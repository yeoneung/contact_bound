"""Fixed-checkpoint comparison and separate contact/center refinement.

Run without arguments to reproduce results. --reuse-enclosures reuses the saved
integer node enclosures and verified Lipschitz constants after hash validation.
No training, reference-based scale selection, or sampled regularity estimates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch
from flint import arb, arb_mat, ctx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from hjrl.verified_arb import (CertifiedCritic, SCALE, ceil_int, floor_int,
                              upper, interval_dict, ridge_cost, ridge_value, held_cost)
import verify_neural_policy as original


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CertifiedJet(CertifiedCritic):
    """Value and signed derivative hull, including every potentially active head."""
    def jet(self, points):
        features, gradients = [], []
        for x in points:
            sc = [(x * arb.pi() * k).sin_cos() for k in range(1, self.freqs + 1)]
            features.append([s for s, c in sc] + [c for s, c in sc])
            gradients.append([arb.pi()*k*c for k, (s, c) in enumerate(sc, 1)]
                             + [-arb.pi()*k*s for k, (s, c) in enumerate(sc, 1)])
        f, df = arb_mat(features), arb_mat(gradients)
        values, derivatives = [], []
        for layers in self.heads:
            a, da = f, df
            for layer, (weights, biases) in enumerate(layers):
                z, dz = a * weights, da * weights
                rows, drows = [], []
                for i in range(len(points)):
                    row, drow = [], []
                    for j, bias in enumerate(biases):
                        v = z[i, j] + bias
                        if layer == len(layers) - 1:
                            row.append(v)
                            drow.append(dz[i, j])
                        else:
                            s = 1 / (1 + (-v).exp())
                            row.append(v*s)
                            drow.append(dz[i, j]*s*(1+v*(1-s)))
                    rows.append(row)
                    drows.append(drow)
                a, da = arb_mat(rows), arb_mat(drows)
            values.append([a[i, 0] for i in range(len(points))])
            derivatives.append([da[i, 0] for i in range(len(points))])
        out, dout = [], []
        for i in range(len(points)):
            v = values[0][i]
            for h in range(1, self.branches):
                v = v.min(values[h][i])
            ceiling = min(ceil_int(values[h][i]*SCALE) for h in range(self.branches))
            possible = [h for h in range(self.branches)
                        if floor_int(values[h][i]*SCALE) <= ceiling]
            d = derivatives[possible[0]][i]
            for h in possible[1:]:
                d = d.union(derivatives[h][i])
            out.append(v)
            dout.append(d)
        return out, dout


def contact_bound(ulo, uhi, glo, ghi, lu, m, n):
    """The original Lipschitz near-contact rule with independently varied grids."""
    start = time.perf_counter()
    nodes = len(ulo)
    assert n >= nodes and n % nodes == 0
    eps, q = arb(1)/m, arb(1)/nodes
    wing = min(nodes//2, ceil_int(2*lu*eps*nodes/2)+2)
    offsets = np.arange(-wing, wing+1, dtype=np.int64)
    rwindow = min(arb(1), arb(2*(wing+2))/nodes)
    threshold = ceil_int(2*(lu+rwindow*m)*q*SCALE)
    penalty_unit, slope_unit = SCALE*m//(2*n*n), SCALE*m//n
    assert penalty_unit*2*n*n == SCALE*m and slope_unit*n == SCALE*m
    assert n*n*penalty_unit+int(max(abs(ulo).max(), abs(uhi).max())) < 2**62
    maxima = {-1: 0, 1: 0}
    witnesses = {-1: 0, 1: 0}
    for begin in range(0, n, 128):
        js = np.arange(begin, min(begin+128, n), dtype=np.int64)
        yn = -n+2*js+1
        base = js*nodes//n
        ix = (base[:, None]+offsets[None, :]) % nodes
        xn = -n+2*ix*(n//nodes)
        displacement = (xn-yn[:, None]+n) % (2*n)-n
        penalty = displacement*displacement*penalty_unit
        slope = abs(displacement)*slope_unit
        for sigma in (-1, 1):
            low = (ulo[ix] if sigma == 1 else -uhi[ix])-penalty
            high = (uhi[ix] if sigma == 1 else -ulo[ix])-penalty
            relevant = high >= low.max(axis=1)[:, None]-threshold
            residual = (uhi[ix]-glo[ix]+slope if sigma == 1
                        else ghi[ix]-ulo[ix]-slope)
            bounded = np.where(relevant, np.maximum(residual, 0), 0)
            value = int(bounded.max())
            if value > maxima[sigma]:
                row = int(np.argmax(bounded)//bounded.shape[1])
                maxima[sigma] = value
                witnesses[sigma] = int(yn[row])
    lv = arb.pi()/2
    le = lv+lv*lv
    loc = (lu+le+m)*q
    comparison = 2*lv*le*eps
    r = arb(1)/n
    zeta = 3*lv*r+r*r/(2*eps)
    center = le*r+zeta+2*(2*zeta*(le+lv+m)).sqrt()
    residual = arb(max(maxima.values()))/SCALE
    return {'reciprocal_scale': m, 'grid_nodes': nodes, 'centers': n,
            'near_contact_residual_upper': interval_dict(residual),
            'location_correction': interval_dict(loc),
            'comparison_correction': interval_dict(comparison),
            'center_correction': interval_dict(center),
            'complete_bound': interval_dict(residual+loc+comparison+center),
            'sign_residuals': {str(s): maxima[s]/SCALE for s in (-1, 1)},
            'witness_centers': {str(s): {'numerator': witnesses[s], 'denominator': n}
                                for s in (-1, 1)},
            'estimator_seconds': time.perf_counter()-start}


def contact_witness(model, lu, m, y, sigma=-1, target_width=arb(1)/2**24):
    """Enclose the exact residual at one center by verified global optimization.

    All contacts lie within 2*L_u/m. Objective upper bounds use the interval
    derivative and the mean-value theorem; incumbent lower bounds use actual
    midpoint evaluations. Only cells provably worse than that incumbent are
    removed. The final minimum residual lower endpoint therefore bounds the
    residual at every surviving true maximizer, even if contacts are tied.
    """
    start = time.perf_counter()
    radius = upper(2*lu/m)
    assert radius < 1
    cells = [(y-radius+2*radius*i/128, y-radius+2*radius*(i+1)/128)
             for i in range(128)]
    incumbent = None
    evaluations = 0
    for level in range(40):
        mids = [(lo+hi)/2 for lo, hi in cells]
        radii = [(hi-lo)/2 for lo, hi in cells]
        boxes = [arb(c, r) for c, r in zip(mids, radii)]
        mid_values = model.evaluate(mids)
        values, derivatives = model.jet(boxes)
        evaluations += len(cells)
        objectives = [sigma*v-(c-y)**2*m/2 for c, v in zip(mids, mid_values)]
        lower = max(floor_int(v*SCALE) for v in objectives)
        incumbent = lower if incumbent is None else max(incumbent, lower)
        retained = []
        for j, (obj, box, radius_cell, du) in enumerate(zip(objectives, boxes, radii, derivatives)):
            psi_derivative = sigma*du-m*(box-y)
            ceiling = ceil_int((obj+radius_cell*psi_derivative.abs_upper())*SCALE)
            if ceiling >= incumbent:
                retained.append(j)
        assert retained, 'Verified global maximizer must survive pruning'
        if max(hi-lo for lo, hi in cells) <= target_width:
            residuals = []
            for j in retained:
                residuals.append((sigma*(values[j]-ridge_cost(boxes[j])
                                         +m*abs(boxes[j]-y))).max(arb(0)))
            lo = min(floor_int(v*SCALE) for v in residuals)
            hi = max(ceil_int(v*SCALE) for v in residuals)
            return {'center': interval_dict(y), 'sign': sigma, 'reciprocal_scale': m,
                    'residual_lower': lo/SCALE, 'residual_upper': hi/SCALE,
                    'lower_numerator': lo, 'upper_numerator': hi, 'denominator': SCALE,
                    'surviving_cells': len(retained), 'cell_evaluations': evaluations,
                    'maximum_cell_width': float(max(cells[j][1]-cells[j][0] for j in retained)),
                    'seconds': time.perf_counter()-start,
                    'claim': 'Enclosure at this center only; lower endpoint also bounds the population residual from below.'}
        cells = [(a, b) for j in retained for a, b in
                 ((cells[j][0], mids[j]), (mids[j], cells[j][1]))]
        if len(cells) > 20000:
            raise RuntimeError('Witness optimization exceeded prescribed cell budget')
    raise RuntimeError('Witness optimization did not reach target width')


def bellman_bound(ulo, uhi, lu, reciprocal_hold):
    start = time.perf_counter()
    nodes = len(ulo)
    h = arb(1)/reciprocal_hold
    gamma = (-h).exp()
    assert nodes % (2*reciprocal_hold) == 0
    shift = nodes//(2*reciprocal_hold)
    ug = [original.ball(int(l), int(u)) for l, u in zip(ulo, uhi)]
    sample_lo = sample_hi = 0
    for i in range(nodes):
        x = arb(-nodes+2*i)/nodes
        qs = [held_cost(x, a, h)+gamma*ug[(i+a*shift) % nodes] for a in (0, -1, 1)]
        defect = abs(ug[i]-qs[0].min(qs[1]).min(qs[2]))
        sample_lo = max(sample_lo, floor_int(defect*SCALE))
        sample_hi = max(sample_hi, ceil_int(defect*SCALE))
    le = arb.pi()/2+arb.pi()**2/4
    # Discounted costs have Lipschitz constant (1-gamma)*L_ell for lambda=1.
    coverage = ((1+gamma)*lu+(1-gamma)*le)/nodes
    original.H = h
    bh = original.hold_bias()
    residual = arb(sample_hi)/SCALE+coverage
    return {'reciprocal_hold': reciprocal_hold, 'grid_nodes': nodes,
            'sample_residual_lower': sample_lo/SCALE,
            'sample_residual_upper': sample_hi/SCALE,
            'residual_coverage_correction': interval_dict(coverage),
            'continuous_residual_upper': interval_dict(residual),
            'hold_bias_upper': interval_dict(bh),
            'hold_correction': interval_dict(bh/(1-gamma)),
            'complete_bound': interval_dict((residual+bh)/(1-gamma)),
            'estimator_seconds': time.perf_counter()-start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-enclosures', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    ctx.prec = 128
    design_path = ROOT/'results/comparison_design.json'
    design = json.loads(design_path.read_text())
    weights = ROOT/'results/verified_neural_models.pt'
    states = torch.load(weights, map_location='cpu', weights_only=True)
    sources = [Path(__file__), ROOT/'code/hjrl/verified_arb.py',
               ROOT/'code/experiments/verify_neural_policy.py']
    signature = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    signature['results/comparison_design.json'] = digest(design_path)
    signature['results/verified_neural_models.pt'] = digest(weights)
    output = ROOT/'results/estimator_comparison.json'
    data = {'precision_bits': 128, 'hashes': signature, 'design': design, 'records': []}
    if args.resume and output.exists():
        data = json.loads(output.read_text())
        assert data['hashes'] == signature
    def save():
        output.write_text(json.dumps(data, indent=2)+'\n', encoding='utf8')
    original.M = 65536
    xs = [arb(-65536+2*i)/65536 for i in range(65536)]
    g = [ridge_cost(x) for x in xs]
    glo = np.array([floor_int(v*SCALE) for v in g], dtype=np.int64)
    ghi = np.array([ceil_int(v*SCALE) for v in g], dtype=np.int64)
    for name, branches in (('MLP', 1), ('Min2', 2)):
        saved = next((r for r in data['records'] if r['critic'] == name), None)
        cache = ROOT/f'results/comparison_enclosures_{name}.npz'
        meta_path = cache.with_suffix('.json')
        if args.reuse_enclosures and cache.exists():
            meta = json.loads(meta_path.read_text())
            assert meta['weights_sha256'] == digest(weights)
            assert meta['arithmetic_source_sha256'] == digest(ROOT/'code/hjrl/verified_arb.py')
            assert meta['original_verifier_sha256'] == digest(ROOT/'code/experiments/verify_neural_policy.py')
            assert meta['enclosures_sha256'] == digest(cache)
            arrays = np.load(cache, allow_pickle=False)
            ulo, uhi = arrays['lower'], arrays['upper']
            lu = arb(meta['L_u_numerator'])/SCALE
            evaluation_seconds = meta['evaluation_seconds']
        else:
            ulo, uhi, lu, evaluation_seconds = original.model_data(states[name], branches, name)
            np.savez_compressed(cache, lower=ulo, upper=uhi)
            meta = {'weights_sha256': digest(weights),
                    'arithmetic_source_sha256': digest(ROOT/'code/hjrl/verified_arb.py'),
                    'original_verifier_sha256': digest(ROOT/'code/experiments/verify_neural_policy.py'),
                    'enclosures_sha256': digest(cache),
                    'L_u_numerator': ceil_int(lu*SCALE), 'denominator': SCALE,
                    'derivative_cells': 65536, 'evaluation_nodes': 65536,
                    'evaluation_seconds': evaluation_seconds}
            meta_path.write_text(json.dumps(meta, indent=2)+'\n')
        model = CertifiedJet(states[name], branches)
        if saved is None:
            error_lo = error_hi = 0
            for x, l, h in zip(xs, ulo, uhi):
                error = abs(original.ball(int(l), int(h))-ridge_value(x))
                error_lo = max(error_lo, floor_int(error*SCALE))
                error_hi = max(error_hi, ceil_int(error*SCALE))
            error = arb(arb(error_lo+error_hi)/(2*SCALE), arb(error_hi-error_lo)/(2*SCALE))
            saved = {'critic': name, 'verified_L_u': interval_dict(lu),
                     'common_verification_seconds': evaluation_seconds,
                     'true_error_enclosure': interval_dict(error.lower().union(error.upper()+(lu+arb.pi()/2)/65536)),
                     'bellman': [], 'contacts': [], 'witnesses': []}
            data['records'].append(saved)
            save()
        for h in design['bellman_reciprocal_holds']:
            if any(r['reciprocal_hold'] == h for r in saved['bellman']):
                continue
            result = bellman_bound(ulo, uhi, lu, h)
            saved['bellman'].append(result)
            save()
            print(name, 'Bellman', h, result['complete_bound']['upper'],
                  'seconds', round(result['estimator_seconds'], 2), flush=True)
        settings = [(m, 16384, 262144) for m in design['contact_scale_scan']['reciprocal_scales']]
        settings += [(64, size, 262144) for size in design['location_refinement']['M']]
        settings += [(64, 16384, size) for size in design['center_refinement']['N']]
        for m, size, n in dict.fromkeys(settings):
            if any((r['reciprocal_scale'], r['grid_nodes'], r['centers']) == (m, size, n)
                   for r in saved['contacts']):
                continue
            stride = 65536//size
            result = contact_bound(ulo[::stride], uhi[::stride], glo[::stride], ghi[::stride], lu, m, n)
            saved['contacts'].append(result)
            save()
            print(name, 'Contact', m, size, n, result['complete_bound']['upper'],
                  'seconds', round(result['estimator_seconds'], 2), flush=True)
        for m in design['contact_scale_scan']['reciprocal_scales']:
            if any(r['reciprocal_scale'] == m for r in saved['witnesses']):
                continue
            row = next(r for r in saved['contacts'] if
                       (r['reciprocal_scale'], r['grid_nodes'], r['centers']) == (m, 16384, 262144))
            center = row['witness_centers']['-1']
            result = contact_witness(model, lu, m, arb(center['numerator'])/center['denominator'])
            saved['witnesses'].append(result)
            save()
            print(name, 'Witness', m, result['residual_lower'], result['residual_upper'],
                  'seconds', round(result['seconds'], 2), flush=True)
        assert all(r['complete_bound']['upper'] >= saved['true_error_enclosure']['upper']
                   for r in saved['bellman']+saved['contacts'])
    data['complete'] = True
    save()
    print('Completed', output, flush=True)


if __name__ == '__main__':
    main()
