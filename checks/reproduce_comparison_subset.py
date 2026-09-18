"""Recompute all Bellman scales, two contact grids and two exact-contact witnesses.

Uses hash-validated cached neural enclosures; does not claim to repeat the global
derivative verification or the entire expensive contact refinement sweep.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from flint import arb, ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
sys.path.insert(0, str(ROOT/'code/experiments'))
from compare_verified_estimators import CertifiedJet, bellman_bound, contact_bound, contact_witness
from hjrl.verified_arb import SCALE, floor_int, ceil_int, ridge_cost


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ctx.prec = 128
    data = json.loads((ROOT/'results/estimator_comparison.json').read_text())
    for name, expected in data['hashes'].items():
        assert digest(ROOT/name) == expected
    states = torch.load(ROOT/'results/verified_neural_models.pt', map_location='cpu', weights_only=True)
    g = [ridge_cost(arb(-16384+2*i)/16384) for i in range(16384)]
    glo = np.array([floor_int(v*SCALE) for v in g], dtype=np.int64)
    ghi = np.array([ceil_int(v*SCALE) for v in g], dtype=np.int64)
    def same(a, b, ignored):
        assert {k: v for k, v in a.items() if k not in ignored} == {
            k: v for k, v in b.items() if k not in ignored}
    for row, branches in zip(data['records'], (1, 2)):
        name = row['critic']
        path = ROOT/f'results/comparison_enclosures_{name}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert digest(path) == meta['enclosures_sha256']
        assert digest(ROOT/'results/verified_neural_models.pt') == meta['weights_sha256']
        assert digest(ROOT/'code/hjrl/verified_arb.py') == meta['arithmetic_source_sha256']
        assert digest(ROOT/'code/experiments/verify_neural_policy.py') == meta['original_verifier_sha256']
        arrays = np.load(path, allow_pickle=False)
        ulo, uhi = arrays['lower'], arrays['upper']
        lu = arb(meta['L_u_numerator'])/SCALE
        for before in row['bellman']:
            after = bellman_bound(ulo, uhi, lu, before['reciprocal_hold'])
            same(before, after, {'estimator_seconds'})
        before = next(c for c in row['contacts'] if
                      (c['reciprocal_scale'], c['grid_nodes'], c['centers']) == (64, 16384, 16384))
        after = contact_bound(ulo[::4], uhi[::4], glo, ghi, lu, 64, 16384)
        same(before, after, {'estimator_seconds'})
        before = next(w for w in row['witnesses'] if w['reciprocal_scale'] == 64)
        center = arb(before['center']['lower_numerator'])/SCALE
        assert before['center']['lower_numerator'] == before['center']['upper_numerator']
        after = contact_witness(CertifiedJet(states[name], branches), lu, 64, center)
        same(before, after, {'seconds'})
    report = {'successful': True, 'bellman_settings_recomputed': 8,
              'contact_settings_recomputed': 2, 'contact_witnesses_recomputed': 2,
              'cached_enclosures_used': True, 'global_derivative_verification_repeated': False,
              'full_contact_sweep_repeated': False, 'all_non_timing_fields_match': True}
    (ROOT/'checks/comparison_subset_reproduction.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
