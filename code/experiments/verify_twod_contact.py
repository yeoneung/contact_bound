"""Reproduce complete 2D certificates for coupled dynamics and bilinear critics."""
from pathlib import Path
import argparse, hashlib, json, platform, sys, time
import numpy as np
from flint import ctx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'code'))
from hjrl.verified_twod import solve_critic, sparse_contact_bound, reference_error, center_data


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', default='cuda', choices=['numpy', 'cpu', 'cuda'])
    parser.add_argument('--grids', default='1024:512,2048:1024,4096:2048')
    parser.add_argument('--scales', default='16,32,64')
    parser.add_argument('--critics', default='32,64')
    parser.add_argument('--validation-side', type=int, default=512)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    ctx.prec = 128
    output = ROOT/'results/twod_contact.json'
    source = ROOT/'code/hjrl/verified_twod.py'
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in [source, Path(__file__), ROOT/'code/hjrl/verified_arb.py']}
    data = dict(dimension=2, precision_bits=128, source_hashes=hashes,
                problem='Coupled periodic speed and running cost; alpha=1/8, beta=1/4',
                environment=dict(python=platform.python_version(), platform=platform.platform()),
                records=[], complete=False)
    if args.resume and output.exists():
        data = json.loads(output.read_text())
        assert data['source_hashes'] == hashes, 'Source changed; start a new verified sweep.'
    def save(): output.write_text(json.dumps(data, indent=2)+'\n', encoding='utf8')
    grids = [tuple(map(int, pair.split(':'))) for pair in args.grids.split(',')]
    scales = list(map(int, args.scales.split(',')))
    centers = {}
    for _, n in grids:
        if n not in centers:
            start = time.perf_counter()
            centers[n] = center_data(n)
            data.setdefault('center_preprocessing_seconds', {})[str(n)] = time.perf_counter()-start
    for ncritic in map(int, args.critics.split(',')):
        path = ROOT/f'results/twod_critic_{ncritic}.npz'
        row = next((r for r in data['records'] if r['critic_side'] == ncritic), None)
        if row is None:
            u, metadata = solve_critic(ncritic, 1/ncritic)
            np.savez_compressed(path, nodes=u)
            row = dict(**metadata, coefficients_sha256=digest(path),
                       reference_error=reference_error(u, args.validation_side), settings=[])
            data['records'].append(row)
            save()
            print('critic', ncritic, 'error', row['reference_error'], flush=True)
        assert digest(path) == row['coefficients_sha256']
        u = np.load(path, allow_pickle=False)['nodes']
        for m, n in grids:
            for scale in scales:
                if any((s['location_side'], s['center_side'], s['reciprocal_scale']) == (m,n,scale) for s in row['settings']):
                    continue
                result = sparse_contact_bound(u, m, n, scale, args.backend, centers[n])
                result['standalone_seconds'] = result['verification_seconds']+data['center_preprocessing_seconds'][str(n)]
                assert result['complete_bound']['upper'] >= row['reference_error']['upper']
                row['settings'].append(result)
                save()
                print('critic', ncritic, 'grid', m, n, '1/eps', scale,
                      'bound', result['complete_bound']['upper'],
                      'seconds', round(result['standalone_seconds'], 2), flush=True)
    data['complete'] = True
    data['design'] = dict(grids=grids, reciprocal_scales=scales, critic_sides=list(map(int,args.critics.split(','))))
    save()


if __name__ == '__main__': main()
