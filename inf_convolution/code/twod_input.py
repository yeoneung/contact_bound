"""Fit a symmetric polynomial to saved numerical values, without PDE/reference fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
from numpy.polynomial import chebyshev as ch

ROOT=Path(__file__).resolve().parents[1]

def fit(nodes,order=4,bits=44):
    n=len(nodes);m=n//2;ids=(m+np.arange(m+1))%n
    values=np.asarray(nodes)[np.ix_(ids,ids)]/2**24
    t=1-np.arange(m+1)/m;basis=ch.chebvander(t,2*order)[:,::2]
    matrix=(basis[:,None,:,None]*basis[None,:,None,:]).reshape((m+1)**2,-1)
    coeffs,_,rank,singular=np.linalg.lstsq(matrix,values.ravel(),rcond=None)
    assert rank==(order+1)**2
    coeffs=coeffs.reshape(order+1,order+1);coeffs=(coeffs+coeffs.T)/2
    nums=[int(round(float(v)*2**bits)) for v in coeffs.ravel()]
    return dict(dimension=2,order=order,denominator_bits=bits,numerators=nums,
                collocation_points=(m+1)**2,unknowns=len(nums),
                fit='least squares to numerical node values only',
                condition_number=float(singular[0]/singular[-1]))

def main():
    folder=ROOT/'results/twod_inputs';folder.mkdir(exist_ok=True)
    for n in [32,64]:
        source=ROOT.parent/f'submission/results/twod_critic_{n}.npz'
        saved_path=folder/f'n{n}.json'
        if source.exists():
            nodes=np.load(source,allow_pickle=False)['nodes']
            original_sha=hashlib.sha256(source.read_bytes()).hexdigest()
        else:
            saved=json.loads(saved_path.read_text());nodes=np.array(saved['node_numerators'],dtype=np.int64)
            original_sha=saved['original_npz_sha256']
        assert np.array_equal(nodes,nodes[np.mod(-np.arange(n),n)])
        assert np.array_equal(nodes,nodes[:,np.mod(-np.arange(n),n)])
        assert np.array_equal(nodes,nodes.T)
        record=dict(n=n,node_denominator_bits=24,node_numerators=nodes.tolist(),
                    original_npz_sha256=original_sha,
                    provenance='Frozen periodic bilinear semi-Lagrangian approximation from submission; no retraining.',
                    polynomial=fit(nodes),smoothing_denominator=n*n)
        (folder/f'n{n}.json').write_text(json.dumps(record,indent=2)+'\n')
        print(n,'data-only polynomial fit saved',flush=True)

if __name__=='__main__':main()
