"""Reproduce the preserved numerical nodes from the explicit HJB data."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def solve(n):
    grid=-1+2*np.arange(n)/n;x,y=np.meshgrid(grid,grid,indexing='ij');h=1/n
    c,d=np.cos(np.pi*x),np.cos(np.pi*y)
    b=1+c*d/4
    g=2-np.abs(np.sin(np.pi*x/2))-np.abs(np.sin(np.pi*y/2))+c*d/8
    slope=np.pi/2*(np.abs(np.cos(np.pi*x/2))+np.abs(np.cos(np.pi*y/2)))
    slope+=np.pi/8*(np.abs(np.sin(np.pi*x))*d+c*np.abs(np.sin(np.pi*y)))
    g+=b*slope
    data=[]
    for a,c in [(0,0),(-1,-1),(-1,1),(1,-1),(1,1)]:
        px,py=((x+h*b*a+1)*n/2)%n,((y+h*b*c+1)*n/2)%n
        ix,iy=np.floor(px).astype(int),np.floor(py).astype(int)
        data.append((ix,iy,px-ix,py-iy))
    gamma=np.exp(-h);u=np.zeros((n,n))
    for iteration in range(10000):
        alternatives=[]
        for ix,iy,ax,ay in data:
            z=((1-ax)*(1-ay)*u[ix,iy]+ax*(1-ay)*u[(ix+1)%n,iy]
               +(1-ax)*ay*u[ix,(iy+1)%n]+ax*ay*u[(ix+1)%n,(iy+1)%n])
            alternatives.append((1-gamma)*g+gamma*z)
        new=np.minimum.reduce(alternatives);change=np.max(np.abs(new-u));u=new
        if change<1e-12:break
    assert change<1e-12
    return np.rint(u*2**24).astype(np.int64)

if __name__=='__main__':
    for n in [32,64]:
        saved=json.loads((ROOT/f'results/twod_inputs/n{n}.json').read_text())
        assert np.array_equal(solve(n),np.asarray(saved['node_numerators']))
        print(n,'semi-Lagrangian nodes reproduced exactly')
