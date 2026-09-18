"""Independent exact-rational, high-precision and backend checks for 2D coverage."""
from pathlib import Path
from fractions import Fraction as F
import json, sys, unittest
import numpy as np
import mpmath as mp
from flint import arb, ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from hjrl.verified_twod import (BASE, SCALE2, constants, critic_grid, critic_lipschitz,
                               trig, problem_arb_from_trig, contact_bound, center_data, sparse_contact_bound)


def bilinear(u, x, y):
    n = len(u)
    tx, ty = ((x+1)*n/2) % n, ((y+1)*n/2) % n
    i, j = tx.numerator//tx.denominator, ty.numerator//ty.denominator
    a, b = tx-i, ty-j
    return ((1-a)*(1-b)*int(u[i,j])+a*(1-b)*int(u[(i+1)%n,j])
            +(1-a)*b*int(u[i,(j+1)%n])+a*b*int(u[(i+1)%n,(j+1)%n]))/BASE


def continuous_contacts(u, center, sign, scale):
    """Enumerate exact patch/edge/corner maxima, independently of grid search."""
    n = len(u)
    y, z = center
    candidates = []
    for i in range(-n,2*n):
        x0 = F(-1)+F(2*i,n); x1=x0+F(2,n)
        for j in range(-n,2*n):
            z0=F(-1)+F(2*j,n); z1=z0+F(2,n)
            a=bilinear(u,x0,z0)
            b=(bilinear(u,x1,z0)-a)/(x1-x0)
            c=(bilinear(u,x0,z1)-a)/(z1-z0)
            d=(bilinear(u,x1,z1)-a-b*(x1-x0)-c*(z1-z0))/((x1-x0)*(z1-z0))
            assert scale > abs(d)
            def evaluate(x,w):
                value=a+b*(x-x0)+c*(w-z0)+d*(x-x0)*(w-z0)
                candidates.append((sign*value-scale*((x-y)**2+(w-z)**2)/2,x,w))
            rhsx=scale*y+sign*(b-d*z0)
            rhsy=scale*z+sign*(c-d*x0)
            det=scale*scale-d*d
            x=(scale*rhsx+sign*d*rhsy)/det
            w=(scale*rhsy+sign*d*rhsx)/det
            if x0 <= x <= x1 and z0 <= w <= z1:evaluate(x,w)
            for x in (x0,x1):
                w=min(z1,max(z0,z+sign*(c+d*(x-x0))/scale));evaluate(x,w)
            for w in (z0,z1):
                x=min(x1,max(x0,y+sign*(b+d*(w-z0))/scale));evaluate(x,w)
    best=max(c[0] for c in candidates)
    return [(x,z) for v,x,z in candidates if v==best]


class TwoDChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):ctx.prec=128;mp.mp.dps=70

    def test_problem_cost_and_viscosity_cones(self):
        c=constants()
        def v(x,y):return 2-abs(mp.sin(mp.pi*x/2))-abs(mp.sin(mp.pi*y/2))+mp.cos(mp.pi*x)*mp.cos(mp.pi*y)/8
        for xn,yn in [(-7,-3),(-3,2),(2,5),(5,7)]:
            x,y=mp.mpf(xn)/8,mp.mpf(yn)/8
            dx,dy=mp.diff(lambda t:v(t,y),x),mp.diff(lambda t:v(x,t),y)
            speed=1+mp.cos(mp.pi*x)*mp.cos(mp.pi*y)/4
            cost=v(x,y)+speed*(abs(dx)+abs(dy))
            va,ba,ga=problem_arb_from_trig(trig(arb(xn)/8),trig(arb(yn)/8))
            for enclosure,value in [(va,v(x,y)),(ba,speed),(ga,cost)]:
                self.assertTrue(enclosure.contains(arb(str(value))))
        # At a ridge intersection all upper slopes lie in [-pi/2,pi/2]^2.
        vv,bb,gg=problem_arb_from_trig(trig(arb(0)),trig(arb(0)))
        for i in (-1,0,1):
            for j in (-1,0,1):
                residual=vv+bb*arb.pi()/2*(abs(i)+abs(j))-gg
                self.assertLessEqual(float(residual.upper()),1e-30)

    def test_dyadic_interpolation_and_derivative_hulls(self):
        u=np.array([[1,2,0,1],[2,3,1,0],[1,0,2,1],[0,1,1,2]],dtype=np.int64)*(BASE//32)
        lo,hi,dxlo,dxhi,dylo,dyhi=critic_grid(u,32)
        for i,j in [(0,0),(1,9),(8,16),(17,23),(31,31)]:
            x,y=F(-1)+F(i,16),F(-1)+F(j,16)
            exact=bilinear(u,x,y)
            self.assertLessEqual(F(int(lo[i,j]),SCALE2),exact)
            self.assertGreaterEqual(F(int(hi[i,j]),SCALE2),exact)
            for a in (-1,1):
                for b in (-1,1):
                    xx,yy=x+F(a,128),y+F(b,128)
                    h=F(1,4096)
                    gx=(bilinear(u,xx+h,yy)-bilinear(u,xx-h,yy))/(2*h)
                    gy=(bilinear(u,xx,yy+h)-bilinear(u,xx,yy-h))/(2*h)
                    self.assertTrue(F(int(dxlo[i,j]),SCALE2)<=gx<=F(int(dxhi[i,j]),SCALE2))
                    self.assertTrue(F(int(dylo[i,j]),SCALE2)<=gy<=F(int(dyhi[i,j]),SCALE2))

    def test_exact_continuous_contacts_survive_joint_coverage(self):
        u=np.array([[1,2,0,1],[2,3,1,0],[1,0,2,1],[0,1,1,2]],dtype=np.int64)*(BASE//32)
        m,n,scale=32,8,8
        lu=critic_lipschitz(u)
        q,r=arb(2).sqrt()/m,arb(2).sqrt()/n
        radius=lu/scale+q+r
        theta=2*lu*r+scale*r*r+(lu+radius*scale)*q
        lo,hi,dxlo,dxhi,dylo,dyhi=critic_grid(u,m)
        nodes=[F(-1)+F(2*i,m) for i in range(m)]
        for sign in (-1,1):
            yj,zj=F(-1)+F(3,n),F(-1)+F(9,n)
            objectives=[sign*bilinear(u,x,z)-scale*(((x-yj+1)%2-1)**2+((z-zj+1)%2-1)**2)/2 for x in nodes for z in nodes]
            maximum=max(objectives)
            for ox,oy in [(0,0),(-1,1),(1,-1)]:
                center=(yj+F(ox,n),zj+F(oy,n))
                for x,z in continuous_contacts(u,center,sign,scale):
                    ix=int(((x+1)*m/2+F(1,2))//1)%m
                    iy=int(((z+1)*m/2+F(1,2))//1)%m
                    objective=objectives[ix*m+iy]
                    gap=maximum-objective
                    self.assertTrue(arb(gap.numerator)/gap.denominator<=theta)
                    ax,ay=(nodes[ix]-yj+1)%2-1,(nodes[iy]-zj+1)%2-1
                    slack=F(scale)*(F(1,m)+F(1,n))
                    for p,left,right in [(sign*scale*ax,dxlo[ix,iy],dxhi[ix,iy]),(sign*scale*ay,dylo[ix,iy],dyhi[ix,iy])]:
                        self.assertTrue(p+slack>=F(int(left),SCALE2))
                        self.assertTrue(p-slack<=F(int(right),SCALE2))

    def test_cpu_gpu_exact_integer_agreement(self):
        import torch
        u=np.array([[1,2,0,1],[2,3,1,0],[1,0,2,1],[0,1,1,2]],dtype=np.int64)*(BASE//32)
        cached=center_data(16)
        a=contact_bound(u,32,16,8,'numpy',cached)
        backend='cuda' if torch.cuda.is_available() else 'cpu'
        b=contact_bound(u,32,16,8,backend,cached)
        for key in ['residual_numerators','filtered_residual_numerators','candidate_counts','complete_bound']:
            self.assertEqual(a[key],b[key])

    def test_sparse_enumeration_matches_dense_search(self):
        u=np.array([[1,2,0,1],[2,3,1,0],[1,0,2,1],[0,1,1,2]],dtype=np.int64)*(BASE//32)
        for scale in (4,8,16):
            for m,n in [(32,16),(64,16)]:
                a=contact_bound(u,m,n,scale,'numpy')
                b=sparse_contact_bound(u,m,n,scale,'numpy')
                for key in ['filtered_residual_numerators','complete_bound']:
                    self.assertEqual(a[key],b[key])
                self.assertEqual({s:v[1] for s,v in a['candidate_counts'].items()},b['filtered_candidate_counts'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TwoDChecks))
    (ROOT/'checks/twod_contact_checks.json').write_text(json.dumps(dict(checks_run=result.testsRun,successful=result.wasSuccessful()),indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
