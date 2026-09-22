"""Additional coefficient problems fixed before any refinement results.

Monomial order: 1,x,y,xy,x^2,y^2. These are prospective tests within the
same four-boundary-branch problem class, not evidence for arbitrary HJB PDEs.
"""
from fractions import Fraction as F
import unknown_junction_2d as problem
from interval_utils import ball

HELDOUT={
    'holdout_a': ([1.1,.2,-.15,.1,0,0],[.75,-.1,.15,-.1,0,.1],[1,.1,-.2,.2,.1,0]),
    'holdout_b': ([.7,-.15,.1,-.1,.1,0],[1.4,.2,-.15,.1,0,0],[1,-.2,.15,-.15,0,.15]),
    'holdout_c': ([1.2,0,.2,-.2,0,.15],[.9,-.15,0,.15,.1,0],[1,.15,.2,-.2,.1,.1]),
    'holdout_d': ([.85,.1,.15,.15,.1,0],[1.15,-.2,-.1,-.1,0,.1],[1,-.15,-.1,.25,.15,.1]),
}
_data=problem.data
_data_arb=problem.data_arb

def values(x,y,case):
    if case not in HELDOUT:return _data(x,y,case)
    def p(c):return c[0]+c[1]*x+c[2]*y+c[3]*x*y+c[4]*x*x+c[5]*y*y
    return tuple(p(c) for c in HELDOUT[case])

def enclosures(x,y,case):
    if case not in HELDOUT:return _data_arb(x,y,case)
    def jet(row):
        c=[ball(F(str(v))) for v in row]
        return (c[0]+c[1]*x+c[2]*y+c[3]*x*y+c[4]*x*x+c[5]*y*y,
                c[1]+c[3]*y+2*c[4]*x,c[2]+c[3]*x+2*c[5]*y)
    a,ax,ay=jet(HELDOUT[case][0]);b,bx,by=jet(HELDOUT[case][1]);g,gx,gy=jet(HELDOUT[case][2])
    return a,b,g,ax,ay,bx,by,gx,gy

def install():
    problem.data=values;problem.data_arb=enclosures
