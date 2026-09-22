"""Polynomial algebra and the preserved auxiliary-reference verifier.

Adapted from submission/code/hjrl/verified_auxiliary.py.
The data-fitted contact reconstruction uses only the generic polynomial routines.

Construct and verify an auxiliary polynomial from the HJB problem data.

The construction solves a collocation least-squares problem. Certification uses
the rounded polynomial, exact algebra, Arb and Bernstein coefficient bounds.
No reference value function or reference error is accepted as an input.
"""
from math import comb, factorial
import numpy as np
from numpy.polynomial import chebyshev as cheb
from flint import arb, arb_mat, ctx
from interval_utils import ceil_int, floor_int

SCALE = 1 << 40

def interval_dict(x):
    lo,hi=floor_int(x*SCALE),ceil_int(x*SCALE)
    return dict(lower_numerator=lo,upper_numerator=hi,denominator=SCALE,lower=lo/SCALE,upper=hi/SCALE)


def problem_data_float(points):
    """Explicit running cost and speed, with t = 1 - |x| in [0,1]."""
    k = np.pi/2
    if len(points) == 1:
        t, = points
        return np.ones_like(t), 1-np.cos(k*t)+k*np.sin(k*t)
    s,t = points
    c,d = np.cos(np.pi*s),np.cos(np.pi*t)
    b = 1+c*d/4
    g = (2-np.cos(k*s)-np.cos(k*t)+c*d/8
         +b*(k*(np.sin(k*s)+np.sin(k*t))
              -np.pi/8*(np.sin(np.pi*s)*d+c*np.sin(np.pi*t))))
    return b,g


def construct(dimension, order, denominator_bits=48):
    """Even Chebyshev basis enforces the periodic-boundary normal derivative."""
    grid = (1-np.cos(np.pi*np.arange(2*order+3)/(2*order+2)))/2
    points = np.meshgrid(*([grid]*dimension), indexing='ij')
    points = [p.ravel() for p in points]
    vals,derivs = [],[]
    for p in points:
        vals.append(cheb.chebvander(p,2*order)[:,::2])
        derivs.append(np.column_stack([cheb.chebval(p,cheb.chebder(np.eye(2*order+1)[2*j]))
                                       for j in range(order+1)]))
    b,g = problem_data_float(points)
    if dimension == 1:
        matrix = vals[0]+b[:,None]*derivs[0]
    else:
        matrix = ((vals[0][:,:,None]*vals[1][:,None,:])
                  +b[:,None,None]*(derivs[0][:,:,None]*vals[1][:,None,:]
                                   +vals[0][:,:,None]*derivs[1][:,None,:])).reshape(len(g),-1)
    coeffs,_,rank,singular = np.linalg.lstsq(matrix,g,rcond=None)
    assert rank == matrix.shape[1]
    nums = [int(round(float(c)*2**denominator_bits)) for c in coeffs]
    return dict(dimension=dimension,order=order,denominator_bits=denominator_bits,
                numerators=nums,collocation_points=len(g),unknowns=len(nums),
                condition_number=float(singular[0]/singular[-1]),
                sampled_residual=float(np.max(np.abs(matrix@coeffs-g))))


def power_coefficients(record):
    """Exact Chebyshev recurrence and dyadic coefficient conversion."""
    degree = 2*record['order']
    ts = [[1],[0,1]]
    for k in range(2,degree+1):
        p=[0]+[2*c for c in ts[-1]]
        for j,c in enumerate(ts[-2]):p[j]-=c
        ts.append(p)
    basis=ts[::2];dim=record['dimension'];out={}
    for index,num in enumerate(record['numerators']):
        indices=(index,) if dim==1 else divmod(index,len(basis))
        for powers in np.ndindex(*[len(basis[j]) for j in indices]):
            c=num
            for j,p in zip(indices,powers):c*=basis[j][p]
            if c:out[powers]=out.get(powers,0)+c
    den=2**record['denominator_bits']
    return {p:arb(c)/den for p,c in out.items() if c}


def add(*polynomials):
    result={}
    for p in polynomials:
        for key,value in p.items():result[key]=result.get(key,arb(0))+value
    return result


def scale(p,a):return {k:v*a for k,v in p.items()}


def multiply(p,q):
    result={}
    for i,a in p.items():
        for j,b in q.items():
            key=tuple(x+y for x,y in zip(i,j))
            result[key]=result.get(key,arb(0))+a*b
    return result


def derivative(p,axis):
    result={}
    for k,v in p.items():
        if k[axis]:
            key=list(k);key[axis]-=1;result[tuple(key)]=v*k[axis]
    return result


def trigonometric_polynomial(dimension,axis,frequency,sine,degree=28):
    result={}
    for n in range(int(sine),degree+1,2):
        key=[0]*dimension;key[axis]=n
        result[tuple(key)]=(-1)**((n-int(sine))//2)*frequency**n/factorial(n)
    # Taylor's theorem on [0,1], using |d^n sin|, |d^n cos| <= 1.
    error=frequency**(degree+1)/factorial(degree+1)
    return result,error


def problem_polynomials(dimension,degree=28):
    one={(0,)*dimension:arb(1)};k=arb.pi()/2
    small=[trigonometric_polynomial(dimension,i,k,False,degree) for i in range(dimension)]
    sins=[trigonometric_polynomial(dimension,i,k,True,degree) for i in range(dimension)]
    if dimension==1:
        g=add(one,scale(small[0][0],-1),scale(sins[0][0],k))
        return one,arb(0),g,small[0][1]+k*sins[0][1]
    cs=[trigonometric_polynomial(2,i,arb.pi(),False,degree) for i in range(2)]
    sn=[trigonometric_polynomial(2,i,arb.pi(),True,degree) for i in range(2)]
    product=multiply(cs[0][0],cs[1][0])
    # Exact trig factors have magnitude <= 1, polynomial factors <= 1+error.
    prod_error=cs[0][1]+cs[1][1]+cs[0][1]*cs[1][1]
    b=add(one,scale(product,arb(1)/4));eb=prod_error/4
    z=add(scale(one,2),scale(small[0][0],-1),scale(small[1][0],-1),scale(product,arb(1)/8))
    ez=small[0][1]+small[1][1]+prod_error/8
    cross=add(multiply(sn[0][0],cs[1][0]),multiply(cs[0][0],sn[1][0]))
    ecross=sum((sn[i][1]+cs[1-i][1]+sn[i][1]*cs[1-i][1] for i in range(2)),arb(0))
    s=add(scale(add(sins[0][0],sins[1][0]),k),scale(cross,-arb.pi()/8))
    es=k*(sins[0][1]+sins[1][1])+arb.pi()/8*ecross
    # |s| <= 2*k+pi/4; |b| <= 5/4.
    eg=ez+arb(5)/4*es+eb*(2*k+arb.pi()/4+es)
    return b,eb,add(z,multiply(b,s)),eg


def bernstein_coefficients(p):
    """Power-to-Bernstein conversion on the unit interval/square."""
    dim=len(next(iter(p)));degrees=[max(k[i] for k in p) for i in range(dim)]
    matrices=[]
    for n in degrees:
        matrices.append(arb_mat([[arb(comb(i,k))/comb(n,k) if k<=i else arb(0)
                                  for k in range(n+1)] for i in range(n+1)]))
    if dim==1:
        values=matrices[0]*arb_mat([[p.get((i,),arb(0))] for i in range(degrees[0]+1)])
    else:
        coeffs=arb_mat([[p.get((i,j),arb(0)) for j in range(degrees[1]+1)]
                       for i in range(degrees[0]+1)])
        values=matrices[0]*coeffs*matrices[1].transpose()
    return values


def bernstein_bounds(p):
    """Convex-hull range, with rigorous endpoint comparisons even at ties."""
    values=bernstein_coefficients(p)
    lo,hi=values[0,0].lower(),values[0,0].upper()
    for i in range(values.nrows()):
        for j in range(values.ncols()):
            lo=lo.min(values[i,j].lower()).lower()
            hi=hi.max(values[i,j].upper()).upper()
    return lo,hi


def certify(record,taylor_degree=28):
    p=power_coefficients(record);dim=record['dimension'];dp=[derivative(p,j) for j in range(dim)]
    gradients=[]
    for j,d in enumerate(dp):
        # All exponents in p are even: the derivative factors exactly as t_j*q_j.
        assert all(k[j]>=1 for k in d)
        quotient={tuple(n-int(i==j) for i,n in enumerate(k)):v for k,v in d.items()}
        lo,hi=bernstein_bounds(quotient)
        assert lo>0, ('Gradient monotonicity not proved',j,str(lo))
        gradients.append({'axis':j,'factored_gradient_lower':interval_dict(lo),
                          'factored_gradient_upper':interval_dict(hi)})
    b,eb,g,eg=problem_polynomials(dim,taylor_degree)
    ds=add(*dp)
    _,dsmax=bernstein_bounds(ds)
    residual=add(p,multiply(b,ds),scale(g,-1))
    lo,hi=bernstein_bounds(residual)
    remainder=eg+eb*dsmax.abs_upper()
    rho=(-lo+remainder).max(hi+remainder)
    return {'residual_bound':interval_dict(rho),'residual_polynomial_range':
            {'lower':interval_dict(lo),'upper':interval_dict(hi)},
            'remainder_bound':interval_dict(remainder),'gradient_checks':gradients,
            'taylor_degree':taylor_degree,'precision_bits':ctx.prec,
            'viscosity_interfaces':'strict downward cusps at x_j=0; C1 periodic joins at |x_j|=1'}
