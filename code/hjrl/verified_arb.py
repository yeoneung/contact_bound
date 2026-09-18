"""Rigorous real-arithmetic enclosures of periodic Fourier/SiLU critics using Arb.

Stored float32 weights are exact dyadic rationals. Fourier features are defined
mathematically as sin(k*pi*x), cos(k*pi*x); training's rounded frequency buffer
is not used as an exact nonperiodic frequency. ctx.prec controls Arb precision.
"""
from flint import arb, arb_mat, ctx

BITS = 40
SCALE = 1 << BITS


def ceil_int(x):
    return int(x.upper().ceil().unique_fmpz())


def floor_int(x):
    return int(x.lower().floor().unique_fmpz())


def upper(x):
    return arb(ceil_int(x*SCALE))/SCALE


def interval_dict(x):
    lo, hi = floor_int(x*SCALE), ceil_int(x*SCALE)
    return {"lower_numerator":lo, "upper_numerator":hi, "denominator":SCALE,
            "lower":lo/SCALE,"upper":hi/SCALE}


class CertifiedCritic:
    def __init__(self, state, branches):
        self.branches = branches
        self.heads = []
        self.freqs = len(state["freqs" if branches == 1 else "heads.0.freqs"])
        for k in range(branches):
            prefix = "" if branches == 1 else f"heads.{k}."
            layers=[]
            for j in (0,2,4,6):
                w=state[f"{prefix}net.{j}.weight"].double().tolist()
                b=state[f"{prefix}net.{j}.bias"].double().tolist()
                layers.append((arb_mat([[arb(v) for v in row] for row in zip(*w)]),
                               [arb(v) for v in b]))
            self.heads.append(layers)

    def evaluate(self, points, derivative=False):
        pi=arb.pi()
        features, gradients=[],[]
        for x in points:
            sc=[(x*pi*k).sin_cos() for k in range(1,self.freqs+1)]
            features.append([a for a,b in sc]+[b for a,b in sc])
            if derivative:
                gradients.append([pi*k*b for k,(a,b) in enumerate(sc,1)]
                                 +[-pi*k*a for k,(a,b) in enumerate(sc,1)])
        f=arb_mat(features)
        df=arb_mat(gradients) if derivative else None
        values, derivs=[],[]
        for layers in self.heads:
            a=f
            da=df
            for layer,(w,bias) in enumerate(layers):
                z=a*w
                dz=da*w if derivative else None
                rows, drows=[],[]
                for i in range(len(points)):
                    row, drow=[],[]
                    for j,b in enumerate(bias):
                        v=z[i,j]+b
                        if layer == len(layers)-1:
                            row.append(v)
                            if derivative:drow.append(dz[i,j])
                        else:
                            s=1/(1+(-v).exp())
                            row.append(v*s)
                            if derivative:drow.append(dz[i,j]*s*(1+v*(1-s)))
                    rows.append(row)
                    if derivative:drows.append(drow)
                a=arb_mat(rows)
                if derivative:da=arb_mat(drows)
            values.append([a[i,0] for i in range(len(points))])
            if derivative:derivs.append([da[i,0] for i in range(len(points))])
        result=[]
        lip=[]
        for i in range(len(points)):
            v=values[0][i]
            for h in range(1,self.branches):v=v.min(values[h][i])
            result.append(v)
            if derivative:
                ceiling=min(ceil_int(values[h][i]*SCALE) for h in range(self.branches))
                possible=[h for h in range(self.branches)
                          if floor_int(values[h][i]*SCALE)<=ceiling]
                lip.append(max(ceil_int(derivs[h][i].abs_upper()*SCALE) for h in possible))
        return (result,lip) if derivative else result


def ridge_value(x):
    return 1-abs((arb.pi()*x/2).sin())


def ridge_cost(x):
    return ridge_value(x)+arb.pi()/2*abs((arb.pi()*x/2).cos())


def held_cost(x, action, h):
    """Exact analytic integration split at integer kinks of the periodic cost.

    On an interval between consecutive integers, both sin(pi*x/2) and
    cos(pi*x/2) have fixed signs. Integrals of exp(-t)sin/cos are elementary.
    x and h are exact dyadic rationals represented by Arb balls of radius zero.
    """
    if action == 0:return (1-(-h).exp())*ridge_cost(x)
    endpoint=x+action*h
    xmin=min(floor_int(x),floor_int(endpoint))
    xmax=max(ceil_int(x),ceil_int(endpoint))
    times=[arb(0),h]
    for k in range(xmin,xmax+1):
        t=(arb(k)-x)/action
        if t>0 and t<h:times.append(t)
    times.sort(key=float)
    omega=arb.pi()*action/2
    theta=arb.pi()*x/2
    denom=1+omega*omega
    def primitive(t,ss,cs):
        st,ct=(theta+omega*t).sin_cos()
        es=(-t).exp()
        isin=es*(-st-omega*ct)/denom
        icos=es*(-ct+omega*st)/denom
        return -es-ss*isin+arb.pi()/2*cs*icos
    total=arb(0)
    for l,r in zip(times[:-1],times[1:]):
        mid=(l+r)/2
        st,ct=(theta+omega*mid).sin_cos()
        ss=1 if st>0 else -1
        cs=1 if ct>0 else -1
        assert st>0 or st<0
        assert ct>0 or ct<0
        total+=primitive(r,ss,cs)-primitive(l,ss,cs)
    return total
