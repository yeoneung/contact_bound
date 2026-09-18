"""Two-dimensional joint contact coverage with exact integer candidate tests.

The supplied critic is the periodic bilinear interpolant of dyadic node values.
Arb encloses problem constants and costs; integer arithmetic covers selection,
Hamiltonian evaluation and a componentwise Clarke-gradient exclusion test.
"""
import time
import numpy as np
from flint import arb, ctx
from .verified_arb import ceil_int, floor_int, interval_dict, upper

BASE = 1 << 24
SCALE2 = 1 << 32
BSCALE = 1 << 24
ALPHA = 1 / 8
BETA = 1 / 4


def problem_float(x, y):
    c, d = np.cos(np.pi*x), np.cos(np.pi*y)
    value = 2-np.abs(np.sin(np.pi*x/2))-np.abs(np.sin(np.pi*y/2))+ALPHA*c*d
    speed = 1+BETA*c*d
    slope = np.pi/2*(np.abs(np.cos(np.pi*x/2))+np.abs(np.cos(np.pi*y/2)))
    slope += ALPHA*np.pi*(np.abs(np.sin(np.pi*x))*d+c*np.abs(np.sin(np.pi*y)))
    return value, speed, value+speed*slope


def trig(x):
    p = arb.pi()
    s, c = (p*x).sin_cos()
    sh, ch = (p*x/2).sin_cos()
    return abs(s), c, abs(sh), abs(ch)


def problem_arb_from_trig(tx, ty):
    sx, cx, hx, kx = tx
    sy, cy, hy, ky = ty
    value = 2-hx-hy+cx*cy/8
    speed = 1+cx*cy/4
    slope = arb.pi()/2*(kx+ky)+arb.pi()/8*(sx*cy+cx*sy)
    return value, speed, value+speed*slope


def constants():
    p, rt = arb.pi(), arb(2).sqrt()
    # |grad b| <= beta*pi, using sin^2(x)cos^2(y)+cos^2(x)sin^2(y) <= 1.
    lv = upper(rt*5*p/8)
    ls = upper(rt*p*p/2)
    ell = upper(lv+arb(5)/4*ls+5*p*p/16)
    return dict(M_f=upper(5*rt/4), L_f=upper(rt*p/4), L_ell=ell,
                reference_L_V=lv)


def solve_critic(n, h, tolerance=1e-12):
    """Unverified construction only; the saved dyadic interpolant is certified later.

    Discounted semi-Lagrangian iteration with a frozen Euler successor and
    left-endpoint cost. The exact value is not used by this iteration.
    """
    start = time.perf_counter()
    nodes = -1+2*np.arange(n)/n
    x, y = np.meshgrid(nodes, nodes, indexing='ij')
    _, b, g = problem_float(x, y)
    actions = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    data = []
    for a, c in actions:
        px, py = ((x+h*b*a+1)*n/2) % n, ((y+h*b*c+1)*n/2) % n
        ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
        data.append((ix, iy, px-ix, py-iy))
    gamma = np.exp(-h)
    u = np.zeros((n, n))
    for iteration in range(10000):
        alternatives = []
        for ix, iy, ax, ay in data:
            z = ((1-ax)*(1-ay)*u[ix, iy]+ax*(1-ay)*u[(ix+1) % n, iy]
                 +(1-ax)*ay*u[ix, (iy+1) % n]+ax*ay*u[(ix+1) % n, (iy+1) % n])
            alternatives.append((1-gamma)*g+gamma*z)
        new = np.minimum.reduce(alternatives)
        change = np.max(np.abs(new-u))
        u = new
        if change < tolerance:
            break
    assert change < tolerance
    dyadic = np.rint(u*BASE).astype(np.int64)
    return dyadic, dict(critic_side=n, hold=h, iterations=iteration+1,
                       last_change=float(change), construction_seconds=time.perf_counter()-start,
                       coefficient_denominator=BASE)


def critic_lipschitz(u):
    n = len(u)
    dx0, dy0 = np.roll(u, -1, 0)-u, np.roll(u, -1, 1)-u
    dx1, dy1 = np.roll(dx0, -1, 1), np.roll(dy0, -1, 0)
    max_squared = max(int(np.max(a*a+b*b)) for a in (dx0, dx1) for b in (dy0, dy1))
    return upper(arb(max_squared).sqrt()*n/(2*BASE))


def critic_grid(u, m):
    """Outward values and derivative hulls on all fine-grid Voronoi cells."""
    n = len(u)
    assert m % n == 0
    ratio = m//n
    i = np.arange(m)
    j, a = i//ratio, i % ratio
    numerator = ((ratio-a[:, None])*(ratio-a[None, :])*u[j[:, None], j[None, :]]
                 +a[:, None]*(ratio-a[None, :])*u[(j[:, None]+1) % n, j[None, :]]
                 +(ratio-a[:, None])*a[None, :]*u[j[:, None], (j[None, :]+1) % n]
                 +a[:, None]*a[None, :]*u[(j[:, None]+1) % n, (j[None, :]+1) % n])
    scaled = numerator*(SCALE2//BASE)
    lo, hi = scaled//(ratio*ratio), -((-scaled)//(ratio*ratio))
    dx = (np.roll(u, -1, 0)-u)*(n//2)*(SCALE2//BASE)
    dy = (np.roll(u, -1, 1)-u)*(n//2)*(SCALE2//BASE)
    left = (j-(a == 0)) % n
    # A derivative is affine in the other coordinate within each coarse patch.
    # Evaluate it at endpoints of the actual fine Voronoi-cell intersection,
    # retaining all patches touching a coarse grid line or vertex.
    lower_x=upper_x=lower_y=upper_y=None
    phases=[a+(a==0)*ratio,a]
    for ix,px in zip((left,j),phases):
        for iy,py in zip((left,j),phases):
            lx=np.clip(2*px-1,0,2*ratio)[:,None]
            hx=np.clip(2*px+1,0,2*ratio)[:,None]
            ly=np.clip(2*py-1,0,2*ratio)[None,:]
            hy=np.clip(2*py+1,0,2*ratio)[None,:]
            d0=dx[ix[:,None],iy[None,:]]
            d1=dx[ix[:,None],(iy[None,:]+1)%n]
            ex0=d0*(2*ratio)+(d1-d0)*ly
            ex1=d0*(2*ratio)+(d1-d0)*hy
            c0=dy[ix[:,None],iy[None,:]]
            c1=dy[(ix[:,None]+1)%n,iy[None,:]]
            ey0=c0*(2*ratio)+(c1-c0)*lx
            ey1=c0*(2*ratio)+(c1-c0)*hx
            lowx=np.minimum(ex0,ex1)//(2*ratio)
            highx=-((-np.maximum(ex0,ex1))//(2*ratio))
            lowy=np.minimum(ey0,ey1)//(2*ratio)
            highy=-((-np.maximum(ey0,ey1))//(2*ratio))
            lower_x=lowx if lower_x is None else np.minimum(lower_x,lowx)
            upper_x=highx if upper_x is None else np.maximum(upper_x,highx)
            lower_y=lowy if lower_y is None else np.minimum(lower_y,lowy)
            upper_y=highy if upper_y is None else np.maximum(upper_y,highy)
    boxes=[lower_x,upper_x,lower_y,upper_y]
    return lo, hi, *boxes


def center_data(n):
    ts = [trig(arb(-n+2*j+1)/n) for j in range(n)]
    arrays = [np.empty((n, n), dtype=np.int64) for _ in range(4)]
    for j in range(n):
        for k in range(n):
            _, b, g = problem_arb_from_trig(ts[j], ts[k])
            values = (floor_int(g*SCALE2), ceil_int(g*SCALE2),
                      floor_int(b*BSCALE), ceil_int(b*BSCALE))
            for a, value in zip(arrays, values):
                a[j, k] = value
    return arrays


def contact_bound(u, m, n, reciprocal_scale, backend='numpy', cached_centers=None):
    start = time.perf_counter()
    assert m >= n and m % n == 0
    q, r, eps = arb(2).sqrt()/m, arb(2).sqrt()/n, arb(1)/reciprocal_scale
    lu = critic_lipschitz(u)
    radius = lu*eps+q+r
    assert radius < 1, 'The implementation uses unique shortest displacements.'
    theta = 2*lu*r+r*r/eps+(lu+radius/eps)*q
    c = constants()
    delta = (lu*q+(c['L_ell']+c['L_f']*radius/eps)*r
             +(c['M_f']+radius)*(q+r)/eps)
    threshold = ceil_int(theta*SCALE2)
    radius_squared = ceil_int(radius*radius*m*m)
    slope_slack = ceil_int(reciprocal_scale*(arb(1)/m+arb(1)/n)*SCALE2)
    fields = critic_grid(u, m)
    gc = cached_centers if cached_centers is not None else center_data(n)
    wing = ceil_int(radius*m/2)+2
    offsets = np.arange(-wing, wing+1, dtype=np.int64)
    ox, oy = np.meshgrid(offsets, offsets, indexing='ij')
    ox, oy = ox.ravel(), oy.ravel()
    punit = SCALE2*reciprocal_scale//(2*m*m)
    sunit = SCALE2*reciprocal_scale//m
    assert punit*2*m*m == SCALE2*reciprocal_scale
    assert sunit*m == SCALE2*reciprocal_scale
    # Guard every product used by signed int64 Hamiltonian arithmetic.
    dmax = 2*(wing+2)
    assert 2*dmax*dmax*punit+int(np.abs(fields[0]).max()) < 2**62
    assert int(np.max(gc[3]))*(2*dmax*sunit)+BSCALE < 2**62
    use_torch = backend != 'numpy'
    if use_torch:
        import torch
        device = backend
        fields = tuple(torch.as_tensor(a, device=device) for a in fields)
        gc = tuple(torch.as_tensor(a.reshape(-1), device=device) for a in gc)
        ox, oy = torch.as_tensor(ox, device=device), torch.as_tensor(oy, device=device)
        arange = lambda a, b: torch.arange(a, b, device=device, dtype=torch.int64)
        minimum = lambda a: a.max(dim=1).values
        where = torch.where
        if backend == 'cuda': torch.cuda.synchronize()
    else:
        gc = tuple(a.reshape(-1) for a in gc)
        arange = lambda a, b: np.arange(a, b, dtype=np.int64)
        minimum = lambda a: a.max(axis=1)
        where = np.where
    maxima = {s: 0 for s in (-1, 1)}
    filtered = {s: 0 for s in (-1, 1)}
    counts = {s: [0, 0] for s in (-1, 1)}
    batch = max(1, min(128, 1000000//len(ox)))
    ulo, uhi, dxlo, dxhi, dylo, dyhi = fields
    glo, ghi, blo, bhi = gc
    for begin in range(0, n*n, batch):
        ids = arange(begin, min(begin+batch, n*n))
        j, k = ids//n, ids % n
        yn, zn = -m+(2*j+1)*(m//n), -m+(2*k+1)*(m//n)
        ix = (((2*j+1)*m//(2*n))[:, None]+ox[None, :]) % m
        iy = (((2*k+1)*m//(2*n))[:, None]+oy[None, :]) % m
        ax = ((-m+2*ix-yn[:, None]+m) % (2*m))-m
        ay = ((-m+2*iy-zn[:, None]+m) % (2*m))-m
        squared = ax*ax+ay*ay
        penalty = squared*punit
        psum = (abs(ax)+abs(ay))*sunit
        for sign in (-1, 1):
            low = (ulo[ix, iy] if sign == 1 else -uhi[ix, iy])-penalty
            high = (uhi[ix, iy] if sign == 1 else -ulo[ix, iy])-penalty
            selected = (high >= minimum(low)[:, None]-threshold) & (squared <= radius_squared)
            if sign == 1:
                hterm = (bhi[ids, None]*psum+BSCALE-1)//BSCALE
                residual = uhi[ix, iy]-glo[ids, None]+hterm-penalty
            else:
                hterm = (blo[ids, None]*psum)//BSCALE
                residual = ghi[ids, None]-ulo[ix, iy]-hterm-penalty
            keep = selected & (sign*ax*sunit+slope_slack >= dxlo[ix, iy])
            keep &= sign*ax*sunit-slope_slack <= dxhi[ix, iy]
            keep &= sign*ay*sunit+slope_slack >= dylo[ix, iy]
            keep &= sign*ay*sunit-slope_slack <= dyhi[ix, iy]
            maxima[sign] = max(maxima[sign], int(where(selected, residual, 0).max()))
            filtered[sign] = max(filtered[sign], int(where(keep, residual, 0).max()))
            counts[sign][0] += int(selected.sum())
            counts[sign][1] += int(keep.sum())
    if use_torch and backend == 'cuda': torch.cuda.synchronize()
    return dict(location_side=m, center_side=n, locations=m*m, centers=n*n,
                reciprocal_scale=reciprocal_scale, backend=backend,
                L_u=interval_dict(lu), slope_restriction='L=L_u; no L_V input',
                radius=interval_dict(radius), threshold=interval_dict(theta),
                coverage_correction=interval_dict(delta),
                residual_numerators={str(s): maxima[s] for s in (-1, 1)},
                filtered_residual_numerators={str(s): filtered[s] for s in (-1, 1)},
                residual_denominator=SCALE2,
                candidate_counts={str(s): counts[s] for s in (-1, 1)},
                one_sided_bounds={str(s): interval_dict(arb(maxima[s])/SCALE2+delta) for s in (-1, 1)},
                filtered_one_sided_bounds={str(s): interval_dict(arb(filtered[s])/SCALE2+delta) for s in (-1, 1)},
                complete_bound=interval_dict(arb(max(filtered.values()))/SCALE2+delta),
                verification_seconds=time.perf_counter()-start)


def reference_error(u, side=1024):
    """Independent validation enclosure; never passed to contact_bound.

    Nodal errors give a lower bound. Bilinear interpolation of V has error
    <= (sup|V_xx|+sup|V_yy|)*h^2/8 on kink-aligned fine cells.
    Since the supplied critic is bilinear there, this controls its error too.
    """
    assert side % len(u) == 0 and side % 2 == 0
    lo, hi, *_ = critic_grid(u, side)
    ts = [trig(arb(-side+2*j)/side) for j in range(side)]
    lower = upper_error = 0
    for j in range(side):
        for k in range(side):
            value, _, _ = problem_arb_from_trig(ts[j], ts[k])
            vl, vh = floor_int(value*SCALE2), ceil_int(value*SCALE2)
            lower = max(lower, int(lo[j, k])-vh, vl-int(hi[j, k]))
            upper_error = max(upper_error, int(hi[j, k])-vl, vh-int(lo[j, k]))
    interpolation = 3*arb.pi()**2/(8*side*side)
    return dict(lower=interval_dict(arb(lower)/SCALE2)['lower'],
                upper=interval_dict(arb(upper_error)/SCALE2+interpolation)['upper'],
                nodal_lower_numerator=lower, nodal_upper_numerator=upper_error,
                denominator=SCALE2, validation_side=side,
                interpolation_correction=interval_dict(interpolation))


def grid_objective_maximum(values, n, scale, wing, backend='numpy'):
    """Exact maximum on each search square by two 1D quadratic-max passes."""
    m=len(values)
    unit=SCALE2*scale//(2*m*m)
    offsets=np.arange(-wing,wing+1,dtype=np.int64)
    centers=-m+(2*np.arange(n,dtype=np.int64)+1)*(m//n)
    indices=(((2*np.arange(n,dtype=np.int64)+1)*m//(2*n))[:,None]+offsets)%m
    displacement=((-m+2*indices-centers[:,None]+m)%(2*m))-m
    penalties=displacement*displacement*unit
    if backend!='numpy':
        import torch
        vals=torch.as_tensor(values,device=backend)
        ix=torch.as_tensor(indices,device=backend)
        pen=torch.as_tensor(penalties,device=backend)
        first=torch.empty((n,m),device=backend,dtype=torch.int64)
        result=torch.empty((n,n),device=backend,dtype=torch.int64)
        for begin in range(0,n,8):
            end=min(begin+8,n)
            first[begin:end]=(vals[ix[begin:end],:]-pen[begin:end,:,None]).max(dim=1).values
        for begin in range(0,n,8):
            end=min(begin+8,n)
            result[:,begin:end]=(first[:,ix[begin:end]]-pen[None,begin:end,:]).max(dim=2).values
        return result.cpu().numpy()
    first=np.empty((n,m),dtype=np.int64)
    result=np.empty((n,n),dtype=np.int64)
    for begin in range(0,n,8):
        end=min(begin+8,n)
        first[begin:end]=(values[indices[begin:end],:]-penalties[begin:end,:,None]).max(axis=1)
    for begin in range(0,n,8):
        end=min(begin+8,n)
        result[:,begin:end]=(first[:,indices[begin:end]]-penalties[None,begin:end,:]).max(axis=2)
    return result


def sparse_contact_bound(u,m,n,reciprocal_scale,backend='numpy',cached_centers=None):
    """Evaluate exactly the filtered candidate set without dense pair enumeration.

    The objective remains nonseparable through u; the quadratic kernel permits
    two exact maximization passes. Derivative boxes enumerate compatible centers
    for each location node. Both steps use signed integer arithmetic.
    """
    start=time.perf_counter()
    assert m>=n and m%n==0
    scale=reciprocal_scale
    q,r=arb(2).sqrt()/m,arb(2).sqrt()/n
    lu=critic_lipschitz(u)
    radius=lu/scale+q+r
    assert radius<1
    theta=2*lu*r+scale*r*r+(lu+radius*scale)*q
    c=constants()
    delta=lu*q+(c['L_ell']+c['L_f']*radius*scale)*r+(c['M_f']+radius)*scale*(q+r)
    threshold=ceil_int(theta*SCALE2)
    radius_squared=ceil_int(radius*radius*m*m)
    slack=ceil_int(scale*(arb(1)/m+arb(1)/n)*SCALE2)
    fields=critic_grid(u,m)
    gc=cached_centers if cached_centers is not None else center_data(n)
    ulo,uhi,dxlo,dxhi,dylo,dyhi=[a.reshape(-1) for a in fields]
    glo,ghi,blo,bhi=[a.reshape(-1) for a in gc]
    wing=ceil_int(radius*m/2)+2
    punit=SCALE2*scale//(2*m*m);sunit=SCALE2*scale//m
    assert punit*2*m*m==SCALE2*scale and sunit*m==SCALE2*scale
    dmax=ceil_int(radius*m)+2
    assert int(bhi.max())*2*dmax*sunit+BSCALE<2**62
    assert 2*dmax*dmax*punit+int(np.abs(ulo).max())<2**62
    maxima={};counts={};enumerated={}
    for sign in (-1,1):
        low=fields[0] if sign==1 else -fields[1]
        gridmax=grid_objective_maximum(low,n,scale,wing,backend).reshape(-1)
        axlo,axhi=(dxlo,dxhi) if sign==1 else (-dxhi,-dxlo)
        aylo,ayhi=(dylo,dyhi) if sign==1 else (-dyhi,-dylo)
        best=count=total=0
        denom=2*(m//n)*SCALE2*scale
        # Coordinate slack is exact because all grid dimensions are powers of two.
        # A location x can represent a contact only at these center coordinates.
        for begin in range(0,m*m,4096):
            end=min(begin+4096,m*m)
            ids=np.arange(begin,end,dtype=np.int64)
            ix,iy=ids//m,ids%m
            ex=(2*ix-(1+m//n)-(m//n))*SCALE2*scale-axhi[ids]*m
            ey=(2*iy-(1+m//n)-(m//n))*SCALE2*scale-ayhi[ids]*m
            fx=(2*ix+(1+m//n)-(m//n))*SCALE2*scale-axlo[ids]*m
            fy=(2*iy+(1+m//n)-(m//n))*SCALE2*scale-aylo[ids]*m
            jl,kl=-((-ex)//denom),-((-ey)//denom)
            jh,kh=fx//denom,fy//denom
            nx,ny=np.maximum(0,jh-jl+1),np.maximum(0,kh-kl+1)
            assert int(nx.max())<=n and int(ny.max())<=n
            multiplicity=nx*ny
            starts=np.cumsum(multiplicity)-multiplicity
            rep=np.repeat(np.arange(len(ids)),multiplicity)
            if len(rep)==0:continue
            offset=np.arange(len(rep),dtype=np.int64)-starts[rep]
            j=(jl[rep]+offset//ny[rep])%n
            k=(kl[rep]+offset%ny[rep])%n
            ii=ids[rep]; jj=j*n+k
            ax=((2*ix[rep]-(2*j+1)*(m//n)+m)%(2*m))-m
            ay=((2*iy[rep]-(2*k+1)*(m//n)+m)%(2*m))-m
            squared=ax*ax+ay*ay; penalty=squared*punit
            high=(uhi[ii] if sign==1 else -ulo[ii])-penalty
            keep=(high>=gridmax[jj]-threshold)&(squared<=radius_squared)
            keep&=sign*ax*sunit+slack>=dxlo[ii]
            keep&=sign*ax*sunit-slack<=dxhi[ii]
            keep&=sign*ay*sunit+slack>=dylo[ii]
            keep&=sign*ay*sunit-slack<=dyhi[ii]
            total+=len(rep);count+=int(keep.sum())
            ii,jj=ii[keep],jj[keep]
            slope=(abs(ax[keep])+abs(ay[keep]))*sunit
            penalty=penalty[keep]
            if sign==1:
                values=uhi[ii]-glo[jj]+(bhi[jj]*slope+BSCALE-1)//BSCALE-penalty
            else:
                values=ghi[jj]-ulo[ii]-(blo[jj]*slope)//BSCALE-penalty
            if len(values):best=max(best,int(values.max()))
        maxima[str(sign)]=best;counts[str(sign)]=count;enumerated[str(sign)]=total
    return dict(location_side=m,center_side=n,locations=m*m,centers=n*n,
                reciprocal_scale=scale,backend=backend,implementation='sparse filtered pairs with two quadratic-max passes',
                L_u=interval_dict(lu),slope_restriction='L=L_u; no L_V input',
                radius=interval_dict(radius),threshold=interval_dict(theta),coverage_correction=interval_dict(delta),
                filtered_residual_numerators=maxima,residual_denominator=SCALE2,
                filtered_candidate_counts=counts,stationarity_pairs_enumerated=enumerated,
                filtered_one_sided_bounds={s:interval_dict(arb(v)/SCALE2+delta) for s,v in maxima.items()},
                complete_bound=interval_dict(arb(max(maxima.values()))/SCALE2+delta),
                verification_seconds=time.perf_counter()-start)
