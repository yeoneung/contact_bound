"""Bernstein distance verifier preserved for the two-dimensional audit."""
import heapq,itertools,time
from flint import arb
from interval_utils import ceil_int,floor_int
from twod_polynomials import SCALE,power_coefficients,bernstein_coefficients

def split_bernstein(values,axis):
    """Exact half-interval de Casteljau subdivision of a tensor coefficient array."""
    a=values if axis==0 else [list(row) for row in zip(*values)]
    n=len(a)-1;m=len(a[0])
    left=[[arb(0) for _ in range(m)] for _ in range(n+1)]
    right=[[arb(0) for _ in range(m)] for _ in range(n+1)]
    for j in range(m):
        column=[a[i][j] for i in range(n+1)]
        left[0][j],right[n][j]=column[0],column[-1]
        for k in range(1,n+1):
            column=[(column[i]+column[i+1])/2 for i in range(len(column)-1)]
            left[k][j],right[n-k][j]=column[0],column[-1]
    if axis:
        left,right=[list(row) for row in zip(*left)],[list(row) for row in zip(*right)]
    return left,right


def bernstein_norm_bounds(values):
    upper=max(ceil_int(v.abs_upper()*SCALE) for row in values for v in row)
    corners=[values[i][j] for i in (0,len(values)-1) for j in (0,len(values[0])-1)]
    lower=max(max(0,floor_int(v*SCALE),floor_int(-v*SCALE)) for v in corners)
    return lower,upper


def bilinear_distance(record,u,tolerance,base=1<<24):
    start=time.perf_counter();n=len(u);assert n%2==0 and (n&(n-1))==0
    coeffs=bernstein_coefficients(power_coefficients(record))
    original=[[coeffs[i,j] for j in range(coeffs.ncols())] for i in range(coeffs.nrows())]
    pieces={(0,0):original}
    # Auxiliary coordinates t=1-|x| have n/2 patches along each positive half.
    for axis in [0,1]:
        for _ in range((n//2).bit_length()-1):
            refined={}
            for index,a in pieces.items():
                for k,child in enumerate(split_bernstein(a,axis)):
                    key=list(index);key[axis]=2*key[axis]+k;refined[tuple(key)]=child
            pieces=refined
    heap=[];serial=itertools.count();lower=0
    degree=coeffs.nrows()-1
    for i in range(n):
        for j in range(n):
            # On the negative half t grows; on the positive half t decreases.
            ti=i if i<n//2 else n-1-i
            tj=j if j<n//2 else n-1-j
            a=pieces[(ti,tj)]
            if i>=n//2:a=a[::-1]
            if j>=n//2:a=[row[::-1] for row in a]
            vertices=[int(u[i,j]),int(u[(i+1)%n,j]),int(u[i,(j+1)%n]),int(u[(i+1)%n,(j+1)%n])]
            difference=[]
            for k in range(degree+1):
                row=[]
                for l in range(degree+1):
                    value=((degree-k)*(degree-l)*vertices[0]+k*(degree-l)*vertices[1]
                           +(degree-k)*l*vertices[2]+k*l*vertices[3])
                    row.append(arb(value)/(degree*degree*base)-a[k][l])
                difference.append(row)
            lo,hi=bernstein_norm_bounds(difference);lower=max(lower,lo)
            heapq.heappush(heap,(-hi,next(serial),0,difference))
    tol=floor_int(arb(tolerance)*SCALE);splits=0
    while -heap[0][0]-lower>tol:
        _,_,depth,a=heapq.heappop(heap)
        for child in split_bernstein(a,depth%2):
            lo,hi=bernstein_norm_bounds(child);lower=max(lower,lo)
            heapq.heappush(heap,(-hi,next(serial),depth+1,child))
        splits+=1
        assert splits<100000,'Distance tolerance not achieved within subdivision limit'
    return dict(lower_numerator=lower,upper_numerator=-heap[0][0],denominator=SCALE,
                lower=lower/SCALE,upper=-heap[0][0]/SCALE,initial_cells=n*n,
                subdivisions=splits,seconds=time.perf_counter()-start)
