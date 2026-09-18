"""Tight signed derivative enclosures for smooth heads and their finite minimum."""
from flint import arb, arb_mat
from .verified_arb import CertifiedCritic, SCALE, ceil_int, floor_int


class SecondOrderCritic(CertifiedCritic):
    def heads_jet(self, points, second=True):
        features, first, curvature = [], [], []
        for x in points:
            frequencies = [arb.pi()*k for k in range(1, self.freqs+1)]
            sc = [(x*k).sin_cos() for k in frequencies]
            features.append([s for s,c in sc]+[c for s,c in sc])
            first.append([k*c for k,(s,c) in zip(frequencies,sc)]
                         +[-k*s for k,(s,c) in zip(frequencies,sc)])
            if second:
                curvature.append([-k*k*s for k,(s,c) in zip(frequencies,sc)]
                                 +[-k*k*c for k,(s,c) in zip(frequencies,sc)])
        f,df=arb_mat(features),arb_mat(first)
        ddf=arb_mat(curvature) if second else None
        outputs=[]
        for layers in self.heads:
            a,da,dda=f,df,ddf
            for layer,(w,biases) in enumerate(layers):
                z,dz=a*w,da*w
                ddz=dda*w if second else None
                rows,drows,ddrows=[],[],[]
                for i in range(len(points)):
                    row,drow,ddrow=[],[],[]
                    for j,b in enumerate(biases):
                        v=z[i,j]+b
                        if layer==len(layers)-1:
                            row.append(v);drow.append(dz[i,j])
                            if second:ddrow.append(ddz[i,j])
                        else:
                            s=1/(1+(-v).exp())
                            derivative=s*(1+v*(1-s))
                            row.append(v*s);drow.append(dz[i,j]*derivative)
                            if second:
                                curvature=s*(1-s)*(2+v*(1-2*s))
                                ddrow.append(ddz[i,j]*derivative+dz[i,j]**2*curvature)
                    rows.append(row);drows.append(drow)
                    if second:ddrows.append(ddrow)
                a,da=arb_mat(rows),arb_mat(drows)
                if second:dda=arb_mat(ddrows)
            outputs.append(([a[i,0] for i in range(len(points))],
                            [da[i,0] for i in range(len(points))],
                            [dda[i,0] for i in range(len(points))] if second else None))
        return outputs

    def tight_derivative_boxes(self, centers, radius):
        mid=self.heads_jet(centers,False)
        boxes=[arb(x,radius) for x in centers]
        full=self.heads_jet(boxes,True)
        lows,highs=[],[]
        for i in range(len(centers)):
            ceiling=min(ceil_int(h[0][i]*SCALE) for h in full)
            possible=[k for k,h in enumerate(full) if floor_int(h[0][i]*SCALE)<=ceiling]
            enclosures=[mid[k][1][i]+arb(0,full[k][2][i].abs_upper()*radius) for k in possible]
            lows.append(min(floor_int(d*SCALE) for d in enclosures))
            highs.append(max(ceil_int(d*SCALE) for d in enclosures))
        return lows,highs
