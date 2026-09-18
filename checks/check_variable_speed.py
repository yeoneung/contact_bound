"""Independent high-precision checks for the variable-speed verification."""
from pathlib import Path
import json,sys,unittest
import mpmath as mp
import numpy as np
import torch
from flint import arb,ctx

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/experiments')]
from hjrl.verified_arb import SCALE,floor_int,ceil_int,CertifiedCritic
from hjrl.verified_variable_speed import VariableSpeed
from compare_variable_speed import load_critic,build_operator,bellman_bound,interval_product,contact_bound

mp.mp.dps=70;ctx.prec=128


def inside(value,enclosure):
    assert mp.mpf(floor_int(enclosure*SCALE))/SCALE<=value<=mp.mpf(ceil_int(enclosure*SCALE))/SCALE


def mp_flow(x,a,t,beta):
    k=mp.sqrt((1-beta)/(1+beta))
    theta=mp.atan2(k*mp.sin(mp.pi*x/2),mp.cos(mp.pi*x/2))
    z=theta+a*mp.pi*mp.sqrt(1-beta*beta)*t/2
    return 2*mp.atan2(mp.sin(z),k*mp.cos(z))/mp.pi


def mp_cost(x,beta):
    return 1-abs(mp.sin(mp.pi*x/2))+(1+beta*mp.cos(mp.pi*x))*mp.pi*abs(mp.cos(mp.pi*x/2))/2


def mp_held(x,a,h,beta):
    if a==0:return (1-mp.exp(-h))*mp_cost(x,beta)
    k=mp.sqrt((1-beta)/(1+beta));omega=mp.pi*mp.sqrt(1-beta*beta)/2
    theta=mp.atan2(k*mp.sin(mp.pi*x/2),mp.cos(mp.pi*x/2))
    cuts=[mp.mpf(0),h]
    for j in range(-3,4):
        t=(j*mp.pi/2-theta)/(a*omega)
        if 0<t<h:cuts.append(t)
    return mp.quad(lambda t:mp.exp(-t)*mp_cost(mp_flow(x,a,t,beta),beta),sorted(cuts))


class Checks(unittest.TestCase):
    def test_contact_known_error(self):
        M=512;lv=arb.pi()/2;lip=ceil_int(lv*SCALE)
        for beta in (0,1,2):
            for shift in (arb(1)/8,arb(1)/2):
                u=[1-abs((arb.pi()*arb(-M+2*i)/(2*M)).sin())-shift for i in range(M)]
                c={'ulo':np.array([floor_int(z*SCALE) for z in u],dtype=np.int64),
                   'uhi':np.array([ceil_int(z*SCALE) for z in u],dtype=np.int64),
                   'dlo':np.full(M,-lip,dtype=np.int64),'dhi':np.full(M,lip,dtype=np.int64),
                   'lu':arb(lip)/SCALE,'base_seconds':0,'derivative_seconds':0}
                r=contact_bound(VariableSpeed(beta),c,16,128)
                self.assertGreaterEqual(r['complete_bound']['upper'],float(shift))

    def test_flow_and_ode(self):
        count=0
        for ib in (0,1,2):
            p=VariableSpeed(ib);beta=mp.mpf(ib)/4
            for numerator in (-63,-32,-1,0,17,63):
                x=mp.mpf(numerator)/64
                for a in (-1,1):
                    t=mp.mpf(1)/16;z=mp_flow(x,a,t,beta)
                    inside(z,p.flow(arb(numerator)/64,a,arb(1)/16))
                    travel=mp.quad(lambda y:1/(1+beta*mp.cos(mp.pi*y)),[x,z])
                    self.assertLess(abs(travel-a*t),mp.mpf('1e-60'))
                    count+=1
        self.assertEqual(count,36)

    def test_cost_quadrature(self):
        count=0
        for ib in (0,1,2):
            p=VariableSpeed(ib);beta=mp.mpf(ib)/4
            for numerator,a in [(-63,-1),(-1,1),(1,-1),(63,1),(17,-1),(0,1),(-64,-1),(21,0)]:
                exact=mp_held(mp.mpf(numerator)/64,a,mp.mpf(1)/16,beta)
                for panels in (2,8):
                    enclosure,_=p.held_cost(arb(numerator)/64,a,arb(1)/16,panels)
                    inside(exact,enclosure)
                count+=1
        self.assertEqual(count,24)

    def test_hold_bias(self):
        for ib in (0,1,2):
            p=VariableSpeed(ib);beta=mp.mpf(ib)/4;h=mp.mpf(1)/16
            bound=mp.mpf(ceil_int(p.hold_bias(arb(1)/16)*SCALE))/SCALE
            for j in range(17):
                d=h*j/16
                x=mp_flow(mp.mpf(1),-1,d,beta)
                value=1-abs(mp.sin(mp.pi*x/2))
                stop=(1-mp.exp(-h))*(mp_cost(x,beta)-value)
                end=mp_flow(x,1,h,beta)
                move=mp_held(x,1,h,beta)+mp.exp(-h)*(1-abs(mp.sin(mp.pi*end/2)))-value
                self.assertLessEqual(min(stop,move),bound)

    def test_successor_enclosures_and_integer_products(self):
        c=load_critic('Min2');p=VariableSpeed(2)
        arrays,meta=build_operator(p,256,16,2,128)
        result=bellman_bound(p,c,arrays,meta)
        self.assertGreater(result['complete_bound']['upper'],0.05533270012438152)
        states=torch.load(ROOT/'results/verified_neural_models.pt',map_location='cpu',weights_only=True)
        net=CertifiedCritic(states['Min2'],2)
        for i,col in [(0,1),(0,2),(1,1),(63,2),(128,1),(255,2)]:
            ix=int(arrays['indices'][i,col]);M=len(c['ulo'])
            dl=min(int(c['dlo'][(ix+j)%M]) for j in (-1,0,1))
            dh=max(int(c['dhi'][(ix+j)%M]) for j in (-1,0,1))
            vl,vh=interval_product([dl],[dh],[arrays['delta_lo'][i,col]],[arrays['delta_hi'][i,col]])
            lo,hi=int(vl[0])+int(c['ulo'][ix]),int(vh[0])+int(c['uhi'][ix])
            X=p.flow(arb(-256+2*i)/256,(0,-1,1)[col],arb(1)/16)
            value=net.evaluate([X])[0]
            self.assertLessEqual(lo,floor_int(value*SCALE));self.assertGreaterEqual(hi,ceil_int(value*SCALE))
        values=[(-3*SCALE,-SCALE,7,19),(-11,5,-9,8),(1,1,1,1)]
        for al,ah,bl,bh in values:
            lo,hi=interval_product(np.array([al]),np.array([ah]),np.array([bl]),np.array([bh]))
            exact=[mp.mpf(a)*b/SCALE for a in (al,ah) for b in (bl,bh)]
            self.assertLessEqual(int(lo[0]),min(exact));self.assertGreaterEqual(int(hi[0]),max(exact))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'successful':result.wasSuccessful(),'checks_run':result.testsRun,'flow_points':36,
            'quadrature_cases':24,'panels_per_case':[2,8],'hold_bias_points':51,'precision_decimal_digits':70,
            'known_contact_error_cases':6,'direct_successor_critic_checks':6}
    (ROOT/'checks/variable_speed_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():raise SystemExit(1)
