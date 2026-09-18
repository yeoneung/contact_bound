"""Independent high-precision checks of the Arb network and held-cost implementation."""
from pathlib import Path
import json
import sys
import unittest
import torch
import mpmath as mp
from flint import arb,ctx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from hjrl.verified_arb import CertifiedCritic,held_cost


class VerificationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx.prec=128
        mp.mp.dps=90
        cls.states=torch.load(ROOT/"results/verified_neural_models.pt",map_location='cpu',weights_only=True)

    def test_network_against_independent_mpmath(self):
        for name,branches in (("MLP",1),("Min2",2)):
            sd=self.states[name]
            net=CertifiedCritic(sd,branches)
            for numerator in (-7,0,5):
                x=mp.mpf(numerator)/8
                outputs=[]
                for head in range(branches):
                    prefix='' if branches==1 else f'heads.{head}.'
                    a=[mp.sin(mp.pi*k*x) for k in range(1,9)]+[mp.cos(mp.pi*k*x) for k in range(1,9)]
                    for index in (0,2,4,6):
                        w=sd[f'{prefix}net.{index}.weight'].tolist()
                        b=sd[f'{prefix}net.{index}.bias'].tolist()
                        z=[mp.fsum(mp.mpf(v)*s for v,s in zip(row,a))+mp.mpf(bias) for row,bias in zip(w,b)]
                        a=[v/(1+mp.exp(-v)) for v in z] if index<6 else z
                    outputs.append(a[0])
                reference=arb(str(min(outputs)))
                enclosure=net.evaluate([arb(numerator)/8])[0]
                self.assertTrue(enclosure.contains(reference),(name,numerator,enclosure,reference))

    def test_held_cost_against_quadrature(self):
        h=mp.mpf(1)/32
        for xn in (-128,-127,-1,0,1,127):
            x=mp.mpf(xn)/128
            for action in (-1,0,1):
                def integrand(t):
                    z=x+action*t
                    g=1-abs(mp.sin(mp.pi*z/2))+mp.pi/2*abs(mp.cos(mp.pi*z/2))
                    return mp.exp(-t)*g
                breaks=[mp.mpf(0),h]
                if action:
                    for k in (-1,0,1):
                        t=(k-x)/action
                        if 0<t<h:breaks.append(t)
                reference=mp.quad(integrand,sorted(breaks))
                enclosure=held_cost(arb(xn)/128,action,arb(1)/32)
                self.assertTrue(enclosure.contains(arb(str(reference))),(xn,action))

    def test_exact_periodicity(self):
        for name,branches in (("MLP",1),("Min2",2)):
            net=CertifiedCritic(self.states[name],branches)
            v=net.evaluate([arb(-1),arb(1)])
            self.assertTrue((v[1]-v[0]).contains(0))

    def test_hold_bias_dominates_independent_point_values(self):
        sys.path.insert(0,str(ROOT/'code/experiments'))
        from verify_neural_policy import hold_bias
        bh=hold_bias()
        gamma=(-arb(1)/32).exp()
        from hjrl.verified_arb import ridge_value
        for i in range(129):
            x=arb(1)-arb(i)/(128*32)
            values=[held_cost(x,a,arb(1)/32)+gamma*ridge_value(x+arb(a)/32)-ridge_value(x)
                    for a in (-1,0,1)]
            defect=values[0].min(values[1]).min(values[2])
            self.assertTrue(bh>=defect)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(VerificationChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'checks_run':result.testsRun,'successful':result.wasSuccessful(),
            'independent_network_points':6,'independent_integrals':18,'hold_bias_points':129}
    (ROOT/'checks/verified_neural_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    sys.exit(not result.wasSuccessful())
