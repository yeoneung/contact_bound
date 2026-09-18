"""Independent or analytic checks for the new verified comparison routines."""
from pathlib import Path
import json
import sys
import unittest
import mpmath as mp
import torch
from flint import arb, ctx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
sys.path.insert(0, str(ROOT/'code/experiments'))
from compare_verified_estimators import CertifiedJet, contact_witness, bellman_bound
from hjrl.verified_arb import ridge_cost, ridge_value, held_cost, SCALE
import verify_neural_policy as original


class AnalyticCosine:
    def evaluate(self, xs):
        return [(arb.pi()*x).cos()/10 for x in xs]

    def jet(self, xs):
        return self.evaluate(xs), [-arb.pi()*(arb.pi()*x).sin()/10 for x in xs]


class ComparisonChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx.prec = 128
        mp.mp.dps = 70
        cls.states = torch.load(ROOT/'results/verified_neural_models.pt',
                                map_location='cpu', weights_only=True)

    def test_signed_derivatives_against_mpmath_differentiation(self):
        for name, branches in (('MLP', 1), ('Min2', 2)):
            state = self.states[name]
            net = CertifiedJet(state, branches)
            def value(x):
                heads = []
                for head in range(branches):
                    prefix = '' if branches == 1 else f'heads.{head}.'
                    a = [mp.sin(mp.pi*k*x) for k in range(1, 9)] + [mp.cos(mp.pi*k*x) for k in range(1, 9)]
                    for index in (0, 2, 4, 6):
                        w = state[f'{prefix}net.{index}.weight'].tolist()
                        b = state[f'{prefix}net.{index}.bias'].tolist()
                        z = [mp.fsum(mp.mpf(v)*s for v, s in zip(row, a))+mp.mpf(bias)
                             for row, bias in zip(w, b)]
                        a = [v/(1+mp.exp(-v)) for v in z] if index < 6 else z
                    heads.append(a[0])
                return min(heads)
            for numerator in (-3, 1):
                x = mp.mpf(numerator)/8
                derivative = mp.diff(value, x)
                _, exact = net.jet([arb(numerator)/8])
                _, interval = net.jet([arb(arb(numerator)/8, arb(1)/2**20)])
                self.assertTrue(exact[0].contains(arb(str(derivative))))
                self.assertTrue(interval[0].contains(exact[0]))

    def test_witness_global_optimization_against_known_contact(self):
        # |u''| <= pi^2/10 < 64: both penalized objectives have a unique
        # stationary maximizer x=y=0, giving a known exact residual.
        model = AnalyticCosine()
        for sigma in (-1, 1):
            result = contact_witness(model, arb.pi()/10, 64, arb(0), sigma,
                                     target_width=arb(1)/2**22)
            exact = (sigma*(arb(1)/10-ridge_cost(arb(0)))).max(arb(0))
            lower = arb(result['lower_numerator'])/SCALE
            upper = arb(result['upper_numerator'])/SCALE
            self.assertTrue(lower <= exact and upper >= exact)
            self.assertLess(result['residual_upper']-result['residual_lower'], 0.0001)

    def test_hold_bias_all_comparison_steps(self):
        for m in (16, 32, 64, 128):
            original.H = h = arb(1)/m
            bh = original.hold_bias()
            gamma = (-h).exp()
            for i in range(65):
                x = 1-h*i/64
                defects = [held_cost(x, a, h)+gamma*ridge_value(x+a*h)-ridge_value(x)
                           for a in (-1, 0, 1)]
                defect = defects[0].min(defects[1]).min(defects[2])
                self.assertTrue(bh >= defect)

    def test_bellman_full_domain_bound_for_zero_critic(self):
        import numpy as np
        nodes = 256
        zeros = np.zeros(nodes, dtype=np.int64)
        result = bellman_bound(zeros, zeros, arb(0), 16)
        # V ranges from zero to one, so this critic has exact uniform error one.
        self.assertGreaterEqual(result['complete_bound']['lower'], 1)
        # Independently evaluate off-grid Bellman residuals by high-precision
        # quadrature, including integration across a kink.
        h = mp.mpf(1)/16
        for xn in (-999, -19, 3, 911):
            x = mp.mpf(xn)/1024
            qs = []
            for action in (-1, 0, 1):
                def integrand(t):
                    z = x+action*t
                    return mp.exp(-t)*(1-abs(mp.sin(mp.pi*z/2))+mp.pi/2*abs(mp.cos(mp.pi*z/2)))
                breaks = [mp.mpf(0), h]
                if action:
                    for k in (-1, 0, 1):
                        t = (k-x)/action
                        if 0 < t < h:
                            breaks.append(t)
                qs.append(mp.quad(integrand, sorted(breaks)))
            bound = mp.mpf(result['continuous_residual_upper']['upper_numerator'])/SCALE
            self.assertLessEqual(min(qs), bound)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonChecks))
    report = {'checks_run': result.testsRun, 'successful': result.wasSuccessful(),
              'signed_derivative_points': 4, 'analytic_contact_signs': 2,
              'hold_bias_points': 260, 'off_grid_quadrature_points': 4}
    (ROOT/'checks/estimator_comparison_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
