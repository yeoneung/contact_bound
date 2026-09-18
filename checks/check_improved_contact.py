"""Independent arithmetic and contact-coverage checks for the sharper estimator."""
from pathlib import Path
import json,sys,unittest
import numpy as np
import torch
import mpmath as mp
from flint import arb,ctx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'code/experiments'))
from hjrl.verified_stationarity import SecondOrderCritic
from hjrl.verified_arb import SCALE,floor_int,ceil_int,ridge_value
from improve_contact_estimator import envelope_bound


def independent_head(state,head,branches,x):
    prefix='' if branches==1 else f'heads.{head}.'
    a=[mp.sin(mp.pi*k*x) for k in range(1,9)]+[mp.cos(mp.pi*k*x) for k in range(1,9)]
    for index in (0,2,4,6):
        w=state[f'{prefix}net.{index}.weight'].tolist()
        b=state[f'{prefix}net.{index}.bias'].tolist()
        z=[mp.fsum(mp.mpf(v)*s for v,s in zip(row,a))+mp.mpf(bias) for row,bias in zip(w,b)]
        a=[v/(1+mp.exp(-v)) for v in z] if index<6 else z
    return a[0]


class ImprovedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ctx.prec=128;mp.mp.dps=70
        cls.states=torch.load(ROOT/'results/verified_neural_models.pt',map_location='cpu',weights_only=True)

    def test_head_curvatures_and_tight_derivative_boxes(self):
        for name,branches in (('MLP',1),('Min2',2)):
            state=self.states[name];model=SecondOrderCritic(state,branches)
            for numerator in (-1,2):
                x=mp.mpf(numerator)/16
                jets=model.heads_jet([arb(numerator)/16])
                for head in range(branches):
                    f=lambda z:independent_head(state,head,branches,z)
                    second=mp.diff(f,x,2)
                    self.assertTrue(jets[head][2][0].contains(arb(str(second))))
                lo,hi=model.tight_derivative_boxes([arb(numerator)/16],arb(1)/65536)
                for direction in (-1,0,1):
                    point=x+mp.mpf(direction)/65536
                    heads=[independent_head(state,h,branches,point) for h in range(branches)]
                    active=min(range(branches),key=lambda h:heads[h])
                    derivative=mp.diff(lambda z:independent_head(state,active,branches,z),point)
                    self.assertLessEqual(mp.mpf(lo[0])/SCALE,derivative)
                    self.assertGreaterEqual(mp.mpf(hi[0])/SCALE,derivative)

    def test_center_cell_candidates_against_continuous_cosine_contacts(self):
        # Every objective is strictly concave on its contact window:
        # |u''| <= pi^2/10 < m. Roots are independent continuous optimizers.
        M,N=128,32
        q,r=mp.mpf(1)/M,mp.mpf(1)/N
        lu=mp.pi/10
        nodes=[-1+mp.mpf(2*i)/M for i in range(M)]
        for m in (4,16,64):
            R=lu/m+r+q
            theta=2*lu*r+m*r*r+(lu+m*R)*q
            for sigma in (-1,1):
                for j in (0,7,15,31):
                    yj=-1+mp.mpf(2*j+1)/N
                    objectives=[]
                    for x in nodes:
                        d=(x-yj+1)%2-1
                        objectives.append(sigma*mp.cos(mp.pi*x)/10-m*d*d/2)
                    maximum=max(objectives)
                    for phase in (-1,0,1):
                        y=yj+phase*r
                        x=mp.findroot(lambda z:-sigma*mp.pi*mp.sin(mp.pi*z)/10-m*(z-y),(y-lu/m,y+lu/m))
                        i=int(mp.floor((x+1)*M/2+mp.mpf('0.5')))%M
                        delta=(nodes[i]-yj+1)%2-1
                        self.assertLessEqual(abs(delta),R+mp.mpf('1e-50'))
                        self.assertGreaterEqual(objectives[i],maximum-theta-mp.mpf('1e-50'))
                        # The necessary stationarity interval must contain the
                        # exact slope, including center-cell boundary points.
                        p=sigma*m*(x-y)
                        self.assertLessEqual(abs(p-sigma*m*delta),m*(q+r)+mp.mpf('1e-50'))

    def test_manufactured_wrong_kinks_both_signs(self):
        M=512
        xs=[arb(-M+2*i)/M for i in range(M)]
        for numerator in (1,2,4):
            c=arb(numerator)/10
            values=[ridge_value(x)-c*(-(1-abs(x))).exp() for x in xs]
            lo=np.array([floor_int(v*SCALE) for v in values],dtype=np.int64)
            hi=np.array([ceil_int(v*SCALE) for v in values],dtype=np.int64)
            for m in (8,32):
                result=envelope_bound(lo,hi,arb.pi()/2+c,m,128)
                # The exact error is c, entirely on the underestimation side.
                bound=result['restricted_one_sided_bounds']['-1']
                self.assertTrue(arb(bound['upper_numerator'])/SCALE>=c)

    def test_saved_ablation_consistency_and_true_error(self):
        shifted=json.loads((ROOT/'results/improved_contact_estimator.json').read_text())
        filtered=json.loads((ROOT/'results/stationary_contact_estimator.json').read_text())
        reference=json.loads((ROOT/'results/estimator_comparison.json').read_text())
        self.assertTrue(shifted['complete'] and filtered['complete'])
        for row in filtered['records']:
            old=next(r for r in shifted['records'] if r['critic']==row['critic'])
            error=next(r for r in reference['records'] if r['critic']==row['critic'])['true_error_enclosure']['upper']
            for result in row['settings']:
                baseline=next(r for r in old['settings'] if
                              (r['grid_nodes'],r['centers'],r['reciprocal_scale'])==
                              (result['grid_nodes'],result['centers'],result['reciprocal_scale']))
                self.assertEqual(result['unfiltered_residual_upper'],baseline['restricted_candidate_residual_upper'])
                self.assertEqual(result['coverage_correction'],baseline['restricted_coverage_correction'])
                self.assertGreaterEqual(result['complete_bound']['upper'],error)
                for sign in ('-1','1'):
                    self.assertLessEqual(result['candidate_counts_after'][sign],result['candidate_counts_before'][sign])
                    self.assertLessEqual(result['one_sided_bounds'][sign]['upper'],baseline['restricted_one_sided_bounds'][sign]['upper'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ImprovedChecks))
    report={'checks_run':result.testsRun,'successful':result.wasSuccessful(),
            'independent_second_derivatives':6,'cell_derivative_points':12,
            'continuous_contact_candidates':72,'wrong_kink_settings':6}
    (ROOT/'checks/improved_contact_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    sys.exit(not result.wasSuccessful())
