"""Fixed comparison design, specified before running the additional cases."""
from fractions import Fraction as F
from asymmetric_input import CASES

ORIGINAL_CASES=('skew','skew_strong','two_peaks')
NEW_CASES={}
for m in (1,2,4):
    for a in ('0.15','0.35'):
        for b in ('0','0.12'):
            name=f'frequency_{m}_asym_{a}_ripple_{b}'
            NEW_CASES[name]=dict(frequency=m,asymmetry=a,ripple=b,
                terms=[(m,1.,0.),(m+1,0.,float(a)),(2*m+1,.1,0.),(7*m,0.,float(b))])

def install_cases():
    for name,row in NEW_CASES.items():CASES[name]=row['terms']

STANDARD_SCALES=tuple(F(k,4) for k in range(4,41))
WIDE_SCALES=tuple(sorted(set(F(k,4) for k in range(1,65))|set(F(k,8) for k in range(8,81))))
POLYNOMIAL_PARAMETERS=tuple((width,degree) for width in range(2,17) for degree in (2,3))
GRIDS=(256,1024)
POLICY_GRIDS=tuple(2**j for j in range(6,15))
TARGETS=(F(1,25),F(1,50),F(1,100))
BUDGETS=(1,3,37)

def specification():
    return dict(original_cases=list(ORIGINAL_CASES),additional_cases=NEW_CASES,
        fixed_grids=list(GRIDS),standard_scales=[[q.numerator,q.denominator] for q in STANDARD_SCALES],
        wide_scales=[[q.numerator,q.denominator] for q in WIDE_SCALES],
        polynomial_parameters=[list(v) for v in POLYNOMIAL_PARAMETERS],
        policy_grids=list(POLICY_GRIDS),targets=[[q.numerator,q.denominator] for q in TARGETS],
        budgets=list(BUDGETS),precision_bits=192,relative_tolerance=[1,50],absolute_tolerance='1e-10',
        purpose='Compare the same input with exhaustive prescribed finite lists, record every failure, and vary the proposal budget and target.',
        scope='The additional family changes frequency, asymmetry and short-scale curvature. All twelve cases and both grids are retained, regardless of outcome.',
        timing='Sequential fixed-input searches. Construction and proposal costs are recorded separately. Timing is descriptive; policy sensitivity reports certified grid counts, not simulated runtimes.',
        pruning='An incumbent certificate may exclude a candidate only if a rigorous lower bound for that candidate exceeds the incumbent upper bound.',
        status='design_fixed_before_additional_case_evaluation')
