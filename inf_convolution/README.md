# Verified inf-convolution experiments

Numerical implementations and saved interval bounds for inf-convolution,
branch correction, and error estimation for Hamilton–Jacobi equations.
The experiments include periodic analytic profiles, asymmetric spline inputs,
local reconstruction, polynomial branch fits in two dimensions, residual
refinement, and variable branch counts on polygons and a coupled 3D example.

## Setup

Use Python 3.12 and run commands from this directory:

```text
python -m pip install -r requirements.txt
```

The saved computations use exact rational input coefficients and outward-rounded
Arb bounds. Timings record the original sequential runs and depend on hardware.
Sampled values propose candidates; full-domain interval bounds certify them.
Unresolved boxes remain in the reported upper bound when a work limit is reached.

## Verify saved computations

```text
python code/check_reconstruction.py
python code/check_branch_stability.py
python code/check_automatic_branches.py
python code/check_robustness.py --full-replay
python code/check_adaptive_extensions.py
python code/check_residual_refinement.py
python code/check_strong_refinement.py
python code/check_free_branches.py
```

The first two checks cover analytic profiles and branch identities. The remaining
commands replay interval computations and may take substantially longer.
Additional `check_numerical_selection.py`, `check_twod_reconstruction.py`, and
`check_twod_methods.py` checks cover the retained initial experiments. The
`verification/` directory contains the completed check reports. The exact inputs
needed to replay these computations are stored in `results/`.

## Rebuild summaries and figures

```text
python scripts/rebuild_summaries.py
```

This regenerates CSV summaries, plots, numerical LaTeX tables and explanatory
summaries from the saved records. The generated tables do not require article
source files. Run drivers that change numerical inputs in a separate checkout.
The source bytes and saved inputs are checked by SHA-256; changing them
invalidates the corresponding recorded checks.

## Saved experiments

| Records | Computation |
|---|---|
| `reconstruction.json` | Analytic profiles and interval effectivity bounds |
| `automatic_branches.json` | Branch construction and error targets on asymmetric inputs |
| `robustness.json` | Frequency, curvature, and complete parameter comparisons |
| `adaptive_local.json` | Local reconstruction and cached interval evaluation |
| `unknown_junction_2d.json` | Two-dimensional fitted branch comparisons |
| `residual_refinement.json` | Shared coefficient refinement for both initializations |
| `strong_refinement.json` | Complete candidate searches with two starts |
| `free_branches.json` | Branch counts determined from the numerical input |

`MANIFEST.sha256` records the packaged source, data and figure bytes. The
public package contains numerical materials and their reproduction instructions.
