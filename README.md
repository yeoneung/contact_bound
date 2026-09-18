# contact_bound

Reproducible experiments for contact residuals, value-error bounds, and policy evaluation on periodic control problems.

The repository contains numerical code, fixed model weights, saved results, interval-enclosure caches, plots, and independent checks.

## Setup

The saved computations and replay commands were checked with Python 3.12 on Windows. Install the dependencies from the repository root:

```sh
python -m pip install -r requirements.txt
```

For neural training and the original navigation diagnostics, use a CUDA-enabled PyTorch installation. Independent checks and the saved-result replays run on CPU. Recorded cache metadata uses Windows path separators; the archived cache-replay commands below were validated on Windows.

## Check the saved results

```sh
python checks/run_checks.py
python checks/run_checks.py --replay
```

The first command runs six independent check suites and an exact-rational contact oracle. The second also recomputes selected configurations:

- Eight Bellman bounds, two contact bounds, and two exact-contact witnesses.
- Five configurations for the shifted and stationarity-filtered estimators.
- Eighteen configurations in the state-dependent-speed comparison.
- Two complete two-dimensional configurations using the NumPy CPU backend.

Replay checks compare numerical fields with saved results, excluding execution times. They validate and reuse cached enclosures; they do not repeat training, the entire parameter sweep, or every global derivative calculation. Check reports are written to `checks/`.

To also rebuild the three flow/cost operators used by the state-dependent replay:

```sh
python checks/reproduce_variable_speed_subset.py --full-operators
```

## Experiment files

| Experiment | Driver in `code/experiments/` | Saved output in `results/` |
| --- | --- | --- |
| Exact-rational one-dimensional certificate | `complete_contact_certificate.py` | `complete_contact_certificate.json` |
| Fixed neural critics and verified policies | `train_verified_neural.py`, `verify_neural_policy.py` | `verified_neural_*` |
| Contact versus Bellman estimators | `compare_verified_estimators.py` | `estimator_comparison.json`, `comparison_enclosures_*` |
| Shifted contact envelope | `improve_contact_estimator.py` | `improved_contact_estimator.json` |
| Stationarity filtering | `refine_contact_stationarity.py` | `stationary_contact_estimator.json`, `contact_derivatives_*` |
| State-dependent speed | `compare_variable_speed.py` | `state_dependent_comparison.json`, `state_dependent_operators/` |
| Coupled two-dimensional dynamics | `verify_twod_contact.py` | `twod_contact.json`, `twod_critic_*.npz` |
| Wrong-kink and ten-seed training diagnostics | `e1_kink.py` | `e1_kink.json` |
| Navigation and policy extraction | `e5_obstacle.py`, `e5_diag.py`, `e5_seeds.py`, `e7_policy_bound.py` | `e5_*`, `e7_policy_bound.json` |
| Sampled clipping correction | `e15_potential.py` | `e15_potential.json` |

The two-dimensional study uses bilinear critics with fixed dyadic coefficients. The one-dimensional neural verification uses the stored MLP and two-branch weights. The `e*` diagnostics use sampled residuals and numerical reference values; their results are not continuum certificates.

The source bytes used by verified computations are preserved for the recorded SHA-256 checks. Git attributes disable newline conversion. Changing a hashed source, weight file, or cache intentionally invalidates those checks.

## Generate plots and CSV summaries

```sh
python scripts/make_improved_contact_artifacts.py
python scripts/make_variable_speed_artifacts.py
python scripts/make_twod_artifacts.py
```

These commands regenerate PDF/PNG plots and their CSV data in `figures/` from the saved results. The original wrong-kink, training, and navigation PNGs are also included; their corresponding diagnostic drivers regenerate them during a full run.

## Run new computations

Full verification drivers write to `results/` relative to their source location. Use a separate checkout for new full runs so the saved results remain available for comparison. A complete one-dimensional verification sequence is:

```sh
python code/experiments/train_verified_neural.py
python code/experiments/verify_neural_policy.py
python code/experiments/compare_verified_estimators.py
python code/experiments/improve_contact_estimator.py
python code/experiments/refine_contact_stationarity.py
python code/experiments/compare_variable_speed.py
```

These steps rebuild weights and enclosures. Numerical training results and execution times can vary with hardware and software; replaying the supplied fixed weights avoids retraining variation. The design JSON files specify the comparison grids and scales.

The exact-rational and two-dimensional studies can run independently:

```sh
python code/experiments/complete_contact_certificate.py
python code/experiments/verify_twod_contact.py --backend cuda
```

Use `--backend numpy` for a CPU-only two-dimensional run. The complete two-dimensional sweep contains 18 configurations and is substantially larger than the two-configuration replay.

The original GPU diagnostics use an isolated working directory:

```sh
python code/run_experiment.py --list
python code/run_experiment.py e1_kink
python code/run_experiment.py e5_seeds
python code/run_experiment.py e7_policy_bound obstacle
python code/run_experiment.py e15_potential
```

Their outputs go to `runs/reproduction/`; the runner copies the supplied diagnostic checkpoints there as needed. `e5_diag` rebuilds the fixed navigation critics. The saved ten-seed training run contains aggregate results and plots, rather than all trained weights. A single-mode `e7_policy_bound obstacle` run writes `e7_policy_bound_obstacle.json`; the saved combined run is `e7_policy_bound.json`.
