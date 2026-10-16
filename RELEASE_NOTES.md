# Publication edition — 16 October 2026

This repository is a curated and repaired derivative of the local graduation-thesis project `Robot_RC` (source files dated December 2025). It is not a byte-for-byte archival release of the thesis source.

## Retained research components

The differential-drive Euler simulator, line/circle/S-curve trajectory families, geometric baseline, and PID control law come from the original project. The ESN retains its original reservoir update, local-frame incremental target, fixed random weights, feature-augmented ridge readout, and physically bounded autoregressive state integration. The MPC retains the original bounded optimization approach, control blocking, periodic angle penalty, and warm-start concept.

## Repairs and simplifications

1. Unified the angle/scaling helpers and repaired imports referring to a missing `utils` module.
2. Repaired MPC cost scope: control-increment calculations previously sat outside the nested objective and referenced local variables unavailable in that scope. The publication controller has an explicit testable objective.
3. Corrected terminal weighting, removed the constant current-state cost, used one consistent control-increment penalty, and shifted warm starts by one simulation step. Removed an unvalidated curvature-dependent initial-speed multiplier.
4. Replaced the broken evaluation entry point. The old entry imported a missing `MPCControllerOptimized` class and passed 4-column standardized output into an ESN fit method requiring 3-column physical next states.
5. Decoded standardized ESN sin/cos outputs before applying unit-circle geometry. The original implementation normalized standardized angle coordinates directly.
6. Reset recurrent state and washout at trial boundaries during fitting; added explicit trial resets during validation. Normalization now uses one reversible scale convention, with a floor for constant features.
7. Restricted the publication ESN interface to the tested 5-input, 4-output incremental mode. Removed unused global-output and batch interfaces; old pickle compatibility is not provided.
8. Regenerated data and trained from scratch. Historical serialized models and cached datasets are excluded because their provenance and compatibility were not established here.

## New evaluation work

- Added a nominal kinematic MPC comparison with the same objective and optimizer.
- Separated training data (seed 0) and validation data (seed 1000); stored scaling statistics are fitted only on training trials.
- Compared all controllers on identical per-case reference paths and initial states, across three evaluation seeds.
- Added physical-unit prediction/tracking metrics, latency distributions, deadline-miss fractions, optimizer status, package versions, and source SHA-256 hashes.
- Generated plots from recorded actual states, rather than plotting reference coordinates under controller labels.
- Added 15 targeted regression tests. Tests and the full 36-run standard benchmark passed locally with the versions in `requirements.txt`.

## Benchmark differences from the historical project

The standard publication profile uses an 80-unit reservoir and an 8-step horizon with 4 control blocks. Historical configuration files mention other sizes and sampling-based MPC. The published controller is L-BFGS-B MPC; it does not claim a sampling or batched-prediction speedup.

Residual learning is disabled because the training simulator and nominal residual model are identical. Otherwise zero residual targets could make results look like evidence for a learned correction where none is needed. The original incremental-learning option is used instead.

The previous `final_colab_test.py` and `colab_test_complete.py` controllers computed state costs from the current state, without rolling candidate actions through a dynamics predictor; their ESN and horizon were not used in that cost. Those alternatives, quick-fix scripts, environment probes, presentation notes, cached bytecode, editor logs, and old figures are omitted from this public edition.

Historical README claims about millimetre tracking, fixed 20 Hz operation, and large speedups have not been independently established. The README now reports the regenerated publication benchmark, including its limitations and the absence of a demonstrated benefit over exact-model MPC.

## Reproducing the recorded environment

The checked-in benchmark used Python 3.12.14, NumPy 2.5.4, SciPy 1.18.1, and Matplotlib 3.11.2 on macOS arm64. `OPENBLAS_NUM_THREADS=1` and `VECLIB_MAXIMUM_THREADS=1` were set for the experiment. Re-run with those environment variables if comparing timing. Timing still varies with hardware, library build, background load, and thread scheduling. Source hashes in the report identify the Python files used for the recorded run.

No historical Git history was available in the supplied directory. Publication starts with a current release commit rather than an invented development history. No third-party reuse license has been selected as part of this cleanup.
