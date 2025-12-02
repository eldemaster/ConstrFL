# HAR Federated Learning

Federated training on the UCI HAR dataset. This project shares the same optimization stack used for MNIST and WISDM (compute-aware partitioning, gradient compression, early stopping) plus an experimental distillation variant that remains unstable and often yields limited gains.

## Features
- Compute-aware partitioning for heterogeneous clients (IID and Non-IID splits supported)
- Gradient compression options (top-k, random-k, threshold)
- Early stopping with configurable patience/delta
- Federated distillation (HAR only, experimental; see `README_DISTILLATION.md` for details)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run
```bash
# Baseline with common optimizations toggled via run-config
flwr run . --run-config 'partition-type="iid" compute-aware=true early-stopping=true'

# Non-IID example
flwr run . --run-config 'partition-type="non-iid" classes-per-partition=3 compute-aware=true'
```

## Distillation Reminder
- Distillation is available only in HAR.
- Current implementation is for comparison purposes; accuracy gains are small and sometimes regress.
- Usage and parameters: see `README_DISTILLATION.md`.
