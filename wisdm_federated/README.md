# WISDM Federated Learning

Federated training on the WISDM HAR dataset. Shares the same optimization stack used for HAR and MNIST: compute-aware partitioning, gradient compression, and early stopping. Distillation is not enabled for this dataset.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run
```bash
# Baseline with shared optimizations configurable via run-config
flwr run . --run-config 'partition-type="iid" compute-aware=true early-stopping=true'

# Non-IID example
flwr run . --run-config 'partition-type="non-iid" classes-per-partition=3 compute-aware=true'
```

## Notes
- Uses the same gradient compression options as the other datasets (top-k, random-k, threshold).
- Early stopping is enabled by default; tweak patience/delta in `pyproject.toml` or via `--run-config`.
- For advanced metrics and workflow details see `README_COMPLETE.md`.
