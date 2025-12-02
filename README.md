# Federated Learning with Optimization Techniques

This repository contains three federated learning implementations for my thesis. All three datasets share the same optimization toolkit (compute-aware partitioning, gradient compression, early stopping) so results are directly comparable. Only HAR includes an experimental distillation variant, which currently delivers limited gains.

## Projects

### 1. HAR Federated (`har_federated/`)
- **Dataset**: UCI Human Activity Recognition
- **Classes**: 6 activity types (walking, sitting, standing, etc.)
- **Features**: Shared optimization stack + federated distillation (experimental, modest impact)

### 2. MNIST Federated (`mnist_federated/`)
- **Dataset**: MNIST handwritten digits
- **Classes**: 10 digit classes (0-9)
- **Features**: Shared optimization stack (compute-aware partitioning, gradient compression, early stopping)

### 3. WISDM Federated (`wisdm_federated/`)
- **Dataset**: WISDM smartphone activity recognition
- **Classes**: 6 activity types
- **Features**: Shared optimization stack (compute-aware partitioning, gradient compression, early stopping)

## Key Features

- **Shared optimization suite**: Compute-aware partitioning, gradient compression (top-k, random-k, threshold), and early stopping across all datasets
- **IID vs Non-IID Partitioning**: Support for both balanced (IID) and heterogeneous (Non-IID) data distributions
- **Federated Distillation (HAR only)**: Optional knowledge transfer route on HAR; currently experimental with small or inconsistent gains
- **Resource Profiling**: Client-side computation and communication metrics

## Prerequisites

- Python 3.11+
- Flower Framework 1.22.0+
- TensorFlow 2.15.0+
- NumPy, Matplotlib, Seaborn

## Setup

Each project has its own virtual environment and dependencies:

```bash
# Navigate to a project directory
cd har_federated  # or mnist_federated, wisdm_federated

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .
```

## Running Experiments

### IID Partitioning (Balanced)
```bash
flwr run . --run-config 'partition-type="iid"'
```

### Non-IID Partitioning (Heterogeneous)
```bash
# HAR/WISDM: 3 classes per client
flwr run . --run-config 'partition-type="non-iid" classes-per-partition=3'

# MNIST: 5 classes per client
flwr run . --run-config 'partition-type="non-iid" classes-per-partition=5'
```

### With Compression
```bash
flwr run . --run-config 'gradient-compression="topk" compression-ratio=0.1'
```

### With Early Stopping
```bash
flwr run . --run-config 'early-stopping=true patience=3 min-delta=0.001'
```

## Analysis Tools

Each project includes analysis scripts:

- **`analyze_run.py`**: Comprehensive analysis of a single run
  ```bash
  python analyze_run.py results/YYYY-MM-DD_HH-MM-SS
  ```

- **`plot_client_metrics.py`**: Per-client resource metrics visualization
  ```bash
  python plot_client_metrics.py results/YYYY-MM-DD_HH-MM-SS
  ```

- **`final_test_evaluation.py`**: Final test set evaluation with detailed metrics
  ```bash
  python final_test_evaluation.py results/YYYY-MM-DD_HH-MM-SS
  ```

## Output Structure

After running an experiment, results are saved in:
```
results/YYYY-MM-DD_HH-MM-SS/
├── server_plots/          # Server-side metrics (loss, accuracy)
├── client_plots/          # Client-side metrics (computation, communication)
├── test_evaluation.png    # Final test metrics
├── generalization_check.png  # Validation vs test comparison
└── metrics/               # Raw metric data
```

## Thesis Context

These implementations explore federated learning optimization for resource-constrained scenarios:
- Communication efficiency through gradient compression
- Training efficiency through early stopping
- Data heterogeneity through IID/Non-IID partitioning
- Knowledge transfer through federated distillation

## License

Academic research code for thesis purposes.
