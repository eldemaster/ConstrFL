# MNIST Federated Learning

Federated MNIST training built on the same optimization stack used for HAR and WISDM: compute-aware partitioning, gradient compression, and early stopping. Distillation is not enabled for this dataset.

## Quick setup
```bash
pip install -e .
```

## Shared optimizations
- Compute-aware partitioning (IID/Non-IID)
- Gradient compression (top-k, random-k, threshold)
- Early stopping (configurable patience/min-delta)

## Compute-Aware Partitioning (NEW!)

**Optimize training time** by assigning more data to powerful clients and less to weak ones, eliminating stragglers!

### Why?

In heterogeneous hardware environments, the slowest client determines the round time:

```
Uniform:         Client A (powerful): 180s → Bottleneck = 180s
                 Client B (weak):     180s

Compute-Aware:   Client A: 60% data → 45s → Bottleneck = 45s ✅ (4x faster!)
                 Client B: 40% data → 45s
```

### Quick Start

```bash
# 1. Profile clients (one-time)
flwr run . --run-config num-server-rounds=1 compute-aware=true

# 2. Training with adaptive partitioning
flwr run . --run-config compute-aware=true

# 3. Analyze speedup
python analyze_compute_aware.py
```

### How It Works

1. **Measures** each client's compute power (CPU, RAM, TensorFlow benchmark)
2. **Assigns** more data to powerful clients, less to weak ones
3. **Eliminates** stragglers → all clients finish simultaneously
4. **Achieves** 2-4x speedup on heterogeneous hardware!

📖 **Full Guide**: See [`COMPUTE_AWARE_GUIDE.md`](COMPUTE_AWARE_GUIDE.md)

---

## Per-Client Metrics & Analysis

This project includes a comprehensive **client metrics tracking system** to analyze heterogeneous client behavior during federated training.

### Features

✅ **Automatic Tracking**: Time, accuracy, loss per client per round  
✅ **Export Formats**: CSV + JSON for easy analysis  
✅ **Visualizations**: 4 comprehensive plot sets  
✅ **Custom Analysis**: Heterogeneity, convergence, contribution scoring  
✅ **Compute-Aware**: NEW! Hardware-aware data partitioning

### Quick Start

```bash
# Run training (metrics auto-collected)
flwr run . --run-config num-server-rounds=10

# Visualize with automatic plots
python plot_client_metrics.py --save

# Run custom analysis
python analyze_client_metrics.py
```

### Output

Metrics are exported to `results/client_metrics_<timestamp>/`:
- `client_summary.csv` - Per-client aggregate stats
- `client_X_history.csv` - Time series for each client
- `round_client_stats.csv` - Cross-client stats per round
- `client_metrics_full.json` - Complete data dump
- `plots/` - Visualization outputs

### Documentation

📖 **Full Guide**: See [`CLIENT_METRICS_GUIDE.md`](CLIENT_METRICS_GUIDE.md)  
📄 **Quick Summary**: See [`CLIENT_METRICS_README.md`](CLIENT_METRICS_README.md)  
🚀 **Compute-Aware**: See [`COMPUTE_AWARE_GUIDE.md`](COMPUTE_AWARE_GUIDE.md)

### Example Metrics

- **Contribution Score**: Weight × Quality (which clients help most?)
- **Data Imbalance**: CV of sample distribution
- **Performance Heterogeneity**: Accuracy variance across clients
- **Training Time**: Identify stragglers and optimize
- **Compute-Aware Speedup**: NEW! 2-4x faster training

---

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
