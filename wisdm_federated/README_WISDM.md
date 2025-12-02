# WISDM Human Activity Recognition - Federated Learning

This project implements federated learning for Human Activity Recognition (HAR) using the WISDM dataset and Flower framework.

## Dataset

The WISDM (Wireless Sensor Data Mining) Activity Prediction dataset contains accelerometer data from smartphones for recognizing daily activities:
- **Activities**: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
- **Sensors**: 3-axis accelerometer (x, y, z)
- **Users**: 36 users
- **Samples**: ~1 million raw sensor readings

## Project Structure

```
wisdm/
├── wisdm/
│   ├── __init__.py
│   ├── task.py              # Model and data loading
│   ├── client_app.py        # Client-side training and evaluation
│   ├── server_app.py        # Server-side aggregation and evaluation
│   └── wisdm_dataset.py     # Dataset download and preprocessing
├── setup_dataset.py         # Script to setup the dataset
├── pyproject.toml          # Project configuration
└── README.md
```

## Setup and Installation

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Download and preprocess the WISDM dataset

```bash
python setup_dataset.py
```

This will:
- Download the WISDM dataset
- Preprocess the raw accelerometer data
- Create time series segments (200 timesteps with 100-step overlap)
- Partition data into 10 federated learning partitions (by user)

## Model Architecture

The model uses a 1D CNN architecture optimized for time series classification:
- Input: (200, 3) - 200 timesteps × 3 accelerometer axes
- 3 Conv1D blocks with BatchNormalization and Dropout
- Global Average Pooling
- Dense layers for classification
- Output: 6 activity classes

## Running Federated Learning

### Local Simulation

Run federated learning with 10 virtual clients:

```bash
flwr run . local-simulation
```

### With SuperLink (Distributed)

1. Start the SuperLink:
```bash
flwr superlink
```

2. Start SuperNodes (clients):
```bash
flwr supernode
```

3. Run the federation:
```bash
flwr run .
```

## Configuration

Edit `pyproject.toml` to customize:
- `num-server-rounds`: Number of federated learning rounds (default: 3)
- `local-epochs`: Number of local training epochs per round (default: 1)
- `batch-size`: Training batch size (default: 32)
- `num-supernodes`: Number of clients in simulation (default: 10)

## Features

✅ **Client-side training**: Each client trains on local user data  
✅ **Client-side evaluation**: Each client evaluates on local test set  
✅ **Server-side evaluation**: Server evaluates global model on centralized test set  
✅ **User-based partitioning**: Data partitioned by user for realistic federated scenario  
✅ **Model persistence**: Final model saved after training

## Results

The system provides:
- Per-round training metrics (loss, accuracy) from each client
- Per-round evaluation metrics from each client
- Centralized evaluation on server after each round
- Final global model performance on centralized test set

## Requirements

- Python 3.8+
- TensorFlow 2.11+
- Flower 1.22+
- NumPy, Pandas, Scikit-learn

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- WISDM Dataset: [WISDM Activity Recognition](https://www.cis.fordham.edu/wisdm/dataset.php)
