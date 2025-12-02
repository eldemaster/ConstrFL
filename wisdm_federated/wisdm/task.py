"""wisdm: A Flower / TensorFlow app for Human Activity Recognition."""

import os
import numpy as np
import keras
from keras import layers
from wisdm.data_utils import (
    load_wisdm_splits,
    get_partition,
    WINDOW_SIZE,
    NUM_FEATURES,
    NUM_CLASSES
)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():
    """
    Define a LIGHTWEIGHT CNN model for time series classification (WISDM HAR).
    
    🆕 Optimized for Raspberry Pi 3 compatibility:
    - Reduced filters: 32 -> 64 -> 128 (instead of 64 -> 128 -> 256)
    - Reduced dense layer: 64 (instead of 128)
    - Target: ~100-120K parameters (vs 241K original)
    - Maintains accuracy while being Pi 3 compatible
    """
    model = keras.Sequential([
        # Input: (200, 3) - 200 time steps, 3 features (x, y, z)
        keras.Input(shape=(WINDOW_SIZE, NUM_FEATURES)),
        
        # First Conv1D block (32 filters - reduced from 64)
        layers.Conv1D(32, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Second Conv1D block (64 filters - reduced from 128)
        layers.Conv1D(64, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # Third Conv1D block (128 filters - reduced from 256)
        layers.Conv1D(128, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.4),
        
        # Dense layers (64 units - reduced from 128)
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


def load_data(partition_id: int, num_partitions: int, data_path: str = None, compute_aware: bool = False, local_val_split: float = 0.0):
    """
    Carica i dati per un client specifico
    
    Args:
        partition_id: ID della partizione (0-based)
        num_partitions: Numero totale di partizioni
        data_path: Path opzionale a partizioni pre-generate
        compute_aware: Se True, usa partizionamento compute-aware
        local_val_split: Fraction of train data for local validation (Phase 4: Early Stopping)
    
    Returns:
        Se local_val_split > 0: (x_train, y_train, x_val, y_val)
        Altrimenti: (x_train, y_train)
    """
    result = get_partition(
        partition_id=partition_id,
        num_partitions=num_partitions,
        data_path=data_path,
        compute_aware=compute_aware,
        local_val_split=local_val_split
    )
    
    return result


def load_global_validation_data():
    """
    Carica il validation set globale per il server
    
    Returns:
        x_val, y_val: Validation set completo
    """
    _, (x_val, y_val), _ = load_wisdm_splits()
    print(f"Global validation set loaded: {len(x_val)} samples")
    return x_val, y_val


def load_global_test_data():
    """
    Carica il test set globale (mai visto durante training)
    
    Returns:
        x_test, y_test: Test set completo
    """
    _, _, (x_test, y_test) = load_wisdm_splits()
    print(f"Global test set loaded: {len(x_test)} samples")
    return x_test, y_test
