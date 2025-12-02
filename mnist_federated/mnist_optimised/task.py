"""
MNIST Task - Model and Data Loading
"""

import os
import numpy as np

# Make TensorFlow less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# Usa keras da tensorflow
keras = tf.keras

from mnist_optimised.data_utils import get_partition, get_global_validation_set, get_global_test_set
import psutil


# 🆕 Rileva dispositivi con poca RAM
TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
IS_LOW_MEMORY = TOTAL_RAM_GB < 2.0  # Pi 3 ha ~1GB


def load_model():
    """
    Crea il modello CNN per MNIST
    
    🆕 Modello UNICO leggero per TUTTI i dispositivi (Pi 3, 4, 5)
    Garantisce compatibilità per aggregazione federata.
    
    Architettura:
    - Conv2D (16 filtri) + MaxPooling
    - Flatten + Dense (64) + Dropout
    - Output (10 classi)
    
    ~13,000 parametri - Abbastanza leggero per Pi 3, abbastanza potente per MNIST
    """
    
    print(f"🔧 Creating LIGHTWEIGHT model for ALL devices ({TOTAL_RAM_GB:.1f}GB RAM)")
    
    # Modello leggero ma UNICO per tutti
    model = keras.Sequential([
        # Input layer
        keras.layers.Input(shape=(28, 28, 1)),
        
        # UN SOLO blocco convoluzionale (leggero)
        keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten e Dense
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        
        # Output layer
        keras.layers.Dense(10, activation='softmax')
    ])
    
    print(f"    📊 Model params: ~{model.count_params():,}")
    if IS_LOW_MEMORY:
        print(f"    ⚠️  Running on LOW MEMORY device - using reduced batch size")
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_data(partition_id, num_partitions, partition_type="iid", compute_aware=False, 
              classes_per_partition=7, local_val_split=0.0):
    """
    Carica MNIST usando partizionamento manuale
    
    🆕 CLIENT-SIDE TESTING:
    Se local_val_split > 0, divide i dati locali in train/val per evaluation locale.
    
    Il server mantiene sempre validation set (per monitoring) e test set (per final evaluation).
    
    Args:
        partition_id: ID del client (0 a num_partitions-1)
        num_partitions: numero totale di partizioni
        partition_type: "iid" o "non-iid"
        compute_aware: Se True, usa compute-aware partitioning (basato su potenza hardware)
        classes_per_partition: Per non-IID, numero di classi per client (default: 7 = moderato)
        local_val_split: frazione di dati locali per validation (default: 0.0 = no split)
    
    Returns:
        Se local_val_split == 0:
            x_train, y_train: numpy arrays (tutti i dati)
        Se local_val_split > 0:
            (x_train, y_train), (x_val, y_val): train e val locali
    """
    result = get_partition(
        partition_id=partition_id,
        num_partitions=num_partitions,
        partition_type=partition_type,
        compute_aware=compute_aware,
        seed=42,
        classes_per_partition=classes_per_partition,
        local_val_split=local_val_split
    )
    
    return result


def load_global_validation_data():
    """
    Carica il validation set globale per valutazione durante il training (ogni round)
    
    Returns:
        x_val, y_val: numpy arrays (12,000 samples)
    """
    x_val, y_val = get_global_validation_set()
    return x_val, y_val


def load_global_test_data():
    """
    Carica il test set globale di MNIST per valutazione finale
    
    Returns:
        x_test, y_test: numpy arrays (10,000 samples)
    """
    x_test, y_test = get_global_test_set()
    return x_test, y_test