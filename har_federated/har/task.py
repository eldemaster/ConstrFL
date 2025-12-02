"""
UCI HAR Task - Model and Data Loading
"""

import os
import numpy as np

# Make TensorFlow less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# Usa keras da tensorflow
keras = tf.keras

from har.data_utils import get_partition, get_global_validation_set, get_global_test_set


def load_model():
    """
    Crea il modello per UCI HAR Dataset
    
    Architettura:
    - Input: 561 features (sensori accelerometro + giroscopio)
    - Dense (128) + Dropout
    - Dense (64) + Dropout
    - Output: 6 classi (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
    """
    
    model = keras.Sequential([
        keras.layers.Input(shape=(561,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(6, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_data(partition_id, num_partitions, partition_type="iid", compute_aware=False, data_path=None, local_val_split=0.0):
    """
    Carica UCI HAR usando partizionamento ibrido
    
    🆕 PHASE 4: Supporta local validation split per early stopping
    
    Args:
        partition_id: ID del client (0 a num_partitions-1)
        num_partitions: numero totale di partizioni
        partition_type: "iid" o "non-iid"
        compute_aware: Se True, usa compute-aware partitioning (basato su potenza hardware)
        data_path: Se specificato, cerca partizioni pre-generate. Se None, usa cache + runtime
        local_val_split: Frazione di training data per validation locale (0.0-0.5, default: 0.0)
    
    Returns:
        Se local_val_split > 0: (x_train, y_train, x_val, y_val)
        Se local_val_split = 0: (x_train, y_train)
    """
    result = get_partition(
        partition_id=partition_id,
        num_partitions=num_partitions,
        partition_type=partition_type,
        compute_aware=compute_aware,
        data_path=data_path,
        local_val_split=local_val_split,
        seed=42
    )
    
    return result


def load_global_validation_data():
    """
    Carica il validation set globale per valutazione durante il training (ogni round)
    
    Returns:
        x_val, y_val: numpy arrays (~15% del dataset totale)
    """
    x_val, y_val = get_global_validation_set()
    return x_val, y_val


def load_global_test_data():
    """
    Carica il test set globale di UCI HAR per valutazione finale
    
    Returns:
        x_test, y_test: numpy arrays (~15% del dataset totale)
    """
    x_test, y_test = get_global_test_set()
    return x_test, y_test
