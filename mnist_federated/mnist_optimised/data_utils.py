"""
MNIST Data Utilities - Manual Partitioning
Gestisce il caricamento e il partizionamento del dataset MNIST senza flwr-datasets
"""

import numpy as np
from typing import Tuple, Optional
import tensorflow as tf

# Usa sempre keras da tensorflow per compatibilità
keras = tf.keras


def load_mnist_keras() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Carica il dataset MNIST usando Keras con split train/val/test
    
    Split standard:
    - Training: 48,000 samples (80% del training originale) -> per client training
    - Validation: 12,000 samples (20% del training originale) -> per server evaluation durante training
    - Test: 10,000 samples (test set originale MNIST) -> per final evaluation
    
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test): tuple di numpy arrays
    """
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalizza pixel values [0, 255] -> [0, 1]
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape per CNN: (N, 28, 28) -> (N, 28, 28, 1)
    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # Split training set in train (80%) e validation (20%)
    # 60,000 samples -> 48,000 train + 12,000 val
    val_size = int(len(x_train_full) * 0.2)
    train_size = len(x_train_full) - val_size
    
    # Shuffle con seed fisso per riproducibilità
    np.random.seed(42)
    indices = np.arange(len(x_train_full))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]
    x_val = x_train_full[val_indices]
    y_val = y_train_full[val_indices]
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_iid_partitions(
    x: np.ndarray, 
    y: np.ndarray, 
    num_partitions: int,
    shuffle: bool = True,
    seed: int = 42
) -> list:
    """
    Crea partizioni IID (Independent and Identically Distributed) dei dati
    
    Args:
        x: features (es. immagini)
        y: labels
        num_partitions: numero di partizioni da creare
        shuffle: se True, mescola i dati prima del partizionamento
        seed: random seed per riproducibilità
    
    Returns:
        list di tuple (x_partition, y_partition) per ogni partizione
    """
    np.random.seed(seed)
    
    # Numero totale di campioni
    num_samples = len(x)
    
    # Crea indici
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    # Divide gli indici in num_partitions parti uguali
    partitions = []
    partition_size = num_samples // num_partitions
    
    for i in range(num_partitions):
        start_idx = i * partition_size
        # Ultima partizione prende tutti i campioni rimanenti
        end_idx = (i + 1) * partition_size if i < num_partitions - 1 else num_samples
        
        partition_indices = indices[start_idx:end_idx]
        x_partition = x[partition_indices]
        y_partition = y[partition_indices]
        
        partitions.append((x_partition, y_partition))
    
    return partitions


def create_non_iid_partitions(
    x: np.ndarray,
    y: np.ndarray,
    num_partitions: int,
    classes_per_partition: int = 7,
    seed: int = 42
) -> list:
    """
    Crea partizioni Non-IID dove ogni client ha solo alcune classi
    
    MODERATO: Default 7 classi su 10 per client (70% overlap)
    ESTREMO: Usa classes_per_partition=2 per solo 2 classi
    
    Args:
        x: features
        y: labels
        num_partitions: numero di partizioni
        classes_per_partition: numero di classi per partizione (default: 7 per non-IID moderato)
        seed: random seed
    
    Returns:
        list di tuple (x_partition, y_partition)
    """
    np.random.seed(seed)
    
    num_classes = len(np.unique(y))
    
    # Organizza dati per classe
    class_indices = {i: np.where(y == i)[0] for i in range(num_classes)}
    
    # Mescola gli indici di ogni classe
    for class_idx in class_indices:
        np.random.shuffle(class_indices[class_idx])
    
    # Assegna classi a ogni partizione CON OVERLAP
    partitions = [set() for _ in range(num_partitions)]
    
    # Strategia: Distribuisci le classi con overlap controllato
    # Ogni classe appare in almeno un client, con overlap tra client vicini
    if classes_per_partition * num_partitions >= num_classes:
        # C'è abbastanza capacità per coprire tutte le classi con overlap
        # Usiamo una sliding window con overlap
        
        # Calcola step size per distribuire uniformemente
        step = max(1, (num_classes - classes_per_partition) / max(1, num_partitions - 1))
        
        for partition_idx in range(num_partitions):
            # Calcola l'offset di partenza per questa partizione
            start_class = int(partition_idx * step)
            
            # Assegna classes_per_partition classi consecutive (con wrapping)
            for i in range(classes_per_partition):
                class_idx = (start_class + i) % num_classes
                partitions[partition_idx].add(class_idx)
        
        # Assicurati che tutte le classi siano coperte
        covered_classes = set()
        for partition in partitions:
            covered_classes.update(partition)
        
        # Se mancano classi, aggiungile ai client con meno classi
        missing_classes = set(range(num_classes)) - covered_classes
        for missing_class in sorted(missing_classes):
            # Trova il client con meno classi
            min_partition = min(range(num_partitions), key=lambda i: len(partitions[i]))
            partitions[min_partition].add(missing_class)
    else:
        # Non c'è abbastanza capacità - distribuisci circolarmente
        class_assignments = list(range(num_classes)) * ((classes_per_partition * num_partitions) // num_classes + 1)
        np.random.shuffle(class_assignments)
        
        for partition_idx in range(num_partitions):
            start_idx = partition_idx * classes_per_partition
            end_idx = start_idx + classes_per_partition
            partitions[partition_idx] = set(class_assignments[start_idx:end_idx])
    
    # Converti set a liste ordinate
    partitions = [sorted(list(p)) for p in partitions]
    
    # 🆕 LOG: Mostra la distribuzione delle classi per client
    print(f"\n📊 Non-IID Partition Distribution ({classes_per_partition} classes per client):")
    for partition_idx, partition_classes in enumerate(partitions):
        print(f"  Client {partition_idx}: classes {partition_classes}")
    print()
    
    # Crea le partizioni effettive
    result_partitions = []
    
    for partition_classes in partitions:
        partition_indices = []
        
        # Per ogni classe in questa partizione
        for class_idx in partition_classes:
            # Prendi una porzione degli esempi di questa classe
            class_data = class_indices[class_idx]
            samples_per_class = len(class_data) // num_partitions
            
            # Assegna i campioni
            start = 0
            end = samples_per_class
            partition_indices.extend(class_data[start:end])
            
            # Rimuovi i campioni assegnati
            class_indices[class_idx] = class_data[end:]
        
        partition_indices = np.array(partition_indices)
        np.random.shuffle(partition_indices)
        
        x_partition = x[partition_indices]
        y_partition = y[partition_indices]
        
        result_partitions.append((x_partition, y_partition))
    
    return result_partitions


def get_partition(
    partition_id: int,
    num_partitions: int,
    partition_type: str = "iid",
    seed: int = 42,
    compute_aware: bool = False,
    local_val_split: float = 0.0,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ottieni una partizione specifica del dataset MNIST per training
    
    🆕 LOCAL VALIDATION SPLIT:
    Se local_val_split > 0, divide i dati locali in train/val per client-side evaluation.
    Questo permette early stopping locale e metriche individuali.
    
    IMPORTANTE: Anche con split locale, il server mantiene validation/test globali separati.
    
    Args:
        partition_id: ID della partizione (0 a num_partitions-1)
        num_partitions: numero totale di partizioni
        partition_type: "iid" o "non-iid"
        seed: random seed
        compute_aware: Se True, usa compute-aware partitioning (basato su potenza hardware)
        local_val_split: frazione di dati locali da usare per validation (default: 0.0 = no split)
                        Es: 0.2 = 80% train, 20% val locale
        **kwargs: parametri aggiuntivi per create_non_iid_partitions
    
    Returns:
        Se local_val_split == 0:
            x_train, y_train: tutti i dati di training per questo client
        Se local_val_split > 0:
            (x_train, y_train), (x_val, y_val): train e validation locali per questo client
    """
    # Carica solo il training set (48,000 samples)
    (x_train_full, y_train_full), (_, _), (_, _) = load_mnist_keras()
    
    # 🆕 COMPUTE-AWARE PARTITIONING
    if compute_aware:
        from mnist_optimised.compute_aware_partitioner import (
            load_compute_profiles,
            create_compute_aware_partitions
        )
        
        print(f"\n{'='*70}")
        print(f"🎯 COMPUTE-AWARE PARTITIONING (Client {partition_id})")
        print(f"{'='*70}")
        
        # Carica profili compute STATICI (devono essere pre-configurati)
        # Ogni client carica gli STESSI profili → partizioni consistenti
        try:
            compute_scores = load_compute_profiles(num_partitions, required=True)
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            print(f"⚠️  Falling back to UNIFORM partitioning\n")
            compute_scores = None
        
        if compute_scores:
            # Crea partizioni adattive usando seed fisso
            # IMPORTANTE: Ogni client esegue lo stesso calcolo con:
            #   - Stesso seed (42)
            #   - Stessi profili compute
            #   - Stesso dataset iniziale
            # → Risultato deterministico e consistente
            partitions_dict = create_compute_aware_partitions(
                x_train_full,
                y_train_full,
                num_partitions,
                compute_scores,
                partition_type=partition_type,
                min_samples=1000,
                seed=seed  # SEED FISSO per consistenza
            )
            
            # Ottieni SOLO la partizione di questo client
            x_train, y_train = partitions_dict[partition_id]
            print(f"{'='*70}\n")
        else:
            # Fallback a uniform se profili non disponibili
            compute_aware = False  # Usa standard partitioning
    
    # STANDARD PARTITIONING (uniform)
    if not compute_aware:
        # Crea partizioni
        if partition_type == "iid":
            partitions = create_iid_partitions(
                x_train_full, 
                y_train_full, 
                num_partitions, 
                seed=seed
            )
        elif partition_type == "non-iid":
            partitions = create_non_iid_partitions(
                x_train_full,
                y_train_full,
                num_partitions,
                seed=seed,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown partition_type: {partition_type}")
        
        # Ottieni la partizione richiesta
        x_train, y_train = partitions[partition_id]
    
    # 🆕 LOCAL VALIDATION SPLIT (se richiesto)
    if local_val_split > 0:
        # Usa seed deterministico per ogni client (basato su partition_id)
        local_seed = seed + partition_id
        np.random.seed(local_seed)
        
        # Calcola dimensioni
        num_samples = len(x_train)
        val_size = int(num_samples * local_val_split)
        train_size = num_samples - val_size
        
        # Shuffle e split
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        x_train_local = x_train[train_indices]
        y_train_local = y_train[train_indices]
        x_val_local = x_train[val_indices]
        y_val_local = y_train[val_indices]
        
        # Statistiche
        unique_classes_train = len(np.unique(y_train_local))
        unique_classes_val = len(np.unique(y_val_local))
        samples_per_class_train = {i: int(np.sum(y_train_local == i)) for i in range(10) if i in y_train_local}
        samples_per_class_val = {i: int(np.sum(y_val_local == i)) for i in range(10) if i in y_val_local}
        
        print(f"Client {partition_id} (with local validation split {local_val_split:.0%}):")
        print(f"  Train: {len(x_train_local)} samples, "
              f"{unique_classes_train} classes, "
              f"distribution: {samples_per_class_train}")
        print(f"  Val:   {len(x_val_local)} samples, "
              f"{unique_classes_val} classes, "
              f"distribution: {samples_per_class_val}")
        
        return (x_train_local, y_train_local), (x_val_local, y_val_local)
    else:
        # No split - return all data as training
        unique_classes = len(np.unique(y_train))
        samples_per_class = {i: int(np.sum(y_train == i)) for i in range(10)}
        print(f"Client {partition_id}: "
              f"{len(x_train)} training samples, "
              f"{unique_classes} classes, "
              f"distribution: {samples_per_class}")
        
        return x_train, y_train


def get_global_validation_set() -> Tuple[np.ndarray, np.ndarray]:
    """
    Ottieni il validation set globale (12,000 samples)
    Usato dal server per valutazione durante il training (ogni round)
    
    Returns:
        x_val, y_val
    """
    (_, _), (x_val, y_val), (_, _) = load_mnist_keras()
    
    print(f"Global validation set loaded: {len(x_val)} samples")
    
    return x_val, y_val


def get_global_test_set() -> Tuple[np.ndarray, np.ndarray]:
    """
    Ottieni il test set globale di MNIST (10,000 samples)
    Usato dal server per valutazione finale (dopo training completo)
    
    Returns:
        x_test, y_test
    """
    (_, _), (_, _), (x_test, y_test) = load_mnist_keras()
    
    print(f"Global test set loaded: {len(x_test)} samples")
    
    return x_test, y_test


def analyze_partition_distribution(partitions: list) -> dict:
    """
    Analizza la distribuzione delle classi nelle partizioni
    
    Args:
        partitions: list di tuple (x, y)
    
    Returns:
        dict con statistiche sulla distribuzione
    """
    stats = {
        'num_partitions': len(partitions),
        'partition_sizes': [],
        'class_distributions': []
    }
    
    for i, (x, y) in enumerate(partitions):
        stats['partition_sizes'].append(len(x))
        
        # Conta classi
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique.astype(int), counts.astype(int)))
        stats['class_distributions'].append(class_dist)
    
    return stats


def print_partition_stats(partitions: list):
    """
    Stampa statistiche sulle partizioni
    """
    stats = analyze_partition_distribution(partitions)
    
    print(f"\n{'='*60}")
    print(f"Partition Statistics")
    print(f"{'='*60}")
    print(f"Number of partitions: {stats['num_partitions']}")
    print(f"Partition sizes: {stats['partition_sizes']}")
    print(f"\nClass distribution per partition:")
    
    for i, dist in enumerate(stats['class_distributions']):
        classes = sorted(dist.keys())
        print(f"  Partition {i}: classes {classes}, "
              f"samples {sum(dist.values())}")
