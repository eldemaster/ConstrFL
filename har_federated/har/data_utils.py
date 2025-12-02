"""
UCI HAR Data Utilities - Hybrid Approach
Gestisce download automatico, caching, e partizionamento del dataset UCI HAR
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import urllib.request
import zipfile
import shutil
from sklearn.model_selection import train_test_split


def get_cache_dir() -> Path:
    """Ottieni la directory di cache per UCI HAR"""
    cache_dir = Path.home() / ".cache" / "har_dataset"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_uci_har(force=False) -> Path:
    """
    Scarica il dataset UCI HAR se non presente in cache
    
    Args:
        force: Se True, forza il re-download anche se già presente
    
    Returns:
        Path al dataset scaricato
    """
    cache_dir = get_cache_dir()
    dataset_dir = cache_dir / "UCI_HAR_Dataset"
    zip_file = cache_dir / "UCI_HAR_Dataset.zip"
    
    # Se già presente e non force, ritorna
    if dataset_dir.exists() and not force:
        print(f"✅ UCI HAR dataset found in cache: {dataset_dir}")
        return dataset_dir
    
    # Download
    print("📥 Downloading UCI HAR dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    
    try:
        urllib.request.urlretrieve(url, zip_file)
        print(f"✅ Downloaded to: {zip_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to download UCI HAR dataset: {e}")
    
    # Extract
    print("📦 Extracting...")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        
        # Lo zip crea "UCI HAR Dataset" invece di "UCI_HAR_Dataset"
        # Rinomina se necessario
        extracted_name = cache_dir / "UCI HAR Dataset"
        if extracted_name.exists() and not dataset_dir.exists():
            extracted_name.rename(dataset_dir)
        
        print(f"✅ Extracted to: {dataset_dir}")
    except Exception as e:
        raise RuntimeError(f"Failed to extract UCI HAR dataset: {e}")
    finally:
        # Cleanup zip file
        if zip_file.exists():
            zip_file.unlink()
    
    return dataset_dir


def load_uci_har_raw(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carica il dataset UCI HAR dai file di testo originali
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    print("📖 Parsing UCI HAR dataset files...")
    
    # Load training data
    X_train = np.loadtxt(dataset_dir / 'train' / 'X_train.txt')
    y_train = np.loadtxt(dataset_dir / 'train' / 'y_train.txt')
    
    # Load test data
    X_test = np.loadtxt(dataset_dir / 'test' / 'X_test.txt')
    y_test = np.loadtxt(dataset_dir / 'test' / 'y_test.txt')
    
    # Convert labels from 1-6 to 0-5
    y_train = y_train - 1
    y_test = y_test - 1
    
    print(f"✅ Loaded: {len(X_train)} train, {len(X_test)} test samples")
    print(f"   Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")
    
    return X_train, y_train, X_test, y_test


def create_cache(force=False) -> Path:
    """
    Crea una cache preprocessata del dataset UCI HAR
    
    Questo velocizza il caricamento successivo (da ~30s a ~1s)
    
    Args:
        force: Se True, ricrea la cache anche se esiste
    
    Returns:
        Path al file di cache
    """
    cache_dir = get_cache_dir()
    cache_file = cache_dir / "uci_har_processed.npz"
    
    # Se cache esiste e non force, ritorna
    if cache_file.exists() and not force:
        print(f"✅ Preprocessed cache found: {cache_file}")
        return cache_file
    
    print("🔄 Creating preprocessed cache (first time only)...")
    
    # Download se necessario
    dataset_dir = download_uci_har()
    
    # Carica dati raw
    X_train, y_train, X_test, y_test = load_uci_har_raw(dataset_dir)
    
    # Combina per poi ripartizionare con train/val/test split custom
    X_all = np.vstack([X_train, X_test])
    y_all = np.hstack([y_train, y_test])
    
    # Salva cache compressa
    np.savez_compressed(
        cache_file,
        X_all=X_all,
        y_all=y_all,
        X_train_orig=X_train,
        y_train_orig=y_train,
        X_test_orig=X_test,
        y_test_orig=y_test
    )
    
    print(f"✅ Cache created: {cache_file}")
    print(f"   Total samples: {len(X_all)}")
    
    return cache_file


def load_uci_har_cached() -> Tuple[np.ndarray, np.ndarray]:
    """
    Carica il dataset UCI HAR dalla cache (veloce!)
    
    Returns:
        X_all, y_all: tutti i dati combinati (train + test originali)
    """
    cache_file = create_cache()  # Crea se non esiste
    
    print(f"⚡ Loading from cache: {cache_file}")
    data = np.load(cache_file)
    
    X_all = data['X_all']
    y_all = data['y_all']
    
    print(f"✅ Loaded {len(X_all)} samples from cache")
    
    return X_all, y_all


def load_uci_har_splits(
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Carica UCI HAR con split train/validation/test
    
    Split strategy:
    - Training: ~70% (per client training)
    - Validation: ~15% (per server evaluation durante training)
    - Test: ~15% (per final evaluation)
    
    Args:
        val_ratio: frazione per validation set
        test_ratio: frazione per test set
        seed: random seed per riproducibilità
    
    Returns:
        (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    # Carica da cache
    X_all, y_all = load_uci_har_cached()
    
    # Split train/temp
    train_ratio = 1.0 - val_ratio - test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=y_all
    )
    
    # Split temp in validation/test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio_adjusted),
        random_state=seed,
        stratify=y_temp
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training:   {len(X_train):5d} samples ({len(X_train)/len(X_all)*100:.1f}%)")
    print(f"   Validation: {len(X_val):5d} samples ({len(X_val)/len(X_all)*100:.1f}%)")
    print(f"   Test:       {len(X_test):5d} samples ({len(X_test)/len(X_all)*100:.1f}%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_iid_partitions(
    x: np.ndarray,
    y: np.ndarray,
    num_partitions: int,
    shuffle: bool = True,
    seed: int = 42
) -> list:
    """
    Crea partizioni IID (Independent and Identically Distributed)
    
    Args:
        x: features
        y: labels
        num_partitions: numero di partizioni
        shuffle: se True, mescola i dati
        seed: random seed
    
    Returns:
        list di tuple (x_partition, y_partition)
    """
    np.random.seed(seed)
    
    num_samples = len(x)
    indices = np.arange(num_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    partitions = []
    partition_size = num_samples // num_partitions
    
    for i in range(num_partitions):
        start_idx = i * partition_size
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
    classes_per_partition: int = 2,
    seed: int = 42
) -> list:
    """
    Crea partizioni Non-IID dove ogni client ha solo alcune classi
    
    Args:
        x: features
        y: labels
        num_partitions: numero di partizioni
        classes_per_partition: numero di classi per partizione (default: 2 per HAR con 6 classi)
        seed: random seed
    
    Returns:
        list di tuple (x_partition, y_partition)
    """
    np.random.seed(seed)
    
    num_classes = len(np.unique(y))
    
    # Organizza dati per classe
    class_indices = {i: np.where(y == i)[0] for i in range(num_classes)}
    
    # Mescola indici per ogni classe
    for class_idx in class_indices:
        np.random.shuffle(class_indices[class_idx])
    
    # Assegna classi a ogni partizione
    partitions = [[] for _ in range(num_partitions)]
    
    # Distribuisci le classi in modo circolare
    for i in range(num_classes):
        for j in range(num_partitions):
            if len(partitions[j]) < classes_per_partition:
                partitions[j].append(i)
    
    # Crea le partizioni effettive
    result_partitions = []
    
    for partition_classes in partitions:
        partition_indices = []
        
        for class_idx in partition_classes:
            class_data = class_indices[class_idx]
            samples_per_class = len(class_data) // num_partitions
            
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
    data_path: Optional[str] = None,
    local_val_split: float = 0.0,  # 🆕 PHASE 4: Local validation split
    **kwargs
) -> Tuple:
    """
    Ottieni una partizione specifica del dataset UCI HAR per training
    
    MODALITÀ IBRIDE:
    1. Se `data_path` è specificato E esiste → usa partizioni pre-generate (preprocessing.py)
    2. Altrimenti → usa cache + partizionamento runtime
    
    🆕 PHASE 4: Supporta local validation split per early stopping
    - Se local_val_split > 0: restituisce (x_train, y_train, x_val, y_val)
    - Se local_val_split = 0: restituisce (x_train, y_train)
    
    Args:
        partition_id: ID della partizione (0 a num_partitions-1)
        num_partitions: numero totale di partizioni
        partition_type: "iid" o "non-iid"
        seed: random seed
        compute_aware: Se True, usa compute-aware partitioning
        data_path: Se specificato, cerca partizioni pre-generate
        local_val_split: Frazione di training data da usare per validation locale (0.0-0.5)
        **kwargs: parametri aggiuntivi
    
    Returns:
        Se local_val_split > 0: (x_train, y_train, x_val, y_val)
        Se local_val_split = 0: (x_train, y_train)
    """
    # MODALITÀ 1: Usa partizioni pre-generate (se disponibili)
    if data_path:
        pregenerated_dir = Path(data_path) / f"client_{partition_id}"
        if pregenerated_dir.exists():
            print(f"\n📁 Using pre-generated partition from: {data_path}")
            x_train = np.load(pregenerated_dir / "X_train.npy")
            y_train = np.load(pregenerated_dir / "y_train.npy")
            
            unique_classes = len(np.unique(y_train))
            samples_per_class = {int(i): int(np.sum(y_train == i)) for i in range(6)}
            print(f"Client {partition_id}: "
                  f"{len(x_train)} training samples, "
                  f"{unique_classes} classes, "
                  f"distribution: {samples_per_class}")
            
            return x_train, y_train
    
    # MODALITÀ 2: Runtime partitioning da cache
    print(f"\n🔄 Creating runtime partition {partition_id}/{num_partitions}")
    
    # Carica solo il training set
    (x_train_full, y_train_full), (_, _), (_, _) = load_uci_har_splits(seed=seed)
    
    # 🆕 COMPUTE-AWARE PARTITIONING
    if compute_aware:
        from har.compute_aware_partitioner import (
            load_compute_profiles,
            create_compute_aware_partitions
        )
        
        print(f"\n{'='*70}")
        print(f"🎯 COMPUTE-AWARE PARTITIONING (Client {partition_id})")
        print(f"{'='*70}")
        
        try:
            compute_scores = load_compute_profiles(num_partitions, required=True)
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: {e}")
            print(f"⚠️  Falling back to UNIFORM partitioning\n")
            compute_scores = None
        
        if compute_scores:
            partitions_dict = create_compute_aware_partitions(
                x_train_full,
                y_train_full,
                num_partitions,
                compute_scores,
                partition_type=partition_type,
                min_samples=500,  # Min per HAR (dataset più piccolo di MNIST)
                seed=seed
            )
            
            x_train, y_train = partitions_dict[partition_id]
            print(f"{'='*70}\n")
        else:
            compute_aware = False
    
    # STANDARD PARTITIONING
    if not compute_aware:
        if partition_type == "iid":
            partitions = create_iid_partitions(
                x_train_full,
                y_train_full,
                num_partitions,
                seed=seed
            )
        elif partition_type == "non-iid":
            classes_per_partition = kwargs.get('classes_per_partition', 2)
            partitions = create_non_iid_partitions(
                x_train_full,
                y_train_full,
                num_partitions,
                classes_per_partition=classes_per_partition,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown partition_type: {partition_type}")
        
        x_train, y_train = partitions[partition_id]
    
    # 🆕 PHASE 4: Local validation split per early stopping
    if local_val_split > 0:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train,
            test_size=local_val_split,
            random_state=seed,
            stratify=y_train
        )
        
        # Statistiche
        unique_classes_train = len(np.unique(y_train))
        samples_per_class_train = {int(i): int(np.sum(y_train == i)) for i in range(6)}
        unique_classes_val = len(np.unique(y_val))
        samples_per_class_val = {int(i): int(np.sum(y_val == i)) for i in range(6)}
        
        print(f"Client {partition_id} (with local validation split {local_val_split*100:.0f}%):")
        print(f"  Train: {len(x_train)} samples, {unique_classes_train} classes, distribution: {samples_per_class_train}")
        print(f"  Val:   {len(x_val)} samples, {unique_classes_val} classes, distribution: {samples_per_class_val}")
        
        return x_train, y_train, x_val, y_val
    else:
        # Statistiche
        unique_classes = len(np.unique(y_train))
        samples_per_class = {int(i): int(np.sum(y_train == i)) for i in range(6)}
        print(f"Client {partition_id}: "
              f"{len(x_train)} training samples, "
              f"{unique_classes} classes, "
              f"distribution: {samples_per_class}")
        
        return x_train, y_train


def get_global_validation_set() -> Tuple[np.ndarray, np.ndarray]:
    """
    Ottieni il validation set globale
    Usato dal server per valutazione durante il training (ogni round)
    
    Returns:
        x_val, y_val
    """
    (_, _), (x_val, y_val), (_, _) = load_uci_har_splits()
    
    print(f"Global validation set loaded: {len(x_val)} samples")
    
    return x_val, y_val


def get_global_test_set() -> Tuple[np.ndarray, np.ndarray]:
    """
    Ottieni il test set globale di UCI HAR
    Usato dal server per valutazione finale (dopo training completo)
    
    Returns:
        x_test, y_test
    """
    (_, _), (_, _), (x_test, y_test) = load_uci_har_splits()
    
    print(f"Global test set loaded: {len(x_test)} samples")
    
    return x_test, y_test


def analyze_partition_distribution(partitions: list) -> dict:
    """
    Analizza la distribuzione delle classi nelle partizioni
    
    Args:
        partitions: list di tuple (x, y)
    
    Returns:
        dict con statistiche
    """
    stats = {
        'num_partitions': len(partitions),
        'partition_sizes': [],
        'class_distributions': []
    }
    
    for i, (x, y) in enumerate(partitions):
        stats['partition_sizes'].append(len(x))
        
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique.astype(int), counts.astype(int)))
        stats['class_distributions'].append(class_dist)
    
    return stats


def print_partition_stats(partitions: list):
    """Stampa statistiche sulle partizioni"""
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


def get_unlabeled_batch(batch_size: int = 100, seed: int = 42) -> np.ndarray:
    """
    🆕 FEDERATED DISTILLATION: Ottieni un batch di dati unlabeled
    
    Il server usa questi dati per la knowledge distillation:
    1. Invia questo batch ai client
    2. I client generano soft predictions su questi dati
    3. Il server aggrega le predictions e fa distillation sul global model
    
    Strategy: Campiona dal validation set (senza usare le label)
    Questo simula uno scenario realistico dove il server ha dati non etichettati
    
    Args:
        batch_size: numero di campioni unlabeled (default: 100)
        seed: random seed per riproducibilità
    
    Returns:
        x_unlabeled: array di shape (batch_size, 561) - solo features, no labels!
    """
    np.random.seed(seed)
    
    # Carica validation set (ma usiamo solo le features)
    (_, _), (x_val, y_val), (_, _) = load_uci_har_splits(seed=seed)
    
    # Campiona random indices
    total_samples = len(x_val)
    if batch_size > total_samples:
        print(f"⚠️  Warning: batch_size ({batch_size}) > available samples ({total_samples})")
        print(f"   Using all {total_samples} samples instead")
        batch_size = total_samples
    
    random_indices = np.random.choice(total_samples, size=batch_size, replace=False)
    x_unlabeled = x_val[random_indices]
    
    print(f"\n🎲 Sampled unlabeled batch:")
    print(f"   Shape: {x_unlabeled.shape}")
    print(f"   Range: [{x_unlabeled.min():.3f}, {x_unlabeled.max():.3f}]")
    print(f"   Mean: {x_unlabeled.mean():.3f}, Std: {x_unlabeled.std():.3f}")
    
    return x_unlabeled
