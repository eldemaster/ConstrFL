"""
WISDM Data Utilities - Hybrid Approach
Gestisce download automatico, caching, e partizionamento del dataset WISDM
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import urllib.request
import tarfile
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# WISDM dataset parameters
WISDM_URL = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
WINDOW_SIZE = 200  # Time series window size
STEP_SIZE = 100    # Sliding window step
NUM_FEATURES = 3   # x, y, z accelerometer axes
NUM_CLASSES = 6    # Activities: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing

# Activity mapping
ACTIVITY_LABELS = {
    'Walking': 0,
    'Jogging': 1,
    'Upstairs': 2,
    'Downstairs': 3,
    'Sitting': 4,
    'Standing': 5
}


def get_cache_dir() -> Path:
    """Ottieni la directory di cache per WISDM"""
    cache_dir = Path.home() / ".cache" / "wisdm_dataset"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_wisdm(force=False) -> Path:
    """
    Scarica il dataset WISDM se non presente in cache
    
    Args:
        force: Se True, forza il re-download anche se già presente
    
    Returns:
        Path al dataset scaricato ed estratto
    """
    cache_dir = get_cache_dir()
    dataset_dir = cache_dir / "WISDM_ar_v1.1"
    tar_file = cache_dir / "WISDM_ar_latest.tar.gz"
    raw_file = dataset_dir / "WISDM_ar_v1.1_raw.txt"
    
    # Se già presente e non force, ritorna
    if raw_file.exists() and not force:
        print(f"✅ WISDM dataset found in cache: {dataset_dir}")
        return raw_file
    
    # Download
    print("📥 Downloading WISDM dataset...")
    url = WISDM_URL
    
    try:
        urllib.request.urlretrieve(url, tar_file)
        print(f"✅ Downloaded to: {tar_file}")
    except Exception as e:
        raise RuntimeError(f"❌ Download failed: {e}")
    
    # Extract
    print("📦 Extracting...")
    try:
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(cache_dir)
        print(f"✅ Extracted to: {dataset_dir}")
    except Exception as e:
        raise RuntimeError(f"❌ Extraction failed: {e}")
    
    if not raw_file.exists():
        raise FileNotFoundError(f"❌ Expected file not found after extraction: {raw_file}")
    
    return raw_file


def load_wisdm_raw(raw_file: Path, window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carica e preprocessa il dataset WISDM grezzo creando segmenti di time series
    
    Args:
        raw_file: Path al file raw WISDM_ar_v1.1_raw.txt
        window_size: Dimensione della finestra temporale (default: 200)
        step_size: Step della sliding window (default: 100)
    
    Returns:
        X: array shape (n_samples, window_size, 3) - segmenti di accelerometro
        y: array shape (n_samples,) - label delle attività (0-5)
        users: array shape (n_samples,) - ID degli utenti
    """
    print("📖 Parsing WISDM dataset...")
    
    # Leggi il file raw
    data = []
    with open(raw_file, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';')
            if line:
                try:
                    parts = line.split(',')
                    if len(parts) == 6 and all(part.strip() for part in parts):
                        data.append([part.strip() for part in parts])
                except:
                    continue
    
    # Crea DataFrame
    df = pd.DataFrame(data, columns=['user_id', 'activity', 'timestamp', 'x', 'y', 'z'])
    
    # Conversioni con gestione errori
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['z'] = pd.to_numeric(df['z'], errors='coerce')
    
    # Rimuovi righe con NaN
    df = df.dropna()
    df['user_id'] = df['user_id'].astype(int)
    
    print(f"   Raw samples: {len(df)}, Users: {df['user_id'].nunique()}")
    print(f"   Activities: {sorted(df['activity'].unique())}")
    
    # Crea segmenti con sliding window
    print(f"🔧 Creating segments (window={window_size}, step={step_size})...")
    segments = []
    labels = []
    user_ids = []
    
    for (user, activity), group in df.groupby(['user_id', 'activity']):
        # Ordina per timestamp
        group = group.sort_values('timestamp')
        data = group[['x', 'y', 'z']].values
        
        # Sliding window
        for i in range(0, len(data) - window_size + 1, step_size):
            segment = data[i:i + window_size]
            if len(segment) == window_size:
                segments.append(segment)
                labels.append(ACTIVITY_LABELS[activity])
                user_ids.append(user)
    
    X = np.array(segments, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    users = np.array(user_ids, dtype=np.int32)
    
    print(f"✅ Created {len(X)} segments from {len(np.unique(users))} users")
    print(f"   Shape: {X.shape}, Classes: {NUM_CLASSES}")
    
    return X, y, users


def create_cache(force=False) -> Path:
    """
    Crea il file di cache preprocessato
    
    Args:
        force: Se True, forza la ricreazione della cache
    
    Returns:
        Path al file di cache .npz
    """
    cache_dir = get_cache_dir()
    cache_file = cache_dir / "wisdm_processed.npz"
    
    # Se esiste e non force, ritorna
    if cache_file.exists() and not force:
        print(f"✅ Preprocessed cache found: {cache_file}")
        return cache_file
    
    print("🔧 Creating preprocessed cache...")
    print("   This is a one-time operation (~30-60s)")
    print("   Future loads will be much faster (~1-2s)")
    
    # Download se necessario
    raw_file = download_wisdm(force=force)
    
    # Carica e preprocessa
    X, y, users = load_wisdm_raw(raw_file, WINDOW_SIZE, STEP_SIZE)
    
    # Salva cache
    np.savez_compressed(
        cache_file,
        X=X,
        y=y,
        users=users
    )
    
    print(f"✅ Cache created: {cache_file}")
    print(f"   Total samples: {len(X)}, Size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    return cache_file


def load_wisdm_cached() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carica il dataset WISDM dalla cache preprocessata
    
    Returns:
        X, y, users
    """
    # Crea cache se non esiste
    cache_file = create_cache(force=False)
    
    print(f"⚡ Loading from cache: {cache_file}")
    data = np.load(cache_file)
    
    X = data['X']
    y = data['y']
    users = data['users']
    
    print(f"✅ Loaded {len(X)} samples from cache")
    
    return X, y, users


def load_wisdm_splits(
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Carica il dataset WISDM e lo divide in train/validation/test
    
    Args:
        val_ratio: Percentuale per validation (default: 15%)
        test_ratio: Percentuale per test (default: 15%)
        seed: Random seed per riproducibilità
    
    Returns:
        (train_data, val_data, test_data) dove ogni tuple è (X, y)
    """
    # Carica dalla cache
    X, y, users = load_wisdm_cached()
    
    # Split in train/(val+test)
    train_size = 1.0 - val_ratio - test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=seed, stratify=y
    )
    
    # Split (val+test) in val/test
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_size_adjusted, random_state=seed, stratify=y_temp
    )
    
    print(f"\n📊 Dataset split:")
    print(f"   Training:    {len(X_train)} samples ({train_size*100:.1f}%)")
    print(f"   Validation:  {len(X_val)} samples ({val_ratio*100:.1f}%)")
    print(f"   Test:        {len(X_test)} samples ({test_ratio*100:.1f}%)")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_partition(
    partition_id: int,
    num_partitions: int,
    data_path: Optional[str] = None,
    compute_aware: bool = False,
    local_val_split: float = 0.0,
    **kwargs
) -> Tuple:
    """
    Ottieni una partizione del dataset WISDM
    
    Supporta tre modalità:
    1. Pre-generated: Usa file .npz già creati (se data_path esiste)
    2. Runtime IID: Partizionamento bilanciato al volo
    3. Runtime Compute-Aware: Partizionamento adattivo basato su hardware
    
    Args:
        partition_id: ID della partizione (0-based)
        num_partitions: Numero totale di partizioni
        data_path: Path opzionale a partizioni pre-generate
        compute_aware: Se True, usa partizionamento compute-aware
        local_val_split: Fraction of train data to use for local validation (0.0-0.5)
        **kwargs: Parametri aggiuntivi (per compatibilità)
    
    Returns:
        Se local_val_split > 0: (X_train, y_train, X_val, y_val)
        Altrimenti: (X_train, y_train)
    """
    # Modalità 1: Pre-generated partitions
    if data_path and Path(data_path).exists():
        partition_file = Path(data_path) / f"partition_{partition_id}.npz"
        if partition_file.exists():
            print(f"📂 Loading pre-generated partition {partition_id} from {partition_file}")
            data = np.load(partition_file)
            return data['X_train'], data['y_train']
    
    # Modalità 2 e 3: Runtime partitioning
    # Carica il training set completo
    (X_train, y_train), _, _ = load_wisdm_splits()
    
    # Compute-aware partitioning
    if compute_aware:
        print(f"⚙️  Using compute-aware partitioning for client {partition_id}")
        try:
            from .compute_aware_partitioner import (
                create_compute_aware_partitions,
                load_compute_profiles
            )
            # Carica i profili compute da file JSON
            compute_scores = load_compute_profiles(num_partitions, required=True)
            
            partitions = create_compute_aware_partitions(
                X_train, y_train, num_partitions, compute_scores
            )
            X_partition, y_partition = partitions[partition_id]
            print(f"   Client {partition_id}: {len(X_partition)} samples (compute-aware)")
            
            # Local validation split (se abilitato)
            if local_val_split > 0:
                from sklearn.model_selection import train_test_split
                X_train_part, X_val, y_train_part, y_val = train_test_split(
                    X_partition, y_partition,
                    test_size=local_val_split,
                    random_state=42,
                    stratify=y_partition
                )
                print(f"   Train: {len(X_train_part)} samples, Val: {len(X_val)} samples")
                return X_train_part, y_train_part, X_val, y_val
            
            return X_partition, y_partition
        except ImportError:
            print("⚠️  Compute-aware module not available, falling back to IID")
    
    # IID partitioning (default)
    print(f"📦 Creating IID partition {partition_id}/{num_partitions}")
    
    # Shuffle con seed basato su partition_id per consistenza
    np.random.seed(42 + partition_id)
    indices = np.random.permutation(len(X_train))
    
    # Calcola gli indici per questa partizione
    partition_size = len(X_train) // num_partitions
    start_idx = partition_id * partition_size
    
    # Ultima partizione prende i rimanenti
    if partition_id == num_partitions - 1:
        end_idx = len(X_train)
    else:
        end_idx = start_idx + partition_size
    
    partition_indices = indices[start_idx:end_idx]
    X_partition = X_train[partition_indices]
    y_partition = y_train[partition_indices]
    
    # Statistiche sulla partizione
    unique, counts = np.unique(y_partition, return_counts=True)
    class_dist = dict(zip(unique.tolist(), counts.tolist()))
    
    print(f"   Client {partition_id}: {len(X_partition)} samples")
    print(f"   Classes distribution: {class_dist}")
    
    # Local validation split (Phase 4: Early Stopping)
    if local_val_split > 0:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_partition, y_partition,
            test_size=local_val_split,
            random_state=42,
            stratify=y_partition
        )
        
        # Stats
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_val, counts_val = np.unique(y_val, return_counts=True)
        class_dist_train = dict(zip(unique_train.tolist(), counts_train.tolist()))
        class_dist_val = dict(zip(unique_val.tolist(), counts_val.tolist()))
        
        print(f"   Train: {len(X_train)} samples, {len(unique_train)} classes, distribution: {class_dist_train}")
        print(f"   Val:   {len(X_val)} samples, {len(unique_val)} classes, distribution: {class_dist_val}")
        
        return X_train, y_train, X_val, y_val
    
    return X_partition, y_partition


# Export main functions
__all__ = [
    'download_wisdm',
    'create_cache',
    'load_wisdm_cached',
    'load_wisdm_splits',
    'get_partition',
    'WINDOW_SIZE',
    'NUM_FEATURES',
    'NUM_CLASSES',
    'ACTIVITY_LABELS'
]
