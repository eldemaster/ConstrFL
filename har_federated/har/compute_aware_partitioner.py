"""
Compute-Aware Data Partitioning
================================

Partizionamento dinamico basato su potenza di calcolo dei client.

Misura la potenza di calcolo di ogni client e assegna più dati
ai client più potenti per ridurre il tempo totale di training
(eliminando stragglers).
"""

import numpy as np
import psutil
import time
from typing import Dict, List, Tuple
from pathlib import Path
import json
import tensorflow as tf


def measure_client_compute_power() -> Tuple[float, Dict]:
    """
    Misura la potenza di calcolo relativa del client
    
    Combina:
    - CPU cores
    - RAM disponibile
    - CPU frequency
    - Benchmark TensorFlow (matrix multiplication)
    
    Returns:
        (compute_score, details_dict)
    """
    details = {}
    
    # 1. CPU cores (weighted)
    cpu_count = psutil.cpu_count(logical=True)
    cpu_score = cpu_count * 1.0
    details['cpu_count'] = cpu_count
    
    # 2. RAM disponibile (GB)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    ram_score = ram_gb * 0.5
    details['ram_gb'] = round(ram_gb, 2)
    
    # 3. CPU frequency (GHz)
    try:
        freq = psutil.cpu_freq()
        if freq:
            freq_ghz = freq.max / 1000.0
            freq_score = freq_ghz
            details['cpu_freq_ghz'] = round(freq_ghz, 2)
        else:
            freq_score = 2.0  # Default
            details['cpu_freq_ghz'] = 2.0
    except:
        freq_score = 2.0
        details['cpu_freq_ghz'] = 2.0
    
    # 4. Benchmark velocità (mini matrix multiplication)
    print(f"  🔬 Running TensorFlow benchmark...")
    start = time.time()
    try:
        # Small matrix multiplication as benchmark
        a = tf.random.normal([500, 500])
        b = tf.random.normal([500, 500])
        _ = tf.matmul(a, b).numpy()
        benchmark_time = time.time() - start
        benchmark_score = 1.0 / benchmark_time  # Faster = higher score
        details['benchmark_time_s'] = round(benchmark_time, 3)
        details['benchmark_score'] = round(benchmark_score, 2)
    except Exception as e:
        print(f"  ⚠️  Benchmark failed: {e}")
        benchmark_score = 1.0
        details['benchmark_time_s'] = None
        details['benchmark_score'] = 1.0
    
    # Combine scores (weights tunable)
    total_score = (
        cpu_score * 2.0 +      # CPU weight
        ram_score * 1.5 +      # RAM weight
        freq_score * 1.0 +     # Frequency weight
        benchmark_score * 3.0  # Benchmark weight (most important)
    )
    
    details['total_score'] = round(total_score, 2)
    
    return total_score, details


def save_compute_profile(partition_id: int, score: float, details: Dict):
    """
    Salva profilo compute del client su disco
    
    Args:
        partition_id: ID del client
        score: Compute score calcolato
        details: Dettagli hardware
    """
    profile_dir = Path("results/compute_profiles")
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    profile = {
        'partition_id': partition_id,
        'compute_score': score,
        'details': details,
        'timestamp': time.time()
    }
    
    profile_path = profile_dir / f"client_{partition_id}_compute.json"
    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=2)
    
    print(f"  💾 Profile saved: {profile_path}")


def load_compute_profiles(num_partitions: int, required: bool = True) -> Dict[int, float]:
    """
    Carica profili compute salvati da file JSON statici
    
    Args:
        num_partitions: Numero di client attesi
        required: Se True, richiede che TUTTI i profili esistano
    
    Returns:
        Dict[partition_id, compute_score]
    
    Raises:
        FileNotFoundError: Se required=True e mancano profili
    """
    profile_dir = Path("results/compute_profiles")
    
    if not profile_dir.exists():
        if required:
            raise FileNotFoundError(
                f"❌ Directory {profile_dir} not found!\n"
                f"Please create compute profiles first:\n"
                f"  mkdir -p {profile_dir}\n"
                f"  # Create client_0_compute.json, client_1_compute.json, etc."
            )
        else:
            print("⚠️  No compute profiles found, using uniform distribution")
            return {i: 1.0 for i in range(num_partitions)}
    
    profiles = {}
    missing = []
    
    for i in range(num_partitions):
        profile_path = profile_dir / f"client_{i}_compute.json"
        
        if profile_path.exists():
            try:
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                    profiles[i] = data['compute_score']
                    device = data.get('details', {}).get('device_name', 'Unknown')
                    print(f"✅ Client {i}: score={data['compute_score']:6.2f} ({device})")
            except Exception as e:
                print(f"❌ Error loading profile for client {i}: {e}")
                missing.append(i)
        else:
            missing.append(i)
    
    # Se mancano profili e sono richiesti, errore
    if missing and required:
        raise FileNotFoundError(
            f"❌ Missing compute profiles for clients: {missing}\n"
            f"Please create: {[f'client_{i}_compute.json' for i in missing]}\n"
            f"In directory: {profile_dir}"
        )
    
    # Altrimenti usa default
    for i in missing:
        profiles[i] = 1.0
        print(f"⚠️  Client {i}: using default score=1.0")
    
    return profiles


def verify_all_profiles_exist(num_partitions: int) -> bool:
    """
    Verifica che esistano profili per tutti i client
    
    NON aspetta, solo controlla se i file esistono.
    
    Args:
        num_partitions: Numero di client attesi
    
    Returns:
        True se tutti i profili esistono, False altrimenti
    """
    profile_dir = Path("results/compute_profiles")
    
    if not profile_dir.exists():
        return False
    
    # Controlla se tutti i profili esistono
    missing = []
    for i in range(num_partitions):
        profile_path = profile_dir / f"client_{i}_compute.json"
        if not profile_path.exists():
            missing.append(i)
    
    if not missing:
        return True
    else:
        print(f"⚠️  Missing profiles for clients: {missing}")
        return False


def compute_adaptive_splits(
    total_samples: int,
    num_partitions: int,
    compute_scores: Dict[int, float],
    min_samples_per_client: int = 1000
) -> Dict[int, int]:
    """
    Calcola la suddivisione dei dati basata su potenza di calcolo
    
    Più potente il client → più dati riceve
    Garantisce un minimo per ogni client per fairness
    
    Args:
        total_samples: Totale sample da partizionare
        num_partitions: Numero di client
        compute_scores: Dict[partition_id, compute_score]
        min_samples_per_client: Minimo garantito per ogni client
    
    Returns:
        Dict[partition_id, num_samples]
    """
    # 1. Calcola proporzioni basate su compute power
    total_score = sum(compute_scores.values())
    proportions = {
        pid: score / total_score 
        for pid, score in compute_scores.items()
    }
    
    # 2. Riserva minimo garantito per ogni client
    reserved = min_samples_per_client * num_partitions
    distributable = total_samples - reserved
    
    if distributable < 0:
        raise ValueError(
            f"Not enough samples! Need at least {reserved}, have {total_samples}"
        )
    
    # 3. Assegna sample proporzionalmente alla potenza
    partition_sizes = {}
    for pid, proportion in proportions.items():
        additional = int(distributable * proportion)
        partition_sizes[pid] = min_samples_per_client + additional
    
    # 4. Aggiusta per rounding errors
    assigned = sum(partition_sizes.values())
    diff = total_samples - assigned
    
    # Distribuisci il resto ai client più potenti
    if diff != 0:
        sorted_clients = sorted(
            compute_scores.keys(), 
            key=lambda x: compute_scores[x], 
            reverse=True
        )
        for i in range(abs(diff)):
            client = sorted_clients[i % num_partitions]
            partition_sizes[client] += 1 if diff > 0 else -1
    
    return partition_sizes


def create_compute_aware_partitions(
    x_data: np.ndarray,
    y_data: np.ndarray,
    num_partitions: int,
    compute_scores: Dict[int, float],
    partition_type: str = "iid",
    min_samples: int = 1000,
    seed: int = 42
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Crea partizioni adattive basate su potenza di calcolo
    
    Args:
        x_data, y_data: Dataset da partizionare
        num_partitions: Numero di client
        compute_scores: Potenza di calcolo per client
        partition_type: "iid" o "non-iid"
        min_samples: Minimo sample garantito per client
        seed: Random seed per riproducibilità
    
    Returns:
        Dict[partition_id, (x_train, y_train)]
    """
    np.random.seed(seed)
    total_samples = len(x_data)
    
    # 1. Calcola split sizes basato su compute power
    partition_sizes = compute_adaptive_splits(
        total_samples, 
        num_partitions, 
        compute_scores,
        min_samples
    )
    
    # 2. Stampa distribuzione
    print(f"\n{'='*70}")
    print(f"🎯 COMPUTE-AWARE PARTITIONING")
    print(f"{'='*70}")
    print(f"Total samples: {total_samples}")
    print(f"Num partitions: {num_partitions}")
    print(f"\nData Distribution:")
    
    for pid in range(num_partitions):
        score = compute_scores[pid]
        size = partition_sizes[pid]
        percentage = (size / total_samples) * 100
        print(f"  Client {pid}: {size:5d} samples ({percentage:5.1f}%) | "
              f"Compute Score: {score:6.2f}")
    print(f"{'='*70}\n")
    
    # 3. Shuffle data per IID
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    x_shuffled = x_data[indices]
    y_shuffled = y_data[indices]
    
    # 4. Create partitions - ORDINA per partition_id per assegnazione consistente
    partitions = {}
    start_idx = 0
    
    # ✅ FIX: Itera sui partition_id in ORDINE ORDINATO per assegnazione deterministica
    # Questo garantisce che partition_sizes[pid] corrisponda ai dati assegnati
    sorted_pids = sorted(partition_sizes.keys())
    
    for pid in sorted_pids:
        size = partition_sizes[pid]
        end_idx = start_idx + size
        
        if partition_type == "iid":
            # IID: sequential chunks da shuffled data
            x_part = x_shuffled[start_idx:end_idx]
            y_part = y_shuffled[start_idx:end_idx]
        
        elif partition_type == "non-iid":
            # Non-IID: per ora usa IID (estendibile)
            # TODO: implementare class-based non-IID
            x_part = x_shuffled[start_idx:end_idx]
            y_part = y_shuffled[start_idx:end_idx]
        
        else:
            raise ValueError(f"Unknown partition_type: {partition_type}")
        
        partitions[pid] = (x_part, y_part)
        start_idx = end_idx
    
    return partitions
