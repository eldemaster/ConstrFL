#!/usr/bin/env python3
"""
Gradient Compression Module
============================

Implementa tecniche di compressione per ridurre la comunicazione nel FL:

1. QUANTIZATION: Riduce la precisione dei pesi (8-bit, 16-bit)
2. TOP-K SPARSIFICATION: Invia solo i K gradienti più grandi

References:
- "Deep Gradient Compression" (Lin et al., 2018)
- "QSGD: Communication-Efficient SGD" (Alistarh et al., 2017)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, NDArrays


# ==================== QUANTIZATION ====================

def quantize_weights(
    weights: NDArrays,
    num_bits: int = 8
) -> Tuple[List[bytes], Dict[str, any]]:
    """
    Quantizza i pesi a num_bits bit
    
    Args:
        weights: Lista di numpy arrays (pesi del modello)
        num_bits: Numero di bit per quantizzazione (8 o 16)
    
    Returns:
        quantized_bytes: Lista di bytes quantizzati
        metadata: Dict con min, max, shape per de-quantizzazione
    """
    if num_bits not in [8, 16]:
        raise ValueError(f"num_bits deve essere 8 o 16, ricevuto: {num_bits}")
    
    quantized_bytes = []
    metadata = {
        'num_bits': num_bits,
        'layers': []
    }
    
    dtype = np.uint8 if num_bits == 8 else np.uint16
    
    for layer_weights in weights:
        # Calcola min/max per de-normalizzazione
        w_min = float(layer_weights.min())
        w_max = float(layer_weights.max())
        
        # Quantizza: [w_min, w_max] → [0, 2^num_bits - 1]
        if w_max > w_min:
            scale = (2**num_bits - 1) / (w_max - w_min)
            quantized = ((layer_weights - w_min) * scale).astype(dtype)
        else:
            # Tutti i valori uguali
            quantized = np.zeros_like(layer_weights, dtype=dtype)
        
        # Salva metadata per de-quantizzazione
        metadata['layers'].append({
            'shape': layer_weights.shape,
            'min': w_min,
            'max': w_max,
            'dtype': str(layer_weights.dtype)
        })
        
        # Converti in bytes
        quantized_bytes.append(quantized.tobytes())
    
    return quantized_bytes, metadata


def dequantize_weights(
    quantized_bytes: List[bytes],
    metadata: Dict[str, any]
) -> NDArrays:
    """
    De-quantizza i pesi compressi
    
    Args:
        quantized_bytes: Lista di bytes quantizzati
        metadata: Metadata da quantize_weights()
    
    Returns:
        weights: Lista di numpy arrays de-quantizzati
    """
    num_bits = metadata['num_bits']
    dtype = np.uint8 if num_bits == 8 else np.uint16
    
    weights = []
    
    for layer_bytes, layer_meta in zip(quantized_bytes, metadata['layers']):
        # Ricostruisci array quantizzato
        quantized = np.frombuffer(layer_bytes, dtype=dtype).reshape(layer_meta['shape'])
        
        # De-quantizza: [0, 2^num_bits - 1] → [w_min, w_max]
        w_min = layer_meta['min']
        w_max = layer_meta['max']
        
        if w_max > w_min:
            scale = (w_max - w_min) / (2**num_bits - 1)
            dequantized = quantized.astype(np.float32) * scale + w_min
        else:
            dequantized = np.full(layer_meta['shape'], w_min, dtype=np.float32)
        
        # Converti al dtype originale
        original_dtype = np.dtype(layer_meta['dtype'])
        weights.append(dequantized.astype(original_dtype))
    
    return weights


def compute_compression_ratio(
    original_weights: NDArrays,
    quantized_bytes: List[bytes]
) -> float:
    """
    Calcola il compression ratio
    
    Returns:
        ratio: compression_ratio (es: 4.0 = 4x compressione)
    """
    # Dimensione originale (float32 = 4 bytes)
    original_size = sum(w.nbytes for w in original_weights)
    
    # Dimensione compressa
    compressed_size = sum(len(b) for b in quantized_bytes)
    
    return original_size / compressed_size if compressed_size > 0 else 1.0


# ==================== TOP-K SPARSIFICATION ====================

def topk_sparsify(
    weights: NDArrays,
    k_percent: float = 0.1
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], Dict[str, any]]:
    """
    Top-K Sparsification: mantieni solo i K% valori più grandi (in abs)
    
    Args:
        weights: Lista di numpy arrays
        k_percent: Percentuale di valori da mantenere (0.01 = 1%, 0.1 = 10%)
    
    Returns:
        sparse_data: Lista di (values, indices, shape) per ogni layer
        metadata: Informazioni per ricostruzione
    """
    if not 0 < k_percent <= 1:
        raise ValueError(f"k_percent deve essere in (0, 1], ricevuto: {k_percent}")
    
    sparse_data = []
    metadata = {
        'k_percent': k_percent,
        'layers': []
    }
    
    for layer_weights in weights:
        # Flatten per trovare top-k
        flat_weights = layer_weights.flatten()
        k = max(1, int(len(flat_weights) * k_percent))
        
        # Trova i K indici con valori assoluti più grandi
        abs_weights = np.abs(flat_weights)
        top_k_indices = np.argpartition(abs_weights, -k)[-k:]
        
        # Estrai valori e indici
        top_k_values = flat_weights[top_k_indices]
        
        # Salva metadata
        metadata['layers'].append({
            'shape': layer_weights.shape,
            'dtype': str(layer_weights.dtype),
            'total_params': len(flat_weights),
            'kept_params': k
        })
        
        sparse_data.append((
            top_k_values,
            top_k_indices,
            np.array(layer_weights.shape)
        ))
    
    return sparse_data, metadata


def topk_desparsify(
    sparse_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    metadata: Dict[str, any]
) -> NDArrays:
    """
    Ricostruisce i pesi da top-k sparse representation
    
    Args:
        sparse_data: Lista di (values, indices, shape)
        metadata: Metadata da topk_sparsify()
    
    Returns:
        weights: Lista di numpy arrays ricostruiti (altri valori = 0)
    """
    weights = []
    
    for (values, indices, shape), layer_meta in zip(sparse_data, metadata['layers']):
        # Crea array di zeri
        original_dtype = np.dtype(layer_meta['dtype'])
        flat_weights = np.zeros(layer_meta['total_params'], dtype=original_dtype)
        
        # Ripristina i valori top-k
        flat_weights[indices] = values
        
        # Reshape alla forma originale
        reconstructed = flat_weights.reshape(tuple(shape))
        weights.append(reconstructed)
    
    return weights


def compute_sparsity_ratio(
    original_weights: NDArrays,
    sparse_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> float:
    """
    Calcola la sparsity ratio
    
    Returns:
        ratio: compression ratio (es: 10.0 = 10x compressione)
    """
    # Parametri originali
    total_params = sum(w.size for w in original_weights)
    
    # Parametri trasmessi (values + indices)
    # Nota: indices sono int64 (8 bytes), values sono float32 (4 bytes)
    transmitted_params = sum(len(values) for values, _, _ in sparse_data)
    transmitted_size = transmitted_params * (4 + 8)  # float32 + int64
    original_size = total_params * 4  # float32
    
    return original_size / transmitted_size if transmitted_size > 0 else 1.0


# ==================== COMPRESSION STRATEGIES ====================

class CompressionStrategy:
    """
    Base class per strategie di compressione
    """
    def compress(self, weights: NDArrays) -> Tuple[any, Dict]:
        raise NotImplementedError
    
    def decompress(self, compressed_data: any, metadata: Dict) -> NDArrays:
        raise NotImplementedError
    
    def get_stats(self) -> Dict:
        raise NotImplementedError


class QuantizationStrategy(CompressionStrategy):
    """
    Strategia di compressione via Quantization
    """
    def __init__(self, num_bits: int = 8):
        self.num_bits = num_bits
        self.compression_ratios = []
    
    def compress(self, weights: NDArrays) -> Tuple[List[bytes], Dict]:
        quantized, metadata = quantize_weights(weights, self.num_bits)
        ratio = compute_compression_ratio(weights, quantized)
        self.compression_ratios.append(ratio)
        return quantized, metadata
    
    def decompress(self, compressed_data: List[bytes], metadata: Dict) -> NDArrays:
        return dequantize_weights(compressed_data, metadata)
    
    def get_stats(self) -> Dict:
        if not self.compression_ratios:
            return {}
        return {
            'compression_type': 'quantization',
            'num_bits': self.num_bits,
            'avg_compression_ratio': np.mean(self.compression_ratios),
            'min_compression_ratio': np.min(self.compression_ratios),
            'max_compression_ratio': np.max(self.compression_ratios),
            'total_compressions': len(self.compression_ratios)
        }


class TopKStrategy(CompressionStrategy):
    """
    Strategia di compressione via Top-K Sparsification
    """
    def __init__(self, k_percent: float = 0.1):
        self.k_percent = k_percent
        self.compression_ratios = []
    
    def compress(self, weights: NDArrays) -> Tuple[List[Tuple], Dict]:
        sparse, metadata = topk_sparsify(weights, self.k_percent)
        ratio = compute_sparsity_ratio(weights, sparse)
        self.compression_ratios.append(ratio)
        return sparse, metadata
    
    def decompress(self, compressed_data: List[Tuple], metadata: Dict) -> NDArrays:
        return topk_desparsify(compressed_data, metadata)
    
    def get_stats(self) -> Dict:
        if not self.compression_ratios:
            return {}
        return {
            'compression_type': 'topk_sparsification',
            'k_percent': self.k_percent,
            'avg_compression_ratio': np.mean(self.compression_ratios),
            'min_compression_ratio': np.min(self.compression_ratios),
            'max_compression_ratio': np.max(self.compression_ratios),
            'total_compressions': len(self.compression_ratios)
        }


class NoCompressionStrategy(CompressionStrategy):
    """
    Nessuna compressione (baseline)
    """
    def compress(self, weights: NDArrays) -> Tuple[NDArrays, Dict]:
        return weights, {'compression_type': 'none'}
    
    def decompress(self, compressed_data: NDArrays, metadata: Dict) -> NDArrays:
        return compressed_data
    
    def get_stats(self) -> Dict:
        return {
            'compression_type': 'none',
            'avg_compression_ratio': 1.0
        }


# ==================== FACTORY ====================

def create_compression_strategy(
    compression_type: str = "none",
    **kwargs
) -> CompressionStrategy:
    """
    Factory per creare strategie di compressione
    
    Args:
        compression_type: "none", "quantization", "topk"
        **kwargs: Parametri specifici (num_bits, k_percent, etc.)
    
    Returns:
        CompressionStrategy instance
    """
    if compression_type == "none":
        return NoCompressionStrategy()
    elif compression_type == "quantization":
        num_bits = kwargs.get('num_bits', 8)
        return QuantizationStrategy(num_bits=num_bits)
    elif compression_type == "topk":
        k_percent = kwargs.get('k_percent', 0.1)
        return TopKStrategy(k_percent=k_percent)
    else:
        raise ValueError(
            f"Tipo di compressione non supportato: {compression_type}. "
            f"Usa: 'none', 'quantization', 'topk'"
        )


# ==================== UTILS ====================

def analyze_compression_impact(
    original_weights: NDArrays,
    compressed_weights: NDArrays
) -> Dict[str, float]:
    """
    Analizza l'impatto della compressione sui pesi
    
    Returns:
        Dict con MSE, MAE, cosine similarity, etc.
    """
    metrics = {}
    
    total_mse = 0
    total_mae = 0
    total_params = 0
    total_dot = 0
    total_norm_orig = 0
    total_norm_comp = 0
    
    for orig, comp in zip(original_weights, compressed_weights):
        diff = orig - comp
        total_mse += np.sum(diff ** 2)
        total_mae += np.sum(np.abs(diff))
        total_params += orig.size
        
        # Cosine similarity
        total_dot += np.sum(orig * comp)
        total_norm_orig += np.sum(orig ** 2)
        total_norm_comp += np.sum(comp ** 2)
    
    metrics['mse'] = total_mse / total_params
    metrics['mae'] = total_mae / total_params
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Cosine similarity
    if total_norm_orig > 0 and total_norm_comp > 0:
        metrics['cosine_similarity'] = (
            total_dot / (np.sqrt(total_norm_orig) * np.sqrt(total_norm_comp))
        )
    else:
        metrics['cosine_similarity'] = 1.0
    
    return metrics
