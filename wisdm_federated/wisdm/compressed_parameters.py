"""
Compressed Parameters Utilities
================================

Helper functions per comprimere/decomprimere Parameters di Flower.

APPROCCIO SEMPLIFICATO:
- Usa Parameters standard di Flower (compatibilità garantita)
- Comprimi i dati PRIMA di convertirli in Parameters
- Il payload compresso viene serializzato come bytes normali

Non serve custom class! Solo helper functions.
"""

from typing import List, Dict, Any, Tuple
from flwr.common import Parameters, NDArrays, ndarrays_to_parameters
import numpy as np

from wisdm.gradient_compression import CompressionStrategy


def compress_to_parameters(
    ndarrays: NDArrays,
    compression_strategy: CompressionStrategy
) -> Tuple[Parameters, Dict[str, Any]]:
    """
    Comprimi NDArrays e crea Parameters standard Flower
    
    Args:
        ndarrays: Lista di numpy arrays (pesi modello)
        compression_strategy: Strategia di compressione
    
    Returns:
        (Parameters, metadata_dict) - Parameters da inviare + stats
    """
    # Comprimi
    compressed_data, metadata = compression_strategy.compress(ndarrays)
    
    # ⚠️ CRITICO: Aggiungi compression_type al metadata
    # (gradient_compression.py non lo include automaticamente)
    compression_type = compression_strategy.__class__.__name__
    if 'Quantization' in compression_type:
        metadata['compression_type'] = 'quantization'
    elif 'TopK' in compression_type:
        metadata['compression_type'] = 'topk'
    else:
        metadata['compression_type'] = 'none'
    
    # Converti compressed_data in NDArrays formato Flower-compatibile
    if metadata['compression_type'] == 'none':
        # No compression: just pass through the original arrays
        compressed_arrays = ndarrays
    
    elif metadata['compression_type'] == 'quantization':
        # compressed_data è lista di bytes
        # Per ogni layer: serializziamo [min, max, shape_dims, shape..., quantized_data]
        compressed_arrays = []
        for layer_bytes, layer_meta in zip(compressed_data, metadata['layers']):
            # Crea header: [min (float32), max (float32), ndim (int32), shape... (int32)]
            w_min = np.array([layer_meta['min']], dtype=np.float32)
            w_max = np.array([layer_meta['max']], dtype=np.float32)
            shape = np.array(layer_meta['shape'], dtype=np.int32)
            ndim = np.array([len(layer_meta['shape'])], dtype=np.int32)
            
            # Concatena: header + quantized_data
            quantized_uint8 = np.frombuffer(layer_bytes, dtype=np.uint8)
            
            # Converti tutto in uint8 per compatibilità
            header = np.concatenate([
                w_min.view(np.uint8),  # 4 bytes
                w_max.view(np.uint8),  # 4 bytes  
                ndim.view(np.uint8),   # 4 bytes
                shape.view(np.uint8)   # 4 * ndim bytes
            ])
            
            combined = np.concatenate([header, quantized_uint8])
            compressed_arrays.append(combined)
    
    elif metadata['compression_type'] == 'topk':
        # compressed_data è lista di (values, indices, shape)
        # Serializziamo come: [k, ndim, shape..., values..., indices...]
        compressed_arrays = []
        for values, indices, shape in compressed_data:
            k = len(values)
            ndim = len(shape)
            
            # Header: k (int32), ndim (int32), shape (int32 array)
            header = np.array([k, ndim] + list(shape), dtype=np.int32)
            
            # Concatena: header + values + indices
            combined = np.concatenate([
                header.astype(np.float32),  # Converti a float32 per uniformità
                values.flatten().astype(np.float32),
                indices.flatten().astype(np.float32)  # indices come float per semplicità
            ])
            compressed_arrays.append(combined)
    
    else:
        raise ValueError(f"Unknown compression type: {metadata['compression_type']}")
    
    # Crea Parameters standard
    parameters = ndarrays_to_parameters(compressed_arrays)
    
    # Aggiungi metadata come attributo (per tracking client-side)
    return parameters, metadata


def decompress_from_parameters(
    parameters,  # Union[Parameters, NDArrays]
    compression_type: str,
    compression_strategy: CompressionStrategy
) -> NDArrays:
    """
    Decomprime Parameters → NDArrays originali
    
    SEMPLIFICATO: metadata viene ricostruito dai parametri stessi!
    
    Args:
        parameters: Parameters da Flower O lista di NDArrays (Flower converte automaticamente)
        compression_type: 'quantization' o 'topk'
        compression_strategy: Strategia per decomprimere
    
    Returns:
        Lista di numpy arrays (pesi originali)
    """
    from flwr.common import parameters_to_ndarrays, Parameters
    
    # 🔧 FIX: Gestisci sia Parameters che lista NDArrays
    if isinstance(parameters, Parameters):
        compressed_arrays = parameters_to_ndarrays(parameters)
    elif isinstance(parameters, list):
        # Flower converte automaticamente Parameters → list nel client
        compressed_arrays = parameters
    else:
        raise TypeError(f"Expected Parameters or list, got {type(parameters)}")
    
    # Handle "none" compression - just pass through
    if compression_type == 'none':
        return compressed_arrays
    
    elif compression_type == 'quantization':
        # Ogni array ha formato: [min (4B), max (4B), ndim (4B), shape (4*ndim B), quantized_data]
        compressed_data = []
        metadata_layers = []
        
        num_bits = compression_strategy.num_bits if hasattr(compression_strategy, 'num_bits') else 8
        
        for arr in compressed_arrays:
            # Leggi header
            w_min = np.frombuffer(arr[0:4].tobytes(), dtype=np.float32)[0]
            w_max = np.frombuffer(arr[4:8].tobytes(), dtype=np.float32)[0]
            ndim = np.frombuffer(arr[8:12].tobytes(), dtype=np.int32)[0]
            
            # Leggi shape
            shape_bytes = arr[12:12+4*ndim]
            shape = tuple(np.frombuffer(shape_bytes.tobytes(), dtype=np.int32))
            
            # Estrai quantized data
            quantized_data = arr[12+4*ndim:]
            
            compressed_data.append(quantized_data.tobytes())
            metadata_layers.append({
                'shape': shape,
                'min': float(w_min),
                'max': float(w_max),
                'dtype': 'float32'
            })
        
        metadata = {
            'compression_type': 'quantization',
            'num_bits': num_bits,
            'layers': metadata_layers
        }
    
    elif compression_type == 'topk':
        # Top-K: ogni array contiene [k, ndim, shape..., values..., indices...]
        compressed_data = []
        metadata_layers = []
        
        for arr in compressed_arrays:
            # Leggi header
            k = int(arr[0])
            ndim = int(arr[1])
            shape = tuple(arr[2:2+ndim].astype(np.int32))
            
            # Values e indices
            start_values = 2 + ndim
            end_values = start_values + k
            values = arr[start_values:end_values]
            indices = arr[end_values:end_values + k].astype(np.int64)
            
            compressed_data.append((values, indices, shape))
            total_params = int(np.prod(shape))
            metadata_layers.append({
                'shape': shape,
                'k': k,
                'dtype': 'float32',
                'total_params': total_params
            })
        
        metadata = {
            'compression_type': 'topk',
            'k_percent': compression_strategy.k_percent if hasattr(compression_strategy, 'k_percent') else 0.1,
            'layers': metadata_layers
        }
    
    else:
        raise ValueError(f"Unknown compression type: {compression_type}")
    
    # Decomprime usando strategia
    return compression_strategy.decompress(compressed_data, metadata)


def get_compression_stats(
    parameters: Parameters,
    metadata: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calcola statistiche di compressione
    
    Args:
        parameters: Parameters compressi
        metadata: Metadata compressione
    
    Returns:
        Dict con stats (ratio, bandwidth saved, ecc.)
    """
    from flwr.common import parameters_to_ndarrays
    
    # 🔧 FIX: Handle "none" compression type (no metadata['layers'])
    if metadata.get('compression_type') == 'none':
        # No compression - just calculate size
        compressed_arrays = parameters_to_ndarrays(parameters)
        size = sum(arr.nbytes for arr in compressed_arrays)
        return {
            'original_size_bytes': int(size),
            'compressed_size_bytes': int(size),
            'compression_ratio': 1.0,
            'bandwidth_saved_percent': 0.0,
            'compression_type': 'none'
        }
    
    # Size compressa
    compressed_arrays = parameters_to_ndarrays(parameters)
    compressed_size = sum(arr.nbytes for arr in compressed_arrays)
    
    # Size originale (da metadata)
    original_size = 0
    
    # 🔧 FIX: Check if layers exist (potrebbero essere vuoti per distillation)
    if 'layers' in metadata and metadata['layers']:
        for layer_meta in metadata['layers']:
            shape = layer_meta['shape']
            original_size += int(np.prod(shape)) * 4  # float32, convert to Python int
    else:
        # Fallback: per distillation o quando non abbiamo metadata layers
        # Usa la compressed_size come original (no compression)
        original_size = compressed_size
    
    ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    
    # 🔧 FIX: Handle division by zero in bandwidth_saved_percent
    if ratio > 0:
        bandwidth_saved_percent = float((1 - 1/ratio) * 100)
    else:
        bandwidth_saved_percent = 0.0
    
    # 🔧 FIX: Convert all values to Python native types (not numpy types)
    return {
        'original_size_bytes': int(original_size),
        'compressed_size_bytes': int(compressed_size),
        'compression_ratio': float(ratio),
        'bandwidth_saved_percent': bandwidth_saved_percent,
        'compression_type': str(metadata['compression_type'])
    }

