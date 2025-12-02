"""
WISDM: Flower ClientApp with Memory Optimizations + Gradient Compression
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import time
import os
import gc
import psutil
import tensorflow as tf
import numpy as np

from wisdm.task import load_model, load_data
from wisdm.gradient_compression import create_compression_strategy
from wisdm.compressed_parameters import decompress_from_parameters
from wisdm.early_stopping import LocalEarlyStopping, AdaptiveEarlyStopping, calculate_compute_savings

# Ottimizzazioni TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Rileva dispositivi con poca RAM
TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
IS_LOW_MEMORY = TOTAL_RAM_GB < 2.0

if IS_LOW_MEMORY:
    print(f"🔧 Low memory device detected ({TOTAL_RAM_GB:.1f}GB). Applying optimizations.")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)

# GPU memory growth (se disponibile)
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Flower ClientApp
app = ClientApp()

# Cache del modello e dei dati
_cached_model = None
_cached_data = {}


def get_model():
    """Ottieni il modello (con cache)"""
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model()
    return _cached_model


def get_data(partition_id, num_partitions, compute_aware=False, data_path=None, local_val_split=0.0):
    """Ottieni i dati (con cache)"""
    global _cached_data
    cache_key = f"{partition_id}_{num_partitions}_ca{compute_aware}_{data_path}_val{local_val_split}"
    
    if cache_key not in _cached_data:
        result = load_data(
            partition_id, 
            num_partitions, 
            compute_aware=compute_aware,
            data_path=data_path,
            local_val_split=local_val_split
        )
        _cached_data[cache_key] = result
    
    return _cached_data[cache_key]


class WISDMClient(NumPyClient):
    """
    Client per WISDM dataset
    
    Il client ha dati di training + opzionalmente validation locale.
    La valutazione globale viene fatta dal server.
    
    🆕 GRADIENT COMPRESSION (Phase 2): Supporta decompressione parametri dal server
    🆕 EARLY STOPPING (Phase 4): Supporta early stopping locale con validation split
    🆕 DEVICE TRACKING: Supporta partition_id e hostname per tracking robusto
    """
    
    def __init__(self, model, x_train, y_train, x_val=None, y_val=None, compression_strategy=None, partition_id=-1):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.has_local_validation = x_val is not None and y_val is not None
        self.compression_strategy = compression_strategy
        self.partition_id = partition_id  # 🆕 Device tracking
    
    def fit(self, parameters, config):
        """Training locale con early stopping, compression, e ottimizzazioni memoria"""
        start_time = time.time()
        
        # 🔥 PHASE 2: Decompressione parametri se compressi
        compression_type = config.get("compression_type", "none")
        decompression_time = 0.0
        if compression_type != "none" and self.compression_strategy is not None:
            try:
                decomp_start = time.time()
                parameters_array = decompress_from_parameters(
                    parameters,
                    compression_type,
                    self.compression_strategy
                )
                decompression_time = time.time() - decomp_start
                print(f"  📥 Decompressed parameters from server ({compression_type}) in {decompression_time*1000:.2f}ms")
            except Exception as e:
                print(f"  ⚠️  Error decompressing parameters: {e}, using as-is")
                parameters_array = parameters
        else:
            parameters_array = parameters
        
        # Setta i parametri ricevuti dal server
        self.model.set_weights(parameters_array)
        
        # Ottieni iperparametri dal config
        max_epochs = config.get("local-epochs", 1)
        batch_size = config.get("batch-size", 32)
        verbose = config.get("verbose", 0)
        
        # 🆕 PHASE 4: Early Stopping config
        early_stopping_enabled = config.get("early_stopping_enabled", False)
        early_stopping_patience = config.get("early_stopping_patience", 3)
        early_stopping_min_delta = config.get("early_stopping_min_delta", 0.001)
        early_stopping_min_epochs = config.get("early_stopping_min_epochs", 0)
        early_stopping_adaptive = config.get("early_stopping_adaptive", False)
        server_round = config.get("server_round", 1)
        num_server_rounds = config.get("num_server_rounds", 3)
        
        # Adatta batch size per dispositivi con poca RAM
        if IS_LOW_MEMORY:
            effective_batch_size = min(batch_size, 16)
            print(f"  ⚠️  Reduced batch size: {batch_size} → {effective_batch_size} (low memory)")
        else:
            effective_batch_size = batch_size
        
        # Setup early stopping se abilitato E abbiamo validation set locale
        early_stopper = None
        actual_epochs = max_epochs
        stopped_early = False
        
        if early_stopping_enabled and self.has_local_validation:
            if early_stopping_adaptive:
                early_stopper = AdaptiveEarlyStopping(
                    base_patience=early_stopping_patience,
                    min_patience=max(2, early_stopping_patience // 2),
                    total_rounds=num_server_rounds,
                    min_delta=early_stopping_min_delta,
                    min_epochs=early_stopping_min_epochs
                )
                early_stopper.set_current_round(server_round)
                print(f"  ⏸️  Adaptive early stopping enabled (base_patience: {early_stopping_patience}, current: {early_stopper.patience}, min_epochs: {early_stopping_min_epochs})")
            else:
                early_stopper = LocalEarlyStopping(
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                    min_epochs=early_stopping_min_epochs
                )
                print(f"  ⏸️  Early stopping enabled (patience: {early_stopping_patience}, min_epochs: {early_stopping_min_epochs})")
        elif early_stopping_enabled and not self.has_local_validation:
            print(f"  ⚠️  Early stopping disabled: no local validation set")
        
        # Training con early stopping
        if early_stopper is not None:
            # Custom training loop epoch-by-epoch
            import copy
            for epoch in range(max_epochs):
                # Train per 1 epoch
                history = self.model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=1,
                    batch_size=effective_batch_size,
                    verbose=0 if IS_LOW_MEMORY else verbose,
                    validation_data=(self.x_val, self.y_val)
                )
                
                # Ottieni loss di validazione
                val_loss = history.history['val_loss'][0]
                
                # Salva pesi correnti
                current_weights = copy.deepcopy(self.model.get_weights())
                
                # Update early stopping e check se fermarsi
                early_stopper.update(val_loss, current_weights)
                
                if early_stopper.should_stop():
                    actual_epochs = epoch + 1
                    stopped_early = True
                    # Restore best weights
                    self.model.set_weights(early_stopper.best_weights)
                    print(f"  ⏹️  Early stopped at epoch {actual_epochs}/{max_epochs} (val_loss: {early_stopper.best_loss:.4f})")
                    break
            
            # Se completato senza early stop
            if not stopped_early:
                actual_epochs = max_epochs
                # Restore best weights comunque
                if early_stopper.best_weights is not None:
                    self.model.set_weights(early_stopper.best_weights)
                print(f"  ✅ Completed all {max_epochs} epochs (best val_loss: {early_stopper.best_loss:.4f})")
            
            # Calcola risparmio compute
            compute_stats = calculate_compute_savings(actual_epochs, max_epochs)
            epochs_saved = compute_stats['epochs_saved']
            compute_saved_pct = compute_stats['compute_saved_pct']
        else:
            # Training normale senza early stopping
            history = self.model.fit(
                self.x_train,
                self.y_train,
                epochs=max_epochs,
                batch_size=effective_batch_size,
                verbose=0 if IS_LOW_MEMORY else verbose,
            )
            actual_epochs = max_epochs
            stopped_early = False
            epochs_saved = 0
            compute_saved_pct = 0.0
        
        fit_duration = time.time() - start_time
        
        # Salva pesi PRIMA di pulire la sessione
        parameters_updated = self.model.get_weights()
        num_examples = len(self.x_train)
        
        # Pulizia memoria
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 🆕 COMPRESSION & BANDWIDTH TRACKING
        from wisdm.compressed_parameters import (
            get_compression_stats, 
            ndarrays_to_parameters,
            compress_to_parameters
        )
        from flwr.common import parameters_to_ndarrays
        
        # Check if compression should be applied
        compression_type = config.get("compression_type", "none")
        if compression_type != "none" and self.compression_strategy is not None:
            # 🗜️ COMPRESS PARAMETERS before returning
            print(f"  🗜️  Applying {compression_type} compression...")
            compressed_parameters, metadata = compress_to_parameters(
                parameters_updated,
                self.compression_strategy
            )
            
            # Convert compressed Parameters → NDArrays for return
            parameters_to_return = parameters_to_ndarrays(compressed_parameters)
            
            # Calculate bandwidth stats with compression metadata
            bandwidth_stats = get_compression_stats(compressed_parameters, metadata)
            
            print(f"  ✅ Compressed: {bandwidth_stats['original_size_bytes']:,} → "
                  f"{bandwidth_stats['compressed_size_bytes']:,} bytes "
                  f"({bandwidth_stats['compression_ratio']:.2f}x, "
                  f"{bandwidth_stats['bandwidth_saved_percent']:.1f}% saved)")
        else:
            # No compression: return original parameters
            parameters_to_return = parameters_updated
            
            # Convert parameters to Flower Parameters format for stats calculation
            params_for_stats = ndarrays_to_parameters(parameters_updated)
            
            # Create metadata
            stats_metadata = {
                'compression_type': compression_type
            }
            
            # Calcola bandwidth stats (funziona anche con compression_type='none')
            bandwidth_stats = get_compression_stats(params_for_stats, stats_metadata)
        
        # 🔧 Get partition_id and hostname for device tracking
        import socket
        hostname = socket.gethostname()
        partition_id = getattr(self, 'partition_id', -1)  # -1 if not set
        
        # Metriche da inviare al server
        metrics = {
            "train_loss": float(history.history["loss"][-1] if not stopped_early else early_stopper.best_loss),
            "train_acc": float(history.history["accuracy"][-1] if "accuracy" in history.history else 0.0),
            "fit_duration": float(fit_duration),
            "num_examples": num_examples,
            "effective_batch_size": effective_batch_size,
            "is_low_memory": IS_LOW_MEMORY,
            # 🆕 Phase 4 metrics
            "actual_epochs": actual_epochs,
            "stopped_early": stopped_early,
            "epochs_saved": epochs_saved,
            "compute_saved_pct": compute_saved_pct,
            # 🆕 BANDWIDTH STATS (sempre presenti!)
            **bandwidth_stats,
            # 🆕 DEVICE TRACKING (per identificare client anche se client_id cambia)
            "partition_id": partition_id,
            "hostname": hostname,
        }
        
        # Aggiungi validation loss se disponibile
        if self.has_local_validation and early_stopper is not None:
            metrics["val_loss"] = float(early_stopper.best_loss)
        
        return parameters_to_return, num_examples, metrics


def get_partition_id_from_hostname(context: Context, num_partitions: int) -> int:
    """
    Determina il partition_id basandosi sull'hostname del dispositivo
    
    Questo garantisce che ogni device mantenga lo stesso partition_id
    anche se si riconnette al SuperLink.
    
    Args:
        context: Flower context
        num_partitions: Numero totale di partizioni
    
    Returns:
        partition_id: ID univoco basato su hostname
    """
    import socket
    import platform
    
    # Prova prima dal node-config (se fornito esplicitamente)
    try:
        partition_id = context.node_config["partition-id"]
        print(f"  ✅ Using partition-id from node-config: {partition_id}")
        return partition_id
    except (KeyError, AttributeError):
        pass
    
    # Altrimenti determina dall'hostname
    hostname = socket.gethostname().lower()
    
    # Mappatura hostname → partition_id
    hostname_map = {
        'condevpi5': 0,  # Pi 5 - più potente
        'condevpi4': 1,  # Pi 4 - medio
        'condevpi3': 2,  # Pi 3 - meno potente
        'localhost': 0,  # Default per local simulation
    }
    
    # Cerca match esatto
    if hostname in hostname_map:
        partition_id = hostname_map[hostname]
        print(f"  ✅ Auto-detected partition-id from hostname '{hostname}': {partition_id}")
        return partition_id
    
    # Cerca match parziale (es. "conDevPi5" → "condevpi5")
    for key, value in hostname_map.items():
        if key in hostname:
            partition_id = value
            print(f"  ✅ Auto-detected partition-id from hostname '{hostname}' (matched '{key}'): {partition_id}")
            return partition_id
    
    # Fallback: usa hash dell'hostname
    partition_id = hash(hostname) % num_partitions
    print(f"  ⚠️  Unknown hostname '{hostname}', using hash-based partition-id: {partition_id}")
    return partition_id


def client_fn(context: Context):
    """Factory function per creare il client"""
    
    # Ottieni configurazione dal context
    num_partitions = context.node_config.get("num-partitions", 2)
    
    # 🆕 Determina partition_id automaticamente dall'hostname (o usa node-config se fornito)
    partition_id = get_partition_id_from_hostname(context, num_partitions)
    compute_aware = context.run_config.get("compute-aware", False)
    data_path = context.run_config.get("data-path", None)
    
    # 🆕 PHASE 2: Compression config
    compression_type = context.run_config.get("compression-type", "none")
    compression_num_bits = context.run_config.get("compression-num-bits", 8)
    compression_k_percent = context.run_config.get("compression-k-percent", 0.1)
    
    compression_strategy = create_compression_strategy(
        compression_type=compression_type,
        num_bits=compression_num_bits,
        k_percent=compression_k_percent
    )
    
    # 🆕 PHASE 4: Early Stopping config
    local_val_split = context.run_config.get("local-val-split", 0.0)
    
    # Log info dispositivo
    print(f"\n{'='*60}")
    print(f"🖥️  Client {partition_id} Starting")
    print(f"{'='*60}")
    print(f"  RAM Total:     {TOTAL_RAM_GB:.2f} GB")
    print(f"  RAM Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"  Low Memory:    {'⚠️  YES' if IS_LOW_MEMORY else '✅ NO'}")
    print(f"  Partition:     {partition_id}/{num_partitions}")
    print(f"  Compute-Aware: {'✅ ENABLED' if compute_aware else '❌ DISABLED'}")
    print(f"  Local Val Split: {local_val_split*100:.0f}% ({'🆕 ENABLED' if local_val_split > 0 else '❌ DISABLED'})")
    print(f"  Compression:   {compression_type.upper()}" + 
          (f" ({compression_num_bits}-bit)" if compression_type == "quantization" else 
           f" (k={compression_k_percent:.1%})" if compression_type == "topk" else ""))
    if data_path:
        print(f"  Data Source:   📁 Pre-generated ({data_path})")
    else:
        print(f"  Data Source:   ⚡ Cache + Runtime")
    print(f"{'='*60}\n")
    
    # Carica i dati (cached) - con optional validation split
    result = get_data(partition_id, num_partitions, compute_aware, data_path, local_val_split)
    
    # Unpack data based on validation split
    if local_val_split > 0:
        x_train, y_train, x_val, y_val = result
        print(f"Client {partition_id} (with local validation split {local_val_split*100:.0f}%):")
        print(f"  Train: {len(x_train)} samples, {len(np.unique(y_train))} classes")
        print(f"  Val:   {len(x_val)} samples, {len(np.unique(y_val))} classes")
    else:
        x_train, y_train = result
        x_val, y_val = None, None
    
    # Crea il modello (cached)
    model = get_model()
    
    # Crea e restituisci il client (con partition_id per device tracking)
    return WISDMClient(model, x_train, y_train, x_val, y_val, compression_strategy, partition_id).to_client()


# Usa client_fn
app = ClientApp(client_fn=client_fn)
