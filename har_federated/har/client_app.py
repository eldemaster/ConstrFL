"""
UCI HAR: Flower ClientApp with Memory Optimizations + Gradient Compression
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters
import time
import os
import gc
import psutil
import tensorflow as tf

from har.task import load_model, load_data
from har.gradient_compression import create_compression_strategy
from har.compressed_parameters import decompress_from_parameters
from har.early_stopping import LocalEarlyStopping, AdaptiveEarlyStopping, calculate_compute_savings  # 🆕 PHASE 4

# 🆕 Ottimizzazioni TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# 🆕 Rileva dispositivi con poca RAM
TOTAL_RAM_GB = psutil.virtual_memory().total / (1024**3)
IS_LOW_MEMORY = TOTAL_RAM_GB < 2.0  # Pi 3 ha ~1GB

if IS_LOW_MEMORY:
    print(f"🔧 Low memory device detected ({TOTAL_RAM_GB:.1f}GB). Applying optimizations.")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)

# 🆕 GPU memory growth (se disponibile)
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
    """Ottieni i dati (con cache) - 🆕 PHASE 4: Supporta local validation split"""
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


class HARClient(NumPyClient):
    """
    Client per UCI HAR dataset
    
    🆕 PHASE 4: Supporta local validation set per early stopping
    🆕 Ottimizzazioni per dispositivi low-memory (Raspberry Pi 3).
    🆕 GRADIENT COMPRESSION (Phase 2): Supporta decompressione parametri dal server
    """
    
    def __init__(self, model, x_train, y_train, x_val=None, y_val=None, compression_strategy=None, partition_id=-1):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val  # 🆕 PHASE 4
        self.y_val = y_val  # 🆕 PHASE 4
        self.has_local_validation = (x_val is not None and y_val is not None)  # 🆕 PHASE 4
        self.compression_strategy = compression_strategy  # 🆕 Phase 2
        self.partition_id = partition_id  # 🆕 Device tracking
    
    def set_parameters(self, parameters):
        """Set model weights from parameters"""
        try:
            print(f"  📥 set_parameters called")
            print(f"     Parameters type: {type(parameters)}")
            print(f"     Parameters length: {len(parameters) if hasattr(parameters, '__len__') else 'N/A'}")
            self.model.set_weights(parameters)
            print(f"     ✅ set_weights successful")
        except Exception as e:
            print(f"     ❌ set_parameters ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def fit(self, parameters, config):
        """Training locale con tracking del tempo e ottimizzazioni memoria + 🆕 PHASE 4: Early Stopping"""
        # Start timer
        start_time = time.time()
        
        # DEBUG: Log di inizio fit
        print(f"\n{'='*60}")
        print(f"🔵 FIT STARTED")
        print(f"   Config: {list(config.keys())}")
        print(f"   Distillation enabled: {config.get('distillation_enabled', False)}")
        print(f"{'='*60}\n")
        
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
        epochs = config.get("local-epochs", 1)
        batch_size = config.get("batch-size", 32)
        verbose = config.get("verbose", 0)
        
        # 🆕 PHASE 4: Early stopping config
        early_stopping_enabled = config.get("early_stopping_enabled", False)
        early_stopping_patience = config.get("early_stopping_patience", 3)
        early_stopping_min_delta = config.get("early_stopping_min_delta", 0.001)
        early_stopping_adaptive = config.get("early_stopping_adaptive", False)
        server_round = config.get("server_round", 1)
        num_server_rounds = config.get("num_server_rounds", 10)
        
        # Adatta batch size per dispositivi con poca RAM
        if IS_LOW_MEMORY:
            effective_batch_size = min(batch_size, 16)
            print(f"  ⚠️  Reduced batch size: {batch_size} → {effective_batch_size} (low memory)")
        else:
            effective_batch_size = batch_size
        
        # 🆕 PHASE 4: Setup Early Stopping
        early_stopper = None
        if early_stopping_enabled and self.has_local_validation:
            if early_stopping_adaptive:
                early_stopper = AdaptiveEarlyStopping(
                    base_patience=early_stopping_patience,
                    min_patience=max(1, early_stopping_patience // 2),
                    total_rounds=num_server_rounds,
                    min_delta=early_stopping_min_delta,
                    restore_best_weights=True,
                    verbose=(verbose > 0)
                )
                early_stopper.set_current_round(server_round)
                print(f"  🎯 Adaptive early stopping enabled (patience: {early_stopper.patience}, round {server_round}/{num_server_rounds})")
            else:
                early_stopper = LocalEarlyStopping(
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                    restore_best_weights=True,
                    verbose=(verbose > 0)
                )
                print(f"  ⏸️  Early stopping enabled (patience: {early_stopping_patience})")
        elif early_stopping_enabled and not self.has_local_validation:
            print(f"  ⚠️  Early stopping requested but no local validation set - DISABLED")
        
        # Training - CUSTOM LOOP PER EARLY STOPPING
        actual_epochs = 0
        stopped_early = False
        
        if early_stopper is not None:
            # 🆕 PHASE 4: Training loop manuale con early stopping
            for epoch in range(epochs):
                # Train per 1 epoch
                history = self.model.fit(
                    self.x_train,
                    self.y_train,
                    epochs=1,
                    batch_size=effective_batch_size,
                    validation_data=(self.x_val, self.y_val),
                    verbose=0
                )
                
                actual_epochs += 1
                
                # Check early stopping
                val_loss = history.history['val_loss'][0]
                weights = self.model.get_weights()
                early_stopper.update(val_loss, weights)
                
                if early_stopper.should_stop():
                    # Restore best weights
                    best_weights = early_stopper.get_best_weights()
                    if best_weights is not None:
                        self.model.set_weights(best_weights)
                    stopped_early = True
                    print(f"  ⏹️  Early stopped at epoch {actual_epochs}/{epochs} (val_loss: {early_stopper.best_loss:.4f})")
                    break
            
            final_val_loss = early_stopper.best_loss
        else:
            # Standard training (no early stopping)
            if self.has_local_validation:
                history = self.model.fit(
                    self.x_train, 
                    self.y_train,
                    epochs=epochs,
                    batch_size=effective_batch_size,
                    validation_data=(self.x_val, self.y_val),
                    verbose=0 if IS_LOW_MEMORY else verbose,
                )
            else:
                history = self.model.fit(
                    self.x_train, 
                    self.y_train,
                    epochs=epochs,
                    batch_size=effective_batch_size,
                    verbose=0 if IS_LOW_MEMORY else verbose,
                )
            actual_epochs = epochs
            final_val_loss = history.history.get('val_loss', [None])[-1]
        
        # End timer
        fit_duration = time.time() - start_time
        
        # � Save weights FIRST
        parameters_updated = self.model.get_weights()
        num_examples_standard = len(self.x_train)
        
        # �🎓 FEDERATED DISTILLATION: Check if distillation is enabled
        distillation_enabled = config.get("distillation_enabled", False)
        
        if distillation_enabled:
            # 🎓 DISTILLATION MODE: Invia soft predictions invece di weights
            import numpy as np
            
            # Ricevi unlabeled data dal server
            unlabeled_data_bytes = config.get("unlabeled_data_bytes", None)
            unlabeled_data_shape = config.get("unlabeled_data_shape", None)
            temperature = config.get("distillation_temperature", 1.0)
            
            try:
                if unlabeled_data_bytes is not None and unlabeled_data_shape is not None:
                    # Deserializza unlabeled data from bytes
                    # Parse shape from string "rows,cols"
                    shape_parts = [int(x) for x in unlabeled_data_shape.split(",")]
                    shape_tuple = tuple(shape_parts)
                    unlabeled_data = np.frombuffer(unlabeled_data_bytes, dtype=np.float64).reshape(shape_tuple)
                    
                    print(f"\n  🎓 DISTILLATION MODE:")
                    print(f"     Unlabeled data shape: {unlabeled_data.shape}")
                    print(f"     Temperature: {temperature}")
                    
                    # Genera soft predictions su unlabeled data
                    logits = self.model.predict(unlabeled_data, verbose=0)
                    
                    # Apply temperature scaling
                    if temperature != 1.0:
                        soft_predictions = tf.nn.softmax(logits / temperature).numpy()
                    else:
                        soft_predictions = tf.nn.softmax(logits).numpy()
                    
                    print(f"     Soft predictions shape: {soft_predictions.shape}")
                    print(f"     Predictions range: [{soft_predictions.min():.3f}, {soft_predictions.max():.3f}]")
                    print(f"     Predictions sum: {soft_predictions.sum(axis=1).mean():.3f} (should be ~1.0)")
                    
                    # Flatten predictions per serializzazione
                    predictions_flat = soft_predictions.flatten()
                    num_examples_distill = len(self.x_train)  # Training set size (per weighted average)
                    
                    # 🎓 RETURN PREDICTIONS invece di weights!
                    parameters_to_send = [predictions_flat]
                    
                    # Salva in metrics il numero effettivo di unlabeled samples
                    distill_num_unlabeled = len(unlabeled_data)
                    
                    print(f"     💾 Communication size:")
                    print(f"        Predictions: {predictions_flat.nbytes / 1024:.1f} KB")
                    
                    # Calcola risparmio comunicazione (vs traditional FedAvg)
                    weights_size = sum(w.nbytes for w in self.model.get_weights()) / 1024
                    predictions_size = predictions_flat.nbytes / 1024
                    reduction_factor = weights_size / predictions_size if predictions_size > 0 else 1
                    
                    print(f"        Weights (FedAvg): {weights_size:.1f} KB")
                    print(f"        🚀 Reduction: {reduction_factor:.1f}x smaller!\n")
                else:
                    # No unlabeled data - fallback
                    print(f"  ⚠️  No unlabeled data provided")
                    parameters_to_send = self.model.get_weights()
                    num_examples_distill = len(self.x_train)
            except Exception as e:
                # Fallback: se c'è un errore, invia weights
                print(f"  ❌ DISTILLATION ERROR: {e}")
                import traceback
                traceback.print_exc()
                print(f"  ⚠️  Falling back to sending weights")
                parameters_to_send = self.model.get_weights()
                num_examples_distill = len(self.x_train)
        else:
            # 🔧 STANDARD MODE: Invia weights (FedAvg tradizionale)
            parameters_to_send = self.model.get_weights()
            num_examples_distill = len(self.x_train)
        
        # Pulizia memoria
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 🆕 COMPRESSION & BANDWIDTH TRACKING
        from har.compressed_parameters import (
            get_compression_stats, 
            ndarrays_to_parameters,
            compress_to_parameters
        )
        from flwr.common import parameters_to_ndarrays
        
        # Check if compression should be applied (only in standard mode, not distillation)
        compression_type = config.get("compression_type", "none")
        if not distillation_enabled and compression_type != "none" and self.compression_strategy is not None:
            # 🗜️ COMPRESS PARAMETERS before returning
            print(f"  🗜️  Applying {compression_type} compression...")
            compressed_parameters, metadata = compress_to_parameters(
                parameters_to_send,
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
            parameters_to_return = parameters_to_send
            
            # Convert parameters to Flower Parameters format for stats calculation
            params_for_stats = ndarrays_to_parameters(parameters_to_send)
            
            # Create metadata
            if distillation_enabled:
                # Distillation mode: metadata speciale
                stats_metadata = {
                    'compression_type': 'distillation',
                    'layers': []  # Empty, non serve
                }
            else:
                # Standard mode: compression_type from config
                stats_metadata = {
                    'compression_type': compression_type
                }
            
            # Calcola bandwidth stats (funziona anche con compression_type='none')
            bandwidth_stats = get_compression_stats(params_for_stats, stats_metadata)
        
        # 🆕 PHASE 4: Compute savings
        compute_stats = calculate_compute_savings(actual_epochs, epochs)
        
        # 🔧 Get partition_id and hostname for device tracking
        import socket
        hostname = socket.gethostname()
        partition_id = getattr(self, 'partition_id', -1)  # -1 if not set
        
        # Metriche da inviare al server
        metrics = {
            "train_loss": float(history.history["loss"][-1]),
            "train_acc": float(history.history["accuracy"][-1]),
            "fit_duration": float(fit_duration),
            "num_examples": num_examples_distill,
            "effective_batch_size": effective_batch_size,
            "is_low_memory": IS_LOW_MEMORY,
            # 🆕 PHASE 4: Early stopping metrics
            "actual_epochs": actual_epochs,
            "max_epochs": epochs,
            "stopped_early": stopped_early,
            "epochs_saved": compute_stats['epochs_saved'],
            "compute_saved_pct": compute_stats['compute_saved_pct'],
            # 🎓 DISTILLATION metrics
            "distillation_enabled": distillation_enabled,
            # 🆕 BANDWIDTH STATS (sempre presenti!)
            **bandwidth_stats,
            # 🆕 DEVICE TRACKING (per identificare client anche se client_id cambia)
            "partition_id": partition_id,
            "hostname": hostname,
        }
        
        # 🎓 Aggiungi num_unlabeled se distillation
        if distillation_enabled and 'distill_num_unlabeled' in locals():
            metrics["num_unlabeled"] = distill_num_unlabeled
        
        if final_val_loss is not None:
            metrics["val_loss"] = float(final_val_loss)
        
        # DEBUG
        print(f"  🐛 DEBUG RETURN:")
        print(f"     parameters_to_return type: {type(parameters_to_return)}")
        print(f"     parameters_to_return length: {len(parameters_to_return)}")
        if isinstance(parameters_to_return, list) and len(parameters_to_return) > 0:
            print(f"     First element type: {type(parameters_to_return[0])}")
            print(f"     First element shape: {parameters_to_return[0].shape if hasattr(parameters_to_return[0], 'shape') else 'N/A'}")
        
        # 🔍 DEBUG: Check if bandwidth_stats are in metrics
        print(f"  🔍 BANDWIDTH DEBUG:")
        bandwidth_keys = [k for k in metrics.keys() if 'bandwidth' in k or 'compression' in k]
        print(f"     Bandwidth keys in metrics: {bandwidth_keys}")
        for k in bandwidth_keys:
            print(f"       {k}: {metrics[k]}")
        
        return parameters_to_return, num_examples_distill, metrics


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
    
    print(f"\n🔵 CLIENT_FN CALLED")
    print(f"   Context type: {type(context)}")
    
    # Ottieni configurazione dal context
    num_partitions = context.node_config.get("num-partitions", 3)
    
    # 🆕 Determina partition_id automaticamente dall'hostname (o usa node-config se fornito)
    partition_id = get_partition_id_from_hostname(context, num_partitions)
    compute_aware = context.run_config.get("compute-aware", False)
    data_path = context.run_config.get("data-path", None)
    
    # 🆕 PHASE 4: Local validation split
    local_val_split = context.run_config.get("local-val-split", 0.0)
    
    # 🆕 PHASE 2: Compression config
    compression_type = context.run_config.get("compression-type", "none")
    compression_num_bits = context.run_config.get("compression-num-bits", 8)
    compression_k_percent = context.run_config.get("compression-k-percent", 0.1)
    
    compression_strategy = create_compression_strategy(
        compression_type=compression_type,
        num_bits=compression_num_bits,
        k_percent=compression_k_percent
    )
    
    # Log info dispositivo
    print(f"\n{'='*60}")
    print(f"🖥️  Client {partition_id} Starting")
    print(f"{'='*60}")
    print(f"  RAM Total:     {TOTAL_RAM_GB:.2f} GB")
    print(f"  RAM Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"  Low Memory:    {'⚠️  YES' if IS_LOW_MEMORY else '✅ NO'}")
    print(f"  Partition:     {partition_id}/{num_partitions}")
    print(f"  Compute-Aware: {'✅ ENABLED' if compute_aware else '❌ DISABLED'}")
    print(f"  Local Val Split: {local_val_split*100:.0f}% ({'🆕 ENABLED' if local_val_split > 0 else '❌ DISABLED'})")  # 🆕 PHASE 4
    print(f"  Compression:   {compression_type.upper()}" + 
          (f" ({compression_num_bits}-bit)" if compression_type == "quantization" else 
           f" (k={compression_k_percent:.1%})" if compression_type == "topk" else ""))
    if data_path:
        print(f"  Data Source:   📁 Pre-generated ({data_path})")
    else:
        print(f"  Data Source:   ⚡ Cache + Runtime")
    print(f"{'='*60}\n")
    
    # 🆕 PHASE 4: Carica i dati (con o senza validation locale)
    data_result = get_data(partition_id, num_partitions, compute_aware, data_path, local_val_split)
    
    if local_val_split > 0:
        x_train, y_train, x_val, y_val = data_result
    else:
        x_train, y_train = data_result
        x_val, y_val = None, None
    
    # Crea il modello (cached)
    model = get_model()
    
    # Crea e restituisci il client (con partition_id per device tracking)
    return HARClient(model, x_train, y_train, x_val, y_val, compression_strategy, partition_id).to_client()


# Usa client_fn
app = ClientApp(client_fn=client_fn)