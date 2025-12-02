"""
MNIST: Flower ClientApp with Memory Optimizations + Gradient Compression
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import time
import os
import gc
import psutil
import tensorflow as tf

from mnist_optimised.task import load_model, load_data
from mnist_optimised.gradient_compression import create_compression_strategy
from mnist_optimised.compressed_parameters import compress_to_parameters, decompress_from_parameters, get_compression_stats
from mnist_optimised.early_stopping import LocalEarlyStopping, AdaptiveEarlyStopping, calculate_compute_savings  # 🆕 PHASE 4
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

# 🆕 Resource Profiling
try:
    from resource_profiler import ResourceProfiler
    PROFILING_ENABLED = True
except ImportError:
    PROFILING_ENABLED = False
    print("⚠️  Resource profiler not available, profiling disabled")

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


def get_data(partition_id, num_partitions, compute_aware=False, partition_type="iid", 
             classes_per_partition=4, local_val_split=0.0):
    """Ottieni i dati (con cache)"""
    global _cached_data
    cache_key = f"{partition_id}_{num_partitions}_ca{compute_aware}_pt{partition_type}_cpp{classes_per_partition}_valsplit{local_val_split}"
    
    if cache_key not in _cached_data:
        result = load_data(partition_id, num_partitions, partition_type=partition_type, 
                          compute_aware=compute_aware, classes_per_partition=classes_per_partition,
                          local_val_split=local_val_split)
        _cached_data[cache_key] = result
    
    return _cached_data[cache_key]


class MNISTClient(NumPyClient):
    """
    Client per MNIST dataset
    
    🆕 CLIENT-SIDE TESTING & EVALUATION:
    - Se local_val_split > 0: il client ha train + validation locale
    - Può eseguire evaluate() su validation set locale per metriche individuali
    - Il server mantiene sempre validation/test globali separati
    
    🆕 GRADIENT COMPRESSION (Phase 2):
    - Supporta quantization (8/16-bit) e top-k sparsification
    - Comprime i pesi prima di inviarli al server → riduce comunicazione
    
    🆕 Ottimizzazioni per dispositivi low-memory (Raspberry Pi 3).
    """
    
    def __init__(self, model, x_train, y_train, x_val=None, y_val=None, 
                 compression_strategy=None, partition_id=0):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val  # 🆕 Local validation set (opzionale)
        self.y_val = y_val  # 🆕 Local validation set (opzionale)
        self.has_local_validation = (x_val is not None and y_val is not None)
        self.compression_strategy = compression_strategy  # 🆕 Phase 2
        self.partition_id = partition_id  # 🆕 Client ID for logging
        
        # 🆕 Resource Profiling
        self.profiler = ResourceProfiler() if PROFILING_ENABLED else None
        self.resource_metrics = []
    
    def fit(self, parameters, config):
        """Training locale con tracking del tempo e ottimizzazioni memoria"""
        # Start timer
        start_time = time.time()
        
        # 🔥 PHASE 2: Decompressione parametri se compressi
        # Server passa compression_type nel config
        compression_type = config.get("compression_type", "none")
        decompression_time = 0.0
        if compression_type != "none" and self.compression_strategy is not None:
            # Parametri compressi - decomprimiamo
            try:
                # 🆕 Profile decompression
                if self.profiler:
                    with self.profiler.profile_operation("decompression"):
                        decomp_start = time.time()
                        parameters_array = decompress_from_parameters(
                            parameters,
                            compression_type,
                            self.compression_strategy
                        )
                        decompression_time = time.time() - decomp_start
                else:
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
            # Parametri standard
            parameters_array = parameters
        
        # Setta i parametri ricevuti dal server
        self.model.set_weights(parameters_array)
        
        # Ottieni iperparametri dal config
        epochs = config.get("local-epochs", 1)
        batch_size = config.get("batch-size", 32)
        verbose = config.get("verbose", 0)
        
        # 🆕 PHASE 4: Early Stopping config
        early_stopping_enabled = config.get("early_stopping_enabled", False)
        early_stopping_patience = config.get("early_stopping_patience", 3)
        early_stopping_min_delta = config.get("early_stopping_min_delta", 0.001)
        early_stopping_adaptive = config.get("early_stopping_adaptive", False)
        server_round = config.get("server_round", 1)  # Per adaptive stopping
        num_server_rounds = config.get("num_server_rounds", 10)  # Totale rounds
        
        # 🆕 Adatta batch size per dispositivi con poca RAM
        if IS_LOW_MEMORY:
            effective_batch_size = min(batch_size, 8)  # 🔧 Ridotto da 16 a 8 per Pi 3
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
                print(f"  � Early stopping enabled (patience: {early_stopping_patience})")
        elif early_stopping_enabled and not self.has_local_validation:
            print(f"  ⚠️  Early stopping requested but no local validation set - DISABLED")
        
        # Training - CUSTOM LOOP PER EARLY STOPPING
        training_time = 0.0
        actual_epochs = 0
        stopped_early = False
        epoch_losses = []
        epoch_accuracies = []
        val_losses = []
        val_accuracies = []
        
        train_start = time.time()
        
        if early_stopper is not None:
            # 🆕 PHASE 4: Training loop manuale con early stopping
            if self.profiler:
                with self.profiler.profile_operation("training"):
                    for epoch in range(epochs):
                        # Train per 1 epoch
                        history = self.model.fit(
                            self.x_train,
                            self.y_train,
                            epochs=1,
                            batch_size=effective_batch_size,
                            verbose=0,
                            validation_data=(self.x_val, self.y_val) if self.has_local_validation else None
                        )
                        
                        actual_epochs += 1
                        train_loss = history.history["loss"][0]
                        train_acc = history.history["accuracy"][0]
                        epoch_losses.append(train_loss)
                        epoch_accuracies.append(train_acc)
                        
                        # Validazione locale per early stopping
                        if self.has_local_validation:
                            val_loss = history.history["val_loss"][0]
                            val_acc = history.history["val_accuracy"][0]
                            val_losses.append(val_loss)
                            val_accuracies.append(val_acc)
                            
                            # Update early stopper
                            weights = self.model.get_weights()
                            early_stopper.update(val_loss, weights)
                            
                            # Check se fermarsi
                            if early_stopper.should_stop():
                                stopped_early = True
                                # Restore best weights
                                best_weights = early_stopper.get_best_weights()
                                if best_weights is not None:
                                    self.model.set_weights(best_weights)
                                print(f"  ⏹️  Early stopped at epoch {actual_epochs}/{epochs} (val_loss: {val_loss:.4f})")
                                break
                    
                    training_time = time.time() - train_start
            else:
                # No profiler
                for epoch in range(epochs):
                    history = self.model.fit(
                        self.x_train,
                        self.y_train,
                        epochs=1,
                        batch_size=effective_batch_size,
                        verbose=0,
                        validation_data=(self.x_val, self.y_val) if self.has_local_validation else None
                    )
                    
                    actual_epochs += 1
                    train_loss = history.history["loss"][0]
                    train_acc = history.history["accuracy"][0]
                    epoch_losses.append(train_loss)
                    epoch_accuracies.append(train_acc)
                    
                    if self.has_local_validation:
                        val_loss = history.history["val_loss"][0]
                        val_acc = history.history["val_accuracy"][0]
                        val_losses.append(val_loss)
                        val_accuracies.append(val_acc)
                        
                        weights = self.model.get_weights()
                        early_stopper.update(val_loss, weights)
                        
                        if early_stopper.should_stop():
                            stopped_early = True
                            best_weights = early_stopper.get_best_weights()
                            if best_weights is not None:
                                self.model.set_weights(best_weights)
                            print(f"  ⏹️  Early stopped at epoch {actual_epochs}/{epochs} (val_loss: {val_loss:.4f})")
                            break
                
                training_time = time.time() - train_start
            
            # Crea history object compatibile
            class FakeHistory:
                def __init__(self, losses, accs):
                    self.history = {
                        "loss": losses,
                        "accuracy": accs
                    }
            
            history = FakeHistory(epoch_losses, epoch_accuracies)
            
        else:
            # 🔧 Training standard (NO early stopping)
            if self.profiler:
                with self.profiler.profile_operation("training"):
                    history = self.model.fit(
                        self.x_train, 
                        self.y_train,
                        epochs=epochs,
                        batch_size=effective_batch_size,
                        verbose=0 if IS_LOW_MEMORY else verbose,
                    )
                    training_time = time.time() - train_start
            else:
                history = self.model.fit(
                    self.x_train, 
                    self.y_train,
                    epochs=epochs,
                    batch_size=effective_batch_size,
                    verbose=0 if IS_LOW_MEMORY else verbose,
                )
                training_time = time.time() - train_start
            
            actual_epochs = epochs
        
        # End timer
        fit_duration = time.time() - start_time
        
        # 🆕 Salva pesi PRIMA di pulire la sessione
        parameters_updated = self.model.get_weights()
        num_examples = len(self.x_train)
        
        # 🆕 PHASE 2: REAL COMPRESSION! 🔥
        # 🔧 MODIFICATO: Calcola bandwidth stats SEMPRE (anche senza compressione)
        compression_stats = {}
        metadata = None
        compression_time = 0.0
        if self.compression_strategy is not None:
            from flwr.common import parameters_to_ndarrays
            
            # 🆕 Profile compression
            if self.profiler:
                with self.profiler.profile_operation("compression"):
                    compress_start = time.time()
                    # Comprimi parametri usando helper function
                    compressed_parameters, metadata = compress_to_parameters(
                        parameters_updated,
                        self.compression_strategy
                    )
                    compression_time = time.time() - compress_start
            else:
                compress_start = time.time()
                compressed_parameters, metadata = compress_to_parameters(
                    parameters_updated,
                    self.compression_strategy
                )
                compression_time = time.time() - compress_start
            
            # 🔧 FIX: Flower NumPyClient richiede NDArrays, non Parameters!
            # Converti Parameters → NDArrays per il return
            parameters_to_return = parameters_to_ndarrays(compressed_parameters)
            
            compression_duration = compression_time  # Alias for backward compatibility
            
            # Ottieni stats compressione
            comp_stats = get_compression_stats(compressed_parameters, metadata)
            compression_stats = {
                "compression_duration": float(compression_duration),
                **comp_stats,
                # ⚠️ CRITICO: Passa solo compression_type (non metadata completi!)
                "compression_type": metadata['compression_type']
            }
            
            print(f"  🗜️  Compressed: {comp_stats['original_size_bytes']:,} → "
                  f"{comp_stats['compressed_size_bytes']:,} bytes "
                  f"({comp_stats['compression_ratio']:.2f}x, "
                  f"{comp_stats['bandwidth_saved_percent']:.1f}% saved)")
        else:
            # No compression - return standard parameters (già NDArrays)
            parameters_to_return = parameters_updated
            
            # 🔧 NUOVO: Calcola bandwidth stats anche SENZA compressione!
            from mnist_optimised.compressed_parameters import ndarrays_to_parameters
            params_for_stats = ndarrays_to_parameters(parameters_updated)
            config_compression_type = config.get("compression_type", "none")
            stats_metadata = {'compression_type': config_compression_type}
            comp_stats = get_compression_stats(params_for_stats, stats_metadata)
            compression_stats = {
                **comp_stats,
                "compression_type": config_compression_type
            }
        
        # 🆕 Pulizia memoria AGGRESSIVA (critico per Pi 3)
        if IS_LOW_MEMORY:
            # Forza garbage collection multiplo
            gc.collect()
            gc.collect()
            gc.collect()
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        # 🆕 Collect resource metrics from profiler
        resource_metrics = {}
        if self.profiler:
            summary = self.profiler.get_summary()
            resource_metrics = {
                "resource_training_cpu_peak": summary.get('training', {}).get('avg_cpu_peak', 0.0),
                "resource_training_memory_delta_mb": summary.get('training', {}).get('avg_memory_delta_mb', 0.0),
                "resource_compression_cpu_peak": summary.get('compression', {}).get('avg_cpu_peak', 0.0),
                "resource_compression_memory_delta_mb": summary.get('compression', {}).get('avg_memory_delta_mb', 0.0),
                "resource_decompression_cpu_peak": summary.get('decompression', {}).get('avg_cpu_peak', 0.0),
                "resource_decompression_memory_delta_mb": summary.get('decompression', {}).get('avg_memory_delta_mb', 0.0),
                "resource_training_time_ms": training_time * 1000,
                "resource_compression_time_ms": compression_time * 1000,
                "resource_decompression_time_ms": decompression_time * 1000,
            }
        
        # 🆕 PHASE 4: Early stopping metrics
        early_stopping_metrics = {}
        if early_stopper is not None:
            compute_savings = calculate_compute_savings(actual_epochs, epochs)
            early_stopping_stats = early_stopper.get_stats()
            
            early_stopping_metrics = {
                "early_stopping_enabled": True,
                "actual_epochs": actual_epochs,
                "max_epochs": epochs,
                "stopped_early": stopped_early,
                "epochs_saved": compute_savings['epochs_saved'],
                "compute_saved_pct": compute_savings['compute_saved_pct'],
                "best_val_loss": early_stopping_stats.get('best_loss', 0.0),
                "early_stopping_wait": early_stopping_stats.get('wait', 0),
            }
            
            # Aggiungi info adaptive se presente
            if early_stopping_adaptive:
                early_stopping_metrics["adaptive_patience"] = early_stopping_stats.get('adaptive_patience', early_stopping_patience)
                early_stopping_metrics["training_progress"] = early_stopping_stats.get('training_progress', 0.0)
        else:
            early_stopping_metrics = {
                "early_stopping_enabled": False,
                "actual_epochs": actual_epochs,
                "max_epochs": epochs,
            }
        
        # 🔧 Get partition_id and hostname for device tracking
        import socket
        hostname = socket.gethostname()
        partition_id = getattr(self, 'partition_id', -1)  # -1 if not set
        
        # Metriche da inviare al server
        metrics = {
            "train_loss": float(history.history["loss"][-1]),
            "train_acc": float(history.history["accuracy"][-1]),
            "fit_duration": float(fit_duration),
            "num_examples": num_examples,
            "effective_batch_size": effective_batch_size,
            "is_low_memory": IS_LOW_MEMORY,
            **compression_stats,       # 🆕 PHASE 2: REAL compression stats
            **resource_metrics,        # 🆕 Resource profiling metrics
            **early_stopping_metrics,  # 🆕 PHASE 4: Early stopping metrics
            # 🆕 DEVICE TRACKING (per identificare client anche se client_id cambia)
            "partition_id": partition_id,
            "hostname": hostname,
        }
        
        return parameters_to_return, num_examples, metrics
    
    def evaluate(self, parameters, config):
        """
        🆕 CLIENT-SIDE EVALUATION (Phase 1)
        
        Valuta il modello su validation set locale (se disponibile).
        Calcola metriche comprehensive: accuracy, loss, precision, recall, F1.
        
        Returns:
            loss, num_examples, metrics
        """
        print(f"\n🔍 [Client {self.partition_id}] evaluate() START")
        
        if not self.has_local_validation:
            # No local validation set - skip evaluation
            # 🔧 Ritorniamo None per indicare che questo client non può fare evaluation
            # Flower interpreterà questo come "skip this client" invece di "failure"
            print(f"  ⚠️  No local validation set - skipping evaluation")
            # Return None to indicate this client should be skipped for evaluation
            # Flower's strategy will handle this gracefully
            return 0.0, 0, {}  # num_examples=0 indica skip
        
        # Start timer
        start_time = time.time()
        
        try:
            # 🔧 FIX: Durante evaluate(), i parametri arrivano GIÀ decompressi (NDArrays)
            # Il server non li comprime durante evaluate, solo durante fit!
            # Quindi non dobbiamo decomprimere, usiamo direttamente
            parameters_array = parameters
            
            # Setta i parametri ricevuti dal server
            self.model.set_weights(parameters_array)
            
            # Evaluate su validation set locale
            loss, accuracy = self.model.evaluate(
                self.x_val, 
                self.y_val,
                verbose=0,
                batch_size=32
            )
        except Exception as e:
            print(f"  ❌ ERROR in model.evaluate: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 🆕 METRICHE DETTAGLIATE con sklearn
        try:
            # Predictions
            y_pred_probs = self.model.predict(self.x_val, verbose=0, batch_size=32)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = self.y_val
            
            # Calcola precision, recall, F1 per-class e macro-average
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            
            # Per-class metrics 
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            # Calculate confidence scores
            confidence_scores = np.max(y_pred_probs, axis=1)
            mean_confidence = float(np.mean(confidence_scores))
            
            # End timer
            eval_duration = time.time() - start_time
            
        except Exception as e:
            print(f"  ❌ ERROR in sklearn metrics: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 🔧 FIX: Converti TUTTO in tipi Python base per serializzazione Ray
        metrics = {
            "val_loss": float(loss),
            "val_accuracy": float(accuracy),
            "val_precision": float(precision),
            "val_recall": float(recall),
            "val_f1": float(f1),
            "val_mean_confidence": float(mean_confidence),
            "eval_duration": float(eval_duration),
            "num_val_examples": int(len(self.x_val)),
            # 🔧 SKIP per-class metrics for now - may cause serialization issues
            # "val_precision_per_class": [float(x) for x in precision_per_class],
            # "val_recall_per_class": [float(x) for x in recall_per_class],
            # "val_f1_per_class": [float(x) for x in f1_per_class],
            # "val_support_per_class": [int(x) for x in support_per_class],
        }
        
        # 🆕 Pulizia memoria (importante per Pi 3)
        if IS_LOW_MEMORY:
            gc.collect()
        
        result = (float(loss), int(len(self.x_val)), metrics)
        print(f"  🎯 [Client {self.partition_id}] evaluate() COMPLETE - returning {len(self.x_val)} samples")
        return result


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
    num_partitions = context.node_config.get("num-partitions", 3)
    
    # 🆕 Determina partition_id automaticamente dall'hostname (o usa node-config se fornito)
    partition_id = get_partition_id_from_hostname(context, num_partitions)
    compute_aware = context.run_config.get("compute-aware", False)  # 🆕
    partition_type = context.run_config.get("partition-type", "iid")  # 🆕
    classes_per_partition = context.run_config.get("classes-per-partition", 7)  # 🆕
    local_val_split = context.run_config.get("local-val-split", 0.0)  # 🆕 Client-side validation
    
    # 🆕 PHASE 2: Gradient Compression
    compression_type = context.run_config.get("compression-type", "none")
    compression_num_bits = context.run_config.get("compression-num-bits", 8)
    compression_k_percent = context.run_config.get("compression-k-percent", 0.1)
    
    compression_strategy = create_compression_strategy(
        compression_type=compression_type,
        num_bits=compression_num_bits,
        k_percent=compression_k_percent
    )
    
    # 🆕 Log info dispositivo
    print(f"\n{'='*60}")
    print(f"🖥️  Client {partition_id} Starting")
    print(f"{'='*60}")
    print(f"  RAM Total:     {TOTAL_RAM_GB:.2f} GB")
    print(f"  RAM Available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"  Low Memory:    {'⚠️  YES' if IS_LOW_MEMORY else '✅ NO'}")
    print(f"  Partition:     {partition_id}/{num_partitions}")
    print(f"  Partition Type: {partition_type.upper()}")
    if partition_type == "non-iid":
        print(f"  Classes/Client: {classes_per_partition}/10")
    print(f"  Compute-Aware: {'✅ ENABLED' if compute_aware else '❌ DISABLED'}")
    print(f"  Local Val Split: {local_val_split:.0%}" + (' (🆕 ENABLED)' if local_val_split > 0 else ' (DISABLED)'))
    print(f"  Compression:   {compression_type.upper()}" + 
          (f" ({compression_num_bits}-bit)" if compression_type == "quantization" else 
           f" (k={compression_k_percent:.1%})" if compression_type == "topk" else ""))
    print(f"{'='*60}\n")
    
    # Carica i dati (cached)
    data_result = get_data(
        partition_id, num_partitions, 
        compute_aware=compute_aware, 
        partition_type=partition_type, 
        classes_per_partition=classes_per_partition,
        local_val_split=local_val_split
    )
    
    # Parse result based on local_val_split
    if local_val_split > 0:
        # Con local validation split
        (x_train, y_train), (x_val, y_val) = data_result
    else:
        # Senza local validation split
        x_train, y_train = data_result
        x_val, y_val = None, None
    
    # Crea il modello (cached)
    model = get_model()
    
    # Crea e restituisci il client con compression strategy
    return MNISTClient(model, x_train, y_train, x_val, y_val, 
                       compression_strategy=compression_strategy,
                       partition_id=partition_id).to_client()


# Usa client_fn
app = ClientApp(client_fn=client_fn)