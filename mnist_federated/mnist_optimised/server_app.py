"""
MNIST: Flower ServerApp with Advanced Metrics + Real Compression
"""

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from mnist_optimised.task import load_model, load_global_validation_data, load_global_test_data
from mnist_optimised.metrics import (
    fit_metrics_aggregation,
    evaluate_metrics_aggregation,
    evaluate_model_detailed,
    ConvergenceTracker,
    EfficiencyTracker,
    calculate_data_heterogeneity,
    print_metrics_summary,
    MetricsLogger
)
from mnist_optimised.client_metrics import (
    client_tracker, 
    enhanced_fit_metrics_aggregation,
    enhanced_evaluate_metrics_aggregation
)
from mnist_optimised.compressed_strategy import CompressedFedAvg
from pathlib import Path

# Trackers globali
convergence_tracker = ConvergenceTracker(patience=5, min_delta=0.001)
efficiency_tracker = EfficiencyTracker()
metrics_logger = None  # Sarà inizializzato in server_fn


def aggregate_bandwidth_stats(metrics):
    """
    🆕 UNIVERSAL BANDWIDTH TRACKING
    
    Aggrega bandwidth stats da tutti i client, funziona con qualsiasi tipo di compressione:
    - none: baseline senza compressione
    - quantization: compressione con quantizzazione
    - distillation: comunicazione tramite predictions
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples from clients
        
    Returns:
        dict: Aggregated bandwidth metrics (total across all clients for this round)
    """
    total_original_bytes = 0
    total_compressed_bytes = 0
    compression_type = 'unknown'
    num_clients_with_stats = 0
    
    # Collect bandwidth stats from all clients
    for num_examples, client_metrics in metrics:
        if 'original_size_bytes' in client_metrics and 'compressed_size_bytes' in client_metrics:
            total_original_bytes += client_metrics['original_size_bytes']
            total_compressed_bytes += client_metrics['compressed_size_bytes']
            compression_type = client_metrics.get('compression_type', 'unknown')
            num_clients_with_stats += 1
    
    # Calculate aggregated stats
    bandwidth_stats = {}
    if num_clients_with_stats > 0:
        # Convert to KB for readability
        original_kb = total_original_bytes / 1024
        compressed_kb = total_compressed_bytes / 1024
        saved_kb = original_kb - compressed_kb
        
        # Calculate compression ratio and percentage
        compression_ratio = original_kb / compressed_kb if compressed_kb > 0 else 1.0
        bandwidth_saved_percent = (saved_kb / original_kb * 100) if original_kb > 0 else 0.0
        
        bandwidth_stats = {
            'round_bandwidth_original_kb': original_kb,
            'round_bandwidth_compressed_kb': compressed_kb,
            'round_bandwidth_saved_kb': saved_kb,
            'round_compression_ratio': compression_ratio,
            'round_bandwidth_saved_percent': bandwidth_saved_percent,
            'bandwidth_compression_type': compression_type
        }
    
    return bandwidth_stats


def get_evaluate_fn(x_val, y_val, num_classes=10, num_server_rounds=10, experiment_name=None):
    """
    Valutazione centralizzata con metriche dettagliate
    
    Usa il VALIDATION SET durante il training per monitorare convergenza.
    Il test set viene usato solo alla fine per valutazione finale.
    
    IMPORTANTE: Salva automaticamente il modello finale dopo l'ultimo round.
    """
    # Storage per le metriche di training dell'ultimo round
    last_training_metrics = {}
    last_distributed_metrics = {}
    
    def evaluate(server_round, parameters_ndarrays, config):
        global metrics_logger
        
        # Start timing
        efficiency_tracker.start_round()
        
        # Carica modello e valuta SUL VALIDATION SET
        model = load_model()
        model.set_weights(parameters_ndarrays)
        
        # Metriche dettagliate
        metrics = evaluate_model_detailed(model, x_val, y_val, num_classes)
        
        # Aggiorna convergence tracker
        convergence_tracker.update(server_round, metrics['loss'], metrics['accuracy'])
        
        # End timing
        round_time = efficiency_tracker.end_round()
        
        # Aggiungi metriche di convergenza ed efficienza
        metrics.update(convergence_tracker.get_metrics())
        metrics.update(efficiency_tracker.get_metrics())
        metrics['round_time'] = round_time
        
        # Print summary
        print_metrics_summary(
            server_round,
            centralized_metrics=metrics,
            distributed_metrics=last_distributed_metrics,
            convergence_metrics=convergence_tracker.get_metrics(),
            efficiency_metrics=efficiency_tracker.get_metrics()
        )
        
        # Log metriche
        if metrics_logger:
            metrics_logger.log_round(
                round_num=server_round,
                centralized_metrics=metrics,
                distributed_metrics=last_distributed_metrics,
                training_metrics=last_training_metrics,
                convergence_metrics=convergence_tracker.get_metrics(),
                efficiency_metrics=efficiency_tracker.get_metrics()
            )
        
        # Check early stopping
        if convergence_tracker.should_stop():
            print("⚠️  Convergence plateau detected! Consider stopping training.")
            # 🔥 SALVA IL MODELLO quando viene rilevato plateau
            print("💾 Saving model due to plateau detection...")
            save_final_model(model, experiment_name)
            export_client_metrics(experiment_name)
        
        # 🆕 SALVA IL MODELLO FINALE dopo l'ultimo round
        if server_round == num_server_rounds:
            save_final_model(model, experiment_name)
            # 🆕 ESPORTA ANCHE LE METRICHE CLIENT
            export_client_metrics(experiment_name)
        
        return metrics['loss'], metrics
    
    # Funzioni helper per salvare metriche intermedie
    evaluate.set_training_metrics = lambda m: last_training_metrics.update(m)
    evaluate.set_distributed_metrics = lambda m: last_distributed_metrics.update(m)
    
    return evaluate


def save_final_model(model, experiment_name=None):
    """
    Salva il modello finale dopo il training
    🆕 Salvato nella sottocartella models/ della run corrente
    
    Args:
        model: Il modello Keras trainato
        experiment_name: Nome dell'esperimento (opzionale, non usato nella nuova struttura)
    """
    # Ottieni il run_dir dal metrics_logger
    global metrics_logger
    
    if metrics_logger and hasattr(metrics_logger, 'models_dir'):
        # 🆕 Usa la cartella models/ della run corrente
        model_path = metrics_logger.models_dir / "final_model.weights.h5"
        run_dir = metrics_logger.run_dir
    else:
        # Fallback (non dovrebbe succedere)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results")
        run_dir = results_dir / f"run_{timestamp}"
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "final_model.weights.h5"
    
    # Salva weights
    model.save_weights(str(model_path))
    
    print(f"\n{'='*70}")
    print(f"💾 MODELLO FINALE SALVATO")
    print(f"{'='*70}")
    print(f"📂 Path: {model_path}")
    print(f"📊 Ora puoi usare:")
    print(f"   python final_test_evaluation.py --model-path {model_path}")
    print(f"   oppure:")
    print(f"   python analyze_run.py {run_dir}")
    print(f"{'='*70}\n")


def export_client_metrics(experiment_name=None):
    """
    Esporta le metriche per-client alla fine del training
    🆕 Salvato nella sottocartella logs/ della run corrente
    
    Args:
        experiment_name: Nome dell'esperimento (opzionale, non usato nella nuova struttura)
    """
    # Ottieni il logs_dir dal metrics_logger
    global metrics_logger
    
    if metrics_logger and hasattr(metrics_logger, 'logs_dir'):
        # 🆕 Usa la cartella logs/ della run corrente
        output_dir = str(metrics_logger.logs_dir / "client_metrics")
    else:
        # Fallback (non dovrebbe succedere)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/run_{timestamp}/logs/client_metrics"
    
    # Esporta
    client_tracker.export_client_data(output_dir)
    
    print(f"\n{'='*70}")
    print(f"📊 CLIENT METRICS ESPORTATE")
    print(f"{'='*70}")
    print(f"📂 Directory: {output_dir}/")
    print(f"📄 File creati:")
    print(f"   - client_summary.csv (stats aggregate per client)")
    print(f"   - client_X_history.csv (time series per ogni client)")
    print(f"   - round_client_stats.csv (stats per round)")
    print(f"   - client_metrics_full.json (dati completi)")
    print(f"{'='*70}\n")


def server_fn(context: fl.common.Context):
    """Construct components that set the ServerApp behaviour."""
    global metrics_logger
    
    # Read run_config
    run_config = context.run_config
    fraction_fit = run_config.get("fraction-fit", 1.0)
    fraction_evaluate = run_config.get("fraction-evaluate", 0.0)  # Default: disabilita evaluate distribuito
    num_server_rounds = run_config["num-server-rounds"]
    local_epochs = run_config["local-epochs"]
    batch_size = run_config["batch-size"]
    verbose = run_config.get("verbose", False)
    num_classes = run_config.get("num-classes", 10)
    experiment_name = run_config.get("experiment-name", None)
    local_val_split = run_config.get("local-val-split", 0.0)  # 🆕 Client-side validation
    
    # 🔥 PHASE 2: Compression config
    compression_type = run_config.get("compression-type", "none")
    compression_num_bits = run_config.get("compression-num-bits", 8)
    compression_k_percent = run_config.get("compression-k-percent", 0.1)
    
    # 🆕 PHASE 4: Early Stopping config
    early_stopping_enabled = run_config.get("early-stopping-enabled", False)
    early_stopping_patience = run_config.get("early-stopping-patience", 3)
    early_stopping_min_delta = run_config.get("early-stopping-min-delta", 0.001)
    early_stopping_adaptive = run_config.get("early-stopping-adaptive", False)
    
    # 🆕 FIXED: L'evaluation distribuita ora funziona correttamente!
    # Se i client hanno local validation, abilitiamo distributed evaluation
    if local_val_split > 0:
        print(f"\n🆕 LOCAL VALIDATION enabled on clients (local-val-split={local_val_split:.0%})")
        print(f"    Clients will use local validation for training monitoring")
        
        # 🔧 AUTO-ENABLE distributed evaluation se non esplicitamente disabilitato
        if fraction_evaluate == 0.0:
            fraction_evaluate = 1.0  # Abilita evaluation su tutti i client con validation locale
            print(f"    ✅ Distributed evaluation AUTO-ENABLED")
        
        print(f"    Server will use BOTH centralized AND distributed evaluation\n")
    elif fraction_evaluate > 0:
        print(f"\n⚠️  WARNING: fraction-evaluate > 0 but local-val-split = 0")
        print(f"    Clients have NO local validation set!")
        print(f"    Distributed evaluation will be skipped by clients.\n")
    
    # Inizializza metrics logger
    metrics_logger = MetricsLogger(experiment_name=experiment_name)
    
    # Initialize model parameters
    ndarrays = load_model().get_weights()
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Load VALIDATION set per evaluation durante training
    print("\n📊 Loading datasets:")
    print("  - Validation set: for monitoring during training")
    x_val, y_val = load_global_validation_data()
    
    # Config function for clients (training)
    def fit_config(server_round: int):
        return {
            "local-epochs": local_epochs,
            "batch-size": batch_size,
            "verbose": verbose,
            # 🆕 PHASE 4: Early stopping config
            "early_stopping_enabled": early_stopping_enabled,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "early_stopping_adaptive": early_stopping_adaptive,
            "server_round": server_round,  # Per adaptive stopping
            "num_server_rounds": num_server_rounds,  # Totale rounds
        }
    
    # 🆕 Config function for clients (evaluation)
    def evaluate_config(server_round: int):
        """
        Configuration passata ai client durante evaluate()
        Importante: deve includere compression_type per decompressione corretta!
        """
        return {
            "compression_type": compression_type,
            "compression_num_bits": compression_num_bits,
            "compression_k_percent": compression_k_percent,
        }
    
    # Ottieni evaluate_fn (usa validation set) + passa num_server_rounds per salvare il modello
    evaluate_fn = get_evaluate_fn(x_val, y_val, num_classes, num_server_rounds, experiment_name)
    
    # Storage per il round corrente (per tracking client)
    current_round = [0]  # Usa lista per mutability in closure
    strategy_ref = [None]  # Reference alla strategy (verrà settata dopo)
    
    # Custom fit metrics aggregation con data heterogeneity E client tracking
    def fit_metrics_with_heterogeneity_and_tracking(metrics):
        current_round[0] += 1  # Incrementa round
        
        # 🆕 USA IL NUOVO TRACKER PER-CLIENT
        aggregated = enhanced_fit_metrics_aggregation(metrics, current_round[0])
        
        # Aggiungi data heterogeneity
        heterogeneity = calculate_data_heterogeneity(metrics)
        aggregated.update(heterogeneity)
        
        # 🆕 UNIVERSAL BANDWIDTH TRACKING - aggrega bandwidth stats da tutti i client
        bandwidth_stats = aggregate_bandwidth_stats(metrics)
        aggregated.update(bandwidth_stats)
        
        # 🐛 DEBUG: Verifica che bandwidth stats siano presenti
        if bandwidth_stats:
            print(f"  🌐 Bandwidth: {bandwidth_stats.get('round_bandwidth_compressed_kb', 0):.2f} KB "
                  f"(type: {bandwidth_stats.get('bandwidth_compression_type', 'unknown')})")
        
        # Salva per il logger
        evaluate_fn.set_training_metrics(aggregated)
        
        # Print training summary
        print(f"\n📊 Training Metrics (Round {current_round[0]}):")
        print(f"  Train Accuracy: {aggregated['train_accuracy']:.4f} "
              f"(range: {aggregated['train_acc_min']:.4f}-{aggregated['train_acc_max']:.4f})")
        print(f"  Train Loss:     {aggregated['train_loss']:.4f}")
        print(f"  Clients:        {aggregated['num_clients']}")
        print(f"  Total Samples:  {aggregated['total_samples']}")
        print(f"  Data Balance:   Gini={aggregated['data_gini']:.3f} "
              f"({'✅ Balanced' if aggregated['data_gini'] < 0.3 else '⚠️ Imbalanced'})")
        
        # 🆕 STAMPA CLIENT METRICS SUMMARY
        client_tracker.print_round_summary(current_round[0])
        
        return aggregated
    
    # Custom evaluate metrics aggregation
    def evaluate_metrics_with_validation(metrics):
        """
        🆕 ENHANCED: Gestisce validation metrics dai client
        
        Questa funzione viene chiamata quando fraction_evaluate > 0 e i client
        hanno local validation set (local_val_split > 0).
        """
        # 🆕 USA LA NUOVA FUNZIONE PER VALIDATION METRICS
        aggregated = enhanced_evaluate_metrics_aggregation(metrics, current_round[0])
        
        # Salva per il logger
        evaluate_fn.set_distributed_metrics(aggregated)
        
        # Print validation summary
        if aggregated:
            print(f"\n🌐 Distributed Validation (Client-Side):")
            print(f"  Avg Accuracy:  {aggregated.get('distributed_val_accuracy', 0):.4f}")
            print(f"  Avg F1 Score:  {aggregated.get('distributed_val_f1', 0):.4f}")
            print(f"  Fairness Gap:  {aggregated.get('val_fairness_gap', 0):.4f} "
                  f"({'✅ Fair' if aggregated.get('val_fairness_gap', 0) < 0.05 else '⚠️ Unfair'})")
            print(f"  Clients:       {aggregated.get('num_clients_evaluated', 0)}")
        
        return aggregated
    
    # 🔥 Define the strategy - USE CompressedFedAvg for REAL compression!
    strategy = CompressedFedAvg(
        # Compression params
        compression_type=compression_type,
        compression_num_bits=compression_num_bits,
        compression_k_percent=compression_k_percent,
        # Standard FedAvg params
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=2,
        min_evaluate_clients=2 if fraction_evaluate > 0 else 0,
        min_available_clients=2,
        initial_parameters=parameters,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,  # 🆕 FIXED: ora passa config anche a evaluate!
        evaluate_fn=evaluate_fn,  # Valutazione centralizzata (validation set)
        fit_metrics_aggregation_fn=fit_metrics_with_heterogeneity_and_tracking,
        evaluate_metrics_aggregation_fn=evaluate_metrics_with_validation if fraction_evaluate > 0 else None,
    )
    
    # 🔥 Setta il riferimento alla strategy nella closure
    strategy_ref[0] = strategy
    
    # Server config
    config = ServerConfig(num_rounds=num_server_rounds)
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting Federated Learning")
    print(f"{'='*70}")
    print(f"Rounds:        {num_server_rounds}")
    print(f"Local Epochs:  {local_epochs}")
    print(f"Batch Size:    {batch_size}")
    print(f"Fraction Fit:  {fraction_fit}")
    print(f"Fraction Eval: {fraction_evaluate}")
    if compression_type != "none":
        print(f"🔥 Compression:  {compression_type.upper()}", end="")
        if compression_type == "quantization":
            print(f" ({compression_num_bits}-bit)")
        elif compression_type == "topk":
            print(f" (k={compression_k_percent*100:.1f}%)")
        else:
            print()
    if experiment_name:
        print(f"Experiment:    {experiment_name}")
    print(f"{'='*70}\n")
    
    # Salva configurazione nel logger
    if metrics_logger:
        metrics_logger.save_final_summary({
            'num_rounds': num_server_rounds,
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'fraction_fit': fraction_fit,
            'fraction_evaluate': fraction_evaluate,
            'num_classes': num_classes
        })
    
    return ServerAppComponents(strategy=strategy, config=config)


# ServerApp
app = ServerApp(server_fn=server_fn)