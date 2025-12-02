"""
UCI HAR: Flower ServerApp with Advanced Metrics + Gradient Compression
"""

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from har.task import load_model, load_global_validation_data, load_global_test_data
from har.compressed_strategy import CompressedFedAvg  # 🆕 Compression support
from har.distillation_strategy import DistillationFedAvg  # 🎓 Federated Distillation
from har.metrics import (
    fit_metrics_aggregation,
    evaluate_metrics_aggregation,
    evaluate_model_detailed,
    ConvergenceTracker,
    EfficiencyTracker,
    calculate_data_heterogeneity,
    print_metrics_summary,
    MetricsLogger
)
from har.client_metrics import client_tracker, enhanced_fit_metrics_aggregation
from pathlib import Path

# Trackers globali
convergence_tracker = ConvergenceTracker(patience=5, min_delta=0.001)
efficiency_tracker = EfficiencyTracker()
metrics_logger = None  # Sarà inizializzato in server_fn


def aggregate_bandwidth_stats(metrics):
    """
    🆕 Aggrega bandwidth stats dai client metrics
    
    Questa funzione raccoglie le statistiche di bandwidth inviate dai client
    e le aggrega per il tracking server-side (anche SENZA compressione).
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples from clients
    
    Returns:
        Dict with aggregated bandwidth stats
    """
    if not metrics:
        return {}
    
    # Raccogli stats da tutti i client
    total_original_bytes = 0
    total_compressed_bytes = 0
    num_clients_with_stats = 0
    compression_type = "none"
    
    for num_examples, client_metrics in metrics:
        # Check se il client ha inviato bandwidth stats
        if 'original_size_bytes' in client_metrics:
            total_original_bytes += client_metrics['original_size_bytes']
            total_compressed_bytes += client_metrics.get('compressed_size_bytes', client_metrics['original_size_bytes'])
            num_clients_with_stats += 1
            compression_type = client_metrics.get('compression_type', 'none')
    
    # Se nessun client ha stats, return empty
    if num_clients_with_stats == 0:
        return {}
    
    # Calcola aggregated stats
    compression_ratio = total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 1.0
    bandwidth_saved_percent = (1 - 1/compression_ratio) * 100 if compression_ratio > 0 else 0.0
    
    return {
        'round_bandwidth_original_kb': total_original_bytes / 1024,
        'round_bandwidth_compressed_kb': total_compressed_bytes / 1024,
        'round_bandwidth_saved_kb': (total_original_bytes - total_compressed_bytes) / 1024,
        'round_compression_ratio': compression_ratio,
        'round_bandwidth_saved_percent': bandwidth_saved_percent,
        'bandwidth_compression_type': compression_type,
    }


def get_evaluate_fn(x_val, y_val, num_classes=6, num_server_rounds=15, experiment_name=None):
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
    🆕 Salva nella cartella models/ del run corrente
    
    Args:
        model: Il modello Keras trainato
        experiment_name: Nome dell'esperimento (opzionale)
    """
    global metrics_logger
    
    # Ottieni il run_dir dal metrics_logger
    if metrics_logger and hasattr(metrics_logger, 'run_dir'):
        # Usa la cartella models/ del run corrente
        run_dir = metrics_logger.run_dir
        models_dir = run_dir / "models"
        models_dir.mkdir(exist_ok=True)
    else:
        # Fallback: crea cartella temporanea (non dovrebbe succedere)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results")
        run_dir = results_dir / f"run_{timestamp}"
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
    
    # Nome file
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
    🆕 Salva nella cartella logs/client_metrics/ del run corrente
    
    Args:
        experiment_name: Nome dell'esperimento (opzionale)
    """
    global metrics_logger
    
    # Ottieni il run_dir dal metrics_logger
    if metrics_logger and hasattr(metrics_logger, 'run_dir'):
        output_dir = str(metrics_logger.run_dir / "logs" / "client_metrics")
    else:
        # Fallback: usa vecchio formato
        if experiment_name:
            output_dir = f"results/client_metrics_{experiment_name}"
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/client_metrics_{timestamp}"
    
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
    fraction_evaluate = run_config.get("fraction-evaluate", 0.0)  # Disabilita evaluate distribuito
    num_server_rounds = run_config["num-server-rounds"]
    local_epochs = run_config["local-epochs"]
    batch_size = run_config["batch-size"]
    verbose = run_config.get("verbose", False)
    num_classes = run_config.get("num-classes", 6)  # UCI HAR ha 6 classi
    experiment_name = run_config.get("experiment-name", None)
    
    # 🆕 PHASE 4: Early stopping config
    early_stopping_enabled = run_config.get("early-stopping-enabled", False)
    early_stopping_patience = run_config.get("early-stopping-patience", 3)
    early_stopping_min_delta = run_config.get("early-stopping-min-delta", 0.001)
    early_stopping_adaptive = run_config.get("early-stopping-adaptive", False)
    
    # Inizializza metrics logger
    metrics_logger = MetricsLogger(experiment_name=experiment_name)
    
    # Initialize model parameters
    ndarrays = load_model().get_weights()
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Load VALIDATION set per evaluation durante training
    print("\n📊 Loading datasets:")
    print("  - Validation set: for monitoring during training")
    x_val, y_val = load_global_validation_data()
    print()
    
    # Config function for clients
    def fit_config(server_round: int):
        return {
            "local-epochs": local_epochs,
            "batch-size": batch_size,
            "verbose": verbose,
            # 🆕 PHASE 4: Early stopping params
            "early_stopping_enabled": early_stopping_enabled,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "early_stopping_adaptive": early_stopping_adaptive,
            "server_round": server_round,
            "num_server_rounds": num_server_rounds,
        }
    
    # Ottieni evaluate_fn (usa validation set) + passa num_server_rounds per salvare il modello
    evaluate_fn = get_evaluate_fn(x_val, y_val, num_classes, num_server_rounds, experiment_name)
    
    # Storage per il round corrente (per tracking client)
    current_round = [0]  # Usa lista per mutability in closure
    
    # Custom fit metrics aggregation con data heterogeneity E client tracking
    def fit_metrics_with_heterogeneity_and_tracking(metrics):
        current_round[0] += 1  # Incrementa round
        
        # USA IL NUOVO TRACKER PER-CLIENT
        aggregated = enhanced_fit_metrics_aggregation(metrics, current_round[0])
        
        # Aggiungi data heterogeneity
        heterogeneity = calculate_data_heterogeneity(metrics)
        aggregated.update(heterogeneity)
        
        # 🆕 BANDWIDTH TRACKING: Aggrega bandwidth stats dai client
        bandwidth_stats = aggregate_bandwidth_stats(metrics)
        aggregated.update(bandwidth_stats)
        
        # Salva per il logger
        evaluate_fn.set_training_metrics(aggregated)
        
        # 🔍 DEBUG: Check if bandwidth stats are in aggregated
        print(f"\n🔍 BANDWIDTH AGGREGATED DEBUG:")
        bandwidth_keys = [k for k in aggregated.keys() if 'bandwidth' in k or 'compression' in k]
        print(f"   Bandwidth keys in aggregated: {bandwidth_keys}")
        for k in bandwidth_keys:
            print(f"     {k}: {aggregated[k]}")
        
        # Print training summary
        print(f"\n📊 Training Metrics (Round {current_round[0]}):")
        print(f"  Train Accuracy: {aggregated['train_accuracy']:.4f} "
              f"(range: {aggregated['train_acc_min']:.4f}-{aggregated['train_acc_max']:.4f})")
        print(f"  Train Loss:     {aggregated['train_loss']:.4f}")
        print(f"  Clients:        {aggregated['num_clients']}")
        print(f"  Total Samples:  {aggregated['total_samples']}")
        print(f"  Data Balance:   Gini={aggregated['data_gini']:.3f} "
              f"({'✅ Balanced' if aggregated['data_gini'] < 0.3 else '⚠️ Imbalanced'})")
        
        # STAMPA CLIENT METRICS SUMMARY
        client_tracker.print_round_summary(current_round[0])
        
        return aggregated
    
    # Custom evaluate metrics aggregation
    def evaluate_metrics_with_fairness(metrics):
        # NOTA: Questa funzione ora NON verrà chiamata perché fraction_evaluate=0
        # I client non hanno più test set locale
        aggregated = evaluate_metrics_aggregation(metrics)
        
        # Salva per il logger
        evaluate_fn.set_distributed_metrics(aggregated)
        
        # Print evaluation summary
        print(f"\n🌐 Distributed Evaluation:")
        print(f"  Avg Accuracy:  {aggregated['distributed_accuracy']:.4f}")
        print(f"  Fairness Gap:  {aggregated['fairness_gap']:.4f} "
              f"({'✅ Fair' if aggregated['fairness_gap'] < 0.05 else '⚠️ Unfair'})")
        print(f"  CV:            {aggregated['eval_acc_cv']:.3f}")
        
        return aggregated
    
    # � FEDERATED DISTILLATION config
    distillation_enabled = run_config.get("distillation-enabled", False)
    distillation_batch_size = run_config.get("distillation-batch-size", 100)
    distillation_temperature = run_config.get("distillation-temperature", 3.0)
    distillation_epochs = run_config.get("distillation-epochs", 5)
    distillation_lr = run_config.get("distillation-lr", 0.001)
    
    # �🆕 Compression config (Fase 2 - Gradient Compression)
    compression_type = run_config.get("compression-type", "none")
    compression_num_bits = run_config.get("compression-num-bits", 8)
    compression_k_percent = run_config.get("compression-k-percent", 0.1)
    
    # Define the strategy - DISTILLATION vs COMPRESSION vs STANDARD
    if distillation_enabled:
        print(f"\n🎓 Using FEDERATED DISTILLATION Strategy")
        print(f"   Batch size: {distillation_batch_size}")
        print(f"   Temperature: {distillation_temperature}")
        print(f"   Distillation epochs: {distillation_epochs}")
        print(f"   Distillation LR: {distillation_lr}\n")
        
        strategy = DistillationFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=2,
            min_evaluate_clients=0,
            min_available_clients=2,
            initial_parameters=parameters,
            on_fit_config_fn=fit_config,
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=fit_metrics_with_heterogeneity_and_tracking,
            evaluate_metrics_aggregation_fn=evaluate_metrics_with_fairness if fraction_evaluate > 0 else None,
            # 🎓 Distillation parameters
            unlabeled_batch_size=distillation_batch_size,
            distillation_temperature=distillation_temperature,
            distillation_epochs=distillation_epochs,
            distillation_lr=distillation_lr,
        )
    else:
        # Standard Compression Strategy o FedAvg
        strategy = CompressedFedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,  # 0.0 = disabilita evaluate distribuito
            min_fit_clients=2,
            min_evaluate_clients=0,  # Non richiede client per evaluate
            min_available_clients=2,
            initial_parameters=parameters,
            on_fit_config_fn=fit_config,
            evaluate_fn=evaluate_fn,  # Solo valutazione centralizzata (validation set)
            fit_metrics_aggregation_fn=fit_metrics_with_heterogeneity_and_tracking,
            evaluate_metrics_aggregation_fn=evaluate_metrics_with_fairness if fraction_evaluate > 0 else None,
            # 🆕 Compression parameters
            compression_type=compression_type,
            compression_num_bits=compression_num_bits,
            compression_k_percent=compression_k_percent,
        )
    
    # Server config
    config = ServerConfig(num_rounds=num_server_rounds)
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting Federated Learning - UCI HAR")
    print(f"{'='*70}")
    print(f"Rounds:        {num_server_rounds}")
    print(f"Local Epochs:  {local_epochs}")
    print(f"Batch Size:    {batch_size}")
    print(f"Fraction Fit:  {fraction_fit}")
    print(f"Fraction Eval: {fraction_evaluate}")
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