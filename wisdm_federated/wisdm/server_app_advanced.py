"""
WISDM: Flower ServerApp with Advanced Metrics
Versione aggiornata con logging completo delle metriche
"""

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from wisdm.task import load_model, load_centralized_data
from wisdm.metrics import (
    fit_metrics_aggregation,
    evaluate_metrics_aggregation,
    evaluate_model_detailed,
    ConvergenceTracker,
    EfficiencyTracker,
    calculate_data_heterogeneity,
    print_metrics_summary,
    MetricsLogger
)

# Trackers globali
convergence_tracker = ConvergenceTracker(patience=5, min_delta=0.001)
efficiency_tracker = EfficiencyTracker()
metrics_logger = None  # Sarà inizializzato in server_fn


def get_evaluate_fn(x_test, y_test, num_classes=6):
    """Valutazione centralizzata con metriche dettagliate per WISDM"""
    # Storage per le metriche di training dell'ultimo round
    last_training_metrics = {}
    last_distributed_metrics = {}
    
    def evaluate(server_round, parameters_ndarrays, config):
        global metrics_logger
        
        # Start timing
        efficiency_tracker.start_round()
        
        # Carica modello e valuta
        model = load_model()
        model.set_weights(parameters_ndarrays)
        
        # Metriche dettagliate
        metrics = evaluate_model_detailed(model, x_test, y_test, num_classes)
        
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
        
        return metrics['loss'], metrics
    
    # Funzioni helper per salvare metriche intermedie
    evaluate.set_training_metrics = lambda m: last_training_metrics.update(m)
    evaluate.set_distributed_metrics = lambda m: last_distributed_metrics.update(m)
    
    return evaluate


def server_fn(context: fl.common.Context):
    """Construct components that set the ServerApp behaviour."""
    global metrics_logger
    
    # Read run_config
    run_config = context.run_config
    fraction_fit = run_config.get("fraction-fit", 1.0)
    fraction_evaluate = run_config.get("fraction-evaluate", 1.0)
    num_server_rounds = run_config["num-server-rounds"]
    local_epochs = run_config["local-epochs"]
    batch_size = run_config["batch-size"]
    verbose = run_config.get("verbose", False)
    num_classes = run_config.get("num-classes", 6)  # WISDM ha 6 classi
    experiment_name = run_config.get("experiment-name", None)
    
    # Inizializza metrics logger
    metrics_logger = MetricsLogger(experiment_name=experiment_name)
    
    # Initialize model parameters
    ndarrays = load_model().get_weights()
    parameters = ndarrays_to_parameters(ndarrays)
    
    # Load global test set
    x_test, y_test = load_centralized_data()
    
    # Config function for clients
    def fit_config(server_round: int):
        return {
            "local-epochs": local_epochs,
            "batch-size": batch_size,
            "verbose": verbose,
        }
    
    # Ottieni evaluate_fn
    evaluate_fn = get_evaluate_fn(x_test, y_test, num_classes)
    
    # Custom fit metrics aggregation con data heterogeneity
    def fit_metrics_with_heterogeneity(metrics):
        aggregated = fit_metrics_aggregation(metrics)
        heterogeneity = calculate_data_heterogeneity(metrics)
        aggregated.update(heterogeneity)
        
        # Salva per il logger
        evaluate_fn.set_training_metrics(aggregated)
        
        # Print training summary
        print(f"\n📊 Training Metrics (Round):")
        print(f"  Train Accuracy: {aggregated['train_accuracy']:.4f} "
              f"(range: {aggregated['train_acc_min']:.4f}-{aggregated['train_acc_max']:.4f})")
        print(f"  Train Loss:     {aggregated['train_loss']:.4f}")
        print(f"  Clients:        {aggregated['num_clients']}")
        print(f"  Total Samples:  {aggregated['total_samples']}")
        print(f"  Data Balance:   Gini={aggregated['data_gini']:.3f} "
              f"({'✅ Balanced' if aggregated['data_gini'] < 0.3 else '⚠️ Imbalanced'})")
        
        return aggregated
    
    # Custom evaluate metrics aggregation
    def evaluate_metrics_with_fairness(metrics):
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
    
    # Define the strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        initial_parameters=parameters,
        on_fit_config_fn=fit_config,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=fit_metrics_with_heterogeneity,
        evaluate_metrics_aggregation_fn=evaluate_metrics_with_fairness,
    )
    
    # Server config
    config = ServerConfig(num_rounds=num_server_rounds)
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting Federated Learning - WISDM HAR")
    print(f"{'='*70}")
    print(f"Rounds:        {num_server_rounds}")
    print(f"Local Epochs:  {local_epochs}")
    print(f"Batch Size:    {batch_size}")
    print(f"Fraction Fit:  {fraction_fit}")
    print(f"Fraction Eval: {fraction_evaluate}")
    print(f"Num Classes:   {num_classes} (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing)")
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
            'num_classes': num_classes,
            'dataset': 'WISDM',
            'task': 'Human Activity Recognition'
        })
    
    return ServerAppComponents(strategy=strategy, config=config)


# ServerApp
app = ServerApp(server_fn=server_fn)
