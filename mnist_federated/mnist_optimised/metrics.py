"""
Advanced Metrics for Federated Learning
Metriche utili per analizzare qualità e fairness del FL
"""

import numpy as np
from typing import List, Tuple, Dict
from flwr.common import Metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time
import json
import csv
from pathlib import Path
from datetime import datetime


# ==================== METRICHE DI TRAINING ====================

def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggrega metriche di training con statistiche avanzate
    """
    # Estrai valori
    train_accs = [m["train_acc"] for _, m in metrics]
    train_losses = [m["train_loss"] for _, m in metrics]
    num_examples = [num for num, _ in metrics]
    
    # Media pesata (standard)
    weighted_acc = sum(m["train_acc"] * num for num, m in metrics) / sum(num_examples)
    weighted_loss = sum(m["train_loss"] * num for num, m in metrics) / sum(num_examples)
    
    # Statistiche di dispersione (quanto variano i client?)
    acc_std = np.std(train_accs)
    acc_min = np.min(train_accs)
    acc_max = np.max(train_accs)
    
    loss_std = np.std(train_losses)
    
    aggregated = {
        # Metriche standard
        "train_accuracy": weighted_acc,
        "train_loss": weighted_loss,
        
        # Dispersione tra client
        "train_acc_std": acc_std,           # Deviazione standard accuracy
        "train_acc_range": acc_max - acc_min,  # Range tra miglior e peggior client
        "train_acc_min": acc_min,           # Peggior client
        "train_acc_max": acc_max,           # Miglior client
        
        "train_loss_std": loss_std,
        
        # Dataset statistics
        "total_samples": sum(num_examples),
        "num_clients": len(metrics),
    }
    
    return aggregated


def evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggrega metriche di evaluation con statistiche di fairness
    """
    # Estrai valori
    eval_accs = [m["eval_acc"] for _, m in metrics]
    eval_losses = [m["eval_loss"] for _, m in metrics]
    num_examples = [num for num, _ in metrics]
    
    # Media pesata
    weighted_acc = sum(m["eval_acc"] * num for num, m in metrics) / sum(num_examples)
    weighted_loss = sum(m["eval_loss"] * num for num, m in metrics) / sum(num_examples)
    
    # Fairness metrics
    acc_std = np.std(eval_accs)
    acc_min = np.min(eval_accs)
    acc_max = np.max(eval_accs)
    
    # Coefficient of Variation (CV) - misura relativa di dispersione
    cv = acc_std / weighted_acc if weighted_acc > 0 else 0
    
    aggregated = {
        # Metriche standard
        "distributed_accuracy": weighted_acc,
        "distributed_loss": weighted_loss,
        
        # Fairness tra client
        "eval_acc_std": acc_std,
        "eval_acc_min": acc_min,
        "eval_acc_max": acc_max,
        "eval_acc_range": acc_max - acc_min,
        "eval_acc_cv": cv,  # <0.1 = buona fairness, >0.2 = poor fairness
        
        # Quanto è equo il modello per tutti i client?
        "fairness_gap": acc_max - acc_min,  # Ideale: <5%
    }
    
    return aggregated


# ==================== METRICHE DI VALUTAZIONE CENTRALIZZATA ====================

def evaluate_model_detailed(model, x_test, y_test, num_classes=10):
    """
    Valutazione dettagliata del modello con metriche per classe
    
    Returns:
        dict con loss, accuracy, precision, recall, f1-score
    """
    # Predizioni
    y_pred_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Se y_test è one-hot encoded, converti
    if len(y_test.shape) > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test
    
    # Loss e accuracy standard
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Metriche per classe
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Weighted (considera class imbalance)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class accuracy
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy),
        
        # Metriche macro (tratta tutte le classi ugualmente)
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        
        # Metriche weighted (considera frequenza classi)
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        
        # Worst-case performance
        "worst_class_acc": float(np.min(per_class_acc)),
        "best_class_acc": float(np.max(per_class_acc)),
        "class_acc_std": float(np.std(per_class_acc)),
        
        # Per-class breakdown
        "per_class_accuracy": [float(acc) for acc in per_class_acc],
    }
    
    return metrics


# ==================== METRICHE DI CONVERGENZA ====================

class ConvergenceTracker:
    """
    Traccia la convergenza del training per rilevare plateau
    """
    
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_round = 0
        self.no_improvement_count = 0
        self.history = []
    
    def update(self, round_num, loss, accuracy):
        """Aggiorna con nuove metriche"""
        self.history.append({
            'round': round_num,
            'loss': loss,
            'accuracy': accuracy
        })
        
        improvement = self.best_loss - loss
        
        if improvement > self.min_delta:
            self.best_loss = loss
            self.best_round = round_num
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
    
    def should_stop(self):
        """Indica se il training dovrebbe fermarsi"""
        return self.no_improvement_count >= self.patience
    
    def get_metrics(self):
        """Ottieni metriche di convergenza"""
        if len(self.history) < 2:
            return {}
        
        recent = self.history[-3:]  # Ultimi 3 round
        avg_recent_loss = np.mean([h['loss'] for h in recent])
        
        return {
            "rounds_since_improvement": self.no_improvement_count,
            "best_round": self.best_round,
            "best_loss": self.best_loss,
            "current_plateau": self.should_stop(),
            "avg_recent_loss": avg_recent_loss,
        }


# ==================== METRICHE DI EFFICIENZA ====================

class EfficiencyTracker:
    """
    Traccia efficienza temporale ed energetica
    """
    
    def __init__(self):
        self.round_times = []
        self.round_start = None
    
    def start_round(self):
        """Marca inizio round"""
        self.round_start = time.time()
    
    def end_round(self):
        """Marca fine round e calcola durata"""
        if self.round_start is not None:
            duration = time.time() - self.round_start
            self.round_times.append(duration)
            self.round_start = None
            return duration
        return 0
    
    def get_metrics(self):
        """Ottieni metriche di efficienza"""
        if not self.round_times:
            return {}
        
        return {
            "avg_round_time": np.mean(self.round_times),
            "total_time": sum(self.round_times),
            "fastest_round": np.min(self.round_times),
            "slowest_round": np.max(self.round_times),
            "time_std": np.std(self.round_times),
        }


# ==================== METRICHE DI DATA HETEROGENEITY ====================

def calculate_data_heterogeneity(metrics: List[Tuple[int, Metrics]]) -> Dict:
    """
    Calcola quanto sono eterogenei i dati tra i client
    Utile per capire se il partizionamento è IID o non-IID
    """
    num_examples = [num for num, _ in metrics]
    
    # Gini coefficient (0 = perfettamente bilanciato, 1 = tutto su un client)
    sorted_examples = sorted(num_examples)
    n = len(sorted_examples)
    cumsum = np.cumsum(sorted_examples)
    gini = (2 * sum((i+1) * x for i, x in enumerate(sorted_examples)) / 
            (n * sum(sorted_examples))) - (n+1)/n
    
    # Coefficient of Variation
    mean_examples = np.mean(num_examples)
    std_examples = np.std(num_examples)
    cv = std_examples / mean_examples if mean_examples > 0 else 0
    
    return {
        "data_gini": gini,  # <0.2 = bilanciato, >0.5 = molto sbilanciato
        "data_cv": cv,
        "data_imbalance_ratio": max(num_examples) / min(num_examples) if min(num_examples) > 0 else 0,
        "min_client_samples": min(num_examples),
        "max_client_samples": max(num_examples),
    }


# ==================== STAMPA METRICHE ====================

# ==================== METRICHE LOGGER ====================

class MetricsLogger:
    """
    Salva le metriche in JSON e CSV per analisi successive
    🆕 Ora organizza tutto in una cartella unificata per run
    """
    
    def __init__(self, experiment_name=None):
        # Nome esperimento con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.exp_name = f"{experiment_name}_{timestamp}"
        else:
            self.exp_name = f"fl_experiment_{timestamp}"
        
        # 🆕 Crea cartella unificata per questa run: results/run_TIMESTAMP/
        self.results_dir = Path("results")
        self.run_dir = self.results_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 🆕 Crea sottocartelle
        self.data_dir = self.run_dir / "data"
        self.models_dir = self.run_dir / "models"
        self.plots_dir = self.run_dir / "plots"
        self.logs_dir = self.run_dir / "logs"
        
        for subdir in [self.data_dir, self.models_dir, self.plots_dir, self.logs_dir]:
            subdir.mkdir(exist_ok=True)
        
        # 🆕 Paths dei file nella sottocartella data/
        self.json_path = self.data_dir / "metrics.json"
        self.csv_path = self.data_dir / "metrics.csv"
        
        # Salva il timestamp per riuso
        self.timestamp = timestamp
        
        # Storage metriche
        self.rounds_data = []
        
        # Inizializza CSV con header
        self._init_csv()
        
        print(f"\n💾 Run directory creata: {self.run_dir}/")
        print(f"   📂 data/    - metrics.json, metrics.csv")
        print(f"   📂 models/  - final_model.weights.h5")
        print(f"   📂 plots/   - grafici e visualizzazioni")
        print(f"   📂 logs/    - client metrics e altri log\n")
    
    def _init_csv(self):
        """Inizializza il file CSV con header"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header base - aggiungeremo colonne dinamicamente
            writer.writerow(['round', 'timestamp'])
    
    def log_round(self, round_num, centralized_metrics, distributed_metrics=None,
                  training_metrics=None, convergence_metrics=None, efficiency_metrics=None):
        """
        Salva le metriche di un round
        """
        timestamp = datetime.now().isoformat()
        
        # Costruisci dizionario completo
        round_data = {
            'round': round_num,
            'timestamp': timestamp,
            'centralized': centralized_metrics or {},
            'distributed': distributed_metrics or {},
            'training': training_metrics or {},
            'convergence': convergence_metrics or {},
            'efficiency': efficiency_metrics or {}
        }
        
        self.rounds_data.append(round_data)
        
        # Salva JSON (sovrascrive con tutti i dati)
        self._save_json()
        
        # Append al CSV
        self._append_csv(round_data)
    
    def _save_json(self):
        """Salva tutti i dati in JSON"""
        with open(self.json_path, 'w') as f:
            json.dump({
                'experiment': self.exp_name,
                'rounds': self.rounds_data
            }, f, indent=2)
    
    def _append_csv(self, round_data):
        """Aggiungi una riga al CSV"""
        # Flatten del dizionario
        flat_data = {'round': round_data['round'], 'timestamp': round_data['timestamp']}
        
        for category, metrics in round_data.items():
            if category not in ['round', 'timestamp'] and isinstance(metrics, dict):
                for key, value in metrics.items():
                    flat_key = f"{category}_{key}"
                    flat_data[flat_key] = value
        
        # 🔧 FIX: Verifica se ci sono nuove colonne rispetto all'header esistente
        needs_header_update = False
        if self.csv_path.exists():
            # Leggi l'header esistente
            import pandas as pd
            try:
                existing_df = pd.read_csv(self.csv_path)
                existing_cols = set(existing_df.columns)
                new_cols = set(flat_data.keys())
                
                # Se ci sono nuove colonne, dobbiamo riscrivere tutto
                if new_cols != existing_cols:
                    needs_header_update = True
                    print(f"🔄 Aggiornamento header CSV: nuove colonne rilevate")
                    print(f"   Nuove: {new_cols - existing_cols}")
            except:
                needs_header_update = True
        else:
            needs_header_update = True
        
        # Se è il primo round O ci sono nuove colonne, riscriviamo tutto il CSV
        if needs_header_update:
            # Riscriviamo tutto il CSV con header aggiornato
            import pandas as pd
            all_rows = []
            
            # Raccogli tutti i round precedenti se esistono
            if self.csv_path.exists():
                try:
                    existing_df = pd.read_csv(self.csv_path)
                    all_rows = existing_df.to_dict('records')
                except:
                    pass
            
            # Aggiungi la nuova riga
            all_rows.append(flat_data)
            
            # Scrivi tutto con il nuovo header completo
            with open(self.csv_path, 'w', newline='') as f:
                # Unione di tutte le chiavi da tutti i round
                all_keys = set()
                for row in all_rows:
                    all_keys.update(row.keys())
                all_keys = ['round', 'timestamp'] + sorted([k for k in all_keys if k not in ['round', 'timestamp']])
                
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                for row in all_rows:
                    writer.writerow(row)
        else:
            # Append normale
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=flat_data.keys())
                writer.writerow(flat_data)
    
    def save_final_summary(self, additional_info=None):
        """
        Salva un summary finale con informazioni aggiuntive
        🆕 Salvato nella sottocartella data/
        """
        summary_path = self.data_dir / "summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"FEDERATED LEARNING EXPERIMENT SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Experiment: {self.exp_name}\n")
            f.write(f"Run Directory: {self.run_dir}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Rounds: {len(self.rounds_data)}\n\n")
            
            if additional_info:
                f.write("Configuration:\n")
                for key, value in additional_info.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Best results
            if self.rounds_data:
                best_acc_round = max(self.rounds_data, 
                                    key=lambda x: x['centralized'].get('accuracy', 0))
                f.write("Best Round:\n")
                f.write(f"  Round: {best_acc_round['round']}\n")
                f.write(f"  Accuracy: {best_acc_round['centralized'].get('accuracy', 0):.4f}\n")
                f.write(f"  Loss: {best_acc_round['centralized'].get('loss', 0):.4f}\n")
                f.write(f"  F1-Score: {best_acc_round['centralized'].get('f1_macro', 0):.4f}\n")
        
        print(f"\n📄 Summary salvato in: {summary_path}")


def print_metrics_summary(round_num, centralized_metrics, distributed_metrics, 
                         convergence_metrics=None, efficiency_metrics=None):
    """
    Stampa un summary leggibile delle metriche
    """
    print(f"\n{'='*70}")
    print(f"ROUND {round_num} SUMMARY")
    print(f"{'='*70}")
    
    # Centralized
    print(f"\n📊 Centralized Evaluation (Validation Set - 12,000 samples):")
    print(f"  Accuracy:  {centralized_metrics.get('accuracy', 0):.4f}")
    print(f"  Loss:      {centralized_metrics.get('loss', 0):.4f}")
    print(f"  F1-Score:  {centralized_metrics.get('f1_macro', 0):.4f}")
    print(f"  Precision: {centralized_metrics.get('precision_macro', 0):.4f}")
    print(f"  Recall:    {centralized_metrics.get('recall_macro', 0):.4f}")
    
    # Distributed
    if distributed_metrics:
        print(f"\n🌐 Distributed Evaluation (Client Test Sets):")
        print(f"  Avg Accuracy:  {distributed_metrics.get('distributed_accuracy', 0):.4f}")
        print(f"  Accuracy Range: {distributed_metrics.get('eval_acc_min', 0):.4f} - "
              f"{distributed_metrics.get('eval_acc_max', 0):.4f}")
        print(f"  Fairness Gap:  {distributed_metrics.get('fairness_gap', 0):.4f} "
              f"({'✅ Good' if distributed_metrics.get('fairness_gap', 1) < 0.05 else '⚠️ Check'})")
    
    # Convergence
    if convergence_metrics:
        print(f"\n📈 Convergence:")
        print(f"  Best Round:    {convergence_metrics.get('best_round', 0)}")
        print(f"  No Improvement: {convergence_metrics.get('rounds_since_improvement', 0)} rounds")
        if convergence_metrics.get('current_plateau', False):
            print(f"  Status:        ⚠️ PLATEAU DETECTED")
    
    # Efficiency
    if efficiency_metrics:
        print(f"\n⏱️  Efficiency:")
        print(f"  Avg Round Time: {efficiency_metrics.get('avg_round_time', 0):.1f}s")
        print(f"  Total Time:     {efficiency_metrics.get('total_time', 0):.1f}s")
    
    print(f"{'='*70}\n")