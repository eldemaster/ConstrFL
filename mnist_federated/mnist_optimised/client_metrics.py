"""
Per-Client Metrics Tracking System
===================================

Sistema per tracciare metriche dettagliate per ogni client:
- Performance individuale (accuracy, loss)
- Tempo di training
- Contributo al modello globale
- Data distribution
- Evoluzione nel tempo
"""

import numpy as np
from typing import List, Tuple, Dict
from flwr.common import Metrics
import time
from pathlib import Path
import json
import pandas as pd


class ClientMetricsTracker:
    """
    Traccia metriche dettagliate per ogni client
    
    🆕 ENHANCED: Traccia anche validation metrics locale per client-side testing
    """
    
    def __init__(self):
        self.client_history = {}  # client_id -> list of round data
        self.round_data = {}      # round -> client data
        self.validation_history = {}  # 🆕 client_id -> list of validation metrics
        
    def log_client_metrics(self, round_num: int, client_id: str, metrics: Dict, 
                          is_validation: bool = False):
        """
        Salva metriche di un client per un round
        
        🆕 ENHANCED: Supporta sia training che validation metrics
        
        Args:
            round_num: Numero del round
            client_id: ID del client (può essere un indice)
            metrics: Dict con metriche del client
            is_validation: Se True, salva in validation_history invece che client_history
        """
        # Inizializza strutture se necessario
        if not is_validation:
            if client_id not in self.client_history:
                self.client_history[client_id] = []
            
            if round_num not in self.round_data:
                self.round_data[round_num] = {}
            
            # Aggiungi timestamp
            metrics['round'] = round_num
            metrics['timestamp'] = time.time()
            
            # Salva
            self.client_history[client_id].append(metrics)
            self.round_data[round_num][client_id] = metrics
        else:
            # 🆕 Validation metrics
            if client_id not in self.validation_history:
                self.validation_history[client_id] = []
            
            # Aggiungi timestamp e round
            metrics['round'] = round_num
            metrics['timestamp'] = time.time()
            
            # Salva
            self.validation_history[client_id].append(metrics)
    
    def get_client_stats(self, client_id: str, include_validation: bool = True) -> Dict:
        """
        Statistiche aggregate per un client
        
        🆕 ENHANCED: Include anche validation metrics se disponibili
        """
        if client_id not in self.client_history:
            return {}
        
        history = self.client_history[client_id]
        
        # Estrai serie temporali TRAINING
        rounds = [h['round'] for h in history]
        accuracies = [h.get('train_acc', 0) for h in history]
        losses = [h.get('train_loss', 0) for h in history]
        times = [h.get('fit_duration', 0) for h in history]
        
        stats = {
            'client_id': client_id,
            'total_rounds': len(rounds),
            'rounds': rounds,
            
            # Training Accuracy stats
            'avg_train_accuracy': np.mean(accuracies),
            'final_train_accuracy': accuracies[-1] if accuracies else 0,
            'best_train_accuracy': np.max(accuracies) if accuracies else 0,
            'worst_train_accuracy': np.min(accuracies) if accuracies else 0,
            'train_accuracy_improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0,
            
            # Training Loss stats
            'avg_train_loss': np.mean(losses),
            'final_train_loss': losses[-1] if losses else 0,
            'best_train_loss': np.min(losses) if losses else float('inf'),
            
            # Time stats
            'avg_fit_time': np.mean(times) if times else 0,
            'total_fit_time': np.sum(times) if times else 0,
            'min_fit_time': np.min(times) if times else 0,
            'max_fit_time': np.max(times) if times else 0,
            
            # Data
            'num_samples': history[-1].get('num_examples', 0),
        }
        
        # 🆕 VALIDATION METRICS
        if include_validation and client_id in self.validation_history:
            val_history = self.validation_history[client_id]
            
            val_accuracies = [h.get('val_accuracy', 0) for h in val_history]
            val_losses = [h.get('val_loss', 0) for h in val_history]
            val_precisions = [h.get('val_precision', 0) for h in val_history]
            val_recalls = [h.get('val_recall', 0) for h in val_history]
            val_f1s = [h.get('val_f1', 0) for h in val_history]
            
            stats.update({
                # Validation Accuracy
                'avg_val_accuracy': np.mean(val_accuracies) if val_accuracies else 0,
                'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0,
                'best_val_accuracy': np.max(val_accuracies) if val_accuracies else 0,
                'worst_val_accuracy': np.min(val_accuracies) if val_accuracies else 0,
                'val_accuracy_improvement': val_accuracies[-1] - val_accuracies[0] if len(val_accuracies) > 1 else 0,
                
                # Validation Loss
                'avg_val_loss': np.mean(val_losses) if val_losses else 0,
                'final_val_loss': val_losses[-1] if val_losses else 0,
                'best_val_loss': np.min(val_losses) if val_losses else float('inf'),
                
                # Validation detailed metrics
                'avg_val_precision': np.mean(val_precisions) if val_precisions else 0,
                'final_val_precision': val_precisions[-1] if val_precisions else 0,
                
                'avg_val_recall': np.mean(val_recalls) if val_recalls else 0,
                'final_val_recall': val_recalls[-1] if val_recalls else 0,
                
                'avg_val_f1': np.mean(val_f1s) if val_f1s else 0,
                'final_val_f1': val_f1s[-1] if val_f1s else 0,
                
                # Train-Val gap (overfitting indicator)
                'train_val_gap': (accuracies[-1] - val_accuracies[-1]) if (accuracies and val_accuracies) else 0,
            })
        
        return stats
    
    def get_round_stats(self, round_num: int, include_validation: bool = True) -> Dict:
        """
        Statistiche per un round specifico (tutti i client)
        
        🆕 ENHANCED: Include validation metrics e fairness
        """
        if round_num not in self.round_data:
            return {}
        
        clients_data = self.round_data[round_num]
        
        # Training metrics
        accuracies = [d.get('train_acc', 0) for d in clients_data.values()]
        losses = [d.get('train_loss', 0) for d in clients_data.values()]
        times = [d.get('fit_duration', 0) for d in clients_data.values()]
        samples = [d.get('num_examples', 0) for d in clients_data.values()]
        
        stats = {
            'round': round_num,
            'num_clients': len(clients_data),
            
            # Training Accuracy distribution
            'mean_train_accuracy': np.mean(accuracies),
            'std_train_accuracy': np.std(accuracies),
            'min_train_accuracy': np.min(accuracies),
            'max_train_accuracy': np.max(accuracies),
            'median_train_accuracy': np.median(accuracies),
            
            # Training Loss distribution
            'mean_train_loss': np.mean(losses),
            'std_train_loss': np.std(losses),
            
            # Time distribution
            'mean_fit_time': np.mean(times),
            'std_fit_time': np.std(times),
            'min_fit_time': np.min(times),
            'max_fit_time': np.max(times),
            'total_fit_time': np.sum(times),
            
            # Data distribution
            'total_samples': sum(samples),
            'samples_per_client': samples,
            'data_imbalance': np.std(samples) / np.mean(samples) if samples else 0,  # CV
        }
        
        # 🆕 VALIDATION METRICS (se disponibili)
        if include_validation:
            val_accuracies = []
            val_losses = []
            val_f1s = []
            
            for client_id in clients_data.keys():
                if client_id in self.validation_history:
                    # Trova validation metrics per questo round
                    client_val = [v for v in self.validation_history[client_id] if v['round'] == round_num]
                    if client_val:
                        val_accuracies.append(client_val[0].get('val_accuracy', 0))
                        val_losses.append(client_val[0].get('val_loss', 0))
                        val_f1s.append(client_val[0].get('val_f1', 0))
            
            if val_accuracies:
                stats.update({
                    # Validation Accuracy distribution
                    'mean_val_accuracy': np.mean(val_accuracies),
                    'std_val_accuracy': np.std(val_accuracies),
                    'min_val_accuracy': np.min(val_accuracies),
                    'max_val_accuracy': np.max(val_accuracies),
                    'median_val_accuracy': np.median(val_accuracies),
                    
                    # Validation Loss
                    'mean_val_loss': np.mean(val_losses) if val_losses else 0,
                    
                    # F1 Score
                    'mean_val_f1': np.mean(val_f1s) if val_f1s else 0,
                    
                    # 🆕 FAIRNESS METRICS
                    'val_fairness_gap': np.max(val_accuracies) - np.min(val_accuracies),  # Max - Min
                    'val_fairness_std': np.std(val_accuracies),  # Standard deviation
                    'val_fairness_cv': np.std(val_accuracies) / np.mean(val_accuracies) if np.mean(val_accuracies) > 0 else 0,  # Coefficient of variation
                })
        
        return stats
    
    def compute_client_contribution(self, round_num: int) -> Dict:
        """
        Calcola il "contributo" di ogni client al training globale
        
        Il contributo è misurato come:
        1. Peso nei parametri aggregati (basato su num_examples)
        2. Quality score (accuracy relativa)
        3. Consistency (quanto è stabile)
        """
        if round_num not in self.round_data:
            return {}
        
        clients_data = self.round_data[round_num]
        
        # Pesi basati su numero di esempi
        total_samples = sum(d.get('num_examples', 0) for d in clients_data.values())
        
        contributions = {}
        for client_id, data in clients_data.items():
            num_examples = data.get('num_examples', 0)
            train_acc = data.get('train_acc', 0)
            train_loss = data.get('train_loss', float('inf'))
            
            # Peso nell'aggregazione (FedAvg standard)
            aggregation_weight = num_examples / total_samples if total_samples > 0 else 0
            
            # Quality score (accuracy normalizzata)
            max_acc = max(d.get('train_acc', 0) for d in clients_data.values())
            quality_score = train_acc / max_acc if max_acc > 0 else 0
            
            # Contribution score (combinazione pesata)
            contribution_score = aggregation_weight * quality_score
            
            contributions[client_id] = {
                'aggregation_weight': aggregation_weight,
                'quality_score': quality_score,
                'contribution_score': contribution_score,
                'num_examples': num_examples,
                'train_acc': train_acc,
                'train_loss': train_loss,
            }
        
        return contributions
    
    def get_client_ranking(self, round_num: int, metric: str = 'contribution_score') -> List[Tuple[str, float]]:
        """
        Classifica i client per una metrica specifica
        
        Args:
            round_num: Round number
            metric: 'contribution_score', 'quality_score', 'train_acc', etc.
        
        Returns:
            List of (client_id, score) ordinata per score decrescente
        """
        contributions = self.compute_client_contribution(round_num)
        
        if metric in ['contribution_score', 'quality_score', 'aggregation_weight']:
            rankings = [(cid, data[metric]) for cid, data in contributions.items()]
        elif metric in ['train_acc', 'num_examples']:
            rankings = [(cid, data[metric]) for cid, data in contributions.items()]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def export_client_data(self, output_dir: str = "results/client_metrics"):
        """
        Esporta tutti i dati dei client in CSV/JSON
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Esporta statistiche per client
        client_stats = []
        for client_id in self.client_history.keys():
            stats = self.get_client_stats(client_id)
            client_stats.append(stats)
        
        if client_stats:
            df = pd.DataFrame(client_stats)
            csv_path = output_path / "client_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"💾 Client summary salvato: {csv_path}")
        
        # 2. Esporta time series per ogni client
        for client_id, history in self.client_history.items():
            df = pd.DataFrame(history)
            csv_path = output_path / f"client_{client_id}_history.csv"
            df.to_csv(csv_path, index=False)
        
        # 3. Esporta statistiche per round
        round_stats = []
        for round_num in sorted(self.round_data.keys()):
            stats = self.get_round_stats(round_num)
            round_stats.append(stats)
        
        if round_stats:
            # Rimuovi liste annidate per CSV
            for stat in round_stats:
                stat.pop('samples_per_client', None)
            
            df = pd.DataFrame(round_stats)
            csv_path = output_path / "round_client_stats.csv"
            df.to_csv(csv_path, index=False)
            print(f"💾 Round stats salvato: {csv_path}")
        
        # 4. Esporta tutto in JSON per analisi approfondite
        json_data = {
            'client_history': self.client_history,
            'round_data': self.round_data,
        }
        
        json_path = output_path / "client_metrics_full.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"💾 Dati completi salvati: {json_path}")
        
        return output_path
    
    def print_round_summary(self, round_num: int):
        """
        Stampa un summary delle metriche client per un round
        
        🆕 ENHANCED: Include validation metrics e fairness
        """
        if round_num not in self.round_data:
            return
        
        stats = self.get_round_stats(round_num, include_validation=True)
        contributions = self.compute_client_contribution(round_num)
        
        print(f"\n{'='*70}")
        print(f"👥 CLIENT METRICS - ROUND {round_num}")
        print(f"{'='*70}")
        
        print(f"\n📊 Training Performance ({stats['num_clients']} clients):")
        print(f"  Accuracy:  {stats['mean_train_accuracy']:.4f} ± {stats['std_train_accuracy']:.4f}")
        print(f"    Range:   {stats['min_train_accuracy']:.4f} - {stats['max_train_accuracy']:.4f}")
        print(f"    Median:  {stats['median_train_accuracy']:.4f}")
        
        # 🆕 VALIDATION METRICS (se disponibili)
        if 'mean_val_accuracy' in stats:
            print(f"\n🎯 Validation Performance:")
            print(f"  Accuracy:  {stats['mean_val_accuracy']:.4f} ± {stats['std_val_accuracy']:.4f}")
            print(f"    Range:   {stats['min_val_accuracy']:.4f} - {stats['max_val_accuracy']:.4f}")
            print(f"    Median:  {stats['median_val_accuracy']:.4f}")
            print(f"  F1 Score:  {stats['mean_val_f1']:.4f}")
            
            # 🆕 FAIRNESS METRICS
            fairness_status = "✅ Fair" if stats['val_fairness_gap'] < 0.05 else "⚠️ Unfair"
            print(f"\n⚖️  Fairness Analysis:")
            print(f"  Gap (Max-Min):    {stats['val_fairness_gap']:.4f} ({fairness_status})")
            print(f"  Std Deviation:    {stats['val_fairness_std']:.4f}")
            print(f"  Coefficient Var:  {stats['val_fairness_cv']:.3f}")
        
        print(f"\n⏱️  Training Time:")
        print(f"  Mean:      {stats['mean_fit_time']:.2f}s ± {stats['std_fit_time']:.2f}s")
        print(f"  Range:     {stats['min_fit_time']:.2f}s - {stats['max_fit_time']:.2f}s")
        print(f"  Total:     {stats['total_fit_time']:.2f}s")
        
        print(f"\n📂 Data Distribution:")
        print(f"  Total samples:    {stats['total_samples']}")
        print(f"  Imbalance (CV):   {stats['data_imbalance']:.3f}")
        
        # Top 3 contributors
        print(f"\n🏆 Top 3 Contributors:")
        rankings = self.get_client_ranking(round_num, 'contribution_score')
        for i, (client_id, score) in enumerate(rankings[:3], 1):
            contrib = contributions[client_id]
            print(f"  {i}. Client {client_id}:")
            print(f"     Contribution: {score:.4f} | Accuracy: {contrib['train_acc']:.4f} | "
                  f"Weight: {contrib['aggregation_weight']:.2%}")
        
        print(f"{'='*70}\n")
    
    def compute_improvement_trajectory(self, client_id: str) -> Dict:
        """
        🆕 Calcola la traiettoria di miglioramento di un client
        
        Analizza come cambia la performance nel tempo per identificare:
        - Convergenza veloce/lenta
        - Plateaus
        - Instabilità (oscillazioni)
        
        Returns:
            Dict con metriche di improvement trajectory
        """
        if client_id not in self.validation_history:
            return {}
        
        val_history = self.validation_history[client_id]
        if len(val_history) < 2:
            return {}
        
        # Estrai accuracies nel tempo
        accuracies = [h.get('val_accuracy', 0) for h in val_history]
        rounds = [h.get('round', 0) for h in val_history]
        
        # Calcola miglioramento round-by-round
        improvements = [accuracies[i] - accuracies[i-1] for i in range(1, len(accuracies))]
        
        # Metriche di trajectory
        trajectory = {
            'client_id': client_id,
            'total_improvement': accuracies[-1] - accuracies[0],
            'avg_improvement_per_round': np.mean(improvements) if improvements else 0,
            'std_improvement': np.std(improvements) if improvements else 0,
            
            # Convergenza
            'is_converged': abs(improvements[-1]) < 0.001 if improvements else False,
            'convergence_speed': (accuracies[-1] - accuracies[0]) / len(accuracies),
            
            # Stabilità
            'is_stable': np.std(improvements) < 0.01 if improvements else False,
            'oscillation_count': sum(1 for i in range(1, len(improvements)) if improvements[i] * improvements[i-1] < 0),
            
            # Plateaus (miglioramento < 0.001 per 2+ round consecutivi)
            'plateau_detected': any(abs(improvements[i]) < 0.001 and abs(improvements[i+1]) < 0.001 
                                   for i in range(len(improvements)-1)),
        }
        
        return trajectory


# Global tracker instance
client_tracker = ClientMetricsTracker()


def enhanced_fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]], round_num: int) -> Metrics:
    """
    Versione enhanced di fit_metrics_aggregation che traccia anche metriche per-client
    
    🆕 ENHANCED: Traccia sia training che validation metrics (se disponibili)
    
    Args:
        metrics: Lista di (num_examples, metrics_dict) dai client
        round_num: Numero del round corrente
    """
    # Track per-client metrics
    for idx, (num_examples, client_metrics) in enumerate(metrics):
        client_id = f"client_{idx}"  # Potremmo usare un ID vero se disponibile
        
        # Aggiungi num_examples alle metriche
        client_metrics_extended = {
            **client_metrics,
            'num_examples': num_examples,
        }
        
        # Salva training metrics
        client_tracker.log_client_metrics(round_num, client_id, client_metrics_extended, is_validation=False)
    
    # Calcola metriche aggregate standard
    from mnist_optimised.metrics import fit_metrics_aggregation
    aggregated = fit_metrics_aggregation(metrics)
    
    # Aggiungi statistiche per-client al risultato
    round_stats = client_tracker.get_round_stats(round_num, include_validation=True)
    
    aggregated.update({
        'client_mean_fit_time': round_stats.get('mean_fit_time', 0),
        'client_std_fit_time': round_stats.get('std_fit_time', 0),
        'client_data_imbalance': round_stats.get('data_imbalance', 0),
    })
    
    # 🆕 Aggiungi validation metrics se disponibili
    if 'mean_val_accuracy' in round_stats:
        aggregated.update({
            'client_mean_val_accuracy': round_stats['mean_val_accuracy'],
            'client_val_fairness_gap': round_stats.get('val_fairness_gap', 0),
            'client_val_fairness_std': round_stats.get('val_fairness_std', 0),
        })
    
    return aggregated


def enhanced_evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]], round_num: int) -> Metrics:
    """
    🆕 Aggregazione per VALIDATION metrics dai client
    
    Questa funzione viene chiamata quando i client eseguono evaluate() locale.
    Traccia le validation metrics e calcola fairness.
    
    Args:
        metrics: Lista di (num_examples, metrics_dict) dai client (evaluation results)
        round_num: Numero del round corrente
    
    Returns:
        Aggregated validation metrics
    """
    if not metrics:
        return {}
    
    # Track per-client validation metrics
    for idx, (num_examples, client_metrics) in enumerate(metrics):
        client_id = f"client_{idx}"
        
        # Salva VALIDATION metrics
        client_tracker.log_client_metrics(round_num, client_id, client_metrics, is_validation=True)
    
    # Aggregate validation metrics
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    # Weighted average per metriche principali
    val_accuracy_weighted = sum(
        num_examples * m.get('val_accuracy', 0) 
        for num_examples, m in metrics
    ) / total_examples if total_examples > 0 else 0
    
    val_loss_weighted = sum(
        num_examples * m.get('val_loss', 0) 
        for num_examples, m in metrics
    ) / total_examples if total_examples > 0 else 0
    
    # Macro average per precision/recall/F1 (tutte le metriche equally important)
    val_precision_macro = np.mean([m.get('val_precision', 0) for _, m in metrics])
    val_recall_macro = np.mean([m.get('val_recall', 0) for _, m in metrics])
    val_f1_macro = np.mean([m.get('val_f1', 0) for _, m in metrics])
    
    # Fairness metrics
    val_accuracies = [m.get('val_accuracy', 0) for _, m in metrics]
    fairness_gap = max(val_accuracies) - min(val_accuracies)
    fairness_std = np.std(val_accuracies)
    
    aggregated = {
        'distributed_val_accuracy': float(val_accuracy_weighted),
        'distributed_val_loss': float(val_loss_weighted),
        'distributed_val_precision': float(val_precision_macro),
        'distributed_val_recall': float(val_recall_macro),
        'distributed_val_f1': float(val_f1_macro),
        'val_fairness_gap': float(fairness_gap),
        'val_fairness_std': float(fairness_std),
        'num_clients_evaluated': len(metrics),
    }
    
    return aggregated
