"""
Per-Client Metrics Tracking System - HAR Version
=================================================

Sistema per tracciare metriche dettagliate per ogni client in UCI HAR:
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
    """
    
    def __init__(self):
        self.client_history = {}  # client_id -> list of round data
        self.round_data = {}      # round -> client data
        
    def log_client_metrics(self, round_num: int, client_id: str, metrics: Dict):
        """
        Salva metriche di un client per un round
        
        Args:
            round_num: Numero del round
            client_id: ID del client (può essere un indice)
            metrics: Dict con metriche del client
        """
        # Inizializza strutture se necessario
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
    
    def get_client_stats(self, client_id: str) -> Dict:
        """
        Statistiche aggregate per un client
        """
        if client_id not in self.client_history:
            return {}
        
        history = self.client_history[client_id]
        
        # Estrai serie temporali
        rounds = [h['round'] for h in history]
        accuracies = [h.get('train_acc', 0) for h in history]
        losses = [h.get('train_loss', 0) for h in history]
        times = [h.get('fit_duration', 0) for h in history]
        
        return {
            'client_id': client_id,
            'total_rounds': len(rounds),
            'rounds': rounds,
            
            # Accuracy stats
            'avg_accuracy': np.mean(accuracies),
            'final_accuracy': accuracies[-1] if accuracies else 0,
            'best_accuracy': np.max(accuracies) if accuracies else 0,
            'worst_accuracy': np.min(accuracies) if accuracies else 0,
            'accuracy_improvement': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0,
            
            # Loss stats
            'avg_loss': np.mean(losses),
            'final_loss': losses[-1] if losses else 0,
            'best_loss': np.min(losses) if losses else float('inf'),
            
            # Time stats
            'avg_fit_time': np.mean(times) if times else 0,
            'total_fit_time': np.sum(times) if times else 0,
            'min_fit_time': np.min(times) if times else 0,
            'max_fit_time': np.max(times) if times else 0,
            
            # Data
            'num_samples': history[-1].get('num_examples', 0),
        }
    
    def get_round_stats(self, round_num: int) -> Dict:
        """
        Statistiche per un round specifico (tutti i client)
        """
        if round_num not in self.round_data:
            return {}
        
        clients_data = self.round_data[round_num]
        
        accuracies = [d.get('train_acc', 0) for d in clients_data.values()]
        losses = [d.get('train_loss', 0) for d in clients_data.values()]
        times = [d.get('fit_duration', 0) for d in clients_data.values()]
        samples = [d.get('num_examples', 0) for d in clients_data.values()]
        
        return {
            'round': round_num,
            'num_clients': len(clients_data),
            
            # Accuracy distribution
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'median_accuracy': np.median(accuracies),
            
            # Loss distribution
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            
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
        """
        if round_num not in self.round_data:
            return
        
        stats = self.get_round_stats(round_num)
        contributions = self.compute_client_contribution(round_num)
        
        print(f"\n{'='*70}")
        print(f"👥 CLIENT METRICS - ROUND {round_num}")
        print(f"{'='*70}")
        
        print(f"\n📊 Performance Distribution ({stats['num_clients']} clients):")
        print(f"  Accuracy:  {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        print(f"    Range:   {stats['min_accuracy']:.4f} - {stats['max_accuracy']:.4f}")
        print(f"    Median:  {stats['median_accuracy']:.4f}")
        
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


# Global tracker instance
client_tracker = ClientMetricsTracker()


def enhanced_fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]], round_num: int) -> Metrics:
    """
    Versione enhanced di fit_metrics_aggregation che traccia anche metriche per-client
    
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
        
        client_tracker.log_client_metrics(round_num, client_id, client_metrics_extended)
    
    # Calcola metriche aggregate standard
    from har.metrics import fit_metrics_aggregation
    aggregated = fit_metrics_aggregation(metrics)
    
    # Aggiungi statistiche per-client al risultato
    round_stats = client_tracker.get_round_stats(round_num)
    
    aggregated.update({
        'client_mean_fit_time': round_stats.get('mean_fit_time', 0),
        'client_std_fit_time': round_stats.get('std_fit_time', 0),
        'client_data_imbalance': round_stats.get('data_imbalance', 0),
    })
    
    return aggregated
