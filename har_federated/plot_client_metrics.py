#!/usr/bin/env python3
"""
Client Metrics Visualization
=============================

Script per visualizzare e analizzare le metriche per-client del training federato.

Usage:
    python plot_client_metrics.py --experiment <nome>
    python plot_client_metrics.py --client-dir results/client_metrics_<nome>
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Stile grafici
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def load_client_data(client_dir):
    """
    Carica i dati delle metriche client
    
    🆕 Supporta il nuovo formato con partition_id tracking per device robusto alle riconnessioni
    """
    client_dir = Path(client_dir)
    
    if not client_dir.exists():
        raise FileNotFoundError(f"Directory non trovata: {client_dir}")
    
    data = {}
    
    # 🆕 Prova a caricare dal nuovo formato JSON (con partition_id)
    full_json_path = client_dir / "client_metrics_full.json"
    if full_json_path.exists():
        import json
        with open(full_json_path, 'r') as f:
            json_data = json.load(f)
        
        # Raggruppa per partition_id invece di client_id
        from collections import defaultdict
        
        partition_histories = defaultdict(list)
        
        for client_id, history in json_data.get('client_history', {}).items():
            for entry in history:
                pid = entry.get('partition_id', -1)
                partition_histories[pid].append(entry)
        
        # Converti in DataFrame per ogni partition
        client_histories = {}
        device_names = {
            0: "Device 0 (Pi 5)",
            1: "Device 1 (Pi 4)",
            2: "Device 2 (Pi 3)",
            -1: "Unknown Device"
        }
        
        for pid, entries in partition_histories.items():
            df = pd.DataFrame(entries)
            device_label = device_names.get(pid, f"Device {pid}")
            client_histories[device_label] = df
        
        data['histories'] = client_histories
        
        # Genera summary e round_stats dal JSON
        # Summary: statistiche aggregate per partition
        summary_data = []
        for pid, entries in partition_histories.items():
            df = pd.DataFrame(entries)
            device_label = device_names.get(pid, f"Device {pid}")
            hostname = entries[0].get('hostname', 'unknown') if entries else 'unknown'
            
            summary_row = {
                'client_id': device_label,
                'hostname': hostname,
                'partition_id': pid,
                'total_rounds': len(entries),
                'final_accuracy': df['train_acc'].iloc[-1] if len(df) > 0 else 0,
                'best_accuracy': df['train_acc'].max() if len(df) > 0 else 0,
                'accuracy_improvement': df['train_acc'].iloc[-1] - df['train_acc'].iloc[0] if len(df) > 1 else 0,
                'avg_fit_time': df['fit_duration'].mean() if 'fit_duration' in df.columns else 0,
                'total_fit_time': df['fit_duration'].sum() if 'fit_duration' in df.columns else 0,
                'num_samples': df['num_examples'].iloc[0] if len(df) > 0 and 'num_examples' in df.columns else 0,
            }
            summary_data.append(summary_row)
        
        data['summary'] = pd.DataFrame(summary_data)
        
        # Round stats: statistiche per round
        round_stats_data = []
        max_round = max(entry['round'] for entries in partition_histories.values() for entry in entries) if partition_histories else 0
        
        for round_num in range(1, max_round + 1):
            # Raccogli metriche di tutti i partition per questo round
            round_entries = []
            for entries in partition_histories.values():
                round_entry = [e for e in entries if e['round'] == round_num]
                if round_entry:
                    round_entries.append(round_entry[0])
            
            if round_entries:
                accs = [e['train_acc'] for e in round_entries]
                times = [e['fit_duration'] for e in round_entries]
                samples = [e['num_examples'] for e in round_entries]
                
                # Data imbalance: coefficient of variation
                cv = np.std(samples) / np.mean(samples) if np.mean(samples) > 0 else 0
                
                round_stats_data.append({
                    'round': round_num,
                    'num_clients': len(round_entries),
                    'mean_accuracy': np.mean(accs),
                    'std_accuracy': np.std(accs),
                    'min_accuracy': np.min(accs),
                    'max_accuracy': np.max(accs),
                    'mean_fit_time': np.mean(times),
                    'std_fit_time': np.std(times),
                    'data_imbalance': cv,
                })
        
        data['round_stats'] = pd.DataFrame(round_stats_data)
        
        print(f"✅ Loaded data using partition_id tracking (robust to client reconnections)")
        return data
    
    # Fallback: vecchio formato CSV
    # Carica summary
    summary_path = client_dir / "client_summary.csv"
    if summary_path.exists():
        data['summary'] = pd.read_csv(summary_path)
    
    # Carica round stats
    round_stats_path = client_dir / "round_client_stats.csv"
    if round_stats_path.exists():
        data['round_stats'] = pd.read_csv(round_stats_path)
    
    # Carica history per ogni client
    client_histories = {}
    for hist_file in client_dir.glob("client_*_history.csv"):
        client_id = hist_file.stem.replace("_history", "")
        client_histories[client_id] = pd.read_csv(hist_file)
    
    data['histories'] = client_histories
    
    print(f"⚠️  Loaded data using old CSV format (client_id based)")
    return data


def plot_client_accuracy_evolution(data, save_path=None):
    """
    Grafico: Evolution of accuracy per ogni client
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    histories = data.get('histories', {})
    
    # Plotta una linea per ogni client
    for client_id, df in histories.items():
        ax.plot(df['round'], df['train_acc'], marker='o', label=client_id, linewidth=2)
    
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Client Training Accuracy Evolution', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Salvato: {save_path}")
    
    plt.show()


def plot_client_training_time(data, save_path=None):
    """
    Grafico: Tempo di training per client
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    histories = data.get('histories', {})
    
    # 1. Time evolution per client
    for client_id, df in histories.items():
        if 'fit_duration' in df.columns:
            ax1.plot(df['round'], df['fit_duration'], marker='o', label=client_id, linewidth=2)
    
    ax1.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Time Evolution per Client', fontsize=13, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Average time comparison (box plot)
    summary = data.get('summary')
    if summary is not None and 'avg_fit_time' in summary.columns:
        clients = summary['client_id'].tolist()
        times = summary['avg_fit_time'].tolist()
        
        ax2.barh(clients, times, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Average Training Time (seconds)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Client', fontsize=12, fontweight='bold')
        ax2.set_title('Average Training Time per Client', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Aggiungi valori sulle barre
        for i, (client, time_val) in enumerate(zip(clients, times)):
            ax2.text(time_val + 0.1, i, f'{time_val:.2f}s', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Salvato: {save_path}")
    
    plt.show()


def plot_client_contribution(data, save_path=None):
    """
    Grafico: Contributo dei client (basato su accuracy e peso)
    """
    summary = data.get('summary')
    if summary is None:
        print("⚠️  Dati summary non disponibili")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Final Accuracy
    ax = axes[0, 0]
    clients = summary['client_id'].tolist()
    # Supporta sia 'final_train_accuracy' (MNIST) che 'final_accuracy' (HAR/WISDM)
    acc_col = 'final_train_accuracy' if 'final_train_accuracy' in summary.columns else 'final_accuracy'
    accuracies = summary[acc_col].tolist()
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(clients)))
    
    bars = ax.barh(clients, accuracies, color=colors, edgecolor='black')
    ax.set_xlabel('Final Training Accuracy', fontsize=11, fontweight='bold')
    ax.set_ylabel('Client', fontsize=11, fontweight='bold')
    ax.set_title('Final Accuracy per Client', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. Accuracy Improvement
    ax = axes[0, 1]
    improve_col = 'train_accuracy_improvement' if 'train_accuracy_improvement' in summary.columns else 'accuracy_improvement'
    if improve_col in summary.columns:
        improvements = summary[improve_col].tolist()
        colors_imp = ['green' if x > 0 else 'red' for x in improvements]
        
        ax.barh(clients, improvements, color=colors_imp, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Accuracy Improvement (Final - Initial)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Client', fontsize=11, fontweight='bold')
        ax.set_title('Learning Progress per Client', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Number of samples (Data contribution)
    ax = axes[1, 0]
    if 'num_samples' in summary.columns:
        samples = summary['num_samples'].tolist()
        total_samples = sum(samples)
        percentages = [s/total_samples*100 for s in samples]
        
        wedges, texts, autotexts = ax.pie(samples, labels=clients, autopct='%1.1f%%',
                                           startangle=90, colors=colors)
        ax.set_title('Data Distribution per Client', fontsize=12, fontweight='bold')
        
        # Style
        for text in texts:
            text.set_fontsize(10)
            text.set_fontweight('bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
    
    # 4. Total training time
    ax = axes[1, 1]
    if 'total_fit_time' in summary.columns:
        total_times = summary['total_fit_time'].tolist()
        
        ax.barh(clients, total_times, color='coral', edgecolor='black')
        ax.set_xlabel('Total Training Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Client', fontsize=11, fontweight='bold')
        ax.set_title('Total Training Time per Client', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Salvato: {save_path}")
    
    plt.show()


def plot_round_statistics(data, save_path=None):
    """
    Grafico: Statistiche aggregate per round
    """
    round_stats = data.get('round_stats')
    if round_stats is None:
        print("⚠️  Dati round_stats non disponibili")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    rounds = round_stats['round'].tolist()
    
    # 1. Accuracy distribution evolution
    ax = axes[0, 0]
    # Supporta sia 'mean_train_accuracy' (MNIST) che 'mean_accuracy' (HAR/WISDM)
    mean_acc_col = 'mean_train_accuracy' if 'mean_train_accuracy' in round_stats.columns else 'mean_accuracy'
    std_acc_col = 'std_train_accuracy' if 'std_train_accuracy' in round_stats.columns else 'std_accuracy'
    min_acc_col = 'min_train_accuracy' if 'min_train_accuracy' in round_stats.columns else 'min_accuracy'
    max_acc_col = 'max_train_accuracy' if 'max_train_accuracy' in round_stats.columns else 'max_accuracy'
    
    ax.plot(rounds, round_stats[mean_acc_col], marker='o', linewidth=2.5, 
            color='#2E86AB', label='Mean')
    ax.fill_between(rounds, 
                     round_stats[mean_acc_col] - round_stats[std_acc_col],
                     round_stats[mean_acc_col] + round_stats[std_acc_col],
                     alpha=0.3, color='#2E86AB', label='±1 Std Dev')
    ax.plot(rounds, round_stats[min_acc_col], linestyle='--', color='red', 
            linewidth=1.5, label='Min (worst client)')
    ax.plot(rounds, round_stats[max_acc_col], linestyle='--', color='green', 
            linewidth=1.5, label='Max (best client)')
    
    ax.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax.set_ylabel('Training Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Client Accuracy Distribution Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Training time distribution
    ax = axes[0, 1]
    ax.plot(rounds, round_stats['mean_fit_time'], marker='o', linewidth=2.5, 
            color='#E63946', label='Mean')
    ax.fill_between(rounds,
                     round_stats['mean_fit_time'] - round_stats['std_fit_time'],
                     round_stats['mean_fit_time'] + round_stats['std_fit_time'],
                     alpha=0.3, color='#E63946', label='±1 Std Dev')
    
    ax.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Client Training Time Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. Data imbalance evolution
    ax = axes[1, 0]
    if 'data_imbalance' in round_stats.columns:
        ax.plot(rounds, round_stats['data_imbalance'], marker='o', linewidth=2.5, 
                color='#9D4EDD')
        ax.axhline(y=0.3, color='orange', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Warning threshold')
        ax.fill_between(rounds, 0, 0.3, alpha=0.2, color='green', label='Good balance')
        
        ax.set_xlabel('Round', fontsize=11, fontweight='bold')
        ax.set_ylabel('Data Imbalance (CV)', fontsize=11, fontweight='bold')
        ax.set_title('Data Distribution Balance', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 4. Client participation
    ax = axes[1, 1]
    ax.plot(rounds, round_stats['num_clients'], marker='o', linewidth=2.5, 
            color='#06A77D')
    ax.set_xlabel('Round', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Active Clients', fontsize=11, fontweight='bold')
    ax.set_title('Client Participation per Round', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Integer ticks
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Salvato: {save_path}")
    
    plt.show()


def print_client_summary(data):
    """
    Stampa un summary testuale delle metriche client
    """
    summary = data.get('summary')
    if summary is None:
        print("⚠️  Dati summary non disponibili")
        return
    
    print("\n" + "="*70)
    print("📊 CLIENT PERFORMANCE SUMMARY")
    print("="*70)
    
    # Supporta sia 'final_train_accuracy' (MNIST) che 'final_accuracy' (HAR/WISDM)
    acc_col = 'final_train_accuracy' if 'final_train_accuracy' in summary.columns else 'final_accuracy'
    best_col = 'best_train_accuracy' if 'best_train_accuracy' in summary.columns else 'best_accuracy'
    improve_col = 'train_accuracy_improvement' if 'train_accuracy_improvement' in summary.columns else 'accuracy_improvement'
    
    for _, row in summary.iterrows():
        print(f"\n🔹 {row['client_id']}")
        print(f"  Rounds participated: {row['total_rounds']}")
        print(f"  Final accuracy:      {row[acc_col]:.4f}")
        print(f"  Best accuracy:       {row[best_col]:.4f}")
        print(f"  Improvement:         +{row[improve_col]:.4f}")
        print(f"  Avg training time:   {row['avg_fit_time']:.2f}s")
        print(f"  Total time:          {row['total_fit_time']:.2f}s")
        print(f"  Data samples:        {row['num_samples']}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualizza metriche per-client')
    parser.add_argument('--experiment', '-e', type=str, default=None,
                       help='Nome esperimento')
    parser.add_argument('--client-dir', '-d', type=str, default=None,
                       help='Directory con metriche client')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Salva grafici come PNG')
    
    args = parser.parse_args()
    
    # Determina directory
    if args.client_dir:
        client_dir = args.client_dir
    elif args.experiment:
        client_dir = f"results/client_metrics_{args.experiment}"
    else:
        # Trova l'ultima directory client_metrics_*
        results_dir = Path("results")
        client_dirs = sorted(results_dir.glob("client_metrics_*"))
        if not client_dirs:
            print("❌ Nessuna directory client_metrics trovata!")
            print("💡 Esegui prima un training con il nuovo sistema")
            return
        client_dir = client_dirs[-1]
        print(f"📂 Usando ultima directory: {client_dir}")
    
    # Carica dati
    try:
        data = load_client_data(client_dir)
        print(f"✅ Dati caricati da: {client_dir}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Stampa summary
    print_client_summary(data)
    
    # Determina save paths
    if args.save:
        plots_dir = Path(client_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        base_name = Path(client_dir).name
        
        save_acc = plots_dir / f"{base_name}_accuracy_evolution.png"
        save_time = plots_dir / f"{base_name}_training_time.png"
        save_contrib = plots_dir / f"{base_name}_contribution.png"
        save_rounds = plots_dir / f"{base_name}_round_stats.png"
    else:
        save_acc = save_time = save_contrib = save_rounds = None
    
    # Genera grafici
    print("\n📈 Generando grafici...")
    
    print("  [1/4] Accuracy evolution...")
    plot_client_accuracy_evolution(data, save_acc)
    
    print("  [2/4] Training time...")
    plot_client_training_time(data, save_time)
    
    print("  [3/4] Client contribution...")
    plot_client_contribution(data, save_contrib)
    
    print("  [4/4] Round statistics...")
    plot_round_statistics(data, save_rounds)
    
    print("\n✅ Analisi completata!")
    
    if args.save:
        print(f"\n💾 Grafici salvati in: {plots_dir}/")


if __name__ == "__main__":
    main()
