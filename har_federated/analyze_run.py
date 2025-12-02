#!/usr/bin/env python3
"""
🎯 UNIFIED RUN ANALYSIS SCRIPT

Analizza completamente una run di federated learning.
Esegue TUTTE le analisi e genera TUTTI i grafici in una sola chiamata.

Usage:
    python analyze_run.py results/run_20251025_123456/
    python analyze_run.py results/run_20251025_123456/ --no-show  # solo salva, non mostra

Analisi eseguite:
    1. 📊 Server-side metrics (8 grafici - analyze_results.py)
    2. 👥 Client-side metrics (4 grafici - plot_client_metrics.py)
    3. 📡 Bandwidth analysis (opzionale - plot_bandwidth_comparison.py)
    4. ✅ Final test evaluation (1 grafico + JSON - final_test_evaluation.py)

Tutti i grafici vengono salvati in: <run_dir>/plots/
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd
import subprocess
import os


# ============================================================================
# COLORS & FORMATTING
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print colorful header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.ENDC}\n")


def print_step(step_num, total_steps, description):
    """Print analysis step"""
    print(f"{Colors.BOLD}{Colors.BLUE}[{step_num}/{total_steps}]{Colors.ENDC} {description}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")


def print_info(message):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ️  {message}{Colors.ENDC}")


# ============================================================================
# VALIDATION
# ============================================================================

def validate_run_directory(run_dir):
    """
    Valida che la directory contenga una run completa
    
    Expected structure:
        run_TIMESTAMP/
        ├── data/
        │   ├── metrics.json
        │   ├── metrics.csv
        │   └── summary.txt
        ├── models/
        │   └── final_model.weights.h5
        ├── plots/
        └── logs/
            └── client_metrics/
    """
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print_error(f"Directory non trovata: {run_dir}")
        return False
    
    if not run_path.is_dir():
        print_error(f"Non è una directory: {run_dir}")
        return False
    
    # Check required subdirectories
    required_dirs = ['data', 'models', 'plots', 'logs']
    missing_dirs = []
    
    for d in required_dirs:
        if not (run_path / d).exists():
            missing_dirs.append(d)
    
    if missing_dirs:
        print_warning(f"Directory mancanti: {', '.join(missing_dirs)}")
        print_info("Creo le directory mancanti...")
        for d in missing_dirs:
            (run_path / d).mkdir(exist_ok=True)
    
    # Check required files
    metrics_json = run_path / "data" / "metrics.json"
    metrics_csv = run_path / "data" / "metrics.csv"
    model_file = run_path / "models" / "final_model.weights.h5"
    client_metrics_dir = run_path / "logs" / "client_metrics"
    
    missing_files = []
    if not metrics_json.exists():
        missing_files.append("data/metrics.json")
    if not metrics_csv.exists():
        missing_files.append("data/metrics.csv")
    if not model_file.exists():
        missing_files.append("models/final_model.weights.h5")
    if not client_metrics_dir.exists():
        missing_files.append("logs/client_metrics/")
    
    if missing_files:
        print_error("File/directory essenziali mancanti:")
        for f in missing_files:
            print(f"  - {f}")
        print_info("Questa potrebbe essere una run incompleta o dalla vecchia struttura")
        return False
    
    print_success(f"Directory validata: {run_dir}")
    return True


def print_run_info(run_dir):
    """Print informazioni sulla run"""
    run_path = Path(run_dir)
    metrics_json = run_path / "data" / "metrics.json"
    summary_txt = run_path / "data" / "summary.txt"
    
    print(f"\n{Colors.BOLD}📁 Run Directory:{Colors.ENDC} {run_path}")
    print(f"{Colors.BOLD}🕒 Timestamp:{Colors.ENDC} {run_path.name.replace('run_', '')}")
    
    # Read summary if exists
    if summary_txt.exists():
        print(f"\n{Colors.BOLD}📄 Summary:{Colors.ENDC}")
        with open(summary_txt, 'r') as f:
            lines = f.readlines()
            # Print first 15 lines
            for line in lines[:15]:
                print(f"  {line.rstrip()}")
            if len(lines) > 15:
                print(f"  ... ({len(lines) - 15} more lines)")
    
    # Read metrics info
    if metrics_json.exists():
        with open(metrics_json, 'r') as f:
            data = json.load(f)
        
        num_rounds = len(data.get('rounds', []))
        print(f"\n{Colors.BOLD}📊 Metrics:{Colors.ENDC}")
        print(f"  Total rounds: {num_rounds}")
        
        if num_rounds > 0:
            last_round = data['rounds'][-1]
            centralized = last_round.get('centralized', {})
            print(f"  Final accuracy: {centralized.get('accuracy', 0):.4f}")
            print(f"  Final loss: {centralized.get('loss', 0):.4f}")
            print(f"  Final F1-score: {centralized.get('f1_macro', 0):.4f}")


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def run_server_analysis(run_dir, show_plots=True):
    """
    Esegue l'analisi server-side (analyze_results.py)
    Genera 8 grafici:
        1. Accuracy & Loss
        2. Detailed Metrics
        3. Fairness Analysis
        4. Convergence Analysis
        5. Client-Server Comparison
        6. Efficiency Analysis
        7. Advanced Metrics
        8. Statistical Summary
    """
    print_step(1, 4, "Server-Side Metrics Analysis")
    
    run_path = Path(run_dir)
    metrics_csv = run_path / "data" / "metrics.csv"
    plots_dir = run_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Importa e usa direttamente le funzioni di analyze_results.py
    try:
        # Carica dati JSON (più affidabile del CSV che può avere colonne variabili)
        metrics_json = run_path / "data" / "metrics.json"
        with open(metrics_json, 'r') as f:
            json_data = json.load(f)
        
        # Converti rounds in DataFrame
        rounds = json_data.get('rounds', [])
        if not rounds:
            print_warning("Nessun round trovato nei dati!")
            return False
        
        # Flatten dei dati
        flat_data = []
        for round_data in rounds:
            flat_row = {'round': round_data['round'], 'timestamp': round_data['timestamp']}
            for category in ['centralized', 'distributed', 'training', 'convergence', 'efficiency']:
                metrics = round_data.get(category, {})
                for key, value in metrics.items():
                    flat_row[f'{category}_{key}'] = value
            flat_data.append(flat_row)
        
        df = pd.DataFrame(flat_data)
        
        # Import functions
        from analyze_results import (
            plot_accuracy_loss,
            plot_metrics_detailed,
            plot_fairness_metrics,
            plot_convergence,
            plot_client_server_comparison,
            plot_efficiency_analysis,
            plot_advanced_metrics,
            plot_statistical_summary,
            print_summary_stats
        )
        
        # Print stats
        print_summary_stats(df)
        
        # Define save paths
        save_paths = {
            'accuracy_loss': plots_dir / "server_accuracy_loss.png",
            'detailed': plots_dir / "server_detailed.png",
            'fairness': plots_dir / "server_fairness.png",
            'convergence': plots_dir / "server_convergence.png",
            'client_server': plots_dir / "server_client_comparison.png",
            'efficiency': plots_dir / "server_efficiency.png",
            'advanced': plots_dir / "server_advanced.png",
            'statistical': plots_dir / "server_statistical.png"
        }
        
        # Generate plots
        print(f"  Generazione grafici server-side...")
        plot_accuracy_loss(df, save_path=save_paths['accuracy_loss'])
        plot_metrics_detailed(df, save_path=save_paths['detailed'])
        plot_fairness_metrics(df, save_path=save_paths['fairness'])
        plot_convergence(df, save_path=save_paths['convergence'])
        plot_client_server_comparison(df, save_path=save_paths['client_server'])
        plot_efficiency_analysis(df, save_path=save_paths['efficiency'])
        plot_advanced_metrics(df, save_path=save_paths['advanced'])
        plot_statistical_summary(df, save_path=save_paths['statistical'])
        
        print_success(f"8 grafici server-side salvati in {plots_dir}/")
        
        if show_plots:
            import matplotlib.pyplot as plt
            plt.show()
        
        return True
        
    except Exception as e:
        print_error(f"Errore nell'analisi server-side: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_client_analysis(run_dir, show_plots=True):
    """
    Esegue l'analisi client-side (plot_client_metrics.py)
    Genera 4 grafici:
        1. Training Progress per Client
        2. Round Statistics
        3. Client Performance Comparison
        4. Performance Distribution
    """
    print_step(2, 4, "Client-Side Metrics Analysis")
    
    run_path = Path(run_dir)
    client_metrics_dir = run_path / "logs" / "client_metrics"
    plots_dir = run_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    try:
        # Import functions
        from plot_client_metrics import (
            load_client_data,
            plot_client_accuracy_evolution,
            plot_client_training_time,
            plot_client_contribution,
            plot_round_statistics
        )
        
        # Load data
        data = load_client_data(str(client_metrics_dir))
        
        # Define save paths
        save_paths = {
            'accuracy': plots_dir / "client_accuracy_evolution.png",
            'training_time': plots_dir / "client_training_time.png",
            'contribution': plots_dir / "client_contribution.png",
            'round_stats': plots_dir / "client_round_stats.png"
        }
        
        # Generate plots
        print(f"  Generazione grafici client-side...")
        plot_client_accuracy_evolution(data, save_path=save_paths['accuracy'])
        plot_client_training_time(data, save_path=save_paths['training_time'])
        plot_client_contribution(data, save_path=save_paths['contribution'])
        plot_round_statistics(data, save_path=save_paths['round_stats'])
        
        print_success(f"4 grafici client-side salvati in {plots_dir}/")
        
        if show_plots:
            import matplotlib.pyplot as plt
            plt.show()
        
        return True
        
    except Exception as e:
        print_error(f"Errore nell'analisi client-side: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_bandwidth_analysis(run_dir, show_plots=True):
    """
    Esegue l'analisi bandwidth (plot_bandwidth_comparison.py)
    
    ⚠️ NOTA: Richiede un file di confronto senza compressione!
    Questa funzione potrebbe non funzionare se non hai una run senza compressione.
    """
    print_step(3, 4, "Bandwidth Analysis")
    
    run_path = Path(run_dir)
    metrics_json = run_path / "data" / "metrics.json"
    plots_dir = run_path / "plots"
    
    # Check if we have bandwidth metrics
    with open(metrics_json, 'r') as f:
        data = json.load(f)
    
    # Check se ci sono metriche di bandwidth in training o efficiency
    has_bandwidth = False
    if data.get('rounds'):
        for round_data in data['rounds']:
            # Check in training section (nuovo formato)
            if 'training' in round_data:
                if any('bandwidth' in key or 'compression' in key for key in round_data['training'].keys()):
                    has_bandwidth = True
                    break
            # Check in efficiency section (vecchio formato, backward compatibility)
            if 'efficiency' in round_data:
                if any(key.startswith('bandwidth_') for key in round_data['efficiency'].keys()):
                    has_bandwidth = True
                    break
    
    if not has_bandwidth:
        print_warning("Nessuna metrica bandwidth trovata in questa run")
        print_info("L'analisi bandwidth richiede una run con compressione abilitata")
        print_info("Skipping bandwidth analysis...")
        return True
    
    # Estrai e mostra statistiche bandwidth
    print("  📊 Compression Statistics:\n")
    
    # Raccogli metriche bandwidth da tutti i round
    bandwidth_data = []
    for round_data in data['rounds']:
        if 'training' in round_data and any('bandwidth' in k for k in round_data['training'].keys()):
            bandwidth_data.append({
                'round': round_data['round'],
                'original_kb': round_data['training'].get('round_bandwidth_original_kb', 0),
                'compressed_kb': round_data['training'].get('round_bandwidth_compressed_kb', 0),
                'saved_kb': round_data['training'].get('round_bandwidth_saved_kb', 0),
                'ratio': round_data['training'].get('round_compression_ratio', 0),
                'total_saved_kb': round_data['training'].get('total_bandwidth_saved_kb', 0)
            })
    
    if bandwidth_data:
        # Calcola totali
        total_original_mb = sum(d['original_kb'] for d in bandwidth_data) / 1024
        total_compressed_mb = sum(d['compressed_kb'] for d in bandwidth_data) / 1024
        total_saved_mb = sum(d['saved_kb'] for d in bandwidth_data) / 1024
        avg_ratio = sum(d['ratio'] for d in bandwidth_data) / len(bandwidth_data)
        savings_percent = (total_saved_mb / total_original_mb * 100) if total_original_mb > 0 else 0
        
        print(f"  📦 Total Data Transfer:")
        print(f"     Original:    {total_original_mb:.2f} MB")
        print(f"     Compressed:  {total_compressed_mb:.2f} MB")
        print(f"     Saved:       {total_saved_mb:.2f} MB ({savings_percent:.1f}%)")
        print(f"\n  🗜️  Compression Ratio: {avg_ratio:.2f}x")
        print(f"\n  ℹ️  Per confronti dettagliati, usa:")
        print(f"     python plot_bandwidth_comparison.py <baseline_run> <compressed_run>")
    
    return True


def plot_test_evaluation(test_metrics, val_metrics=None, plots_dir=None):
    """
    Crea un grafico visuale completo con metriche e visualizzazioni
    
    Args:
        test_metrics: dizionario con metriche test
        val_metrics: dizionario con metriche validation (opzionale)
        plots_dir: directory dove salvare il grafico
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    test_acc = test_metrics.get('accuracy', 0)
    test_loss = test_metrics.get('loss', 0)
    
    # === MAIN METRICS PLOT ===
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
    values = [
        test_metrics.get('accuracy', 0) * 100,
        test_metrics.get('f1_macro', 0) * 100,
        test_metrics.get('precision_macro', 0) * 100,
        test_metrics.get('recall_macro', 0) * 100
    ]
    colors = ['#2ecc71' if v >= 90 else '#f39c12' if v >= 80 else '#e74c3c' for v in values]
    
    bars = ax1.barh(metrics, values, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Score (%)', fontsize=14, fontweight='bold')
    ax1.set_title('Test Set Performance Metrics', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.tick_params(axis='y', labelsize=12)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax1.text(val + 1, i, f'{val:.2f}%', va='center', fontsize=13, fontweight='bold')
    
    # Add loss info
    ax1.text(0.02, 0.02, f'Loss: {test_loss:.4f}', transform=ax1.transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1.5))
    
    plt.tight_layout()
    
    # Save main metrics
    if plots_dir:
        save_path = plots_dir / "test_evaluation.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ Test metrics saved in {plots_dir}/")
    
    plt.close()
    
    # === GENERALIZATION CHECK (Separate plot) ===
    if val_metrics:
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        
        val_acc = val_metrics.get('accuracy', 0) * 100
        test_acc_pct = test_acc * 100
        delta = test_acc_pct - val_acc
        
        x = ['Validation', 'Test']
        y = [val_acc, test_acc_pct]
        colors_comp = ['#3498db', '#2ecc71' if delta >= -1 else '#e74c3c']
        
        bars2 = ax2.bar(x, y, color=colors_comp, alpha=0.85, edgecolor='black', linewidth=1.5, width=0.6)
        ax2.set_ylim(max(0, min(y) - 5), 100)
        ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax2.set_title('Generalization Check', fontsize=16, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.tick_params(axis='both', labelsize=12)
        
        # Value labels
        for bar, val in zip(bars2, y):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Delta annotation
        delta_color = '#27ae60' if abs(delta) < 1 else '#f39c12' if abs(delta) < 3 else '#e74c3c'
        ax2.text(0.5, 0.95, f'Δ = {delta:+.2f}%', transform=ax2.transAxes,
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=delta_color, alpha=0.3, edgecolor=delta_color, linewidth=2))
        
        # Quality indicator
        if abs(delta) < 1:
            quality_text = 'EXCELLENT'
            quality_color = '#27ae60'
        elif abs(delta) < 3:
            quality_text = 'GOOD'
            quality_color = '#f39c12'
        else:
            quality_text = 'CHECK'
            quality_color = '#e74c3c'
        
        ax2.text(0.5, 0.05, quality_text, transform=ax2.transAxes,
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=quality_color)
        
        plt.tight_layout()
        
        # Save generalization check
        if plots_dir:
            save_path_gen = plots_dir / "generalization_check.png"
            plt.savefig(save_path_gen, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✅ Generalization check saved in {plots_dir}/")
        
        plt.close()


def run_test_evaluation(run_dir, show_plots=True):
    """
    Esegue la valutazione finale sul test set (final_test_evaluation.py)
    Confronta validation vs test performance
    """
    print_step(4, 4, "Final Test Set Evaluation")
    
    run_path = Path(run_dir)
    model_path = run_path / "models" / "final_model.weights.h5"
    metrics_json = run_path / "data" / "metrics.json"
    plots_dir = run_path / "plots"
    
    try:
        # Import functions
        from final_test_evaluation import (
            load_final_model,
            evaluate_on_test_set,
            print_test_results,
            compare_with_validation,
            save_test_results
        )
        
        # Load model
        print(f"  Caricamento modello: {model_path}")
        model = load_final_model(str(model_path))
        
        # Evaluate on test set
        print(f"  Valutazione su test set (10,000 samples)...")
        test_metrics = evaluate_on_test_set(model)
        
        # Print results
        print_test_results(test_metrics)
        
        # Compare with validation (if available)
        with open(metrics_json, 'r') as f:
            data = json.load(f)
        
        val_metrics = None
        if data.get('rounds'):
            last_round = data['rounds'][-1]
            val_metrics = last_round.get('centralized', {})
            
            if val_metrics:
                print(f"\n  Confronto validation vs test:")
                compare_with_validation(test_metrics=test_metrics)
        
        # Generate visualization plot
        print(f"  Generazione grafico test evaluation...")
        plot_test_evaluation(test_metrics, val_metrics, plots_dir)
        
        # Save results JSON
        output_file = plots_dir / "test_evaluation.json"
        save_test_results(test_metrics, str(output_file))
        print_success(f"Grafico test evaluation salvato in {plots_dir}/")
        
        return True
        
    except Exception as e:
        print_error(f"Errore nella valutazione finale: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='🎯 Analisi completa di una run di federated learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analizza ultima run
    python analyze_run.py results/run_20251025_123456/
    
    # Analizza senza mostrare grafici
    python analyze_run.py results/run_20251025_123456/ --no-show
    
    # Trova e analizza ultima run automaticamente
    python analyze_run.py --latest

Output:
    Tutti i file vengono salvati in <run_dir>/plots/
    - Server-side: 8 grafici (server_*.png)
    - Client-side: 4 grafici (client_*.png)
    - Test evaluation: test_evaluation.png + test_evaluation.json
        """
    )
    
    parser.add_argument(
        'run_dir',
        nargs='?',
        type=str,
        help='Path alla directory della run (es: results/run_20251025_123456/)'
    )
    
    parser.add_argument(
        '--latest', '-l',
        action='store_true',
        help='Analizza automaticamente l\'ultima run disponibile'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Non mostrare i grafici interattivi (solo salva)'
    )
    
    parser.add_argument(
        '--skip-server',
        action='store_true',
        help='Salta analisi server-side'
    )
    
    parser.add_argument(
        '--skip-client',
        action='store_true',
        help='Salta analisi client-side'
    )
    
    parser.add_argument(
        '--skip-bandwidth',
        action='store_true',
        help='Salta analisi bandwidth'
    )
    
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Salta valutazione test finale'
    )
    
    args = parser.parse_args()
    
    # Determine run directory
    if args.latest:
        # Find latest run
        results_dir = Path("results")
        run_dirs = sorted(results_dir.glob("run_*"))
        if not run_dirs:
            print_error("Nessuna run trovata in results/")
            print_info("Esegui prima un training con il nuovo sistema")
            return 1
        run_dir = run_dirs[-1]
        print_info(f"Usando ultima run: {run_dir}")
    elif args.run_dir:
        run_dir = args.run_dir
    else:
        parser.print_help()
        return 1
    
    # Print header
    print_header("FEDERATED LEARNING RUN ANALYSIS")
    
    # Validate directory
    if not validate_run_directory(run_dir):
        return 1
    
    # Print run info
    print_run_info(run_dir)
    
    # Run analyses
    print_header("RUNNING ANALYSES")
    
    show_plots = not args.no_show
    results = []
    
    if not args.skip_server:
        success = run_server_analysis(run_dir, show_plots)
        results.append(("Server-side", success))
    
    if not args.skip_client:
        success = run_client_analysis(run_dir, show_plots)
        results.append(("Client-side", success))
    
    if not args.skip_bandwidth:
        success = run_bandwidth_analysis(run_dir, show_plots)
        results.append(("Bandwidth", success))
    
    if not args.skip_test:
        success = run_test_evaluation(run_dir, show_plots)
        results.append(("Test evaluation", success))
    
    # Summary
    print_header("ANALYSIS COMPLETE")
    
    print(f"{Colors.BOLD}Results:{Colors.ENDC}")
    for name, success in results:
        status = f"{Colors.GREEN}✅ SUCCESS{Colors.ENDC}" if success else f"{Colors.RED}❌ FAILED{Colors.ENDC}"
        print(f"  {name:20s} {status}")
    
    run_path = Path(run_dir)
    plots_dir = run_path / "plots"
    
    print(f"\n{Colors.BOLD}Output:{Colors.ENDC}")
    print(f"  📂 Plots directory: {plots_dir}/")
    
    # Count generated files
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        json_files = list(plots_dir.glob("*.json"))
        print(f"  📊 Grafici generati: {len(plot_files)}")
        print(f"  📄 File JSON: {len(json_files)}")
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 Analisi completata!{Colors.ENDC}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
