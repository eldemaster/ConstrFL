#!/usr/bin/env python3
"""
Final Test Set Evaluation
==========================

Questo script valuta il modello FINALE sul TEST SET (10,000 samples).

IMPORTANTE:
- Durante training: usiamo VALIDATION set (12K) per monitoring
- Dopo training: usiamo TEST set (10K) per valutazione finale imparziale

Il test set NON è MAI visto durante il training → no data leakage!
"""

import argparse
import json
from pathlib import Path
from mnist_optimised.task import load_model, load_global_test_data
from mnist_optimised.metrics import evaluate_model_detailed


def load_final_model(model_path=None):
    """
    Carica il modello finale
    
    Args:
        model_path: Path ai weights salvati. Se None, usa modello iniziale.
    """
    model = load_model()
    
    if model_path and Path(model_path).exists():
        print(f"📂 Caricando weights da: {model_path}")
        model.load_weights(model_path)
    else:
        print("⚠️  Nessun weight file specificato - usando modello iniziale (random)")
        print("💡 TIP: Salva il modello finale durante training per usare questo script")
    
    return model


def evaluate_on_test_set(model, verbose=True):
    """
    Valuta il modello sul TEST SET originale MNIST
    
    Returns:
        dict: Metriche dettagliate
    """
    # Carica TEST SET (10,000 samples - mai visto durante training)
    x_test, y_test = load_global_test_data()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🧪 FINAL TEST SET EVALUATION")
        print(f"{'='*70}")
        print(f"📊 Test Set: {len(x_test)} samples (MNIST original test set)")
        print(f"⚠️  Questo set NON è mai stato visto durante il training")
        print(f"{'='*70}\n")
    
    # Valuta con metriche dettagliate
    metrics = evaluate_model_detailed(model, x_test, y_test, num_classes=10)
    
    return metrics


def print_test_results(metrics):
    """
    Stampa i risultati del test in modo leggibile
    """
    print(f"\n{'='*70}")
    print(f"📊 FINAL TEST RESULTS (10,000 samples)")
    print(f"{'='*70}")
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Loss:        {metrics['loss']:.4f}")
    
    print(f"\n📈 Classification Metrics:")
    print(f"  F1-Score:    {metrics['f1_macro']:.4f} (macro)")
    print(f"  Precision:   {metrics['precision_macro']:.4f} (macro)")
    print(f"  Recall:      {metrics['recall_macro']:.4f} (macro)")
    
    print(f"\n⚖️  Per-Class Performance:")
    print(f"  Best Class:  {metrics['best_class_acc']:.4f}")
    print(f"  Worst Class: {metrics['worst_class_acc']:.4f}")
    print(f"  Std Dev:     {metrics['class_acc_std']:.4f}")
    
    # Per-class breakdown se disponibile
    if 'per_class_accuracy' in metrics:
        print(f"\n📋 Per-Class Accuracy:")
        for cls, acc in enumerate(metrics['per_class_accuracy']):
            print(f"  Digit {cls}:  {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\n{'='*70}\n")


def save_test_results(metrics, output_path="results/final_test_evaluation.json"):
    """
    Salva i risultati del test in JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Risultati salvati in: {output_path}")


def compare_with_validation(experiment_name=None, test_metrics=None):
    """
    Confronta i risultati del test con quelli della validation
    """
    if not experiment_name:
        # Trova ultimo esperimento
        results_dir = Path("results")
        json_files = sorted(results_dir.glob("fl_experiment_*.json"))
        if not json_files:
            print("⚠️  Nessun esperimento trovato per confronto")
            return
        json_path = json_files[-1]
    else:
        json_path = Path(f"results/{experiment_name}.json")
    
    if not json_path.exists():
        print(f"⚠️  File non trovato: {json_path}")
        return
    
    # Carica risultati validation (ultimo round)
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not data['rounds']:
        print("⚠️  Nessun round trovato nell'esperimento")
        return
    
    last_round = data['rounds'][-1]
    val_acc = last_round['centralized']['accuracy']
    val_loss = last_round['centralized']['loss']
    val_f1 = last_round['centralized'].get('f1_macro', 0)
    
    print(f"\n{'='*70}")
    print(f"📊 VALIDATION vs TEST COMPARISON")
    print(f"{'='*70}")
    print(f"Esperimento: {json_path.stem}")
    print(f"Round finale: {last_round['round']}")
    
    if test_metrics:
        test_acc = test_metrics['accuracy']
        test_loss = test_metrics['loss']
        test_f1 = test_metrics.get('f1_macro', 0)
        
        acc_diff = test_acc - val_acc
        loss_diff = test_loss - val_loss
        f1_diff = test_f1 - val_f1
        
        print(f"\nMetrica              Validation (12K)    Test (10K)       Δ")
        print(f"-" * 70)
        print(f"  Accuracy:          {val_acc:.4f}            {test_acc:.4f}        {acc_diff:+.4f}")
        print(f"  Loss:              {val_loss:.4f}            {test_loss:.4f}        {loss_diff:+.4f}")
        if val_f1 > 0:
            print(f"  F1-Score (macro):  {val_f1:.4f}            {test_f1:.4f}        {f1_diff:+.4f}")
        
        # Analisi generalizzazione
        print(f"\n⚖️  Generalizzazione:")
        diff = abs(acc_diff)
        if diff < 0.02:
            print(f"   ✅ OTTIMA (|Δ| = {diff:.4f} < 0.02)")
            print(f"      Il modello generalizza perfettamente")
        elif diff < 0.05:
            print(f"   ✅ BUONA (|Δ| = {diff:.4f} < 0.05)")
            print(f"      Leggera differenza, accettabile")
        else:
            print(f"   ⚠️  ATTENZIONE (|Δ| = {diff:.4f} > 0.05)")
            if acc_diff < 0:
                print(f"      Possibile underfitting o dataset bias")
            else:
                print(f"      Possibile overfitting")
    else:
        print(f"\n                    Validation (12K)    Test (10K)       Δ")
        print(f"  Accuracy:         {val_acc:.4f}          ----         ----")
        print(f"  Loss:             {val_loss:.4f}          ----         ----")
        print(f"\n💡 Esegui prima l'evaluation con --model-path per vedere il confronto")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Valuta il modello finale sul test set MNIST (10K samples)'
    )
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default=None,
        help='Path ai weights del modello (es: final_model.weights.h5)'
    )
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        default=None,
        help='Nome esperimento per confronto con validation'
    )
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Salva i risultati in JSON'
    )
    
    args = parser.parse_args()
    
    # Carica modello
    model = load_final_model(args.model_path)
    
    # Valuta sul test set
    metrics = evaluate_on_test_set(model)
    
    # Stampa risultati
    print_test_results(metrics)
    
    # Salva se richiesto
    if args.save:
        save_test_results(metrics)
    
    # Confronta con validation se richiesto
    if args.experiment or not args.model_path:
        compare_with_validation(args.experiment, test_metrics=metrics if args.model_path else None)
    
    # Avvertimenti finali
    if not args.model_path:
        print("\n" + "⚠️ " * 25)
        print("⚠️  ATTENZIONE: Hai valutato il modello INIZIALE (random weights)!")
        print("⚠️  Per valutare il modello TRAINATO:")
        print("⚠️    1. Salva il modello finale durante training")
        print("⚠️    2. Usa: python final_test_evaluation.py --model-path model.h5")
        print("⚠️ " * 25 + "\n")


if __name__ == "__main__":
    main()
