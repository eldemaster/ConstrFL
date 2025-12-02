#!/usr/bin/env python3
"""
Final Test Set Evaluation - UCI HAR Dataset
============================================

Questo script valuta il modello FINALE sul TEST SET (1545 samples).

IMPORTANTE:
- Durante training: usiamo VALIDATION set (1545 samples) per monitoring
- Dopo training: usiamo TEST set (1545 samples) per valutazione finale imparziale

Il test set NON è MAI visto durante il training → no data leakage!
"""

import argparse
import json
from pathlib import Path
from har.task import load_model, load_global_test_data
from har.metrics import evaluate_model_detailed


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
    Valuta il modello sul TEST SET UCI HAR
    
    Returns:
        dict: Metriche dettagliate
    """
    # Carica TEST SET (1545 samples - mai visto durante training)
    x_test, y_test = load_global_test_data()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🧪 FINAL TEST SET EVALUATION - UCI HAR Dataset")
        print(f"{'='*70}")
        print(f"📊 Test Set: {len(x_test)} samples (15% of total dataset)")
        print(f"⚠️  Questo set NON è mai stato visto durante il training")
        print(f"📱 Activities: 6 classes (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,")
        print(f"               SITTING, STANDING, LAYING)")
        print(f"{'='*70}\n")
    
    # Valuta con metriche dettagliate (6 classi per HAR)
    metrics = evaluate_model_detailed(model, x_test, y_test, num_classes=6)
    
    return metrics


def print_test_results(metrics):
    """
    Stampa i risultati del test in modo leggibile
    """
    print(f"\n{'='*70}")
    print(f"📊 FINAL TEST RESULTS (1,545 samples - UCI HAR)")
    print(f"{'='*70}")
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Loss:        {metrics['loss']:.4f}")
    
    print(f"\n📈 Classification Metrics:")
    print(f"  F1-Score:    {metrics['f1_macro']:.4f} (macro)")
    print(f"  Precision:   {metrics['precision_macro']:.4f} (macro)")
    print(f"  Recall:      {metrics['recall_macro']:.4f} (macro)")
    
    print(f"\n⚖️  Per-Class Performance (6 activity classes):")
    print(f"  Best Class:  {metrics['best_class_acc']:.4f}")
    print(f"  Worst Class: {metrics['worst_class_acc']:.4f}")
    print(f"  Std Dev:     {metrics['class_acc_std']:.4f}")
    
    print(f"\n🏃 Activity Recognition Quality:")
    if metrics['accuracy'] >= 0.95:
        print(f"  ⭐⭐⭐ ECCELLENTE (>95%)")
    elif metrics['accuracy'] >= 0.90:
        print(f"  ⭐⭐ OTTIMO (90-95%)")
    elif metrics['accuracy'] >= 0.85:
        print(f"  ⭐ BUONO (85-90%)")
    else:
        print(f"  ⚠️  Da migliorare (<85%)")
    
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
        test_f1 = test_metrics['f1_macro']
        
        delta_acc = test_acc - val_acc
        delta_loss = test_loss - val_loss
        delta_f1 = test_f1 - val_f1
        
        print(f"\n                    Validation         Test            Δ")
        print(f"                    (1545 samples)     (1545 samples)")
        print(f"  Accuracy:         {val_acc:.4f}          {test_acc:.4f}        {delta_acc:+.4f}")
        print(f"  Loss:             {val_loss:.4f}          {test_loss:.4f}        {delta_loss:+.4f}")
        print(f"  F1-Score:         {val_f1:.4f}          {test_f1:.4f}        {delta_f1:+.4f}")
        
        print(f"\n💡 Interpretazione:")
        if abs(delta_acc) < 0.02:
            print(f"  ✅ Generalizzazione eccellente (differenza < 2%)")
        elif abs(delta_acc) < 0.05:
            print(f"  ✅ Buona generalizzazione (differenza < 5%)")
        else:
            if delta_acc < 0:
                print(f"  ⚠️  Possibile overfitting (test peggiore di validation)")
            else:
                print(f"  ⚠️  Test migliore di validation (inaspettato - verificare)")
    else:
        print(f"\n                    Validation         Test            Δ")
        print(f"  Accuracy:         {val_acc:.4f}          ----         ----")
        print(f"  Loss:             {val_loss:.4f}          ----         ----")
        print(f"\n💡 Esegui prima l'evaluation con --model-path per vedere il confronto")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Valuta il modello finale sul test set UCI HAR (1545 samples)'
    )
    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default=None,
        help='Path ai weights del modello (es: results/fl_experiment_*_final_model.weights.h5)'
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
    parser.add_argument(
        '--compare', '-c',
        action='store_true',
        help='Confronta con validation set',
        default=True
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
    if args.compare:
        compare_with_validation(args.experiment, test_metrics=metrics if args.model_path else None)
    
    # Avvertimenti finali
    if not args.model_path:
        print("\n" + "⚠️ " * 25)
        print("⚠️  ATTENZIONE: Hai valutato il modello INIZIALE (random weights)!")
        print("⚠️  Per valutare il modello TRAINATO:")
        print("⚠️    1. Il modello finale è già stato salvato durante training")
        print("⚠️    2. Usa: python final_test_evaluation.py --model-path results/fl_experiment_*_final_model.weights.h5")
        print("⚠️ " * 25 + "\n")


if __name__ == "__main__":
    main()
