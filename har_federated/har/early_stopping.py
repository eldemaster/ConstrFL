"""
Local Early Stopping Module
============================

Implementa patience-based early stopping per training locale sui client.
Riduce il compute time fermando il training quando il miglioramento è marginale.

Design:
- LocalEarlyStopping: fixed patience per tutti i round
- AdaptiveEarlyStopping: patience dinamica (alta all'inizio, bassa alla fine)

Best Practice:
- Monitora validation loss locale
- Salva best weights automaticamente
- Restore best weights se early stop

Author: Generated for FL Optimization Roadmap
Date: 29 Ottobre 2025
"""

import numpy as np
from typing import Optional, List
import copy


class LocalEarlyStopping:
    """
    Patience-based early stopping per training locale
    
    Monitora validation loss e ferma training se non c'è improvement
    per patience epochs consecutivi.
    
    Args:
        patience: Numero di epochs senza improvement prima di stop (default: 3)
        min_delta: Miglioramento minimo per considerare "improvement" (default: 0.001)
        restore_best_weights: Se True, restore weights con best val_loss (default: True)
        verbose: Se True, stampa info quando stopping triggered (default: False)
    
    Example:
        >>> early_stop = LocalEarlyStopping(patience=3, min_delta=0.001)
        >>> for epoch in range(max_epochs):
        ...     train_loss = model.fit(...)
        ...     val_loss = model.evaluate(x_val, y_val)
        ...     weights = model.get_weights()
        ...     early_stop.update(val_loss, weights)
        ...     if early_stop.should_stop():
        ...         best_weights = early_stop.get_best_weights()
        ...         model.set_weights(best_weights)
        ...         break
    """
    
    def __init__(
        self, 
        patience: int = 3,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
        verbose: bool = False
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # State tracking
        self.best_loss = float('inf')
        self.best_weights = None
        self.wait = 0  # Numero di epochs senza improvement
        self.stopped_epoch = 0
        self.stopped = False
        
        # History
        self.loss_history: List[float] = []
        
    def update(self, val_loss: float, weights: Optional[List] = None) -> bool:
        """
        Aggiorna stato early stopping con nuovo validation loss
        
        Args:
            val_loss: Validation loss corrente
            weights: Weights correnti del modello (opzionale)
        
        Returns:
            bool: True se c'è stato improvement, False altrimenti
        """
        self.loss_history.append(val_loss)
        
        # Check for improvement
        # Improvement = riduzione di almeno min_delta
        improvement = self.best_loss - val_loss
        
        if improvement > self.min_delta:
            # C'è improvement significativo
            self.best_loss = val_loss
            self.wait = 0
            
            # Salva best weights
            if self.restore_best_weights and weights is not None:
                self.best_weights = copy.deepcopy(weights)
            
            if self.verbose:
                print(f"    📉 Val loss improved to {val_loss:.4f}")
            
            return True
        else:
            # NO improvement
            self.wait += 1
            
            if self.verbose:
                print(f"    ⏸️  No improvement ({self.wait}/{self.patience})")
            
            return False
    
    def should_stop(self) -> bool:
        """
        Verifica se training dovrebbe fermarsi
        
        Returns:
            bool: True se patience esaurita, False altrimenti
        """
        if self.wait >= self.patience:
            self.stopped = True
            self.stopped_epoch = len(self.loss_history)
            
            if self.verbose:
                print(f"    ⏹️  Early stopping triggered at epoch {self.stopped_epoch}")
                print(f"        Best val_loss: {self.best_loss:.4f}")
            
            return True
        
        return False
    
    def get_best_weights(self) -> Optional[List]:
        """
        Restituisce i best weights salvati
        
        Returns:
            Best weights o None se restore_best_weights=False
        """
        return self.best_weights
    
    def get_stats(self) -> dict:
        """
        Statistiche early stopping
        
        Returns:
            Dict con statistiche: stopped, stopped_epoch, best_loss, etc.
        """
        return {
            'stopped': self.stopped,
            'stopped_epoch': self.stopped_epoch,
            'total_epochs': len(self.loss_history),
            'best_loss': self.best_loss,
            'final_loss': self.loss_history[-1] if self.loss_history else None,
            'wait': self.wait,
            'improvement': self.best_loss - self.loss_history[-1] if self.loss_history else 0,
        }
    
    def reset(self):
        """Reset stato per nuovo round"""
        self.best_loss = float('inf')
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stopped = False
        self.loss_history = []


class AdaptiveEarlyStopping(LocalEarlyStopping):
    """
    Early stopping con patience ADATTIVA
    
    La patience diminuisce durante il FL training:
    - Round iniziali: alta patience (exploration)
    - Round finali: bassa patience (exploitation)
    
    Formula:
        patience_t = max(min_patience, base_patience * (1 - round_t / total_rounds))
    
    Args:
        base_patience: Patience massima (round iniziali)
        min_patience: Patience minima (round finali)
        total_rounds: Totale FL rounds previsti
        min_delta: Threshold per improvement
        restore_best_weights: Se True, restore best weights
        verbose: Se True, stampa info
    
    Example:
        >>> # FL con 10 rounds totali
        >>> early_stop = AdaptiveEarlyStopping(
        ...     base_patience=5, min_patience=2, total_rounds=10
        ... )
        >>> 
        >>> # Round 1: patience = 5 (exploration)
        >>> early_stop.set_current_round(1)
        >>> 
        >>> # Round 10: patience = 2 (exploitation)
        >>> early_stop.set_current_round(10)
    """
    
    def __init__(
        self,
        base_patience: int = 5,
        min_patience: int = 2,
        total_rounds: int = 10,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
        verbose: bool = False
    ):
        self.base_patience = base_patience
        self.min_patience = min_patience
        self.total_rounds = total_rounds
        self.current_round = 1
        
        # Calcola patience iniziale
        initial_patience = self._calculate_patience(1)
        
        super().__init__(
            patience=initial_patience,
            min_delta=min_delta,
            restore_best_weights=restore_best_weights,
            verbose=verbose
        )
    
    def _calculate_patience(self, round_num: int) -> int:
        """
        Calcola patience per round corrente
        
        Formula: patience_t = max(min_patience, base_patience * (1 - t/T))
        
        Args:
            round_num: Numero del round corrente (1-indexed)
        
        Returns:
            Patience calcolata per questo round
        """
        # Fraction di FL training completato
        progress = (round_num - 1) / self.total_rounds  # 0.0 → 1.0
        
        # Patience decresce linearmente
        adaptive_patience = self.base_patience * (1 - progress)
        
        # Clamp a min_patience
        patience = max(self.min_patience, int(adaptive_patience))
        
        return patience
    
    def set_current_round(self, round_num: int):
        """
        Aggiorna round corrente e ricalcola patience
        
        Chiamare all'inizio di ogni FL round!
        
        Args:
            round_num: Numero del FL round corrente (1-indexed)
        """
        self.current_round = round_num
        self.patience = self._calculate_patience(round_num)
        
        if self.verbose:
            print(f"    🔄 Adaptive patience for round {round_num}: {self.patience}")
        
        # Reset per nuovo round
        self.reset()
    
    def get_stats(self) -> dict:
        """
        Statistiche con info adaptive
        
        Returns:
            Dict con statistiche + adaptive info
        """
        base_stats = super().get_stats()
        
        # Aggiungi info adaptive
        base_stats.update({
            'adaptive_patience': self.patience,
            'base_patience': self.base_patience,
            'min_patience': self.min_patience,
            'current_round': self.current_round,
            'total_rounds': self.total_rounds,
            'training_progress': (self.current_round - 1) / self.total_rounds,
        })
        
        return base_stats


def calculate_compute_savings(actual_epochs: int, max_epochs: int) -> dict:
    """
    Calcola risparmio computazionale da early stopping
    
    Args:
        actual_epochs: Epochs realmente eseguiti
        max_epochs: Epochs massimi configurati
    
    Returns:
        Dict con: compute_saved_pct, epochs_saved, efficiency_ratio
    """
    epochs_saved = max_epochs - actual_epochs
    compute_saved_pct = (epochs_saved / max_epochs) * 100 if max_epochs > 0 else 0
    efficiency_ratio = actual_epochs / max_epochs if max_epochs > 0 else 1.0
    
    return {
        'actual_epochs': actual_epochs,
        'max_epochs': max_epochs,
        'epochs_saved': epochs_saved,
        'compute_saved_pct': compute_saved_pct,
        'efficiency_ratio': efficiency_ratio,
    }
