"""
🆕 FEDERATED DISTILLATION STRATEGY

Implementa Federated Distillation invece di FedAvg:
- I client inviano SOFT PREDICTIONS invece di model weights
- Il server aggrega le predictions e usa knowledge distillation

Vantaggi:
- 200x riduzione della comunicazione (10KB vs 2MB)
- Privacy migliorata (predictions vs gradients)
- Supporta modelli eterogenei
"""

import numpy as np
from logging import INFO
from typing import Dict, List, Optional, Tuple, Union

import tensorflow as tf
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from har.data_utils import get_unlabeled_batch


class DistillationFedAvg(FedAvg):
    """
    Federated Distillation Strategy
    
    Invece di aggregare i pesi del modello (FedAvg tradizionale),
    questa strategia:
    1. Invia unlabeled data ai client
    2. Riceve soft predictions dai client
    3. Aggrega le predictions (media)
    4. Usa knowledge distillation per aggiornare il global model
    """
    
    def __init__(
        self,
        *args,
        unlabeled_batch_size: int = 100,
        distillation_temperature: float = 3.0,
        distillation_epochs: int = 5,
        distillation_lr: float = 0.001,
        **kwargs
    ):
        """
        Args:
            unlabeled_batch_size: dimensione del batch unlabeled per distillation
            distillation_temperature: temperatura per softmax scaling (default: 3.0)
            distillation_epochs: numero di epoche per distillation training
            distillation_lr: learning rate per distillation
        """
        super().__init__(*args, **kwargs)
        
        self.unlabeled_batch_size = unlabeled_batch_size
        self.distillation_temperature = distillation_temperature
        self.distillation_epochs = distillation_epochs
        self.distillation_lr = distillation_lr
        
        # Cache per unlabeled data (riutilizzato in tutti i round)
        self.unlabeled_data = None
        self.global_model = None
        
        log(INFO, f"🎓 Federated Distillation Strategy initialized:")
        log(INFO, f"   Unlabeled batch size: {unlabeled_batch_size}")
        log(INFO, f"   Temperature: {distillation_temperature}")
        log(INFO, f"   Distillation epochs: {distillation_epochs}")
        log(INFO, f"   Distillation LR: {distillation_lr}")
    
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Inizializza i parametri del global model"""
        log(INFO, "Initializing global model parameters...")
        
        # Ottieni parametri iniziali dal primo client disponibile
        initial_parameters = super().initialize_parameters(client_manager)
        # Salva per poter inizializzare il modello globale coerentemente
        self.initial_parameters = initial_parameters
        
        # Genera unlabeled data (una volta sola)
        if self.unlabeled_data is None:
            log(INFO, "🎲 Generating unlabeled batch for distillation...")
            self.unlabeled_data = get_unlabeled_batch(
                batch_size=self.unlabeled_batch_size
            )
        
        return initial_parameters
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configura i client per training
        
        IMPORTANTE: Invia unlabeled_data ai client insieme ai parametri
        """
        config = {}
        
        # Parametri standard (epochs, batch_size, etc.)
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # 🆕 DISTILLATION: Aggiungi unlabeled data al config
        config["distillation_enabled"] = True
        config["distillation_temperature"] = self.distillation_temperature
        
        # Serializza unlabeled data per invio ai client
        # NOTA: Flower non supporta liste in config, usiamo bytes e string
        if self.unlabeled_data is not None:
            # Shape as string "rows,cols" (Flower doesn't support list[int])
            config["unlabeled_data_shape"] = f"{self.unlabeled_data.shape[0]},{self.unlabeled_data.shape[1]}"
            # Serialize as bytes (Flower supports bytes in config)
            config["unlabeled_data_bytes"] = self.unlabeled_data.tobytes()
        
        # Ottieni client disponibili
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients
        )
        
        log(INFO, f"🚀 Round {server_round}: sending unlabeled data to {len(clients)} clients")
        log(INFO, f"   Unlabeled batch size: {self.unlabeled_data.shape[0]}")
        log(INFO, f"   Communication size: ~{self.unlabeled_data.nbytes / 1024:.1f} KB")
        
        # 🔧 FIX: Restituisci FitIns objects, non solo config!
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggrega i risultati dai client
        
        🆕 DISTILLATION: Invece di aggregare pesi, aggrega soft predictions
        """
        if not results:
            log(INFO, f"⚠️  NO RESULTS! Failures: {len(failures)}")
            if failures:
                for i, failure in enumerate(failures[:5]):  # Log primi 5 failures
                    log(INFO, f"   Failure {i} type: {type(failure)}")
                    log(INFO, f"   Failure {i} content: {failure}")
                    if hasattr(failure, '__dict__'):
                        log(INFO, f"   Failure {i} dict: {failure.__dict__}")
            return None, {}
        
        log(INFO, f"\n{'='*70}")
        log(INFO, f"🎓 FEDERATED DISTILLATION - Round {server_round}")
        log(INFO, f"{'='*70}")
        
        # Estrai soft predictions dai risultati
        # NOTA: I client inviano predictions invece di weights!
        soft_predictions_list = []
        num_samples_list = []
        
        for client_proxy, fit_res in results:
            # 🔧 Gestisci sia FitRes object che dict (serializzazione Ray)
            if isinstance(fit_res, dict):
                # fit_res è un dict (serializzato da Ray)
                parameters = fit_res["parameters"]
                num_examples = fit_res["num_examples"]
                metrics = fit_res["metrics"]
            else:
                # fit_res è un FitRes object
                parameters = fit_res.parameters
                num_examples = fit_res.num_examples
                metrics = fit_res.metrics
            
            # Le predictions sono nei parameters
            predictions_flat = parameters_to_ndarrays(parameters)
            
            # Reshape predictions: (num_unlabeled_samples, num_classes)
            # NOTA: num_examples è il training set size del client (per weighted average),
            #       num_unlabeled è nelle metriche
            num_unlabeled = metrics.get("num_unlabeled", self.unlabeled_batch_size)
            num_classes = len(predictions_flat[0]) // num_unlabeled
            
            predictions = predictions_flat[0].reshape(num_unlabeled, num_classes)
            soft_predictions_list.append(predictions)
            # Usa training set size per weighted average
            num_samples_list.append(num_examples)
            
            log(INFO, f"   Client {client_proxy.cid}: {num_unlabeled} predictions received")
            log(INFO, f"      Shape: {predictions.shape}, "
                     f"Range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        # Aggregazione: media delle soft predictions (weighted by num_samples)
        total_samples = sum(num_samples_list)
        weights = [n / total_samples for n in num_samples_list]
        
        aggregated_predictions = np.zeros_like(soft_predictions_list[0])
        for predictions, weight in zip(soft_predictions_list, weights):
            aggregated_predictions += predictions * weight
        
        log(INFO, f"\n📊 Aggregated predictions:")
        log(INFO, f"   Shape: {aggregated_predictions.shape}")
        log(INFO, f"   Range: [{aggregated_predictions.min():.3f}, {aggregated_predictions.max():.3f}]")
        log(INFO, f"   Sum per sample: {aggregated_predictions.sum(axis=1).mean():.3f} "
                 f"(should be ~1.0 for probabilities)")
        
        # 🎓 KNOWLEDGE DISTILLATION: Train global model con aggregated predictions
        log(INFO, f"\n🔥 Training global model via knowledge distillation...")
        log(INFO, f"   Distillation epochs: {self.distillation_epochs}")
        log(INFO, f"   Learning rate: {self.distillation_lr}")
        
        # Ottieni current global model weights dai risultati CLIENT
        # NOTA: I risultati contengono predictions, non weights!
        # Dobbiamo usare i weights che avevamo inviato ai client inizialmente
        # Per ora, creiamo un nuovo modello trainato e lo aggiorniamo
        
        # WORKAROUND: Se è il primo round, usa initial_parameters
        if self.global_model is None:
            from har.task import load_model
            self.global_model = load_model()
            if getattr(self, 'initial_parameters', None) is not None:
                initial_weights = parameters_to_ndarrays(self.initial_parameters)
                self.global_model.set_weights(initial_weights)
        
        current_weights = self.global_model.get_weights()
        
        # Crea modello temporaneo per distillation
        from har.task import load_model
        distill_model = load_model()
        distill_model.set_weights(current_weights)
        
        # Knowledge Distillation Loss: KL-divergence tra aggregated predictions e model predictions
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.distillation_lr)
        
        distillation_history = []
        
        for epoch in range(self.distillation_epochs):
            with tf.GradientTape() as tape:
                # Predizioni del modello su unlabeled data
                model_predictions = distill_model(self.unlabeled_data, training=True)
                
                # KL-divergence loss (knowledge distillation loss)
                kl_loss = tf.keras.losses.KLDivergence()( 
                    aggregated_predictions,
                    model_predictions
                )
            
            # Backpropagation
            gradients = tape.gradient(kl_loss, distill_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, distill_model.trainable_variables))
            
            distillation_history.append(float(kl_loss.numpy()))
        
        avg_distill_loss = np.mean(distillation_history)
        log(INFO, f"   Distillation loss: {avg_distill_loss:.4f}")
        log(INFO, f"   Loss history: {[f'{l:.4f}' for l in distillation_history[:3]]}...")
        
        # Ottieni i nuovi pesi dopo distillation
        new_weights = distill_model.get_weights()
        new_parameters = ndarrays_to_parameters(new_weights)
        
        # Aggiorna anche il modello globale in memoria per i round successivi
        self.global_model.set_weights(new_weights)
        log(INFO, f"✅ Global model updated via knowledge distillation")
        log(INFO, f"{'='*70}\n")
        
        # Aggregazione metriche dai client
        metrics_aggregated = {}
        
        # Aggregazione loss (gestendo sia dict che FitRes)
        losses = []
        for _, fit_res in results:
            res_metrics = fit_res["metrics"] if isinstance(fit_res, dict) else fit_res.metrics
            if "loss" in res_metrics:
                losses.append(res_metrics.get("loss", 0))
        if losses:
            metrics_aggregated["loss"] = float(np.mean(losses))
        
        # Aggregazione accuracy (gestendo sia dict che FitRes)
        accuracies = []
        for _, fit_res in results:
            res_metrics = fit_res["metrics"] if isinstance(fit_res, dict) else fit_res.metrics
            if "accuracy" in res_metrics:
                accuracies.append(res_metrics.get("accuracy", 0))
        if accuracies:
            metrics_aggregated["accuracy"] = float(np.mean(accuracies))
        
        # Aggiungi distillation metrics
        metrics_aggregated["distillation_loss"] = avg_distill_loss
        metrics_aggregated["num_clients_aggregated"] = len(results)
        
        # 🆕 CRITICAL: Call fit_metrics_aggregation_fn for bandwidth stats & client tracking!
        if self.fit_metrics_aggregation_fn is not None:
            # Prepara metrics nel formato atteso: [(num_examples, metrics_dict), ...]
            metrics_for_aggregation = []
            for _, fit_res in results:
                num_ex = fit_res["num_examples"] if isinstance(fit_res, dict) else fit_res.num_examples
                res_metrics = fit_res["metrics"] if isinstance(fit_res, dict) else fit_res.metrics
                metrics_for_aggregation.append((num_ex, res_metrics))
            
            # Chiama la funzione di aggregazione personalizzata
            custom_aggregated = self.fit_metrics_aggregation_fn(metrics_for_aggregation)
            
            # Merge con metrics_aggregated (custom metrics hanno priorità)
            metrics_aggregated.update(custom_aggregated)
        
        return new_parameters, metrics_aggregated
