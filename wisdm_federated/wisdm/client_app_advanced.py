"""
WISDM: Flower ClientApp with Enhanced Metrics
Versione migliorata con metriche dettagliate
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from wisdm.task import load_model, load_data

# Flower ClientApp
app = ClientApp()

# Cache del modello e dei dati
_cached_model = None
_cached_data = {}


def get_model():
    """Ottieni il modello (con cache)"""
    global _cached_model
    if _cached_model is None:
        _cached_model = load_model()
    return _cached_model


def get_data(partition_id, num_partitions):
    """Ottieni i dati (con cache)"""
    global _cached_data
    cache_key = f"{partition_id}_{num_partitions}"
    
    if cache_key not in _cached_data:
        x_train, y_train, x_test, y_test = load_data(
            partition_id, 
            num_partitions
        )
        _cached_data[cache_key] = (x_train, y_train, x_test, y_test)
    
    return _cached_data[cache_key]


class WISDMClient(NumPyClient):
    """Client per WISDM dataset con metriche avanzate"""
    
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    
    def fit(self, parameters, config):
        """Training locale con metriche dettagliate"""
        # Setta i parametri ricevuti dal server
        self.model.set_weights(parameters)
        
        # Ottieni iperparametri dal config
        epochs = config.get("local-epochs", 1)
        batch_size = config.get("batch-size", 32)
        verbose = config.get("verbose", 0)
        
        # Training
        history = self.model.fit(
            self.x_train, 
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        # Restituisci parametri aggiornati
        parameters_updated = self.model.get_weights()
        num_examples = len(self.x_train)
        
        # Metriche da inviare al server
        metrics = {
            "train_loss": float(history.history["loss"][-1]),
            "train_acc": float(history.history["accuracy"][-1]),
        }
        
        return parameters_updated, num_examples, metrics
    
    def evaluate(self, parameters, config):
        """Valutazione locale con metriche dettagliate"""
        self.model.set_weights(parameters)
        
        loss, accuracy = self.model.evaluate(
            self.x_test, 
            self.y_test,
            verbose=0
        )
        
        num_examples = len(self.x_test)
        
        metrics = {
            "eval_loss": float(loss),
            "eval_acc": float(accuracy),
        }
        
        return loss, num_examples, metrics


def client_fn(context: Context):
    """Factory function per creare il client"""
    
    # Ottieni configurazione dal context
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config.get("num-partitions", 2)
    
    # Carica i dati (cached)
    x_train, y_train, x_test, y_test = get_data(partition_id, num_partitions)
    
    # Crea il modello (cached)
    model = get_model()
    
    # Crea e restituisci il client
    return WISDMClient(model, x_train, y_train, x_test, y_test).to_client()


# Usa client_fn
app = ClientApp(client_fn=client_fn)
