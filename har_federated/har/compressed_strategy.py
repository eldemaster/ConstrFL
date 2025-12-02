"""
Compressed FedAvg Strategy
==========================

FedAvg con compressione REALE dei gradienti/parametri.

FUNZIONAMENTO:
1. Client comprime parametri → invia compressi
2. Server decomprime → aggrega (FedAvg) → ri-comprime
3. Server invia parametri compressi ai client
"""

from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    FitIns,
    FitRes,
    EvaluateIns,
    Parameters,
    Scalar,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from har.gradient_compression import create_compression_strategy
from har.compressed_parameters import compress_to_parameters, decompress_from_parameters, get_compression_stats


class CompressedFedAvg(FedAvg):
    """
    FedAvg with REAL gradient compression support
    
    ARCHITETTURA:
    - I client comprimono e inviano Parameters standard (con payload compresso)
    - Il server de-comprime e aggrega (FedAvg standard)
    - Il server ri-comprime e invia Parameters compressi ai client
    
    Metadata viene passato tramite FitRes.metrics e FitIns.config
    """
    
    def __init__(
        self,
        compression_type: str = "none",
        compression_num_bits: int = 8,
        compression_k_percent: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Crea strategia di compressione
        self.compression_strategy = create_compression_strategy(
            compression_type=compression_type,
            num_bits=compression_num_bits,
            k_percent=compression_k_percent
        )
        
        self.compression_type = compression_type
        self.total_bandwidth_saved = 0  # Track cumulative savings
        self.last_bandwidth_metrics = {}  # Store last round bandwidth metrics
        
        print(f"\n🔧 Server using REAL compression: {compression_type.upper()}")
        if compression_type == "quantization":
            print(f"   Quantization: {compression_num_bits}-bit")
        elif compression_type == "topk":
            print(f"   Top-K: {compression_k_percent*100:.1f}% sparsity")
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results con de-compressione
        """
        if not results:
            return None, {}
        
        # Se non c'è compressione, usa FedAvg standard
        if self.compression_type == "none":
            return super().aggregate_fit(server_round, results, failures)
        
        # 🔥 DE-COMPRIMI i parametri prima dell'aggregazione
        decompressed_results = []
        total_compressed_bytes = 0
        total_original_bytes = 0
        
        for client, fit_res in results:
            try:
                # Controlla se metrics contengono compression_type (marker)
                metrics = fit_res.metrics or {}
                if 'compression_type' in metrics:
                    # Parametri compressi - decomprimiamo
                    compression_type = metrics['compression_type']
                    
                    # De-comprimi → NDArrays (metadata ricostruito automaticamente!)
                    ndarrays = decompress_from_parameters(
                        fit_res.parameters,
                        compression_type,
                        self.compression_strategy
                    )
                    
                    # Track bandwidth
                    comp_stats = metrics  # già contiene compressed_size_bytes etc
                    total_compressed_bytes += comp_stats.get('compressed_size_bytes', 0)
                    total_original_bytes += comp_stats.get('original_size_bytes', 0)
                    
                    # Ri-crea Parameters standard per aggregazione
                    decompressed_params = ndarrays_to_parameters(ndarrays)
                    
                    # Sostituisci con parametri decompressi
                    new_fit_res = FitRes(
                        status=fit_res.status,
                        parameters=decompressed_params,
                        num_examples=fit_res.num_examples,
                        metrics=fit_res.metrics
                    )
                    
                    decompressed_results.append((client, new_fit_res))
                else:
                    # Parametri standard (no compression)
                    decompressed_results.append((client, fit_res))
                    
            except Exception as e:
                print(f"⚠️  Error decompressing from client: {e}")
                # Fallback: usa parametri come sono
                decompressed_results.append((client, fit_res))
        
        # Log bandwidth savings
        if total_original_bytes > 0:
            saved_bytes = total_original_bytes - total_compressed_bytes
            self.total_bandwidth_saved += saved_bytes
            ratio = total_original_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 1.0
            
            print(f"\n  📊 Round {server_round} Bandwidth:")
            print(f"     Original:  {total_original_bytes:,} bytes")
            print(f"     Compressed: {total_compressed_bytes:,} bytes")
            print(f"     Ratio: {ratio:.2f}x, Saved: {saved_bytes/1024:.1f} KB ({(1-1/ratio)*100:.1f}%)")
            print(f"     Total saved so far: {self.total_bandwidth_saved/1024:.1f} KB\n")
        
        # Aggregazione FedAvg standard sui parametri de-compressi
        aggregated_params, metrics = super().aggregate_fit(
            server_round, decompressed_results, failures
        )
        
        # Aggiungi compression stats alle metriche
        if total_original_bytes > 0:
            metrics["round_bandwidth_original_kb"] = total_original_bytes / 1024
            metrics["round_bandwidth_compressed_kb"] = total_compressed_bytes / 1024
            metrics["round_bandwidth_saved_kb"] = saved_bytes / 1024
            metrics["round_compression_ratio"] = ratio
            metrics["total_bandwidth_saved_kb"] = self.total_bandwidth_saved / 1024
            
            # 🔥 STORE per accesso successivo
            self.last_bandwidth_metrics = {
                "round_bandwidth_original_kb": total_original_bytes / 1024,
                "round_bandwidth_compressed_kb": total_compressed_bytes / 1024,
                "round_bandwidth_saved_kb": saved_bytes / 1024,
                "round_compression_ratio": ratio,
                "total_bandwidth_saved_kb": self.total_bandwidth_saved / 1024,
                "avg_compression_ratio": ratio  # Alias per compatibilità
            }
        
        return aggregated_params, metrics
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure fit con parametri compressi da inviare ai client
        """
        # Get standard config
        config_pairs = super().configure_fit(server_round, parameters, client_manager)
        
        # Se non c'è compressione, return as-is
        if self.compression_type == "none":
            return config_pairs
        
        # 🔥 COMPRIMI i parametri prima di inviarli ai client
        try:
            # Converti parameters → ndarrays
            ndarrays = parameters_to_ndarrays(parameters)
            
            # Comprimi usando helper function
            compressed_params, metadata = compress_to_parameters(
                ndarrays,
                self.compression_strategy
            )
            
            # Sostituisci con parametri compressi in tutte le FitIns
            # Aggiungi compression_type nel config per decompressione client-side
            compressed_pairs = []
            for client, fit_ins in config_pairs:
                # Aggiungi solo compression_type al config (metadata ricostruito dal client!)
                config = dict(fit_ins.config)
                config["compression_type"] = metadata['compression_type']
                
                new_fit_ins = FitIns(
                    parameters=compressed_params,
                    config=config
                )
                compressed_pairs.append((client, new_fit_ins))
            
            return compressed_pairs
            
        except Exception as e:
            print(f"⚠️  Error compressing parameters: {e}")
            return config_pairs
    
    def get_bandwidth_metrics(self) -> Dict[str, float]:
        """Ritorna le bandwidth metrics dell'ultimo round"""
        return self.last_bandwidth_metrics
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        🆕 Configure evaluate - NON comprimere i parametri!
        
        Durante evaluate, inviamo parametri NON compressi ai client
        perché è solo evaluation, non training.
        """
        # Use parent's configure_evaluate WITHOUT compression
        return super().configure_evaluate(server_round, parameters, client_manager)


