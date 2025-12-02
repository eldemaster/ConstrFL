# WISDM Federated Learning - Sistema Completo con Metriche Avanzate

## 🎯 Overview

Questo progetto implementa un sistema di **Federated Learning** per il riconoscimento di attività umane (HAR - Human Activity Recognition) usando il dataset **WISDM**, con un **sistema avanzato di metriche** ispirato al progetto MNIST.
Condivide lo stesso stack di ottimizzazioni applicato agli altri dataset (compute-aware partitioning, gradient compression, early stopping); la distillation non è prevista qui.

### Features Principali

- ✅ **Federated Learning** con Flower Framework
- ✅ **Metriche Avanzate**: Accuracy, F1, Precision, Recall, Fairness, Convergence, Efficiency
- ✅ **Logging Automatico**: JSON + CSV per ogni esperimento
- ✅ **Analisi Post-Training**: 6 grafici automatici + summary testuale
- ✅ **Data Heterogeneity Tracking**: Gini coefficient, CV, imbalance ratio
- ✅ **Convergence Detection**: Plateau detection e early stopping suggestions
- ✅ **Fairness Monitoring**: Gap tra performance dei client

## 📊 Dataset WISDM

**WISDM (Wireless Sensor Data Mining)**:
- **Task**: Human Activity Recognition
- **Input**: Accelerometer data (x, y, z) da smartphone
- **Classes**: 6 attività
  - 0: Walking
  - 1: Jogging
  - 2: Upstairs
  - 3: Downstairs
  - 4: Sitting
  - 5: Standing
- **Window Size**: 200 time steps
- **Features**: 3 (x, y, z acceleration)

## 🚀 Quick Start

### 1. Setup Dataset

```bash
cd /home/ubuntu/WISDMtestSuperlink/wisdm

# Scarica e preprocessa il dataset WISDM
python setup_dataset.py

# Questo crea:
# - data/wisdm/partitions/partition_0.npz
# - data/wisdm/partitions/partition_1.npz
# - ... (2+ partizioni per default)
```

### 2. Attiva Sistema Metriche Avanzate

```bash
# Attiva il sistema con metriche complete
./activate_advanced_metrics.sh

# Questo:
# - Fa backup dei file originali
# - Attiva server_app_advanced.py e client_app_advanced.py
# - Verifica dipendenze
# - Crea directory results/
```

### 3. Training Federato

```bash
# Test veloce (2 rounds)
flwr run . --run-config "num-server-rounds=2 experiment-name=quick_test"

# Training completo (10+ rounds)
flwr run . --run-config "num-server-rounds=10 experiment-name=baseline local-epochs=3"
```

**Output durante training**:
```
🚀 Starting Federated Learning - WISDM HAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rounds:        10
Local Epochs:  3
Batch Size:    32
Num Classes:   6 (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing)
Experiment:    baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💾 Metriche salvate in:
   JSON: results/baseline_20231209_143022.json
   CSV:  results/baseline_20231209_143022.csv

[Round 1/10]
📊 Centralized Evaluation (Global Test Set):
  Accuracy:  0.7234
  F1-Score:  0.7012
  ...
```

### 4. Analisi Risultati

```bash
# Analisi interattiva dell'ultimo esperimento
python analyze_results.py

# Con salvataggio grafici
python analyze_results.py --save

# Esperimento specifico
python analyze_results.py --experiment baseline_20231209_143022 --save
```

**Output**:
```
📊 STATISTICHE RIASSUNTIVE - WISDM HAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Performance Finale:
  Best Accuracy:    0.8734 (Round 8)
  Final Accuracy:   0.8701
  Best F1-Score:    0.8612
  ...

📈 Generando grafici...
  [1/6] Accuracy & Loss base...
  [2/6] Metriche dettagliate...
  [3/6] Fairness metrics...
  [4/6] Convergence analysis...
  [5/6] Client-Server comparison...
  [6/6] Efficiency analysis...

✅ Analisi completata!
💾 Grafici salvati in: results/plots/
```

## 📁 Struttura Progetto

```
wisdm/
├── wisdm/                          # Package principale
│   ├── __init__.py
│   ├── task.py                     # Model & data loading
│   ├── wisdm_dataset.py            # Dataset preprocessing
│   ├── metrics.py                  # ⭐ Sistema metriche completo
│   ├── server_app.py               # Server base (backup)
│   ├── server_app_advanced.py      # ⭐ Server con metriche
│   ├── client_app.py               # Client base (backup)
│   └── client_app_advanced.py      # ⭐ Client con metriche
│
├── data/                           # Dataset (auto-generato)
│   └── wisdm/
│       └── partitions/
│           ├── partition_0.npz
│           └── partition_1.npz
│
├── results/                        # Risultati esperimenti (auto-generato)
│   ├── *.json                      # Dati completi
│   ├── *.csv                       # Dati tabulari
│   ├── *_summary.txt               # Summary testuali
│   └── plots/                      # Grafici PNG
│
├── models/                         # Modelli salvati (opzionale)
│   └── wisdm_final_model.keras
│
├── setup_dataset.py                # ⭐ Setup automatico dataset
├── analyze_results.py              # ⭐ Analisi e grafici
├── activate_advanced_metrics.sh    # ⭐ Attiva sistema metriche
├── restore_original.sh             # Ripristina file originali
│
├── pyproject.toml                  # Configurazione Flower
├── README.md                       # Questo file
├── README_METRICS.md               # ⭐ Guida metriche (user)
├── METRICS_GUIDE.md                # ⭐ Guida metriche (tecnica)
├── IMPLEMENTATION_SUMMARY.md       # ⭐ Dettagli implementazione
└── COMPARISON_GUIDE.md             # ⭐ Confronto MNIST vs WISDM
```

## 🎓 Documentazione

### Per Utenti
- **README.md** (questo file): Quick start e overview
- **README_METRICS.md**: Guida completa al sistema di metriche
- **METRICS_GUIDE.md**: Interpretazione metriche e best practices

### Per Sviluppatori
- **IMPLEMENTATION_SUMMARY.md**: Dettagli architetturali e workflow
- **COMPARISON_GUIDE.md**: Confronto con MNIST, riuso del codice

## 📊 Metriche Tracciate

### Performance (Centralized)
- **Accuracy**: Proporzione predizioni corrette
- **Loss**: Cross-entropy loss
- **F1-Score**: Media armonica precision/recall
- **Precision/Recall**: Per-class e weighted

### Fairness (Distributed)
- **Fairness Gap**: Differenza tra miglior e peggior client
- **Eval Accuracy Range**: Min-Max tra client
- **Coefficient of Variation**: Dispersione relativa

### Data Heterogeneity
- **Gini Coefficient**: Bilanciamento dati (0=perfetto, 1=tutto su 1 client)
- **Data CV**: Variabilità numero sample
- **Imbalance Ratio**: Max/Min samples

### Convergence
- **Best Round**: Round con miglior loss
- **Rounds Since Improvement**: Contatore plateau
- **Plateau Detection**: Boolean per early stopping

### Efficiency
- **Round Time**: Tempo per round
- **Total Time**: Tempo totale training
- **Convergence Speed**: Velocità riduzione loss

### Per-Class (WISDM)
- **Worst/Best Class Accuracy**: Range performance tra attività
- **Class Accuracy Std**: Variabilità tra classi

## 🎯 Use Cases

### 1. Training Baseline
```bash
# Run standard
flwr run . --run-config "num-server-rounds=10 experiment-name=baseline"

# Analisi
python analyze_results.py --save

# Review
cat results/baseline_*_summary.txt
```

### 2. Hyperparameter Tuning
```bash
# Esperimento 1: Learning rate 0.001
flwr run . --run-config "experiment-name=lr001"

# Esperimento 2: Learning rate 0.0001
# (modifica task.py e rirun)
flwr run . --run-config "experiment-name=lr0001"

# Confronto
python analyze_results.py -e lr001_* --save
python analyze_results.py -e lr0001_* --save
```

### 3. Fairness Analysis
```bash
# Training con metriche fairness attive
flwr run . --run-config "experiment-name=fairness_test"

# Analizza fairness gap
python analyze_results.py -e fairness_test_*

# Se gap alto (>0.10), considera:
# - Ribilanciamento partizioni
# - FedProx invece di FedAvg
# - Data augmentation
```

### 4. Convergence Study
```bash
# Training con plateau detection
flwr run . --run-config "num-server-rounds=30 experiment-name=convergence"

# Analisi convergenza
python analyze_results.py -e convergence_*

# Trova best round e ferma prima per risparmiare tempo
```

## 🔧 Configurazione

### pyproject.toml

```toml
[tool.flwr.app.config]
num-server-rounds = 10      # Numero round
local-epochs = 3            # Epoch locali per client
batch-size = 32             # Batch size
verbose = false             # Verbosity training
num-classes = 6             # Classi WISDM (non modificare)
experiment-name = "test"    # Nome esperimento
```

### Run-time Override

```bash
# Override configurazione al run
flwr run . --run-config "
  num-server-rounds=20
  local-epochs=5
  batch-size=64
  experiment-name=custom_config
"
```

## 🛠️ Troubleshooting

### Problema: "No module named 'wisdm.metrics'"
**Causa**: Sistema avanzato non attivato
**Fix**:
```bash
./activate_advanced_metrics.sh
```

### Problema: "Partition not found"
**Causa**: Dataset non preparato
**Fix**:
```bash
python setup_dataset.py
```

### Problema: Grafici non si generano
**Causa**: Dipendenze visualizzazione mancanti
**Fix**:
```bash
pip install matplotlib seaborn pandas scikit-learn
```

### Problema: Metriche CSV corrotte
**Causa**: Interruzione durante scrittura
**Fix**: Script ripara automaticamente da JSON:
```bash
python analyze_results.py  # Ripara e analizza
```

### Problema: Fairness gap molto alto (>0.20)
**Causa**: Dati client troppo eterogenei
**Analisi**:
1. Controlla Gini coefficient (se >0.5, ribilancia)
2. Verifica per-class accuracy (alcune attività difficili?)
3. Considera strategie diverse (FedProx, FedNova)

### Problema: Plateau immediato (round 2-3)
**Causa**: Learning rate troppo basso o modello inadeguato
**Fix**:
1. Aumenta learning rate in `task.py`
2. Aumenta local epochs
3. Verifica architettura modello

## 📈 Best Practices

### 1. Naming Convention
```bash
# Usa nomi descrittivi che includono configurazione
flwr run . --run-config "experiment-name=wisdm_fedavg_iid_lr001_e3"
# Format: dataset_strategy_partition_lr_epochs
```

### 2. Incremental Development
```bash
# Step 1: Test veloce
flwr run . --run-config "num-server-rounds=2 experiment-name=test"

# Step 2: Short run
flwr run . --run-config "num-server-rounds=5 experiment-name=dev"

# Step 3: Full run
flwr run . --run-config "num-server-rounds=20 experiment-name=final"
```

### 3. Version Control Risultati
```bash
# Committa JSON e grafici importanti
git add results/baseline_*_summary.txt
git add results/plots/baseline_*.png
git commit -m "Add baseline experiment results"
```

### 4. Documentazione Esperimenti
```markdown
# experiments_log.md
## Baseline - 2023-12-09
- Config: 10 rounds, 3 epochs, batch=32
- Best accuracy: 0.8734 (round 8)
- Fairness gap: 0.0456 (good)
- Gini: 0.23 (balanced)
- Notes: Good baseline, consider more rounds
```

## 🔬 Advanced Topics

### Custom Metrics
Aggiungi metriche personalizzate in `wisdm/metrics.py`:

```python
def my_custom_metric(model, x_test, y_test):
    # Calcola metrica custom
    predictions = model.predict(x_test)
    return custom_calculation(predictions, y_test)

# In evaluate_model_detailed, aggiungi:
metrics['my_metric'] = my_custom_metric(model, x_test, y_test)
```

### Different Strategies
Modifica `server_app_advanced.py` per usare FedProx, FedAdam, etc.:

```python
from flwr.server.strategy import FedProx

strategy = FedProx(
    proximal_mu=0.1,  # Proximal term
    ...
)
```

### Non-IID Partitioning
Modifica `wisdm_dataset.py` per creare partizioni non-IID basate su classi:

```python
# Esempio: Client 0 vede più Walking, Client 1 più Jogging
```

## 🚀 Next Steps

1. ✅ Setup dataset: `python setup_dataset.py`
2. ✅ Attiva metriche: `./activate_advanced_metrics.sh`
3. ✅ Test veloce: `flwr run . --run-config "num-server-rounds=2"`
4. ✅ Baseline: `flwr run . --run-config "num-server-rounds=10 experiment-name=baseline"`
5. ✅ Analisi: `python analyze_results.py --save`
6. 📊 Esperimenti varianti (lr, epochs, clients)
7. 📝 Documentazione risultati
8. 📈 Confronto con letteratura

## 📚 Riferimenti

- **Flower Framework**: https://flower.dev/
- **WISDM Dataset**: https://www.cis.fordham.edu/wisdm/dataset.php
- **Federated Learning**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- **Fairness in FL**: Mohri et al., "Agnostic Federated Learning" (2019)

## 🤝 Credits

- **MNIST Baseline**: Sistema metriche originale da `/home/ubuntu/MNISTtestSuperlink/mnist/`
- **WISDM Adaptation**: Porting e ottimizzazione per HAR
- **Flower Framework**: https://flower.dev/

---

## 📞 Support

**Domande su**:
- Metriche → Vedi `METRICS_GUIDE.md`
- Setup → Vedi `README_METRICS.md`
- Implementazione → Vedi `IMPLEMENTATION_SUMMARY.md`
- Confronto MNIST → Vedi `COMPARISON_GUIDE.md`

**Quick Help**:
```bash
# Lista file documentazione
ls -1 *.md

# Cerca in tutta la doc
grep -r "keyword" *.md
```

---

**Ready to start?**
```bash
./activate_advanced_metrics.sh
flwr run . --run-config "experiment-name=first_run"
python analyze_results.py --save
```

🎉 **Happy Federated Learning!**
