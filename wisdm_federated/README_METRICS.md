# Advanced Metrics System - WISDM Federated Learning

## 📊 Sistema di Metriche Completo

Questo progetto implementa un sistema avanzato di tracking delle metriche durante il training federato, ispirato al progetto MNIST ma ottimizzato per Human Activity Recognition con WISDM.

## ✨ Features

- ✅ **Metriche Centralizzate**: Accuracy, Loss, F1, Precision, Recall
- ✅ **Metriche Distribuite**: Fairness tra client, variabilità
- ✅ **Data Heterogeneity**: Gini coefficient, CV, imbalance ratio
- ✅ **Convergence Tracking**: Plateau detection, best round tracking
- ✅ **Efficiency Metrics**: Timing per round, velocità convergenza
- ✅ **Per-Class Metrics**: Performance per ogni attività (6 classi WISDM)
- ✅ **Logging Automatico**: JSON + CSV per ogni esperimento
- ✅ **Visualizzazioni**: 6+ grafici automatici post-training

## 🚀 Quick Start

### 1. Training con Metriche

Usa il server avanzato invece di quello base:

```bash
# Invece di:
# flwr run .

# Usa server_app_advanced
# Modifica pyproject.toml o usa:
export FLWR_SERVER_APP=wisdm.server_app_advanced:app
flwr run . --run-config "experiment-name=wisdm_har_test"
```

**Oppure**, sostituisci temporaneamente `server_app.py`:

```bash
cd wisdm/wisdm
cp server_app.py server_app_backup.py
cp server_app_advanced.py server_app.py
cd ../..
flwr run .
```

### 2. Analisi Post-Training

```bash
# Analizza ultimo esperimento
python analyze_results.py

# Analizza esperimento specifico
python analyze_results.py --experiment wisdm_har_test_20231209_143022

# Salva grafici come PNG
python analyze_results.py --save
```

### 3. Visualizza Risultati

```bash
# Summary testuale
cat results/wisdm_har_test_*_summary.txt

# JSON completo
cat results/wisdm_har_test_*.json

# CSV per Excel/Pandas
cat results/wisdm_har_test_*.csv
```

## 📁 Struttura File

```
wisdm/
├── wisdm/
│   ├── metrics.py              # ⭐ Sistema metriche completo
│   ├── server_app_advanced.py  # ⭐ Server con metriche
│   ├── client_app_advanced.py  # ⭐ Client con metriche
│   ├── server_app.py          # Versione base (da sostituire)
│   └── client_app.py          # Versione base
├── analyze_results.py         # ⭐ Script analisi e grafici
├── results/                   # Auto-creata
│   ├── *.json                 # Dati completi
│   ├── *.csv                  # Dati tabellari
│   ├── *_summary.txt          # Summary testuale
│   └── plots/                 # Grafici salvati
├── METRICS_GUIDE.md          # ⭐ Guida completa metriche
└── README_METRICS.md         # Questo file
```

## 📊 Metriche Tracciate

### Durante il Training (ogni round)

**Server (Centralized Evaluation)**:
```
Round 5/10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Centralized Evaluation (Global Test Set):
  Accuracy:  0.8723
  Loss:      0.3421
  F1-Score:  0.8654
  Precision: 0.8712
  Recall:    0.8598

🌐 Distributed Evaluation (Client Test Sets):
  Avg Accuracy:  0.8591
  Accuracy Range: 0.7823 - 0.9012
  Fairness Gap:  0.1189 ⚠️ Check

📈 Convergence:
  Best Round:    4
  No Improvement: 1 rounds

⏱️  Efficiency:
  Avg Round Time: 12.3s
  Total Time:     61.5s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Client (Training Metrics)**:
```
📊 Training Metrics (Round):
  Train Accuracy: 0.8912 (range: 0.8234-0.9456)
  Train Loss:     0.2876
  Clients:        4
  Total Samples:  8543
  Data Balance:   Gini=0.23 ✅ Balanced
```

### File Salvati

**JSON** (`wisdm_har_test_20231209_143022.json`):
```json
{
  "experiment": "wisdm_har_test_20231209_143022",
  "dataset": "WISDM",
  "task": "Human Activity Recognition",
  "rounds": [
    {
      "round": 1,
      "timestamp": "2023-12-09T14:30:22.123456",
      "centralized": {
        "accuracy": 0.7234,
        "loss": 0.6543,
        "f1_macro": 0.7012,
        "precision_macro": 0.7123,
        "recall_macro": 0.6912,
        ...
      },
      "distributed": {
        "distributed_accuracy": 0.7098,
        "fairness_gap": 0.0856,
        ...
      },
      "training": {
        "train_accuracy": 0.7456,
        "data_gini": 0.23,
        ...
      },
      "convergence": {...},
      "efficiency": {...}
    },
    ...
  ]
}
```

**CSV** (importabile in Excel/Pandas):
```csv
round,timestamp,centralized_accuracy,centralized_loss,distributed_accuracy,...
1,2023-12-09T14:30:22,0.7234,0.6543,0.7098,...
2,2023-12-09T14:30:35,0.7845,0.5432,0.7612,...
...
```

## 📈 Grafici Generati

### 1. Accuracy & Loss
- Trend principale nel tempo
- Confronto centralized vs distributed

### 2. Detailed Metrics
- Precision, Recall, F1-Score
- Evolution nel tempo

### 3. Fairness Metrics
- Fairness gap tra client
- Data Gini coefficient
- Training accuracy range
- Round time efficiency

### 4. Convergence Analysis
- Rounds since improvement
- Best loss evolution
- Plateau detection

### 5. Client-Server Comparison
- Centralized vs Distributed accuracy
- Loss comparison
- Overfitting detection (train vs test)
- Client heterogeneity evolution

### 6. Efficiency Analysis
- Round time evolution
- Accuracy vs time (sample efficiency)
- Learning rate (accuracy improvement per round)
- Convergence speed (loss reduction rate)

## 🎯 Use Cases

### 1. Debugging Training

**Problema**: Accuracy non migliora

**Metriche da guardare**:
- Convergence → Plateau detected?
- Training → Overfitting (train >> test)?
- Fairness → Alcuni client performano malissimo?
- Data → Gini molto alto (dati sbilanciati)?

### 2. Ottimizzazione Hyperparameters

**Metriche chiave**:
- Best Round → Quando fermarsi
- Efficiency → Tempo per convergenza
- Fairness → Equità tra client
- Per-Class → Bilanciamento classi

### 3. Confronto Strategie

Confronta metriche tra:
- FedAvg vs FedProx
- IID vs Non-IID partition
- 2 vs 4 vs 8 clients
- Diversi learning rate

**Esempio**:
```bash
# Esperimento 1: FedAvg
flwr run . --run-config "experiment-name=fedavg_test"

# Esperimento 2: FedProx (se implementato)
flwr run . --run-config "experiment-name=fedprox_test"

# Confronta
python analyze_results.py -e fedavg_test_*
python analyze_results.py -e fedprox_test_*
```

### 4. Paper/Report Writing

Le metriche salvate forniscono:
- 📊 Grafici publication-ready (alta risoluzione)
- 📈 Dati numerici precisi (CSV)
- 📄 Summary testuale (copy-paste)
- 🔬 Riproducibilità (JSON completo)

## 🛠️ Personalizzazione

### Modificare Configurazione Metriche

In `pyproject.toml` aggiungi:

```toml
[tool.flwr.app.config]
num-server-rounds = 10
local-epochs = 3
batch-size = 32
experiment-name = "wisdm_custom"
num-classes = 6  # WISDM
verbose = false
```

### Cambiare Soglie

In `wisdm/wisdm/metrics.py`:

```python
# Convergence patience
convergence_tracker = ConvergenceTracker(
    patience=5,      # Cambia qui
    min_delta=0.001  # Cambia qui
)

# Fairness threshold (nel print)
if fairness_gap < 0.05:  # Cambia qui
    print("✅ Good")
```

### Aggiungere Nuove Metriche

1. **Calcolo** in `metrics.py`:
```python
def my_new_metric(model, x, y):
    # Calcola metrica custom
    return value
```

2. **Aggregazione** in `server_app_advanced.py`:
```python
metrics.update({
    "my_metric": my_new_metric(model, x_test, y_test)
})
```

3. **Visualizzazione** in `analyze_results.py`:
```python
def plot_my_metric(df):
    plt.plot(df['round'], df['centralized_my_metric'])
    plt.title('My Custom Metric')
```

## 📚 Differenze vs MNIST

| Feature | MNIST | WISDM HAR |
|---------|-------|-----------|
| **Num Classes** | 10 (digits) | 6 (activities) |
| **Input Shape** | (28, 28, 1) | (200, 3) |
| **Task** | Image Classification | Time Series Classification |
| **Model** | CNN 2D | CNN 1D |
| **Per-Class** | Digits 0-9 | Walking, Jogging, etc. |
| **Challenge** | Visual recognition | Temporal patterns |

**Metriche comuni**:
- ✅ Accuracy, Loss, F1
- ✅ Fairness, Convergence
- ✅ Efficiency tracking

**Metriche specifiche WISDM**:
- 🔍 Focus su attività fisiche (6 classi)
- 🔍 Temporal window analysis
- 🔍 Sensor data quality

## 🔧 Troubleshooting

### Problema: Metriche non vengono salvate

**Soluzione**: Verifica di usare `server_app_advanced.py`:

```bash
cd wisdm/wisdm
ls -la server_app*.py
# Se esiste solo server_app.py (versione base), hai due opzioni:

# Opzione 1: Rinomina (backup vecchio)
mv server_app.py server_app_old.py
mv server_app_advanced.py server_app.py

# Opzione 2: Modifica pyproject.toml
# [tool.flwr.app.components]
# serverapp = "wisdm.server_app_advanced:app"
```

### Problema: Grafici non si generano

**Causa**: Mancano dipendenze

```bash
pip install pandas matplotlib seaborn scikit-learn
```

### Problema: CSV corrotto

**Soluzione**: Viene riparato automaticamente dal JSON:

```bash
python analyze_results.py
# Lo script rileva e ripara CSV automaticamente
```

### Problema: Troppi esperimenti in results/

**Pulizia**:

```bash
# Mantieni solo gli ultimi 5
cd results
ls -t *.json | tail -n +6 | xargs rm
ls -t *.csv | tail -n +6 | xargs rm
```

## 📖 Documentation Links

- **Guida Completa Metriche**: `METRICS_GUIDE.md`
- **Implementation Details**: `IMPLEMENTATION_SUMMARY.md` (se presente)
- **Flower Docs**: https://flower.dev/docs/
- **WISDM Dataset**: https://www.cis.fordham.edu/wisdm/

## 🎓 Best Practices

1. **Nomina gli esperimenti**:
   ```bash
   flwr run . --run-config "experiment-name=test_lr001"
   ```

2. **Salva sempre i grafici** per confronti:
   ```bash
   python analyze_results.py --save
   ```

3. **Documenta configurazioni** nel summary:
   - Learning rate
   - Num clients
   - Partitioning strategy

4. **Monitora fairness**: In FL reale è cruciale

5. **Early stopping**: Se plateau detected, ferma

## 💡 Tips & Tricks

- **Fast iteration**: Usa pochi rounds per debug, poi aumenta
- **Baseline first**: Training centralizzato come confronto
- **Log everything**: Meglio troppi dati che troppo pochi
- **Version control**: Git committa i JSON dei risultati
- **Reproduce**: JSON + config → risultati riproducibili

## 🚀 Next Steps

1. ✅ Sistema metriche funzionante
2. 📊 Analisi baseline (training centralizzato)
3. 🔬 Esperimenti con configurazioni diverse
4. 📈 Confronto IID vs Non-IID
5. 📝 Write-up risultati

---

**Domande?** Vedi `METRICS_GUIDE.md` per dettagli tecnici.

**Ready to start?**
```bash
flwr run . --run-config "experiment-name=first_test"
python analyze_results.py --save
```
