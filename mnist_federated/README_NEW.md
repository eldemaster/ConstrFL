# 🌸 Federated Learning MNIST con Flower

Sistema di Federated Learning per il dataset MNIST con metriche avanzate e suddivisione corretta dei dati.

Condivide lo stesso stack di ottimizzazioni usato negli altri dataset della tesi (compute-aware partitioning, gradient compression, early stopping). Nessuna distillation è attiva per MNIST.

---

## 📊 Architettura Dataset

### Suddivisione Standard FL

```
MNIST Dataset (70,000 samples)
├── Training Set (48,000 - 68.6%)
│   └── Partizionato tra CLIENT per training federato
│   └── IID o Non-IID distribution
│
├── Validation Set (12,000 - 17.1%)
│   └── Solo SERVER per monitoring durante training
│   └── Early stopping, convergenza
│
└── Test Set (10,000 - 14.3%)
    └── Solo SERVER per valutazione finale
    └── Mai usato durante training (no data leakage)
```

### ✅ Best Practices Implementate

- ✅ Client con SOLO dati di training (no test locale)
- ✅ Server con validation (monitoring) e test (final eval)
- ✅ Nessun data leakage
- ✅ Paradigma FL standard
- ✅ Train/Val/Test split corretto

**Documentazione completa**: [`DATA_SPLIT_GUIDE.md`](DATA_SPLIT_GUIDE.md)

---

## 🚀 Quick Start

### Prerequisiti

```bash
python >= 3.11
tensorflow == 2.15.0
flwr >= 1.22.0
```

### Installazione

```bash
cd mnist_v2
pip install -e .
```

### Training Federato

```bash
# Training normale (20 rounds, 10 clients)
flwr run .

# Risultati salvati automaticamente in results/
```

### Analisi Risultati

```bash
# Analizza ultimo esperimento + grafici
python analyze_results.py

# Salva grafici come PNG
python analyze_results.py --save

# Analizza esperimento specifico
python analyze_results.py --experiment fl_experiment_20251020_123456
```

---

## 📁 Struttura Progetto

```
mnist_v2/
├── mnist/
│   ├── __init__.py
│   ├── client_app.py       # Client (solo training)
│   ├── server_app.py       # Server (evaluation centralizzata)
│   ├── task.py             # Model + data loading
│   ├── data_utils.py       # Partizionamento dataset
│   └── metrics.py          # Metriche avanzate
│
├── results/                # Esperimenti salvati
│   ├── *.json             # Dati strutturati
│   ├── *.csv              # Dati tabulari
│   ├── *_summary.txt      # Riepiloghi
│   └── plots/             # Grafici salvati
│
├── pyproject.toml          # Configurazione Flower
├── analyze_results.py      # Script analisi
├── test_data_split.py      # Verifica dataset split
│
├── DATA_SPLIT_GUIDE.md     # Guida suddivisione dati
├── SUMMARY_DATA_SPLIT.md   # Riepilogo modifiche
├── METRICS_GUIDE.md        # Guida metriche
├── IMPLEMENTATION_SUMMARY.md
└── README.md               # Questo file
```

---

## ⚙️ Configurazione

### `pyproject.toml`

```toml
[tool.flwr.app.config]
num-server-rounds = 20           # Numero di round
local-epochs = 3                 # Epoch locali per client
batch-size = 64                  # Batch size training
fraction-fit = 0.6               # % client per round (60%)
fraction-evaluate = 0.0          # Disabilitato (no test locale)
# experiment-name = "my_exp"     # Nome personalizzato

[tool.flwr.federations.local-simulation]
options.num-supernodes = 10      # Numero di client simulati
```

### Partizionamento

**IID (default)**
```python
# In data_utils.py
partition_type = "iid"  # Distribuzione uniforme delle classi
```

**Non-IID**
```python
# In data_utils.py
partition_type = "non-iid"
classes_per_partition = 2  # Solo 2 classi per client
```

---

## 📊 Metriche Salvate

### Performance
- ✅ Accuracy (centralized su validation)
- ✅ Loss
- ✅ Precision, Recall, F1-Score (macro/weighted)
- ✅ Per-class accuracy (best/worst/std)

### Fairness
- ✅ Fairness gap (differenza tra client)
- ✅ Coefficient of variation
- ✅ Accuracy range

### Data Heterogeneity
- ✅ Gini coefficient
- ✅ Data imbalance ratio
- ✅ Samples distribution

### Convergence
- ✅ Rounds since improvement
- ✅ Best loss/accuracy
- ✅ Plateau detection
- ✅ Early stopping signals

### Efficiency
- ✅ Round time (avg/min/max/std)
- ✅ Total training time
- ✅ Time variance

**Guida completa**: [`METRICS_GUIDE.md`](METRICS_GUIDE.md)

---

## 📈 Grafici Automatici

Lo script `analyze_results.py` genera 8 tipi di grafici:

1. **Accuracy & Loss**: Evoluzione base
2. **Detailed Metrics**: F1, Precision, Recall
3. **Fairness Metrics**: Gap, Gini, CV
4. **Convergence**: Plateau detection
5. **Client-Server Comparison**: Centralized vs Distributed
6. **Efficiency Analysis**: Tempo, sample efficiency
7. **Advanced Metrics**: Per-class, data distribution
8. **Statistical Summary**: Distribuzioni, correlazioni, tabelle

Tutti salvabili in alta risoluzione (300 DPI) con `--save`.

---

## 🧪 Testing

### Verifica Suddivisione Dataset

```bash
python test_data_split.py

# Output:
✅ Training set:    48000 samples (68.6%)
✅ Validation set:  12000 samples (17.1%)
✅ Test set:        10000 samples (14.3%)
✅ Shape corretti: (N, 28, 28, 1)
✅ Normalizzazione OK: [0, 1]
✅ Tutti i client hanno 10 classi (IID)
✅ TUTTI I TEST SUPERATI!
```

### Test Training Veloce

```bash
# Modifica temporaneamente pyproject.toml:
# num-server-rounds = 2
# options.num-supernodes = 2

flwr run .

# Verifica che:
# - Client ricevano solo training data
# - Server usi validation set
# - Nessun evaluate distribuito
```

---

## 📊 Esempio Output Training

### Durante Training (ogni round)

```
======================================================================
ROUND 3 SUMMARY
======================================================================

📊 Centralized Evaluation (Validation Set - 12,000 samples):
  Accuracy:  0.9456
  Loss:      0.1823
  F1-Score:  0.9442
  Precision: 0.9467
  Recall:    0.9434

📊 Training Metrics (Aggregated from Clients):
  Train Accuracy: 0.9234 (range: 0.9012-0.9456)
  Train Loss:     0.2145
  Clients:        6 active
  Total Samples:  28800
  Data Balance:   Gini=0.145 (✅ Balanced)

🔄 Convergence:
  Rounds Since Improvement: 0
  Best Loss: 0.1823 (current)
  Status: ✅ Improving

⚡ Efficiency:
  Round Time: 15.3s
  Total Time: 45.9s

💾 Metriche salvate in:
   JSON: results/fl_experiment_20251020_143022.json
   CSV:  results/fl_experiment_20251020_143022.csv
======================================================================
```

### Dopo Training

```bash
python analyze_results.py --save

📂 Caricando esperimento: fl_experiment_20251020_143022
📊 Dati caricati: 20 rounds

======================================================================
📊 STATISTICHE RIASSUNTIVE
======================================================================

🎯 Performance Finale (Validation Set):
  Best Accuracy:    0.9678 (Round 18)
  Final Accuracy:   0.9654
  Best Loss:        0.1234 (Round 19)
  Final Loss:       0.1289

⚡ Efficienza:
  Avg Round Time:   14.5s
  Total Time:       290.3s

📈 Generando 8 grafici...
💾 Grafici salvati in: results/plots/
✅ Analisi completata!
```

---

## 🔬 Esperimenti Personalizzati

### Cambiare Nome Esperimento

In `pyproject.toml`:
```toml
[tool.flwr.app.config]
experiment-name = "mnist_iid_baseline"
```

### Confrontare Esperimenti

```python
import pandas as pd
import matplotlib.pyplot as plt

# Carica due esperimenti
df1 = pd.read_csv('results/baseline.csv')
df2 = pd.read_csv('results/improved.csv')

# Filtra dal round 1
df1 = df1[df1['round'] >= 1]
df2 = df2[df2['round'] >= 1]

# Confronta accuracy
plt.figure(figsize=(10, 6))
plt.plot(df1['round'], df1['centralized_accuracy'], label='Baseline', marker='o')
plt.plot(df2['round'], df2['centralized_accuracy'], label='Improved', marker='s')
plt.xlabel('Round')
plt.ylabel('Accuracy (Validation)')
plt.title('Comparison: Baseline vs Improved')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Export per Paper (LaTeX)

```python
import pandas as pd

df = pd.read_csv('results/experiment.csv')

# Filtra round interessanti
final_results = df[df['round'].isin([1, 5, 10, 15, 20])]

# Seleziona colonne
export = final_results[[
    'round', 
    'centralized_accuracy', 
    'centralized_loss',
    'centralized_f1_macro',
    'efficiency_avg_round_time'
]]

# Salva per LaTeX
export.to_latex('results_table.tex', index=False, float_format='%.4f')
```

---

## 🎓 Workflow Completo

### 1. Setup
```bash
cd mnist_v2
pip install -e .
python test_data_split.py  # Verifica setup
```

### 2. Training
```bash
# Modifica pyproject.toml se necessario
# (num-server-rounds, batch-size, etc.)

flwr run .
```

### 3. Analisi
```bash
# Analisi base
python analyze_results.py

# Salva grafici
python analyze_results.py --save

# Analisi personalizzata
python  # Apri Python REPL
>>> import pandas as pd
>>> df = pd.read_csv('results/latest.csv')
>>> df[df['round'] >= 1].describe()
```

### 4. Final Test (TODO)
```python
# Script da implementare
from mnist.task import load_model, load_global_test_data

model = load_model()
model.load_weights('final_model.h5')

x_test, y_test = load_global_test_data()  # 10K test samples
# ... evaluate sul test set
```

---

## 📚 Documentazione Completa

- **[DATA_SPLIT_GUIDE.md](DATA_SPLIT_GUIDE.md)**: Guida completa sulla suddivisione dati
- **[SUMMARY_DATA_SPLIT.md](SUMMARY_DATA_SPLIT.md)**: Riepilogo modifiche recenti
- **[METRICS_GUIDE.md](METRICS_GUIDE.md)**: Guida alle metriche
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**: Sistema di logging
- **[ANALYZE.md](ANALYZE.md)**: Template analisi Jupyter

---

## 🐛 Troubleshooting

### Problema: Import errors
```bash
pip install -e .  # Reinstalla
```

### Problema: "No rounds data"
```bash
# Assicurati di aver eseguito almeno un training
flwr run .
```

### Problema: Grafici non partono da round 1
```bash
# Già risolto! analyze_results.py filtra automaticamente
python analyze_results.py
```

### Problema: Client hanno test data
```bash
# Verifica con:
python test_data_split.py

# Se fallisce, reinstalla:
pip install -e .
```

---

## 🔧 Dipendenze

```toml
flwr >= 1.22.0         # Flower framework
tensorflow == 2.15.0   # Deep learning
scikit-learn == 1.6.1  # Metriche ML
pandas >= 1.3.0        # Data analysis
matplotlib >= 3.4.0    # Plotting
seaborn >= 0.11.0      # Statistical plots
numpy >= 1.21.0        # Numerical computing
```

---

## 📊 Statistiche Progetto

- **Linee di codice**: ~2,500
- **File Python**: 8
- **Metriche tracciare**: 30+
- **Grafici automatici**: 8 tipi
- **Formati output**: JSON, CSV, PNG, TXT

---

## 🎯 Caratteristiche

- ✅ Federated Learning standard
- ✅ Train/Val/Test split corretto
- ✅ IID e Non-IID support
- ✅ Metriche avanzate (performance, fairness, convergence, efficiency)
- ✅ Logging automatico (JSON/CSV)
- ✅ Grafici pronti all'uso
- ✅ Early stopping
- ✅ Convergence tracking
- ✅ Export per paper (LaTeX)
- ✅ Testing automatico
- ✅ Documentazione completa

---

## 📧 Info

**Versione**: 2.0 (Data Split Refactoring)  
**Data**: Ottobre 2025  
**Framework**: Flower ≥1.22.0  
**Dataset**: MNIST (70K samples)  
**Status**: ✅ Production Ready

---

## 🚀 Next Steps

1. ⏳ Implementare final test evaluation script
2. ⏳ Salvare best model checkpoint
3. ⏳ Confronto validation vs test metrics
4. ⏳ Multi-run aggregation (mean ± std)
5. ⏳ Hyperparameter tuning support
6. ⏳ More datasets (CIFAR-10, Fashion-MNIST)

---

✨ **Happy Federated Learning!** 🌸
