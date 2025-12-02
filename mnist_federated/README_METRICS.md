# 📊 Sistema di Logging e Analisi Metriche - Quick Start

## 🎯 In Breve

Ora **tutte le metriche del training federato vengono salvate automaticamente** in:
- 📄 **JSON**: Dati strutturati per analisi programmatiche
- 📊 **CSV**: Formato tabulare per pandas/Excel
- 📝 **TXT**: Summary testuale con statistiche finali

## 🚀 Quick Start

### 1. Esegui il Training

```bash
cd /home/ubuntu/MNISTtestSuperlink/mnist
flwr run .
```

Le metriche vengono salvate automaticamente in `results/`:
```
results/
├── fl_experiment_20251009_143022.json
├── fl_experiment_20251009_143022.csv
└── fl_experiment_20251009_143022_summary.txt
```

### 2. Analizza i Risultati

```bash
# Analizza l'ultimo esperimento con grafici interattivi
python analyze_results.py

# Salva anche i grafici come immagini PNG
python analyze_results.py --save
```

**Output:**
- ✅ Statistiche riassuntive nel terminale
- 📈 4 grafici interattivi (Accuracy/Loss, Detailed, Fairness, Convergence)
- 💾 (Opzionale) Grafici salvati in `results/plots/` come PNG ad alta risoluzione

### 3. Test con Dati Fittizi

```bash
# Genera dati di esempio per testare l'analisi
python generate_sample_data.py --rounds 10

# Analizza i dati generati
python analyze_results.py
```

## 📚 Documentazione Completa

- **`IMPLEMENTATION_SUMMARY.md`** - Riepilogo completo dell'implementazione
- **`METRICS_GUIDE.md`** - Guida dettagliata con esempi avanzati
- **`ANALYZE.md`** - Template per Jupyter Notebook con blocchi di codice pronti

## 📊 Cosa Viene Salvato

**Performance:**
- Accuracy, Loss, F1-Score, Precision, Recall
- Metriche per classe (best/worst)

**Fairness:**
- Fairness gap tra client
- Variance e range di accuracy

**Data Heterogeneity:**
- Gini coefficient
- Data imbalance ratio

**Convergence:**
- Rounds since improvement
- Best round detection
- Plateau detection

**Efficiency:**
- Round time (avg/min/max)
- Total training time

## 💡 Tips

### Nome Personalizzato Esperimento

Modifica `pyproject.toml`:
```toml
[tool.flwr.app.config]
experiment-name = "mnist_baseline"  # Nome descrittivo
```

### Analisi con Pandas

```python
import pandas as pd

df = pd.read_csv('results/latest.csv')
print(f"Best accuracy: {df['centralized_accuracy'].max():.4f}")
print(f"At round: {df.loc[df['centralized_accuracy'].idxmax(), 'round']}")
```

### Confronto Esperimenti

```python
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('results/baseline.csv')
df2 = pd.read_csv('results/improved.csv')

plt.plot(df1['round'], df1['centralized_accuracy'], label='Baseline')
plt.plot(df2['round'], df2['centralized_accuracy'], label='Improved')
plt.legend()
plt.show()
```

## 🎨 Grafici Disponibili

1. **Accuracy & Loss**: Evoluzione centralizzata e distribuita
2. **Detailed Metrics**: Accuracy, F1, Precision, Recall
3. **Fairness & Heterogeneity**: Gap, Gini, Range, Time
4. **Convergence**: Tracking miglioramenti

## ✅ Checklist

- [x] Sistema di logging implementato
- [x] Salvataggio automatico JSON/CSV
- [x] Script di analisi con grafici
- [x] Documentazione completa
- [x] Esempi e test
- [x] Dipendenze installate (pandas, matplotlib, seaborn)

## 🔧 Comandi Utili

```bash
# Training
flwr run .

# Analisi
python analyze_results.py
python analyze_results.py --save  # Con salvataggio grafici
python analyze_results.py --experiment <nome>  # Esperimento specifico

# Test
python generate_sample_data.py --rounds 10

# Verifica file generati
ls -lh results/
```

## 📞 Help

```bash
python analyze_results.py --help
python generate_sample_data.py --help
```

---

**Pronto per il training federato con analisi automatica! 🚀**
