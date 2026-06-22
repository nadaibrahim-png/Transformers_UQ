# Toward Robust Deep Learning
## Evaluating Uncertainty in Transformer-Based Architectures for Scientific Data
**Nada Elhassan — University of Messina | ICTP Poster, July 2026**

---

## Project Structure

```
ictp-poster-2026/
│
├── config.py               ← All settings in one place (change dataset, hyperparams here)
├── main.py                 ← Entry point: runs full experiment end-to-end
├── requirements.txt
│
├── data/
│   └── loader.py           ← Dataset loading, preprocessing, OOD creation
│
├── models/
│   └── transformer.py      ← TabTransformer architecture with MC Dropout head
│
├── training/
│   └── trainer.py          ← Training loop with early stopping
│
├── evaluation/
│   ├── metrics.py          ← ECE, NLL, Accuracy, reliability diagram data
│   └── uncertainty.py      ← Standard / MC Dropout / Temperature Scaling inference
│
├── visualization/
│   └── plots.py            ← 5 poster-quality figures
│
└── figures/                ← Output figures (auto-created on first run)
```

---

## Quickstart (Google Colab)

```python
# Cell 1 — Install
!pip install netcal --quiet

# Cell 2 — Clone your repo
!git clone https://github.com/YOUR_USERNAME/ictp-poster-2026.git
%cd ictp-poster-2026

# Cell 3 — Run everything
from main import run
results, history = run()
```

---

## Switching Datasets

Edit `config.py`:
```python
DATASET = "magic"   # "magic" | "higgs" | "wine"
```

| Dataset | Domain | Samples | Features | Task |
|---------|--------|---------|----------|------|
| `magic` | Astroparticle physics | 19,020 | 10 | Binary |
| `higgs` | Particle physics (HEP) | 50k subset | 28 | Binary |
| `wine`  | Chemistry (fast debug) | 178 | 13 | 3-class |

---

## Paper References

- **[GAL16]** Gal & Ghahramani. *Dropout as a Bayesian Approximation.* ICML 2016. arXiv:1506.02142
- **[GUO17]** Guo et al. *On Calibration of Modern Neural Networks.* ICML 2017. arXiv:1706.04599
- **[LAK17]** Lakshminarayanan et al. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017. arXiv:1612.01474
- **[VAS17]** Vaswani et al. *Attention is All You Need.* NeurIPS 2017.
