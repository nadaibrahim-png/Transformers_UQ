# =============================================================================
# colab_ensemble.py — Deep Ensemble training + evaluation for Colab
# =============================================================================
# Run this AFTER the main training notebook has already run and produced:
#   - train_loader, val_loader, test_loader
#   - A single trained `model` (used as reference for architecture kwargs)
#
# Cells:
#   Cell A — Train Deep Ensemble (M=5 members, ~5× training time)
#   Cell B — Evaluate Ensemble vs MC Dropout side-by-side
#   Cell C — OOD entropy comparison (Ensemble vs MC Dropout)
#   Cell D — Per-class ECE comparison (all 5 methods)
#   Cell E — Epistemic uncertainty decomposition (unique to Ensemble)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL A — Train Deep Ensemble  (run once, ~10-15 min on GPU)
# ─────────────────────────────────────────────────────────────────────────────

import torch
import numpy as np
from evaluation.uncertainty import DeepEnsemble
from models.transformer import FTTransformer
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Architecture kwargs must match your trained single model ──
MODEL_KWARGS = dict(
    input_dim    = config.INPUT_DIM,        # 50
    n_classes    = config.N_CLASSES,        # 2
    d_model      = config.D_MODEL,          # 64
    n_heads      = config.N_HEADS,          # 8
    n_layers     = config.N_LAYERS,         # 2
    d_ff         = config.D_FF,             # 256
    dropout      = config.DROPOUT,          # 0.1
)

ensemble = DeepEnsemble(
    model_class  = FTTransformer,
    model_kwargs = MODEL_KWARGS,
    M            = 5,
)

ensemble.train_all(
    train_loader = train_loader,
    val_loader   = val_loader,
    epochs       = config.EPOCHS,
    lr           = config.LEARNING_RATE,
    seeds        = [0, 1, 2, 3, 4],
    verbose      = True,
)

# Save weights so you don't need to retrain
ensemble.save("/content/drive/MyDrive/ensemble_weights")
print("✓ Ensemble saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL B — Evaluate: Ensemble vs MC Dropout (ECE / NLL / Brier / Balanced Acc)
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
from evaluation.uncertainty import predict_mc_dropout
from evaluation.metrics import compute_all, compute_ece_per_class

# ── MC Dropout ──
print("Running MC Dropout (T=30)...")
mcd_probs, mcd_labels, mcd_entropy = predict_mc_dropout(model, test_loader, n_samples=30)
mcd_metrics = compute_all(mcd_probs, mcd_labels)

# ── Deep Ensemble ──
print("Running Deep Ensemble (M=5)...")
ens_probs, ens_labels, ens_entropy, ens_epistemic = ensemble.predict(test_loader)
ens_metrics = compute_all(ens_probs, ens_labels)

# ── Print comparison table ──
print("\n" + "="*60)
print(f"{'Metric':<22} {'MC Dropout':>15} {'Deep Ensemble':>15}")
print("="*60)
for key in ['ece', 'nll', 'brier', 'balanced_accuracy']:
    mcd_val = mcd_metrics.get(key, float('nan'))
    ens_val = ens_metrics.get(key, float('nan'))
    print(f"{key:<22} {mcd_val:>15.4f} {ens_val:>15.4f}")
print("="*60)

# ── Bar chart ──
metrics_to_plot = ['ece', 'nll', 'brier']
labels_plot     = ['ECE', 'NLL', 'Brier Score']
x = np.arange(len(metrics_to_plot))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 4))
bars1 = ax.bar(x - width/2,
               [mcd_metrics[m] for m in metrics_to_plot],
               width, label='MC Dropout (T=30)', color='#007A78', alpha=0.88)
bars2 = ax.bar(x + width/2,
               [ens_metrics[m] for m in metrics_to_plot],
               width, label='Deep Ensemble (M=5)', color='#004B9B', alpha=0.88)

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels_plot, fontsize=11)
ax.set_ylabel('Score (lower = better)', fontsize=11)
ax.set_title('Epistemic UQ Methods — Calibration Metrics', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0, max(mcd_metrics['nll'], ens_metrics['nll']) * 1.2)
plt.tight_layout()
plt.savefig('fig_ensemble_vs_mcd.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig_ensemble_vs_mcd.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL C — OOD Entropy: Ensemble vs MC Dropout
# ─────────────────────────────────────────────────────────────────────────────

from torch.utils.data import DataLoader, TensorDataset

def make_ood_loader(test_loader, sigma=3.0):
    """Inject Gaussian noise into all features."""
    X_list, y_list = [], []
    for X_batch, y_batch in test_loader:
        X_list.append(X_batch)
        y_list.append(y_batch)
    X = torch.cat(X_list)
    y = torch.cat(y_list)
    X_ood = X + sigma * torch.randn_like(X)
    return DataLoader(TensorDataset(X_ood, y), batch_size=512, shuffle=False)

ood_loader = make_ood_loader(test_loader, sigma=3.0)

# MC Dropout OOD
_, _, mcd_ood_entropy = predict_mc_dropout(model, ood_loader, n_samples=30)

# Ensemble OOD
_, _, ens_ood_entropy, ens_ood_epistemic = ensemble.predict(ood_loader)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
COLORS = {'mcd': '#007A78', 'ens': '#004B9B'}

for ax, (ind_ent, ood_ent, title, color) in zip(axes, [
    (mcd_entropy,  mcd_ood_entropy,  'MC Dropout (T=30)',   COLORS['mcd']),
    (ens_entropy,  ens_ood_entropy,  'Deep Ensemble (M=5)', COLORS['ens']),
]):
    ax.hist(ind_ent, bins=60, alpha=0.65, color=color,     label='In-Distribution', density=True)
    ax.hist(ood_ent, bins=60, alpha=0.55, color='#B91C1C', label='OOD (σ=3.0)',      density=True)
    ax.axvline(np.median(ind_ent), color=color,     linestyle='--', linewidth=1.5,
               label=f'IND median = {np.median(ind_ent):.3f}')
    ax.axvline(np.median(ood_ent), color='#B91C1C', linestyle='--', linewidth=1.5,
               label=f'OOD median = {np.median(ood_ent):.3f}')
    ratio = np.median(ood_ent) / (np.median(ind_ent) + 1e-9)
    ax.set_title(f'{title}\n{ratio:.1f}× entropy separation', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predictive Entropy H(y|x)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

plt.suptitle('OOD Detection via Predictive Entropy', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_ood_ensemble_vs_mcd.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig_ood_ensemble_vs_mcd.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL D — Per-class ECE: all 5 methods side-by-side
# ─────────────────────────────────────────────────────────────────────────────

from evaluation.uncertainty import (
    predict_standard, predict_temperature_scaled, predict_isotonic,
    TemperatureScaler
)

# Standard
std_probs, std_labels = predict_standard(model, test_loader)

# Temperature Scaling
ts = TemperatureScaler(model).to(DEVICE)
ts.fit(val_loader)
ts_probs, ts_labels = predict_temperature_scaled(ts, test_loader)

# Isotonic
iso_probs, iso_labels, _ = predict_isotonic(model, val_loader, test_loader)

# Per-class ECE for all methods
methods = {
    'Standard':       (std_probs,  std_labels),
    'MC Dropout':     (mcd_probs,  mcd_labels),
    'Temp. Scaling':  (ts_probs,   ts_labels),
    'Isotonic':       (iso_probs,  iso_labels),
    'Deep Ensemble':  (ens_probs,  ens_labels),
}
COLORS_5 = {
    'Standard':      '#B91C1C',
    'MC Dropout':    '#007A78',
    'Temp. Scaling': '#166534',
    'Isotonic':      '#6D28D9',
    'Deep Ensemble': '#004B9B',
}

print("\nPer-class ECE:")
print(f"{'Method':<18} {'νₑ ECE':>10} {'νμ ECE':>10}")
print("-"*40)
per_class_eces = {}
for name, (probs, labels) in methods.items():
    pce = compute_ece_per_class(probs, labels)
    per_class_eces[name] = pce
    print(f"{name:<18} {pce.get(0, float('nan')):>10.4f} {pce.get(1, float('nan')):>10.4f}")

# Grouped bar chart
class_names  = ['νₑ signal (class 0)', 'νμ background (class 1)']
method_names = list(methods.keys())
n_methods    = len(method_names)
x            = np.arange(len(class_names))
width        = 0.15

fig, ax = plt.subplots(figsize=(10, 5))
for i, name in enumerate(method_names):
    pce = per_class_eces[name]
    vals = [pce.get(c, 0.0) for c in [0, 1]]
    offset = (i - n_methods/2 + 0.5) * width
    bars = ax.bar(x + offset, vals, width, label=name,
                  color=COLORS_5[name], alpha=0.88)
    for bar in bars:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.0005,
                    f'{bar.get_height():.4f}',
                    ha='center', va='bottom', fontsize=7.5, rotation=45)

ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=11)
ax.set_ylabel('ECE (lower = better)', fontsize=11)
ax.set_title('Per-Class ECE — All 5 UQ Methods', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('fig_perclass_ece_5methods.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig_perclass_ece_5methods.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL E — Epistemic Uncertainty Decomposition (Ensemble only)
# ─────────────────────────────────────────────────────────────────────────────
# Total uncertainty = Aleatoric + Epistemic
# MI(y; θ | x) = H[p̄] − (1/M) Σₘ H[pₘ]

aleatoric = ens_entropy - ens_epistemic     # aleatoric = total − epistemic

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, data, title, color in zip(axes,
    [ens_entropy,   ens_epistemic, aleatoric],
    ['Total Uncertainty\nH[p̄]',
     'Epistemic\nMutual Information',
     'Aleatoric\nH[p̄] − MI'],
    ['#004B9B', '#B91C1C', '#007A78'],
):
    ax.hist(data, bins=60, color=color, alpha=0.8, density=True)
    ax.axvline(np.median(data), color='black', linestyle='--', linewidth=1.5,
               label=f'median={np.median(data):.4f}')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Entropy (nats)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)

plt.suptitle('Deep Ensemble — Uncertainty Decomposition (In-Distribution)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_uncertainty_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig_uncertainty_decomposition.png")

print("\n✓ All ensemble cells complete.")
print("Figures: fig_ensemble_vs_mcd.png | fig_ood_ensemble_vs_mcd.png |",
      "fig_perclass_ece_5methods.png | fig_uncertainty_decomposition.png")
