# =============================================================================
# colab_swag.py — SWAG training + full 3-way epistemic UQ comparison
# =============================================================================
# Run AFTER the main notebook (model already trained) AND colab_ensemble.py.
# Requires: model, train_loader, val_loader, test_loader already in scope,
#           plus mcd_probs/mcd_entropy and ens_probs/ens_entropy from ensemble.
#
# Cells:
#   Cell A — SWAG collection phase (~5 min)
#   Cell B — SWAG inference on test set
#   Cell C — 3-way calibration table (MC Dropout / Ensemble / SWAG)
#   Cell D — 3-way OOD entropy histograms
#   Cell E — Per-class ECE: all 6 methods
#   Cell F — Epistemic uncertainty comparison (violin plots)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# CELL A — SWAG collection phase
# ─────────────────────────────────────────────────────────────────────────────

import torch
import numpy as np
import matplotlib.pyplot as plt
from evaluation.uncertainty import SWAG
from evaluation.metrics import compute_all, compute_ece_per_class

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Important: SWAG continues training FROM your already-trained model.
# Do NOT re-initialise model weights.
swag = SWAG(model, max_snapshots=20)

swag.collect_during_training(
    train_loader  = train_loader,
    val_loader    = val_loader,
    epochs        = 20,          # 20 high-LR epochs → 20 snapshots
    lr            = 5e-4,        # higher than Adam LR to explore the basin
    collect_every = 1,
    verbose       = True,
)

# Save SWAG statistics (no need to retrain next time)
swag.save("/content/drive/MyDrive/swag_stats.pt")
print("✓ SWAG ready.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL B — SWAG inference
# ─────────────────────────────────────────────────────────────────────────────

print("Running SWAG inference (K=30 samples)...")
swag_probs, swag_labels, swag_entropy = swag.predict(test_loader, K=30)
swag_metrics = compute_all(swag_probs, swag_labels)

print(f"\nSWAG results:")
for k, v in swag_metrics.items():
    print(f"  {k:<22} {v:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL C — 3-way calibration comparison table + bar chart
# ─────────────────────────────────────────────────────────────────────────────
# Requires mcd_metrics and ens_metrics from colab_ensemble.py Cell B

from evaluation.uncertainty import predict_mc_dropout
from evaluation.metrics import compute_all

# Re-run MC Dropout if not already in scope
try:
    mcd_metrics
except NameError:
    print("Re-running MC Dropout...")
    mcd_probs, mcd_labels, mcd_entropy = predict_mc_dropout(model, test_loader, n_samples=30)
    mcd_metrics = compute_all(mcd_probs, mcd_labels)

try:
    ens_metrics
except NameError:
    print("Re-running Ensemble (load from Drive)...")
    from evaluation.uncertainty import DeepEnsemble
    from models.transformer import FTTransformer
    import config
    ensemble = DeepEnsemble(FTTransformer, dict(
        input_dim=config.INPUT_DIM, n_classes=config.N_CLASSES,
        d_model=config.D_MODEL, n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS, d_ff=config.D_FF, dropout=config.DROPOUT,
    ), M=5)
    ensemble.load("/content/drive/MyDrive/ensemble_weights")
    ens_probs, ens_labels, ens_entropy, ens_epistemic = ensemble.predict(test_loader)
    ens_metrics = compute_all(ens_probs, ens_labels)

# ── Print comparison table ──
methods_3 = {
    'MC Dropout (T=30)':   mcd_metrics,
    'Deep Ensemble (M=5)': ens_metrics,
    'SWAG (T=20, K=30)':   swag_metrics,
}
COLORS_3 = ['#007A78', '#004B9B', '#6D28D9']

print("\n" + "="*65)
print(f"{'Metric':<22} {'MC Dropout':>14} {'Ensemble':>14} {'SWAG':>14}")
print("="*65)
for key in ['ece', 'nll', 'brier', 'balanced_accuracy']:
    vals = [m.get(key, float('nan')) for m in methods_3.values()]
    best = min(vals) if key != 'balanced_accuracy' else max(vals)
    row  = f"{key:<22}"
    for v in vals:
        marker = " ✓" if abs(v - best) < 1e-6 else "  "
        row += f" {v:>12.4f}{marker}"
    print(row)
print("="*65)

# ── Bar chart ──
metric_keys   = ['ece', 'nll', 'brier']
metric_labels = ['ECE', 'NLL', 'Brier Score']
x     = np.arange(len(metric_keys))
width = 0.26

fig, ax = plt.subplots(figsize=(9, 4.5))
for i, (name, metrics, color) in enumerate(
    zip(methods_3.keys(), methods_3.values(), COLORS_3)
):
    offset = (i - 1) * width
    bars = ax.bar(x + offset,
                  [metrics[k] for k in metric_keys],
                  width, label=name, color=color, alpha=0.88)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f'{bar.get_height():.4f}',
                ha='center', va='bottom', fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylabel('Score (lower = better)', fontsize=11)
ax.set_title('Epistemic UQ Methods — Calibration Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig_3way_calibration.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig_3way_calibration.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL D — 3-way OOD entropy histograms
# ─────────────────────────────────────────────────────────────────────────────

from torch.utils.data import DataLoader, TensorDataset

def make_ood_loader(loader, sigma=3.0):
    X_list, y_list = [], []
    for X_batch, y_batch in loader:
        X_list.append(X_batch); y_list.append(y_batch)
    X = torch.cat(X_list)
    y = torch.cat(y_list)
    return DataLoader(TensorDataset(X + sigma * torch.randn_like(X), y),
                      batch_size=512, shuffle=False)

ood_loader = make_ood_loader(test_loader, sigma=3.0)

_, _, mcd_ood_ent  = predict_mc_dropout(model, ood_loader, n_samples=30)
_, _, ens_ood_ent, _ = ensemble.predict(ood_loader)
_, _, swag_ood_ent = swag.predict(ood_loader, K=30)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

for ax, (ind_ent, ood_ent, title, color) in zip(axes, [
    (mcd_entropy,  mcd_ood_ent,  'MC Dropout (T=30)',    '#007A78'),
    (ens_entropy,  ens_ood_ent,  'Deep Ensemble (M=5)',  '#004B9B'),
    (swag_entropy, swag_ood_ent, 'SWAG (T=20, K=30)',    '#6D28D9'),
]):
    ind_med = np.median(ind_ent)
    ood_med = np.median(ood_ent)
    ratio   = ood_med / (ind_med + 1e-9)

    ax.hist(ind_ent, bins=60, alpha=0.65, color=color,     label='IND', density=True)
    ax.hist(ood_ent, bins=60, alpha=0.50, color='#B91C1C', label='OOD (σ=3.0)', density=True)
    ax.axvline(ind_med, color=color,     linestyle='--', lw=1.8,
               label=f'IND med={ind_med:.3f}')
    ax.axvline(ood_med, color='#B91C1C', linestyle='--', lw=1.8,
               label=f'OOD med={ood_med:.3f}')
    ax.set_title(f'{title}\n{ratio:.1f}× separation', fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Predictive Entropy H(y|x)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8.5)

plt.suptitle('OOD Detection — 3 Epistemic UQ Methods', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig_ood_3way.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig_ood_3way.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL E — Per-class ECE: all 6 methods (Standard + MCD + TS + Iso + Ens + SWAG)
# ─────────────────────────────────────────────────────────────────────────────

from evaluation.uncertainty import (
    predict_standard, predict_temperature_scaled,
    predict_isotonic, TemperatureScaler
)

std_probs, std_labels = predict_standard(model, test_loader)

ts = TemperatureScaler(model).to(DEVICE)
ts.fit(val_loader)
ts_probs, ts_labels = predict_temperature_scaled(ts, test_loader)

iso_probs, iso_labels, _ = predict_isotonic(model, val_loader, test_loader)

all_methods = {
    'Standard':       (std_probs,  std_labels),
    'MC Dropout':     (mcd_probs,  mcd_labels),
    'Temp. Scaling':  (ts_probs,   ts_labels),
    'Isotonic':       (iso_probs,  iso_labels),
    'Ensemble':       (ens_probs,  ens_labels),
    'SWAG':           (swag_probs, swag_labels),
}
COLORS_6 = {
    'Standard':      '#B91C1C',
    'MC Dropout':    '#007A78',
    'Temp. Scaling': '#166534',
    'Isotonic':      '#6D28D9',
    'Ensemble':      '#004B9B',
    'SWAG':          '#92400E',
}

print("\nPer-class ECE (class 0 = νₑ signal, class 1 = νμ background):")
print(f"{'Method':<18} {'νₑ (rare)':>12} {'νμ (majority)':>14}")
print("-"*46)

pce_all = {}
for name, (probs, labels) in all_methods.items():
    pce = compute_ece_per_class(probs, labels)
    pce_all[name] = pce
    print(f"{name:<18} {pce.get(0,float('nan')):>12.4f} {pce.get(1,float('nan')):>14.4f}")

# Grouped bar chart
classes      = ['νₑ signal (rare)', 'νμ background (majority)']
method_names = list(all_methods.keys())
n_m          = len(method_names)
x            = np.arange(len(classes))
width        = 0.13

fig, ax = plt.subplots(figsize=(11, 5))
for i, name in enumerate(method_names):
    pce  = pce_all[name]
    vals = [pce.get(c, 0.0) for c in [0, 1]]
    off  = (i - n_m / 2 + 0.5) * width
    bars = ax.bar(x + off, vals, width,
                  label=name, color=COLORS_6[name], alpha=0.88)
    for bar in bars:
        if bar.get_height() > 0.0002:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.0003,
                    f'{bar.get_height():.4f}',
                    ha='center', va='bottom', fontsize=7, rotation=50)

ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.set_ylabel('Per-class ECE (lower = better)', fontsize=11)
ax.set_title('Per-Class ECE — All 6 UQ Methods', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
plt.tight_layout()
plt.savefig('fig_perclass_ece_6methods.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig_perclass_ece_6methods.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL F — Violin plot: epistemic uncertainty distributions (IND vs OOD)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

for ax, (ind_ent, ood_ent, title, color) in zip(axes, [
    (mcd_entropy,  mcd_ood_ent,  'MC Dropout',    '#007A78'),
    (ens_entropy,  ens_ood_ent,  'Deep Ensemble', '#004B9B'),
    (swag_entropy, swag_ood_ent, 'SWAG',          '#6D28D9'),
]):
    # Subsample for violin (max 5k points for speed)
    ind_s = np.random.choice(ind_ent, min(5000, len(ind_ent)), replace=False)
    ood_s = np.random.choice(ood_ent, min(5000, len(ood_ent)), replace=False)

    parts = ax.violinplot([ind_s, ood_s], positions=[0, 1],
                          showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['In-Distribution', 'OOD (σ=3.0)'], fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel('Predictive Entropy', fontsize=10)

plt.suptitle('Uncertainty Distributions — IND vs OOD', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_violin_ood.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig_violin_ood.png")

print("\n✓ All SWAG cells complete.")
print("New figures:")
print("  fig_3way_calibration.png")
print("  fig_ood_3way.png")
print("  fig_perclass_ece_6methods.png")
print("  fig_violin_ood.png")
