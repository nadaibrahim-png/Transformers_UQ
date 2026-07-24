# ============================================================
# Run these cells in Colab to generate the two missing figures
# Then download with files.download()
# ============================================================

# ── CELL A: Reliability Diagram (4 panels) ──────────────────
import matplotlib.pyplot as plt
import numpy as np
from evaluation.metrics import get_reliability_data
from evaluation.uncertainty import (
    predict_standard, predict_mc_dropout,
    predict_temperature_scaled, predict_isotonic,
    TemperatureScaler
)

std_probs,  std_labels  = predict_standard(model, loaders["test"])
mc_probs,   mc_labels,  mc_entropy = predict_mc_dropout(model, loaders["test"])
ts_model = TemperatureScaler(model).to(DEVICE)
ts_model.fit(loaders["val"])
ts_probs,   ts_labels   = predict_temperature_scaled(ts_model, loaders["test"])
iso_probs,  iso_labels, _ = predict_isotonic(model, loaders["val"], loaders["test"])

methods = [
    ("Standard\n(Baseline)",    std_probs,  std_labels,  "#B91C1C"),
    ("MC Dropout\n(T=30)",      mc_probs,   mc_labels,   "#007A78"),
    ("Temperature\nScaling",    ts_probs,   ts_labels,   "#166534"),
    ("Isotonic\nRegression",    iso_probs,  iso_labels,  "#6D28D9"),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
fig.suptitle("Reliability Diagrams — Confidence vs. Accuracy", fontsize=15, fontweight="bold", y=1.02)

for ax, (name, probs, labels, color) in zip(axes, methods):
    bin_confs, bin_accs, bin_sizes = get_reliability_data(probs, labels, n_bins=10)
    ece = sum(
        (bin_sizes[i] / len(labels)) * abs(bin_confs[i] - bin_accs[i])
        for i in range(10) if bin_sizes[i] > 0
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Perfect")
    mask = bin_sizes > 0
    ax.bar(bin_confs[mask], bin_accs[mask], width=0.08, alpha=0.7,
           color=color, edgecolor="white", label=f"ECE={ece:.4f}")
    ax.bar(bin_confs[mask], bin_confs[mask], width=0.08, alpha=0.15,
           color=color, edgecolor="none")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_title(name, fontsize=12, fontweight="bold", color=color)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Accuracy", fontsize=11)
plt.tight_layout()
plt.savefig("/content/reliability_diagram_4panel.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.show()

from google.colab import files
files.download("/content/reliability_diagram_4panel.png")


# ── CELL B: Entropy Histogram + AUROC ────────────────────────
from sklearn.metrics import roc_auc_score

ind_entropy = mc_entropy
_, _, ood_entropy = predict_mc_dropout(model, loaders["ood"])

all_entropy  = np.concatenate([ind_entropy, ood_entropy])
all_labels_ood = np.concatenate([np.zeros(len(ind_entropy)), np.ones(len(ood_entropy))])
auroc = roc_auc_score(all_labels_ood, all_entropy)

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, max(all_entropy.max(), 0.8), 60)
ax.hist(ind_entropy, bins=bins, alpha=0.65, color="#004B9B", label="In-Distribution (IND)", density=True)
ax.hist(ood_entropy, bins=bins, alpha=0.65, color="#B91C1C", label="OOD (σ=3.0 noise)", density=True)
ax.axvline(np.median(ind_entropy), color="#004B9B", lw=2, ls="--", label=f"Median IND = {np.median(ind_entropy):.3f}")
ax.axvline(np.median(ood_entropy), color="#B91C1C", lw=2, ls="--", label=f"Median OOD = {np.median(ood_entropy):.3f}")
ax.set_xlabel("Predictive Entropy H(x)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title(f"MC Dropout Predictive Entropy: IND vs OOD\nAUROC = {auroc:.3f}", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.text(0.97, 0.95, f"AUROC = {auroc:.3f}", transform=ax.transAxes,
        fontsize=14, fontweight="bold", color="#166534", ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAF4EA", edgecolor="#166534"))
plt.tight_layout()
plt.savefig("/content/ood_entropy_auroc.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.show()
print(f"AUROC: {auroc:.4f} | Median IND: {np.median(ind_entropy):.4f} | Median OOD: {np.median(ood_entropy):.4f}")

from google.colab import files
files.download("/content/ood_entropy_auroc.png")
