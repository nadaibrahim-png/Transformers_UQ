# =============================================================================
# visualization/plots.py — Poster-quality figures
# =============================================================================
# Generates 7 figures for the ICTP A0 poster:
#   fig1 — Reliability diagrams (3-panel: Standard / MC / Temp Scaling)
#   fig2 — ECE bar chart (3 methods, base config)
#   fig3 — Predictive entropy: in-distribution vs OOD
#   fig4 — Results summary table (base config)
#   fig5 — Training curves (loss + accuracy)
#   fig6 — Attention heatmap (implicit feature selection over 50 features)
#   fig7 — Hyperparameter ablation sweep (ECE across all configs)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from evaluation.metrics import get_reliability_data

plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "std": "#e05c5c",   # red    — standard baseline
    "mc":  "#4e91d9",   # blue   — MC Dropout
    "ts":  "#2ecc71",   # green  — Temperature Scaling
    "ood": "#f39c12",   # orange — OOD data
}

os.makedirs(config.FIGURES_DIR, exist_ok=True)


def _savefig(name):
    path = os.path.join(config.FIGURES_DIR, name)
    plt.savefig(path, dpi=config.DPI, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")


# ── Figure 1: Reliability Diagrams ───────────────────────────────────────────

def plot_reliability_diagrams(results):
    """
    Three-panel reliability diagram: Standard vs MC Dropout vs Temp Scaling.

    Paper connection [GUO17 Figure 1]:
        'A perfectly calibrated model has outputs that match the diagonal.
         The gap between the bars and the diagonal is the ECE contribution.'
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    titles = {
        "std": f"Standard (Baseline)\nECE = {results['std']['metrics']['ece']:.4f}",
        "mc":  f"MC Dropout ({config.MC_SAMPLES} passes)\nECE = {results['mc']['metrics']['ece']:.4f}",
        "ts":  f"Temperature Scaling\nECE = {results['ts']['metrics']['ece']:.4f}",
    }

    for ax, (key, title) in zip(axes, titles.items()):
        probs  = results[key]["probs"]
        labels = results[key]["labels"]
        color  = COLORS[key]

        bin_confs, bin_accs, _ = get_reliability_data(probs, labels, config.N_BINS)
        gap = bin_confs - bin_accs

        ax.bar(bin_confs, bin_accs, width=0.07, alpha=0.85,
               color=color, label="Accuracy", zorder=3)
        ax.bar(bin_confs, np.clip(gap, 0, None), width=0.07,
               bottom=bin_accs, alpha=0.35,
               color="#e74c3c", label="Gap (overconfidence)", zorder=3)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5,
                label="Perfect calibration", zorder=4)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Accuracy",   fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle(
        "Reliability Diagrams — MiniBooNE Particle Identification\n"
        "(bars should align with the diagonal for perfect calibration)",
        fontsize=12, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    _savefig("fig1_reliability_diagrams.png")


# ── Figure 2: ECE Bar Chart ───────────────────────────────────────────────────

def plot_ece_comparison(results):
    """Bar chart comparing ECE across the three methods. Lower = better."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    labels_  = ["Standard\n(Baseline)",
                 f"MC Dropout\n({config.MC_SAMPLES} passes)",
                 "Temperature\nScaling"]
    ece_vals = [results[k]["metrics"]["ece"] for k in ("std", "mc", "ts")]
    colors_  = [COLORS["std"], COLORS["mc"], COLORS["ts"]]

    bars = ax.bar(labels_, ece_vals, color=colors_, width=0.45,
                  edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, ece_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("ECE  (lower = better calibration)", fontsize=11)
    ax.set_title("Expected Calibration Error by Method", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(ece_vals) * 1.45)
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    _savefig("fig2_ece_comparison.png")


# ── Figure 3: Predictive Entropy — In-distribution vs OOD ────────────────────

def plot_entropy_ood(uncertainty_test, uncertainty_ood):
    """
    Histogram comparing predictive entropy on in-distribution vs OOD data.

    Paper connection [GAL16 §4]:
        A well-calibrated model assigns HIGH entropy to OOD inputs
        and LOW entropy to in-distribution inputs.
    Well-separated histograms → the model knows when it doesn't know.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(uncertainty_test, bins=25, alpha=0.72, color=COLORS["mc"],
            density=True, label="In-distribution (test)")
    ax.hist(uncertainty_ood,  bins=25, alpha=0.72, color=COLORS["ood"],
            density=True, label=f"OOD (Gaussian noise σ={config.OOD_NOISE})")

    ax.axvline(np.median(uncertainty_test), color=COLORS["mc"],
               linestyle="--", linewidth=1.8,
               label=f"Median IND = {np.median(uncertainty_test):.3f}")
    ax.axvline(np.median(uncertainty_ood), color=COLORS["ood"],
               linestyle="--", linewidth=1.8,
               label=f"Median OOD = {np.median(uncertainty_ood):.3f}")

    ax.set_xlabel("Predictive Entropy  H(y|x) = −Σ p log p", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        f"Predictive Entropy: In-Distribution vs Out-of-Distribution\n"
        f"(MC Dropout, {config.MC_SAMPLES} passes)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _savefig("fig3_entropy_ood.png")


# ── Figure 4: Results Summary Table ──────────────────────────────────────────

def plot_results_table(results, dataset_name, T_value):
    """Clean summary table of Accuracy / ECE / NLL for the poster."""
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.axis("off")

    def m(key): return results[key]["metrics"]

    rows = [
        ["Standard (Baseline)",
         f"{m('std')['accuracy']:.4f}",
         f"{m('std')['ece']:.4f}",
         f"{m('std')['nll']:.4f}",
         "—"],
        [f"MC Dropout ({config.MC_SAMPLES} passes)",
         f"{m('mc')['accuracy']:.4f}",
         f"{m('mc')['ece']:.4f}",
         f"{m('mc')['nll']:.4f}",
         f"{config.MC_SAMPLES} stochastic passes"],
        ["Temperature Scaling",
         f"{m('ts')['accuracy']:.4f}",
         f"{m('ts')['ece']:.4f}",
         f"{m('ts')['nll']:.4f}",
         f"T = {T_value:.3f}"],
    ]
    col_labels = ["Method", "Accuracy ↑", "ECE ↓", "NLL ↓", "Notes"]
    row_colors = [["#fdecea"]*5, ["#e8f4fd"]*5, ["#eafaf1"]*5]

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
        cellColours=row_colors
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.2)

    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].get_text().set_color("white")
        table[(0, j)].get_text().set_fontweight("bold")

    ax.set_title(f"Calibration Results — {dataset_name}",
                 fontsize=13, fontweight="bold", pad=24)
    plt.tight_layout()
    _savefig("fig4_results_table.png")


# ── Figure 5: Training Curves ─────────────────────────────────────────────────

def plot_training_curves(history):
    """Train/val loss and val accuracy over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], color=COLORS["mc"],  label="Train loss")
    ax1.plot(epochs, history["val_loss"],   color=COLORS["std"], label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Curves — FT-Transformer", fontweight="bold")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], color=COLORS["ts"])
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Validation Accuracy", fontweight="bold")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    _savefig("fig5_training_curves.png")


# ── Figure 6: Attention Heatmap — Implicit Feature Selection ─────────────────

def plot_attention_heatmap(attn_weights, feature_names):
    """
    Heatmap of attention weights from the last Transformer layer.

    Row 0 = CLS token. The CLS row shows how much the model attends to
    each of the 50 MiniBooNE kinematic features when making its prediction.
    High attention weight ≈ the model considers this feature important.

    This is the "implicit feature selection" figure — the Transformer learns
    which features matter without any manual filtering or preprocessing.

    Args:
        attn_weights  : (n_tokens, n_tokens) array — from model.get_attention_weights()
                        n_tokens = n_features + 1 (CLS + 50 features)
        feature_names : list of feature name strings (length n_features)
    """
    # CLS row: how strongly CLS attends to each feature
    cls_attn = attn_weights[0, 1:]   # shape (n_features,) — skip CLS-to-CLS

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [1, 3]})

    # ── Top panel: CLS attention bar chart (most interpretable) ──────────────
    sorted_idx  = np.argsort(cls_attn)[::-1]
    sorted_attn = cls_attn[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    top_n = min(20, len(sorted_names))   # show top 20 for readability
    colors_bar = plt.cm.Blues(
        np.linspace(0.4, 0.9, top_n)[::-1]
    )
    ax1.bar(range(top_n), sorted_attn[:top_n], color=colors_bar)
    ax1.set_xticks(range(top_n))
    ax1.set_xticklabels(sorted_names[:top_n], fontsize=9, rotation=45, ha="right")
    ax1.set_ylabel("Attention weight", fontsize=10)
    ax1.set_title(
        "CLS Token Attention — Top 20 Most-Attended Features\n"
        "(implicit feature selection learned by FT-Transformer)",
        fontsize=11, fontweight="bold"
    )

    # ── Bottom panel: full attention matrix heatmap ───────────────────────────
    token_labels = ["CLS"] + list(feature_names)
    im = ax2.imshow(attn_weights, cmap="Blues", aspect="auto",
                    vmin=0, vmax=attn_weights.max())
    plt.colorbar(im, ax=ax2, fraction=0.02, pad=0.02, label="Attention weight")

    # Label axes (only show every 5th feature for readability)
    tick_step = 5
    ticks     = list(range(0, len(token_labels), tick_step))
    tick_lbls = [token_labels[i] for i in ticks]
    ax2.set_xticks(ticks); ax2.set_xticklabels(tick_lbls, fontsize=8, rotation=45)
    ax2.set_yticks(ticks); ax2.set_yticklabels(tick_lbls, fontsize=8)
    ax2.set_xlabel("Key token (feature being attended to)", fontsize=10)
    ax2.set_ylabel("Query token", fontsize=10)
    ax2.set_title("Full Attention Matrix — Last Encoder Layer",
                  fontsize=11, fontweight="bold")

    plt.suptitle(
        "FT-Transformer Attention Weights over MiniBooNE Kinematic Features",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _savefig("fig6_attention_heatmap.png")


# ── Figure 7: Hyperparameter Sweep Results ───────────────────────────────────

def plot_sweep_results(sweep_results):
    """
    Compare ECE across all hyperparameter configurations for all three
    UQ methods. Shows which architecture produces the best-calibrated model
    before and after applying MC Dropout / Temperature Scaling.

    Args:
        sweep_results : list of dicts from run_sweep()
                        each with {'label', 'cfg', 'results'}
    """
    labels   = [r["label"] for r in sweep_results]
    ece_std  = [r["results"]["std"]["metrics"]["ece"] for r in sweep_results]
    ece_mc   = [r["results"]["mc"]["metrics"]["ece"]  for r in sweep_results]
    ece_ts   = [r["results"]["ts"]["metrics"]["ece"]  for r in sweep_results]
    acc_std  = [r["results"]["std"]["metrics"]["accuracy"] for r in sweep_results]

    x    = np.arange(len(labels))
    w    = 0.25

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(labels)*1.2), 9))

    # ── ECE comparison ────────────────────────────────────────────────────────
    ax1.bar(x - w,   ece_std, w, color=COLORS["std"], alpha=0.85, label="Standard")
    ax1.bar(x,       ece_mc,  w, color=COLORS["mc"],  alpha=0.85, label=f"MC Dropout ({config.MC_SAMPLES})")
    ax1.bar(x + w,   ece_ts,  w, color=COLORS["ts"],  alpha=0.85, label="Temp Scaling")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("ECE  (lower = better)", fontsize=11)
    ax1.set_title(
        "ECE Across Hyperparameter Configurations\n"
        "(ablation: one parameter varied at a time)",
        fontsize=12, fontweight="bold"
    )
    ax1.legend(fontsize=9)
    ax1.axhline(0, color="black", linewidth=0.6)

    # Annotate best (lowest ECE per method)
    for ece_vals, color in [(ece_ts, COLORS["ts"])]:
        best_idx = int(np.argmin(ece_vals))
        ax1.annotate(f"Best\n{ece_vals[best_idx]:.3f}",
                     xy=(x[best_idx] + w, ece_vals[best_idx]),
                     xytext=(0, 10), textcoords="offset points",
                     ha="center", fontsize=8, color=color, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    # ── Accuracy comparison ───────────────────────────────────────────────────
    ax2.plot(x, acc_std, "o-", color=COLORS["std"], linewidth=2,
             markersize=6, label="Accuracy (Standard)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Test Accuracy", fontsize=11)
    ax2.set_title("Test Accuracy Across Configurations", fontsize=12, fontweight="bold")
    ax2.set_ylim(max(0, min(acc_std) - 0.05), min(1, max(acc_std) + 0.05))
    ax2.axhline(np.mean(acc_std), color="gray", linestyle="--",
                linewidth=1, label=f"Mean = {np.mean(acc_std):.3f}")
    ax2.legend(fontsize=9)

    # Vertical separators between parameter groups
    group_size = len(config.SWEEP[list(config.SWEEP.keys())[0]])
    for i in range(group_size, len(labels), group_size):
        ax1.axvline(i - 0.5, color="gray", linestyle=":", linewidth=1.2)
        ax2.axvline(i - 0.5, color="gray", linestyle=":", linewidth=1.2)

    # Label parameter groups
    param_names = list(config.SWEEP.keys())
    for g, pname in enumerate(param_names):
        mid = g * group_size + group_size / 2 - 0.5
        ax1.text(mid, ax1.get_ylim()[1] * 0.95, pname.replace("_", " "),
                 ha="center", fontsize=9, color="gray",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7))

    plt.suptitle("Hyperparameter Ablation Study — FT-Transformer on MiniBooNE",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig("fig7_sweep_results.png")
