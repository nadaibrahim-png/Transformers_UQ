# =============================================================================
# visualization/plots.py — Poster-quality figures
# =============================================================================
# Generates the 4 figures you need for your ICTP A0 poster.
# All figures saved as high-res PNGs in the figures/ directory.

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
    Side-by-side reliability diagrams for Standard, MC Dropout, and Temp. Scaling.

    Paper connection [GUO17 Figure 1]:
        'A perfectly calibrated model has outputs that match the diagonal.
         The gap between the bars and the diagonal is the ECE contribution.'

    Args:
        results : dict with keys 'std', 'mc', 'ts', each containing
                  {'probs': array, 'labels': array, 'metrics': dict}
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    titles = {
        "std": f"Standard (Baseline)\nECE = {results['std']['metrics']['ece']:.4f}",
        "mc":  f"MC Dropout (T={config.MC_SAMPLES} passes)\nECE = {results['mc']['metrics']['ece']:.4f}",
        "ts":  f"Temperature Scaling\nECE = {results['ts']['metrics']['ece']:.4f}",
    }

    for ax, (key, title) in zip(axes, titles.items()):
        probs  = results[key]["probs"]
        labels = results[key]["labels"]
        color  = COLORS[key]

        bin_confs, bin_accs, _ = get_reliability_data(probs, labels, config.N_BINS)
        gap = bin_confs - bin_accs   # positive = overconfident

        # Accuracy bars
        ax.bar(bin_confs, bin_accs, width=0.07, alpha=0.85,
               color=color, label="Accuracy", zorder=3)
        # Overconfidence gap (red fill)
        ax.bar(bin_confs, np.clip(gap, 0, None), width=0.07,
               bottom=bin_accs, alpha=0.35,
               color="#e74c3c", label="Gap (overconfidence)", zorder=3)

        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5,
                label="Perfect calibration", zorder=4)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Accuracy",   fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

    plt.suptitle(
        "Reliability Diagrams — Calibration Before and After UQ Methods\n"
        "(bars should align with the diagonal for perfect calibration)",
        fontsize=12, fontweight="bold", y=1.03
    )
    plt.tight_layout()
    _savefig("fig1_reliability_diagrams.png")


# ── Figure 2: ECE Bar Chart ───────────────────────────────────────────────────

def plot_ece_comparison(results):
    """
    Bar chart comparing ECE across the three methods.
    Lower bar = better calibration.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    labels_  = ["Standard\n(Baseline)", f"MC Dropout\n(T={config.MC_SAMPLES})", "Temperature\nScaling"]
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
        'A model with good uncertainty should assign HIGH entropy to OOD
         inputs and LOW entropy to in-distribution inputs.'
    Well-separated histograms → the model knows when it doesn't know.

    Args:
        uncertainty_test : predictive entropy on test (in-distribution) data
        uncertainty_ood  : predictive entropy on OOD (corrupted) data
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.hist(uncertainty_test, bins=25, alpha=0.72, color=COLORS["mc"],
            density=True, label="In-distribution (test)")
    ax.hist(uncertainty_ood,  bins=25, alpha=0.72, color=COLORS["ood"],
            density=True, label=f"OOD (noise σ={config.OOD_NOISE})")

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
        f"(MC Dropout, {config.MC_SAMPLES} samples)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    _savefig("fig3_entropy_ood.png")


# ── Figure 4: Results Summary Table ──────────────────────────────────────────

def plot_results_table(results, dataset_name, T_value):
    """
    Clean summary table of Accuracy / ECE / NLL for the poster.

    Args:
        results      : dict with 'std', 'mc', 'ts' metrics
        dataset_name : string name of the dataset (for title)
        T_value      : learned temperature scalar
    """
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.axis("off")

    def m(key): return results[key]["metrics"]

    rows = [
        ["Standard (Baseline)",
         f"{m('std')['accuracy']:.4f}",
         f"{m('std')['ece']:.4f}",
         f"{m('std')['nll']:.4f}",
         "—"],
        [f"MC Dropout (T={config.MC_SAMPLES} passes)",
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
    row_colors = [
        ["#fdecea"] * 5,
        ["#e8f4fd"] * 5,
        ["#eafaf1"] * 5,
    ]

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

    ax.set_title(f"Results Summary — {dataset_name}",
                 fontsize=13, fontweight="bold", pad=24)
    plt.tight_layout()
    _savefig("fig4_results_table.png")


# ── Figure 5: Active Learning Learning Curves ────────────────────────────────

def plot_learning_curves(al_results):
    """
    Learning curves: test accuracy vs number of labeled samples,
    for Random / Entropy / BALD acquisition strategies.

    Paper connection [GAL17 Figure 1]:
        'BALD and entropy-based acquisition reach high accuracy with
         significantly fewer labeled samples than random selection.'

    This is the KEY RESULT of your active learning experiment.
    A good acquisition function curve rises faster and plateaus higher.

    Args:
        al_results : dict returned by run_all_strategies()
                     {"random": {"labeled_counts": [...], "accuracies": [...]}, ...}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    style = {
        "random":  {"color": COLORS["std"], "linestyle": "--",  "marker": "o", "label": "Random (baseline)"},
        "entropy": {"color": COLORS["mc"],  "linestyle": "-",   "marker": "s", "label": "Entropy (MC Dropout)"},
        "bald":    {"color": COLORS["ts"],  "linestyle": "-",   "marker": "^", "label": "BALD [Gal 2017]"},
    }

    for strategy, data in al_results.items():
        s = style[strategy]
        ax.plot(data["labeled_counts"], data["accuracies"],
                color=s["color"], linestyle=s["linestyle"],
                marker=s["marker"], markersize=5, linewidth=2,
                label=s["label"])

    ax.set_xlabel("Number of Labeled Samples", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(
        "Active Learning — Learning Curves\n"
        "Entropy & BALD acquisition vs Random baseline",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1)
    ax.axvline(config.AL_INITIAL_SIZE, color="gray", linestyle=":",
               linewidth=1.2, label="Seed size")

    # Annotate final accuracies
    for strategy, data in al_results.items():
        final_acc = data["accuracies"][-1]
        final_n   = data["labeled_counts"][-1]
        ax.annotate(f"{final_acc:.3f}",
                    xy=(final_n, final_acc),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=style[strategy]["color"],
                    fontweight="bold")

    plt.tight_layout()
    _savefig("fig6_active_learning_curves.png")


# ── Figure 6 (bonus): BALD score distribution ────────────────────────────────

def plot_bald_scores(bald_scores, entropy_scores):
    """
    Compare BALD scores vs raw entropy scores on the pool.
    Shows that BALD filters out aleatoric noise (high entropy but low BALD).

    Paper connection [GAL17 §3.1]:
        'BALD measures epistemic uncertainty — it ignores noise in the data
         and focuses on model ignorance.'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(entropy_scores, bins=30, color=COLORS["mc"], alpha=0.8, density=True)
    ax1.set_title("Entropy Scores\n(total uncertainty)", fontweight="bold")
    ax1.set_xlabel("H[y|x, D]")
    ax1.set_ylabel("Density")

    ax2.hist(bald_scores, bins=30, color=COLORS["ts"], alpha=0.8, density=True)
    ax2.set_title("BALD Scores\n(epistemic uncertainty only)", fontweight="bold")
    ax2.set_xlabel("I(y; ω | x, D)")
    ax2.set_ylabel("Density")

    plt.suptitle("Acquisition Function Score Distributions",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _savefig("fig7_bald_vs_entropy.png")


# ── Figure 7 (bonus): Training curves ────────────────────────────────────────

def plot_training_curves(history):
    """
    Train/val loss and val accuracy over epochs.
    Useful for your Methods section — shows the model converged cleanly.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], color=COLORS["mc"],  label="Train loss")
    ax1.plot(epochs, history["val_loss"],   color=COLORS["std"], label="Val loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training Curves", fontweight="bold")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], color=COLORS["ts"])
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("Validation Accuracy", fontweight="bold")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    _savefig("fig5_training_curves.png")
