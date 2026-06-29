# =============================================================================
# evaluation/extended_evaluation.py — Extended calibration experiments
# =============================================================================
# Five new experiments beyond the baseline:
#
#   1. Calibration robustness under noise (σ sweep)
#      — Plot ECE & entropy vs Gaussian noise level σ ∈ {0,0.5,1,2,3}
#        for all four UQ methods. Shows *when* each method breaks down.
#
#   2. Full metric table (ECE + NLL + Brier Score)
#      — Proper scoring rules alongside ECE.
#        Brier Score penalises overconfidence continuously (no binning).
#
#   3. Attention-based feature importance
#      — CLS-row attention weights averaged over the test set.
#        Ranks all 50 MiniBooNE kinematic features by how much the
#        model attends to them — implicit feature selection.
#
#   4. Isotonic Regression vs Temperature Scaling
#      — Compare parametric (TS, 1 param) vs non-parametric (IR, flexible)
#        post-hoc calibration. TS often wins on small val sets.
#
#   5. Per-class calibration
#      — ECE split by νₑ signal vs νμ background.
#        Critical for rare-event physics: signal purity > raw accuracy.
#
# Usage (in Colab after training):
#   from evaluation.extended_evaluation import run_all_extended
#   results = run_all_extended(model, train_loader, val_loader, test_loader)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from evaluation.metrics import (
    compute_all, compute_ece, compute_brier_score, compute_nll,
    compute_ece_per_class
)
from evaluation.uncertainty import (
    predict_standard, predict_mc_dropout, predict_temperature_scaled,
    predict_isotonic, TemperatureScaler
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = {
    "Standard":    "#B91C1C",   # red
    "MC Dropout":  "#007A78",   # teal
    "Temp. Scaling": "#166534", # green
    "Isotonic":    "#6D28D9",   # purple
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build an OOD DataLoader with Gaussian noise injected
# ─────────────────────────────────────────────────────────────────────────────

def _make_noisy_loader(test_loader, sigma):
    """
    Return a DataLoader whose features have been corrupted by N(0, σ²) noise.
    σ=0 returns the original clean test set (in-distribution baseline).
    """
    from torch.utils.data import TensorDataset, DataLoader
    all_X, all_y = [], []
    for X_batch, y_batch in test_loader:
        all_X.append(X_batch)
        all_y.append(y_batch)
    X = torch.cat(all_X)
    y = torch.cat(all_y)
    if sigma > 0:
        X = X + torch.randn_like(X) * sigma
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=test_loader.batch_size or 512, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1 — Calibration robustness under varying noise
# ─────────────────────────────────────────────────────────────────────────────

def run_noise_sweep(model, val_loader, test_loader, ts_model, ir_scaler,
                    sigmas=None):
    """
    Sweep Gaussian noise level σ and record ECE + mean entropy for each
    UQ method at each noise level.

    Args:
        model      : trained FTTransformer
        val_loader : validation DataLoader (not corrupted)
        test_loader: clean test DataLoader
        ts_model   : fitted TemperatureScaler
        ir_scaler  : fitted IsotonicScaler
        sigmas     : list of σ values (default: [0, 0.5, 1.0, 2.0, 3.0])

    Returns:
        results : dict[method_name][sigma] → {'ece': float, 'entropy': float}
    """
    sigmas = sigmas or [0, 0.5, 1.0, 2.0, 3.0]
    methods = ["Standard", "MC Dropout", "Temp. Scaling", "Isotonic"]
    results = {m: {"ece": [], "entropy": []} for m in methods}

    print("Running noise sweep...")
    for sigma in sigmas:
        print(f"  σ = {sigma:.1f}", end=" | ", flush=True)
        noisy_loader = _make_noisy_loader(test_loader, sigma)

        # Standard
        p, _ = predict_standard(model, noisy_loader)
        results["Standard"]["ece"].append(compute_ece(p, _))
        results["Standard"]["entropy"].append(
            float(np.mean(-np.sum(p * np.log(p + 1e-9), axis=1))))

        # MC Dropout
        p_mc, _, unc = predict_mc_dropout(model, noisy_loader)
        results["MC Dropout"]["ece"].append(compute_ece(p_mc, _))
        results["MC Dropout"]["entropy"].append(float(unc.mean()))

        # Temperature Scaling
        p_ts, _ = predict_temperature_scaled(ts_model, noisy_loader)
        results["Temp. Scaling"]["ece"].append(compute_ece(p_ts, _))
        results["Temp. Scaling"]["entropy"].append(
            float(np.mean(-np.sum(p_ts * np.log(p_ts + 1e-9), axis=1))))

        # Isotonic (apply scaler to standard probs)
        p_raw, _ = predict_standard(model, noisy_loader)
        p_ir = ir_scaler.predict(p_raw)
        results["Isotonic"]["ece"].append(compute_ece(p_ir, _))
        results["Isotonic"]["entropy"].append(
            float(np.mean(-np.sum(p_ir * np.log(p_ir + 1e-9), axis=1))))

        print("done")

    return sigmas, results


def plot_noise_sweep(sigmas, results, save_path=None):
    """
    Two-panel figure: ECE vs σ (left) and Mean Entropy vs σ (right).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Calibration & Uncertainty Under Gaussian Noise (σ sweep)",
                 fontsize=14, fontweight="bold", y=1.02)

    for method, color in COLORS.items():
        ls = "--" if method == "Standard" else "-"
        axes[0].plot(sigmas, results[method]["ece"],
                     marker="o", color=color, label=method, linewidth=2, ls=ls)
        axes[1].plot(sigmas, results[method]["entropy"],
                     marker="o", color=color, label=method, linewidth=2, ls=ls)

    for ax, title, ylabel in zip(
        axes,
        ["ECE vs Noise Level", "Mean Predictive Entropy vs Noise Level"],
        ["ECE (lower = better)", "Mean Entropy H (higher = more uncertain)"]
    ):
        ax.set_xlabel("Noise σ (std of injected Gaussian)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5, label="IND")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2 — Full metric table (ECE + NLL + Brier)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_metrics(model, val_loader, test_loader, ts_model, ir_scaler):
    """
    Compute all metrics for all four UQ methods on the clean test set.

    Headline metric is balanced_accuracy (not plain accuracy) to account
    for MiniBooNE's 72/28 class imbalance.

    Returns:
        table : dict[method_name] → dict with all metrics from compute_all()
    """
    print("Computing full metric table...")
    table = {}

    p, y = predict_standard(model, test_loader)
    table["Standard"] = compute_all(p, y)

    p_mc, y, _ = predict_mc_dropout(model, test_loader)
    table["MC Dropout"] = compute_all(p_mc, y)

    p_ts, y = predict_temperature_scaled(ts_model, test_loader)
    table["Temp. Scaling"] = compute_all(p_ts, y)

    p_raw, y = predict_standard(model, test_loader)
    p_ir = ir_scaler.predict(p_raw)
    table["Isotonic"] = compute_all(p_ir, y)

    # Print — balanced accuracy as headline
    hdr = f"\n{'Method':<20} {'Bal.Acc':>9} {'Acc':>7} {'νₑ Recall':>11} {'νμ Recall':>11} {'ECE':>8} {'NLL':>8} {'Brier':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in table.items():
        print(f"{name:<20} {m['balanced_accuracy']:>9.4f} {m['accuracy']:>7.4f} "
              f"{m['signal_recall']:>11.4f} {m['background_recall']:>11.4f} "
              f"{m['ece']:>8.4f} {m['nll']:>8.4f} {m['brier']:>8.4f}")

    return table


def plot_metric_bars(table, save_path=None):
    """
    Three grouped bar charts: ECE, NLL, Brier Score side by side.
    """
    methods = list(table.keys())
    colors  = [COLORS[m] for m in methods]
    metrics = ["ece", "nll", "brier"]
    titles  = ["Expected Calibration Error (ECE)", "Negative Log-Likelihood (NLL)",
               "Brier Score"]
    ylabels = ["ECE ↓", "NLL ↓", "Brier ↓"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Calibration Metrics — All Four Methods",
                 fontsize=14, fontweight="bold")

    for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
        vals = [table[m][metric] for m in methods]
        bars = ax.bar(methods, vals, color=colors, edgecolor="white",
                      linewidth=0.5, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.18)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3 — Attention-based feature importance
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_importance(model, test_loader, top_k=20, save_path=None):
    """
    Extract CLS-row attention weights from the last Transformer layer,
    averaged over all test samples. Ranks features by attention weight.

    The CLS token aggregates information from all feature tokens; high
    attention weight on feature i means the model heavily uses feature i
    when making its predictions.

    Args:
        model      : trained FTTransformer
        test_loader: DataLoader for test set
        top_k      : how many top features to show in the bar chart

    Returns:
        feature_weights : (n_features,) mean CLS attention to each feature
        ranked_indices  : feature indices sorted by attention (highest first)
    """
    print("Extracting attention weights...")
    model.eval()
    all_weights = []

    for X_batch, _ in test_loader:
        w = model.get_attention_weights(X_batch.to(DEVICE))
        # w shape: (batch, n_tokens, n_tokens) or (n_tokens, n_tokens)
        # CLS is token 0 → row 0, columns 1: are features
        if w.ndim == 3:
            cls_row = w[:, 0, 1:]    # (batch, n_features)
            all_weights.append(cls_row.mean(axis=0))
        else:
            cls_row = w[0, 1:]       # (n_features,)
            all_weights.append(cls_row)

    feature_weights = np.array(all_weights).mean(axis=0)   # (n_features,)
    ranked_indices  = np.argsort(feature_weights)[::-1]     # highest first

    # Plot
    top_idx = ranked_indices[:top_k]
    top_w   = feature_weights[top_idx]
    labels  = [f"F{i+1:02d}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_k))[::-1]
    bars = ax.bar(labels, top_w, color=bar_colors, edgecolor="white")
    ax.set_xlabel("MiniBooNE Feature (F01–F50)", fontsize=12)
    ax.set_ylabel("Mean CLS Attention Weight", fontsize=12)
    ax.set_title(
        f"Feature Importance via CLS Attention — Top {top_k} of 50 Features\n"
        "(Averaged over test set, last Transformer layer)",
        fontsize=13, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)

    # Annotate top 5
    for bar, val in zip(bars[:5], top_w[:5]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top_w.max() * 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#004B9B")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()
    return feature_weights, ranked_indices, fig


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 4 — Isotonic Regression vs Temperature Scaling
# ─────────────────────────────────────────────────────────────────────────────

def plot_isotonic_vs_ts(model, val_loader, test_loader, ts_model, save_path=None):
    """
    Reliability diagram comparison: uncalibrated vs Temperature Scaling
    vs Isotonic Regression. Shows the actual calibration curves (not just
    a single ECE number) for a richer visual comparison.
    """
    from evaluation.metrics import get_reliability_data

    # Raw
    p_raw, y = predict_standard(model, test_loader)
    # TS
    p_ts, _  = predict_temperature_scaled(ts_model, test_loader)
    # IR
    p_raw2, _ = predict_standard(model, test_loader)
    p_ir_val, y_val = predict_standard(model, val_loader)
    from evaluation.uncertainty import IsotonicScaler
    ir = IsotonicScaler()
    ir.fit(p_ir_val, y_val)
    p_ir = ir.predict(p_raw2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Reliability Diagrams: Post-Hoc Calibration Comparison",
                 fontsize=14, fontweight="bold")

    for ax, probs, name, color in zip(
        axes,
        [p_raw, p_ts, p_ir],
        ["Uncalibrated (Baseline)", "Temperature Scaling (T=1.282)", "Isotonic Regression"],
        [COLORS["Standard"], COLORS["Temp. Scaling"], COLORS["Isotonic"]]
    ):
        bc, ba, bs = get_reliability_data(probs, y)
        ece_val = compute_ece(probs, y)

        # Gap fill (miscalibration region)
        ax.bar(bc, ba, width=0.09, alpha=0.7, color=color, label="Model accuracy")
        ax.bar(bc, np.abs(bc - ba), bottom=np.minimum(bc, ba),
               width=0.09, alpha=0.25, color="red", label="Gap (miscalib.)")
        # Perfect calibration diagonal
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
        ax.set_xlabel("Confidence", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{name}\nECE = {ece_val:.4f}", fontsize=12,
                     fontweight="bold", color=color)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 5 — Per-class calibration
# ─────────────────────────────────────────────────────────────────────────────

def run_per_class_calibration(model, test_loader, ts_model, ir_scaler,
                              class_names=None, save_path=None):
    """
    Compute and plot per-class ECE for all methods.

    For MiniBooNE:
        Class 0 = νμ background  (72% of dataset — majority)
        Class 1 = νₑ signal      (28% of dataset — rare events)

    A model could have low overall ECE but be poorly calibrated for the
    rare signal class — this reveals that asymmetry.

    Args:
        class_names : list of class labels (default: ["νμ Bkg (0)", "νₑ Sig (1)"])

    Returns:
        per_class : dict[method] → {class_idx: ece}
    """
    class_names = class_names or ["νμ Background (class 0)", "νₑ Signal (class 1)"]
    print("Computing per-class ECE...")

    # Get probabilities
    p_std, y  = predict_standard(model, test_loader)
    p_mc, _, _ = predict_mc_dropout(model, test_loader)
    p_ts, _   = predict_temperature_scaled(ts_model, test_loader)
    p_ir      = ir_scaler.predict(p_std.copy())

    per_class = {}
    for name, probs in [("Standard", p_std), ("MC Dropout", p_mc),
                        ("Temp. Scaling", p_ts), ("Isotonic", p_ir)]:
        per_class[name] = compute_ece_per_class(probs, y)

    # Print
    print(f"\n{'Method':<20} {'νμ Bkg (ECE)':>15} {'νₑ Sig (ECE)':>15}")
    print("-" * 52)
    for name, pc in per_class.items():
        print(f"{name:<20} {pc[0]:>15.4f} {pc[1]:>15.4f}")

    # Plot grouped bars
    methods    = list(per_class.keys())
    x          = np.arange(len(methods))
    width      = 0.35
    ece_bkg    = [per_class[m][0] for m in methods]
    ece_sig    = [per_class[m][1] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, ece_bkg, width, label=class_names[0],
                   color="#B91C1C", alpha=0.85)
    bars2 = ax.bar(x + width/2, ece_sig, width, label=class_names[1],
                   color="#004B9B", alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0003,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("ECE (lower = better)", fontsize=12)
    ax.set_title("Per-Class Calibration Error — Signal vs Background\n"
                 "Reveals whether the model is equally well-calibrated for rare νₑ events",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()
    return per_class, fig


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_extended(model, train_loader, val_loader, test_loader,
                     figure_dir=None):
    """
    Run all five extended experiments in sequence.

    Call this in Colab after the model is trained and the standard
    calibration results have been produced.

    Args:
        model        : trained FTTransformer on DEVICE
        train_loader : not used currently (reserved for future methods)
        val_loader   : validation DataLoader (used to fit calibrators)
        test_loader  : test DataLoader
        figure_dir   : directory to save figures (default: config.FIGURES_DIR)

    Returns:
        results : dict with keys 'noise_sweep', 'metrics', 'per_class',
                  'feature_weights', 'ranked_features'
    """
    fig_dir = figure_dir or getattr(config, "FIGURES_DIR", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 60)
    print("EXTENDED EVALUATION — 5 experiments")
    print("=" * 60)

    # ── Fit calibrators once ──────────────────────────────────────────────────
    print("\n[Setup] Fitting Temperature Scaler...")
    ts_model = TemperatureScaler(model).to(DEVICE)
    T = ts_model.fit(val_loader)

    print("\n[Setup] Fitting Isotonic Scaler...")
    p_val, y_val = predict_standard(model, val_loader)
    from evaluation.uncertainty import IsotonicScaler
    ir_scaler = IsotonicScaler()
    ir_scaler.fit(p_val, y_val)

    # ── Experiment 1: Noise sweep ─────────────────────────────────────────────
    print("\n[1/5] Noise sweep (σ ∈ {0, 0.5, 1.0, 2.0, 3.0})")
    sigmas, noise_results = run_noise_sweep(
        model, val_loader, test_loader, ts_model, ir_scaler)
    plot_noise_sweep(sigmas, noise_results,
                     save_path=os.path.join(fig_dir, "noise_sweep.png"))

    # ── Experiment 2: Full metrics ────────────────────────────────────────────
    print("\n[2/5] Full metric table (ECE + NLL + Brier)")
    metrics_table = run_full_metrics(model, val_loader, test_loader,
                                     ts_model, ir_scaler)
    plot_metric_bars(metrics_table,
                     save_path=os.path.join(fig_dir, "metrics_all.png"))

    # ── Experiment 3: Feature importance ─────────────────────────────────────
    print("\n[3/5] Attention-based feature importance")
    feat_weights, ranked_idx, _ = run_feature_importance(
        model, test_loader,
        save_path=os.path.join(fig_dir, "feature_importance.png"))

    # ── Experiment 4: Isotonic vs TS reliability diagram ─────────────────────
    print("\n[4/5] Isotonic Regression vs Temperature Scaling")
    plot_isotonic_vs_ts(model, val_loader, test_loader, ts_model,
                        save_path=os.path.join(fig_dir, "reliability_comparison.png"))

    # ── Experiment 5: Per-class calibration ──────────────────────────────────
    print("\n[5/5] Per-class calibration (νₑ signal vs νμ background)")
    per_class, _ = run_per_class_calibration(
        model, test_loader, ts_model, ir_scaler,
        save_path=os.path.join(fig_dir, "per_class_ece.png"))

    print("\n" + "=" * 60)
    print(f"All figures saved to: {fig_dir}/")
    print("=" * 60)

    return {
        "noise_sweep":      (sigmas, noise_results),
        "metrics":          metrics_table,
        "per_class":        per_class,
        "feature_weights":  feat_weights,
        "ranked_features":  ranked_idx,
        "temperature":      T,
    }
