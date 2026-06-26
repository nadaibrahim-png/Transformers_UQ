# =============================================================================
# main.py — Hyperparameter ablation study with calibration evaluation
# =============================================================================
# Experiment: How does FT-Transformer calibration change under different
# architectural configurations?
#
# Ablation design: vary ONE hyperparameter at a time, others fixed to defaults.
#   d_model  ∈ {32, 64, 128}   — model capacity
#   n_layers ∈ {1, 2, 4}       — depth
#   dropout  ∈ {0.1, 0.2, 0.3} — regularisation + MC uncertainty
#
# For each config, three UQ methods are evaluated:
#   1. Standard inference   — baseline, typically overconfident
#   2. MC Dropout           — 30 stochastic passes [GAL16]
#   3. Temperature Scaling  — single-parameter post-hoc fix [GUO17]
#
# Key metric: ECE (Expected Calibration Error) — lower is better.
# =============================================================================

import torch
import numpy as np
import os
import config

from data        import get_dataset, preprocess, make_ood, get_loaders
from models      import FTTransformer
from training    import train
from evaluation  import (compute_all,
                          predict_standard,
                          predict_mc_dropout,
                          TemperatureScaler,
                          predict_temperature_scaled)
from visualization import (plot_reliability_diagrams,
                            plot_ece_comparison,
                            plot_entropy_ood,
                            plot_results_table,
                            plot_training_curves,
                            plot_sweep_results,
                            plot_attention_heatmap)

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(config.FIGURES_DIR, exist_ok=True)


# ── Single-config evaluation ──────────────────────────────────────────────────

def evaluate_config(model, loaders):
    """
    Run all three UQ methods on a trained model and return metrics dict.

    Returns:
        results : {
            'std': {'probs', 'labels', 'metrics'},
            'mc':  {'probs', 'labels', 'metrics', 'unc_test', 'unc_ood'},
            'ts':  {'probs', 'labels', 'metrics', 'T'},
        }
    """
    # Standard
    probs_std, labels = predict_standard(model, loaders["test"])

    # MC Dropout
    probs_mc,  _, unc_test = predict_mc_dropout(model, loaders["test"])
    _,         _, unc_ood  = predict_mc_dropout(model, loaders["ood"])

    # Temperature Scaling — move to DEVICE to avoid device mismatch
    ts = TemperatureScaler(model).to(DEVICE)
    T  = ts.fit(loaders["val"])
    probs_ts, _ = predict_temperature_scaled(ts, loaders["test"])

    return {
        "std": {
            "probs":   probs_std,
            "labels":  labels,
            "metrics": compute_all(probs_std, labels),
        },
        "mc": {
            "probs":    probs_mc,
            "labels":   labels,
            "metrics":  compute_all(probs_mc, labels),
            "unc_test": unc_test,
            "unc_ood":  unc_ood,
        },
        "ts": {
            "probs":   probs_ts,
            "labels":  labels,
            "metrics": compute_all(probs_ts, labels),
            "T":       T,
        },
    }


# ── Hyperparameter ablation sweep ─────────────────────────────────────────────

def run_sweep(loaders, n_features, n_classes):
    """
    Ablation study: vary d_model / n_layers / dropout one at a time.

    For each config:
        train FT-Transformer → evaluate Standard / MC Dropout / Temp Scaling

    Returns:
        sweep_results : list of dicts, each containing:
            {label, cfg, results, history}
    """
    base = {
        "d_model":  config.D_MODEL,
        "n_layers": config.N_LAYERS,
        "dropout":  config.DROPOUT,
    }

    sweep_results = []
    seen = set()   # avoid re-running identical configs

    for param_name, values in config.SWEEP.items():
        for val in values:
            cfg = {**base, param_name: val}
            key = (cfg["d_model"], cfg["n_layers"], cfg["dropout"])
            if key in seen:
                continue
            seen.add(key)

            label = f"{param_name}={val}"
            print(f"\n{'='*60}")
            print(f"  Config: {cfg}  [{label}]")
            print(f"{'='*60}")

            model = FTTransformer(
                n_features=n_features,
                n_classes=n_classes,
                d_model=cfg["d_model"],
                nhead=4,               # 4 divides 32, 64, 128 equally
                num_layers=cfg["n_layers"],
                dropout=cfg["dropout"],
            )
            print(f"  Parameters: {model.count_parameters():,}")

            history = train(model, loaders, epochs=config.SWEEP_EPOCHS)
            results = evaluate_config(model, loaders)

            _print_config_table(label, results)

            sweep_results.append({
                "label":   label,
                "cfg":     cfg,
                "results": results,
                "history": history,
                "model":   model,
            })

    return sweep_results


# ── Base config — full figures ─────────────────────────────────────────────────

def run_base_config(loaders, n_features, n_classes, dataset_name, feature_names):
    """
    Train the DEFAULT config (d_model=64, n_layers=2, dropout=0.2) at full
    epochs and generate all poster figures for that single config.
    """
    print(f"\n{'='*60}")
    print("  BASE CONFIG — full training + all figures")
    print(f"{'='*60}")

    model = FTTransformer(n_features=n_features, n_classes=n_classes)
    print(f"  Parameters: {model.count_parameters():,}")

    history = train(model, loaders)
    results = evaluate_config(model, loaders)
    T_value = results["ts"]["T"]

    _print_config_table("Base config", results)

    # ── Poster figures ────────────────────────────────────────────────────────
    print(f"\n── Generating figures → {config.FIGURES_DIR}/ ──")

    plot_training_curves(history)
    plot_reliability_diagrams(results)
    plot_ece_comparison(results)
    plot_entropy_ood(
        results["mc"]["unc_test"],
        results["mc"]["unc_ood"]
    )
    plot_results_table(results, dataset_name, T_value)

    # Attention heatmap — implicit feature selection
    sample_x = next(iter(loaders["test"]))[0][:256].to(DEVICE)
    attn = model.get_attention_weights(sample_x)
    plot_attention_heatmap(attn, feature_names)

    return results, model


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_all():
    """Load data, run base config figures, then run hyperparameter sweep."""

    # ── Data ─────────────────────────────────────────────────────────────────
    X, y, n_classes, dataset_name, feature_names = get_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = preprocess(X, y)
    X_ood   = make_ood(X_test)
    loaders = get_loaders(X_train, y_train, X_val, y_val, X_test, y_test, X_ood)
    n_features = X_train.shape[1]

    print(f"\nDataset  : {dataset_name}")
    print(f"Features : {n_features}  |  Classes: {n_classes}")
    print(f"Device   : {DEVICE}")

    # ── Base config ───────────────────────────────────────────────────────────
    base_results, base_model = run_base_config(
        loaders, n_features, n_classes, dataset_name, feature_names
    )

    # ── Ablation sweep ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  HYPERPARAMETER ABLATION SWEEP")
    print(f"{'='*60}")

    sweep_results = run_sweep(loaders, n_features, n_classes)
    plot_sweep_results(sweep_results)

    print("\n✓ All experiments complete.")
    print(f"✓ Figures saved to {config.FIGURES_DIR}/")

    return base_results, sweep_results


# ── Helper ────────────────────────────────────────────────────────────────────

def _print_config_table(label, results):
    print(f"\n  {'─'*52}")
    print(f"  Results for: {label}")
    print(f"  {'─'*52}")
    print(f"  {'Method':<26} {'Accuracy':>9} {'ECE':>9} {'NLL':>9}")
    print(f"  {'─'*52}")
    for key, name in [("std", "Standard (Baseline)"),
                      ("mc",  f"MC Dropout (T={config.MC_SAMPLES})"),
                      ("ts",  "Temperature Scaling")]:
        m = results[key]["metrics"]
        print(f"  {name:<26} {m['accuracy']:>9.4f} {m['ece']:>9.4f} {m['nll']:>9.4f}")
    if "T" in results["ts"]:
        print(f"  Learned T = {results['ts']['T']:.4f}")
    print(f"  {'─'*52}")


if __name__ == "__main__":
    run_all()
