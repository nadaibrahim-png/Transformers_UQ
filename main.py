# =============================================================================
# main.py — Entry point: runs the full experiment end-to-end
# =============================================================================
# Run in Colab:
#   !pip install netcal --quiet
#   !python main.py
#
# Or from a notebook:
#   from main import run_calibration, run_active_learning_experiment, run_all
#
# To switch dataset: change config.DATASET before running.
# =============================================================================

import torch
import numpy as np
import config

from data           import get_dataset, preprocess, make_ood, get_loaders
from models         import TabTransformer
from training       import train
from evaluation     import (compute_all,
                             predict_standard,
                             predict_mc_dropout,
                             TemperatureScaler,
                             predict_temperature_scaled)
from active_learning import run_all_strategies
from visualization  import (plot_reliability_diagrams,
                             plot_ece_comparison,
                             plot_entropy_ood,
                             plot_results_table,
                             plot_training_curves,
                             plot_learning_curves)

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


# ── Experiment A: Calibration ─────────────────────────────────────────────────

def run_calibration(X_train, y_train, X_val, y_val, X_test, y_test, X_ood,
                    n_features, n_classes, dataset_name):
    """
    Train a Transformer and evaluate three calibration methods:
    Standard baseline, MC Dropout, and Temperature Scaling.
    Generates figures 1–5.
    """
    print("\n" + "="*60)
    print("EXPERIMENT A: Calibration & Uncertainty")
    print("="*60)

    loaders = get_loaders(X_train, y_train, X_val, y_val, X_test, y_test, X_ood)

    # Train
    model = TabTransformer(n_features=n_features, n_classes=n_classes)
    print(f"Parameters: {model.count_parameters():,}")
    history = train(model, loaders)

    # Inference — Standard
    probs_std, labels_test = predict_standard(model, loaders["test"])

    # Inference — MC Dropout
    probs_mc, _, uncertainty_test = predict_mc_dropout(model, loaders["test"])
    _,        _, uncertainty_ood  = predict_mc_dropout(model, loaders["ood"])

    # Inference — Temperature Scaling
    print("\n── Fitting Temperature Scaling ──")
    ts = TemperatureScaler(model)
    T_value = ts.fit(loaders["val"])
    probs_ts, _ = predict_temperature_scaled(ts, loaders["test"])

    # Metrics
    results = {
        "std": {"probs": probs_std, "labels": labels_test, "metrics": compute_all(probs_std, labels_test)},
        "mc":  {"probs": probs_mc,  "labels": labels_test, "metrics": compute_all(probs_mc,  labels_test)},
        "ts":  {"probs": probs_ts,  "labels": labels_test, "metrics": compute_all(probs_ts,  labels_test)},
    }

    _print_calibration_table(results, T_value)

    # Figures
    print(f"\n── Generating calibration figures → {config.FIGURES_DIR}/ ──")
    plot_training_curves(history)
    plot_reliability_diagrams(results)
    plot_ece_comparison(results)
    plot_entropy_ood(uncertainty_test, uncertainty_ood)
    plot_results_table(results, dataset_name, T_value)

    return results, model


# ── Experiment B: Active Learning ─────────────────────────────────────────────

def run_active_learning_experiment(X_train, y_train, X_test, y_test, n_classes):
    """
    Compare three acquisition strategies on the same dataset:
      - Random  (baseline)
      - Entropy (max predictive entropy, MC Dropout)
      - BALD    (epistemic uncertainty only)

    Paper connection [GAL17]:
        'We show that BALD and entropy-based acquisition consistently
         outperform random baselines across datasets.'

    Generates figure 6 (learning curves).
    """
    print("\n" + "="*60)
    print("EXPERIMENT B: Active Learning")
    print("="*60)
    print(f"  Initial seed  : {config.AL_INITIAL_SIZE} samples")
    print(f"  Query budget  : {config.AL_QUERY_SIZE} per round")
    print(f"  Rounds        : {config.AL_N_ROUNDS}")
    print(f"  Epochs/round  : {config.AL_EPOCHS_PER_ROUND}")

    al_results = run_all_strategies(
        X_train, y_train,
        X_test,  y_test,
        n_classes=n_classes
    )

    # Print summary
    print(f"\n{'Strategy':<12} {'Final Acc':>10} {'Labeled':>10}")
    print("-" * 35)
    for strategy, data in al_results.items():
        print(f"{strategy:<12} {data['accuracies'][-1]:>10.4f} "
              f"{data['labeled_counts'][-1]:>10}")

    # Figure
    print(f"\n── Generating active learning figure → {config.FIGURES_DIR}/ ──")
    plot_learning_curves(al_results)

    return al_results


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_all():
    """Run both experiments end-to-end."""

    # ── Data ─────────────────────────────────────────────────────────────────
    X, y, n_classes, dataset_name, feature_names = get_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = preprocess(X, y)
    X_ood = make_ood(X_test)
    n_features = X_train.shape[1]

    print(f"\nDataset   : {dataset_name}")
    print(f"Features  : {n_features}  |  Classes: {n_classes}")

    # ── Experiment A: Calibration ─────────────────────────────────────────────
    cal_results, model = run_calibration(
        X_train, y_train, X_val, y_val, X_test, y_test, X_ood,
        n_features, n_classes, dataset_name
    )

    # ── Experiment B: Active Learning ─────────────────────────────────────────
    al_results = run_active_learning_experiment(
        X_train, y_train, X_test, y_test, n_classes
    )

    print("\n✓ All experiments complete. Figures saved to figures/")
    return cal_results, al_results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_calibration_table(results, T_value):
    print(f"\n{'='*62}")
    print(f"{'CALIBRATION RESULTS':^62}")
    print(f"{'='*62}")
    print(f"{'Method':<28} {'Accuracy':>9} {'ECE':>9} {'NLL':>9}")
    print(f"{'-'*62}")
    for key, label in [("std", "Standard (Baseline)"),
                        ("mc",  f"MC Dropout (T={config.MC_SAMPLES})"),
                        ("ts",  "Temperature Scaling")]:
        m = results[key]["metrics"]
        print(f"{label:<28} {m['accuracy']:>9.4f} {m['ece']:>9.4f} {m['nll']:>9.4f}")
    print(f"{'='*62}")
    print(f"Learned temperature T = {T_value:.4f}")


if __name__ == "__main__":
    run_all()
