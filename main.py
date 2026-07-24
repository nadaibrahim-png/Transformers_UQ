# =============================================================================
# main.py  —  FT-Transformer: calibration + epistemic UQ (5 methods)
# =============================================================================
# BASE CONFIG runs all five UQ methods:
#   1. Standard inference        (baseline, single forward pass)
#   2. MC Dropout  T=30          [GAL16]
#   3. Temperature Scaling       [GUO17]
#   4. Deep Ensembles  M=5       [LAKSHMINARAYANAN17]
#   5. SWAG-Diagonal   K=30      [MADDOX19]
#
# ABLATION SWEEP uses Standard / MC / TS only (Ensemble+SWAG too costly xN).
# =============================================================================
import torch
import numpy as np
import os
import config

from data        import get_dataset, preprocess, make_ood, get_loaders
from models      import FTTransformer
from training    import train
from evaluation  import (
    compute_all, compute_ece_per_class, compute_balanced_accuracy,
    predict_standard, predict_mc_dropout,
    TemperatureScaler, predict_temperature_scaled,
    DeepEnsemble, SWAG,
)
from visualization import (
    plot_reliability_diagrams, plot_ece_comparison, plot_entropy_ood,
    plot_results_table, plot_training_curves, plot_sweep_results,
    plot_attention_heatmap,
)

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(config.FIGURES_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# evaluate_config  —  Standard / MC / TS  (used by both base and sweep)
# ----------------------------------------------------------------------------

def evaluate_config(model, loaders):
    probs_std, labels = predict_standard(model, loaders["test"])

    probs_mc, _, unc_test = predict_mc_dropout(model, loaders["test"])
    _,        _, unc_ood  = predict_mc_dropout(model, loaders["ood"])

    ts = TemperatureScaler(model).to(DEVICE)
    T  = ts.fit(loaders["val"])
    probs_ts, _ = predict_temperature_scaled(ts, loaders["test"])

    return {
        "std": {"probs": probs_std, "labels": labels,
                "metrics": compute_all(probs_std, labels)},
        "mc":  {"probs": probs_mc,  "labels": labels,
                "metrics": compute_all(probs_mc, labels),
                "unc_test": unc_test, "unc_ood": unc_ood},
        "ts":  {"probs": probs_ts,  "labels": labels,
                "metrics": compute_all(probs_ts, labels), "T": T},
    }


# ----------------------------------------------------------------------------
# evaluate_base_full  —  adds Ensemble + SWAG
# ----------------------------------------------------------------------------

def evaluate_base_full(model, loaders, model_kwargs):
    results = evaluate_config(model, loaders)

    # -- Method 4: Deep Ensemble ----------------------------------------------
    print(f"\n-- Deep Ensemble (M={config.ENSEMBLE_M}) --")
    ensemble = DeepEnsemble(FTTransformer, model_kwargs, M=config.ENSEMBLE_M)
    ensemble.train_all(loaders["train"], loaders["val"],
                       epochs=config.EPOCHS,
                       seeds=list(range(config.ENSEMBLE_M)))

    p_e, l_e, u_e, epi_e = ensemble.predict(loaders["test"])
    _,   _,   u_eo, epi_eo = ensemble.predict(loaders["ood"])
    results["ensemble"] = {
        "probs": p_e, "labels": l_e, "metrics": compute_all(p_e, l_e),
        "unc_test": u_e, "unc_ood": u_eo,
        "epistemic_test": epi_e, "epistemic_ood": epi_eo,
        "object": ensemble,
    }

    # -- Method 5: SWAG-Diagonal ----------------------------------------------
    print(f"\n-- SWAG (epochs={config.SWAG_EPOCHS}, K={config.SWAG_K}) --")
    swag = SWAG(model, max_snapshots=config.SWAG_EPOCHS)
    swag.collect_during_training(loaders["train"], loaders["val"],
                                 epochs=config.SWAG_EPOCHS, lr=config.SWAG_LR)

    p_s, l_s, u_s = swag.predict(loaders["test"], K=config.SWAG_K)
    _,   _,   u_so = swag.predict(loaders["ood"],  K=config.SWAG_K)
    results["swag"] = {
        "probs": p_s, "labels": l_s, "metrics": compute_all(p_s, l_s),
        "unc_test": u_s, "unc_ood": u_so, "object": swag,
    }

    return results


# ----------------------------------------------------------------------------
# Ablation sweep
# ----------------------------------------------------------------------------

def run_sweep(loaders, n_features, n_classes):
    base  = {"d_model": config.D_MODEL, "n_layers": config.N_LAYERS,
             "dropout": config.DROPOUT}
    sweep_results = []
    seen = set()

    for param_name, values in config.SWEEP.items():
        for val in values:
            cfg = {**base, param_name: val}
            key = (cfg["d_model"], cfg["n_layers"], cfg["dropout"])
            if key in seen:
                continue
            seen.add(key)
            label = f"{param_name}={val}"
            print(f"\n{'='*60}\n  Config: {cfg}  [{label}]\n{'='*60}")

            model = FTTransformer(n_features=n_features, n_classes=n_classes,
                                  d_model=cfg["d_model"], nhead=4,
                                  num_layers=cfg["n_layers"],
                                  dropout=cfg["dropout"])
            print(f"  Parameters: {model.count_parameters():,}")
            history = train(model, loaders, epochs=config.SWEEP_EPOCHS)
            results = evaluate_config(model, loaders)
            _print_config_table(label, results)
            sweep_results.append({"label": label, "cfg": cfg,
                                   "results": results, "history": history,
                                   "model": model})

    return sweep_results


# ----------------------------------------------------------------------------
# Base config  —  full run + all figures
# ----------------------------------------------------------------------------

def run_base_config(loaders, n_features, n_classes, dataset_name, feature_names):
    print(f"\n{'='*60}\n  BASE CONFIG — all 5 UQ methods\n{'='*60}")

    model = FTTransformer(n_features=n_features, n_classes=n_classes)
    print(f"  Parameters: {model.count_parameters():,}")
    history = train(model, loaders)

    model_kwargs = dict(n_features=n_features, n_classes=n_classes,
                        d_model=config.D_MODEL, nhead=config.N_HEAD,
                        num_layers=config.N_LAYERS, dropout=config.DROPOUT)

    results = evaluate_base_full(model, loaders, model_kwargs)
    T_value = results["ts"]["T"]
    _print_base_table(results)

    # Per-class ECE
    print("\n-- Per-class ECE --")
    for key, name in [("std","Standard"), ("mc","MC Dropout"),
                      ("ts","Temp. Scaling"), ("ensemble","Ensemble"),
                      ("swag","SWAG")]:
        pce = compute_ece_per_class(results[key]["probs"], results[key]["labels"])
        ba  = compute_balanced_accuracy(results[key]["probs"], results[key]["labels"])
        print(f"  {name:<20}  ve_ECE={pce[0]:.4f}  vm_ECE={pce[1]:.4f}"
              f"  BalAcc={ba:.4f}")

    # Standard figures
    print(f"\n-- Generating figures -> {config.FIGURES_DIR}/ --")
    plot_training_curves(history)
    plot_reliability_diagrams(results)
    plot_ece_comparison(results)
    plot_entropy_ood(results["mc"]["unc_test"], results["mc"]["unc_ood"])
    plot_results_table(results, dataset_name, T_value)
    sample_x = next(iter(loaders["test"]))[0][:256].to(DEVICE)
    plot_attention_heatmap(model.get_attention_weights(sample_x), feature_names)

    # Extra figures: 3-way OOD + epistemic decomposition
    _plot_ood_comparison(results)
    _plot_epistemic_decomposition(results)

    return results, model


# ----------------------------------------------------------------------------
# Extra figure helpers
# ----------------------------------------------------------------------------

def _plot_ood_comparison(results):
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle("Predictive Entropy: IND vs OOD  (3 methods)", fontsize=12)
        panels = [
            ("mc",       f"MC Dropout (T={config.MC_SAMPLES})",        "#4C72B0"),
            ("ensemble", f"Deep Ensemble (M={config.ENSEMBLE_M})", "#DD8452"),
            ("swag",     f"SWAG (K={config.SWAG_K})",             "#55A868"),
        ]
        for ax, (key, name, color) in zip(axes, panels):
            ut = results[key]["unc_test"]
            uo = results[key]["unc_ood"]
            ax.hist(ut, bins=50, density=True, alpha=0.6,
                    color=color,    label=f"IND  med={np.median(ut):.3f}")
            ax.hist(uo, bins=50, density=True, alpha=0.6,
                    color="orange", label=f"OOD  med={np.median(uo):.3f}")
            ax.axvline(np.median(ut), color=color,        linestyle="--", lw=1.5)
            ax.axvline(np.median(uo), color="darkorange", linestyle="--", lw=1.5)
            ax.set_title(name, fontsize=11)
            ax.set_xlabel("H(y|x)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=9)
        plt.tight_layout()
        path = os.path.join(config.FIGURES_DIR, "fig_ood_3way.png")
        fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")
    except Exception as e:
        print(f"  [warn] _plot_ood_comparison: {e}")


def _plot_epistemic_decomposition(results):
    try:
        import matplotlib.pyplot as plt
        ens       = results["ensemble"]
        total     = ens["unc_test"]
        epistemic = ens["epistemic_test"]
        aleatoric = np.maximum(total - epistemic, 0.0)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle("Uncertainty Decomposition  (Deep Ensemble, test set)\n"
                     "H[p_bar] = MI(y;theta|x)  +  E[H[p|x,theta]]", fontsize=11)
        for ax, arr, label, color in zip(
            axes,
            [total, epistemic, aleatoric],
            ["Total  H[p_bar]", "Epistemic  MI", "Aleatoric  E[H]"],
            ["steelblue", "firebrick", "seagreen"],
        ):
            ax.hist(arr, bins=60, color=color, alpha=0.75)
            med = np.median(arr)
            ax.axvline(med, color="black", linestyle="--", lw=1.5,
                       label=f"median={med:.3f}")
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("Entropy (nats)")
            ax.set_ylabel("Count")
            ax.legend(fontsize=9)
        plt.tight_layout()
        path = os.path.join(config.FIGURES_DIR, "fig_uncertainty_decomposition.png")
        fig.savefig(path, dpi=config.DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")
    except Exception as e:
        print(f"  [warn] _plot_epistemic_decomposition: {e}")


# ----------------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------------

def run_all():
    X, y, n_classes, dataset_name, feature_names = get_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = preprocess(X, y)
    X_ood   = make_ood(X_test)
    loaders = get_loaders(X_train, y_train, X_val, y_val, X_test, y_test, X_ood)
    n_features = X_train.shape[1]

    print(f"\nDataset  : {dataset_name}")
    print(f"Features : {n_features}  |  Classes: {n_classes}")
    print(f"Device   : {DEVICE}")

    base_results, _ = run_base_config(
        loaders, n_features, n_classes, dataset_name, feature_names
    )

    print(f"\n{'='*60}\n  ABLATION SWEEP  (Standard / MC / TS)\n{'='*60}")
    sweep_results = run_sweep(loaders, n_features, n_classes)
    plot_sweep_results(sweep_results)

    print("\n  All experiments complete.")
    print(f"  Figures saved to {config.FIGURES_DIR}/")
    return base_results, sweep_results


# ----------------------------------------------------------------------------
# Print helpers
# ----------------------------------------------------------------------------

def _print_config_table(label, results):
    print(f"\n  {'─'*58}\n  Results for: {label}\n  {'─'*58}")
    print(f"  {'Method':<28} {'Accuracy':>9} {'ECE':>9} {'NLL':>9}")
    print(f"  {'─'*58}")
    for key, name in [("std", "Standard (Baseline)"),
                      ("mc",  f"MC Dropout (T={config.MC_SAMPLES})"),
                      ("ts",  "Temperature Scaling")]:
        m = results[key]["metrics"]
        print(f"  {name:<28} {m['accuracy']:>9.4f} {m['ece']:>9.4f} {m['nll']:>9.4f}")
    if "T" in results["ts"]:
        print(f"  Learned T = {results['ts']['T']:.4f}")
    print(f"  {'─'*58}")


def _print_base_table(results):
    print(f"\n  {'─'*72}\n  BASE CONFIG — all 5 UQ methods\n  {'─'*72}")
    print(f"  {'Method':<30} {'Accuracy':>9} {'ECE':>9} {'NLL':>9} {'Brier':>8}")
    print(f"  {'─'*72}")
    for key, name in [
        ("std",      "Standard (Baseline)"),
        ("mc",       f"MC Dropout (T={config.MC_SAMPLES})"),
        ("ts",       "Temperature Scaling"),
        ("ensemble", f"Deep Ensemble (M={config.ENSEMBLE_M})"),
        ("swag",     f"SWAG (K={config.SWAG_K})"),
    ]:
        m = results[key]["metrics"]
        brier = m.get("brier", float("nan"))
        print(f"  {name:<30} {m['accuracy']:>9.4f} {m['ece']:>9.4f}"
              f" {m['nll']:>9.4f} {brier:>8.4f}")
    print(f"  Learned T = {results['ts']['T']:.4f}")
    print(f"  {'─'*72}")


if __name__ == "__main__":
    run_all()
