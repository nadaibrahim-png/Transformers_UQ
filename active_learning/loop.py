# =============================================================================
# active_learning/loop.py — Pool-based active learning loop
# =============================================================================
# Implements the standard pool-based active learning protocol:
#
#   Round 0:  Train on a small initial labeled set (seed set)
#   Round r:  Score unlabeled pool → query K samples → add to labeled set
#             → retrain model from scratch → evaluate on test set
#   Repeat for N rounds, tracking test accuracy at each round.
#
# Paper connection [GAL17 §4]:
#   "We start with a small labeled dataset and iteratively select
#    informative points from the unlabeled pool."
#
# We compare three acquisition strategies:
#   - Random  (baseline)
#   - Entropy (max predictive entropy via MC Dropout)
#   - BALD    (epistemic uncertainty only)
#
# The result is a LEARNING CURVE: accuracy vs number of labeled samples.
# A good acquisition function reaches high accuracy with fewer labels.

import numpy as np
import torch
import copy
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config
from data       import to_loader
from models     import TabTransformer
from training   import train
from evaluation import compute_accuracy, predict_standard
from .acquisition import ACQUISITION_FUNCTIONS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_active_learning(
    X_train, y_train,
    X_test,  y_test,
    n_classes,
    strategy="entropy",
    initial_size=None,
    query_size=None,
    n_rounds=None,
    epochs_per_round=None,
    verbose=True,
):
    """
    Run one full active learning experiment with a given acquisition strategy.

    Args:
        X_train, y_train : full training pool (unlabeled in AL terms)
        X_test,  y_test  : held-out test set (never queried)
        n_classes        : number of output classes
        strategy         : "random" | "entropy" | "bald"
        initial_size     : number of seed labeled samples (default: config.AL_INITIAL_SIZE)
        query_size       : samples to query per round   (default: config.AL_QUERY_SIZE)
        n_rounds         : number of AL rounds          (default: config.AL_N_ROUNDS)
        epochs_per_round : training epochs per round    (default: config.AL_EPOCHS_PER_ROUND)

    Returns:
        labeled_counts : list of labeled set sizes at each round
        accuracies     : list of test accuracies at each round
    """
    initial_size     = initial_size     or config.AL_INITIAL_SIZE
    query_size       = query_size       or config.AL_QUERY_SIZE
    n_rounds         = n_rounds         or config.AL_N_ROUNDS
    epochs_per_round = epochs_per_round or config.AL_EPOCHS_PER_ROUND

    acquire = ACQUISITION_FUNCTIONS[strategy]
    n_features = X_train.shape[1]

    # ── Initialise labeled / unlabeled split ──────────────────────────────────
    # Seed set: stratified random sample so all classes are represented
    all_indices = np.arange(len(X_train))
    labeled_idx = _stratified_seed(y_train, initial_size, n_classes)
    pool_idx    = np.setdiff1d(all_indices, labeled_idx)

    labeled_counts = []
    accuracies     = []

    if verbose:
        print(f"\n── Active Learning: {strategy.upper()} ──")
        print(f"  Initial labeled: {len(labeled_idx)} | "
              f"Pool: {len(pool_idx)} | "
              f"Query/round: {query_size} | "
              f"Rounds: {n_rounds}")

    for round_idx in range(n_rounds + 1):
        n_labeled = len(labeled_idx)

        # ── Train model on current labeled set ───────────────────────────────
        model = TabTransformer(n_features=n_features, n_classes=n_classes)

        # Make a small val split from labeled set (20%) for trainer
        if n_labeled >= 10:
            val_cut  = max(2, int(0.2 * n_labeled))
            lbl_perm = np.random.permutation(labeled_idx)
            lbl_val  = lbl_perm[:val_cut]
            lbl_tr   = lbl_perm[val_cut:]
        else:
            lbl_tr = lbl_val = labeled_idx

        loaders = {
            "train": to_loader(X_train[lbl_tr], y_train[lbl_tr], shuffle=True),
            "val":   to_loader(X_train[lbl_val], y_train[lbl_val]),
        }

        # Suppress per-epoch prints inside the loop for cleanliness
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            train(model, loaders, epochs=epochs_per_round)

        # ── Evaluate on test set ─────────────────────────────────────────────
        test_loader = to_loader(X_test, y_test)
        probs, labels = predict_standard(model, test_loader)
        acc = compute_accuracy(probs, labels)

        labeled_counts.append(n_labeled)
        accuracies.append(acc)

        if verbose:
            print(f"  Round {round_idx:2d} | Labeled: {n_labeled:4d} | Test acc: {acc:.4f}")

        # ── Query next batch (skip on last round) ────────────────────────────
        if round_idx < n_rounds and len(pool_idx) >= query_size:
            pool_loader = to_loader(X_train[pool_idx], y_train[pool_idx])
            query_local = acquire(model, pool_loader, k=query_size)
            query_global = pool_idx[query_local]

            labeled_idx = np.concatenate([labeled_idx, query_global])
            pool_idx    = np.setdiff1d(pool_idx, query_global)

    return labeled_counts, accuracies


def run_all_strategies(X_train, y_train, X_test, y_test, n_classes, **kwargs):
    """
    Run active learning with all three strategies and return results dict.
    Used by main.py and visualization.

    Returns:
        results : {"random": (counts, accs), "entropy": ..., "bald": ...}
    """
    results = {}
    for strategy in ["random", "entropy", "bald"]:
        counts, accs = run_active_learning(
            X_train, y_train, X_test, y_test, n_classes,
            strategy=strategy, **kwargs
        )
        results[strategy] = {"labeled_counts": counts, "accuracies": accs}
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stratified_seed(y, n, n_classes):
    """
    Sample n indices with at least one example per class.
    Ensures the seed set is not accidentally class-imbalanced.
    """
    indices = []
    per_class = max(1, n // n_classes)
    for c in range(n_classes):
        class_idx = np.where(y == c)[0]
        chosen    = np.random.choice(class_idx, size=min(per_class, len(class_idx)), replace=False)
        indices.extend(chosen.tolist())
    # Top up to exactly n if needed
    remaining = np.setdiff1d(np.arange(len(y)), indices)
    shortfall = n - len(indices)
    if shortfall > 0:
        indices.extend(np.random.choice(remaining, size=shortfall, replace=False).tolist())
    return np.array(indices[:n])
