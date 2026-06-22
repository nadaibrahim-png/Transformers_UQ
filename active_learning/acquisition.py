# =============================================================================
# active_learning/acquisition.py — Acquisition functions
# =============================================================================
# An acquisition function scores each unlabeled sample and returns the indices
# of the K most "informative" ones to query (i.e. send for labeling).
#
# Three functions are implemented:
#
#   1. random_acquisition   — baseline: pick K samples at random
#   2. entropy_acquisition  — pick K samples with highest predictive entropy
#                             Paper connection [GAL17 §2]: "query the point
#                             about which the model is most uncertain"
#   3. bald_acquisition     — Bayesian Active Learning by Disagreement (BALD)
#                             Paper connection [GAL17 §3.1 / HOUL11]:
#                             maximises mutual information between prediction
#                             and model parameters — captures epistemic
#                             uncertainty only, not aleatoric noise.
#
# All functions share the same signature:
#   inputs  : model, pool DataLoader, k (budget), n_mc_samples
#   returns : numpy array of K indices into the pool

import numpy as np
import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mc_samples(model, loader, n_samples):
    """
    Collect T stochastic softmax outputs for every sample in the pool.

    Returns:
        all_samples : (n_pool, T, n_classes)  — T predictions per sample
    """
    model.enable_mc_dropout()
    all_samples = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(DEVICE)
            # (T, batch, n_classes)
            batch_samples = torch.stack([
                torch.softmax(model(X_batch), dim=-1)
                for _ in range(n_samples)
            ], dim=0)
            # → (batch, T, n_classes)
            all_samples.append(batch_samples.permute(1, 0, 2).cpu().numpy())

    return np.vstack(all_samples)   # (n_pool, T, n_classes)


# ── 1. Random ─────────────────────────────────────────────────────────────────

def random_acquisition(model, pool_loader, k, n_mc_samples=None):
    """
    Baseline: select K samples uniformly at random from the pool.
    No model inference needed.
    """
    n_pool = sum(X.shape[0] for X, _ in pool_loader)
    return np.random.choice(n_pool, size=k, replace=False)


# ── 2. Entropy ────────────────────────────────────────────────────────────────

def entropy_acquisition(model, pool_loader, k, n_mc_samples=None):
    """
    Maximum Entropy: query the K samples with the highest predictive entropy.

        H[y | x, D] = -Σ_c  p̄_c · log p̄_c

    where p̄_c = (1/T) Σ_t p_c^(t)  is the mean MC Dropout prediction.

    Paper connection [GAL17 §2]:
        "We select the point x* = argmax H[y | x, D_train]"
        This picks samples where the average prediction is most spread out
        across classes — the model "doesn't know which class to pick."

    Limitation: entropy captures TOTAL uncertainty (epistemic + aleatoric).
    BALD below separates them.
    """
    n_mc_samples = n_mc_samples or config.MC_SAMPLES
    samples = _mc_samples(model, pool_loader, n_mc_samples)
    # Mean across MC passes → (n_pool, n_classes)
    mean_probs = samples.mean(axis=1)
    # Predictive entropy → (n_pool,)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-9), axis=1)
    # Return indices of top-K highest entropy
    return np.argsort(entropy)[-k:][::-1]


# ── 3. BALD ───────────────────────────────────────────────────────────────────

def bald_acquisition(model, pool_loader, k, n_mc_samples=None):
    """
    BALD — Bayesian Active Learning by Disagreement.

    Paper connection [GAL17 §3.1 / HOUL11]:
        BALD(x) = H[y | x, D] − E_{ω~q(ω)} [ H[y | x, ω] ]

        = Predictive entropy  −  Mean entropy of individual MC passes

        = TOTAL uncertainty   −  ALEATORIC uncertainty
        = EPISTEMIC uncertainty alone

    Why is this better than plain entropy?
        A sample near a decision boundary but with noisy labels has HIGH
        total entropy but LOW BALD score — the disagreement is due to
        noise, not model ignorance. BALD filters that out.

    In practice: BALD queries samples where the MC Dropout models
    DISAGREE with each other — i.e. different "neural network samples"
    give very different predictions.

    Args:
        n_mc_samples : T — number of stochastic forward passes
        k            : query budget per round

    Returns:
        indices of top-K samples by BALD score
    """
    n_mc_samples = n_mc_samples or config.MC_SAMPLES
    samples = _mc_samples(model, pool_loader, n_mc_samples)
    # samples: (n_pool, T, n_classes)

    # Predictive entropy of the mean → H[y | x, D]
    mean_probs = samples.mean(axis=1)                          # (n_pool, n_classes)
    H_total = -np.sum(mean_probs * np.log(mean_probs + 1e-9), axis=1)  # (n_pool,)

    # Mean entropy of each MC sample → E[H[y | x, ω]]
    H_each  = -np.sum(samples * np.log(samples + 1e-9), axis=2)  # (n_pool, T)
    H_mean  = H_each.mean(axis=1)                                  # (n_pool,)

    # BALD = epistemic uncertainty
    bald_scores = H_total - H_mean                                 # (n_pool,)

    return np.argsort(bald_scores)[-k:][::-1]


# ── Registry ──────────────────────────────────────────────────────────────────

ACQUISITION_FUNCTIONS = {
    "random":  random_acquisition,
    "entropy": entropy_acquisition,
    "bald":    bald_acquisition,
}
