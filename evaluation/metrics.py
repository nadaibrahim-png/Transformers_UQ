# =============================================================================
# evaluation/metrics.py — Calibration metrics
# =============================================================================
# Paper connections:
#   ECE + Reliability diagrams : Guo et al. 2017 (arXiv:1706.04599)
#   NLL                        : standard probabilistic metric

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


def compute_ece(probs, labels, n_bins=None):
    """
    Expected Calibration Error — measures the gap between confidence and accuracy.

    Paper connection [GUO17 §3]:
        ECE = Σ_b (|B_b| / n) × |acc(B_b) − conf(B_b)|
        where B_b is the set of samples whose confidence falls in bin b.

    Perfect calibration → ECE = 0.
    Typical untreated neural net → ECE = 0.05–0.15.

    Args:
        probs  : (n_samples, n_classes) softmax probabilities
        labels : (n_samples,) true class indices
        n_bins : number of confidence bins (default: config.N_BINS = 10)

    Returns:
        ece : scalar float
    """
    n_bins = n_bins or config.N_BINS
    confidences = probs.max(axis=1)          # highest class probability
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc  = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_w    = mask.sum() / len(labels)
        ece     += bin_w * abs(bin_conf - bin_acc)

    return float(ece)


def compute_nll(probs, labels):
    """
    Negative Log-Likelihood — penalises confident wrong predictions heavily.

    NLL = -(1/n) Σ log p(true_class | xᵢ)

    A model that says 99% and is wrong gets punished far more than one
    that says 55% and is wrong. Lower is better.

    Args:
        probs  : (n_samples, n_classes) softmax probabilities
        labels : (n_samples,) true class indices

    Returns:
        nll : scalar float
    """
    n = len(labels)
    true_probs = probs[np.arange(n), labels]
    return float(-np.mean(np.log(true_probs + 1e-9)))


def compute_accuracy(probs, labels):
    """Top-1 accuracy."""
    return float((probs.argmax(axis=1) == labels).mean())


def compute_all(probs, labels):
    """
    Compute all three calibration metrics at once.

    Returns:
        dict with keys: 'accuracy', 'ece', 'nll'
    """
    return {
        "accuracy": compute_accuracy(probs, labels),
        "ece":      compute_ece(probs, labels),
        "nll":      compute_nll(probs, labels),
    }


def get_reliability_data(probs, labels, n_bins=None):
    """
    Compute bin-level data for reliability diagrams.

    Returns:
        bin_confs : (n_bins,) average confidence per bin
        bin_accs  : (n_bins,) average accuracy per bin
        bin_sizes : (n_bins,) number of samples per bin
    """
    n_bins = n_bins or config.N_BINS
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confs  = np.zeros(n_bins)
    bin_accs   = np.zeros(n_bins)
    bin_sizes  = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_confs[i] = (lo + hi) / 2
            continue
        bin_confs[i] = confidences[mask].mean()
        bin_accs[i]  = correct[mask].mean()
        bin_sizes[i] = mask.sum()

    return bin_confs, bin_accs, bin_sizes
