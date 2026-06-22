# =============================================================================
# evaluation/metrics.py — Calibration metrics: ECE, NLL, Accuracy
# =============================================================================
# Paper connection [GUO17]:
#   ECE  — Eq. 3: bins confidences and compares to accuracy per bin
#   NLL  — standard probabilistic loss; penalises confident wrong predictions
#   Reliability diagram — visual companion to ECE (Figure 1 in Guo 2017)

import numpy as np


def compute_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error (ECE).

    Paper connection [GUO17 Eq. 3]:
        ECE = Σ_m (|B_m| / n) * |acc(B_m) - conf(B_m)|
    where B_m = samples whose confidence falls in bin m.

    Perfect calibration → ECE = 0.
    Overconfident model → ECE > 0 (confidence > accuracy).

    Args:
        probs  : (n_samples, n_classes) softmax probabilities
        labels : (n_samples,) true integer class labels
        n_bins : number of equal-width confidence bins

    Returns:
        ece : scalar float
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = (predictions[mask] == labels[mask]).mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_nll(probs, labels):
    """
    Negative Log-Likelihood (NLL).

    Paper connection [GUO17 §3.1]:
        NLL = -(1/n) Σ log p(y_i | x_i)
    A well-calibrated model that is wrong should assign LOW confidence
    to the wrong class, keeping NLL low.

    Args:
        probs  : (n_samples, n_classes) softmax probabilities
        labels : (n_samples,) true integer class labels

    Returns:
        nll : scalar float
    """
    eps = 1e-9
    correct_probs = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(correct_probs + eps)))


def compute_accuracy(probs, labels):
    """Top-1 classification accuracy."""
    return float((probs.argmax(axis=1) == labels).mean())


def compute_all(probs, labels, n_bins=10):
    """
    Compute accuracy, ECE, and NLL in one call.

    Returns:
        dict with keys: accuracy, ece, nll
    """
    return {
        "accuracy": compute_accuracy(probs, labels),
        "ece":      compute_ece(probs, labels, n_bins),
        "nll":      compute_nll(probs, labels),
    }


def get_reliability_data(probs, labels, n_bins=10):
    """
    Compute per-bin accuracy and confidence for reliability diagrams.

    Paper connection [GUO17 Figure 1]:
        Returns the data needed to draw the reliability diagram.
        Perfect calibration → bin_accs ≈ bin_confs (points on diagonal).

    Returns:
        bin_confs : mean confidence per bin
        bin_accs  : mean accuracy per bin
        bin_sizes : number of samples per bin
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_confs, bin_accs, bin_sizes = [], [], []
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_confs.append(confidences[mask].mean())
        bin_accs.append((predictions[mask] == labels[mask]).mean())
        bin_sizes.append(int(mask.sum()))

    return np.array(bin_confs), np.array(bin_accs), np.array(bin_sizes)
