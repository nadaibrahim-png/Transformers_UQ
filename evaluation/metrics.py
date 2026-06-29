# =============================================================================
# evaluation/metrics.py — Calibration metrics
# =============================================================================
# Paper connections:
#   ECE + Reliability diagrams : Guo et al. 2017 (arXiv:1706.04599)
#   NLL                        : standard probabilistic metric
#   Brier Score                : Brier 1950; Murphy 1973

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


def compute_brier_score(probs, labels):
    """
    Brier Score — mean squared error between predicted probabilities and one-hot labels.

    BS = (1/n) Σ Σ_c (p(c|xᵢ) − 1[yᵢ=c])²

    Range [0, 2] for multi-class (normalised to [0, 1] for binary).
    Lower is better. A perfect model scores 0; random guessing scores ~0.5.

    Unlike ECE (binning-based), Brier Score is a proper scoring rule that
    penalises both overconfidence AND underconfidence continuously.

    Paper connection: Brier (1950); Murphy (1973).

    Args:
        probs  : (n_samples, n_classes) softmax probabilities
        labels : (n_samples,) true class indices

    Returns:
        brier : scalar float
    """
    n, n_classes = probs.shape
    # Build one-hot targets
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def compute_accuracy(probs, labels):
    """Top-1 accuracy. Biased toward majority class — use balanced_accuracy instead."""
    return float((probs.argmax(axis=1) == labels).mean())


def compute_balanced_accuracy(probs, labels):
    """
    Balanced accuracy — average per-class recall, unaffected by class imbalance.

    Balanced Acc = (1/C) Σ_c  TP_c / (TP_c + FN_c)
                 = mean of per-class recall scores

    For MiniBooNE (72% νμ / 28% νₑ):
        A model always predicting νμ → accuracy 0.72, balanced accuracy 0.50.
        Random chance → balanced accuracy 0.50 regardless of class ratio.
        This is the headline metric for imbalanced datasets.

    Args:
        probs  : (n_samples, n_classes) softmax probabilities
        labels : (n_samples,) true class indices

    Returns:
        balanced_acc : scalar float in [0, 1]
    """
    predictions = probs.argmax(axis=1)
    classes     = np.unique(labels)
    per_class_recall = []
    for c in classes:
        mask = labels == c
        if mask.sum() == 0:
            continue
        recall = (predictions[mask] == c).mean()
        per_class_recall.append(float(recall))
    return float(np.mean(per_class_recall))


def compute_per_class_recall(probs, labels, class_names=None):
    """
    Per-class recall (sensitivity) for each class.

    For MiniBooNE:
        Class 0 (νμ background) recall = background rejection rate
        Class 1 (νₑ signal)     recall = signal efficiency

    Signal efficiency is the primary physics metric — it answers:
    'Of all the real νₑ events, how many does the model correctly flag?'

    Args:
        probs       : (n_samples, n_classes) softmax probabilities
        labels      : (n_samples,) true class indices
        class_names : optional list of names, e.g. ["νμ Bkg", "νₑ Sig"]

    Returns:
        recall_dict : dict mapping class name (or index) → recall float
    """
    predictions = probs.argmax(axis=1)
    classes     = np.unique(labels)
    recall_dict = {}
    for c in classes:
        mask = labels == c
        recall = float((predictions[mask] == c).mean()) if mask.sum() > 0 else 0.0
        key = class_names[int(c)] if class_names else int(c)
        recall_dict[key] = recall
    return recall_dict


def compute_all(probs, labels):
    """
    Compute all calibration and classification metrics.

    Primary metric is balanced_accuracy (not accuracy) to account for
    MiniBooNE's 72/28 class imbalance.

    Returns:
        dict with keys:
            'balanced_accuracy' — headline metric (class-fair)
            'accuracy'          — plain accuracy (for comparison with literature)
            'signal_recall'     — νₑ recall / signal efficiency
            'background_recall' — νμ recall / background rejection
            'ece'               — Expected Calibration Error
            'nll'               — Negative Log-Likelihood
            'brier'             — Brier Score
    """
    recall = compute_per_class_recall(probs, labels)
    return {
        "balanced_accuracy": compute_balanced_accuracy(probs, labels),
        "accuracy":          compute_accuracy(probs, labels),
        "signal_recall":     recall.get(1, recall.get(0, 0.0)),   # class 1 = νₑ
        "background_recall": recall.get(0, 0.0),                   # class 0 = νμ
        "ece":               compute_ece(probs, labels),
        "nll":               compute_nll(probs, labels),
        "brier":             compute_brier_score(probs, labels),
    }


def compute_ece_per_class(probs, labels, n_bins=None):
    """
    Per-class ECE — compute ECE separately for each true class.

    Answers: 'Is the model better calibrated for signal (νₑ) than background (νμ)?'
    Critical for rare-event physics where signal purity matters more than
    raw accuracy.

    Args:
        probs  : (n_samples, n_classes) softmax probabilities
        labels : (n_samples,) true class indices

    Returns:
        per_class_ece : dict mapping class_index → ECE float
    """
    n_bins = n_bins or config.N_BINS
    classes = np.unique(labels)
    per_class_ece = {}
    for c in classes:
        mask = labels == c
        per_class_ece[int(c)] = compute_ece(probs[mask], labels[mask], n_bins)
    return per_class_ece


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
