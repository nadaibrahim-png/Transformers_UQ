# =============================================================================
# evaluation/uncertainty.py — UQ inference methods
# =============================================================================
# Three methods are implemented here:
#   1. Standard  — single forward pass, dropout OFF   [baseline]
#   2. MC Dropout — 30 stochastic forward passes       [GAL16]
#   3. Temperature Scaling — post-hoc logit rescaling  [GUO17]

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Method 1: Standard Inference ─────────────────────────────────────────────

def predict_standard(model, loader):
    """
    Single deterministic forward pass — dropout OFF.
    This is the baseline: shows how overconfident the model is
    before any uncertainty correction.

    Paper connection [GUO17]: modern NNs are overconfident even on test data
    they classify correctly.

    Returns:
        probs  : (n_samples, n_classes) softmax probabilities
        labels : (n_samples,) true class labels
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(DEVICE))
            probs  = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.numpy())

    return np.vstack(all_probs), np.concatenate(all_labels)


# ── Method 2: MC Dropout ─────────────────────────────────────────────────────

def predict_mc_dropout(model, loader, n_samples=None):
    """
    Monte Carlo Dropout: run N stochastic forward passes, average softmax.

    Paper connection [GAL16 Algorithm 1]:
        At test time:
            for t = 1 .. T:
                ŷ_t = softmax(f_dropout(x))
            predictive mean = (1/T) Σ ŷ_t
            predictive variance = (1/T) Σ ŷ_t² - mean²

    The mean is our calibrated prediction.
    The variance is our uncertainty estimate (epistemic uncertainty).

    The key trick: model.enable_mc_dropout() sets only the MC Dropout
    layer to train mode (stochastic), while keeping all other layers
    in eval mode (deterministic BN etc.).

    Args:
        n_samples : number of stochastic forward passes (default: config.MC_SAMPLES)

    Returns:
        probs      : (n_samples, n_classes) mean softmax across T passes
        labels     : (n_samples,) true class labels
        uncertainty: (n_samples,) predictive entropy H = -Σ p log p
    """
    n_samples = n_samples or config.MC_SAMPLES
    model.enable_mc_dropout()   # eval + mc_dropout.train()

    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)

            # Stack T softmax outputs → (T, batch, n_classes)
            samples = torch.stack([
                torch.softmax(model(X_batch), dim=-1)
                for _ in range(n_samples)
            ], dim=0)

            # Predictive mean → (batch, n_classes)
            mean_probs = samples.mean(dim=0)

            all_probs.append(mean_probs.cpu().numpy())
            all_labels.append(y_batch.numpy())

    probs  = np.vstack(all_probs)
    labels = np.concatenate(all_labels)

    # Predictive entropy: H = -Σ p_c * log(p_c)
    # High H → model is uncertain (good for OOD inputs)
    uncertainty = -np.sum(probs * np.log(probs + 1e-9), axis=1)

    return probs, labels, uncertainty


# ── Method 3: Temperature Scaling ────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Wraps a trained model and applies temperature scaling to its logits.

    Paper connection [GUO17 §3.3]:
        σ_T(f(x)/T) where T > 0 is a single scalar parameter.
        T > 1 → softer predictions (less overconfident).
        T is fit by minimising NLL on the VALIDATION SET (not test set).
        Accuracy does not change — only calibration improves.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))  # start at T=1 (no change)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def fit(self, val_loader):
        """
        Optimise T on the validation set using L-BFGS + NLL loss.
        L-BFGS converges in very few steps for this single-parameter problem.
        """
        self.model.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_logits.append(self.model(X_batch.to(DEVICE)).cpu())
                val_labels.append(y_batch)
        val_logits = torch.cat(val_logits)
        val_labels = torch.cat(val_labels)

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=500)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(val_logits / self.temperature, val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        T = self.temperature.item()
        print(f"  Learned temperature T = {T:.4f}  ({'overconfident → softened' if T > 1 else 'underconfident → sharpened'})")
        return T


def predict_temperature_scaled(ts_model, loader):
    """
    Run inference with temperature scaling applied.

    Args:
        ts_model : fitted TemperatureScaler instance

    Returns:
        probs  : (n_samples, n_classes) calibrated softmax probabilities
        labels : (n_samples,) true class labels
    """
    ts_model.model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            scaled_logits = ts_model(X_batch.to(DEVICE))
            probs = torch.softmax(scaled_logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.numpy())

    return np.vstack(all_probs), np.concatenate(all_labels)
