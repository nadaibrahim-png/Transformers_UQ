# =============================================================================
# evaluation/uncertainty.py — UQ inference methods
# =============================================================================
# Four methods:
#   1. Standard          — single forward pass, dropout OFF   [baseline]
#   2. MC Dropout        — 30 stochastic forward passes       [GAL16]
#   3. Temperature Scaling — post-hoc logit rescaling         [GUO17]
#   4. Isotonic Regression — non-parametric post-hoc scaling  [ZADROZNY02]

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
    Baseline: shows how overconfident the model is before any UQ correction.

    Paper connection [GUO17]: 'Modern NNs are overconfident even when wrong.'

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
    Monte Carlo Dropout: run T stochastic forward passes, average softmax.

    Paper connection [GAL16 Algorithm 1]:
        Keep dropout ON at test time.
        Predictive mean  = (1/T) Σ softmax(f_dropout(x))
        Predictive entropy H = -Σ p log p  (uncertainty estimate)

    model.enable_mc_dropout() keeps ONLY the mc_dropout layer stochastic;
    everything else stays in eval mode.

    Returns:
        probs       : (n_samples, n_classes) mean softmax across T passes
        labels      : (n_samples,) true class labels
        uncertainty : (n_samples,) predictive entropy H = -Σ p log p
    """
    n_samples = n_samples or config.MC_SAMPLES
    model.enable_mc_dropout()

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

    # Predictive entropy: high H → uncertain (good signal for OOD inputs)
    uncertainty = -np.sum(probs * np.log(probs + 1e-9), axis=1)

    return probs, labels, uncertainty


# ── Method 3: Temperature Scaling ────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Wraps a trained model and applies temperature scaling to its logits.

    Paper connection [GUO17 §3.3]:
        scaled_logits = logits / T
        T is fit by minimising NLL on the VALIDATION SET (not test set).
        T > 1 → softer predictions (less overconfident).
        Accuracy does not change — only calibration improves.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Initialise temperature on the same device as the model
        model_device = next(model.parameters()).device
        self.temperature = nn.Parameter(torch.ones(1, device=model_device) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def fit(self, val_loader):
        """
        Optimise T on the validation set using L-BFGS + NLL loss.
        L-BFGS converges in very few steps for this 1-parameter problem.
        """
        self.model.eval()
        val_logits, val_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_logits.append(self.model(X_batch.to(DEVICE)).cpu())
                val_labels.append(y_batch)

        val_logits = torch.cat(val_logits)
        val_labels = torch.cat(val_labels)

        # Move temperature to CPU for optimisation (logits are on CPU)
        temperature_cpu = nn.Parameter(self.temperature.data.cpu())
        optimizer  = optim.LBFGS([temperature_cpu], lr=0.01, max_iter=500)
        criterion  = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(val_logits / temperature_cpu, val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Sync back to original device
        with torch.no_grad():
            self.temperature.copy_(temperature_cpu.to(self.temperature.device))

        T = self.temperature.item()
        direction = "overconfident → softened" if T > 1 else "underconfident → sharpened"
        print(f"  Learned temperature T = {T:.4f}  ({direction})")
        return T


def predict_temperature_scaled(ts_model, loader):
    """
    Run inference with temperature scaling applied.

    Args:
        ts_model : fitted TemperatureScaler instance (already .to(DEVICE))

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


# ── Method 4: Isotonic Regression ────────────────────────────────────────────

class IsotonicScaler:
    """
    Non-parametric post-hoc calibration via Isotonic Regression.

    Paper connection [ZADROZNY02]:
        Fits a monotonic step function mapping raw confidence → calibrated
        probability on the VALIDATION set. No distributional assumption
        (unlike Temperature Scaling which assumes logit shift).

    Tradeoff vs Temperature Scaling:
        + More flexible — can correct any monotonic miscalibration pattern.
        − Needs more val data to avoid overfitting the calibration map.
        − One IR fitted per class (one-vs-rest), then outputs renormalised.

    For MiniBooNE (binary): fits 2 isotonic regressors on [p(νₑ), p(νμ)],
    each independently mapped, then softmax-normalised.

    Paper: Zadrozny & Elkan 2002 — "Transforming Classifier Scores into
           Accurate Multiclass Probability Estimates" (KDD 2002).
    """

    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self._IsotonicRegression = IsotonicRegression
        self.regressors = None   # list of fitted IR, one per class
        self.n_classes  = None

    def fit(self, val_probs, val_labels):
        """
        Fit one isotonic regressor per class on validation probabilities.

        Args:
            val_probs  : (n_val, n_classes) softmax probabilities from model
            val_labels : (n_val,) true class indices
        """
        self.n_classes  = val_probs.shape[1]
        self.regressors = []

        for c in range(self.n_classes):
            y_binary = (val_labels == c).astype(float)   # 1 if true class = c
            ir = self._IsotonicRegression(out_of_bounds="clip")
            ir.fit(val_probs[:, c], y_binary)
            self.regressors.append(ir)

        print(f"  Isotonic Regression fitted on {len(val_labels)} val samples "
              f"({self.n_classes} regressors)")

    def predict(self, probs):
        """
        Apply fitted isotonic regressors and renormalise to sum to 1.

        Args:
            probs : (n_samples, n_classes) raw softmax probabilities

        Returns:
            calibrated_probs : (n_samples, n_classes)
        """
        if self.regressors is None:
            raise RuntimeError("Call fit() before predict().")

        calibrated = np.column_stack([
            self.regressors[c].predict(probs[:, c])
            for c in range(self.n_classes)
        ])

        # Renormalise — IR outputs are not guaranteed to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)   # avoid div-by-zero
        return calibrated / row_sums


def predict_isotonic(model, val_loader, test_loader):
    """
    Fit Isotonic Regression on validation set and evaluate on test set.

    Args:
        model       : trained FTTransformer (eval mode)
        val_loader  : DataLoader for validation set (used to fit IR)
        test_loader : DataLoader for test set (evaluated after calibration)

    Returns:
        probs_cal : (n_test, n_classes) isotonic-calibrated probabilities
        labels    : (n_test,) true class labels
        scaler    : fitted IsotonicScaler (for reuse on OOD sets)
    """
    # 1. Get raw validation probabilities to fit the regressor
    val_probs, val_labels = predict_standard(model, val_loader)

    # 2. Fit isotonic regressor
    scaler = IsotonicScaler()
    scaler.fit(val_probs, val_labels)

    # 3. Get raw test probabilities and apply calibration
    test_probs, test_labels = predict_standard(model, test_loader)
    probs_cal = scaler.predict(test_probs)

    return probs_cal, test_labels, scaler
