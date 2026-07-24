# =============================================================================
# evaluation/uncertainty.py — UQ inference methods
# =============================================================================
# Five methods:
#   1. Standard          — single forward pass, dropout OFF   [baseline]
#   2. MC Dropout        — 30 stochastic forward passes       [GAL16]
#   3. Temperature Scaling — post-hoc logit rescaling         [GUO17]
#   4. Isotonic Regression — non-parametric post-hoc scaling  [ZADROZNY02]
#   5. Deep Ensemble     — M=5 independently trained models   [LAKSHMINARAYANAN17]

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


# ── Method 5: Deep Ensemble ───────────────────────────────────────────────────

class DeepEnsemble:
    """
    Deep Ensembles: train M independent models from different random seeds,
    average their softmax outputs at inference.

    Paper connection [LAKSHMINARAYANAN17 — "Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles" NeurIPS 2017]:
        - Each model trained independently with a different random seed.
        - Diversity comes from random weight initialisation + data shuffling.
        - Predictive mean  p̄(y|x) = (1/M) Σₘ pₘ(y|x)
        - Epistemic uncertainty via disagreement between members:
            MI(y; θ | x) ≈ H[p̄] − (1/M) Σₘ H[pₘ]
          (Total entropy minus average member entropy = mutual information)

    Why better than MC Dropout:
        - Members explore genuinely different loss landscape modes.
        - MC Dropout samples around a single MAP estimate; ensemble samples
          across M different MAP estimates — broader coverage of the posterior.
        - Empirically: better calibration + OOD detection [OVADIA19].

    Usage:
        ensemble = DeepEnsemble(model_class, model_kwargs, M=5)
        ensemble.train_all(train_loader, val_loader, epochs=50)
        probs, labels, uncertainty = ensemble.predict(test_loader)
    """

    def __init__(self, model_class, model_kwargs: dict, M: int = 5):
        """
        Args:
            model_class  : the model constructor (e.g. FTTransformer)
            model_kwargs : dict of constructor arguments (d_model, n_heads, etc.)
            M            : number of ensemble members (default 5)
        """
        self.model_class  = model_class
        self.model_kwargs = model_kwargs
        self.M            = M
        self.models       = []          # list of trained nn.Module

    def train_all(self, train_loader, val_loader, epochs=None, lr=None,
                  seeds=None, verbose=True):
        """
        Train M models independently from different random seeds.

        Args:
            train_loader : DataLoader for training set
            val_loader   : DataLoader for validation set (used for early-stop loss)
            epochs       : number of epochs per member (default: config.EPOCHS)
            lr           : learning rate (default: config.LEARNING_RATE)
            seeds        : list of M ints; defaults to [0, 1, 2, 3, 4]
            verbose      : print progress
        """
        epochs = epochs or getattr(config, 'EPOCHS', 50)
        lr     = lr     or getattr(config, 'LEARNING_RATE', 1e-3)
        seeds  = seeds  or list(range(self.M))

        self.models = []

        for m, seed in enumerate(seeds):
            if verbose:
                print(f"\n── Ensemble member {m+1}/{self.M}  (seed={seed}) ──")

            # Reproducibility for this member
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = self.model_class(**self.model_kwargs).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float('inf')
            best_state    = None

            for epoch in range(epochs):
                # ── Train ──
                model.train()
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(X_batch.to(DEVICE)), y_batch.to(DEVICE))
                    loss.backward()
                    optimizer.step()

                # ── Validate ──
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        val_loss += criterion(
                            model(X_batch.to(DEVICE)), y_batch.to(DEVICE)
                        ).item()

                val_loss /= len(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = {k: v.clone() for k, v in model.state_dict().items()}

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"   epoch {epoch+1:3d}/{epochs}  val_loss={val_loss:.4f}")

            # Restore best checkpoint for this member
            model.load_state_dict(best_state)
            model.eval()
            self.models.append(model)
            if verbose:
                print(f"   ✓ member {m+1} done — best val_loss={best_val_loss:.4f}")

        if verbose:
            print(f"\n✓ Deep Ensemble ready — {self.M} members trained.")

    def predict(self, loader):
        """
        Average softmax predictions across all M ensemble members.

        Returns:
            probs       : (n_samples, n_classes) mean softmax probability
            labels      : (n_samples,) true class labels
            uncertainty : (n_samples,) predictive entropy H[p̄]  — total uncertainty
            epistemic   : (n_samples,) mutual information = H[p̄] − mean_m H[pₘ]
        """
        if not self.models:
            raise RuntimeError("No trained models. Call train_all() first.")

        all_member_probs = []   # list of M arrays (n, n_classes)
        all_labels       = None

        for model in self.models:
            model.eval()
            member_probs, labels = predict_standard(model, loader)
            all_member_probs.append(member_probs)
            all_labels = labels

        # Stack → (M, n_samples, n_classes)
        stacked = np.stack(all_member_probs, axis=0)

        # Predictive mean → (n_samples, n_classes)
        mean_probs = stacked.mean(axis=0)

        # Total uncertainty: entropy of the mean
        total_entropy = -np.sum(
            mean_probs * np.log(mean_probs + 1e-9), axis=1
        )

        # Average member entropy → aleatoric component
        member_entropies = -np.sum(
            stacked * np.log(stacked + 1e-9), axis=2
        ).mean(axis=0)                             # (n_samples,)

        # Epistemic uncertainty: mutual information = total − aleatoric
        epistemic = np.maximum(total_entropy - member_entropies, 0.0)

        return mean_probs, all_labels, total_entropy, epistemic

    def save(self, directory: str):
        """Save all member weights to directory/member_{m}.pt"""
        os.makedirs(directory, exist_ok=True)
        for m, model in enumerate(self.models):
            path = os.path.join(directory, f"member_{m}.pt")
            torch.save(model.state_dict(), path)
        print(f"Saved {self.M} ensemble members to '{directory}/'")

    def load(self, directory: str):
        """Load member weights previously saved with save()."""
        self.models = []
        for m in range(self.M):
            path = os.path.join(directory, f"member_{m}.pt")
            model = self.model_class(**self.model_kwargs).to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            self.models.append(model)
        print(f"Loaded {self.M} ensemble members from '{directory}/'")


# ── Method 6: SWAG (Stochastic Weight Averaging-Gaussian) ────────────────────

class SWAG:
    """
    SWAG-Diagonal: fit a Gaussian over the SGD weight trajectory and sample
    at inference — cheap single-training-run Bayesian approximation.

    Paper connection [MADDOX19 — "A Simple Baseline for Bayesian Deep Learning"
    NeurIPS 2019]:
        After standard training, continue with a high/cyclical LR for C epochs.
        Every epoch, record the current weight vector θₜ.
        From T snapshots compute:
            θ_SWA  = (1/T) Σ θₜ           ← SWA mean (better point estimate)
            σ²_diag = (1/T) Σ θₜ² − θ_SWA²  ← diagonal posterior variance

        At inference, sample K weight vectors:
            θ* = θ_SWA + σ_diag ⊙ ε,  ε ~ N(0, I)

        Average K softmax outputs → predictive mean + entropy.

    Why SWAG vs Ensembles:
        - SWAG needs only ONE training run (snapshots from the end of training).
        - Ensembles need M full training runs.
        - SWAG posterior covers a connected region around ONE basin;
          Ensembles cover M different basins → more diverse but more expensive.

    Why SWAG vs MC Dropout:
        - SWAG samples from a Gaussian over ALL weights (full posterior approx).
        - MC Dropout samples binary masks — a much coarser approximation.
        - SWAG often matches or beats MC Dropout on calibration [OVADIA19].

    Usage:
        swag = SWAG(model)
        swag.collect_during_training(train_loader, val_loader, epochs=20, lr=5e-4)
        probs, labels, uncertainty = swag.predict(test_loader, K=30)
    """

    def __init__(self, model, max_snapshots: int = 20):
        """
        Args:
            model         : a trained nn.Module (FTTransformer)
            max_snapshots : rolling window size T (default 20)
        """
        self.model         = model
        self.max_snapshots = max_snapshots

        self._snapshots    = []          # list of flat CPU tensors
        self._mean         = None        # θ_SWA  — flat tensor
        self._std          = None        # σ_diag — flat tensor (std, not var)
        self._param_names  = None
        self._param_shapes = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _flatten(self) -> torch.Tensor:
        """Flatten all trainable parameters into a single CPU vector."""
        return torch.cat([p.detach().cpu().view(-1) for p in self.model.parameters()])

    def _unflatten_and_set(self, flat: torch.Tensor):
        """Write a flat weight vector back into model parameters."""
        offset = 0
        for shape, param in zip(self._param_shapes, self.model.parameters()):
            numel = shape.numel()
            param.data.copy_(
                flat[offset: offset + numel].view(shape).to(DEVICE)
            )
            offset += numel

    # ── Snapshot collection ───────────────────────────────────────────────────

    def collect(self):
        """Record the current model weights as one snapshot."""
        self._snapshots.append(self._flatten())
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)      # rolling window

    def fit(self):
        """
        Compute SWA mean and diagonal standard deviation from collected snapshots.
        Must be called after collect() has been called at least twice.
        """
        if len(self._snapshots) < 2:
            raise RuntimeError(
                f"Need ≥2 snapshots, have {len(self._snapshots)}. "
                "Run collect_during_training() first."
            )

        stacked = torch.stack(self._snapshots, dim=0)   # (T, num_params)

        self._mean = stacked.mean(dim=0)
        var        = stacked.var(dim=0)
        self._std  = torch.sqrt(var + 1e-6)             # diagonal σ, avoid √0

        # Cache shapes for unflatten
        self._param_names  = [n for n, _ in self.model.named_parameters()]
        self._param_shapes = [p.shape for p in self.model.parameters()]

        n_params = self._mean.numel()
        print(f"  SWAG fitted: T={len(self._snapshots)} snapshots, "
              f"{n_params:,} params, "
              f"mean σ = {self._std.mean().item():.4f}")

    # ── SWAG training phase ───────────────────────────────────────────────────

    def collect_during_training(self, train_loader, val_loader,
                                epochs: int = 20, lr: float = 5e-4,
                                collect_every: int = 1, verbose: bool = True):
        """
        Continue training with a constant high LR (SGD) and collect snapshots.

        This is the SWAG-specific phase that runs AFTER the main Adam training.
        A high LR causes the optimizer to explore the loss basin rather than
        converge — the resulting trajectory gives diverse weight snapshots.

        Args:
            train_loader  : DataLoader for training set
            val_loader    : DataLoader for validation (monitoring only)
            epochs        : SWAG collection epochs (20–30 is typical)
            lr            : learning rate — higher than initial (e.g. 5e-4)
            collect_every : collect a snapshot every N epochs
            verbose       : print per-epoch progress
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        if verbose:
            print(f"SWAG collection phase: {epochs} epochs, LR={lr}")

        for epoch in range(epochs):
            # ── Train step ──
            self.model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(
                    self.model(X_batch.to(DEVICE)), y_batch.to(DEVICE)
                )
                loss.backward()
                optimizer.step()

            # ── Collect snapshot ──
            if (epoch + 1) % collect_every == 0:
                self.collect()

            if verbose and (epoch + 1) % 5 == 0:
                # Quick val loss for monitoring
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        val_loss += criterion(
                            self.model(X_batch.to(DEVICE)), y_batch.to(DEVICE)
                        ).item()
                val_loss /= len(val_loader)
                print(f"  epoch {epoch+1:3d}/{epochs}  "
                      f"val_loss={val_loss:.4f}  "
                      f"snapshots={len(self._snapshots)}")

        self.fit()
        print(f"✓ SWAG ready.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, loader, K: int = 30):
        """
        Sample K weight vectors from the SWAG-Diagonal posterior and average.

        Sampling:
            θ* = θ_SWA + σ_diag ⊙ ε,   ε ~ N(0, I)

        Returns:
            probs       : (n_samples, n_classes) mean softmax
            labels      : (n_samples,) true class labels
            uncertainty : (n_samples,) predictive entropy H[p̄]
        """
        if self._mean is None:
            raise RuntimeError("Call collect_during_training() (or fit()) first.")

        # Save current weights to restore after sampling
        original_flat = self._flatten()

        all_sample_probs = []
        all_labels       = None

        for _ in range(K):
            # Sample θ* from the diagonal Gaussian
            eps       = torch.randn_like(self._mean)
            theta_star = self._mean + self._std * eps
            self._unflatten_and_set(theta_star)

            self.model.eval()
            member_probs, labels = predict_standard(self.model, loader)
            all_sample_probs.append(member_probs)
            all_labels = labels

        # Restore original weights (SWA mean or last training state)
        self._unflatten_and_set(self._mean)

        stacked    = np.stack(all_sample_probs, axis=0)   # (K, n, n_classes)
        mean_probs = stacked.mean(axis=0)
        uncertainty = -np.sum(
            mean_probs * np.log(mean_probs + 1e-9), axis=1
        )

        return mean_probs, all_labels, uncertainty

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save SWAG statistics (mean, std) and model architecture info."""
        torch.save({
            'mean':         self._mean,
            'std':          self._std,
            'param_shapes': self._param_shapes,
            'param_names':  self._param_names,
            'n_snapshots':  len(self._snapshots),
        }, path)
        print(f"SWAG statistics saved to '{path}'")

    def load(self, path: str):
        """Load SWAG statistics previously saved with save()."""
        ckpt = torch.load(path, map_location='cpu')
        self._mean         = ckpt['mean']
        self._std          = ckpt['std']
        self._param_shapes = ckpt['param_shapes']
        self._param_names  = ckpt['param_names']
        print(f"SWAG statistics loaded from '{path}' "
              f"(T={ckpt['n_snapshots']} snapshots)")
