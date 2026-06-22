# =============================================================================
# TOWARD ROBUST DEEP LEARNING: Evaluating Uncertainty in Transformers
# ICTP Poster Experiment — Google Colab Ready
# Author: Nada Elhassan, University of Messina
# =============================================================================
#
# HOW TO USE:
#   Copy this entire file into a Colab notebook (one big cell, or split by
#   the ── SECTION ── markers). Run top to bottom.
#
# PAPER CONNECTIONS (read these alongside the code):
#   [GAL16]  Gal & Ghahramani 2016 — arXiv:1506.02142  (MC Dropout)
#   [GUO17]  Guo et al. 2017       — arXiv:1706.04599  (ECE, calibration)
#   [LAK17]  Lakshminarayanan 2017  — arXiv:1612.01474  (Deep Ensembles)
# =============================================================================


# ── SECTION 0: INSTALL & IMPORTS ────────────────────────────────────────────

# Run this cell first in Colab
# !pip install netcal --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# netcal gives us ECE and reliability diagrams out of the box
# Paper connection: implements the ECE formula from [GUO17] Section 3.3
try:
    from netcal.metrics import ECE
    from netcal.presentation import ReliabilityDiagram
    NETCAL = True
except ImportError:
    NETCAL = False
    print("netcal not installed — run:  !pip install netcal")

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── SECTION 1: DATASET ───────────────────────────────────────────────────────
#
# We use the Wine dataset (sklearn built-in).
# 178 samples, 13 chemical features, 3 wine classes.
# Small enough to train fast on Colab CPU, real enough to be scientific.
#
# For your poster: describe this as a tabular scientific dataset where
# features represent physical/chemical measurements — exactly the kind
# of data where model overconfidence is dangerous.

print("\n── Loading Wine dataset ──")
data = load_wine()
X, y = data.data.astype(np.float32), data.target

# Split: 60% train / 20% val (for temperature scaling) / 20% test
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.25, random_state=42, stratify=y_tv)

# Standardise — critical for Transformer training stability
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Simulate Out-of-Distribution (OOD) data by adding strong Gaussian noise
# Paper connection: [GUO17] and [GAL16] evaluate on OOD inputs to show
# that standard models remain overconfident even when they should be uncertain.
X_ood = X_test + np.random.normal(0, 3.0, X_test.shape).astype(np.float32)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, OOD: {X_ood.shape}")
print(f"Classes: {data.target_names}")
N_FEATURES = X_train.shape[1]   # 13
N_CLASSES  = len(np.unique(y))  # 3

# Convert to tensors
def to_loader(X, y, batch_size=32, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = to_loader(X_train, y_train, shuffle=True)
val_loader   = to_loader(X_val,   y_val)
test_loader  = to_loader(X_test,  y_test)
ood_loader   = to_loader(X_ood,   y_test)  # same labels, corrupted features


# ── SECTION 2: MODEL — TRANSFORMER FOR TABULAR DATA ─────────────────────────
#
# Architecture:
#   Each of the 13 features is treated as a "token" (like words in NLP).
#   A linear layer embeds each scalar into a D-dimensional vector.
#   A Transformer encoder layer applies self-attention across the 13 tokens.
#   A classifier head produces 3 class logits.
#
# Dropout is added in TWO places:
#   1. Inside the Transformer (standard regularisation)
#   2. Before the classifier head (this is what MC Dropout uses at test time)
#
# Paper connection [GAL16]: "We show that a neural network with arbitrary
# depth and non-linearities, with dropout applied before every weight layer,
# is equivalent to an approximation to the probabilistic deep Gaussian process."
#
# Key insight: during TRAINING dropout is always on (standard).
#              During TESTING we normally turn it off — but MC Dropout
#              keeps it ON to sample different "versions" of the model.

class TabTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_model=64, nhead=4,
                 num_layers=2, dropout=0.2):
        super().__init__()

        # Embed each feature scalar → d_model-dimensional vector
        self.feature_embed = nn.Linear(1, d_model)

        # Positional embedding (one per feature / "token")
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder (self-attention across the 13 feature tokens)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ★ MC Dropout layer — this is the key one we keep ON at test time
        self.mc_dropout = nn.Dropout(p=dropout)

        # Classifier: mean-pool tokens → linear → class logits
        self.classifier = nn.Linear(d_model, n_classes)

        self.d_model = d_model

    def forward(self, x):
        # x: (batch, n_features)
        # Reshape to (batch, n_features, 1) then embed to (batch, n_features, d_model)
        x = self.feature_embed(x.unsqueeze(-1))
        x = x + self.pos_embed

        # Self-attention across feature tokens
        x = self.transformer(x)

        # Mean pool across feature dimension → (batch, d_model)
        x = x.mean(dim=1)

        # ★ MC Dropout applied here — on during both train AND test (for MC)
        x = self.mc_dropout(x)

        # Logits
        return self.classifier(x)

    def enable_mc_dropout(self):
        """
        Sets only the mc_dropout layer to training mode (keeps dropout active).
        The rest stays in eval mode (BatchNorm etc. behave correctly).
        Paper connection [GAL16] Algorithm 1: keep dropout on at test time.
        """
        self.eval()
        self.mc_dropout.train()


# ── SECTION 3: TRAINING ──────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs=60, lr=1e-3):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validation
        model.eval()
        val_losses, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                val_losses.append(criterion(logits, y_batch).item())
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        va = accuracy_score(all_labels, all_preds)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_acc'].append(va)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | train_loss={tl:.4f} | val_loss={vl:.4f} | val_acc={va:.3f}")

    model.load_state_dict(best_state)
    print(f"\nBest val loss: {best_val_loss:.4f}")
    return history

print("\n── Training Transformer ──")
model = TabTransformer(N_FEATURES, N_CLASSES, d_model=64, nhead=4, num_layers=2, dropout=0.2)
history = train_model(model, train_loader, val_loader, epochs=80)


# ── SECTION 4: STANDARD INFERENCE (BASELINE) ─────────────────────────────────
#
# Standard evaluation: model.eval() → dropout OFF → single forward pass.
# We collect softmax probabilities (confidences) and true labels.
#
# Paper connection [GUO17] Section 3: "Modern neural networks are poorly
# calibrated — they tend to be overconfident."
# We will SHOW this with a reliability diagram and ECE score.

def get_predictions(model, loader, mc_dropout=False, n_samples=30):
    """
    Collect softmax probabilities and true labels from a dataloader.

    Args:
        mc_dropout : if True, keep dropout ON (MC sampling) [GAL16]
        n_samples  : how many stochastic forward passes for MC Dropout
    """
    if mc_dropout:
        model.enable_mc_dropout()  # eval + mc_dropout.train()
    else:
        model.eval()

    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)

            if mc_dropout:
                # ★ MC Dropout: run n_samples forward passes, average softmax
                # Paper connection [GAL16] Eq. 12: Monte Carlo approximation
                # of the posterior predictive distribution.
                # Each pass drops different neurons → different "model sample"
                samples = torch.stack([
                    torch.softmax(model(X_batch), dim=-1)
                    for _ in range(n_samples)
                ], dim=0)           # (n_samples, batch, n_classes)
                probs = samples.mean(dim=0)  # average across samples
            else:
                probs = torch.softmax(model(X_batch), dim=-1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.numpy())

    return np.vstack(all_probs), np.concatenate(all_labels)

print("\n── Standard inference (dropout OFF) ──")
probs_std, labels_test = get_predictions(model, test_loader, mc_dropout=False)

# Accuracy
preds_std = probs_std.argmax(axis=1)
acc_std   = accuracy_score(labels_test, preds_std)
print(f"Test accuracy (standard): {acc_std:.4f}")

# NLL — Negative Log-Likelihood
# Paper connection [GUO17]: NLL penalises confident wrong predictions.
# Lower is better.
eps = 1e-9
nll_std = -np.mean(np.log(probs_std[np.arange(len(labels_test)), labels_test] + eps))
print(f"NLL (standard): {nll_std:.4f}")

# ECE — Expected Calibration Error
# Paper connection [GUO17] Eq. 3: bins confidences, compares to accuracy.
if NETCAL:
    ece_metric = ECE(bins=10)
    ece_std = ece_metric.measure(probs_std, labels_test)
    print(f"ECE (standard): {ece_std:.4f}")
else:
    # Manual ECE if netcal not installed
    def compute_ece(probs, labels, n_bins=10):
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences > bins[i]) & (confidences <= bins[i+1])
            if mask.sum() == 0:
                continue
            bin_acc  = (predictions[mask] == labels[mask]).mean()
            bin_conf = confidences[mask].mean()
            ece += mask.mean() * abs(bin_acc - bin_conf)
        return ece
    ece_std = compute_ece(probs_std, labels_test)
    print(f"ECE (standard, manual): {ece_std:.4f}")


# ── SECTION 5: MC DROPOUT ────────────────────────────────────────────────────
#
# Now we run 30 stochastic forward passes with dropout ACTIVE.
# The mean of the softmax outputs = our uncertainty-aware prediction.
# The variance across passes = epistemic uncertainty.
#
# Paper connection [GAL16] Algorithm 1:
#   "At test time: for t=1..T, run ŷ_t = f_dropout(x).
#    Predictive mean: (1/T) Σ ŷ_t"
#
# KEY POINT: we don't change the model at all — just call model.train()
# on the dropout layer. This is why MC Dropout is so easy to implement.

print("\n── MC Dropout inference (30 samples) ──")
probs_mc, _ = get_predictions(model, test_loader, mc_dropout=True, n_samples=30)

preds_mc = probs_mc.argmax(axis=1)
acc_mc   = accuracy_score(labels_test, preds_mc)
nll_mc   = -np.mean(np.log(probs_mc[np.arange(len(labels_test)), labels_test] + eps))
ece_mc   = ece_metric.measure(probs_mc, labels_test) if NETCAL else compute_ece(probs_mc, labels_test)

print(f"Accuracy (MC Dropout): {acc_mc:.4f}")
print(f"NLL      (MC Dropout): {nll_mc:.4f}")
print(f"ECE      (MC Dropout): {ece_mc:.4f}")

# Also get MC predictions on OOD data — should show higher uncertainty
probs_mc_ood, _ = get_predictions(model, ood_loader, mc_dropout=True, n_samples=30)


# ── SECTION 6: TEMPERATURE SCALING ──────────────────────────────────────────
#
# Temperature scaling: divide logits by scalar T before softmax.
# T > 1 softens overconfident predictions (reduces peak probability).
# T is optimised on the VALIDATION SET using NLL loss.
#
# Paper connection [GUO17] Section 3.3:
#   "We apply a single scalar parameter T > 0 to the logits...
#    we find T using the validation set."
#
# This is the SIMPLEST calibration method — no retraining needed.

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        # T starts at 1.0 (= no change), optimiser will move it up
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        # Divide logits by T — softens the distribution when T > 1
        return logits / self.temperature

def collect_logits(model, loader):
    """Collect raw logits (before softmax) — needed to fit temperature."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(DEVICE))
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)
    return torch.cat(all_logits), torch.cat(all_labels)

print("\n── Temperature Scaling ──")
# Fit T on validation set
val_logits, val_labels = collect_logits(model, val_loader)

temp_model = TemperatureScaling()
optimizer_t = optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=500)
criterion_t = nn.CrossEntropyLoss()

def closure():
    optimizer_t.zero_grad()
    scaled_logits = temp_model(val_logits)
    loss = criterion_t(scaled_logits, val_labels)
    loss.backward()
    return loss

optimizer_t.step(closure)
T_learned = temp_model.temperature.item()
print(f"Learned temperature T = {T_learned:.4f}  (>1 means model was overconfident)")

# Apply temperature scaling to test set
test_logits, _ = collect_logits(model, test_loader)
scaled_logits   = temp_model(test_logits).detach()
probs_ts        = torch.softmax(scaled_logits, dim=-1).numpy()

preds_ts = probs_ts.argmax(axis=1)
acc_ts   = accuracy_score(labels_test, preds_ts)
nll_ts   = -np.mean(np.log(probs_ts[np.arange(len(labels_test)), labels_test] + eps))
ece_ts   = ece_metric.measure(probs_ts, labels_test) if NETCAL else compute_ece(probs_ts, labels_test)

print(f"Accuracy (Temp. Scaling): {acc_ts:.4f}")
print(f"NLL      (Temp. Scaling): {nll_ts:.4f}")
print(f"ECE      (Temp. Scaling): {ece_ts:.4f}")


# ── SECTION 7: FIGURES FOR YOUR POSTER ───────────────────────────────────────
#
# We generate 4 publication-quality figures.
# Save each as a high-res PNG to use in your A0 poster.

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'std': '#e05c5c', 'mc': '#4e91d9', 'ts': '#2ecc71', 'ood': '#f39c12'}


# ── FIGURE 1: Reliability Diagrams (before vs after) ─────────────────────────
#
# Paper connection [GUO17] Figure 1: "A perfectly calibrated model follows
# the diagonal. Overconfident models fall below it."
# This is the most important figure for your poster.

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

def plot_reliability(ax, probs, labels, title, color, n_bins=10):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_sizes = [], [], []

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if mask.sum() == 0:
            continue
        bin_accs.append((predictions[mask] == labels[mask]).mean())
        bin_confs.append(confidences[mask].mean())
        bin_sizes.append(mask.sum())

    bin_accs  = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    gap       = bin_confs - bin_accs   # positive = overconfident

    # Gap bars (overconfidence shown in red)
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.85, color=color, label='Accuracy', zorder=3)
    ax.bar(bin_confs, gap, width=0.08, bottom=bin_accs, alpha=0.35,
           color='#e74c3c', label='Gap (overconfidence)', zorder=3)

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Perfect calibration', zorder=4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')

plot_reliability(axes[0], probs_std, labels_test, 'Standard (Baseline)', COLORS['std'])
plot_reliability(axes[1], probs_mc,  labels_test, 'MC Dropout (30 passes)', COLORS['mc'])
plot_reliability(axes[2], probs_ts,  labels_test, 'Temperature Scaling', COLORS['ts'])

plt.suptitle('Figure 1 — Reliability Diagrams\n(bars should align with diagonal for perfect calibration)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('fig1_reliability_diagrams.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved: fig1_reliability_diagrams.png")


# ── FIGURE 2: ECE Comparison Bar Chart ───────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
methods = ['Standard\n(Baseline)', 'MC Dropout\n(30 samples)', 'Temperature\nScaling']
ece_vals = [ece_std, ece_mc, ece_ts]
colors   = [COLORS['std'], COLORS['mc'], COLORS['ts']]
bars = ax.bar(methods, ece_vals, color=colors, width=0.5, edgecolor='white', linewidth=1.5)

# Value labels on bars
for bar, val in zip(bars, ece_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('ECE (lower is better)', fontsize=11)
ax.set_title('Figure 2 — Expected Calibration Error\nby Uncertainty Method', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(ece_vals) * 1.4)
ax.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('fig2_ece_comparison.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved: fig2_ece_comparison.png")


# ── FIGURE 3: Predictive Entropy — In-distribution vs OOD ────────────────────
#
# Predictive entropy H = -Σ p_c * log(p_c)
# Paper connection [GAL16] Section 3.3:
#   "A model with good uncertainty should assign HIGH entropy to OOD inputs
#    and LOW entropy to in-distribution inputs."
# Well-separated histograms = your model knows when it doesn't know.

def predictive_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-9), axis=1)

H_test = predictive_entropy(probs_mc)        # in-distribution
H_ood  = predictive_entropy(probs_mc_ood)    # out-of-distribution

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(H_test, bins=20, alpha=0.7, color=COLORS['mc'], label='In-distribution (test)', density=True)
ax.hist(H_ood,  bins=20, alpha=0.7, color=COLORS['ood'], label='OOD (corrupted input)', density=True)
ax.axvline(np.median(H_test), color=COLORS['mc'],  linestyle='--', linewidth=1.5, label=f'Median IND = {np.median(H_test):.2f}')
ax.axvline(np.median(H_ood),  color=COLORS['ood'], linestyle='--', linewidth=1.5, label=f'Median OOD = {np.median(H_ood):.2f}')
ax.set_xlabel('Predictive Entropy  H(y|x)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Figure 3 — Predictive Entropy: In-Distribution vs OOD\n(MC Dropout, 30 samples)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('fig3_entropy_ood.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved: fig3_entropy_ood.png")


# ── FIGURE 4: Results Summary Table ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 2.8))
ax.axis('off')

table_data = [
    ['Standard (Baseline)', f'{acc_std:.3f}', f'{ece_std:.4f}', f'{nll_std:.4f}', '—'],
    ['MC Dropout (T=30)',    f'{acc_mc:.3f}',  f'{ece_mc:.4f}',  f'{nll_mc:.4f}',  f'T=30 passes'],
    ['Temperature Scaling',  f'{acc_ts:.3f}',  f'{ece_ts:.4f}',  f'{nll_ts:.4f}',  f'T={T_learned:.2f}'],
]
col_labels = ['Method', 'Accuracy ↑', 'ECE ↓', 'NLL ↓', 'Notes']
colors_rows = [
    ['#fdecea', '#fdecea', '#fdecea', '#fdecea', '#fdecea'],
    ['#e8f4fd', '#e8f4fd', '#e8f4fd', '#e8f4fd', '#e8f4fd'],
    ['#eafaf1', '#eafaf1', '#eafaf1', '#eafaf1', '#eafaf1'],
]

table = ax.table(
    cellText=table_data, colLabels=col_labels,
    cellLoc='center', loc='center',
    cellColours=colors_rows
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 2.0)

# Bold header
for j in range(len(col_labels)):
    table[(0, j)].set_facecolor('#2c3e50')
    table[(0, j)].get_text().set_color('white')
    table[(0, j)].get_text().set_fontweight('bold')

ax.set_title('Figure 4 — Calibration Results Summary (Wine Dataset)', fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('fig4_results_table.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved: fig4_results_table.png")


# ── SECTION 8: SUMMARY PRINTOUT ──────────────────────────────────────────────

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"{'Method':<25} {'Accuracy':>9} {'ECE':>9} {'NLL':>9}")
print("-"*60)
print(f"{'Standard (Baseline)':<25} {acc_std:>9.4f} {ece_std:>9.4f} {nll_std:>9.4f}")
print(f"{'MC Dropout (T=30)':<25} {acc_mc:>9.4f} {ece_mc:>9.4f}  {nll_mc:>9.4f}")
print(f"{'Temperature Scaling':<25} {acc_ts:>9.4f} {ece_ts:>9.4f}  {nll_ts:>9.4f}")
print("="*60)
print(f"\nLearned Temperature T = {T_learned:.4f}")
print("\nFigures saved:")
print("  fig1_reliability_diagrams.png  ← KEY FIGURE for poster")
print("  fig2_ece_comparison.png")
print("  fig3_entropy_ood.png")
print("  fig4_results_table.png")
print("\n✓ Experiment complete! Use these figures in your ICTP poster.")
