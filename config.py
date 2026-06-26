# =============================================================================
# config.py — Central configuration for all experiments
# =============================================================================
# Change values here to switch datasets, tune hyperparameters, or
# adjust training settings. No other file needs to be edited.

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET = "miniboone"      # options: "miniboone" | "magic"

# ── Model (base / default config) ────────────────────────────────────────────
D_MODEL    = 64    # embedding dimension per feature token
N_HEAD     = 4     # attention heads (D_MODEL must be divisible by N_HEAD)
N_LAYERS   = 2     # number of Transformer encoder layers
DROPOUT    = 0.2   # dropout rate (regularisation + MC Dropout at test time)

# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS       = 80
BATCH_SIZE   = 256   # larger batch suits MiniBooNE (130 k samples)
LR           = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED  = 42

# ── Uncertainty ──────────────────────────────────────────────────────────────
MC_SAMPLES = 30    # stochastic forward passes for MC Dropout [GAL16]
N_BINS     = 10    # bins for ECE / reliability diagrams      [GUO17]
OOD_NOISE  = 3.0   # Gaussian noise std for OOD test set

# ── Hyperparameter Ablation Study ────────────────────────────────────────────
# One parameter is varied at a time; others stay at the base values above.
# Each entry produces 3 trained models (one per value) × 3 UQ methods = 9 runs.
SWEEP = {
    "d_model":  [32, 64, 128],   # model capacity  (embedding size)
    "n_layers": [1,  2,  4],     # depth           (number of encoder layers)
    "dropout":  [0.1, 0.2, 0.3], # regularisation  (also controls MC uncertainty)
}
SWEEP_EPOCHS = 60   # fewer epochs per sweep run to save time

# ── Output ───────────────────────────────────────────────────────────────────
FIGURES_DIR = "figures"
DPI         = 150
