# =============================================================================
# config.py — Central configuration for all experiments
# =============================================================================
# Change values here to switch datasets, tune hyperparameters, or
# adjust training settings. No other file needs to be edited.

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET = "magic"          # options: "magic" | "higgs" | "wine"
HIGGS_SUBSET = 50_000      # only used when DATASET="higgs"

# ── Model ────────────────────────────────────────────────────────────────────
D_MODEL    = 64    # embedding dimension for each feature token
N_HEAD     = 4     # number of attention heads (D_MODEL must be divisible by N_HEAD)
N_LAYERS   = 2     # number of Transformer encoder layers
DROPOUT    = 0.2   # dropout rate (used for both regularisation and MC Dropout)

# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS      = 80
BATCH_SIZE  = 64
LR          = 1e-3
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

# ── Uncertainty ──────────────────────────────────────────────────────────────
MC_SAMPLES   = 30    # number of stochastic forward passes for MC Dropout
N_BINS       = 10    # number of bins for ECE and reliability diagrams
OOD_NOISE    = 3.0   # std of Gaussian noise added to create OOD test data

# ── Active Learning ──────────────────────────────────────────────────────────
AL_INITIAL_SIZE     = 20    # seed labeled set size (samples per class × n_classes)
AL_QUERY_SIZE       = 10    # samples to query per round
AL_N_ROUNDS         = 15    # number of active learning rounds
AL_EPOCHS_PER_ROUND = 40    # training epochs per round (fewer than full training)

# ── Output ───────────────────────────────────────────────────────────────────
FIGURES_DIR = "figures"   # folder where poster figures are saved
DPI         = 200         # figure resolution
