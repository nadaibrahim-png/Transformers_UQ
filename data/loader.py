# =============================================================================
# data/loader.py — Dataset loading and preprocessing
# =============================================================================
# Paper connection [GUO17]: "We evaluate calibration on held-out test data."
# We split into train / val / test.
# Val is used exclusively for Temperature Scaling — not for model selection.
# OOD data is created by adding strong Gaussian noise to the test features.

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


def load_magic():
    """
    MAGIC Gamma Telescope — UCI (astroparticle physics)
    Task: classify cosmic ray events as gamma (signal) vs hadron (background).
    19,020 samples | 10 physical features | binary classification
    Features: Cherenkov image parameters (Flenner & Heck 1999)
    """
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "magic/magic04.data")
    df = pd.read_csv(url, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = (df.iloc[:, -1].values == "g").astype(int)  # g=gamma(1), h=hadron(0)
    n_classes = 2
    name = "MAGIC Gamma Telescope"
    feature_names = ["fLength", "fWidth", "fSize", "fConc", "fConc1",
                     "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"]
    return X, y, n_classes, name, feature_names


def load_higgs(n_samples=None):
    """
    HIGGS Boson — UCI / TensorFlow Datasets (particle physics)
    Task: classify collision events as Higgs signal vs background.
    Up to 11M samples | 28 kinematic features | binary classification
    Paper: Baldi et al. 2014 — arXiv:1402.4735
    """
    n_samples = n_samples or config.HIGGS_SUBSET
    try:
        import tensorflow_datasets as tfds
        ds = tfds.load("higgs", split=f"train[:{n_samples}]", as_supervised=False)
        df = tfds.as_dataframe(ds)
        y = df.pop("class_label").values.astype(int)
        X = df.values.astype(np.float32)
    except ImportError:
        raise ImportError(
            "Install tensorflow-datasets:  !pip install tensorflow-datasets"
        )
    n_classes = 2
    name = f"HIGGS Boson (n={n_samples:,})"
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return X, y, n_classes, name, feature_names


def load_wine_dataset():
    """
    Wine Quality — sklearn built-in (chemistry / food science)
    Task: classify wine into 3 quality classes from chemical measurements.
    178 samples | 13 features | 3-class classification
    Fast to train — good for initial debugging.
    """
    data = load_wine()
    X = data.data.astype(np.float32)
    y = data.target
    n_classes = len(np.unique(y))
    name = "Wine (sklearn)"
    feature_names = data.feature_names
    return X, y, n_classes, name, feature_names


# ── Dataset registry ─────────────────────────────────────────────────────────
DATASETS = {
    "magic": load_magic,
    "higgs": load_higgs,
    "wine":  load_wine_dataset,
}


def get_dataset(name=None):
    """Load dataset by name (defaults to config.DATASET)."""
    name = name or config.DATASET
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}")
    print(f"\n── Loading dataset: {name} ──")
    return DATASETS[name]()


def preprocess(X, y, random_seed=None):
    """
    Split into train / val / test and standardise features.
    Returns numpy arrays (train/val/test) + fitted scaler.

    Split ratios: 60% train | 20% val | 20% test
    Val set is reserved for Temperature Scaling only [GUO17 §3.3].
    """
    seed = random_seed or config.RANDOM_SEED

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, random_state=seed, stratify=y_tv
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler


def make_ood(X_test, noise_std=None):
    """
    Create Out-of-Distribution data by corrupting test features with noise.
    Paper connection [GAL16 §4]: a well-calibrated model should assign
    HIGH predictive entropy to OOD inputs.
    """
    std = noise_std or config.OOD_NOISE
    return X_test + np.random.normal(0, std, X_test.shape).astype(np.float32)


def to_loader(X, y, batch_size=None, shuffle=False):
    """Wrap numpy arrays into a PyTorch DataLoader."""
    bs = batch_size or config.BATCH_SIZE
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def get_loaders(X_train, y_train, X_val, y_val, X_test, y_test, X_ood):
    """Bundle all splits into DataLoaders."""
    return {
        "train": to_loader(X_train, y_train, shuffle=True),
        "val":   to_loader(X_val,   y_val),
        "test":  to_loader(X_test,  y_test),
        "ood":   to_loader(X_ood,   y_test),   # same labels, corrupted features
    }
