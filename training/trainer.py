# =============================================================================
# training/trainer.py — Model training loop
# =============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, loaders, epochs=None, lr=None):
    """
    Train the FT-Transformer with Adam + CosineAnnealingLR + gradient clipping.
    Saves best checkpoint by validation loss and restores it at the end.

    Args:
        model   : FTTransformer instance
        loaders : dict with 'train' and 'val' DataLoaders
        epochs  : number of training epochs (default: config.EPOCHS)
        lr      : learning rate (default: config.LR)

    Returns:
        history : dict with train_loss, val_loss, val_acc per epoch
    """
    epochs = epochs or config.EPOCHS
    lr     = lr     or config.LR

    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Cosine annealing: smoothly decays LR to near-zero over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state    = None
    history       = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\n── Training ({DEVICE}) — {model.count_parameters():,} parameters ──")

    for epoch in range(epochs):

        # ── Training pass ────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for X_batch, y_batch in loaders["train"]:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # ── Validation pass ──────────────────────────────────────────────────
        model.eval()
        val_losses, preds, trues = [], [], []
        with torch.no_grad():
            for X_batch, y_batch in loaders["val"]:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                val_losses.append(criterion(logits, y_batch).item())
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                trues.extend(y_batch.cpu().numpy())

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        va = accuracy_score(trues, preds)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        # Save best model state by val loss
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"train_loss={tl:.4f}  val_loss={vl:.4f}  val_acc={va:.3f}")

    # Restore best weights
    model.load_state_dict(best_state)
    print(f"\n  Best val loss: {best_val_loss:.4f}")
    return history
