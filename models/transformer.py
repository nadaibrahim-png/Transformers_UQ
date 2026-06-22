# =============================================================================
# models/transformer.py — Transformer architecture for tabular data
# =============================================================================
# Architecture overview:
#   Each of the N input features is treated as a "token" (like a word in NLP).
#   A linear layer embeds each scalar → D-dimensional vector.
#   A Transformer encoder applies self-attention across the N feature tokens.
#   Mean pooling → MC Dropout → linear classifier head.
#
# Paper connection [GAL16]:
#   Dropout is placed right before the classifier head. At test time we call
#   enable_mc_dropout() to keep ONLY that layer stochastic, while BatchNorm
#   and other layers stay in eval mode — this is Algorithm 1 from Gal 2016.

import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


class TabTransformer(nn.Module):
    """
    Transformer encoder for tabular (scientific) data.

    Each input feature is treated as a token. Self-attention allows the model
    to learn interactions between features — e.g. how fLength and fWidth
    jointly determine whether a gamma-ray event is real.

    Args:
        n_features : number of input features (columns in your dataset)
        n_classes  : number of output classes
        d_model    : embedding dimension for each feature token
        nhead      : number of self-attention heads
        num_layers : number of stacked Transformer encoder layers
        dropout    : dropout probability (used in Transformer + MC Dropout head)
    """

    def __init__(self, n_features, n_classes,
                 d_model=None, nhead=None, num_layers=None, dropout=None):
        super().__init__()
        d_model    = d_model    or config.D_MODEL
        nhead      = nhead      or config.N_HEAD
        num_layers = num_layers or config.N_LAYERS
        dropout    = dropout    or config.DROPOUT

        # ── 1. Feature embedding ─────────────────────────────────────────────
        # Each scalar feature → d_model-dimensional vector.
        # This is equivalent to the word embedding step in NLP.
        self.feature_embed = nn.Linear(1, d_model)

        # Learnable positional embedding: one per feature (not per time step).
        # Tells the Transformer which "slot" each feature occupies.
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── 2. Transformer encoder ───────────────────────────────────────────
        # Self-attention: each feature attends to every other feature.
        # Paper connection [VASWANI17]: "Attention is All You Need"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True         # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── 3. MC Dropout head ───────────────────────────────────────────────
        # ★ This is the key layer for MC Dropout inference.
        # During training: acts as normal regularisation dropout.
        # During MC inference: kept active (stochastic) via enable_mc_dropout().
        # Paper connection [GAL16 §3]: "dropout before every weight layer"
        self.mc_dropout = nn.Dropout(p=dropout)

        # Final classifier: mean-pooled representation → class logits
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

        self.n_features = n_features
        self.n_classes  = n_classes
        self.d_model    = d_model

        self._init_weights()

    def _init_weights(self):
        """Xavier initialisation for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x : (batch_size, n_features)  — standardised input features
        Returns:
            logits : (batch_size, n_classes)  — raw scores (before softmax)
        """
        # Reshape → (batch, n_features, 1) → embed → (batch, n_features, d_model)
        x = self.feature_embed(x.unsqueeze(-1))

        # Add positional embedding
        x = x + self.pos_embed

        # Self-attention across feature tokens → (batch, n_features, d_model)
        x = self.transformer(x)

        # Mean pool across feature dimension → (batch, d_model)
        x = x.mean(dim=1)

        # ★ MC Dropout — stochastic at test time when enable_mc_dropout() called
        x = self.mc_dropout(x)

        # Class logits → (batch, n_classes)
        return self.classifier(x)

    def enable_mc_dropout(self):
        """
        Set the whole model to eval (deterministic) EXCEPT the MC Dropout layer.
        Call this before MC Dropout inference — do NOT call model.eval() after.

        Paper connection [GAL16] Algorithm 1:
            'Keep dropout active during test time to approximate the posterior
             predictive distribution via Monte Carlo sampling.'
        """
        self.eval()
        self.mc_dropout.train()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
