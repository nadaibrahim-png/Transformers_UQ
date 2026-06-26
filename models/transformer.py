# =============================================================================
# models/transformer.py — FT-Transformer for tabular scientific data
# =============================================================================
# Architecture: Gorishniy et al. 2021 — "Revisiting Deep Learning Models
#               for Tabular Data" (arXiv:2106.05346)
#
# Key difference from TabTransformer:
#   TabTransformer embeds ONLY categorical features → attention is inactive
#   on continuous-only datasets like MiniBooNE.
#
#   FT-Transformer gives each continuous feature its OWN linear projection
#   (separate weight matrix per feature), so every feature becomes a proper
#   token and self-attention operates over all 50 kinematic variables.
#
# MC Dropout connection [GAL16]:
#   A dedicated dropout layer sits before the classifier head.
#   enable_mc_dropout() keeps it stochastic at test time while the rest of
#   the model stays deterministic — approximating a Bayesian posterior.
#
# Attention weights connection (poster):
#   get_attention_weights() extracts how much each feature is attended to.
#   This is "implicit feature selection" — the model learns which of the
#   50 MiniBooNE kinematic variables matter most, without manual filtering.

import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular data.

    Each continuous feature xᵢ is projected to a d_model-dimensional token
    via its own learned linear map:  tᵢ = Wᵢ · xᵢ + bᵢ

    A learnable [CLS] token is prepended; its output after all Transformer
    layers feeds the classifier head.

    Args:
        n_features : number of input features (50 for MiniBooNE)
        n_classes  : number of output classes (2 for binary)
        d_model    : token embedding dimension
        nhead      : number of self-attention heads (must divide d_model)
        num_layers : number of stacked Transformer encoder layers
        dropout    : dropout probability (regularisation + MC Dropout head)
    """

    def __init__(self, n_features, n_classes,
                 d_model=None, nhead=None, num_layers=None, dropout=None):
        super().__init__()
        d_model    = d_model    or config.D_MODEL
        nhead      = nhead      or config.N_HEAD
        num_layers = num_layers or config.N_LAYERS
        dropout    = dropout    or config.DROPOUT

        self.n_features = n_features
        self.n_classes  = n_classes
        self.d_model    = d_model

        # ── 1. Feature Tokenizer ─────────────────────────────────────────────
        # One Linear(1 → d_model) per feature — each feature has its OWN
        # weight, unlike TabTransformer which shares a single embedding.
        # Paper [GOR21 §3.1]: "each feature is tokenized independently."
        self.feature_tokenizers = nn.ModuleList([
            nn.Linear(1, d_model, bias=True) for _ in range(n_features)
        ])

        # ── 2. [CLS] token ───────────────────────────────────────────────────
        # Learnable classification token prepended to the feature sequence.
        # After Transformer layers, the CLS output summarises the whole input.
        # Analogous to [CLS] in BERT (Devlin et al. 2019).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── 3. Transformer Encoder ───────────────────────────────────────────
        # Self-attention: every token (feature) attends to every other token.
        # dim_feedforward = 4 × d_model follows the original FT-Transformer.
        # norm_first=True (Pre-LN) gives more stable training [XIONG20].
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

        # ── 4. MC Dropout head ───────────────────────────────────────────────
        # ★ This single layer is the engine of MC Dropout inference.
        # Training: normal regularisation dropout.
        # Test (MC mode): kept stochastic via enable_mc_dropout().
        # Paper connection [GAL16 §3]: each stochastic forward pass samples
        # a different set of dropped neurons → approximates Bayesian posterior.
        self.mc_dropout = nn.Dropout(p=dropout)

        # ── 5. Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes)
        )

        self._init_weights()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _init_weights(self):
        """Xavier initialisation for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _tokenize(self, x):
        """
        Convert raw features to token sequence with CLS token.

        Args:
            x : (batch, n_features) — standardised input
        Returns:
            tokens : (batch, n_features + 1, d_model)
                     token 0 = CLS, tokens 1..n = feature tokens
        """
        # Each feature through its own projection → (batch, n_features, d_model)
        feature_tokens = torch.stack([
            self.feature_tokenizers[i](x[:, i:i+1])
            for i in range(self.n_features)
        ], dim=1)

        # Prepend CLS token → (batch, n_features+1, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls, feature_tokens], dim=1)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x):
        """
        Args:
            x : (batch, n_features) — standardised input features
        Returns:
            logits : (batch, n_classes) — raw scores (pre-softmax)
        """
        tokens = self._tokenize(x)                    # (B, F+1, D)
        out    = self.transformer(tokens)              # (B, F+1, D)
        cls_out = out[:, 0, :]                        # CLS token → (B, D)
        return self.classifier(self.mc_dropout(cls_out))

    # ── MC Dropout ────────────────────────────────────────────────────────────

    def enable_mc_dropout(self):
        """
        Set the whole model to eval (deterministic) EXCEPT the MC Dropout
        layer. Call this before every MC Dropout inference run.

        Paper connection [GAL16] Algorithm 1:
            'Keep dropout active at test time to sample from the approximate
             posterior over model weights.'
        """
        self.eval()
        self.mc_dropout.train()

    # ── Attention weights for feature importance ──────────────────────────────

    def get_attention_weights(self, x):
        """
        Extract averaged attention weights from the LAST Transformer layer.

        This reveals which features the model attends to most — equivalent
        to implicit feature selection learned end-to-end from data.

        For your poster: row 0 of the returned matrix = how strongly the
        CLS token attends to each of the 50 MiniBooNE features.

        Args:
            x : (batch, n_features) tensor on the model's device
        Returns:
            weights : (n_features+1, n_features+1) numpy array
                      averaged over the batch dimension
                      axis-0 = query token, axis-1 = key token
        """
        self.eval()
        with torch.no_grad():
            tokens = self._tokenize(x)

            # Pass through all encoder layers except the last
            for layer in self.transformer.layers[:-1]:
                tokens = layer(tokens)

            # Last layer: explicitly request attention weights
            last_layer  = self.transformer.layers[-1]
            tokens_norm = last_layer.norm1(tokens)       # Pre-LN
            _, weights  = last_layer.self_attn(
                tokens_norm, tokens_norm, tokens_norm,
                need_weights=True,
                average_attn_weights=True                # average over heads
            )
        # weights: (batch, n_tokens, n_tokens) → average over batch
        return weights.mean(dim=0).cpu().numpy()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
