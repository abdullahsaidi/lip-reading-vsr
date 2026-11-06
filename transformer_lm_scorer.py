"""
transformer_lm_scorer.py — Adapt our trained Transformer LM to BatchBeamSearch.

Background:
  BatchBeamSearch expects each scorer to implement BatchScorerInterface
  (or BatchPartialScorerInterface), which means it must provide:
    - batch_score(ys, states, xs) -> (logp, new_states)
        where ys: (n_batch, ylen) prefix tokens
              states: list of decoder states for each batch item
              xs: encoder feature (we IGNORE this -- LM doesn't use audio/video)
        returns logp: (n_batch, vocab_size) log-probabilities for the NEXT token
                states: new states (we use cached past_kv via PyTorch's standard
                        approach -- but a simple recompute-each-step also works)

  We implement the simple "recompute each step" version. This is slower than
  caching past keys/values but it's bug-proof and the LM is small (30M params),
  so per-step latency is acceptable.

  Token-id contract: the LM was trained on the SAME SentencePiece vocabulary
  as the visual model, ids 0..vocab_size-1. So no token mapping is needed.

  CRITICAL: log-probabilities are computed across the FULL vocab. The visual
  model's beam search will combine these with decoder/CTC scores using:
      combined = w_decoder * logp_decoder + w_ctc * logp_ctc + w_lm * logp_lm
  All terms must be in log-probability space (not raw logits), which is what
  log_softmax gives us.
"""

import json
import os
from typing import Any, List, Tuple

import torch
import torch.nn.functional as F

from espnet.nets.scorer_interface import BatchScorerInterface


# ---------------------------------------------------------------------------
# We re-import the model class from train_lm.py so the saved config matches.
# With train_lm.py on PYTHONPATH it is importable as a module.
# But to keep this scorer file self-contained we copy the architecture verbatim
# below (small, ~200 lines). If you change the LM architecture, change BOTH.
# ---------------------------------------------------------------------------

import torch.nn as nn


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout, block_size):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout_p = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))


class _TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttention(d_model, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class _TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_layer=6, n_head=8, d_model=512,
                 d_ff=2048, dropout=0.1, block_size=256, pad_id=-1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model, n_head, d_ff, dropout, block_size)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie input/output embeddings (must match training)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        """idx: (B, T) -> logits (B, T, vocab_size)"""
        B, T = idx.shape
        # If T exceeds block_size, truncate from the LEFT (keep most recent context).
        # In beam search this happens when a beam reaches a long prefix.
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            T = self.block_size
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        # Note: we don't apply dropout at inference but it's a no-op in eval mode.
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Loader: reconstruct the trained LM from a best.pt produced by train_lm.py
# ---------------------------------------------------------------------------

def load_transformer_lm(ckpt_path, device):
    """Load a TransformerLM from a train_lm.py checkpoint.

    The checkpoint is a dict with keys:
        model_state_dict: parameters
        config: dict of architecture hyperparameters
        step / val_ppl / args: metadata (ignored here)
    """
    print(f"[lm-scorer] loading LM checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = _TransformerLM(**config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"[lm-scorer] loaded LM with vocab={config['vocab_size']}, "
          f"n_layer={config['n_layer']}, d_model={config['d_model']}, "
          f"block_size={config['block_size']}")
    return model


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------

class TransformerLMScorer(BatchScorerInterface):
    """Wrap the LM as a BatchScorerInterface for shallow fusion.

    State strategy: simple "no state, recompute every step." We just keep the
    full prefix and recompute logits at every call. For a 30M LM this is ~5ms
    per step on a 5090, which is acceptable inside beam search.

    A more performance-tuned implementation would cache past_kv across steps,
    but that's invasive and easy to get wrong. We choose correctness over speed.

    WARNING: the visual model's eos token id is `model.odim - 1`.
    The LM was trained with eos at id 5049 too (see meta.json from tokenize_corpus.py).
    These match, so beam search's eos handling carries through naturally.
    """

    def __init__(self, lm, vocab_size, eos_id):
        self.lm = lm
        self.vocab_size = vocab_size
        self.eos_id = eos_id

    @torch.no_grad()
    def init_state(self, x):
        """Get initial state for one beam. We don't use stateful caching, just
        return None as a placeholder.
        Args:
            x: encoder feature (ignored; LM doesn't use it)
        """
        return None

    @torch.no_grad()
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score the next-token distribution for each beam.

        Args:
            ys: (n_batch, ylen) -- prefix token ids (already on device)
            states: list of length n_batch (we ignore each element)
            xs: encoder feature (ignored)

        Returns:
            logp: (n_batch, vocab_size) -- log-probabilities for next token
            states: new states list (we just return [None] * n_batch)
        """
        # Forward through LM. ys already includes <sos>=eos_id at position 0
        # by BatchBeamSearch convention.
        logits = self.lm(ys)            # (B, T, V_lm)
        # Take logits at the LAST position -- that's the prediction for the
        # *next* token after the prefix.
        last_logits = logits[:, -1, :]  # (B, V_lm)

        # IMPORTANT: the LM's vocabulary may be slightly larger than the visual
        # model's (e.g. LM has 5050 entries, visual has 5049). We slice to
        # self.vocab_size (the visual vocab) so the shape matches what beam
        # search expects when adding scores together. Any LM-only ids are
        # dropped -- they're unused by the visual model anyway.
        if last_logits.size(-1) != self.vocab_size:
            last_logits = last_logits[:, : self.vocab_size]

        # Convert to log-probabilities so it's compatible with the other scorers.
        logp = F.log_softmax(last_logits.float(), dim=-1)
        return logp, [None] * ys.size(0)
