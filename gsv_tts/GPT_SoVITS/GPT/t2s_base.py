"""Shared base classes for GPT Text-to-Semantic decoders.

Concrete implementations live in:
    - ``t2s_model.py``   — standard ``F.scaled_dot_product_attention``
    - ``t2s_model_flash_attn.py`` — ``flash_attn_with_kvcache``
"""
from __future__ import annotations

from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .utils import sample
from .embedding import SinePositionalEmbedding, TokenEmbedding


# ────────────────────────────────────────────────────────────────────
# 1.  T2SBlock — individual transformer block
# ────────────────────────────────────────────────────────────────────

class _T2SBlockBase(nn.Module):
    """Shared init and post-attention residual/MLP/norm path.

    Subclasses MUST override ``decode_next_token`` (the attention
    mechanism differs between SDPA and flash_attn_with_kvcache).

    ``process_prompt`` is also left to subclasses because the KV cache
    tensor layout differs.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    # ── to be overridden by subclasses ──

    def process_prompt(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def decode_next_token(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


# ────────────────────────────────────────────────────────────────────
# 2.  T2STransformer — stack of T2SBlock
# ────────────────────────────────────────────────────────────────────

class _T2STransformerBase(nn.Module):
    """Shared init and process_prompt.  decode_next_token is subclassed."""

    def __init__(self, num_blocks: int, blocks: List[_T2SBlockBase]):
        super().__init__()
        self.num_blocks: int = num_blocks
        self.blocks = nn.ModuleList(blocks)

    def process_prompt(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_cache_len: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(self.num_blocks):
            x = self.blocks[i].process_prompt(
                x, k_cache[i], v_cache[i], attn_mask,
            )
        kv_cache_len.fill_(x.shape[1])
        return x

    def decode_next_token(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


# ────────────────────────────────────────────────────────────────────
# 3.  Bucket — CUDA Graph metadata
# ────────────────────────────────────────────────────────────────────

class _BucketBase:
    """Fields common to both SDPA and flash-attention code paths."""
    cuda_graph = None
    graph_xy_pos: torch.Tensor = None
    graph_xy_dec: torch.Tensor = None
    kv_cache_len: torch.Tensor = None
    k_cache: torch.Tensor = None
    v_cache: torch.Tensor = None
    max_kv_cache: int = None
    batch_size: int = None


# ────────────────────────────────────────────────────────────────────
# 4.  Text2SemanticDecoderBase — main decoder
# ────────────────────────────────────────────────────────────────────

class Text2SemanticDecoderBase(nn.Module):
    """Shared ``__init__``, ``process_batch_data``, and ``process_single_data``.

    Subclasses must implement:
        - ``decode_next_token`` (T2SBlock + T2STransformer)
        - ``_build_t2s_block(hidden_dim, num_heads) -> _T2SBlockBase``
        - ``initialize_runtime``  (different KV cache layout)
        - ``_setup_bucket_extras``  (decode_attn_mask, batch_indices if needed)
        - ``_update_decode_mask`` / ``_reset_decode_mask`` (no-ops for flash_attn)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]

        self.suppressed_tokens = [280, 486, self.EOS]

        self.bert_proj = nn.Linear(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim, self.phoneme_vocab_size, self.p_dropout,
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True,
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim, self.vocab_size, self.p_dropout,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True,
        )

        self.ar_predict_layer = nn.Linear(self.model_dim, self.vocab_size, bias=False)

        blocks = []
        for _ in range(self.num_layers):
            blocks.append(self._build_t2s_block(self.model_dim, self.num_head))
        self.t2s_transformer = _T2STransformerBase(self.num_layers, blocks)

        self.cuda_graph_buckets = {}

    # ── hooks for subclasses ──

    def _build_t2s_block(self, hidden_dim: int, num_heads: int) -> _T2SBlockBase:
        raise NotImplementedError

    def initialize_runtime(self, dtype, device, gpt_cache):
        raise NotImplementedError

    def _setup_bucket_extras(self, bucket):
        """Set extra fields on a bucket (e.g. decode_attn_mask, batch_indices)."""
        pass

    def _update_decode_mask(self, bucket, batch_indices=None):
        """Per-step update of decode mask (no-op for flash_attn)."""
        pass

    def _reset_decode_mask(self, max_bucket, batch_indices=None):
        """Reset decode mask when a new sequence enters a slot."""
        pass

    def _get_cache_seq_dim(self, kv_layout: str = "BH") -> int:
        """Return the dimension index for the sequence length in KV cache."""
        return 2 if kv_layout == "BH" else 1  # BH=heads-major, BS=seq-major

    # ────────────────────────────────────────────────────────────
    # Shared data-processing methods (IDENTICAL in both files)
    # ────────────────────────────────────────────────────────────

    def _process_batch_data(self, x, y, bert_feature, x_lens, y_lens):
        device = x.device
        B = x.shape[0]

        xy_lens = x_lens + y_lens

        xy_len = xy_lens.max()
        x_len = x_lens.max()
        y_len = y_lens.max()

        xy_indices = torch.arange(xy_len, device=device).unsqueeze(0)
        x_mask1 = xy_indices < x_lens
        indices = torch.arange(x_len, device=device)
        x_mask2 = indices.unsqueeze(0) < x_lens

        y_mask1 = (x_lens <= xy_indices) & (xy_indices < xy_lens)
        indices = torch.arange(y_len, device=device).unsqueeze(0)
        y_mask2 = indices < y_lens

        last_token_mask = xy_indices == xy_lens - 1

        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature)
        x = self.ar_text_position(x)

        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_pos = torch.zeros((B, xy_len, self.model_dim), dtype=y_pos.dtype, device=device)
        xy_pos[x_mask1] = x[x_mask2]
        xy_pos[y_mask1] = y_pos[y_mask2]

        # 音素可以关注自身(双向),但不能关注音频 音频可以关注自身(因果),也能关注音素(双向)
        prompt_attn_mask = torch.zeros((B, xy_len, xy_len), dtype=torch.bool, device=device)

        x_attn_mask = x_mask1.unsqueeze(1).expand(-1, x_len, -1).clone()
        prompt_attn_mask[x_mask1] = x_attn_mask[x_mask2]

        y_attn_mask = x_mask1.unsqueeze(1).expand(-1, y_len, -1).clone()
        tril_mask = torch.tril(torch.ones(B, y_len, xy_len, dtype=torch.bool, device=device))
        mask = xy_indices < (xy_len - x_lens)
        mask = mask.unsqueeze(1).expand(-1, y_len, -1)
        y_attn_mask[~y_attn_mask] = tril_mask[mask]
        prompt_attn_mask[y_mask1] = y_attn_mask[y_mask2]

        prompt_attn_mask = prompt_attn_mask.unsqueeze(1).expand(-1, self.num_head, -1, -1)
        # PyTorch convention: True = mask (do NOT attend).
        # Our construction uses True = "can attend", so invert.
        prompt_attn_mask = ~prompt_attn_mask

        return xy_pos, last_token_mask, prompt_attn_mask

    def process_single_data(self, x, y, bert_feature):
        x_len = x.shape[1]
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature)
        x = self.ar_text_position(x)

        y_len = y.shape[1]
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_pos = torch.concat([x, y_pos], dim=1)

        B, device = x.shape[0], x.device

        # 音素可以关注自身(双向),但不能关注音频 音频可以关注自身(因果),也能关注音素(双向)
        x_attn_mask = F.pad(
            torch.ones((x_len, x_len), dtype=torch.bool),
            (0, y_len), value=False,
        )
        y_attn_mask = F.pad(
            torch.tril(torch.ones((y_len, y_len), dtype=torch.bool)),
            (x_len, 0), value=True,
        )
        prompt_attn_mask = (
            torch.concat([x_attn_mask, y_attn_mask], dim=0)
            .unsqueeze(0).unsqueeze(0)
            .expand(B, self.num_head, -1, -1)
            .to(device=device, dtype=torch.bool)
        )
        # PyTorch convention: True = mask. Our construction uses True = can attend.
        prompt_attn_mask = ~prompt_attn_mask

        return xy_pos, prompt_attn_mask
