"""Speaker configuration and weight data structures for multi-speaker TTS."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SpeakerConfig:
    """Configuration for a single speaker in the multi-speaker TTS engine.

    Each speaker has their own fine-tuned GPT/SoVITS model paths and reference
    audio for extracting timbre embeddings (ge/prompt).
    """

    name: str
    """Unique speaker identifier used for inference routing."""

    gpt_model_path: str
    """Path to the speaker's fine-tuned GPT checkpoint (.ckpt or safetensors dir)."""

    sovits_model_path: str
    """Path to the speaker's fine-tuned SoVITS checkpoint (.pth or safetensors dir)."""

    spk_audio_path: str
    """Path to the speaker's reference audio file (for timbre embedding extraction)."""

    prompt_audio_path: str | None = None
    """Optional path to prompt audio (style/tone reference). Uses spk_audio_path if None."""

    prompt_audio_text: str | None = None
    """Transcription of the prompt audio. Required if prompt_audio_path is set."""


@dataclass
class SpeakerWeights:
    """Pre-extracted speaker-specific weights and cached features.

    These are the lightweight per-speaker components (~5-15% of a full model)
    that get injected into the shared backbone at inference time via copy_().
    All tensors are stored on CPU to minimize VRAM pressure.
    """

    name: str
    """Speaker name matching SpeakerConfig.name."""

    # ── GPT speaker-specific weights ──

    gpt_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    """Speaker-specific GPT layer weights, keyed by parameter path.

    Includes:
      - ar_predict_layer.{weight}    (prediction head)
      - t2s_transformer.blocks.{N-2,N-1}.*  (last 2 transformer blocks)

    Keys match Text2SemanticDecoder's named_parameters() names.
    Tensors are on CPU; copied to GPU via copy_() at inference time.
    """

    # ── SoVITS speaker-specific weights ──

    sovits_weights: dict[str, torch.Tensor] = field(default_factory=dict)
    """Speaker-specific SoVITS layer weights, keyed by parameter path.

    Includes (by prefix match):
      - ref_enc.*       (MelStyleEncoder — core timbre extractor)
      - sv_emb.*        (SV embedding projection, v2Pro+ only)
      - ge_to512.*      (ge dimension adapter, v2Pro+ only)
      - prelu.*         (post-fusion activation, v2Pro+ only)

    Keys match SynthesizerTrn's named_parameters() names.
    Tensors are on CPU.
    """

    # ── Pre-computed features (cached once at init time) ──

    ge: torch.Tensor | None = None
    """Pre-computed speaker embedding (global embedding).
    
    Extracted from the speaker's reference audio via SoVITS's ref_enc and SV model.
    Shape: [1, gin_channels, 1]. Stored on GPU for fast inference access.
    """

    prompt: torch.Tensor | None = None
    """Pre-computed prompt semantic tokens.
    
    Extracted from prompt_audio_path via CNHubert + VQ quantizer.
    Provides style/tone reference for GPT inference.
    """

    phones1: list[int] | None = None
    """Phoneme IDs of the prompt text (pre-computed G2P result)."""

    bert1: torch.Tensor | None = None
    """BERT features of the prompt text (pre-computed CNRoberta output)."""

    sv_emb: torch.Tensor | None = None
    """Speaker verification embedding from ERes2NetV2.
    
    Used by SoVITS.get_ge() for v2Pro+ models. Stored per audio_path
    but cached here for convenience during inference.
    """
