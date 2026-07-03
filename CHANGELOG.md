# Changelog

## [Unreleased] — multi-speaker-inference

### Added

- **MultiSpeakerTTS**: Multi-speaker TTS engine with shared model backbone and per-speaker lightweight weights. Multiple fine-tuned speakers can coexist in a single session without reloading models. (`gsv_tts/MultiSpeaker.py`)
- **SpeakerConfig**: Dataclass for defining a speaker's model paths, reference audio, and prompt configuration. (`gsv_tts/SpeakerWeights.py`)
- **SpeakerWeights**: Dataclass for storing extracted per-speaker weights (~5-15% of full model size) and pre-computed features (ge, prompt, BERT). (`gsv_tts/SpeakerWeights.py`)
- **ConfigMismatchError**: Raised when a speaker model's architecture is incompatible with the base model. Exported in `gsv_tts.__init__`.
- **Config compatibility validation**: Automatic validation of 13 critical architecture keys (GPT: `hidden_dim`, `embedding_dim`, `head`, `n_layer`, `vocab_size`, `phoneme_vocab_size`; SoVITS: `gin_channels`, `inter_channels`, `hidden_channels`, `filter_channels`, `n_heads`, `n_layers`, `upsample_initial_channel`, `version`) before weight extraction.
- **v2Pro ↔ v2ProPlus version compatibility**: v2Pro and v2ProPlus models with matching structural dimensions (`upsample_initial_channel` etc.) are treated as compatible since they share identical code paths (`is_v2pro`, `sv_emb`, `ge_to512`, `prelu`).
- **Automatic degradation fallback**: Incompatible speakers automatically fall back to full model loading (~800MB VRAM) without blocking compatible speakers from using the shared backbone.
- **True GPU batching in `infer_batched`**: Same-speaker texts are delegated to `TTS.infer_batched()` for true GPU parallelism. Multi-speaker batches are grouped by speaker and processed sequentially per group.
- **Runtime speaker management**: `add_speaker()` and `remove_speaker()` for dynamic speaker registration.
- **Speaker weight extraction** (`gsv_tts/Loader.py`):
  - `extract_speaker_gpt_weights()`: Extracts prediction head + last N transformer blocks
  - `extract_speaker_sovits_weights()`: Extracts `ref_enc`, `sv_emb`, `ge_to512`, `prelu`, `dec.cond`, `flow.*.enc.cond_layer.*`
  - `load_shared_gpt()` / `load_shared_sovits()`: Load base models with speaker-specific layers randomized
- **hf-mirror download source**: Added `hf-mirror.com` as a faster alternative for users in China. Auto-selects best mirror via latency check. (`gsv_tts/Download.py`)
- **`ensure_default_models()`**: Automatic download of `s1v3.ckpt` and `s2Gv2ProPlus.pth` with fallback chain across mirrors. (`gsv_tts/Download.py`)
- **SoVITS sharing verification test**: Self-consistency test with MCD and mel cosine similarity metrics. (`tests/test_sovits_sharing.py`)

### Fixed

- SoVITS speaker weight detection now correctly includes `dec.cond.*` and `flow.*.enc.cond_layer.*` layers (previously missed during weight extraction).
- `SpeakerWeights` dataclass now includes `spk_audio_path`, `prompt_audio_path`, `prompt_audio_text` fields that were missing from the initial implementation.

### Changed

- `MultiSpeakerTTS._activate_shared_models()`: Extracted cache lifecycle management into a context manager for cleaner shared model registration and cleanup.
- `MultiSpeakerTTS.infer_batched()`: Rewritten from per-text sequential loop to grouped-by-speaker batch delegation with true GPU parallelism.

---

## [0.4.5] — 2025-06-29

### Added
- Initial PyPI release `gsv-tts-lite==0.4.5`
- CUDA Graph, Nested KV Cache, Continuous Batching optimizations
- Flash Attention support
- Multi-speaker fusion (`spk_audio_path` as dict with weights)
- Word-level subtitle timestamp alignment
- Voice conversion (`infer_vc`)
- Speaker verification (`verify_speaker`)
- WebRTC real-time API
- WebUI presets persistence
- CUDA, MPS (Apple Silicon), and CPU backend support
- CN/JP/EN language support
- V2, V2Pro, V2ProPlus model support
