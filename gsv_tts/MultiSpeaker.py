"""Multi-speaker TTS engine with shared model backbone and per-speaker weights.

Core idea:
- One shared GPT backbone + one shared SoVITS backbone (loaded once at init)
- Each speaker adds only ~5-15% lightweight weights (predict_layer, ref_enc, etc.)
- At inference time, speaker weights are injected via copy_() (CUDA Graph safe)
- All speakers can be used in a single session without reloading models
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Generator

import torch
import numpy as np

from .TTS import TTS
from .SpeakerWeights import SpeakerConfig, SpeakerWeights
from .Loader import (
    extract_speaker_gpt_weights,
    extract_speaker_sovits_weights,
    load_shared_gpt,
    load_shared_sovits,
)
from .TextProcessor import get_phones_and_bert, sub2text_index
from .Player import AudioClip


logger = logging.getLogger(__name__)


def _resolve_param(model: torch.nn.Module, param_path: str) -> torch.nn.Parameter:
    """Resolve a dotted parameter path (e.g. 't2s_transformer.blocks.0.qkv.weight')
    to the actual nn.Parameter in the model tree."""
    parts = param_path.split(".")
    obj = model
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    if not isinstance(obj, torch.nn.Parameter):
        raise TypeError(f"Expected nn.Parameter at '{param_path}', got {type(obj)}")
    return obj


class MultiSpeakerTTS:
    """Multi-speaker TTS engine — shared backbone + per-speaker lightweight weights.

    Usage:
        speakers = [
            SpeakerConfig(name="alice", gpt_model_path="alice_gpt.ckpt",
                          sovits_model_path="alice_sovits.pth",
                          spk_audio_path="alice_ref.wav"),
            SpeakerConfig(name="bob",   gpt_model_path="bob_gpt.ckpt",
                          sovits_model_path="bob_sovits.pth",
                          spk_audio_path="bob_ref.wav"),
        ]
        tts = MultiSpeakerTTS(speakers=speakers, use_bert=True)
        audio = tts.infer("alice", "Hello world!")
        audio.play()
        tts.audio_queue.wait()
    """

    def __init__(
        self,
        speakers: list[SpeakerConfig],
        base_gpt_path: str | None = None,
        base_sovits_path: str | None = None,
        models_dir: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        use_flash_attn: bool = False,
        use_bert: bool = False,
        shared_gpt_layers: int | None = None,
        gpt_cache: list[tuple[int, int]] | None = None,
        sovits_cache: list[int] | None = None,
    ):
        """Initialize multi-speaker TTS engine.

        Args:
            speakers: List of speaker configurations. Each speaker gets its own
                      fine-tuned prediction head and speaker embedding components.
            base_gpt_path: Path to the base GPT model for shared backbone.
                           Defaults to ~/.cache/gsv/s1v3.ckpt.
            base_sovits_path: Path to the base SoVITS model for shared backbone.
                              Defaults to ~/.cache/gsv/s2Gv2ProPlus.pth.
            models_dir: Override models directory.
            device: Override device (cuda/mps/cpu).
            dtype: Override dtype (float32/float16/bfloat16).
            use_flash_attn: Enable Flash Attention for GPT.
            use_bert: Pre-load Chinese BERT at init.
            shared_gpt_layers: Number of GPT transformer layers to share.
                               None = n_layer - 2 (keep last 2 per speaker).
            gpt_cache: Override GPT CUDA Graph static cache sizes.
            sovits_cache: Override SoVITS CUDA Graph static cache sizes.
        """
        if not speakers:
            raise ValueError("At least one SpeakerConfig is required.")

        # ── Create underlying TTS instance (shared infrastructure) ──
        tts_kwargs = {
            "models_dir": models_dir,
            "device": device,
            "dtype": dtype,
            "use_flash_attn": use_flash_attn,
            "use_bert": use_bert,
        }
        if gpt_cache is not None:
            tts_kwargs["gpt_cache"] = gpt_cache
        if sovits_cache is not None:
            tts_kwargs["sovits_cache"] = sovits_cache
        self._tts = TTS(**{k: v for k, v in tts_kwargs.items() if v is not None})

        # ── Load shared backbones ──
        if base_gpt_path is None:
            base_gpt_path = str(Path(self._tts.models_dir) / "s1v3.ckpt")
        if base_sovits_path is None:
            base_sovits_path = str(Path(self._tts.models_dir) / "s2Gv2ProPlus.pth")

        logger.info(f"Loading shared GPT backbone from: {base_gpt_path}")
        self._shared_gpt = load_shared_gpt(
            base_gpt_path, self._tts.tts_config, shared_layers=shared_gpt_layers
        )
        logger.info(f"Loading shared SoVITS backbone from: {base_sovits_path}")
        self._shared_sovits = load_shared_sovits(base_sovits_path, self._tts.tts_config)

        # ── Extract per-speaker weights ──
        self._speakers: dict[str, SpeakerWeights] = {}
        self._shared_gpt_layers = shared_gpt_layers  # store for logging

        for i, spk in enumerate(speakers):
            logger.info(
                f"Extracting speaker weights [{i + 1}/{len(speakers)}]: {spk.name}"
            )
            self._add_speaker(spk)

        # ── Active speaker tracking ──
        self._active_speaker: str | None = None

        # ── Expose shared resources ──
        self.audio_queue = self._tts.audio_queue
        self.samplerate = self._tts.samplerate

        logger.info(
            f"MultiSpeakerTTS ready with {len(self._speakers)} speaker(s). "
            f"Shared GPT layers: {self._shared_gpt_layers or 'n_layer - 2'}"
        )

    # ==================================================================
    # Speaker management
    # ==================================================================

    def _add_speaker(self, spk: SpeakerConfig):
        """Extract and cache speaker-specific weights and features."""
        weights = SpeakerWeights(name=spk.name)

        # Extract model weights from checkpoints
        weights.gpt_weights = extract_speaker_gpt_weights(
            spk.gpt_model_path,
            self._tts.tts_config,
            shared_layers=self._shared_gpt_layers,
        )
        weights.sovits_weights = extract_speaker_sovits_weights(
            spk.sovits_model_path,
            self._tts.tts_config,
        )

        # Pre-compute speaker embedding (ge) via the SoVITS model
        # We load the full SoVITS model temporarily to get ge, then discard it
        self._tts.load_sovits_model(spk.sovits_model_path)
        self._tts.cache_spk_audio(spk.spk_audio_path, sovits_model=spk.sovits_model_path)
        spk_cache = self._tts.spk_audio_cache[spk.spk_audio_path]
        weights.ge = spk_cache["ge"][spk.sovits_model_path]
        weights.sv_emb = spk_cache.get("sv_emb")

        # Pre-compute prompt features (if prompt audio is configured)
        prompt_audio = spk.prompt_audio_path or spk.spk_audio_path
        prompt_text = spk.prompt_audio_text
        if prompt_text is not None:
            self._tts.cache_prompt_audio(
                prompt_audio_paths=prompt_audio,
                prompt_audio_texts=prompt_text,
            )
            prompt_cache = self._tts.prompt_audio_cache[prompt_audio]
            weights.prompt = prompt_cache["prompt"]
            weights.phones1 = prompt_cache["phones1"]
            weights.bert1 = prompt_cache["bert1"]
        else:
            logger.warning(
                f"Speaker '{spk.name}' has no prompt_audio_text — "
                "prompt features will need to be provided at inference time."
            )

        # Clean up: unload the temporarily loaded full model
        self._tts.unload_sovits_model(spk.sovits_model_path)

        self._speakers[spk.name] = weights
        logger.info(
            f"  Speaker '{spk.name}' ready: "
            f"{len(weights.gpt_weights)} GPT keys, "
            f"{len(weights.sovits_weights)} SoVITS keys"
        )

    def add_speaker(self, spk: SpeakerConfig):
        """Add a new speaker at runtime."""
        if spk.name in self._speakers:
            raise ValueError(f"Speaker '{spk.name}' already exists.")
        self._add_speaker(spk)

    def remove_speaker(self, name: str):
        """Remove a speaker at runtime."""
        if name not in self._speakers:
            raise ValueError(f"Speaker '{name}' not found.")
        del self._speakers[name]
        if self._active_speaker == name:
            self._active_speaker = None
        logger.info(f"Removed speaker: {name}")

    @property
    def speaker_names(self) -> list[str]:
        """List all registered speaker names."""
        return list(self._speakers.keys())

    # ==================================================================
    # Weight injection
    # ==================================================================

    def _apply_speaker(self, name: str):
        """Inject speaker-specific weights into the shared backbone via copy_().

        This is safe for CUDA Graphs because copy_() modifies tensor values
        in-place without changing memory addresses.
        """
        if self._active_speaker == name:
            return

        w = self._speakers[name]
        device = self._tts.tts_config.device
        dtype = self._tts.tts_config.dtype

        # Inject GPT weights
        for param_path, tensor in w.gpt_weights.items():
            target = _resolve_param(self._shared_gpt.t2s_model, param_path)
            target.data.copy_(tensor.to(device=device, dtype=dtype))

        # Inject SoVITS weights
        for param_path, tensor in w.sovits_weights.items():
            target = _resolve_param(self._shared_sovits.vq_model, param_path)
            target.data.copy_(tensor.to(device=device, dtype=dtype))

        self._active_speaker = name

    # ==================================================================
    # Inference
    # ==================================================================

    @torch.inference_mode()
    def infer(
        self,
        speaker: str,
        text: str,
        prompt_audio_path: str | None = None,
        prompt_audio_text: str | None = None,
        return_subtitles: bool = False,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
    ) -> AudioClip:
        """Generate speech for a single speaker.

        Args:
            speaker: Speaker name (must match a registered SpeakerConfig.name).
            text: Text to synthesize.
            prompt_audio_path: Override prompt audio for style reference.
            prompt_audio_text: Override prompt audio transcription.
            return_subtitles: Return word-level timestamp subtitles.
            top_k, top_p, temperature: GPT sampling parameters.
            repetition_penalty: GPT repetition penalty.
            noise_scale: SoVITS decoder noise scale.
            speed: Playback speed (1.0 = normal).

        Returns:
            AudioClip with generated audio.
        """
        with self._tts._infer_lock:
            self._apply_speaker(speaker)
            w = self._speakers[speaker]

            # Text pre-processing
            if self._tts._contains_chinese(text):
                self._tts._ensure_bert_loaded()
            if not self._tts._check_pause(text):
                text += "."

            if len(text) > 20:
                logger.info(f"[{speaker}] Starting inference: '{text[:20]}...'")
            else:
                logger.info(f"[{speaker}] Starting inference: '{text}'")

            # Prompt features (use override or cached)
            if prompt_audio_path is not None and prompt_audio_text is not None:
                self._tts.cache_prompt_audio(
                    prompt_audio_paths=prompt_audio_path,
                    prompt_audio_texts=prompt_audio_text,
                )
                pc = self._tts.prompt_audio_cache[prompt_audio_path]
                prompt, phones1, bert1 = pc["prompt"], pc["phones1"], pc["bert1"]
            elif w.prompt is not None:
                prompt, phones1, bert1 = w.prompt, w.phones1, w.bert1
            else:
                raise ValueError(
                    f"Speaker '{speaker}' has no cached prompt features. "
                    "Provide prompt_audio_path + prompt_audio_text, "
                    "or set them in SpeakerConfig."
                )

            ge = w.ge
            t2s_model = self._shared_gpt.t2s_model
            vq_model = self._shared_sovits.vq_model

            # Text → phones + BERT
            logger.info("Processing text to phones and BERT features...")
            phones2, word2ph, bert2, norm_text = get_phones_and_bert(
                text, self._tts.tts_config
            )
            all_phoneme_ids = (
                torch.LongTensor(phones1 + phones2)
                .to(self._tts.tts_config.device)
                .unsqueeze(0)
            )
            bert = torch.cat([bert1, bert2]).unsqueeze(0)

            # GPT inference
            logger.info("Running GPT inference (Text-to-Semantic)...")
            pred_semantic = t2s_model.infer(
                all_phoneme_ids, prompt, bert,
                top_k=top_k, top_p=top_p, temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

            # SoVITS inference
            logger.info("Running SoVITS inference (Semantic-to-Waveform)...")
            phones2_tensor = (
                torch.LongTensor(phones2)
                .to(self._tts.tts_config.device)
                .unsqueeze(0)
            )
            audio, attn = vq_model.decode(
                pred_semantic, phones2_tensor, ge,
                noise_scale=noise_scale, speed=speed,
            )

            # Post-processing (reuse TTS helpers)
            audio = audio[0, 0, :]
            assign = self._tts._viterbi_monotonic(attn)

            if return_subtitles:
                subtitles = self._tts._get_subtitles(word2ph, assign, speed)
                if not self._tts._check_pause(subtitles[-1]["text"]):
                    subtitles.append({
                        "text": word2ph["word"][-1],
                        "start_s": subtitles[-1]["end_s"],
                        "end_s": subtitles[-1]["end_s"],
                    })
                subtitles[-1]["end_s"] += 0.2
                subtitles = sub2text_index(subtitles, norm_text, text)
            else:
                subtitles = []

            head_offset = self._tts._find_head_threshold_offsets(audio)
            audio = audio[head_offset:]
            if subtitles:
                self._tts._increment_subtitle_times(
                    subtitles, -head_offset / self.samplerate
                )
                subtitles[0]["start_s"] = max(0, subtitles[0]["start_s"])

            audio = audio.float().cpu().numpy()
            max_audio = np.abs(audio).max()
            if max_audio > 1:
                audio = audio / max_audio
            audio = np.concatenate([
                audio,
                np.zeros((int(0.2 * self.samplerate),), dtype=audio.dtype),
            ])

            audio_len_s = len(audio) / self.samplerate
            logger.info(f"[{speaker}] Complete. Generated {audio_len_s:.2f}s audio.")

            return AudioClip(
                self.audio_queue, audio, self.samplerate, audio_len_s, subtitles, text
            )

    def infer_batched(
        self,
        speaker_texts: list[tuple[str, str]],
        prompt_audio_paths: str | list[str] | None = None,
        prompt_audio_texts: str | list[str] | None = None,
        return_subtitles: bool = False,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        bert_batch_size: int = 20,
        sovits_batch_size: int = 10,
    ) -> list[AudioClip]:
        """Batch inference — all items use the current active speaker's weights.

        Note: Currently all items in one batch must use the SAME speaker.
        Multi-speaker batches (mixing speakers) are not yet supported
        due to the sequential nature of weight injection.

        Args:
            speaker_texts: List of (speaker_name, text) tuples.
            ... (other args same as infer())

        Returns:
            List of AudioClip results.
        """
        # Validate all texts use the same speaker
        speakers_seen = set(s for s, _ in speaker_texts)
        if len(speakers_seen) > 1:
            logger.warning(
                "infer_batched received multiple speakers. "
                "Processing sequentially grouped by speaker."
            )

        results = []
        for speaker, text in speaker_texts:
            result = self.infer(
                speaker=speaker,
                text=text,
                prompt_audio_path=(
                    prompt_audio_paths if isinstance(prompt_audio_paths, str)
                    else prompt_audio_paths[speaker_texts.index((speaker, text))]
                    if isinstance(prompt_audio_paths, list) else None
                ),
                prompt_audio_text=(
                    prompt_audio_texts if isinstance(prompt_audio_texts, str)
                    else prompt_audio_texts[speaker_texts.index((speaker, text))]
                    if isinstance(prompt_audio_texts, list) else None
                ),
                return_subtitles=return_subtitles,
                top_k=top_k, top_p=top_p, temperature=temperature,
                repetition_penalty=repetition_penalty,
                noise_scale=noise_scale, speed=speed,
            )
            results.append(result)

        return results

    def infer_stream(
        self,
        speaker: str,
        text: str,
        prompt_audio_path: str | None = None,
        prompt_audio_text: str | None = None,
        stream_chunk: int = 25,
        overlap_len: int = 5,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
        noise_scale: float = 0.5,
        speed: float = 1.0,
        return_subtitles: bool = False,
        debug: bool = False,
    ) -> Generator[AudioClip, None, None]:
        """Streaming inference — yields audio chunks as they are generated.

        Currently falls back to non-streaming infer() per text segment.
        Full streaming with per-chunk weight injection will be added later.

        Args:
            speaker: Speaker name.
            text: Text to synthesize.
            stream_chunk: Token count per streaming chunk.
            overlap_len: Overlap tokens between chunks.
            ... (other args same as infer())
        """
        # For now, decompose text into segments and infer each segment
        from .TextProcessor import cut_text
        segments = cut_text(text, self._tts)
        for seg_text in segments:
            yield self.infer(
                speaker=speaker,
                text=seg_text,
                prompt_audio_path=prompt_audio_path,
                prompt_audio_text=prompt_audio_text,
                return_subtitles=return_subtitles,
                top_k=top_k, top_p=top_p, temperature=temperature,
                repetition_penalty=repetition_penalty,
                noise_scale=noise_scale, speed=speed,
            )

    def _empty_cache(self):
        """Release unused GPU memory."""
        self._tts._empty_cache()
