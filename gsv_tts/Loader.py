import os
import json
import torch
import hashlib
from io import BytesIO
from safetensors.torch import load_model

from .Config import Config
from .GPT_SoVITS.SoVITS.models import SynthesizerTrn
from .GPT_SoVITS.GPT.t2s_model import Text2SemanticDecoder
from .GPT_SoVITS import utils

import sys
sys.modules['utils'] = utils


head2version = {
    b"01": "v2",
    b"05": "v2Pro",
    b"06": "v2ProPlus",
}
hash_pretrained_dict = {
    "dc3c97e17592963677a4a1681f30c653": "v2",  # s2G488k.pth#sovits_v1_pretrained
    "6642b37f3dbb1f76882b69937c95a5f3": "v2",  # s2G2333K.pth#sovits_v2_pretrained
    "c7e9fce2223f3db685cdfa1e6368728a": "v2Pro",  # s2Gv2Pro.pth#sovits_v2Pro_pretrained
    "66b313e39455b57ab1b0bc0b239c9d0a": "v2ProPlus",  # s2Gv2ProPlus.pth#sovits_v2ProPlus_pretrained
}


class Sovits:
    def __init__(self, vq_model, hps):
        self.vq_model: SynthesizerTrn = vq_model
        self.hps = hps

def get_hash_from_file(sovits_path):
    with open(sovits_path, "rb") as f:
        data = f.read(8192)
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()

def load_sovits(sovits_path):
    hash = get_hash_from_file(sovits_path)

    f = open(sovits_path, "rb")
    meta = f.read(2)

    version = head2version.get(meta)
    if version is None: version = hash_pretrained_dict.get(hash)
    
    if meta != b"PK":
        data = b"PK" + f.read()
        bio = BytesIO()
        bio.write(data)
        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False), version
    return torch.load(sovits_path, map_location="cpu", weights_only=False), version

def get_sovits_weights(sovits_path, tts_config: Config):
    if os.path.isdir(sovits_path):
        with open(os.path.join(sovits_path, "hps.json"), "r") as f:
            hps = json.load(f)
        hps = utils.DictToAttrRecursive(hps)

        with torch.device("meta"):
            vq_model = SynthesizerTrn(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **vars(hps.model),
            )
        
        vq_model.dec.remove_weight_norm()
        vq_model = vq_model.to_empty(device=tts_config.device)
        vq_model = vq_model.to(tts_config.dtype)
        load_model(vq_model, os.path.join(sovits_path, "model.safetensors"))
    else:
        dict_s2, version = load_sovits(sovits_path)
        
        hps = utils.DictToAttrRecursive(dict_s2["config"])
        hps.model.semantic_frame_rate = "25hz"
        if version is None:
            assert getattr(hps.model, 'version', None) in ["v2", "v2Pro", "v2ProPlus"], "The Sovits model is not the v2/v2pro/v2proplus version. Please check the model file."
        else:
            hps.model.version = version
        
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **vars(hps.model),
        )

        vq_model.load_state_dict(dict_s2["weight"], strict=False)
        vq_model.dec.remove_weight_norm()
        vq_model.to(tts_config.device, tts_config.dtype)

    vq_model.eval()
    vq_model.initialize_runtime(tts_config.dtype, tts_config.device, tts_config.sovits_cache)

    sovits = Sovits(vq_model, hps)

    return sovits


class Gpt:
    def __init__(self, t2s_model, config):
        self.t2s_model: Text2SemanticDecoder = t2s_model
        self.config = config

def get_gpt_weights(gpt_path, tts_config: Config):
    if os.path.isdir(gpt_path):
        with open(os.path.join(gpt_path, "config.json"), "r") as f:
            config = json.load(f)

        with torch.device("meta"):
            if tts_config.use_flash_attn:
                from .GPT_SoVITS.GPT.t2s_model_flash_attn import Text2SemanticDecoder as Text2SemanticDecoder_flash_attn
                t2s_model = Text2SemanticDecoder_flash_attn(config)
            else:
                t2s_model = Text2SemanticDecoder(config)
        
        t2s_model = t2s_model.to_empty(device=tts_config.device)
        t2s_model = t2s_model.to(tts_config.dtype)
        load_model(t2s_model, os.path.join(gpt_path, "model.safetensors"))
    else:
        dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
        config = dict_s1["config"]
        
        w_key_map = [
            ['self_attn.in_proj_weight', 'qkv.weight'],
            ['self_attn.in_proj_bias', 'qkv.bias'],
            ['self_attn.out_proj.weight', 'out_proj.weight'],
            ['self_attn.out_proj.bias', 'out_proj.bias'],
            ['linear1.weight', 'mlp.0.weight'],
            ['linear1.bias', 'mlp.0.bias'],
            ['linear2.weight', 'mlp.2.weight'],
            ['linear2.bias', 'mlp.2.bias'],
            ['norm1.weight', 'norm1.weight'],
            ['norm1.bias', 'norm1.bias'],
            ['norm2.weight', 'norm2.weight'],
            ['norm2.bias', 'norm2.bias']
        ]

        for i in range(config["model"]["n_layer"]):
            original_l_key = f'model.h.layers.{i}.'
            new_l_key = f't2s_transformer.blocks.{i}.'
            for original_w_key, new_w_key in w_key_map:
                dict_s1["weight"][new_l_key+new_w_key] = dict_s1["weight"].pop(original_l_key+original_w_key)
        
        dict_s1["weight"] = {
            k.replace("model.", "", 1) if k.startswith("model.") else k: v 
            for k, v in dict_s1["weight"].items()
        }

        if tts_config.use_flash_attn:
            from .GPT_SoVITS.GPT.t2s_model_flash_attn import Text2SemanticDecoder as Text2SemanticDecoder_flash_attn
            t2s_model = Text2SemanticDecoder_flash_attn(config)
        else:
            t2s_model = Text2SemanticDecoder(config)
        
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.to(tts_config.device, tts_config.dtype)

    t2s_model.eval()
    t2s_model.initialize_runtime(tts_config.dtype, tts_config.device, tts_config.gpt_cache)

    gpt = Gpt(t2s_model, config)

    return gpt


# ============================================================
# Multi-Speaker Support — Speaker weight extraction & shared model loading
# ============================================================

# GPT state dict keys that are speaker-specific (extracted per speaker)
_GPT_SPEAKER_PREFIXES = [
    "ar_predict_layer.",
    "t2s_transformer.blocks.",  # filtered by block index at runtime
]

# SoVITS state dict keys that are speaker-specific
_SOVITS_SPEAKER_PREFIXES = [
    "ref_enc.",
    "sv_emb.",
    "ge_to512.",
    "prelu.",
    "dec.cond.",
]


def _is_sovits_speaker_weight(key: str) -> bool:
    """Return whether a SoVITS state key belongs to per-speaker conditioning."""
    if any(key.startswith(prefix) for prefix in _SOVITS_SPEAKER_PREFIXES):
        return True
    return key.startswith("flow.") and ".enc.cond_layer." in key


def _load_gpt_state_dict(gpt_path: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load GPT checkpoint state_dict and config, handling both .ckpt and safetensors.

    Returns (state_dict, config) where state_dict keys use the model-internal naming
    (e.g. 'ar_predict_layer.weight', 't2s_transformer.blocks.0.qkv.weight').
    """
    if os.path.isdir(gpt_path):
        # Safetensors: read config + model.safetensors
        with open(os.path.join(gpt_path, "config.json"), "r") as f:
            config = json.load(f)
        # Build a temporary model on meta device to get state_dict keys, then load
        with torch.device("meta"):
            t2s_model = Text2SemanticDecoder(config)
        state_dict = {}
        load_model(t2s_model, os.path.join(gpt_path, "model.safetensors"))
        for name, param in t2s_model.named_parameters():
            state_dict[name] = param.data.clone().cpu()
        del t2s_model
    else:
        dict_s1 = torch.load(gpt_path, map_location="cpu", weights_only=False)
        config = dict_s1["config"]
        w_key_map = [
            ['self_attn.in_proj_weight', 'qkv.weight'],
            ['self_attn.in_proj_bias', 'qkv.bias'],
            ['self_attn.out_proj.weight', 'out_proj.weight'],
            ['self_attn.out_proj.bias', 'out_proj.bias'],
            ['linear1.weight', 'mlp.0.weight'],
            ['linear1.bias', 'mlp.0.bias'],
            ['linear2.weight', 'mlp.2.weight'],
            ['linear2.bias', 'mlp.2.bias'],
            ['norm1.weight', 'norm1.weight'],
            ['norm1.bias', 'norm1.bias'],
            ['norm2.weight', 'norm2.weight'],
            ['norm2.bias', 'norm2.bias']
        ]
        for i in range(config["model"]["n_layer"]):
            original_l_key = f'model.h.layers.{i}.'
            new_l_key = f't2s_transformer.blocks.{i}.'
            for original_w_key, new_w_key in w_key_map:
                dict_s1["weight"][new_l_key + new_w_key] = dict_s1["weight"].pop(
                    original_l_key + original_w_key
                )
        dict_s1["weight"] = {
            k.replace("model.", "", 1) if k.startswith("model.") else k: v
            for k, v in dict_s1["weight"].items()
        }
        state_dict = dict_s1["weight"]

    return state_dict, config


def _load_sovits_state_dict(sovits_path: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load SoVITS checkpoint state_dict and hps, handling both .pth and safetensors.

    Returns (state_dict, hps) where state_dict keys match SynthesizerTrn's named_parameters().
    """
    if os.path.isdir(sovits_path):
        with open(os.path.join(sovits_path, "hps.json"), "r") as f:
            hps = json.load(f)
        # For safetensors, we can read weights directly without building a model
        from safetensors import safe_open
        state_dict = {}
        with safe_open(os.path.join(sovits_path, "model.safetensors"), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                state_dict[key] = sf.get_tensor(key)
    else:
        dict_s2, version = load_sovits(sovits_path)
        hps = dict_s2["config"]
        if version is not None:
            hps["model"]["version"] = version
        state_dict = dict_s2["weight"]

    return state_dict, hps


def extract_speaker_gpt_weights(
    gpt_path: str,
    tts_config: Config,
    shared_layers: int | None = None,
) -> dict[str, torch.Tensor]:
    """Extract speaker-specific GPT weights from a full checkpoint.

    Extracts:
      - ar_predict_layer.* (prediction head — always speaker-specific)
      - t2s_transformer.blocks[N-shared_count : N].* (last few transformer blocks)

    Args:
        gpt_path: Path to the GPT checkpoint (.ckpt file or safetensors directory).
        tts_config: TTS Config instance (unused currently, for future compatibility).
        shared_layers: Number of transformer layers to keep shared.
                       None = auto-detect: n_layer - 2 (keep last 2 layers per speaker).

    Returns:
        Dict mapping parameter paths to CPU tensors.
        Example: {"ar_predict_layer.weight": tensor, "t2s_transformer.blocks.23.qkv.weight": tensor}
    """
    state_dict, config = _load_gpt_state_dict(gpt_path)
    n_layer = config["model"]["n_layer"]

    if shared_layers is None:
        shared_layers = max(0, n_layer - 2)

    speaker_weights = {}

    for key, tensor in state_dict.items():
        # Always extract prediction head
        if key.startswith("ar_predict_layer."):
            speaker_weights[key] = tensor.cpu()

        # Extract transformer blocks beyond shared_layers
        if key.startswith("t2s_transformer.blocks."):
            # Parse block index from key: "t2s_transformer.blocks.{i}.*"
            rest = key[len("t2s_transformer.blocks."):]
            block_idx_str = rest.split(".", 1)[0]
            try:
                block_idx = int(block_idx_str)
            except ValueError:
                continue
            if block_idx >= shared_layers:
                speaker_weights[key] = tensor.cpu()

    return speaker_weights


def extract_speaker_sovits_weights(
    sovits_path: str,
    tts_config: Config,
) -> dict[str, torch.Tensor]:
    """Extract speaker-specific SoVITS weights from a full checkpoint.

    Extracts (by key prefix match):
      - ref_enc.*            (speaker embedding extractor)
      - sv_emb.*             (SV embedding projection, v2Pro+ only)
      - ge_to512.*           (ge dimension adapter, v2Pro+ only)
      - prelu.*              (post-fusion activation, v2Pro+ only)
      - flow.*.enc.cond_layer.* (WN conditioning projections)
      - dec.cond.*           (vocoder conditioning projection)

    Args:
        sovits_path: Path to the SoVITS checkpoint (.pth file or safetensors directory).
        tts_config: TTS Config instance (unused currently).

    Returns:
        Dict mapping parameter paths to CPU tensors.
    """
    state_dict, hps = _load_sovits_state_dict(sovits_path)

    speaker_weights = {}

    for key, tensor in state_dict.items():
        if _is_sovits_speaker_weight(key):
            speaker_weights[key] = tensor.cpu()

    return speaker_weights


def load_shared_gpt(
    gpt_path: str,
    tts_config: Config,
    shared_layers: int | None = None,
) -> Gpt:
    """Load GPT model as a shared backbone — speaker-specific layers are randomly initialized.

    This loads the full model but zeroes out weights for layers that will be
    per-speaker (predict_layer + last few transformer blocks). At inference time,
    these are overwritten via copy_() with speaker-specific weights from
    extract_speaker_gpt_weights().

    Args:
        gpt_path: Path to the base GPT checkpoint.
        tts_config: TTS Config instance.
        shared_layers: Number of transformer layers to keep as shared.
                       None = n_layer - 2.

    Returns:
        Gpt wrapper with valid shared weights and randomized speaker-specific weights.
    """
    state_dict, config = _load_gpt_state_dict(gpt_path)
    n_layer = config["model"]["n_layer"]

    if shared_layers is None:
        shared_layers = max(0, n_layer - 2)

    # Build model
    if tts_config.use_flash_attn:
        from .GPT_SoVITS.GPT.t2s_model_flash_attn import Text2SemanticDecoder as T2SD
        t2s_model = T2SD(config)
    else:
        t2s_model = Text2SemanticDecoder(config)

    # Randomize speaker-specific layers (they'll be filled at inference time)
    randomized_keys = set()
    for key in state_dict:
        if key.startswith("ar_predict_layer."):
            randomized_keys.add(key)
        if key.startswith("t2s_transformer.blocks."):
            rest = key[len("t2s_transformer.blocks."):]
            block_idx_str = rest.split(".", 1)[0]
            try:
                block_idx = int(block_idx_str)
            except ValueError:
                continue
            if block_idx >= shared_layers:
                randomized_keys.add(key)

    for key in randomized_keys:
        # Replace with random init matching the expected shape
        if key in state_dict:
            state_dict[key] = torch.empty_like(state_dict[key]).normal_()

    t2s_model.load_state_dict(state_dict)
    t2s_model = t2s_model.to(tts_config.device, tts_config.dtype)
    t2s_model.eval()
    t2s_model.initialize_runtime(tts_config.dtype, tts_config.device, tts_config.gpt_cache)

    return Gpt(t2s_model, config)


def load_shared_sovits(
    sovits_path: str,
    tts_config: Config,
) -> Sovits:
    """Load SoVITS model as a shared backbone — speaker-specific layers are randomly initialized.

    Speaker-specific layers (ref_enc, sv_emb, ge_to512, prelu, flow cond,
    decoder cond) are set to random.
    These will be overwritten via copy_() at inference time with speaker-specific
    weights from extract_speaker_sovits_weights().

    Args:
        sovits_path: Path to the base SoVITS checkpoint.
        tts_config: TTS Config instance.

    Returns:
        Sovits wrapper with valid shared weights and randomized speaker-specific weights.
    """
    state_dict, hps_raw = _load_sovits_state_dict(sovits_path)
    hps = utils.DictToAttrRecursive(hps_raw)
    hps.model.semantic_frame_rate = "25hz"

    # Detect version if not already set
    if not hasattr(hps.model, 'version') or hps.model.version is None:
        version = hash_pretrained_dict.get(get_hash_from_file(sovits_path) if not os.path.isdir(sovits_path) else None)
        hps.model.version = version or "v2"

    # Randomize speaker-specific layers
    for key in list(state_dict.keys()):
        if _is_sovits_speaker_weight(key):
            state_dict[key] = torch.empty_like(state_dict[key]).normal_()

    # Build model and load (partially randomized) state_dict
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **vars(hps.model),
    )
    vq_model.load_state_dict(state_dict, strict=False)
    vq_model.dec.remove_weight_norm()
    vq_model.to(tts_config.device, tts_config.dtype)
    vq_model.eval()
    vq_model.initialize_runtime(tts_config.dtype, tts_config.device, tts_config.sovits_cache)

    return Sovits(vq_model, hps)
