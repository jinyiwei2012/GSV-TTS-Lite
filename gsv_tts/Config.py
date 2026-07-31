"""GPU auto-detection and configuration for GSV-TTS-Lite.

Device/dtype selection is cached on first access so that ``import Config`` does
not immediately initialise CUDA (useful when the module is imported for
documentation or introspection).
"""

import os
import logging
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# -----------------------------------------------------------
# Named constants for SM version thresholds
# -----------------------------------------------------------
SM_AMPERE_MIN = 8.0    # Ampere+ (A100, A6000, RTX 30xx): bfloat16
SM_TURING_MIN = 7.0    # Turing/Volta (T4, V100, RTX 20xx): float16
SM_PASCAL     = 6.1    # Pascal (P100, GTX 10xx): float32
SM_MIN_SUPPORTED = 5.3  # Minimum CUDA capability we support
FULLWIDTH_OFFSET = 0xFEE0


def get_cuda_device_info(idx: int, quiet: bool = False):
    """Get CUDA device info for a GPU at *idx*."""
    if not torch.cuda.is_available() or idx >= torch.cuda.device_count():
        return None

    try:
        props = torch.cuda.get_device_properties(idx)
    except Exception:
        return None

    name = props.name
    major, minor = props.major, props.minor
    sm_version = major + minor / 10.0
    mem_gb = props.total_memory / (1024 ** 3)

    if sm_version < SM_MIN_SUPPORTED:
        if not quiet:
            logging.info(
                "GPU %d (%s) SM %.1f below minimum %.1f, skipping",
                idx, name, sm_version, SM_MIN_SUPPORTED,
            )
        return None

    device = torch.device(f"cuda:{idx}")

    # GTX 16 series (Turing SM 7.5 but lacking tensor-core float16 perf)
    is_16_series = (major == 7 and minor == 5) and ("16" in name)

    if sm_version == SM_PASCAL or is_16_series:
        return device, torch.float32, sm_version, mem_gb

    if sm_version >= SM_AMPERE_MIN:
        return device, torch.bfloat16, sm_version, mem_gb

    if sm_version >= SM_TURING_MIN:
        return device, torch.float16, sm_version, mem_gb

    return device, torch.float32, sm_version, mem_gb


def get_mps_device_info():
    """Get Apple Silicon MPS device info."""
    if not torch.backends.mps.is_available():
        return None
    try:
        device = torch.device("mps")
        return device, torch.float32, 0.0, 0.0
    except Exception:
        return None


def choose_attention_backend(batch=1, heads=8, seq=128, head_dim=64, dtype=torch.float16):
    """Probe the fastest available SDPA backend on the current device."""
    if not torch.cuda.is_available():
        logging.info("SDPBackend: MATH")
        return SDPBackend.MATH

    k = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=dtype)

    probes = []
    for q_len in (seq, 1):
        q = torch.randn(batch, heads, q_len, head_dim, device="cuda", dtype=dtype)
        mask = torch.zeros(batch, heads, q_len, seq, device="cuda", dtype=torch.bool)
        probes.append((q, mask))

    def is_usable(backend):
        try:
            for q, mask in probes:
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return True
        except Exception:
            return False

    if is_usable(SDPBackend.CUDNN_ATTENTION):
        logging.info("SDPBackend: CUDNN_ATTENTION")
        return SDPBackend.CUDNN_ATTENTION
    elif is_usable(SDPBackend.EFFICIENT_ATTENTION):
        logging.info("SDPBackend: EFFICIENT_ATTENTION")
        return SDPBackend.EFFICIENT_ATTENTION
    else:
        logging.info("SDPBackend: MATH")
        return SDPBackend.MATH


# -----------------------------------------------------------
# Cached lazy device / dtype detection
# -----------------------------------------------------------
_DEVICE_CACHE: torch.device | None = None
_DTYPE_CACHE: torch.dtype | None = None


def _detect_device() -> torch.device:
    """Run full device detection (called once, cache on first access)."""
    global _DTYPE_CACHE
    # CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        available_devices = []
        for i in range(gpu_count):
            info = get_cuda_device_info(i)
            if info is not None:
                available_devices.append(info)
        if available_devices:
            # Prefer highest SM version, then most memory
            best = max(available_devices, key=lambda x: (x[2], x[3]))
            _DTYPE_CACHE = best[1]
            return best[0]

    # MPS (Apple Silicon)
    mps = get_mps_device_info()
    if mps is not None:
        _DTYPE_CACHE = torch.float32
        return mps[0]

    # CPU fallback
    _DTYPE_CACHE = torch.float32
    return torch.device("cpu")


def get_device() -> torch.device:
    """Return best available device (cached on first call)."""
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        _DEVICE_CACHE = _detect_device()
    return _DEVICE_CACHE


def get_dtype() -> torch.dtype:
    """Return best dtype for the detected device (cached on first call)."""
    global _DTYPE_CACHE
    if _DTYPE_CACHE is None:
        _detect_device()
    return _DTYPE_CACHE


# Backward-compatible module-level aliases — lazily populated so that
# ``import Config`` does not immediately probe CUDA.
_device: torch.device | None = None
_dtype: torch.dtype | None = None


def _lazy_init():
    global _device, _dtype
    if _device is None:
        _device = get_device()
    if _dtype is None:
        _dtype = get_dtype()
    return _device, _dtype


# -----------------------------------------------------------
# Config classes
# -----------------------------------------------------------

class Config:
    def __init__(self):
        dev, dt = _lazy_init()
        self.dtype = dt
        self.device = dev


class GlobalConfig:
    """Holds global path and G2P-module state (separate from device config)."""

    def __init__(self):
        self.models_dir = None
        self.use_jieba_fast = None
        self.chinese_g2p = None
        self.japanese_g2p = None
        self.english_g2p = None


global_config = GlobalConfig()

# Probe the best SDPA backend once at import time (kept for API compatibility).
SDPBACKEND = choose_attention_backend()
