"""Single source of truth for ``model_key -> (served name, base URL)``.

Every vLLM/OpenRouter call site used to keep its own ``model_name_map`` +
``model_server_map``. They drifted (e.g. the 9B served on :8010 was still mapped
to :8001 in some files; the 27B was mapped to :8002, which is actually qwen2-vl).
This module replaces all of those copies.

Per-host ports differ, so any URL can be overridden by an env var
``VLLM_<KEY>_URL`` — the key upper-cased with ``-`` / ``.`` turned into ``_``::

    export VLLM_QWEN35_VL_URL=http://localhost:8010/v1
    export VLLM_QWEN35_27B_URL=http://gpu-box-2:8012/v1

Edit the registry / set the env var here, never re-hardcode a port at a call
site. ``model_key`` is the suffix after the last ``_`` of the runner's model
string (e.g. ``vllm_qwen35-vl`` -> ``qwen35-vl``), matching the existing
``payload["model"].split("_")[-1]`` convention.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

_OPENROUTER = "https://openrouter.ai/api/v1"
_DEFAULT_URL = "http://localhost:8000/v1"

# model_key -> (served/HF model name, default base URL)
_REGISTRY: dict[str, Tuple[str, str]] = {
    # ---- local vLLM ----
    "qwen2.5-vl":    ("Qwen/Qwen2.5-VL-7B-Instruct",   "http://localhost:8000/v1"),
    "ui-tars":       ("ByteDance-Seed/UI-TARS-1.5-7B", "http://localhost:8001/v1"),
    "qwen2-vl":      ("Qwen/Qwen2-VL-7B-Instruct",     "http://localhost:8002/v1"),
    "websight":      ("WenyiWU0111/websight-7B_combined", "http://localhost:8003/v1"),
    "qwen2.5-vl-3b": ("Qwen/Qwen2.5-VL-3B-Instruct",   "http://localhost:8004/v1"),
    "cogagent":      ("zai-org/cogagent-9b-20241220",  "http://localhost:8005/v1"),
    "ui-ins-7b":     ("Tongyi-MiA/UI-Ins-7B",          "http://localhost:8006/v1"),
    "qwen3-vl":      ("Qwen/Qwen3-VL-8B-Instruct",     "http://localhost:8007/v1"),
    "qwen35-vl":     ("Qwen/Qwen3.5-9B",               "http://localhost:8010/v1"),
    "qwen35-27b":    ("Qwen/Qwen3.5-27B",              "http://localhost:8012/v1"),
    # ---- OpenRouter ----
    "ui-ins-32b":        ("Tongyi-MiA/UI-Ins-32B",                _OPENROUTER),
    "qwen2.5-vl-32b":    ("qwen/qwen2.5-vl-32b-instruct",         _OPENROUTER),
    "glm":               ("thudm/glm-4.1v-9b-thinking",           _OPENROUTER),
    "gemini":            ("google/gemini-2.5-pro",                _OPENROUTER),
    "gpt-4o":            ("openai/gpt-4o",                        _OPENROUTER),
    "claude-sonnet":     ("anthropic/claude-sonnet-4.6",          _OPENROUTER),
    "claude-opus":       ("anthropic/claude-opus-4.6",            _OPENROUTER),
    "qwen2.5-coder":     ("qwen/qwen2.5-coder-7b-instruct",       _OPENROUTER),
    "qwen3-coder":       ("qwen/qwen3-coder-30b-a3b-instruct",    _OPENROUTER),
    "qwen3.7-plus":      ("qwen/qwen3.7-plus",                    _OPENROUTER),
    "kimi-k2.6":         ("moonshotai/kimi-k2.6",                 _OPENROUTER),
    "qwen3.5-397b-a17b": ("qwen/qwen3.5-397b-a17b",              _OPENROUTER),
    "minimax-m3":        ("minimax/minimax-m3",                   _OPENROUTER),
}


def _env_url(model_key: str) -> Optional[str]:
    return os.environ.get("VLLM_" + re.sub(r"[-.]", "_", model_key).upper() + "_URL")


def model_name(model_key: str) -> str:
    """Served/HF model name for a key; unknown keys pass through unchanged."""
    entry = _REGISTRY.get(model_key)
    return entry[0] if entry else model_key


def server_url(model_key: str) -> str:
    """Base URL for a key, overridable via ``VLLM_<KEY>_URL``; unknown -> default."""
    env = _env_url(model_key)
    if env:
        return env
    entry = _REGISTRY.get(model_key)
    return entry[1] if entry else _DEFAULT_URL


def resolve(model_key: str) -> Tuple[str, str]:
    """``(served_name, base_url)`` for a key — the common call-site pattern."""
    return model_name(model_key), server_url(model_key)


def is_known(model_key: str) -> bool:
    """Whether the key is in the registry (vs an ad-hoc / custom endpoint that a
    caller resolves through its own fallback, e.g. a per-run decomposer URL)."""
    return model_key in _REGISTRY


def is_openrouter(model_key: str) -> bool:
    return server_url(model_key) == _OPENROUTER
