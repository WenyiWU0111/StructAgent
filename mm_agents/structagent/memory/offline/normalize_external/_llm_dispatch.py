"""LLM dispatch helpers for the offline normalize scripts.

Reconstructed 2026-06-16: these generic helpers previously lived in the (now
removed) ``memory/runtime/recheck/_common.py``. That recheck package hosted the
dead L_v2/L_v3 milestone/done-gate exemplar machinery and was deleted, but it also
held this model resolver, which the external-dataset normalize scripts still use.

Local Qwen via vLLM with thinking OFF (the OpenRouter wire-format does not take the
chat_template_kwargs flag, so only the vLLM path sets it — otherwise Qwen keeps
thinking ON and blows past max_tokens before reaching the JSON answer). sonnet/opus
or a fully-qualified name route to OpenRouter.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

from mm_agents.structagent._paths import REPO_ROOT
from mm_agents import model_endpoints as ME

_NO_THINK: Dict = {"chat_template_kwargs": {"enable_thinking": False}}


def load_dotenv() -> None:
    """Best-effort: populate os.environ (esp. OPENROUTER_API_KEY) from a .env."""
    for p in (Path.cwd() / ".env", REPO_ROOT / ".env"):
        try:
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass


def resolve_reviewer_model(choice: str) -> Tuple[str, str, Dict]:
    """Map a model choice to ``(model_name, base_url, extra_body)``.

    choice ∈ {"qwen" (=9B), "qwen35-27b", "sonnet", "opus"} or a fully-qualified
    OpenRouter model name. Local Qwen → vLLM (thinking off); others → OpenRouter.
    """
    c = (choice or os.environ.get("VERIFIER_RECHECK_MODEL", "qwen")).strip()
    if c in ("qwen", "qwen35-9b", "qwen3.5-9b", "qwen-9b"):
        return (*ME.resolve("qwen35-vl"), dict(_NO_THINK))
    if c in ("qwen35-27b", "qwen3.5-27b", "27b", "qwen-27b"):
        return (*ME.resolve("qwen35-27b"), dict(_NO_THINK))
    if c == "sonnet":
        return "anthropic/claude-sonnet-4-5", "https://openrouter.ai/api/v1", {}
    if c == "opus":
        return "anthropic/claude-opus-4", "https://openrouter.ai/api/v1", {}
    # fully-qualified model name → OpenRouter
    return c, "https://openrouter.ai/api/v1", {}
