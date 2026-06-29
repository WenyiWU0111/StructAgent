"""LLM client mixin: vLLM + DashScope dispatch + system-prompt composition.

Provides three methods folded into StructAgent:
  call_llm(payload, model)            — DashScope / vLLM dispatch
  _call_llm_vllm(payload)             — vLLM (or OpenRouter-via-vLLM) backend
  _compose_system_prompt(base, role)  — append runtime debug patches

All three preserve the original ``self.X`` access patterns; nothing was
re-shaped semantically.
"""

import logging
import os
import time
from typing import Any, Dict

from mm_agents import model_endpoints as ME
import backoff
import openai
from requests.exceptions import SSLError
from google.api_core.exceptions import (
    InvalidArgument,
    ResourceExhausted,
    InternalServerError,
    BadRequest,
)


MAX_RETRY_TIMES = 5

_logger = logging.getLogger("desktopenv.qwen25vl_agent_planner")


class LLMClient:
    """LLM dispatch + system-prompt composition. Folded into ``StructAgent``
    via inheritance — methods access ``self.model`` / ``self.max_tokens`` /
    etc on the composed instance.
    """

    @backoff.on_exception(
        backoff.constant,
        (
            SSLError,
            openai.RateLimitError,
            openai.BadRequestError,
            openai.InternalServerError,
            InvalidArgument,
            ResourceExhausted,
            InternalServerError,
            BadRequest,
        ),
        interval=30,
        max_tries=5,
    )
    def call_llm(self, payload, model):
        messages = payload["messages"]
        if model.startswith("vllm"):
            return self._call_llm_vllm(payload)

        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"Model '{model}' routes to the DashScope API but DASHSCOPE_API_KEY "
                "is not set. Export it (see .env.example) or select a vLLM/OpenRouter "
                "model instead."
            )

        client = openai.OpenAI(base_url=base_url, api_key=api_key)

        for _ in range(MAX_RETRY_TIMES):
            _logger.info("Generating content with Qwen model: %s", model)
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                return response.choices[0].message.content
            except Exception as e:
                _logger.error("Error calling Qwen model: %s", e)
                time.sleep(5)
                continue
        return ""

    def _call_llm_vllm(self, payload):
        """Call LLM via vLLM backend (same logic as agent.py / qwen3vl_agent.py)."""
        model_key = payload["model"].split("_")[-1]
        model_name_, server_url = ME.resolve(model_key)
        api_key = os.environ.get("OPENROUTER_API_KEY") or "EMPTY"
        _logger.info("Generating content with vLLM model: %s (server: %s)", payload["model"], server_url)
        client = openai.OpenAI(base_url=server_url, api_key=api_key)
        # Qwen3.5-VL thinking mode toggle. Default OFF (clean structured
        # output, lower latency). Enable via env QWEN35_THINKING=1; when
        # enabled, also bump max_tokens up to QWEN35_THINKING_MAX_TOKENS
        # (default 15000) so the <think> block has headroom before the
        # actual answer is generated.
        extra_kwargs: Dict[str, Any] = {}
        max_tokens_override = payload.get("max_tokens", self.max_tokens)
        # Both qwen35-vl (local vLLM :8001) and qwen35-27b (local vLLM :8002,
        # serve_qwen35_27b.sh) share the QWEN35_THINKING env var — both Qwen 3.5
        # family. The thinking wire format differs by backend: OpenRouter expects
        # `reasoning.enabled`, vLLM expects `chat_template_kwargs.enable_thinking`.
        # We branch on server_url (not model_key), so a future OpenRouter-hosted
        # qwen35-* still gets the right format from its alias entry above.
        if model_key == "qwen3.7-plus":
            # qwen3.7-plus (OpenRouter): thinking ON by default for best quality.
            # Reasoning comes back in a SEPARATE `reasoning` field, so `content`
            # stays clean. Bump max_tokens to a high floor so the reasoning trace
            # + final answer never get truncated (response completeness). Turn off
            # with QWEN37_THINKING=0; raise the floor with QWEN37_THINKING_MAX_TOKENS.
            thinking_on = os.environ.get("QWEN37_THINKING", "1").strip().lower() in ("1", "true", "yes", "on")
            extra_kwargs["extra_body"] = {"reasoning": {"enabled": thinking_on}}
            if thinking_on:
                try:
                    floor = int(os.environ.get("QWEN37_THINKING_MAX_TOKENS", "16000"))
                except ValueError:
                    floor = 16000
                if max_tokens_override < floor:
                    max_tokens_override = floor
        elif model_key in ("qwen35-vl", "qwen35-27b"):
            # ``force_thinking`` on the payload overrides the global switch —
            # the feasibility judge (attribution/feasibility_verdict.py) needs
            # Qwen-3.5 thinking ON regardless of QWEN35_THINKING (validated:
            # thinking-off 误伤's). No-op for non-Qwen-3.5 models (this branch
            # is skipped). See memory ``project-ca-infeasible-design``.
            thinking_on = bool(payload.get("force_thinking")) or \
                os.environ.get("QWEN35_THINKING", "").strip().lower() in ("1", "true", "yes", "on")
            if "openrouter.ai" in server_url:
                extra_kwargs["extra_body"] = {"reasoning": {"enabled": thinking_on}}
            else:
                extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking_on}}
            if thinking_on:
                try:
                    floor = int(os.environ.get("QWEN35_THINKING_MAX_TOKENS", "15000"))
                except ValueError:
                    floor = 15000
                if max_tokens_override < floor:
                    max_tokens_override = floor
        # OpenRouter's Cloudflare provider silently truncates long completions on
        # Qwen coder models; route around it.
        if "openrouter.ai" in server_url:
            eb = extra_kwargs.setdefault("extra_body", {})
            # Some OpenRouter models serve VISION on only specific providers;
            # default routing lands on text-only providers that silently drop the
            # image (empty/garbled completion). Lock each to a provider verified
            # to serve its vision endpoint with accurate grounding coords.
            _VISION_PROVIDER = {
                "kimi-k2.6": "SiliconFlow",
                "qwen3.5-397b-a17b": "Alibaba",
            }
            if model_key in _VISION_PROVIDER:
                eb["provider"] = {"only": [_VISION_PROVIDER[model_key]]}
            else:
                eb["provider"] = {"ignore": ["Cloudflare"]}
        response = client.chat.completions.create(
            model=model_name_,
            messages=payload["messages"],
            temperature=payload.get("temperature", self.temperature),
            top_p=payload.get("top_p", self.top_p),
            max_tokens=max_tokens_override,
            **extra_kwargs,
        )
        return response.choices[0].message.content or ""

    def _compose_system_prompt(self, base: str, role: str = "planner") -> str:
        """Append any runtime prompt patches (set by the online debug loop)
        to the given base system prompt. Returns ``base`` unchanged when no
        patch is active.

        Args:
            base: the hard-coded system prompt for this call site
            role: which agent role this prompt is for — "planner" / "actor".
                Patches target planner/actor independently so an
                actor-level hard_override doesn't leak into planner context
                and a planner-level strategic guidance doesn't confuse the
                actor's tool calling.
        """
        osworld_block = (
            "OSWorld VM environment (constant across all tasks):\n"
            "  • Linux user: 'user' (HOME=/home/user); '~' expands to "
            "'/home/user'.\n"
            "  • Default sudo password for 'user' is 'password'. Use "
            "this when a command requires sudo.\n"
            "  • Open a terminal with ctrl+alt+t (gnome-terminal)."
        )
        base_with_env = osworld_block + "\n\n" + base
        return base_with_env
