"""LLM-as-judge for WebVoyager / Mind2Web-executable tasks.

Reproduces the PMI WebVoyager judge prompt verbatim (planner-matter-
inference/benchmarks/webvoyager_evaluation/evaluator.py) so scores are
directly comparable to that repo's published numbers.

Inputs:
  • intent           — task instruction
  • agent_answer     — text emitted via finished(answer="...") at DONE
  • screenshots_b64  — base64-encoded screenshots (we use last 5)

Output:
  (score: float in {0.0, 1.0}, raw_response: str)

Model: any chat-completions endpoint; rollouts use qwen35-27b via
OpenRouter. Pass an ``openai.OpenAI``-style client whose
``.chat.completions.create(...)`` accepts vision content.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable, List, Optional, Tuple

from mm_agents import model_endpoints as ME

logger = logging.getLogger(__name__)


# ── PMI judge prompt — verbatim from
# benchmarks/webvoyager_evaluation/evaluator.py:80-130 so we can compare
# scores against their published numbers directly. ─────────────────────


_JUDGE_SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the last 5 screens showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'.
You should provide the 'SUCCESS' or 'NOT SUCCESS' between <result> and </result>."""


_USER_TEMPLATE = "TASK: {task}\n        Result Response: {answer}"


# ── Answer extraction ─────────────────────────────────────────────────


_FINISHED_RE = re.compile(
    r"finished\s*\(\s*answer\s*=\s*[\"']?(?P<a>.*?)[\"']?\s*\)",
    re.IGNORECASE | re.DOTALL,
)


def extract_web_answer(text: Optional[str]) -> str:
    """Pull the ``finished(answer="...")`` payload out of an agent reply.

    Tolerates single/double quotes and trailing whitespace. Returns ""
    when no match — caller can decide whether to score against the
    screenshots alone or skip.
    """
    if not text:
        return ""
    m = _FINISHED_RE.search(text)
    if not m:
        return ""
    return (m.group("a") or "").strip().strip("'").strip('"')


# ── Judge call ────────────────────────────────────────────────────────


def judge_webvoyager(
    intent: str,
    agent_answer: str,
    screenshots_b64: Iterable[str],
    client: Any,
    *,
    model: str = "Qwen/Qwen3.5-9B",
    temperature: float = 0.8,
    max_tokens: int = 1024,
    n_screenshots: int = 5,
    retries: int = 2,
) -> Tuple[float, str]:
    """Run the LLM judge. Returns (score, raw_response).

    score: 1.0 on SUCCESS, 0.0 on NOT SUCCESS (or any judge failure).

    Defaults to local Qwen3.5-9B (vision-capable, served from the same
    vllm endpoint as the actor / decomposer) for stability. OpenRouter
    qwen3.5-27b also works but has been seen to return empty content
    on long screenshots-heavy prompts; pass a different client +
    model="qwen/qwen3.5-27b" to opt in.
    """
    if not intent:
        logger.warning("[WebJudge] empty intent; scoring 0")
        return 0.0, "<no intent>"

    ss_list: List[str] = [s for s in screenshots_b64 if s][-n_screenshots:]

    user_text = _USER_TEMPLATE.format(task=intent, answer=agent_answer or "")
    user_content: List[dict] = [{"type": "text", "text": user_text}]
    for ss in ss_list:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{ss}"},
        })

    # FORCE thinking OFF for Qwen3.5 family — otherwise the model
    # spends the entire max_tokens budget in <think> mode and never
    # emits the <result>SUCCESS</result> verdict.
    base_url = str(getattr(client, "base_url", "") or "")
    if "openrouter.ai" in base_url:
        extra_body = {"reasoning": {"enabled": False},
                      "provider": {"ignore": ["Cloudflare"]}}
    else:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    raw = ""
    for attempt in range(max(1, retries + 1)):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
        except Exception as e:
            logger.error("[WebJudge] client error attempt=%d/%d: %s",
                         attempt + 1, retries + 1, e)
            if attempt == retries:
                return 0.0, f"<error: {e}>"
            continue
        if not getattr(resp, "choices", None):
            logger.warning("[WebJudge] response has no choices attempt=%d "
                           "(raw resp=%r)", attempt + 1, resp)
            if attempt == retries:
                return 0.0, "<no choices>"
            continue
        raw = (resp.choices[0].message.content or "").strip()
        if not raw:
            # OpenRouter occasionally 200's with empty content under
            # load; treat as transient and retry.
            logger.warning(
                "[WebJudge] empty content attempt=%d/%d "
                "(model=%s, n_screenshots=%d, finish_reason=%s)",
                attempt + 1, retries + 1, model, len(ss_list),
                getattr(resp.choices[0], "finish_reason", "?"))
            if attempt == retries:
                return 0.0, "<empty content from judge>"
            continue
        break

    verdict = raw.lower()
    m = re.search(r"<result>(.*?)</result>", verdict, re.DOTALL)
    if m:
        verdict = (m.group(1) or "").strip()
    if "not success" in verdict or "notsuccess" in verdict:
        return 0.0, raw
    if "success" in verdict:
        return 1.0, raw
    logger.warning("[WebJudge] no SUCCESS/NOT SUCCESS verdict; raw: %s",
                   raw[:300])
    return 0.0, raw


# ── Client factories ──────────────────────────────────────────────────


_DEFAULT_LOCAL_VLLM_URL = ME.server_url("qwen35-vl")  # single source of truth


def make_local_vllm_client(base_url: Optional[str] = None) -> Any:
    """Local vllm (Qwen3.5-9B / qwen35-vl). The default vllm endpoint
    served by the actor stack — reusing it for the judge means no extra
    network hop, no OpenRouter quota issues, no empty-content surprises.
    """
    import openai
    return openai.OpenAI(
        base_url=base_url or _DEFAULT_LOCAL_VLLM_URL,
        api_key="EMPTY")


def make_openrouter_client(api_key: Optional[str] = None) -> Any:
    """Fallback OpenRouter client for the qwen/qwen3.5-27b judge model.
    Only consulted when ``judge_model`` clearly points at a remote model.
    """
    import os
    import openai
    key = (api_key
           or os.environ.get("OPENROUTER_API_KEY")
           or os.environ.get("OPEN_ROUTER_API_KEY"))
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set; cannot reach remote judge")
    return openai.OpenAI(base_url="https://openrouter.ai/api/v1",
                          api_key=key)


def make_judge_client(model: str) -> Tuple[Any, str]:
    """Resolve (client, real_model_name) for a judge ``model`` string.

    Routing rules (resolved via the model_endpoints registry so any
    served local alias works — e.g. a stronger ``vllm_qwen35-27b`` judge,
    which the Online-Mind2Web 3-stage grader needs; the 9B is too weak for
    its per-screenshot relevance scoring):
      • ``provider/model`` (slash) or ``anthropic/`` / ``openai/`` /
        ``google/`` / ``qwen/`` prefix → OpenRouter.
      • A registry alias (optionally ``vllm_``-prefixed) that maps to a
        local endpoint → local vllm at its URL + served model name.
      • Anything else → local 9B (``Qwen/Qwen3.5-9B``).

    Returns (openai.OpenAI client, fully-qualified model name suitable
    for the resolved endpoint).
    """
    from mm_agents import model_endpoints as ME
    m = (model or "").strip()
    lower = m.lower()
    # Explicit remote provider routing.
    has_provider = "/" in m or lower.startswith(
        ("anthropic/", "openai/", "google/", "qwen/"))
    if has_provider:
        return make_openrouter_client(), m
    # Local vllm alias via the registry (strip an optional vllm_ prefix).
    key = m[len("vllm_"):] if lower.startswith("vllm_") else m
    try:
        if ME.is_known(key) and not ME.is_openrouter(key):
            return (make_local_vllm_client(ME.server_url(key)),
                    ME.model_name(key))
    except Exception:
        pass
    # Unknown — default to local 9B.
    return make_local_vllm_client(), "Qwen/Qwen3.5-9B"
