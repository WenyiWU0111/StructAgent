"""Mind2Web evaluation bridge.

Captures the agent's text answer when needed, collects key screenshots, and
scores a finished web episode with the answer-blind Online-Mind2Web grader
(``web_judge_online_m2w.WebJudge``). Also hosts the judge LLM-client factory.
WebVoyager / MMInA grading are not part of this release.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from typing import Any, List, Optional

from mm_agents import model_endpoints as ME
from web_judge_online_m2w import WebJudge

logger = logging.getLogger("desktopenv.experiment")

_FINISHED_RE = re.compile(
    r"finished\s*\(\s*answer\s*=\s*[\"\']?(?P<a>.*?)[\"\']?\s*\)",
    re.IGNORECASE | re.DOTALL,
)

_DEFAULT_LOCAL_VLLM_URL = ME.server_url("qwen35-vl")

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






# ── Sentinel extraction ───────────────────────────────────────────────


_LEGACY_ANSWER_RE = re.compile(
    r"\bANSWER\s*\(\s*(.+?)\s*\)", re.IGNORECASE | re.DOTALL)


def extract_text_answer(text: Optional[str]) -> str:
    """Accept either ``finished(answer="...")`` (new) or ``ANSWER(...)``
    (MMInA-era + wiki_search_executor). Returns "" on no match."""
    if not text:
        return ""
    ans = extract_web_answer(text)
    if ans:
        return ans
    m = _LEGACY_ANSWER_RE.search(text)
    return (m.group(1) or "").strip().strip("'").strip('"') if m else ""


def _b64_screenshot(obs) -> Optional[str]:
    ss = (obs or {}).get("screenshot")
    return base64.b64encode(ss).decode("ascii") if ss else None


def _fetch_current_url(env) -> str:
    """Pull Chrome's current focused-tab URL via the CDP HTTP endpoint.
    Best-effort — returns "" on any failure. Called from the runner before
    the force-extract step so the helper stays env-decoupled."""
    vm_ip = getattr(env, "vm_ip", None)
    if not vm_ip:
        return ""
    port = getattr(env, "chromium_port", 9222)
    try:
        import requests
        resp = requests.get(f"http://{vm_ip}:{port}/json", timeout=5)
        for tab in resp.json():
            if tab.get("type") == "page":
                return tab.get("url", "") or ""
    except Exception as e:
        logger.warning("[force-extract] CDP URL fetch failed: %s", e)
    return ""


# ── Force-extract prompts (run when agent never emitted finished(answer=)) ──


_FORCE_EXTRACT_SYSTEM = """You are an answer-extraction assistant for a GUI agent that just finished a web task. The agent never emitted the ``finished(answer="...")`` sentinel during its trajectory, so we are asking you ONCE — looking at the LAST FEW SCREENSHOTS plus the LIVE PAGE URL — to emit it now.

YOUR OUTPUT MUST BE EXACTLY ONE LINE of the form:

    finished(answer="<your answer>")

Nothing else — no Thought, no explanation, no markdown. The string inside the quotes is what gets graded.

CHOOSING WHAT GOES INSIDE THE QUOTES:
- If the evaluator looks at PAGE URL (mostly multi-hop navigation tasks
  asking the agent to "rent a car", "book a hotel", "search youtube" —
  the EVAL MODE / SUBDOMAIN hint will say URL) — emit the CURRENT URL
  given to you below VERBATIM. Do NOT try to read the URL bar from
  the screenshot pixels (often truncated / OCR-fragile); use the
  CURRENT URL field as ground truth.
- If the evaluator looks at TEXT (Wikipedia QA — SUBDOMAIN says TEXT)
  — read the screenshots and emit a concise English answer (one
  sentence, ≤ 30 words). The URL is irrelevant here.
- If you cannot find an answer in either source, emit
  ``finished(answer="UNKNOWN")``.

NEVER refuse, never explain, never describe the screen — just the one finished() line."""


_FORCE_EXTRACT_USER_PREFIX = """TASK:
{task}

EVAL MODE: {mode}{subdomain_hint}

CURRENT URL (verbatim from Chrome DevTools — use this for URL-eval tasks):
{current_url}

LAST {n} SCREENSHOT(S) of the trajectory (most recent last):"""


_SUBDOMAIN_HINT_FOR_EXTRACT = {
    "mmina_wikipedia":
        "  → expect TEXT answer (LLM fuzzy-match against a reference sentence; CURRENT URL is IRRELEVANT for this subdomain).",
    "mmina_compare":
        "  → expect URL answer (substring match against page URL; COPY the CURRENT URL above verbatim into finished(answer=\"...\")).",
    "mmina_normal":
        "  → expect URL answer (substring match against page URL; COPY the CURRENT URL above verbatim into finished(answer=\"...\")).",
    "mmina_multi567":
        "  → expect URL answer (substring match against page URL; COPY the CURRENT URL above verbatim into finished(answer=\"...\")).",
    "mmina_multipro":
        "  → expect URL answer (substring match against page URL; COPY the CURRENT URL above verbatim into finished(answer=\"...\")).",
}


def _force_extract_answer(
    *,
    call_llm,
    model_arg: str,
    instruction: str,
    last_screenshots_b64: List[str],
    current_url: str,
    eval_mode: str,
    text_answer_domain: Optional[str],
) -> str:
    """One-shot LLM call asking for ``finished(answer="...")`` based on
    (a) the trajectory's last few screenshots and (b) the live page URL
    captured by the runner (pulled via CDP). Returns "" on any failure
    (caller logs).

    ``call_llm`` is any ``(payload, model) -> str`` callable: the planner
    agent's ``.call_llm`` in the main harness, or the generic
    OpenAI-compatible caller from ``_make_client_llm_caller`` for baseline
    frameworks whose agents have no such method.
    """
    if not last_screenshots_b64:
        return ""
    sub_hint = _SUBDOMAIN_HINT_FOR_EXTRACT.get(text_answer_domain or "", "")
    user_text = _FORCE_EXTRACT_USER_PREFIX.format(
        task=instruction[:1500],
        mode=eval_mode,
        n=len(last_screenshots_b64),
        current_url=(current_url or "(unavailable)"),
        subdomain_hint=("\n" + sub_hint) if sub_hint else "",
    )
    user_content: List[dict] = [{"type": "text", "text": user_text}]
    for ss in last_screenshots_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{ss}"},
        })
    user_content.append({
        "type": "text",
        "text": ("\nEmit the single finished(answer=\"...\") line now. "
                 "Output nothing else."),
    })
    payload = {
        "model": model_arg,
        "messages": [
            {"role": "system", "content": _FORCE_EXTRACT_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
    }
    # LLMClient.call_llm signature is (payload, model) — model must be
    # passed as the second positional arg (the top-level vllm/non-vllm
    # routing branch reads it via ``model.startswith("vllm")``). Omitting
    # it silently failed every single force-extract call with
    # "missing 1 required positional argument: 'model'".
    resp = (call_llm(payload, model_arg) or "").strip()
    ans = extract_text_answer(resp)
    if ans:
        return ans
    # Fallback A: Qwen sometimes drops the ``finished(...)`` wrapper and
    # emits just ``answer="Paris"`` or ``answer=Paris``. Salvage it.
    m = re.search(r"answer\s*=\s*[\"']?(.+?)[\"']?\s*$", resp,
                  re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().strip("'").strip('"')
    # Fallback B: model emitted ONLY a short bare answer (one short line
    # with no Thought/JSON/markdown). Use the stripped response as-is.
    # Cap at 150 chars — bare answers should be a URL or a short
    # sentence; anything longer is almost certainly the model reasoning
    # out loud, which is worse than empty (it pollutes substring matching).
    if resp and "\n" not in resp and len(resp) <= 150:
        return resp.strip("'").strip('"')
    return ""


def _make_client_llm_caller(judge_model: str):
    """``(payload, model) -> str`` caller for harnesses whose agents have
    no ``.call_llm`` (the baseline frameworks: Agent-S3 / VLAA-GUI /
    OS-Symphony). Routes through the same endpoint family as the judge;
    thinking is forced OFF for the local Qwen3.5 default so reasoning
    text never pollutes the one-line ``finished(answer=...)`` output."""
    client, real_model = make_judge_client(judge_model)

    def _call(payload: dict, _model_arg: str) -> str:
        kwargs = {}
        if "qwen3.5" in real_model.lower():
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}}
        resp = client.chat.completions.create(
            model=real_model,
            messages=payload["messages"],
            temperature=payload.get("temperature", 0.0),
            max_tokens=payload.get("max_tokens", 400),
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    return _call


# ── Eval dispatch ─────────────────────────────────────────────────────


def _run_judge(
    *,
    eval_mode: str,
    example: dict,
    instruction: str,
    agent_answer: str,
    screenshots_b64: List[str],
    env,
    judge_client: Any,
    judge_model: str,
    n_screenshots_for_judge: int,
    action_history: Optional[List[str]] = None,
) -> tuple[float, str]:
    """Pick the right eval path. Returns (score, judge_raw)."""
    if eval_mode == "gold_reference":
        # gold-reference (string / URL / fuzzy match) eval was used by the
        # MMInA benchmark, which is not part of this release. OSWorld tasks
        # are scored by the environment's functional checker; Mind2Web is
        # scored by the answer-blind Online-Mind2Web grader below.
        raise NotImplementedError(
            "gold_reference eval (MMInA) is not included in this release; "
            "use OSWorld functional eval or the Mind2Web 'llm_judge' path."
        )

    # Mind2Web → Online-Mind2Web 3-stage grader (arXiv:2504.01382). It is
    # ANSWER-BLIND: judges from the RAW task + LLM-extracted key-points +
    # per-screenshot relevance + action history, and NEVER reads the
    # agent's self-reported ``finished(answer=...)``. This avoids the
    # WebVoyager judge's systematic false-FAIL where a correct navigation
    # is failed merely because the extracted answer lacks the literal
    # finished() sentinel. We pass the RAW intent (not the rewritten
    # finished()-preamble instruction) so Stage-1 key-point extraction
    # can't turn the sentinel mandate into a bogus key point.
    _dom = (example.get("_text_answer_domain") or "")
    _is_mind2web = (
        _dom.startswith("mind2web")
        or (example.get("evaluator") or {})
            .get("expected", {}).get("type") == "raw_intent")
    if eval_mode == "llm_judge" and _is_mind2web:
        # The Online-Mind2Web 3-stage grader needs a CAPABLE vision judge for
        # its per-screenshot relevance scoring. The 9B scores ~every shot "1"
        # (kept=0 → every task false-fails), so force-upgrade a 9B judge to the
        # 27B. A different (non-9B) judge passed explicitly is respected.
        # Override the upgrade target via M2W_JUDGE_MODEL.
        _weak = (not judge_model) or "9b" in str(judge_model).lower() \
            or "qwen35-vl" in str(judge_model).lower()
        if judge_client is None or _weak:
            target = (os.environ.get("M2W_JUDGE_MODEL", "vllm_qwen35-27b")
                      if _weak else judge_model)
            try:
                judge_client, judge_model = make_judge_client(target)
                if _weak:
                    logger.warning(
                        "[Eval] mind2web online grader needs a strong judge; "
                        "using %s (the 9B scores everything irrelevant)",
                        judge_model)
            except Exception as e:
                logger.error("[Eval] no judge client (%s); scoring 0", e)
                return 0.0, f"<no judge client: {e}>"
        raw_intent = (
            (example.get("evaluator") or {})
                .get("expected", {}).get("intent")
            or example.get("instruction")
            or instruction or "")
        raw_intent = re.sub(r"^\s*\[[^\]]+\]\s*", "", raw_intent).strip()
        try:
            from web_judge_online_m2w import WebJudge
            wj = WebJudge(client=judge_client, model=judge_model)
            score, judge_raw = wj.judge(
                task=raw_intent,
                action_history=list(action_history or []),
                screenshots_b64=screenshots_b64,
            )
        except Exception as e:
            logger.error("[Eval] online-mind2web judge failed: %s", e)
            return 0.0, f"<online-m2w error: {e}>"
        return max(0.0, min(1.0, float(score))), judge_raw

    # No other web grader ships in this release. WebVoyager (response-believing)
    # grading was removed; every web task here is a Mind2Web task scored by the
    # answer-blind Online-Mind2Web grader above.
    raise NotImplementedError(
        "Only the Online-Mind2Web grader is included; the task did not resolve "
        "to a mind2web eval (check example['_text_answer_domain'])."
    )


# ── Baseline-framework helpers (Agent-S3 / VLAA-GUI / OS-Symphony) ───


def is_mind2web_task(example: dict) -> bool:
    """Whether an OSWorld-format task config is a Mind2Web web task.

    Mind2Web tasks are graded by the answer-blind Online-Mind2Web grader
    (``score_text_answer``) instead of ``env.evaluate()``. Their configs are
    tagged either with ``_eval_mode`` (set by the dispatcher) or with
    ``evaluator.func == "llm_judge_webvoyager"`` (set by the task converter).
    Plain OSWorld tasks return False.
    """
    if example.get("_eval_mode"):
        return True
    return (example.get("evaluator") or {}).get("func") == "llm_judge_webvoyager"


def collect_step_screenshots_b64(
    example_result_dir: str, limit: int = 8
) -> List[str]:
    """Collect the trajectory's step screenshots already persisted by a
    baseline harness (step_0.png, step_N[_suffix].png), oldest→newest,
    capped to the last ``limit`` frames (the judge reads the last 5)."""
    import glob

    def _num(p: str) -> int:
        m = re.search(r"step_(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1

    paths = sorted(
        glob.glob(os.path.join(example_result_dir, "step_*.png")), key=_num)
    out: List[str] = []
    for p in paths[-limit:]:
        try:
            with open(p, "rb") as fh:
                out.append(base64.b64encode(fh.read()).decode("ascii"))
        except Exception as e:
            logger.warning("[screenshots] unreadable %s: %s", p, e)
    return out


# ── Shared scorer (main harness + baseline frameworks) ───────────────


def _needs_answer_channel(domain: Optional[str]) -> bool:
    """Whether a text-answer domain's SCORING reads the agent's final answer,
    i.e. the agent must emit ``finished(answer="...")``:
      - webvoyager — the WebVoyager judge reads the Result Response.
      - mmina_*    — gold-reference string/URL/fuzzy match against the answer.
    mind2web is graded by the ANSWER-BLIND Online-Mind2Web grader (task +
    key-points + screenshots + action history), so it needs NO answer channel:
    it runs as a normal OSWorld task and terminates via the DONE-gate +
    done-auditor (identical to env.evaluate desktop tasks)."""
    d = (domain or "").lower()
    return d == "webvoyager" or d.startswith("mmina")


def score_text_answer(
    *,
    example: dict,
    instruction: str,
    agent_answer: str,
    screenshots_b64: List[str],
    env,
    example_result_dir: str,
    judge_client: Any = None,
    judge_model: Optional[str] = None,
    n_screenshots_for_judge: int = 5,
    force_extract_llm: Any = None,
    force_extract_model: Optional[str] = None,
    action_history: Optional[List[str]] = None,
) -> float:
    """Score one finished text-answer episode and persist artifacts.

    Single scoring path for EVERY harness (main planner runner and the
    Agent-S3 / VLAA-GUI / OS-Symphony baselines call this same function),
    so all table rows are judged by byte-identical logic:

      1. If ``agent_answer`` is empty, run the force-extract rescue
         (last 3 screenshots + live CDP URL → one LLM call) — the same
         rescue the main runs had, applied uniformly for fairness.
      2. Dispatch eval by ``example["_eval_mode"]``:
         gold_reference → mmina_adapter.evaluate_mmina;
         llm_judge (default) → web_judge.judge_webvoyager.
      3. Write result.txt / answer.txt / judge.json into
         ``example_result_dir``; return the score.

    ``force_extract_llm`` is a ``(payload, model) -> str`` callable; when
    None a generic OpenAI-compatible caller on the judge endpoint is used.
    Call this while the env is still alive — the URL force-extract reads
    the live Chrome tab via CDP.
    """
    eval_mode = example.get("_eval_mode") or "llm_judge"
    if judge_model is None:
        judge_model = (os.environ.get("JUDGE_MODEL", "").strip()
                       or "vllm_qwen35-vl")
    agent_answer = (agent_answer or "").strip()
    text_answer_domain = (
        example.get("_text_answer_domain")
        or example.get("_mmina_subdomain"))

    if not agent_answer and screenshots_b64 and _needs_answer_channel(text_answer_domain):
        try:
            caller = force_extract_llm or _make_client_llm_caller(judge_model)
            current_url = _fetch_current_url(env)
            logger.info("[force-extract] current URL: %r",
                        (current_url or "")[:200])
            agent_answer = _force_extract_answer(
                call_llm=caller,
                model_arg=force_extract_model or judge_model,
                instruction=instruction,
                last_screenshots_b64=screenshots_b64[-3:],
                current_url=current_url,
                eval_mode=eval_mode,
                text_answer_domain=text_answer_domain,
            )
            if agent_answer:
                logger.info("[force-extract] recovered answer: %r",
                            agent_answer[:200])
            else:
                logger.info("[force-extract] LLM emitted nothing usable")
        except Exception as e:
            logger.warning("[force-extract] failed: %s", e)

    logger.info("Final answer (mode=%s): %r", eval_mode, agent_answer[:300])

    # Judge client used by BOTH eval modes (gold_reference needs it for
    # _llm_fuzzy_match of wikipedia text answers).
    if judge_client is None:
        try:
            judge_client, judge_model = make_judge_client(judge_model)
            logger.info("[judge] client built: model=%s endpoint=%s",
                        judge_model,
                        getattr(judge_client, "base_url", "?"))
        except Exception as e:
            logger.warning("[judge] could not build client (%s) — "
                           "gold_reference fuzzy will degrade to "
                           "substring; LLM judge will fail", e)
            judge_client = None

    score, judge_raw = _run_judge(
        eval_mode=eval_mode,
        example=example,
        instruction=instruction,
        agent_answer=agent_answer,
        screenshots_b64=screenshots_b64,
        env=env,
        judge_client=judge_client,
        judge_model=judge_model,
        n_screenshots_for_judge=n_screenshots_for_judge,
        action_history=action_history,
    )
    logger.info("Score: %.2f  (judge raw excerpt: %s)",
                score, judge_raw[:200])

    with open(os.path.join(example_result_dir, "result.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"{score}\n")
    with open(os.path.join(example_result_dir, "answer.txt"), "w",
              encoding="utf-8") as f:
        f.write(agent_answer or "(no answer)")
    with open(os.path.join(example_result_dir, "judge.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "score": score,
            "eval_mode": eval_mode,
            "agent_answer": agent_answer,
            "judge_model": (judge_model if eval_mode == "llm_judge"
                            else None),
            "judge_raw_response": judge_raw,
            "n_screenshots_seen": len(screenshots_b64),
        }, f, indent=2, ensure_ascii=False)

    return score


# ── Main loop ─────────────────────────────────────────────────────────


