"""Unified single-example loop for text-answer tasks.

Covers three benchmarks today, all of which share the same shape:
  1. The agent navigates a browser (or any GUI surface).
  2. At task end, it emits a textual answer string via a sentinel.
  3. The score is computed AFTER env termination from
       (a) the answer string + (b) the trajectory's screenshots
       (and optionally the gold reference data when available).

Benchmarks routed here (and the eval they dispatch to):

  • WebVoyager / Mind2Web-executable  → ``web_judge.judge_webvoyager``
    LLM-as-judge over (intent + answer + last 5 screenshots). No gold
    reference; the judge model decides SUCCESS / NOT SUCCESS. Routed
    when ``example.get("_eval_mode")`` is ``"llm_judge"`` (the default).

  • MMInA  → ``mmina_adapter.evaluate_mmina``
    Gold-reference eval (string match / URL match / LLM fuzzy match).
    More precise than LLM-as-judge when gold answers exist. Routed when
    ``example.get("_eval_mode") == "gold_reference"``.

Sentinel: the planner prompt instructs the model to emit
``finished(answer="...")``. The agent's detection layer ALSO accepts the
legacy ``ANSWER(...)`` shape for backward-compat with wiki_search_executor
and any MMInA-era replay. Both populate ``self._text_answer``.
"""
from __future__ import annotations

import base64
import dataclasses
import datetime
import json
import logging
import os
import re
import time
from typing import Any, List, Optional

from lib_results_logger import log_task_completion, write_trajectory_html
from web_judge import (
    extract_web_answer,
    judge_webvoyager,
    make_judge_client,
)

logger = logging.getLogger("desktopenv.experiment")


def _dump_agent_runtime_config(agent, example_result_dir: str) -> None:
    cfg = getattr(agent, "cfg", None)
    if cfg is None:
        return
    try:
        payload = (
            dataclasses.asdict(cfg)
            if dataclasses.is_dataclass(cfg)
            else dict(getattr(cfg, "__dict__", {}) or {})
        )
        payload.update({
            "verifier_model": getattr(agent, "verifier_model", None),
            "planner_model": getattr(agent, "model", None),
            "decomposer_model": getattr(agent, "decomposer_model", None),
            "grounding_model": getattr(agent, "grounding_model", None),
            "effective_enable_boundary_verify": getattr(
                agent, "_boundary_verify_on", None),
            "effective_enable_actor_burst": getattr(
                agent, "_actor_burst_on", None),
            "effective_actor_burst_max_turns": getattr(
                agent, "_ACTOR_BURST_MAX_TURNS", None),
            "effective_actor_burst_no_effect_limit": getattr(
                agent, "_ACTOR_BURST_NO_EFFECT_LIMIT", None),
            "effective_actor_burst_debug": getattr(
                agent, "_actor_burst_debug", None),
            "effective_enable_done_auditor": getattr(
                agent, "enable_done_auditor", None),
            "done_auditor_model": getattr(agent, "done_auditor_model", None),
            "done_auditor_max_per_task": getattr(
                agent, "done_auditor_max_per_task", None),
            "done_auditor_budget_remaining": getattr(
                agent, "_done_audit_budget_remaining", None),
            "done_audit_reject_count": getattr(
                agent, "_done_audit_reject_count", None),
            "effective_enable_stuck_diagnosis_injection": getattr(
                agent, "enable_stuck_diagnosis_injection", None),
        })
        with open(
            os.path.join(example_result_dir, "structagent_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("Failed to dump StructAgent config: %s", e)


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

    # Default: LLM-as-judge (WebVoyager grader). Picks local vllm by
    # default for stability; set JUDGE_MODEL=qwen/qwen3.5-27b for OpenRouter.
    if judge_client is None:
        try:
            judge_client, judge_model = make_judge_client(judge_model)
        except Exception as e:
            logger.error("[Eval] no judge client (%s); scoring 0", e)
            return 0.0, f"<no judge client: {e}>"
    score, judge_raw = judge_webvoyager(
        intent=instruction,
        agent_answer=agent_answer,
        screenshots_b64=screenshots_b64,
        client=judge_client,
        model=judge_model,
        n_screenshots=n_screenshots_for_judge,
    )
    return max(0.0, min(1.0, float(score))), judge_raw


# ── Baseline-framework helpers (Agent-S3 / VLAA-GUI / OS-Symphony) ───


def text_answer_eval_mode(example: dict) -> Optional[str]:
    """Detect whether an OSWorld-format task config is a text-answer task.

    Returns ``"gold_reference"`` / ``"llm_judge"`` or None for plain
    OSWorld tasks (whose score comes from env.evaluate()). Detection keys:
    the MMInA converter stamps ``_eval_mode``; mind2web / webvoyager
    configs carry ``evaluator.func == "llm_judge_webvoyager"``.
    """
    if example.get("_eval_mode"):
        return example["_eval_mode"]
    if (example.get("evaluator") or {}).get("func") == "llm_judge_webvoyager":
        return "llm_judge"
    return None


def rewrite_text_answer_instruction(
    instruction: str, sentinel_prefix: str
) -> str:
    """W-P0: task hints name the main harness's ``finished(answer=...)``
    sentinel. Swap in the calling framework's own answer channel (e.g.
    ``agent.done(return_value=`` for S3/VLAA), and for tasks whose raw
    intent carries no hint at all (mind2web), append the same emission
    preamble the main harness uses — with the framework's action name.
    Semantics unchanged; only the action spelling differs per framework.
    """
    out = instruction.replace("finished(answer=", sentinel_prefix)
    if sentinel_prefix not in out:
        out = (
            out.rstrip()
            + "\n\nIMPORTANT: This task requires you to emit a final "
              "answer. After completing the navigation/search/filter "
              "steps, your last action MUST be "
              f"``{sentinel_prefix}\"<your answer here>\")`` — a concise "
              "sentence summarising what you found or what was done. "
              "Without this final emission the task is scored as failed."
        )
    return out


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


def run_single_example_text_answer(
    agent, env, example, max_steps, instruction, args, example_result_dir,
    scores,
    judge_client: Any = None,
    judge_model: Optional[str] = None,
    n_screenshots_for_judge: int = 5,
):
    """Run one text-answer task and write result.txt + answer.txt.

    Dispatches eval based on ``example["_eval_mode"]``:
      - ``"gold_reference"`` → mmina_adapter.evaluate_mmina (string /
                                URL / fuzzy-match against gold)
      - ``"llm_judge"`` (default) → web_judge.judge_webvoyager
                                     (vision LLM over screenshots)
    """
    eval_mode = example.get("_eval_mode") or "llm_judge"
    # Resolve judge model: explicit arg > env > local default. Local
    # vllm Qwen3.5-9B is the most reliable option (no quota / no
    # rate-limit empty-content surprises seen with OpenRouter).
    if judge_model is None:
        judge_model = (os.environ.get("JUDGE_MODEL", "").strip()
                       or "vllm_qwen35-vl")

    runtime_logger = setup_logger(example, example_result_dir)
    env.reset(task_config=example)

    try:
        agent.reset(runtime_logger, vm_ip=env.vm_ip)
    except Exception:
        agent.reset(runtime_logger)

    related_apps = example.get("related_apps", ["chrome"])
    # Mirror lib_run_single's per-task agent setup so we don't drop
    # state that the agent / decomposer / memory layer expects.
    agent._related_apps = list(related_apps)
    agent._current_domain = related_apps[0] if related_apps else "chrome"
    agent._vm_ip = getattr(env, "vm_ip", None)
    agent._chromium_port = getattr(env, "chromium_port", 9222)
    # All debug dumpers (planner_memory_debug, perceiver_debug,
    # stuck_diagnosis, digester audit, audit_debug, etc.) read
    # ``self._results_dir`` and silent-skip when it's None. Set per
    # task so artifacts land in the standard OSWorld per-task subdir.
    agent._results_dir = example_result_dir
    _dump_agent_runtime_config(agent, example_result_dir)
    # OSWorld convention: tag the instruction with [<domain>] so the
    # planner / decomposer's domain-specific hint injection picks the
    # right block. Without this, the planner sees an un-tagged web task
    # and may inject the wrong domain context.
    _instr_tag = (
        "multi_apps" if len(related_apps) > 1 else
        (related_apps[0] if related_apps else "chrome"))
    if not instruction.lstrip().startswith("["):
        instruction = f"[{_instr_tag}] {instruction}"
    # Answer channel (``finished(answer="...")``) is needed ONLY by domains
    # whose SCORING reads the agent's answer: webvoyager (the judge reads it)
    # and mmina_* (gold-reference match). mind2web is graded by the answer-blind
    # Online-Mind2Web grader, so it runs as a NORMAL OSWorld task — it
    # terminates via the DONE-gate + done-auditor (no finished(answer) preamble,
    # sentinel, or planner instruction).
    _ans_domain = (example.get("_text_answer_domain")
                   or example.get("_web_domain")
                   or example.get("_mmina_domain") or "")
    _answer_channel = _needs_answer_channel(_ans_domain)
    if (_answer_channel
            and "finished(answer" not in instruction
            and "ANSWER(" not in instruction):
        instruction = (
            instruction.rstrip()
            + "\n\nIMPORTANT: This task requires you to emit a final "
              "answer. After completing the navigation/search/filter "
              "steps, your last action MUST be "
              "``finished(answer=\"<your answer here>\")`` — a concise "
              "sentence summarising what you found or what was done. "
              "Without this final emission the task is scored as "
              "failed."
        )

    # Answer-channel flag: drives the finished(answer) sentinel detection
    # (loop.py) + the planner's finished(answer) instruction. OFF for mind2web
    # → the agent uses the normal OSWorld DONE logic (DONE-gate + done-auditor).
    agent._is_text_answer_task = _answer_channel
    agent._text_answer = None
    agent._text_answer_domain = (
        example.get("_text_answer_domain")
        or example.get("_web_domain")
        or example.get("_mmina_domain")
        or getattr(agent, "_text_answer_domain", None))

    # Chrome boot wait. Same value as MMInA / web runners.
    time.sleep(20)
    obs = env._get_obs()

    done = False
    step_idx = 0
    steps_for_html: List[dict] = []
    initial_plan = None
    memory_text = None
    all_actions: List[str] = []
    all_responses: List[str] = []
    final_screenshots_b64: List[str] = []

    env.controller.start_recording()

    while not done and step_idx < max_steps:
        response, actions = agent.predict(
            instruction, obs, env=env, wall_step_idx=step_idx)

        if step_idx == 0:
            initial_plan = getattr(agent, "current_plan", None)
            memory_text = getattr(agent, "memory_steps_text", None)

        current_plan = getattr(agent, "current_plan", None)
        next_step_hint = getattr(agent, "next_step_hint", None)
        observations = getattr(agent, "observations", None)
        current_subgoal = getattr(agent, "current_subgoal", None)
        planner_decision = getattr(agent, "planner_decision", None)
        planner_response = getattr(agent, "planner_response", None)

        all_responses.append(response or "")

        if getattr(agent, "all_done_after_step", False):
            logger.info("Agent reported done — terminating episode.")
            ss = _b64_screenshot(obs)
            if ss:
                final_screenshots_b64.append(ss)
            done = True
            break

        for action in actions:
            action_timestamp = datetime.datetime.now().strftime(
                "%Y%m%d@%H%M%S%f")
            logger.info("Step %d: %s", step_idx + 1, action)
            all_actions.append(str(action))

            obs, reward, done, info = env.step(
                action, args.sleep_after_execution)

            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                }))
                f.write("\n")

            screenshot_b64 = _b64_screenshot(obs)
            if screenshot_b64:
                final_screenshots_b64.append(screenshot_b64)
            steps_for_html.append({
                "screenshot_b64": screenshot_b64,
                "response": response,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "plan": current_plan,
                "next_step_hint": next_step_hint,
                "observations": observations,
                "current_subgoal": current_subgoal,
                "planner_decision": planner_decision,
                "planner_response": planner_response,
            })
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1

    time.sleep(3)

    # ── Answer extraction (dual-sentinel) ──────────────────────────────
    agent_answer = (getattr(agent, "_text_answer", None) or "").strip()
    if not agent_answer:
        for txt in reversed(all_responses):
            cand = extract_text_answer(txt)
            if cand:
                agent_answer = cand
                break
    if not agent_answer:
        pr = getattr(agent, "planner_response", "") or ""
        agent_answer = extract_text_answer(pr)

    # ── Force-extract + judge + persist via the SHARED scorer ──────────
    # score_text_answer is the single scoring path for every harness
    # (this planner runner and the S3/VLAA/Symphony baselines), so all
    # rows are judged byte-identically. force_extract_model preserves the
    # old behaviour here (the agent's own model alias drives the rescue
    # call, not the judge's).
    score = score_text_answer(
        example=example,
        instruction=instruction,
        agent_answer=agent_answer,
        screenshots_b64=final_screenshots_b64,
        env=env,
        example_result_dir=example_result_dir,
        judge_client=judge_client,
        judge_model=judge_model,
        n_screenshots_for_judge=n_screenshots_for_judge,
        force_extract_llm=(getattr(agent, "call_llm", None)
                           if getattr(agent, "_is_text_answer_task", False)
                           else None),
        force_extract_model=(getattr(agent, "model", None)
                             or "vllm_qwen35-vl"),
        action_history=all_actions,
    )
    scores.append(score)

    # write_trajectory_html no longer accepts initial_plan — Step 1's
    # ``plan`` field is byte-identical to it (see lib_results_logger
    # docstring). Pass only the supported kwargs.
    write_trajectory_html(
        example_result_dir,
        instruction=instruction,
        steps=steps_for_html,
        final_score=score,
        memory_text=memory_text,
    )
    log_task_completion(example, score, example_result_dir, args)
    env.controller.end_recording(
        os.path.join(example_result_dir, "recording.mp4"))


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(
        f"text_answer.{example.get('id', 'unknown')}")
    runtime_logger.setLevel(logging.INFO)
    for h in list(runtime_logger.handlers):
        runtime_logger.removeHandler(h)
    runtime_logger.addHandler(
        logging.FileHandler(os.path.join(example_result_dir, "runtime.log"))
    )
    return runtime_logger
