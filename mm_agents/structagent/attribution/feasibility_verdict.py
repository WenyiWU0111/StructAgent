"""Evidence-driven feasibility verdict — Fix 1 for the infeasible-task problem.

Fires after the agent force-replans ``>= K`` times while stuck. A feasibility
judge (the verifier model) reads the accumulated evidence and decides INFEASIBLE
(emit ``FAIL``) vs CONTINUE replanning. Post-hoc rather than an up-front gate:
a small model can't judge feasibility at t=0, but after K stuck replans there's
enough evidence (paths walked, outcomes never reached, workarounds attempted)
to make the call.

CONSISTENCY CONTRACT (read before editing the prompt or brief): this module is
the single source of truth for the three artifacts validated on V4 replay
(12/12: 2/2 infeasible detected, 0 FP; Sonnet + local Qwen3.5-9B, thinking ON):
FEASIBILITY_SYSTEM, extract_brief_from_log, build_user_message + parse_verdict.
The replay harness (/tmp/feasibility_replay/validate.py) imports these exact
symbols, so validated == production by construction. The brief is built from the
live runtime.log because the ``<evidence>`` stream only exists as logged LLM
output — no structured in-memory field to read. RE-RUN validate.py after any
change to the prompt or extraction.

Design invariants (each grounded in a replay failure):
  * Reason from FACTS, not the verifier's pass/fail verdict — verifiers
    false-reject feasible tasks (over-strict regex/format/encoding); trusting
    the verdict caused the only false-positive in the thinking-off run.
  * INFEASIBLE requires DIRECT evidence the target is ABSENT; "stuck" or
    "verifier keeps rejecting" is not enough.
  * Conservative — anything not a clean INFEASIBLE → CONTINUE. A false
    INFEASIBLE fails a winnable task (high cost); a miss just keeps trying.
"""
from __future__ import annotations

import re
from typing import Tuple

INFEASIBLE = "INFEASIBLE"
CONTINUE = "CONTINUE"
PARSE_FAIL = "PARSE_FAIL"


# ═══════════════════ 1. system prompt (validated verbatim) ════════════════ #

FEASIBILITY_SYSTEM = """You are a FEASIBILITY JUDGE for a GUI agent on an Ubuntu VM. The agent
has gotten STUCK — it replanned 8+ times without completing the task. Decide whether
the task is INFEASIBLE (genuinely cannot be done on this VM by a GUI agent) or the
agent should CONTINUE (the failures are tactical and a better attempt could succeed).

The "verifier feedback" below is a set of CHECK RESULTS — useful FACTS (return codes,
whether a pattern was found in a file, what is visible on screen), but its pass/fail
JUDGEMENT is NOT authoritative: verifiers FALSE-REJECT feasible tasks (over-strict
regexes, format/encoding mismatches like http:// vs no-protocol, – vs '-'). Reason
from the underlying FACTS, not from the verifier's verdict.

INFEASIBLE — only when the FACTS directly show the goal genuinely cannot be reached:
- The agent's ACTIONS SUCCEED (commands return 0, clicks land, files get written) yet the
  TARGET STATE is repeatedly absent — because the goal needs a feature/setting/option/key
  that DOES NOT EXIST in the software (e.g. writes a config key, return 0, but the key is
  never honored / not a real setting; opened the app's flags/menus and the option isn't there).
- The agent exhausted the legitimate paths (the app's own settings/menus) and the
  feature simply isn't there.
- The agent is now reaching for WORKAROUNDS resting on an UNVERIFIED premise (editing a
  config key it never confirmed exists, creating files/symlinks, unrequested extensions).
  Reaching for creative workarounds to do a simple task is itself an infeasibility signal.
- A required input file/resource genuinely does not exist on the VM.

CONTINUE — the failures are TACTICAL, or the negative signal is unreliable:
- Actions are ERRORING (wrong coordinates, command typos, transient file locks, focus
  issues) — technique problems, not impossibility.
- The agent is making PARTIAL progress toward the goal.
- The legitimate path hasn't been fully explored yet.
- CRUCIAL: the action VISIBLY SUCCEEDED (star filled, toggle flipped, file CONTAINS the
  target value) but a verifier check rejected it on a FORMAT/REGEX/ENCODING technicality.
  That is a broken check, NOT infeasibility — the agent did the thing → CONTINUE.
- The task is really a QUESTION / advice request the agent should ANSWER (not a GUI
  manipulation) — wrong-action-space, NOT infeasibility → CONTINUE.

Decision rule: INFEASIBLE ONLY when the agent's own observations DIRECTLY show the target
feature/setting/resource is ABSENT. "The verifier keeps rejecting" or "no obvious GUI
action" is NOT sufficient. Being stuck is NOT proof of infeasibility — agents are often
just bad at feasible tasks. False-positives are very costly; when in doubt, CONTINUE.

Output exactly two lines:
VERDICT: INFEASIBLE   (or)   VERDICT: CONTINUE
REASON: <one sentence citing the specific evidence>"""


# ═══════════════════ 2. evidence brief (validated verbatim) ═══════════════ #

def extract_brief_from_log(log_text: str, replan_cutoff: int) -> str:
    """Build the evidence brief from runtime-log TEXT, using only what was logged
    before the ``replan_cutoff``-th ``force_replan`` (the trigger). Strategy tags,
    planner bottlenecks, verifier ``<evidence>`` checks. Harness and live loop both
    call this — do not fork it.
    """
    lines = log_text.splitlines()
    replan_idx = [i for i, l in enumerate(lines) if "force_replan" in l]
    cut = replan_idx[replan_cutoff - 1] if len(replan_idx) >= replan_cutoff else len(lines)
    head = "\n".join(lines[:cut])

    strategies = re.findall(r"<strategy>(.*?)</strategy>", head, re.DOTALL)
    bottlenecks = [m.split("Bottleneck:", 1)[1].strip()
                   for m in re.findall(r"PA-Planner\] Bottleneck:[^\n]*", head)]
    # per-step post-state checks; drop the prompt-template line
    evid = re.findall(r"<evidence>(.*?)(?:</evidence>|\n)", head, re.DOTALL)
    evid = [e.strip()[:240] for e in evid if "One short sentence" not in e and e.strip()]

    def fmt(items, n):
        items = [i.strip().replace("\n", " ")[:200] for i in items if i.strip()]
        items = items[-n:]
        return "\n".join(f"  {j+1}. {x}" for j, x in enumerate(items)) or "  (none)"

    return (f"### Strategies attempted (most recent {min(len(strategies),10)})\n{fmt(strategies,10)}\n\n"
            f"### Per-attempt bottleneck diagnoses\n{fmt(bottlenecks,10)}\n\n"
            f"### Check results (FACTS: return codes, file/screen observations — the pass/fail verdict may be a false-reject)\n{fmt(evid,10)}")


# ═══════════════ 3. user wrapper + parser (validated verbatim) ════════════ #

def build_user_message(instruction: str, brief: str) -> str:
    return (f"## Task\n{instruction}\n\n"
            "## Evidence (what the agent tried and how it failed, up to when it got stuck)\n\n"
            f"{brief}")


def _strip_thinking(txt: str) -> str:
    """Qwen-3.5 (thinking on) via vLLM injects the opening <think> into the prompt,
    so the completion carries only the closing </think>; return everything after it.
    No-op for non-thinking output."""
    if "</think>" in txt:
        return txt.split("</think>", 1)[1]
    return txt


def parse_verdict(text: str) -> Tuple[str, str]:
    """Parse the judge reply → (verdict, reason), verdict ∈ {INFEASIBLE, CONTINUE,
    PARSE_FAIL}. The harness keeps PARSE_FAIL distinct; production maps anything
    != INFEASIBLE to CONTINUE (conservative)."""
    txt = _strip_thinking(text or "")
    m = re.search(r"VERDICT:\s*(INFEASIBLE|CONTINUE)", txt, re.I)
    v = m.group(1).upper() if m else PARSE_FAIL
    rm = re.search(r"REASON:\s*(.+)", txt)
    reason = (rm.group(1).strip() if rm else txt.strip())[:300]
    return v, reason


# ═══════════════════════ 4. production entry point ════════════════════════ #

def feasibility_verdict(
    call_llm,
    model: str,
    instruction: str,
    brief: str,
    max_tokens: int = 6144,
    logger=None,
) -> Tuple[str, str]:
    """Run the feasibility judge via ``call_llm`` on the verifier model. Returns
    ``(verdict, reason)``, verdict ∈ {INFEASIBLE, CONTINUE} (PARSE_FAIL and errors
    collapse to CONTINUE — conservative).

    ``force_thinking`` makes the llm-client force thinking ON for Qwen-3.5
    regardless of the global ``QWEN35_THINKING`` switch — the judge false-positives
    without it (validated). No-op for Claude / other models."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": FEASIBILITY_SYSTEM},
            {"role": "user", "content": build_user_message(instruction, brief)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.9,
        "force_thinking": True,
    }
    try:
        raw = call_llm(payload, model)
    except Exception as e:  # never let the judge crash the loop
        if logger:
            logger.warning("[FeasibilityVerdict] call failed (%s) — defaulting CONTINUE", e)
        return CONTINUE, f"(judge error: {e})"
    v, reason = parse_verdict(raw or "")
    if logger:
        logger.info("[FeasibilityVerdict] raw_verdict=%s reason=%r", v, reason[:200])
    return (INFEASIBLE if v == INFEASIBLE else CONTINUE), reason
