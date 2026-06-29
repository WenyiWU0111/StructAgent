"""Comparative-MCQ judge for bBoN (see DESIGN.md).

ONE prompt, all N candidates at once — comparative selection beats independent
scoring (Agent-S3 finding). For EACH rubric outcome the judge decides which
candidates satisfy it BY THE EVIDENCE (especially the final state), explicitly told
NOT to trust any agent self-claim, then picks the single best candidate.

Reuses the agent's ``call_llm`` + a VLM judge model (needs to see screenshots).
``select_best`` is a pure function of its inputs (no agent state) so eval_judge can
replay it on a frozen rollout set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, List

from mm_agents.structagent.bbon.narrative import BehaviorNarrative


@dataclass
class Selection:
    index: int                 # chosen rollout (0-based); -1 if no candidates
    rationale: str = ""        # the judge's cited contrast (for eval + debugging)
    raw: str = ""              # raw judge response


JUDGE_SYSTEM = (
    "You are an exacting judge selecting which candidate ACTUALLY completed a "
    "computer-use task. You get a SUCCESS RUBRIC (the criteria) and, per candidate, "
    "a behavior storyboard: objective actions + observed evidence + screenshots at "
    "the decisive steps, and the FINAL STATE.\n"
    "METHOD:\n"
    "  - For EACH rubric outcome, decide which candidates satisfy it FROM THE "
    "EVIDENCE — weight the FINAL STATE most. Do NOT trust any 'AGENT SELF-CLAIM' "
    "line; agents frequently believe they succeeded when they did not.\n"
    "  - Watch for an outcome reached then BROKEN later (a value set then deleted / "
    "overwritten): compare the BEFORE vs AFTER screenshots.\n"
    "  - INFEASIBLE tasks: a candidate may emit FAIL / impossible (it judged the task "
    "undoable on this UI). If MOST candidates gave up and NONE shows the goal truly "
    "achieved, the task is likely infeasible — then a FAIL candidate is the CORRECT "
    "pick; do NOT choose one that merely 'looks busy' or 'attempted' it.\n"
    "  - Then pick the SINGLE candidate that best satisfies ALL outcomes. If several "
    "tie, prefer the one whose final state most directly shows every outcome met."
)

_OUTPUT_INSTRUCTION = (
    "\nWork CANDIDATE BY CANDIDATE first — judge each on its OWN merits against the "
    "rubric, do NOT compare across candidates yet. For EACH candidate, in order, "
    "output exactly one line:\n"
    "<verdict>LETTER: DONE|NOT_DONE — for each rubric outcome state whether THIS "
    "candidate's FINAL STATE shows it met, citing concrete evidence (a11y row / "
    "visible element / final screenshot); name any outcome left unmet</verdict>\n"
    "Only after all per-candidate verdicts, choose the single best among those you "
    "marked DONE (if none are DONE, pick the closest and say so). Output EXACTLY:\n"
    "<choice>LETTER</choice><cite>one sentence citing the decisive evidence</cite>"
)


def _text(s: str) -> dict:
    return {"type": "text", "text": s}


def _img(b64: str):
    if not b64:
        return None
    url = b64 if str(b64).startswith("data:") else f"data:image/png;base64,{b64}"
    return {"type": "image_url", "image_url": {"url": url}}


def _letter(i: int) -> str:
    return chr(ord("A") + i)


def build_judge_prompt(task: str, rubric: List[str],
                       narratives: List[BehaviorNarrative]) -> list:
    """Comparative-MCQ message payload (system + one multi-image user turn).

    Candidates are A, B, C…; each contributes its ``to_text()`` storyboard followed
    by its decisive screenshots (BEFORE/AFTER on revert frames, plus the final
    state), all labelled by candidate + step so the judge can cross-reference.
    """
    rubric_txt = "\n".join(f"  - {r}" for r in rubric) or "  (no compiled outcomes)"
    content: list = [_text(
        f"TASK: {task}\n\nSUCCESS RUBRIC (shared by all candidates):\n{rubric_txt}\n\n"
        f"There are {len(narratives)} candidates (A..{_letter(len(narratives) - 1)}). "
        "Each candidate's storyboard + screenshots follow."
    )]
    for i, nar in enumerate(narratives):
        c = _letter(i)
        content.append(_text(f"\n===== CANDIDATE {c} =====\n{nar.to_text()}\n"
                             f"[CANDIDATE {c} screenshots:]"))
        for t in nar.turning_points:
            _a = f"\n    a11y@step: {t.a11y[:500]}" if getattr(t, "a11y", "") else ""
            if t.before_b64:
                content += [_text(f"[{c}] step {t.step} BEFORE:"), _img(t.before_b64),
                            _text(f"[{c}] step {t.step} AFTER — {t.action}:{_a}"), _img(t.after_b64)]
            else:
                content += [_text(f"[{c}] step {t.step} — {t.action}:{_a}"), _img(t.after_b64)]
        fe = nar.final_evidence
        if fe and fe.last_screenshot_b64:
            content += [_text(f"[{c}] FINAL STATE:"), _img(fe.last_screenshot_b64)]
    content.append(_text(_OUTPUT_INSTRUCTION))
    content = [c for c in content if c]  # drop missing images
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": content}]


def _tok_to_idx(tok: str, n: int):
    tok = (tok or "").strip()
    if not tok:
        return None
    if tok[0].isalpha():
        idx = ord(tok[0].upper()) - ord("A")
        return idx if 0 <= idx < n else None
    if tok.isdigit():
        v = int(tok)
        # candidates are letters (A=1st); a digit is off-spec — read 1-based
        # first ("1" -> A -> 0), then 0-based ("0" -> 0).
        if 1 <= v <= n:
            return v - 1
        if 0 <= v < n:
            return v
    return None


def _parse_choice(resp: str, n: int):
    """Strict verdict parse: a clean ``<choice>X</choice>`` (single letter/digit),
    after dropping any ``<think>`` reasoning trailer. Off-format answers return
    None so ``select_best`` can RE-ASK the judge — a retry beats guessing at a
    garbled verdict (``<choice>A and C</choice>`` means the judge was unsure, not
    'A'; a missing tag means it didn't answer, not whatever ``<cite>`` mentions)."""
    resp = resp or ""
    if "</think>" in resp:
        resp = resp.rsplit("</think>", 1)[-1]
    m = re.search(r"<choice>\s*([A-Za-z0-9])\s*</choice>", resp)
    return _tok_to_idx(m.group(1), n) if m else None


def _cite_of(resp: str) -> str:
    m = re.search(r"<cite>(.*?)</cite>", resp or "", re.DOTALL)
    return m.group(1).strip() if m else ""


def select_best(task: str, rubric: List[str], narratives: List[BehaviorNarrative],
                call_llm: Callable, model: str, *, max_tries: int = 3,
                temperature: float = 0.0) -> Selection:
    """Run the comparative judge over the N narratives; parse the pick.

    On an unparseable verdict we RE-ASK the judge (up to ``max_tries``) with an
    explicit format reminder rather than guessing at a garbled answer. Only if
    every attempt fails do we fall back to candidate 0, flagged in ``rationale``
    (so eval separates parse failures from genuine judge errors). Pure given a
    deterministic ``call_llm`` so eval_judge can replay it.
    """
    if not narratives:
        return Selection(index=-1, rationale="no candidates")
    if len(narratives) == 1:
        return Selection(index=0, rationale="single candidate")
    messages = build_judge_prompt(task, rubric, narratives)
    resp = ""
    for attempt in range(max(1, max_tries)):
        msgs = messages if attempt == 0 else messages + [
            {"role": "assistant", "content": resp or "(no answer)"},
            {"role": "user", "content":
             "Your reply did not contain a valid verdict. Respond with EXACTLY "
             "this and nothing else: <choice>X</choice><cite>one sentence</cite> "
             "— where X is a SINGLE candidate letter (A, B, C, …)."}]
        resp = call_llm({"model": model, "messages": msgs,
                         "max_tokens": 2500, "temperature": temperature}, model) or ""
        idx = _parse_choice(resp, len(narratives))
        if idx is not None:
            return Selection(index=idx, rationale=_cite_of(resp), raw=resp)
    return Selection(index=0,
                     rationale=f"[parse-failed after {max(1, max_tries)} tries] "
                               + _cite_of(resp), raw=resp)


def select_best_voting(task: str, rubric: List[str],
                       narratives: List[BehaviorNarrative],
                       call_llm: Callable, model: str, *,
                       n_votes: int = 5, temperature: float = 0.7,
                       max_tries: int = 2) -> Selection:
    """Self-consistency judge: run ``select_best`` ``n_votes`` times at a sampling
    temperature and take the MAJORITY pick. Ties break toward the lowest index
    (stable). Cuts single-shot judge noise. Falls back to a single deterministic
    pick when there's nothing to vote on.
    """
    if len(narratives) <= 1:
        return select_best(task, rubric, narratives, call_llm, model)
    from collections import Counter
    votes: List[int] = []
    cites: dict = {}
    for _ in range(max(1, n_votes)):
        s = select_best(task, rubric, narratives, call_llm, model,
                        max_tries=max_tries, temperature=temperature)
        if isinstance(s.index, int) and 0 <= s.index < len(narratives) \
                and "[parse-failed" not in s.rationale:
            votes.append(s.index)
            cites.setdefault(s.index, s.rationale)
    if not votes:
        return Selection(index=0, rationale="[voting: all tries unparseable]")
    # most_common is order-stable for ties on Counter from a list; enforce
    # lowest-index tie-break explicitly for determinism.
    top = max(set(votes), key=lambda i: (votes.count(i), -i))
    return Selection(index=top,
                     rationale=f"[vote {votes.count(top)}/{len(votes)}] "
                               + cites.get(top, ""))
