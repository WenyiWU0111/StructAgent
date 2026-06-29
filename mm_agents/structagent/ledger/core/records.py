"""Ledger records — structured data the ledger accumulates per task / outcome.

  - Evidence / Artifact      : verifier-output artifacts (one per verify run).
  - Failure                  : per-Outcome failed-branch record.
  - Fact                     : captured working-memory value.
  - build_* (bundle_assembler): verify-trace -> Evidence record.

Merged from evidence.py / failure.py / bundle_assembler.py / fact.py
(Phase 1 declutter: 4 tiny dataclass modules -> one records module).
Intra-deps (Failure->Artifact, bundle_assembler->Evidence) are now
same-module references; ordering preserves them."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple




# =========================================================================
# from evidence.py
# =========================================================================


# ──────────────────────────────────────────────────────────────────────
# Evidence status — set by the cascade machinery (A8+).
#
# LIVE        — verifier confirmed this outcome and nothing has
#               invalidated it yet (default state for fresh Evidence).
# STALE       — the source outcome was REVERTED via reverify (A8); any
#               downstream outcome whose Evidence consumed this slot
#               should not trust it any more.
# INVALIDATED — explicitly marked invalid by something else (e.g.
#               human intervention via the planned ``invalidate_evidence``
#               action). Not reached automatically in A2.
# ──────────────────────────────────────────────────────────────────────
EVIDENCE_LIVE = "live"
EVIDENCE_STALE = "stale"
EVIDENCE_INVALIDATED = "invalidated"


@dataclass
class Artifact:
    """A sub-piece of evidence — a screenshot crop, a matched file line
    region, an extracted a11y row, etc. Kept loose-typed (kind + blob)
    so we can grow the set without schema migrations.

    A2: nothing populates this yet. A3+ uses screenshot crops and
    matched-line snippets when bundle_assembler has them in hand.
    """
    kind: str                                # "screenshot_crop" | "matched_lines" | "a11y_row" | ...
    payload: Any                             # JSON-serializable
    label: Optional[str] = None


@dataclass
class Evidence:
    """One verifier run's structured output, attached to the Outcome.

    Built by ``bundle_assembler.build()`` immediately after
    ``verify_with_trace`` returns and the KeyNode is decided.
    Appended to ``Outcome.evidence`` (list, latest last). Persists
    in traj.jsonl alongside the rest of the ledger.
    """
    triggering_keynode: Optional[Any]        # KeyNode (the detection that fired this verify)
    verifier_kind: str                       # file_grep / url_match / a11y_match / shell_command / calc_verify / ...
    verifier_trace: Dict[str, Any]           # full trace from verify_with_trace (no truncation)
    artifacts: List[Artifact] = field(default_factory=list)
    bound_slots: Dict[str, Any] = field(default_factory=dict)
    strategy_text: Optional[str] = None      # the planner strategy that drove this verify, if any
    step_span: Tuple[int, int] = (0, 0)      # (first_step_in_attempt, verify_step)
    valid_at_step: int = 0                   # invalidation horizon
    status: str = EVIDENCE_LIVE              # LIVE | STALE | INVALIDATED
    verdict: Optional[bool] = None           # True/False/None — what the verifier returned



# =========================================================================
# from failure.py
# =========================================================================


# Levels mirror the existing FailedPath levels for compat at A4.
FAILURE_STRATEGY = "strategy"
FAILURE_ACTION = "action"


# Evidence strength tags mirror the FailedPath ones.
ES_AUTO = "auto"
ES_PLANNER = "planner"
ES_KEYNODE = "keynode"


@dataclass
class Failure:
    """A dead-end record at one of two granularities (see ``level``).

    Same semantics as FailedPath but scoped per-Outcome (A4 migration).
    Fields preserved 1:1 so the relocation does not change persistence
    shape — only the *container* changes.
    """
    path: str
    why: str
    level: str = FAILURE_ACTION              # "strategy" | "action"
    action_sequence: List[str] = field(default_factory=list)
    evidence_strength: str = ES_AUTO         # "auto" | "planner" | "keynode"
    first_observed_at_step: int = 0
    first_observed_at_subgoal: str = ""
    last_observed_at_step: int = 0
    observation_count: int = 1

    # New A2 fields. Default-empty so legacy data round-trips through
    # __init__ without TypeErrors.
    related_strategy_id: Optional[str] = None
    artifacts: List[Artifact] = field(default_factory=list)



# =========================================================================
# from bundle_assembler.py
# =========================================================================


def build(
    *,
    outcome: Any,
    verdict: Optional[bool],
    trace: Optional[Dict[str, Any]],
    triggering_keynode: Optional[Any] = None,
    step_idx: int = 0,
    strategy_text: Optional[str] = None,
    bound_slots: Optional[Dict[str, Any]] = None,
) -> Evidence:
    """Build one Evidence record from a verify_with_trace run.

    Inputs:
      outcome             — the Outcome being verified (for verifier_kind
                            and step-span derivation; we read
                            outcome.verify.kind and
                            outcome.last_satisfy_step).
      verdict             — the verifier's True/False/None return.
      trace               — the trace dict the verifier returned.
                            None / non-dict → coerced to empty dict.
      triggering_keynode  — the KeyNode that was decided (may be None
                            on a failed verify or when KeyNode emission
                            was suppressed).
      step_idx            — the step index this verify ran in.
      strategy_text       — the planner's <strategy> text (if any).
      bound_slots         — slot values bound by THIS verify run (the
                            verifier slot binder fills these). Empty
                            dict on a verify with no slot bindings.

    Returns a fully-populated Evidence ready to be appended to
    outcome.evidence. status is LIVE; cascade machinery (A8) may flip
    it later.
    """
    verifier_kind = "?"
    if outcome is not None and getattr(outcome, "verify", None) is not None:
        verifier_kind = getattr(outcome.verify, "kind", "?")

    # step_span = (first step in this attempt, this verify step).
    # "first step in this attempt" = last_satisfy_step + 1 if the
    # outcome has been verified before (we're re-verifying after a
    # revert/recheck); otherwise 0 (the agent has been working on it
    # from the start).
    last_sat = getattr(outcome, "last_satisfy_step", None) if outcome is not None else None
    first_step = (last_sat + 1) if isinstance(last_sat, int) else 0
    span: Tuple[int, int] = (first_step, step_idx)

    return Evidence(
        triggering_keynode=triggering_keynode,
        verifier_kind=verifier_kind,
        verifier_trace=dict(trace) if isinstance(trace, dict) else {},
        artifacts=[],
        bound_slots=dict(bound_slots) if isinstance(bound_slots, dict) else {},
        strategy_text=strategy_text,
        step_span=span,
        valid_at_step=step_idx,
        status=EVIDENCE_LIVE,
        verdict=verdict,
    )



# =========================================================================
# from fact.py
# =========================================================================


# Type tags — informative only; consumers may inspect when rendering.
FACT_TYPE_STR = "str"
FACT_TYPE_LIST_STR = "list[str]"
FACT_TYPE_PATH = "path"
FACT_TYPE_URL = "url"
FACT_TYPE_INT = "int"
FACT_TYPE_FLOAT = "float"
FACT_TYPE_JSON = "json"


@dataclass
class Fact:
    """One captured working-memory value.

    Fields:
      name                  — flat key (e.g. "unpaid_emails", "doc_path")
      value                 — JSON-serializable payload
      type                  — informative type tag (see FACT_TYPE_* above)
      source_outcome_id     — Outcome id that produced this value,
                              or ``"global"`` for task-wide captures
                              (which live in ``ledger.global_facts``)
      source_step           — step at which the value was captured
      last_verified_at_step — step the value was last re-verified live
                              (mirrors source_step on creation; bumped
                              when a verifier confirms the source again)
    """
    name: str
    value: Any
    type: str = FACT_TYPE_STR
    source_outcome_id: str = "global"
    source_step: int = 0
    last_verified_at_step: int = 0


# ──────────────────────────────────────────────────────────────────────
# Rendering — A6 switches the planner / decomposer prompts from the
# legacy ``_notebook`` dict + ``actions.handlers.notebook.render`` to the per-
# outcome Facts view rendered by this function.
# ──────────────────────────────────────────────────────────────────────

_TRUNC_NOTE_LEN = 240


def _truncate(s: str, n: int = _TRUNC_NOTE_LEN) -> str:
    return s if len(s) <= n else (s[:n] + " …[truncated]")


def render_working_memory(
    facts: Dict[str, "Fact"],
    *,
    focus_outcome_id: Optional[str] = None,
) -> Optional[str]:
    """Render the live ``working_memory()`` view into a prompt block.

    Output mirrors the legacy ``[Notebook]`` block but groups Facts by
    their source (focus outcome first, other outcomes, then global)
    and surfaces ``_kind=extract_info`` payloads as a structured
    sub-section per file. Returns None when there's nothing live.
    """
    if not facts:
        return None

    # Bucket by source: focus outcome → others → global.
    focus_bucket: list = []
    other_bucket: list = []
    global_bucket: list = []
    for name, f in facts.items():
        owner = f.source_outcome_id
        if owner == "global":
            global_bucket.append((name, f))
        elif focus_outcome_id is not None and owner == focus_outcome_id:
            focus_bucket.append((name, f))
        else:
            other_bucket.append((name, f))

    out: list = [
        "[Working memory — typed Facts captured this task. Per-outcome "
        "Facts go stale when their source Outcome REVERTS; global "
        "Facts (env paths, task-wide context) stay live always. Read "
        "here BEFORE re-observing or re-extracting; the planner does "
        "not need to re-derive values that already live here.]"
    ]

    def _render_bucket(bucket, header):
        if not bucket:
            return
        # split notes vs extract_info payloads (latter has structured
        # {_kind: "extract_info", query, extracted}).
        notes, extracts = [], []
        for name, f in bucket:
            if (isinstance(f.value, dict)
                    and f.value.get("_kind") == "extract_info"):
                extracts.append((name, f))
            else:
                notes.append((name, f))
        if notes:
            out.append(f"\n{header} — notes ({len(notes)}):")
            for name, f in notes:
                v = f.value
                # _notebook-wrapped {"_kind":"note","value":...} payloads
                # land here in mixed dual-write states; surface the
                # inner value when present.
                if isinstance(v, dict) and v.get("_kind") == "note":
                    v = v.get("value")
                if isinstance(v, (dict, list)):
                    v_s = json.dumps(v, ensure_ascii=False)
                else:
                    v_s = str(v)
                out.append(f"  {name} = {_truncate(v_s)!r}")
        if extracts:
            out.append(f"\n{header} — extracted files ({len(extracts)}):")
            for name, f in extracts:
                extracted = f.value.get("extracted") or {}
                query = f.value.get("query") or ""
                out.append(f"  {name}")
                if query:
                    out.append(f"    query: {_truncate(str(query), 120)}")
                if isinstance(extracted, dict):
                    for k, v in extracted.items():
                        if isinstance(v, (dict, list)):
                            v_s = json.dumps(v, ensure_ascii=False)
                        else:
                            v_s = str(v)
                        out.append(f"    {k}: {_truncate(v_s)}")
                else:
                    out.append(f"    {_truncate(str(extracted))}")

    if focus_outcome_id and focus_bucket:
        _render_bucket(focus_bucket, f"From focus outcome `{focus_outcome_id}`")
    if other_bucket:
        _render_bucket(other_bucket, "From other outcomes")
    if global_bucket:
        _render_bucket(global_bucket, "Global / task-wide")

    return "\n".join(out)
