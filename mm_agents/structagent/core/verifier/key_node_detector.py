"""KeyNode detector — one multimodal LLM call per planner turn.

Looks at (prev observation, current observation, actor action, outcomes
table) and emits a list of `KeyNode` recording meaningful state
transitions:
  - outcome_satisfied / outcome_invalidated   (drives the OutcomeStateCache)
  - value_committed / navigation / dialog_opened / dialog_closed
    (auxiliary signals that enrich the event stream — not gated by
    outcome state)

Conservative output discipline:
  * every KeyNode MUST cite an `evidence` string (a11y row or visual phrase);
    no evidence → drop the KeyNode at parse time
  * confidence is an ordinal label `low|medium|high` (LLM self-rated floats
    are unreliable; ordinals stay calibratable)
  * empty list is the correct answer when nothing meaningful happened
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from mm_agents.structagent.ledger.core.timeline import (
    KeyNode,
    StepRecord,
    CONF_LOW,
    CONF_MEDIUM, CONF_HIGH, KN_OUTCOME_SATISFIED,
    KN_OUTCOME_INVALIDATED, KN_VALUE_COMMITTED,
    KN_NAVIGATION, KN_DIALOG_OPENED,
    KN_DIALOG_CLOSED, OUTCOME_PENDING,
    OUTCOME_VERIFIED,
)
# A2: silent-observer assembler. Builds an Evidence per verify_with_trace
# run and appends to outcome.evidence; nothing reads outcome.evidence
# yet — A3 wires the planner's Verifier-Feedback block to it.
from mm_agents.structagent.ledger.core.records import (
    build as _build_evidence,
)

logger = logging.getLogger("desktopenv.key_node_detector")

_VALID_KINDS = {
    KN_OUTCOME_SATISFIED, KN_OUTCOME_INVALIDATED,
    KN_VALUE_COMMITTED, KN_NAVIGATION,
    KN_DIALOG_OPENED, KN_DIALOG_CLOSED,
}
_VALID_CONF = {CONF_LOW, CONF_MEDIUM, CONF_HIGH}


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """You are the KeyNode detector for a GUI automation agent.
Each turn you receive two consecutive screen observations plus the actor
action that ran between them, the current authoritative outcome states,
and a machine-computed a11y delta. Your job is to emit a JSON list of
``KeyNode`` records describing meaningful state transitions caused by
this turn.

Two KeyNode kinds are LOAD-BEARING (they flip the outcome state machine):

  outcome_satisfied
    A pending outcome's evidence is now concretely visible on the
    CURRENT screen. Quote the specific a11y row or visual phrase as
    evidence. Do NOT emit when the evidence is only about the
    INTENT to satisfy ("user just clicked the filter") — wait for
    the next turn where the post-click state is observable.

  outcome_invalidated
    A previously verified outcome's underlying STATE was undone — NOT
    merely scrolled out of view, NOT merely "I cannot verify it".
    Distinguish carefully:
      • "off-screen but still applied" → NOT invalidated. Filter chips
        in URL params, applied-filters summary section, page result
        counts that reflect the filter — these prove the state holds
        even if the input control itself isn't visible. Emit nothing.
      • "refined / extended value still containing the required one"
        → NOT invalidated. If the hint requires a value to be present
        (substring, list item, selection) and the new state still
        contains it alongside additions, the hint still holds. Emit
        nothing.
      • "actively undone" → invalidated. Examples:
          - URL no longer contains the filter query param
          - "Selected" label / "Remove X filter" link disappeared
            from the active-filters bar
          - Toggle / radio / checkbox flipped to opposite state
          - Page navigated to a different URL that resets state

    MECHANICAL CHECK before emitting outcome_invalidated:
      Step A — name the underlying state that was verified.
      Step B — quote SPECIFIC current evidence that the state was
               REVERSED (not just hidden):
        Allowed: URL diff, toggle now shows opposite state in a11y
                 (`checked` flag now absent, `selected` gone), value
                 field now shows different content, page navigated
                 to a state-reset URL.
        DISALLOWED phrases — if your evidence reads like any of these,
        DO NOT emit invalidated:
          "control no longer visible", "dialog is closed",
          "I cannot verify", "state unverified", "evidence is gone",
          "off-screen", "we cannot confirm".
        Absence of evidence ≠ evidence of reversal. Emit nothing.

  outcome_re_satisfied (re-emit outcome_satisfied with kind="outcome_satisfied")
    A previously REVERTED outcome's evidence is freshly visible again.
    Treat exactly like a first-time satisfied: emit outcome_satisfied
    with current evidence. This recovers from false-positive reverts
    or genuine fix-it-back actions.

Auxiliary KeyNode kinds (record when meaningfully observed; do not gate
state, just enrich the timeline):

  value_committed   — input field went from focused/typing to committed
  navigation        — URL or top-level page changed
  dialog_opened     — modal / popup / context menu opened
  dialog_closed     — modal / popup / context menu closed

Each KeyNode MUST include:
  kind        — one of the kinds above
  target      — for outcome_*: the exact outcome_id; for others: a short
                human label ("address bar dropdown", "Print dialog")
  evidence    — concrete quote from a11y rows or visual description,
                ≤ 240 chars. NO evidence → don't emit the node.
  confidence  — "low" | "medium" | "high"
                Use "high" only when the evidence is unambiguous.
                "medium" is the default when the evidence appears in
                a11y but with some interpretation needed.
                "low" when the visual/a11y is partly occluded or you
                are inferring across an animation.

A11Y STATE COLUMN — read this carefully, it is the leading source of
false-positive outcome_satisfied for toggle/checkbox/radio outcomes:

  Each a11y row is tab-separated:    <tag>\\t<text>\\t<state>
  The state column lists ONLY attributes whose value is TRUE; absent
  attributes are FALSE. Relevant flags:
    `checked`  → toggle / check-box is ON / ticked
    `selected` → radio-button / menu-item is the chosen one
    `focused`  → has keyboard focus  (NOT relevant to on/off)
    `enabled`  → control is INTERACTABLE (user can click it).
                 Every active control has `enabled`. ``enabled`` does
                 NOT mean "the feature is on" or "the toggle is flipped".

  Therefore for a toggle / check-box outcome ("toggle X is in 'on'
  state", "Black filter is checked"):
    Row state contains `checked`        → outcome_satisfied is allowed
    Row state contains only `enabled`   → control exists but is OFF;
    (or `enabled,focused`)                outcome is NOT satisfied,
                                          do NOT emit outcome_satisfied
    No matching row at all              → cannot tell from a11y; fall
                                          back to visual cue, otherwise
                                          emit nothing
  Same rule for radio/menu outcomes with `selected`.

  Do NOT confuse "enabled = clickable" with "feature is on". Cite
  the literal state column so the rule is mechanically checkable.

DUAL-CHANNEL VISUAL GATE — for any outcome whose ``verify.kind ==
"a11y_match"`` and where the outcomes table lists `visual gates ...
MUST HOLD / MUST NOT HOLD` clauses, judging is TWO-CHANNEL:

  Channel 1 (a11y) — the row-level constraint authored by the ledger
                     (tag / text_contains / state_contains).
  Channel 2 (visual) — the listed visual_must_hold and
                     visual_must_not_hold clauses, applied to the
                     CURRENT screenshot.

  RULE: emit ``outcome_satisfied`` ONLY IF
    Channel 1 succeeds  AND
    every MUST HOLD clause is true on the current screenshot  AND
    every MUST NOT HOLD clause is false on the current screenshot.

  If ANY visual clause fails — even when the a11y row matches —
  treat the outcome as STILL pending. Do NOT emit ``outcome_satisfied``;
  do NOT emit ``outcome_invalidated`` either (we are not undoing
  anything, just refusing to commit prematurely). Emit nothing for
  that outcome and let the next turn re-check.

  Cite each visual gate verdict explicitly in <reasoning>:

    visual_must_hold[0]:    HOLDS    "<short cite of screenshot>"
    visual_must_hold[1]:    FAILS    "<reason — the dialog is still there>"
    visual_must_not_hold[0]: HOLDS-FALSE / HOLDS-TRUE → ...

  This stops a single transient overlay row (file picker tree,
  command palette item, tooltip) from triggering a false satisfied
  verdict when the action has not yet committed.

OFFICE DOCUMENT STRUCTURE BLOCK — when the user message includes a
``[Document structure block — AUTHORITATIVE for font / color / size
/ cell values; supersedes screenshot for structural properties]``
section, that block is the ground truth for STRUCTURAL outcomes on
libreoffice_calc / libreoffice_impress / libreoffice_writer:

  font weight (bold)            → look for ``bold`` keyword on the
                                   shape's line
  underline / italic / strike   → ``underline`` / ``italic`` / ``strike``
  font color                    → ``#RRGGBB`` hex value on the shape
  font name                     → e.g. ``Calibri`` / ``Liberation Sans``
  font size                     → ``Npt``
  shape size                    → ``size=(Wcm,Hcm)``
  table cell text               → ``cell[r,c] "..."``
  cell value (calc)             → printed in the sheet section
  slide / sheet count           → ``N slides`` / ``Sheet[name]`` headers
  paragraph alignment           → ``left`` / ``right`` / ``center``
                                   on the shape's line

For these properties the block is READ DIRECTLY from the UNO document
model. The screenshot's pixel rendering can be misleading — fonts like
"Open Sans Bold" or "Lato Black" already look weighted at normal
CharWeight, so visual inspection routinely calls a bolded-via-op
shape "not bold". When the block shows ``bold`` on the shape line for
a "bold" outcome, the outcome is satisfied; quote the SHAPE LINE from
the block in your evidence.

OUTPUT FORMAT — strictly two blocks in this order, no other text:

<reasoning>
For each outcome currently in the table, emit ONE bullet:

  - {outcome_id} (was {pending|verified|reverted}):
      hint:        "{paraphrase of evidence_hint}"
      observation: "{what current a11y/screen shows for this hint}"
      hint_holds_now: YES | NO
        ↑ Answer the hint as a yes/no QUESTION about the CURRENT screen.
        Required text still appears (even as substring of a longer
        value)? still YES. Required state still set (even if input is
        off-screen, URL still has it, indicator chip still shown)?
        still YES. Only NO when the literal requirement is genuinely
        absent / contradicted on screen.

        BUT — when the hint is about a SETTING / KEYBINDING / LIST-ROW
        STATE (a checkbox value, a keybinding row present-or-absent, a
        list item count, a config value), the answer must be derived
        from the CONTROL or DATA ROW itself, NOT from search-result
        labels ("Showing N matches"), the contents of an input/search
        field, or count badges. A "0 results" filter view is NOT
        evidence that the underlying data was changed — the same view
        appears for an empty initial state. If you cannot directly
        observe the control / data row (it is off-screen, the list is
        collapsed, or only the search summary is visible), answer NO
        and let the next turn re-check.
      verdict: derived strictly from the previous-state × hint_holds_now:
        was pending  + hint_holds_now=YES → SATISFIED
        was reverted + hint_holds_now=YES → SATISFIED   (re-verify)
        was verified + hint_holds_now=NO  → INVALIDATED
        all other combinations             → NO_CHANGE

If you also detected non-outcome events (navigation, dialog, value
committed), list them after the outcomes:

  - aux: {kind} → {target} — {short evidence}
</reasoning>

<patch>
{
  "key_nodes": [
    {"kind": "outcome_satisfied", "target": "<outcome_id>",
     "evidence": "<concrete quote>", "confidence": "high|medium|low"},
    ...
  ]
}
</patch>

Rules:
- Emit a KeyNode ONLY when the verdict is SATISFIED or INVALIDATED for
  outcomes; NO_CHANGE produces nothing for that outcome.
- An outcome that was already verified should NOT be re-emitted as
  outcome_satisfied — only emit invalidated/satisfied at TRANSITIONS.
- An outcome that is currently REVERTED CAN be re-emitted as
  outcome_satisfied if its evidence is freshly visible — this recovers
  from false-positive reverts.
- Be conservative on outcome_invalidated: distinguish "off-screen but
  still applied" (e.g., URL params, active-filter chips) from "actively
  undone" (URL diff, selected-state gone, toggle flipped). Prefer
  NO_CHANGE if you cannot quote concrete UNDOING evidence — false-
  positive reverts wreck runs more than missed reverts (the next turn
  has another chance to detect a true revert).
- If nothing changed meaningfully, emit empty key_nodes: []."""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def detect_key_nodes(
    *,
    prev_step: Optional[StepRecord],
    current_step: StepRecord,
    required_outcomes: List[Any],            # List[Outcome]; duck-typed (.id, .description, .evidence_hint, .verify)
    current_outcome_states: Dict[str, str],  # {oid: pending|verified|reverted}
    actor_action_summary: str,
    a11y_delta_block: Optional[str],         # already-rendered WindowDelta.to_prompt_block()
    call_llm: Callable,
    model: Optional[str] = None,
    max_tokens: int = 2000,
    snapshot: Optional[Any] = None,           # SubgoalSnapshot — perceiver's pre-distilled view (auxiliary)
    doc_inspect_block: Optional[str] = None,  # office-domain SP/DP/SP block (authoritative for font / color / size / cell values; supersedes the screenshot for structural properties)
    env: Optional[Any] = None,                # used for deterministic file_grep verify
    instruction: str = "",                    # used by verifier for app detection / fallback path
    results_dir: Optional[str] = None,        # debug dump target dir (e.g. <results>/<task>/)
    debug_step_idx: Optional[int] = None,     # explicit step idx for debug filenames
    agent_state: Optional[Dict[str, Any]] = None,  # per-task state (_ledger, _ledger_slots, _physical_domain, ...)
    earlier_steps: Optional[List[StepRecord]] = None,  # extra frames OLDER than prev (oldest→newest) for the visual judge
) -> List[KeyNode]:
    """KeyNode detection with hybrid deterministic + LLM dispatch.

    Per-outcome flow:
      1. If outcome carries a ``verify`` spec AND state is pending/reverted,
         run the deterministic verifier (file_grep / a11y_match / url_match):
           - True  → emit outcome_satisfied with conf=high; SKIP the LLM
                     for this outcome.
           - False → emit nothing (outcome stays pending). Trust the
                     deterministic verdict — file content / DOM / URL
                     is the ground truth, not LLM inference.
           - None  → uncertain (path unresolvable, file unreadable,
                     a11y row not visible). Fall back to the LLM path
                     for this outcome.

      2. If outcome already verified, OR has no verify spec, OR verifier
         returned None, hand it to the LLM (legacy path) — same prompt
         as before, just with a smaller required_outcomes list.

    The LLM call is skipped entirely if every outcome got a deterministic
    verdict (saves ~1 LLM call per step on the common case).

    When ``snapshot`` is provided, the perceiver's pre-verified clauses,
    relevant_controls and transition summary are injected into the LLM
    prompt as a strong prior — KeyNode treats the perceiver as having
    already done part of the verification work.
    """
    if not required_outcomes:
        return []

    # ─── Debug input dump (zero-cost when results_dir is None) ───
    from mm_agents.structagent.core.verifier import key_node_debug
    a11y_text = current_step.a11y or ""
    dbg_step = debug_step_idx if debug_step_idx is not None else current_step.step_idx
    key_node_debug.dump_keynode_input(
        results_dir=results_dir,
        step_idx=dbg_step,
        required_outcomes=required_outcomes,
        current_states=current_outcome_states,
        a11y_text=a11y_text,
        actor_action_summary=actor_action_summary,
    )

    # ─── Phase 1: deterministic verify pass ───
    from mm_agents.structagent.core.verifier.verifiers import verify_with_trace
    deterministic_kns: List[KeyNode] = []
    needs_llm: List[Any] = []
    per_outcome_trace: List[Dict[str, Any]] = []  # for debug dump

    # Step-0 baseline-skip. At step 0 the actor has not had a turn yet,
    # so every outcome that requires actor work would verify=False —
    # and any timing-sensitive verify (e.g. a UNO verify → LO not fully
    # ready right after snapshot reset) risks blocking the entire 110s
    # HTTP read budget per outcome before the first action even runs.
    # Treat all outcomes as pending without inspecting state; the next
    # round (after the actor's first turn) does the real check.
    _step_idx = getattr(current_step, "step_idx", 0)
    if _step_idx == 0:
        logger.info(
            "[KeyNodeDetector] step=0 baseline verify skipped for %d "
            "outcome(s) — running fresh from snapshot, no work yet",
            len(required_outcomes),
        )
        for o in required_outcomes:
            per_outcome_trace.append({
                "outcome_id": o.id,
                "previous_state": current_outcome_states.get(
                    o.id, OUTCOME_PENDING),
                "verify_kind": (o.verify.kind if getattr(o, "verify", None)
                                else None),
                "deterministic_verdict": None,
                "deterministic_trace": {
                    "reason": "step=0 baseline skipped (no actor turn yet)"
                },
                "took_llm_path": False,
                "emitted_keynode": None,
            })
        key_node_debug.dump_keynode_verdicts(
            results_dir=results_dir,
            step_idx=dbg_step,
            per_outcome=per_outcome_trace,
        )
        return []

    # R2 — physical-app scope gate. For outcomes whose verify needs the
    # focused window's UI (a11y_match / url_match) and whose outcome.app
    # differs from the currently-focused app, skip BOTH deterministic
    # and LLM paths this step. The outcome stays at its previous state.
    # Rationale: ``_extract_active_url`` reads whatever URL is in the
    # current a11y dump; ``a11y_match`` looks for controls in that dump
    # — if the wrong app is focused, both produce noise. Best to
    # explicitly defer the verification to a turn where the right
    # window is forward.
    _physical = (agent_state or {}).get("_physical_domain")
    _UI_DEPENDENT_KINDS = ("a11y_match", "url_match")

    def _app_mismatch(outcome) -> bool:
        if not _physical:
            return False
        v = getattr(outcome, "verify", None)
        if v is None or v.kind not in _UI_DEPENDENT_KINDS:
            return False
        oa = getattr(outcome, "app", None)
        if not oa:
            return False
        return str(oa).strip().lower() != str(_physical).strip().lower()

    for o in required_outcomes:
        cur_state = current_outcome_states.get(o.id, OUTCOME_PENDING)
        verify = getattr(o, "verify", None)
        rec: Dict[str, Any] = {
            "outcome_id": o.id,
            "previous_state": cur_state,
            "verify_kind": verify.kind if verify else None,
            "deterministic_verdict": None,
            "deterministic_trace": None,
            "took_llm_path": False,
            "emitted_keynode": None,
        }
        # R2 — app scope short-circuit (before any deterministic /
        # LLM dispatch). Logged for offline replay.
        if _app_mismatch(o):
            rec["deterministic_trace"] = {
                "reason": (
                    f"app scope mismatch: outcome.app={getattr(o, 'app', None)!r} "
                    f"physical={_physical!r} — UI-dependent verify "
                    "deferred to a turn where the correct window is focused"
                ),
                "app_scope_skipped": True,
            }
            logger.info(
                "[KeyNode-AppScope] step=%d %s skipped "
                "(outcome.app=%s, physical=%s)",
                current_step.step_idx, o.id,
                getattr(o, "app", None), _physical,
            )
            per_outcome_trace.append(rec)
            continue
        # Already-verified outcomes still go to the LLM so it can detect
        # invalidation. Pending/reverted with verify spec try deterministic
        # — except a11y_match, which by design routes to the LLM (the
        # original v1 KeyNode flow that compares before/after a11y +
        # screenshot). Substring-matching a11y rows on a single frame
        # would catch transient cues and miss state transitions; only
        # file_grep / url_match are reliably deterministic.
        # A1: outcome's own DONE-rejection counter drives the fallback
        # to LLM judge. Was previously read off a separate
        # OutcomeStateCache; now lives on the Outcome itself.
        _fall_back_after_rejections = o.should_fall_back_to_llm()
        # Route to the LLM only when deterministic verification cannot
        # apply: no verify spec, a11y_match (transient UI requires the
        # before/after a11y+screenshot comparison the v1 LLM flow does),
        # or the K-fallback (the deterministic verifier persistently
        # disagrees with DONE rejections).
        #
        # NOTE on already-verified outcomes: previously they ALWAYS went
        # to LLM "to detect invalidation". But deterministic kinds
        # (file_grep / url_match / shell_command / calc_verify /
        # impress_verify) re-read live VM state every turn — if the
        # user un-did the action (file deleted, dconf flipped, URL
        # navigated, document mutated), verdict drops to False and we
        # can emit outcome_invalidated directly. The LLM-based
        # invalidation check on top of a deterministic-verifiable
        # outcome is actively harmful: terminal scroll history / popup
        # leftovers can show stale error rows that fool the LLM into
        # invalidating an outcome the live deterministic verify still
        # confirms.
        _must_use_llm = (
            verify is None
            or verify.kind == "a11y_match"
            or _fall_back_after_rejections
        )
        if _must_use_llm:
            needs_llm.append(o)
            rec["took_llm_path"] = True
            if _fall_back_after_rejections:
                k = o.get_done_rejection_count()
                rec["deterministic_trace"] = {
                    "reason": (
                        f"deterministic verifier suspected wrong "
                        f"({k} DONE rejections without flip); falling "
                        f"back to LLM judge"
                    ),
                }
                logger.info(
                    "[KeyNode-Fallback] step=%d outcome=%s rejections=%d "
                    "→ LLM judge takes over",
                    current_step.step_idx, o.id, k,
                )
            else:
                rec["deterministic_trace"] = (
                    {"reason": "no verify spec on outcome"}
                    if verify is None
                    else {"reason": "a11y_match always uses LLM (v1 flow)"}
                )
            per_outcome_trace.append(rec)
            continue
        try:
            verdict, trace = verify_with_trace(
                verify, env=env, a11y_text=a11y_text, instruction=instruction,
                slots=(agent_state or {}).get("_ledger_slots"),
            )
        except Exception as e:
            logger.warning("[KeyNodeDetector] verifier crashed for %s: %s",
                           o.id, e)
            verdict, trace = None, {"verdict_reason": f"crash: {e}"}
        # A1: persist the verifier trace directly onto the Outcome so
        # the planner's force_replan prompt can render concrete
        # "[Verifier Feedback]" diagnostics (patterns checked,
        # per-pattern hits, file size, file changed since baseline).
        # Must happen on EVERY verify run — not just satisfied — because
        # the planner needs to see the failed run details, not only
        # success.
        try:
            if isinstance(trace, dict):
                o.record_verifier_trace(trace)
        except Exception as e:
            logger.info("[KeyNodeDetector] record_verifier_trace failed: %s", e)

        # In-step self-healing loop: when a PENDING outcome's verifier
        # returns False AND the diagnoser classifies the failure as
        # ``verifier_mismatch``, re-author the spec and re-run the
        # verifier within the SAME step — actor never sees the broken
        # spec. Up to ``INSTEP_REAUTHOR_MAX_ROUNDS`` cycles; shares
        # A1: budget tracked on outcome.patch_applied.
        if (
            cur_state != OUTCOME_VERIFIED
            and verdict is False
            and call_llm is not None
        ):
            verify, verdict, trace, rounds_log = _instep_reauthor_loop(
                outcome=o,
                current_verify=verify,
                current_trace=trace,
                env=env,
                a11y_text=a11y_text,
                instruction=instruction,
                call_llm=call_llm,
                model=model,
                agent_state=agent_state,
                current_step=current_step,
            )
            # Persist per-round details into the step's debug dump so
            # each spec rewrite + re-verify is replayable offline.
            if rounds_log:
                rec["instep_reauthor_rounds"] = rounds_log
                rec["instep_reauthor_patches_after"] = o.patch_applied

        rec["deterministic_verdict"] = verdict
        rec["deterministic_trace"] = trace

        # ─── A2 (silent observer): build Evidence for this verify run ───
        # We append once per concrete (True/False) verdict. None verdicts
        # route to the LLM path; the LLM path's KeyNode is recorded in
        # the timeline but we don't synthesize a fake "verifier ran"
        # Evidence for it (no trace to capture). triggering_keynode is
        # left None at A2; A3+ may patch it from the emitted KeyNode.
        if verdict in (True, False):
            try:
                ev = _build_evidence(
                    outcome=o,
                    verdict=verdict,
                    trace=trace,
                    triggering_keynode=None,
                    step_idx=current_step.step_idx,
                    strategy_text=(agent_state or {}).get("_current_strategy_text"),
                    bound_slots=(agent_state or {}).get("_ledger_slots"),
                )
                o.evidence.append(ev)
            except Exception as e:
                logger.info("[KeyNodeDetector] evidence build failed: %s", e)

        if cur_state == OUTCOME_VERIFIED:
            # Already verified — looking for invalidation only.
            # - verdict=True  → state still holds; no transition, emit nothing
            # - verdict=False → state was un-done; emit outcome_invalidated
            # - verdict=None  → ambiguous (resolved_path gone, crash); fall
            #                   back to LLM judge for invalidation reasoning
            if verdict is False:
                kn = KeyNode(
                    kind=KN_OUTCOME_INVALIDATED,
                    target=o.id,
                    evidence=(
                        f"[{verify.kind}] live re-verify returned False; "
                        f"underlying state was reversed"
                    )[:240],
                    confidence=CONF_HIGH,
                    detected_at_step=current_step.step_idx,
                    source="deterministic",
                )
                deterministic_kns.append(kn)
                rec["emitted_keynode"] = {
                    "kind": kn.kind, "target": kn.target,
                    "evidence": kn.evidence, "confidence": kn.confidence,
                }
                logger.info(
                    "[KeyNode-Deterministic] step=%d %s INVALIDATED via %s "
                    "(live re-verify now False)",
                    current_step.step_idx, o.id, verify.kind,
                )
            elif verdict is True:
                logger.info(
                    "[KeyNode-Deterministic] step=%d %s still verified "
                    "via %s (live re-verify True; no invalidation)",
                    current_step.step_idx, o.id, verify.kind,
                )
            else:
                # verdict=None → fall back to LLM to disambiguate
                needs_llm.append(o)
                rec["took_llm_path"] = True
            per_outcome_trace.append(rec)
            continue

        # cur_state is pending or reverted — looking for satisfaction.
        if verdict is True:
            ev = _format_verify_evidence(verify)
            kn = KeyNode(
                kind=KN_OUTCOME_SATISFIED,
                target=o.id,
                evidence=ev,
                confidence=CONF_HIGH,
                detected_at_step=current_step.step_idx,
                source="deterministic",
            )
            deterministic_kns.append(kn)
            rec["emitted_keynode"] = {
                "kind": kn.kind, "target": kn.target,
                "evidence": kn.evidence, "confidence": kn.confidence,
            }
            logger.info(
                "[KeyNode-Deterministic] step=%d %s satisfied via %s "
                "(no LLM call needed)",
                current_step.step_idx, o.id, verify.kind,
            )
        elif verdict is False:
            logger.info(
                "[KeyNode-Deterministic] step=%d %s NOT satisfied via %s "
                "(skipping LLM)", current_step.step_idx, o.id, verify.kind,
            )
        else:
            # None — uncertain; let the LLM decide.
            needs_llm.append(o)
            rec["took_llm_path"] = True
        per_outcome_trace.append(rec)

    # ─── Phase 2: LLM call for outcomes still needing inference ───
    llm_messages: Optional[List[Dict[str, Any]]] = None
    raw: Optional[str] = None
    if needs_llm:
        sub_states = {o.id: current_outcome_states.get(o.id, OUTCOME_PENDING)
                      for o in needs_llm}
        llm_messages = _build_messages(
            prev_step=prev_step,
            current_step=current_step,
            required_outcomes=needs_llm,
            current_outcome_states=sub_states,
            actor_action_summary=actor_action_summary,
            a11y_delta_block=a11y_delta_block,
            snapshot=snapshot,
            doc_inspect_block=doc_inspect_block,
            earlier_steps=earlier_steps,
        )
        try:
            raw = call_llm(
                {"model": model, "messages": llm_messages,
                 "max_tokens": max_tokens, "temperature": 0.0, "top_p": 0.9},
                model,
            )
        except Exception as e:
            logger.warning("[KeyNodeDetector] LLM call failed: %s", e)
            raw = None

    llm_kns: List[KeyNode] = []
    if needs_llm and raw:
        sub_states = {o.id: current_outcome_states.get(o.id, OUTCOME_PENDING)
                      for o in needs_llm}
        llm_kns = _parse_response(raw, current_step.step_idx, needs_llm,
                                   sub_states)

        # ─── Hard gate: unresolved-slot block on LLM judge satisfactions ───
        # The LLM judge looks at a11y/screenshot and can hallucinate
        # satisfaction for an outcome whose verify spec embeds a slot
        # reference that's still UNBOUND — e.g. an outcome
        # "the extracted URL is loaded" with verify ``${target_url}``,
        # judged satisfied because there happens to be *some* URL
        # visible on screen. Block that here: KN_OUTCOME_SATISFIED is
        # only credible when every slot the verify spec mentions has
        # been bound by the perceiver. Otherwise demote to "uncertain"
        # and leave the outcome pending — the planner will keep pushing
        # the actor to expose the value.
        from mm_agents.structagent.ledger.core.ledger import find_slot_refs
        slots = (agent_state or {}).get("_ledger_slots") or {}
        outcomes_by_id = {o.id: o for o in needs_llm}
        filtered: List[KeyNode] = []
        for kn in llm_kns:
            if kn.kind != KN_OUTCOME_SATISFIED:
                filtered.append(kn)
                continue
            o = outcomes_by_id.get(kn.target)
            spec = getattr(o, "verify", None) if o else None
            unresolved: List[str] = []
            if spec is not None:
                # Scan every string-valued spec field for ${name} refs.
                from dataclasses import asdict
                try:
                    spec_blob = asdict(spec)
                except Exception:
                    spec_blob = {}
                import json as _json
                refs = find_slot_refs(_json.dumps(spec_blob, default=str))
                unresolved = [r for r in refs if (slots.get(r) is None
                              or getattr(slots.get(r), "value", None) is None)]
            if unresolved:
                logger.info(
                    "[KeyNode-Gate] step=%d %s LLM-satisfied REJECTED — "
                    "verify references unbound slot(s) %s; outcome stays "
                    "pending until perceiver binds them",
                    current_step.step_idx, kn.target, unresolved,
                )
                # Mark in trace for offline debug
                for rec in per_outcome_trace:
                    if rec["outcome_id"] == kn.target:
                        rec["llm_satisfied_rejected_unbound"] = unresolved
                        break
                continue
            filtered.append(kn)
        llm_kns = filtered

        # annotate the trace records with what the LLM emitted
        emitted_by_target = {k.target: k for k in llm_kns}
        for rec in per_outcome_trace:
            if rec["took_llm_path"] and rec["outcome_id"] in emitted_by_target:
                kn = emitted_by_target[rec["outcome_id"]]
                rec["emitted_keynode"] = {
                    "kind": kn.kind, "target": kn.target,
                    "evidence": kn.evidence, "confidence": kn.confidence,
                }

    # ─── Debug dumps (per-outcome verdict + LLM artifacts) ───
    key_node_debug.dump_keynode_verdicts(
        results_dir=results_dir,
        step_idx=dbg_step,
        per_outcome=per_outcome_trace,
    )
    if llm_messages is not None or raw:
        key_node_debug.dump_keynode_llm(
            results_dir=results_dir,
            step_idx=dbg_step,
            messages=llm_messages,
            raw_response=raw,
        )

    return deterministic_kns + llm_kns


def _instep_reauthor_loop(
    *,
    outcome,
    current_verify,
    current_trace,
    env,
    a11y_text,
    instruction,
    call_llm,
    model,
    agent_state,
    current_step,
):
    """One pending outcome's deterministic verifier returned False on
    this step. Try to heal the spec WITHIN this step rather than
    surfacing a possibly-broken verdict to the planner.

    Loop body: diagnose(trace) → if cause=='verifier_mismatch' AND
    patch budget left, reauthor(spec, trace, diag) → re-run verifier.
    Stops when verdict=True, cause=='planner_action', reauthor fails,
    or ``INSTEP_REAUTHOR_MAX_ROUNDS`` reached.

    Mutates ``outcome.verify`` and ``cache.patch_applied[outcome.id]``
    in place. Returns ``(verify_spec, verdict, trace, rounds_log)``.

    ``rounds_log`` is a list of per-round dicts capturing each
    (old spec → diagnosis → cause → new spec → re-verify result)
    transition, intended for the debug dump so a human can replay why
    the verifier converged (or gave up).
    """
    from dataclasses import asdict
    from mm_agents.structagent.ledger.core.timeline import (
        INSTEP_REAUTHOR_MAX_ROUNDS,
        VERIFIER_REAUTHOR_MAX_PATCHES,
        _diagnose_outcome,
    )
    from mm_agents.structagent.core.verifier.verifiers import verify_with_trace
    from mm_agents.structagent.ledger.core.initializer import reauthor_outcome_verifier

    def _spec_to_dict(s):
        if s is None:
            return None
        try:
            return asdict(s)
        except Exception:
            return {"kind": getattr(s, "kind", "?"), "repr": str(s)[:200]}

    verify = current_verify
    verdict = False
    trace = current_trace
    rounds_log: List[Dict[str, Any]] = []

    # Step-0 gate. At step 0 the actor has not had a turn yet, so any
    # outcome that requires actor work will naturally verify=False —
    # this is the ``planner_action`` cause (actor hasn't acted), NOT
    # ``verifier_mismatch`` (spec is wrong). The diagnoser empirically
    # mis-classifies step-0 verdicts as verifier_mismatch (the trace
    # has no "what the actor wrote" signal to anchor template (a) on),
    # which triggers up to INSTEP_REAUTHOR_MAX_ROUNDS wasted LLM calls
    # at task start producing equivalent specs that all still fail.
    # Skip reauthor until at least one actor turn has happened.
    step_idx = getattr(current_step, "step_idx", 0)
    if step_idx == 0:
        logger.info(
            "[InStep-Reauthor] %s step=0 — skipping reauthor "
            "(actor has not had a turn yet; verify=False is the "
            "expected pre-action state, not a spec bug)",
            outcome.id,
        )
        return verify, verdict, trace, rounds_log

    # initial_obs slot for reauthor — pass the current step's screenshot
    # since spec rewriting cares about NOW, not task start. Reauthor
    # tolerates None.
    initial_obs = {
        "screenshot": getattr(current_step, "screenshot_bytes", None),
    }
    domain = (agent_state or {}).get("domain")

    for rnd in range(INSTEP_REAUTHOR_MAX_ROUNDS):
        round_rec: Dict[str, Any] = {
            "round_idx": rnd,
            "old_spec": _spec_to_dict(verify),
            "diagnosis": None,
            "cause": None,
            "new_spec": None,
            "verdict_after": None,
            "trace_after": None,
            "stop_reason": None,
        }

        # Budget guard — shared with post-step apply_reauthor_pass.
        if outcome.patch_applied >= VERIFIER_REAUTHOR_MAX_PATCHES:
            logger.info(
                "[InStep-Reauthor] %s round=%d patch budget exhausted (%d) — stopping",
                outcome.id, rnd, outcome.patch_applied,
            )
            round_rec["stop_reason"] = "patch_budget_exhausted"
            rounds_log.append(round_rec)
            break

        # Diagnose (uses trace signature cache — _trace_signature was
        # updated to include exit_code + stdout head so different specs
        # produce fresh diagnoses).
        diag_text = _diagnose_outcome(
            outcome, verify, trace,
            call_llm=call_llm, model=model,
        )
        cause = outcome.last_diagnosis_cause or ""
        round_rec["diagnosis"] = (diag_text or "")[:1000]
        round_rec["cause"] = cause or "unknown"
        if cause != "verifier_mismatch":
            logger.info(
                "[InStep-Reauthor] %s round=%d cause=%s — stopping (no rewrite)",
                outcome.id, rnd, cause or "unknown",
            )
            round_rec["stop_reason"] = f"cause={cause or 'unknown'}"
            rounds_log.append(round_rec)
            break

        # Reauthor — produce a new VerifySpec for THIS outcome.
        try:
            new_spec = reauthor_outcome_verifier(
                instruction=instruction,
                outcome=outcome,
                diagnosis_text=diag_text,
                last_trace=trace,
                initial_obs=initial_obs,
                call_llm=call_llm,
                model=model,
                a11y_text=a11y_text,
                active_url=None,
                domain=domain,
            )
        except Exception as e:
            logger.warning(
                "[InStep-Reauthor] %s round=%d reauthor crashed: %s",
                outcome.id, rnd, e,
            )
            round_rec["stop_reason"] = f"reauthor_crashed: {e}"
            rounds_log.append(round_rec)
            break
        if new_spec is None or not new_spec.is_valid():
            logger.info(
                "[InStep-Reauthor] %s round=%d reauthor returned None/invalid — stopping",
                outcome.id, rnd,
            )
            round_rec["new_spec"] = _spec_to_dict(new_spec)
            round_rec["stop_reason"] = "reauthor_none_or_invalid"
            rounds_log.append(round_rec)
            break

        # Commit new spec + bump patch counter (shared budget).
        outcome.verify = new_spec
        outcome.patch_applied = outcome.patch_applied + 1
        verify = new_spec
        round_rec["new_spec"] = _spec_to_dict(new_spec)

        # Re-run verifier with the new spec.
        try:
            verdict, trace = verify_with_trace(
                new_spec, env=env, a11y_text=a11y_text,
                instruction=instruction,
                slots=(agent_state or {}).get("_ledger_slots"),
            )
        except Exception as e:
            logger.warning(
                "[InStep-Reauthor] %s round=%d re-verify crashed: %s",
                outcome.id, rnd, e,
            )
            trace = {"verdict_reason": f"re-verify crash: {e}"}
            verdict = None
            round_rec["verdict_after"] = verdict
            round_rec["trace_after"] = trace
            round_rec["stop_reason"] = f"reverify_crashed: {e}"
            rounds_log.append(round_rec)
            break

        # Re-record trace (overwrites the pre-rewrite one for this step).
        try:
            if isinstance(trace, dict):
                outcome.record_verifier_trace(trace)
        except Exception as e:
            logger.info("[InStep-Reauthor] record_verifier_trace failed: %s", e)

        round_rec["verdict_after"] = verdict
        round_rec["trace_after"] = trace

        logger.info(
            "[InStep-Reauthor] %s round=%d new_kind=%s verdict=%s",
            outcome.id, rnd, new_spec.kind, verdict,
        )
        if verdict is True:
            logger.info(
                "[InStep-Reauthor] %s HEALED at round=%d (kind=%s)",
                outcome.id, rnd, new_spec.kind,
            )
            round_rec["stop_reason"] = "healed"
            rounds_log.append(round_rec)
            break

        # Still failing — keep looping (cause re-evaluated next iteration
        # against the new trace).
        round_rec["stop_reason"] = "continue"
        rounds_log.append(round_rec)
    else:
        # Loop ran to MAX_ROUNDS without break.
        if rounds_log:
            rounds_log[-1]["stop_reason"] = "max_rounds_exhausted"

    return verify, verdict, trace, rounds_log


def _format_verify_evidence(verify) -> str:
    """Compact human-readable evidence string for a deterministic match."""
    if verify.kind == "file_grep":
        path = verify.file_path or "?"
        pat = ", ".join(verify.patterns or [])[:160]
        return f"[file_grep] {path} matched {pat}"[:240]
    if verify.kind == "url_match":
        return f"[url_match] active URL matched /{verify.url_pattern}/"[:240]
    if verify.kind == "a11y_match":
        bits = []
        if verify.tag: bits.append(f"tag~{verify.tag}")
        if verify.text_contains: bits.append(f"text contains {verify.text_contains}")
        if verify.state_contains: bits.append(f"state contains {verify.state_contains}")
        return f"[a11y_match] {'; '.join(bits)}"[:240]
    return f"[{verify.kind}] deterministic verify passed"


# --------------------------------------------------------------------------- #
# Message construction
# --------------------------------------------------------------------------- #

_MAX_A11Y_CHARS = 6000


def _truncate_a11y(a11y: Optional[str]) -> str:
    if not a11y:
        return "(no a11y)"
    if len(a11y) <= _MAX_A11Y_CHARS:
        return a11y
    return a11y[:_MAX_A11Y_CHARS] + f"\n…[truncated, {len(a11y)} total chars]"


def _outcomes_table(
    required_outcomes: List[Any],
    current_states: Dict[str, str],
) -> str:
    """Render the per-outcome state table for the KeyNode prompt.

    For ``a11y_match`` outcomes, also emit the dual-channel safety net
    (``visual_must_hold`` / ``visual_must_not_hold``) the ledger
    initializer wrote, so the LLM judges them BOTH on a11y rows AND
    on the screenshot. Failure of any visual clause keeps the outcome
    at its previous state — the LLM must not emit ``outcome_satisfied``
    in that case (see system prompt).
    """
    lines = ["outcome_id | current_state | evidence_hint"]
    lines.append("-" * 60)
    for o in required_outcomes:
        st = current_states.get(o.id, OUTCOME_PENDING)
        lines.append(
            f"{o.id} | {st} | {(o.evidence_hint or '')[:160]}"
        )
        verify = getattr(o, "verify", None)
        if verify is None:
            continue
        # Only a11y_match is gated by visual clauses (per design — file_grep
        # / url_match are deterministic). Other kinds: skip rendering even
        # if the fields happen to be set, to avoid confusing the LLM.
        if getattr(verify, "kind", None) != "a11y_match":
            continue
        must_hold = list(getattr(verify, "visual_must_hold", []) or [])
        must_not = list(getattr(verify, "visual_must_not_hold", []) or [])
        if must_hold or must_not:
            lines.append(
                f"  └ visual gates for `{o.id}` (a11y match alone is "
                f"NOT sufficient — ALL of these must also hold on the "
                f"screenshot before emitting outcome_satisfied):"
            )
            for c in must_hold:
                lines.append(f"      MUST HOLD     : {c}")
            for c in must_not:
                lines.append(f"      MUST NOT HOLD : {c}")
        else:
            # Author left them empty; warn the LLM that this outcome is
            # judged on a11y alone (legacy) so it knows to be conservative.
            lines.append(
                f"  └ visual gates for `{o.id}`: NONE — a11y-only "
                f"judging (be conservative; transient overlay rows can "
                f"create false positives)"
            )
    return "\n".join(lines)


_VERDICT_GLYPH = {"yes": "✓", "no": "✗", "partial": "~", "unknown": "?"}


def _render_perceiver_prior(snapshot: Any) -> str:
    """Render the SubgoalSnapshot as a 'PERCEIVER PRE-VERIFICATION' block
    for the KeyNode detector prompt. Strong prior — KeyNode treats the
    perceiver as having already attempted clause-level verification on
    the same screen, so KeyNode focuses on edge cases / overrides.

    Duck-typed access; returns a graceful empty-ish block on attribute
    failure so KeyNode doesn't crash on a malformed snapshot.
    """
    try:
        lines: List[str] = [
            "[PERCEIVER PRE-VERIFICATION (auxiliary signal — NOT authoritative)]",
            "A focused perceiver attempted to verify the current subgoal's",
            "expected_post_state on THIS same screen + a11y. Its findings are",
            "a HINT, not ground truth — the perceiver can overreach (claim",
            "visual content beyond the subgoal's expected_post_state, infer",
            "from task keywords / filenames rather than from the screenshot).",
            "Independently verify each `visual_must_hold` / `visual_must_not_hold`",
            "clause of every outcome against the CURRENT screenshot yourself",
            "before accepting any perceiver-suggested transition.",
            "",
        ]
        sg = getattr(snapshot, "subgoal_text", "") or ""
        ep = getattr(snapshot, "expected_post_state", "") or ""
        if sg:
            lines.append(f"  Subgoal:  {sg}")
        if ep:
            lines.append(f"  Expected: {ep}")
        sc = getattr(snapshot, "subgoal_check", None)
        if sc is not None:
            overall = getattr(sc, "overall", "?")
            lines.append(f"  Subgoal-check overall: {str(overall).upper()}")
            for c in (getattr(sc, "clauses", []) or []):
                v = getattr(c, "verdict", "?")
                glyph = _VERDICT_GLYPH.get(str(v), "?")
                lines.append(f"    {glyph} {getattr(c, 'clause', '') or ''}")
                ev = getattr(c, "evidence", "") or ""
                if ev:
                    lines.append(f"        evidence: {ev}")
        controls = getattr(snapshot, "relevant_controls", None) or []
        if controls:
            lines.append("  Relevant controls (subgoal-relevant subset):")
            for ctl in controls:
                label = getattr(ctl, "label", "") or ""
                role = getattr(ctl, "role", "") or "?"
                state = getattr(ctl, "state", []) or []
                bbox = getattr(ctl, "bbox", None)
                state_str = f"[{', '.join(state)}]" if state else ""
                bbox_str = f" bbox={bbox}" if bbox else ""
                lines.append(f"    - {role} {label!r} {state_str}{bbox_str}")
        ts = getattr(snapshot, "transition_summary", "") or ""
        if ts:
            lines.append(f"  Transition (NL diff): {ts}")
        blockers = getattr(snapshot, "unexpected_blockers", None) or []
        if blockers:
            lines.append("  Unexpected blockers:")
            for b in blockers:
                lines.append(f"    • {b}")
        coverage = getattr(snapshot, "coverage", None)
        if coverage:
            lines.append(f"  Perceiver coverage: {coverage}")
        lines.extend([
            "",
            "Hard rules for using this prior:",
            "  • clause verdict='yes' is a HINT — still independently verify",
            "    the claimed evidence appears in the current a11y / screenshot,",
            "    AND that the evidence is on-topic for the outcome's visual",
            "    gates. Perceiver evidence that mentions content NOT listed in",
            "    the subgoal's expected_post_state (e.g. claims about canvas",
            "    contents when the subgoal was only 'app opens') is overreach",
            "    — IGNORE it.",
            "  • clause verdict='no' → DO NOT emit outcome_satisfied for",
            "    that outcome this turn (the perceiver explicitly observed",
            "    the post-state did NOT happen).",
            "  • clause verdict='unknown' OR coverage='partial' → be",
            "    conservative; emit outcome_satisfied ONLY with strong",
            "    independent evidence on the screenshot.",
            "  • unexpected_blockers present → do NOT emit outcome_satisfied",
            "    for any outcome that requires interaction with the blocked",
            "    region (modal/popup obscures the target).",
        ])
        return "\n".join(lines)
    except Exception as e:
        logger.warning("[KeyNode] perceiver-prior render failed: %s", e)
        return "[PERCEIVER PRE-VERIFICATION] (render failed; ignoring prior)"


def _build_messages(
    *,
    prev_step: Optional[StepRecord],
    current_step: StepRecord,
    required_outcomes: List[Any],
    current_outcome_states: Dict[str, str],
    actor_action_summary: str,
    a11y_delta_block: Optional[str],
    snapshot: Optional[Any] = None,           # SubgoalSnapshot — perceiver pre-distilled
    doc_inspect_block: Optional[str] = None,  # office SP/DP block (authoritative for structural props)
    earlier_steps: Optional[List[StepRecord]] = None,  # frames OLDER than prev (oldest→newest)
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT}
    ]

    # User text + images for current observation (always present)
    user_blocks: List[Dict[str, Any]] = []

    user_blocks.append({"type": "text", "text": (
        "[Outcomes (current state machine view)]\n"
        + _outcomes_table(required_outcomes, current_outcome_states)
    )})

    user_blocks.append({"type": "text", "text": (
        f"[Actor action that ran between prev and current observations]\n"
        f"{actor_action_summary or '(unknown)'}"
    )})

    # Earlier observations (screenshots only — extra frames so the judge sees a
    # multi-step workflow, e.g. a print→save-dialog flow, not just prev+current.
    # A short-lived UI state visible only mid-workflow — a save dialog that
    # confirms a file landed before it closes — would otherwise be invisible to
    # a 2-frame judge. a11y omitted here to bound tokens; prev+current carry the
    # full a11y.) Oldest first so the sequence reads chronologically.
    for _es in (earlier_steps or []):
        if getattr(_es, "screenshot_b64", None):
            user_blocks.append({"type": "text", "text": (
                f"[EARLIER screenshot — step {_es.step_idx} "
                f"(before PREV; for workflow context)]")})
            user_blocks.append({"type": "image_url", "image_url": {
                "url": _b64_to_url(_es.screenshot_b64)}})

    # Previous observation
    if prev_step is not None:
        if prev_step.screenshot_b64:
            user_blocks.append({"type": "text",
                                "text": "[PREV screenshot]"})
            user_blocks.append({"type": "image_url", "image_url": {
                "url": _b64_to_url(prev_step.screenshot_b64)
            }})
        user_blocks.append({"type": "text", "text": (
            f"[PREV a11y tree (step {prev_step.step_idx}); URL: {prev_step.url or '?'}]\n"
            + _truncate_a11y(prev_step.a11y)
        )})
    else:
        user_blocks.append({"type": "text",
                            "text": "[no previous observation — first turn]"})

    # Current observation
    if current_step.screenshot_b64:
        user_blocks.append({"type": "text",
                            "text": "[CURRENT screenshot]"})
        user_blocks.append({"type": "image_url", "image_url": {
            "url": _b64_to_url(current_step.screenshot_b64)
        }})
    from mm_agents.structagent.ledger.support.a11y_legend import with_legend
    user_blocks.append({"type": "text", "text": (
        f"[CURRENT a11y tree (step {current_step.step_idx}); URL: {current_step.url or '?'}]\n"
        + with_legend(_truncate_a11y(current_step.a11y),
                       header="a11y rows (tag\\ttext\\tstate):")
    )})

    # Machine a11y delta block (from compute_window_delta) — auxiliary signal
    if a11y_delta_block:
        user_blocks.append({"type": "text", "text": (
            "[Machine-computed a11y delta (already diffed for you)]\n"
            + with_legend(a11y_delta_block,
                          header="delta rows (tag\\ttext\\tstate, prefixed with +/-):")
        )})

    # Perceiver pre-distilled view — strong prior. Treat the perceiver's
    # clause verdicts and relevant_controls as "another LLM has already
    # looked at this same screen and tried to verify". Use as a starting
    # point; override only when you have specific contrary visual /
    # a11y evidence.
    if snapshot is not None:
        user_blocks.append({"type": "text",
                            "text": _render_perceiver_prior(snapshot)})

    # Office structured block (SlidePerceiver / SheetPerceiver /
    # DocPerceiver). For libreoffice_calc / libreoffice_impress /
    # libreoffice_writer this is read DIRECTLY from the UNO document
    # model — every font property, color hex, alignment, cell value,
    # shape size is the ground truth. When checking outcomes that
    # name structural properties (bold / underline / italic / color
    # / size / font name / alignment / cell-value), prefer this block
    # over screenshot pixel guesses. Screenshots show rendered output
    # which can mask subtle font-weight changes; the SP block doesn't.
    if doc_inspect_block:
        user_blocks.append({"type": "text", "text": (
            "[Document structure block — AUTHORITATIVE for font / "
            "color / size / cell values; supersedes screenshot for "
            "structural properties]\n" + doc_inspect_block
        )})

    user_blocks.append({"type": "text", "text": (
        "Now produce the <reasoning> + <patch> as specified in the system prompt."
    )})

    messages.append({"role": "user", "content": user_blocks})
    return messages


def _b64_to_url(b64: str) -> str:
    if b64.startswith("data:image"):
        return b64
    return f"data:image/png;base64,{b64}"


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

_PATCH_RE = re.compile(r"<patch>\s*(\{.*?\})\s*</patch>", re.DOTALL | re.IGNORECASE)


def _parse_response(
    raw: str,
    step_idx: int,
    required_outcomes: List[Any],
    current_outcome_states: Dict[str, str],
) -> List[KeyNode]:
    """Extract KeyNodes from the LLM response. Tolerant of fence/whitespace
    drift; drops any KeyNode that fails validation (missing evidence,
    invalid kind/confidence, target not in outcomes for outcome_* kinds)."""
    if not raw:
        return []
    text = raw.strip()
    m = _PATCH_RE.search(text)
    payload: Optional[Dict[str, Any]] = None
    if m:
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        # Fallback: find first balanced {...} object. If a <patch> opening
        # tag is present (even without a closing tag — common when the
        # response is truncated mid-emit), restrict the brace search to
        # text after that tag so we don't accidentally swallow literal
        # braces inside the <reasoning> block (e.g. `{outcome_id}` notes).
        search_from = 0
        patch_open = re.search(r"<patch>", text, re.IGNORECASE)
        if patch_open is not None:
            search_from = patch_open.end()
        first = text.find("{", search_from)
        last = text.rfind("}")
        if first != -1 and last > first:
            try:
                payload = json.loads(text[first:last + 1])
            except json.JSONDecodeError:
                payload = None
    if payload is None or not isinstance(payload, dict):
        # Distinguish truncation (no <patch> opened, no {} present) from
        # real parse failures (malformed JSON) — the former means raise
        # max_tokens, the latter means the model emitted bad JSON.
        has_patch_open = "<patch>" in text.lower()
        has_brace = "{" in text
        if not has_patch_open and not has_brace:
            logger.warning(
                "[KeyNodeDetector] response truncated before reaching "
                "<patch> block (no JSON in output, %d chars). Likely "
                "max_tokens hit during <reasoning>. raw_tail=%r",
                len(text), text[-200:],
            )
        else:
            logger.warning(
                "[KeyNodeDetector] could not parse patch JSON: head=%r tail=%r",
                text[:200], text[-200:],
            )
        return []

    raw_nodes = payload.get("key_nodes") or []
    if isinstance(raw_nodes, dict):
        raw_nodes = [raw_nodes]   # tolerate a single bare key-node object
    if not isinstance(raw_nodes, list):
        return []

    valid_outcome_ids = {o.id for o in required_outcomes}
    out: List[KeyNode] = []
    for nd in raw_nodes:
        if not isinstance(nd, dict):
            continue
        kind = (nd.get("kind") or "").strip().lower()
        target = (nd.get("target") or "").strip()
        evidence = (nd.get("evidence") or "").strip()
        confidence = (nd.get("confidence") or "").strip().lower()

        if kind not in _VALID_KINDS:
            logger.warning("[KeyNodeDetector] dropping kind=%r (not in %s)",
                           kind, _VALID_KINDS)
            continue
        if confidence not in _VALID_CONF:
            # Tolerate floats like "0.9" → coerce
            if confidence.replace(".", "").isdigit():
                f = float(confidence)
                confidence = CONF_HIGH if f >= 0.75 else (CONF_MEDIUM if f >= 0.4 else CONF_LOW)
            else:
                logger.warning("[KeyNodeDetector] dropping confidence=%r", confidence)
                continue
        if not evidence:
            logger.warning("[KeyNodeDetector] dropping kind=%s target=%s — no evidence",
                           kind, target)
            continue
        if not target:
            continue

        # Outcome-kind validation: target must reference a real outcome,
        # and the transition must be coherent with current state.
        if kind in (KN_OUTCOME_SATISFIED, KN_OUTCOME_INVALIDATED):
            if target not in valid_outcome_ids:
                logger.warning("[KeyNodeDetector] dropping outcome target=%r — not in required_outcomes",
                               target)
                continue
            cur = current_outcome_states.get(target, OUTCOME_PENDING)
            if kind == KN_OUTCOME_SATISFIED and cur == OUTCOME_VERIFIED:
                # already verified — silent dedup, do not re-emit
                continue
            if kind == KN_OUTCOME_INVALIDATED and cur != OUTCOME_VERIFIED:
                # cannot invalidate something that wasn't verified
                logger.info("[KeyNodeDetector] dropping invalidated for %r — current state=%r",
                            target, cur)
                continue

        out.append(KeyNode(
            kind=kind,
            target=target[:120],
            evidence=evidence[:240],
            confidence=confidence,
            detected_at_step=step_idx,
            source="llm",          # parsed from the LLM judge's <patch>
        ))

    return out
