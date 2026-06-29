"""Progress Ledger v2 — outcome state derived from Event Timeline.

The ledger is now a thin wrapper around an ``OutcomeStateCache`` (in
``timeline.py``) that the agent feeds with KeyNodes from the event-
timeline detector. Outcome state derives from the cache (verified /
pending / reverted) and supports REVERSION — a previously-verified
outcome can flip back to pending if its evidence is invalidated.

Backward compatibility (v1 fields preserved):
  • ``done: List[DoneEntry]`` is still appended on satisfied so
    legacy tools (trajectory.html replay, old summarizer fallback)
    keep parsing trajectories.
  • ``failed_paths: List[FailedPath]`` unchanged — strategy-level
    failures are not modelled by KeyNodes (kept in the slow
    boundary-summarizer path).
  • ``add_done`` / ``add_failed_path`` / ``can_accept_done_claim`` /
    ``pending`` / ``all_done`` / ``to_prompt_block`` / ``to_dict`` /
    ``from_dict`` keep their v1 signatures so the existing planner
    agent code continues to work without changes.

The new authoritative state lives in ``self.outcome_cache``. Calls to
``add_done`` synthesize a ``KeyNode`` and fold it into the cache so v1
code paths (e.g. legacy summarizer) still update the ground truth.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from mm_agents.structagent.ledger.core.timeline import (
    KeyNode, StepRecord,
    render_outcome_view,
    KN_OUTCOME_SATISFIED, KN_OUTCOME_INVALIDATED, CONF_HIGH,
    OUTCOME_PENDING, OUTCOME_VERIFIED, OUTCOME_REVERTED,
    MAX_REVERT_COUNT, DONE_REJECTED_FALLBACK_THRESHOLD,
)
from mm_agents.structagent.ledger.core.records import Evidence
from mm_agents.structagent.ledger.core.records import Failure
from mm_agents.structagent.ledger.core.records import Fact


# --------------------------------------------------------------------------- #
# Helpers for FailedPath construction (2-layer dead-end tracking)
#
# The agent calls these when:
#   - building ``action_sequence`` from raw actor responses
#     (``abstract_actor_response``)
#   - deciding whether the planner's new <strategy> represents a real
#     strategic change vs just refined wording (``strategy_similar``)
#
# Both helpers live here (next to ``add_failed_path``) so the agent
# doesn't need module-level helpers polluting its file. They're plain
# functions — no class instance required.
# --------------------------------------------------------------------------- #

def abstract_actor_response(resp: Optional[str]) -> str:
    """Reduce a verbose actor response to a one-line summary suitable
    for the ``action_sequence`` field of a FailedPath.

    Actor responses look like:
        "Thought: Plan: click the 'Products' link in the top nav
         Action: click(start_box='(800,200)')"

    The ``Plan:`` line is the actor's own NL summary of what it
    intended this turn — usually the right level of abstraction for
    showing the planner ``what was tried``. Falls back to ``[re-grounded] X``
    for the 2-stage re-grounding response, then to the first non-empty
    line truncated.
    """
    if not resp:
        return "(empty)"
    m = re.search(r"Plan:\s*([^\n]+)", resp)
    if m:
        return m.group(1).strip()[:140]
    m2 = re.search(r"\[2-stage re-grounded\]\s*([^\n]+)", resp)
    if m2:
        return f"[re-grounded] {m2.group(1).strip()[:120]}"
    first_line = resp.split("\n", 1)[0].strip()
    return first_line[:140] if first_line else "(empty)"


# Stopwords for strategy-similarity matching. Drops generic English +
# generic UI verbs that two semantically distinct strategies would both
# carry, so the discriminating tokens (specific UI element names, page
# routes, distinct method names) dominate the Jaccard score.
_STRATEGY_STOPWORDS = frozenset({
    "the", "a", "an", "to", "of", "in", "on", "for", "with", "by", "as", "at",
    "and", "or", "from", "into", "this", "that", "be", "is", "are",
    "use", "via", "through", "find", "get", "go", "navigate", "reach",
    "access", "open", "view", "see", "show", "click", "press", "type",
    "page", "section", "area", "button", "link",
})


def _strategy_signature(text: str) -> set:
    """Token set used for fuzzy strategy comparison."""
    if not text:
        return set()
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in _STRATEGY_STOPWORDS}


# Default Jaccard threshold for the MECHANICAL-fallback path of
# ``strategy_similar`` — used when no LLM judge is supplied. Token-level
# matching is rigid and prone to false-positive switches on simple
# rewording, so callers SHOULD prefer the ``judge=`` path (an LLM
# callable) when available; the agent's ``_commit_strategy_change``
# wires this up. Mechanical mode stays available for offline tests
# and for the safety-net when the LLM path errors out.
STRATEGY_SIMILAR_JACCARD = 0.7


def strategy_similar(
    prior: Optional[str], new: Optional[str],
    *,
    judge: Optional[Any] = None,
    actions_under_prior: Optional[List[str]] = None,
    progress_signal: str = "unknown",
    threshold: float = STRATEGY_SIMILAR_JACCARD,
) -> bool:
    """Whether the planner is still pursuing the prior strategy (SAME)
    vs has abandoned it for a structurally different approach (DIFFERENT).

    Modes:
      - ``judge`` is provided (callable returning bool): use the judge.
        Intended for an LLM-backed check that consumes the prior +
        new strategy AND the actor's progress signal, so it can tell
        "planner reworded / narrowed the same approach" (SAME) apart
        from "planner abandoned the prior approach because it failed"
        (DIFFERENT). The judge call signature is:
            judge(prior, new, actions_under_prior, progress_signal)
        ``actions_under_prior`` is a list of one-line action summaries
        run while ``prior`` was the committed strategy (default []).
        ``progress_signal`` is one of:
            "advanced" — at least one outcome / KeyNode verified
                          while prior was committed (prior approach
                          was producing real forward motion)
            "stuck"    — no outcome progress; actor returned IMPOSSIBLE
                          / repeated the same failed action / no
                          KeyNode fired
            "unknown"  — caller could not determine (judge should err
                          on the SAME side to avoid false dead-ends)
      - ``judge`` is None: fall back to token Jaccard >= ``threshold``.
        Imprecise but dependency-free; used by offline tests where no
        live LLM is available.

    Caller responsibilities:
      - Cache repeated lookups (the agent does this in
        ``_commit_strategy_change``) — this function does not.
      - Handle judge errors. If the judge raises, we treat the result
        as "not similar" to avoid silently swallowing the error.
    """
    if not prior or not new:
        return False
    if judge is not None:
        try:
            return bool(judge(
                prior, new,
                actions_under_prior or [],
                progress_signal,
            ))
        except Exception:
            return False
    sa, sb = _strategy_signature(prior), _strategy_signature(new)
    if not sa or not sb:
        return False
    return (len(sa & sb) / len(sa | sb)) >= threshold


# --------------------------------------------------------------------------- #
# Judge — LLM-backed SAME / DIFFERENT verdict on a prior→new strategy
# transition. Owns the prompt, the parsing, and the cache lookup; the
# caller injects the actual LLM call so this module stays free of any
# vLLM / openai dependency.
# --------------------------------------------------------------------------- #

# How many of the most-recent actor responses we surface to the judge.
# 8 is a balance: enough to show progress vs stuck-loop pattern; not
# so much that the prompt blows past a few-shot window.
_JUDGE_ACTIONS_TAIL = 8


def _judge_signal_explainer(progress_signal: str) -> str:
    return {
        "advanced": (
            "actor was making real forward progress under the prior "
            "strategy (at least one milestone outcome verified)"
        ),
        "stuck": (
            "actor was NOT making outcome progress under the prior "
            "strategy (no milestone verified; likely IMPOSSIBLE / "
            "repeated failed actions / no-op)"
        ),
        "unknown": (
            "progress under the prior strategy is inconclusive — lean "
            "toward SAME unless the new strategy is obviously a "
            "different entry point / interaction pattern"
        ),
    }.get(progress_signal, "progress is inconclusive")


def _build_judge_prompt(
    prior: str, new: str,
    actions_under_prior: List[str], progress_signal: str,
) -> str:
    actions_block = "\n".join(
        f"  - {a[:160]}"
        for a in (actions_under_prior or [])[-_JUDGE_ACTIONS_TAIL:]
    ) or "  (none recorded)"
    return (
        "You are deciding whether a task-planning agent has ABANDONED "
        "its prior strategy (DIFFERENT) or is still pursuing it "
        "(SAME).\n\n"
        "DEFINITIONS:\n\n"
        "  SAME — the planner is continuing or refining the same "
        "high-level approach. The new strategy is one of:\n"
        "    (a) the prior approach reworded\n"
        "    (b) the prior approach narrowed to describe the next "
        "sub-step within the same path (especially when the actor was "
        "visibly advancing along that path)\n"
        "    (c) the prior approach focused on a different aspect of "
        "the same path (same entry point + same interaction pattern)\n\n"
        "  DIFFERENT — the planner has abandoned the prior approach for "
        "a structurally different one. Telltale signs: different UI "
        "entry point, different interaction pattern, different "
        "page/route, AND usually the actor was stuck under the prior "
        "approach (which is why the planner pivoted).\n\n"
        "DECISION RULE: progress signal matters.\n"
        "  - If actor was ADVANCING under prior, the planner almost "
        "certainly did NOT abandon it — verdict is SAME unless the new "
        "strategy is unambiguously a different approach.\n"
        "  - If actor was STUCK under prior and the new strategy "
        "describes a different entry/pattern → verdict is DIFFERENT.\n"
        "  - If progress is unknown, default to SAME (avoid false "
        "dead-ends).\n\n"
        "EXAMPLES:\n\n"
        "Example 1 (SAME — narrowed but advancing):\n"
        "  PRIOR: \"Navigate via top-nav Products link to use the "
        "page-internal search\"\n"
        "  ACTIONS:\n"
        "    - click 'I Consent' button (success)\n"
        "    - click 'Products' link (success — URL changed to "
        "/products)\n"
        "    - click search input (success — input focused)\n"
        "  PROGRESS: advanced\n"
        "  NEW: \"Use the search functionality on the Products page "
        "to query for 'lamp'\"\n"
        "  Verdict: SAME — actor was successfully advancing along the "
        "prior path; the new strategy is just narrowing to the next "
        "sub-step within that same path.\n\n"
        "Example 2 (DIFFERENT — pivoted because stuck):\n"
        "  PRIOR: \"Open Chrome Settings via the three-dot menu\"\n"
        "  ACTIONS:\n"
        "    - click(1905,14) — no UI change\n"
        "    - click(1905,14) retry — no change\n"
        "    - click(1905,14) — IMPOSSIBLE\n"
        "  PROGRESS: stuck\n"
        "  NEW: \"Use direct URL navigation to chrome://settings in the "
        "address bar\"\n"
        "  Verdict: DIFFERENT — three-dot menu approach failed "
        "(IMPOSSIBLE × 3); planner pivoted to address-bar URL approach "
        "(different entry point).\n\n"
        "Example 3 (SAME — reworded, no progress info):\n"
        "  PRIOR: \"Use the catalog site's homepage search\"\n"
        "  ACTIONS: (none recorded)\n"
        "  PROGRESS: unknown\n"
        "  NEW: \"Use the search input on the catalog site's homepage\"\n"
        "  Verdict: SAME — same approach, just reworded.\n\n"
        "NOW CLASSIFY:\n\n"
        f"  PRIOR: {prior}\n"
        f"  ACTIONS:\n{actions_block}\n"
        f"  PROGRESS: {progress_signal} "
        f"({_judge_signal_explainer(progress_signal)})\n"
        f"  NEW: {new}\n\n"
        "Answer with one word only: SAME or DIFFERENT."
    )


def _parse_judge_verdict(response: str) -> bool:
    """LLM returns 'SAME' or 'DIFFERENT'. SAME ⇒ True (no FailedPath);
    anything else ⇒ False (treat as switch / record the prior)."""
    return "SAME" in (response or "").upper()[:30]


def judge_strategy_change(
    *,
    prior: str, new: str,
    actions_under_prior: List[str],
    progress_signal: str,
    llm_call: Any,
    cache: Optional[Dict] = None,
) -> bool:
    """LLM-backed judge: True if planner is still pursuing prior (SAME),
    False if planner has abandoned prior for a structurally different
    approach (DIFFERENT).

    Args:
      prior, new: the two strategy strings being compared.
      actions_under_prior: actor responses run while ``prior`` was the
        committed strategy. The judge uses these as evidence of whether
        prior was making progress.
      progress_signal: "advanced" / "stuck" / "unknown" — derived from
        ledger outcome cache by ``commit_strategy_change``.
      llm_call: callable (prompt: str) -> str. The agent injects this
        so the ledger module stays free of LLM-client dependencies.
      cache: optional dict for memoization. Key is
        (prior, new, progress_signal) so re-judging under a changed
        signal correctly issues a fresh LLM call.

    Behavior:
      - Empty prior / empty new       → False (no comparison possible)
      - prior == new (after strip)    → True (skip LLM)
      - Cache hit                     → return cached
      - LLM call raises               → False (treat as not-similar)
        — agent's ``commit_strategy_change`` will then record prior as
          a FailedPath, which is the safe-side error
    """
    prior_key = (prior or "").strip()[:300]
    new_key = (new or "").strip()[:300]
    if not prior_key or not new_key:
        return False
    if prior_key == new_key:
        return True
    cache_key = (prior_key, new_key, progress_signal)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    prompt = _build_judge_prompt(
        prior_key, new_key, actions_under_prior, progress_signal,
    )
    try:
        resp = llm_call(prompt) or ""
    except Exception:
        return False
    same = _parse_judge_verdict(resp)
    if cache is not None:
        cache[cache_key] = same
    return same


# --------------------------------------------------------------------------- #
# Strategy-change tracker — single entry point for the agent
#
# The agent maintains 3 fields on itself for two-layer FailedPath tracking:
#     current_plan_strategy            : Optional[str]
#     current_strategy_started_at_step : int
#     replans_within_current_strategy  : int
#
# At every plan-commit site (step==0 install + REPLAN swap) the agent
# packages those into a ``StrategyState``, calls ``commit_strategy_change``
# with the new <strategy> from the just-parsed planner response, and
# assigns the returned state back. The function is also responsible for
# recording the strategy-LEVEL FailedPath in the ledger when a switch
# is detected — keeping all "what does it mean for the strategy to
# change" logic in one place rather than scattered in the agent.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StrategyState:
    current_strategy: Optional[str]
    started_at_step: int
    replans_within_strategy: int


def _derive_progress_signal(
    ledger: Optional["ProgressLedger"], span_start: int, step_idx: int,
) -> str:
    """Inspect the ledger's outcome cache to decide whether the prior
    strategy's span saw real forward motion.

    Returns:
      "advanced" — at least one outcome was satisfied within the span
                    [span_start, step_idx]. Concrete evidence the prior
                    approach was producing real progress.
      "stuck"    — no outcome satisfied in the span (and we have a
                    cache to consult).
      "unknown"  — no ledger / no outcome cache available; caller should
                    treat as inconclusive.
    """
    if ledger is None:
        return "unknown"
    # A1: per-outcome last_satisfy_step (was OutcomeStateCache map).
    any_step = False
    for o in getattr(ledger, "required_outcomes", []) or []:
        at = o.last_satisfy_step
        if at is None:
            continue
        any_step = True
        if span_start <= at <= step_idx:
            return "advanced"
    if not any_step:
        return "stuck"
    return "stuck"


def commit_strategy_change(
    *,
    ledger: Optional["ProgressLedger"],
    new_strategy: Optional[str],
    prior: StrategyState,
    actions: List[str],
    current_subgoal: Optional[str],
    step_idx: int,
    judge: Optional[Any] = None,
    strategy_history: Optional[List[Any]] = None,
) -> StrategyState:
    """Apply a fresh plan commit to the agent's strategy-tracking state.

    Branches:
      - Empty ``new_strategy`` (parser hiccup): treat as "same strategy"
        so we don't false-positive a switch — bump the counter.
      - No prior strategy (first plan of the task): install
        ``new_strategy`` and start the span at ``step_idx``.
      - ``strategy_similar(prior, new, judge=judge, ...)`` returns
        True (SAME): same approach, just refined wording or narrowed
        scope while still pursuing the same path — bump the counter.
        The judge sees the prior + new strategy AND the actor's
        progress under prior (``actions_under_prior`` slice of the
        action history + ``progress_signal`` derived from the
        ledger's outcome cache), so it can tell "narrowed but still
        on track" apart from "abandoned because failed".
      - Otherwise (DIFFERENT): switched. Record the OLD strategy as
        a strategy-level FailedPath in ``ledger`` (if non-None),
        then install ``new_strategy`` and reset the counter / span.
    """
    from mm_agents.structagent.ledger.core.exploration import StrategyEvent

    new_clean = (new_strategy or "").strip()
    prior_clean = (prior.current_strategy or "").strip()

    if not new_clean:
        return StrategyState(
            current_strategy=prior.current_strategy,
            started_at_step=prior.started_at_step,
            replans_within_strategy=prior.replans_within_strategy + 1,
        )

    # A4: capture the focus outcome at event time so the curator can
    # attribute resulting FailedPaths per-outcome (target_outcome_id).
    _focus_oid = None
    if ledger is not None:
        try:
            _focus = ledger.next_focus_outcome()
            _focus_oid = _focus.id if _focus is not None else None
        except Exception:
            _focus_oid = None

    if not prior_clean:
        # First plan installed — append an "install" event so the
        # curator sees the start of the strategy chain.
        if strategy_history is not None:
            strategy_history.append(StrategyEvent(
                kind="install",
                step_idx=step_idx,
                new_strategy=new_clean,
                span_started_at_step=step_idx,
                current_subgoal=current_subgoal,
                focus_outcome_id=_focus_oid,
            ))
        return StrategyState(
            current_strategy=new_clean,
            started_at_step=step_idx,
            replans_within_strategy=0,
        )

    actions_under_prior = list(actions[prior.started_at_step:])
    progress_signal = _derive_progress_signal(
        ledger, prior.started_at_step, step_idx,
    )
    action_summaries = [
        abstract_actor_response(a) for a in actions_under_prior
    ][-8:]
    span_steps = step_idx - prior.started_at_step

    if strategy_similar(
        prior_clean, new_clean,
        judge=judge,
        actions_under_prior=actions_under_prior,
        progress_signal=progress_signal,
    ):
        # Same approach reworded — append a "same" event to the history
        # so the curator can see how long the planner has been stuck on
        # this strategy. Do NOT write to ledger.failed_paths here; the
        # curator owns that decision based on full history.
        if strategy_history is not None:
            strategy_history.append(StrategyEvent(
                kind="same",
                step_idx=step_idx,
                prior_strategy=prior_clean,
                new_strategy=new_clean,
                span_started_at_step=prior.started_at_step,
                span_steps=span_steps,
                actions_under_prior=action_summaries,
                progress_signal=progress_signal,
                current_subgoal=current_subgoal,
                focus_outcome_id=_focus_oid,
            ))
        return StrategyState(
            current_strategy=prior.current_strategy,
            started_at_step=prior.started_at_step,
            replans_within_strategy=prior.replans_within_strategy + 1,
        )

    # Strategy switch — record the event into history. The LLM curator
    # decides whether/how this becomes a strategy-level FailedPath
    # (it may merge with an existing entry, drop as actor-flailing,
    # etc.) the next time it runs, post-REPLAN.
    if strategy_history is not None:
        strategy_history.append(StrategyEvent(
            kind="switch",
            step_idx=step_idx,
            prior_strategy=prior_clean,
            new_strategy=new_clean,
            span_started_at_step=prior.started_at_step,
            span_steps=span_steps,
            actions_under_prior=action_summaries,
            progress_signal=progress_signal,
            current_subgoal=current_subgoal,
            focus_outcome_id=_focus_oid,
        ))
    return StrategyState(
        current_strategy=new_clean,
        started_at_step=step_idx,
        replans_within_strategy=0,
    )


# --------------------------------------------------------------------------- #
# Record types — unchanged from v1 (preserved for trajectory.json compat)
# --------------------------------------------------------------------------- #

@dataclass
class VerifySpec:
    """Structured verification directive attached to an Outcome.

    Authored by ``init_ledger`` once at task start, consumed by the
    KeyNode dispatcher every step. Three flat ``kind`` values, no
    nesting — simple to author, simple to execute, hard to write wrong.

    kind="file_grep"
      Read a file from the VM, regex-match patterns against its bytes.
      Optionally require all patterns to occur within ``proximity_lines``
      lines of each other (for list-entry tasks where multiple fields
      must belong to the same JSON object — e.g. a keybinding entry
      with both 'key' and 'command' set correctly).
      Required: ``file_path``, ``patterns``.
      Optional: ``all_must_match`` (default True), ``proximity_lines``.

    kind="url_match"
      Regex-match against the active tab URL extracted from a11y.
      Required: ``url_pattern``.

    kind="a11y_match"
      Find an a11y row whose tag, text and state all match the spec.
      ``text_contains`` and ``state_contains`` are AND'd; substring
      semantics for text, exact-flag semantics for state.
      Required: at least one of ``tag``, ``text_contains``,
      ``state_contains``.

    On any verification with kind="file_grep" where ``file_path`` cannot
    be resolved (ENOENT after fallback search), the dispatcher falls
    back to the legacy LLM KeyNode path — preventing false negatives
    when init_ledger writes a wrong path."""

    kind: str  # "file_grep" | "url_match" | "a11y_match" | "shell_command" | "calc_verify" | "impress_verify"

    # file_grep
    file_path: Optional[str] = None
    patterns: Optional[List[str]] = None
    all_must_match: bool = True
    proximity_lines: Optional[int] = None

    # calc_verify — flat-JSON Calc verification. List of check dicts;
    # each has shape ``{"op": "<cell_value|sheet_exists|pivot_exists|
    # cell_format|sort_order|freeze_pane|duplicate_range_match>", ...kwargs}``.
    # All checks must pass (AND). Framework deterministically builds
    # the underlying python3 -c command from the structured dicts — the
    # init_ledger LLM no longer writes Python source for verify bodies,
    # so the "fake from uno_helper import" / unescaped-quote / missing-
    # PASS-token classes of bugs are eliminated by construction.
    calc_checks: Optional[List[Dict[str, Any]]] = None

    # impress_verify — flat-JSON Impress verification. Same shape as
    # ``calc_checks`` but routed through ``IMPRESS_VERIFY_OPS``.
    impress_checks: Optional[List[Dict[str, Any]]] = None

    # writer_verify — flat-JSON Writer verification. Same shape as
    # ``calc_checks`` but routed through ``WRITER_VERIFY_OPS``.
    writer_checks: Optional[List[Dict[str, Any]]] = None

    # url_match
    url_pattern: Optional[str] = None

    # a11y_match
    tag: Optional[str] = None
    text_contains: Optional[List[str]] = None
    state_contains: Optional[List[str]] = None

    # shell_command — for tasks whose ground truth lives in a CLI tool's
    # stdout (e.g. ``which spotify`` to confirm a binary is on PATH,
    # ``gsettings get ...`` for dconf-backed settings, ``ps -ef | grep`` for
    # process state). The dispatcher executes ``command`` on the VM and
    # checks: (a) exit code matches ``expected_exit_code`` (default 0),
    # AND (b) at least one of ``expected_substring`` matches stdout (case-
    # insensitive substring) AND none of ``forbidden_substring`` appears.
    #
    # ``command`` is an argv list (each element a separate argument), NOT a
    # raw shell string — the executor passes them straight to the VM with
    # NO shell interpretation, so quoting / $-expansion / pipe / redirect
    # do NOT work. Use multi-arg argv form (e.g. ``["gsettings", "get",
    # "org.gnome...", "idle-dim"]``).
    command: Optional[List[str]] = None
    expected_substring: Optional[List[str]] = None
    forbidden_substring: Optional[List[str]] = None
    expected_exit_code: Optional[int] = 0

    # Dual-channel safety net for a11y_match (only consumed by the LLM
    # KeyNode path — file_grep / url_match are deterministic and don't
    # use these fields). Mandatory for a11y_match: init_ledger MUST
    # populate at least one ``visual_must_hold`` clause; ``a11y_match``
    # outcomes that lack it are flagged by the parser and the LLM is
    # told the visual gate is unspecified, which falls back to the
    # legacy a11y-only behavior.
    #
    # ``visual_must_hold``  : list of free-form visual clauses (each a
    #                         single sentence) that MUST hold true on
    #                         the current screenshot for the outcome to
    #                         be marked satisfied. AND-semantics across
    #                         the list. Examples:
    #                            "The Open Folder dialog is no longer
    #                             visible on screen."
    #                            "The project folder appears under the
    #                             EXPLORER tree in the LEFT sidebar of
    #                             VS Code's main window."
    # ``visual_must_not_hold``: list of clauses that MUST be FALSE.
    #                         Same single-sentence form. Examples:
    #                            "Any modal dialog with Open / Cancel
    #                             buttons is currently visible."
    #
    # Failure of any clause keeps the outcome at its previous state
    # (does not emit invalidation) — the simpler logic the user
    # asked for.
    visual_must_hold: Optional[List[str]] = None
    visual_must_not_hold: Optional[List[str]] = None

    def is_valid(self) -> bool:
        """Sanity check the spec is fillable for its kind. Used by
        ``init_ledger``'s parser to drop malformed entries silently
        rather than crash mid-verify."""
        if self.kind == "file_grep":
            return bool(self.file_path) and bool(self.patterns)
        if self.kind == "url_match":
            return bool(self.url_pattern)
        if self.kind == "a11y_match":
            # [FIX a11y_visual_only_valid] visual_must_hold (LLM-judged
            # natural-language clause) is also a valid grounding for an
            # ``a11y_match`` outcome — no static a11y tag/text/state
            # anchor is required when the success criterion is best
            # phrased as a sentence the screenshot must satisfy. Roll
            # back by removing the ``self.visual_must_hold`` branch.
            return any([self.tag, self.text_contains,
                        self.state_contains, self.visual_must_hold])
        if self.kind == "calc_verify":
            if not self.calc_checks or not isinstance(self.calc_checks, list):
                return False
            from mm_agents.structagent.actions.calc.actions import (
                CALC_VERIFY_OPS, validate_calc_verify_check,
            )
            for chk in self.calc_checks:
                if not isinstance(chk, dict):
                    return False
                op = chk.get("op")
                if op not in CALC_VERIFY_OPS:
                    return False
                if not validate_calc_verify_check(chk):
                    return False
            return True
        if self.kind == "impress_verify":
            if not self.impress_checks or not isinstance(self.impress_checks, list):
                return False
            from mm_agents.structagent.actions.impress.actions import (
                IMPRESS_VERIFY_OPS, validate_impress_verify_check,
            )
            for chk in self.impress_checks:
                if not isinstance(chk, dict):
                    return False
                op = chk.get("op")
                if op not in IMPRESS_VERIFY_OPS:
                    return False
                if not validate_impress_verify_check(chk):
                    return False
            return True
        if self.kind == "writer_verify":
            if not self.writer_checks or not isinstance(self.writer_checks, list):
                return False
            from mm_agents.structagent.actions.writer.actions import (
                WRITER_VERIFY_OPS, validate_writer_verify_check,
            )
            for chk in self.writer_checks:
                if not isinstance(chk, dict):
                    return False
                if chk.get("op") not in WRITER_VERIFY_OPS:
                    return False
                if not validate_writer_verify_check(chk):
                    return False
            return True
        if self.kind == "shell_command":
            if not self.command:
                return False
            # Hard whitelist on argv[0]. The init_ledger prompt lists
            # exactly six shapes (which / dpkg / snap / pgrep / test /
            # gsettings / python3); anything else is either a forbidden
            # multi-step pipeline (find / tar / awk / ls / cat / grep /
            # sed / bash …) or an unsourced gsettings/dconf path. Reject
            # so the outcome falls back to verify=None → KeyNode
            # LLM-judge handles it.
            if self.command[0] not in (
                "which", "dpkg", "snap", "pgrep", "test", "gsettings",
                "python3",
            ):
                return False
            # ``test`` produces NO stdout — its verdict is exit-code only,
            # so a substring rule is meaningless. Every other binary needs
            # at least one substring (positive or negative); otherwise the
            # spec passes on exit==0 alone and never discriminates.
            if self.command[0] == "test":
                return True
            # ``python3`` is only valid in the form
            # ``["python3", "-c", "<expr>"]`` — anything else (script
            # paths, modules) lets the actor sneak in arbitrary code.
            # The expression itself must print a PASS token, so we also
            # require at least one expected_substring (typically "PASS").
            if self.command[0] == "python3":
                if len(self.command) != 3 or self.command[1] != "-c":
                    return False
                if not self.expected_substring:
                    return False
                return True
            return bool(self.expected_substring or self.forbidden_substring)
        return False


@dataclass
class Outcome:
    """A milestone state the task evaluator (or the agent reasoning
    about completion) should observe to consider that subgoal done.

    Design rules (enforced via the init_ledger prompt; see
    ``progress_ledger/initializer.py``):

      * Each outcome must be a GATING MILESTONE — a state whose
        regression would cascade and block downstream work. Path
        waypoints (e.g. "search bar visible right now", "consent
        dialog dismissed") are NOT outcomes; they're transient UI
        cues the planner can break down into subgoals.

      * ``evidence_hint`` should anchor on the most DURABLE signal
        available — persisted file/config > URL/cookies > stable
        DOM state > transient UI focus. The lower in this hierarchy,
        the more frequently the KeyNode detector will flap the
        outcome between verified ↔ reverted. Kept as free-text for
        the planner prompt's semantic context (the planner reads
        this to know what target state to drive toward).

      * ``verify`` is the structured machine-executable counterpart of
        ``evidence_hint``. The KeyNode dispatcher prefers ``verify``
        when present (deterministic, no hallucination). When None
        (legacy outcomes) or when ``verify`` cannot be evaluated
        (e.g. file_grep with unresolvable file_path), the dispatcher
        falls back to the legacy LLM-based KeyNode path.

      * ``depends_on`` lists outcome ids that must be satisfied
        before this one can be reasonably attempted. The cache does
        NOT enforce ordering (the KeyNode detector may verify
        outcomes in any sequence) — the field is consumed by
        ``next_focus_outcome`` to drive single-focus planner
        prompting and by the renderer to show structure.
    """
    id: str
    description: str
    evidence_hint: str
    depends_on: List[str] = field(default_factory=list)
    verify: Optional[VerifySpec] = None
    # Multi-app tasks only: the application this outcome is verified in
    # (e.g. "chrome", "libreoffice_calc"). ``None`` for single-domain
    # tasks. Drives the agent's dynamic ``_current_domain`` switching —
    # the domain follows the app of the outcome currently in focus.
    app: Optional[str] = None

    # ---- A1: per-outcome state (was OutcomeStateCache) ---------------- #
    # Authoritative state. PENDING → VERIFIED on outcome_satisfied KeyNode,
    # VERIFIED → REVERTED on outcome_invalidated. apply_keynode() is the
    # only mutator; ProgressLedger.apply_step() drives it for each KN in
    # a fresh StepRecord.
    state: str = OUTCOME_PENDING
    revert_count: int = 0
    done_rejected_count: int = 0
    last_satisfy_step: Optional[int] = None
    last_invalidate_step: Optional[int] = None
    last_verifier_trace: Dict[str, Any] = field(default_factory=dict)
    file_baseline_size: Optional[int] = None
    file_baseline_hash: Optional[str] = None
    last_diagnosis_signature: Optional[str] = None
    last_diagnosis: Optional[str] = None
    last_diagnosis_cause: Optional[str] = None
    verifier_mismatch_strikes: int = 0
    patch_applied: int = 0

    # A8-V0: cached StuckDiagnosis result (see stuck_diagnosis.py).
    # Memoized on the outcome itself; reset to None on outcome_satisfied
    # so a verified-then-reverted outcome gets a fresh diagnosis next
    # stuck cycle. Kept as a plain dict to keep Outcome's asdict
    # serialization simple.
    last_stuck_diagnosis: Optional[Dict[str, Any]] = None
    last_stuck_diagnosis_signature: Optional[str] = None

    # ---- A2: per-Outcome Evidence + Failure containers ---------------- #
    # ``evidence`` is appended once per verify_with_trace run by
    # bundle_assembler.build(). A2 is silent-observer: nobody reads
    # it yet. ``failures`` is empty until A4 (the migration moves
    # per-outcome subsets out of ProgressLedger.failed_paths).
    evidence: List[Evidence] = field(default_factory=list)
    failures: List[Failure] = field(default_factory=list)

    # ---- A5: working-memory Facts owned by this Outcome --------------- #
    # ``facts`` holds typed captured values (the structured replacement
    # for the legacy flat ``_notebook`` dict). Populated by note_write
    # / extract_info auto-mirror / perceiver slot binder when those
    # call sites know which outcome they're producing for.
    # ``"ambient"`` sentinel Outcome owns Facts unattributable to a
    # real outcome (environment paths probed at task start, free-form
    # captures). A5 ships *dual write* — Facts are written alongside
    # ``_notebook`` but nothing reads them yet. A6 switches readers.
    facts: Dict[str, Fact] = field(default_factory=dict)

    # ---- A1: convenience state predicates ----------------------------- #

    def is_verified(self) -> bool:
        return self.state == OUTCOME_VERIFIED

    def is_pending(self) -> bool:
        return self.state == OUTCOME_PENDING

    def is_reverted(self) -> bool:
        return self.state == OUTCOME_REVERTED

    def get_done_rejection_count(self) -> int:
        return self.done_rejected_count

    def should_fall_back_to_llm(self, threshold: Optional[int] = None) -> bool:
        """True iff this outcome's DONE-rejection counter has reached the
        threshold — caller should drop deterministic verify and hand the
        outcome to the LLM judge."""
        if threshold is None:
            try:
                import os as _os
                threshold = int(_os.environ.get(
                    "DONE_REJECTED_FALLBACK_THRESHOLD",
                    DONE_REJECTED_FALLBACK_THRESHOLD,
                ))
            except (ValueError, TypeError):
                threshold = DONE_REJECTED_FALLBACK_THRESHOLD
        return self.done_rejected_count >= threshold

    def record_verifier_trace(self, trace: Dict[str, Any]) -> None:
        """Stash the most-recent verify_with_trace trace dict. Also seeds
        file_baseline_size / file_baseline_hash on first observation so
        ``file_changed_since_baseline()`` can tell whether the actor's
        edits are reaching disk."""
        if not isinstance(trace, dict):
            return
        self.last_verifier_trace = dict(trace)
        size = trace.get("file_size_bytes")
        if isinstance(size, int) and self.file_baseline_size is None:
            self.file_baseline_size = size
        ch = trace.get("file_content_hash")
        if isinstance(ch, str) and self.file_baseline_hash is None:
            self.file_baseline_hash = ch

    def file_changed_since_baseline(self) -> Optional[bool]:
        """True iff the verifier-target file size or hash differs from
        baseline; False iff confirmed unchanged; None iff baseline /
        current data missing."""
        last = self.last_verifier_trace or {}
        if self.file_baseline_size is None:
            return None
        cur_size = last.get("file_size_bytes")
        if cur_size is None:
            return None
        if self.file_baseline_size != cur_size:
            return True
        if self.file_baseline_hash is not None:
            cur_hash = last.get("file_content_hash")
            if cur_hash is not None:
                return self.file_baseline_hash != cur_hash
        return False

    def apply_keynode(self, kn: "KeyNode", step_idx: int) -> None:
        """Update self.state from a single KeyNode targeting this outcome.
        Called by ProgressLedger.apply_step()."""
        if kn.kind == KN_OUTCOME_SATISFIED:
            self.state = OUTCOME_VERIFIED
            self.last_satisfy_step = step_idx
            # A8-V0: drop any cached stuck diagnosis — if this outcome
            # ever reverts and we get stuck again, the next cycle should
            # diagnose fresh against the new (post-revert) state.
            self.last_stuck_diagnosis = None
            self.last_stuck_diagnosis_signature = None
        elif kn.kind == KN_OUTCOME_INVALIDATED:
            if self.state != OUTCOME_VERIFIED:
                return
            if step_idx <= (self.last_satisfy_step or -1):
                return
            if self.revert_count >= MAX_REVERT_COUNT:
                return  # flap protection
            self.state = OUTCOME_REVERTED
            self.last_invalidate_step = step_idx
            self.revert_count += 1


@dataclass
class DoneEntry:
    """Legacy v1 record. Still appended on outcome_satisfied for
    backward compat with trajectory.html readers and the old
    summarizer's ``add_done`` path. The authoritative state is in
    ``outcome_cache``."""
    outcome_id: str
    verified_at_step: int
    verified_at_subgoal: str
    evidence: str


@dataclass
class FailedPath:
    """A dead-end record at one of two granularities (see ``level``).

    Two-layer model:

      level == "strategy"
        ``path`` is the high-level approach the planner declared in
        its <plan>'s <strategy> tag (e.g. "Use the catalog site's
        main-page search to find a product"). Recorded when the planner
        SWITCHES strategy across REPLANs — i.e. the planner itself
        abandoned the old strategy. Subsequent plans must not
        propose this strategy again.

      level == "action"
        ``path`` is the stuck subgoal text the planner kept emitting
        without progress (e.g. "Click the magnifying glass icon to
        submit"). Recorded by the cadence trigger ONLY when the
        planner kept the same strategy across the stuck REPLANs —
        i.e. the strategy is presumed sound; only the specific
        tactical execution failed. Suggests trying alternative
        tactics within the SAME strategy (different element, keyboard
        shortcut, alt coordinates).

    ``action_sequence`` carries one-line summaries of the actor's
    actual attempts — evidence that the path was tried in earnest.
    Cap at 12 entries to keep the prompt-render bounded.
    """
    path: str
    why: str
    first_observed_at_step: int
    first_observed_at_subgoal: str
    # NEW: which layer this dead-end lives at — see class docstring.
    # Default "action" for back-compat with persisted records that
    # don't carry the field (most legacy entries are subgoal-text-style,
    # i.e. action-level by today's semantics).
    level: str = "action"
    # Provenance / quality of the evidence behind this dead-end:
    #   "auto"     — heuristic trigger (REPLAN cadence, same-coord, etc.)
    #   "planner"  — planner self-claimed via <dead_end_review>/<dead_end_claim>
    #   "keynode"  — derived from KeyNode invalidation causal chain
    # Default "auto" for back-compat with persisted records that don't
    # carry this field.
    evidence_strength: str = "auto"
    # How many times this dead-end has been observed/re-confirmed.
    # Bumped ONLY by ``add_failed_path`` dedup hits (the cadence
    # trigger firing again on the same strategy — i.e. the strategy
    # was actually RE-ATTEMPTED). Retrospective mentions in the
    # planner's <dead_end_review> do NOT bump this — see
    # ``merge_planner_review`` for the rationale.
    observation_count: int = 1
    # Last step at which we saw evidence of this dead-end (any source).
    # Defaults to first_observed_at_step for newly-created entries; no
    # explicit bump needed at construction time.
    last_observed_at_step: int = 0
    # Concrete actor actions attempted under this strategy. Each entry
    # is a one-line summary of an actor invocation (typically
    # ``actor_response_summary`` text such as "click(1885,140)" or
    # "type 'foo'"). Populated by the cadence trigger from
    # ``self.actions[-K:]`` at fire time, and extended on subsequent
    # dedup hits with new distinct entries (capped at 12). Empty for
    # planner-claimed entries (``ref:new``) since the planner doesn't
    # know the actor's exact actions.
    action_sequence: List[str] = field(default_factory=list)
    # Provenance marker — when True, this dead-end was authored by the
    # strong-model rescue agent (not curator). Curator's drop_existing
    # logic must NOT re-open a strong-model-authored path: the rescue
    # call had full context and deliberately declared it dead.
    strong_model_origin: bool = False

    # A4 (per-outcome failures): which Outcome this dead-end was failing
    # against. None means uncategorized (legacy entries, ambient
    # actor-flailing failures unattributable to a single outcome).
    # Populated by the curator via the focus_outcome_id annotation on
    # the originating StrategyEvent (see commit_strategy_change in
    # this file). Read by render_outcome_view to filter the failures
    # block per-focus-outcome — i.e. the planner replanning O3 only
    # sees O3's dead-ends, not O1/O2 noise.
    target_outcome_id: Optional[str] = None


@dataclass
class InitialContext:
    open_tabs: List[str] = field(default_factory=list)
    active_url: Optional[str] = None
    active_app: Optional[str] = None
    window_title: Optional[str] = None
    # Filtered VM-environment snapshot produced once per task by
    # ``probes.probe_environment`` then digested by
    # ``perceiver.perceive_environment``. Plain text, ≤ ~1.5KB.
    # Lists task-relevant identifiers (binaries, config files,
    # GSettings schemas, processes) directly observed on the VM —
    # the agent never invents them. None when the probe was unavailable
    # or returned empty.
    environment_probe: Optional[str] = None
    # Disk-content probe of files surfaced in [Possibly useful files]
    # — produced once at task start by ``file_probes.probe_useful_files``.
    # Spreadsheets go through the SheetPerceiver renderer, docx / pptx
    # arrive as plain-text summaries (soffice --convert-to txt). Used
    # by init_ledger ONLY (the live UNO perceivers take over on later
    # turns); not rendered into per-turn planner prompts.
    useful_file_contents: Optional[str] = None


# --------------------------------------------------------------------------- #
# Slot — placeholder for a value the init_ledger LLM doesn't know yet
# --------------------------------------------------------------------------- #

# ``${slot_name}`` references inside verify-spec strings. Names are
# restricted to Python-identifier form so the parser can't be fooled
# into capturing regex metacharacters as a slot name when the spec is
# itself a regex (url_pattern, file_grep patterns).
_SLOT_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class VerifySpecSlot:
    """A typed placeholder for a value the init_ledger LLM cannot know yet
    (URL extracted from a page, filename chosen at runtime, OCR-read
    number, ...). Outcomes reference slots with ``${slot_name}`` inside
    their verify spec; the verifier substitutes bound values before
    running, and short-circuits to UNCERTAIN (None) when any reference
    is still unbound — preventing the deterministic path from rubber-
    stamping a vacuous pattern like ``.*``.

    Bind-once: ``value`` is set exactly once, by the perceiver (or any
    other authoritative resolver) the first time evidence is visible.
    Subsequent observations do NOT overwrite — this avoids active-URL
    drift turning into spec drift.
    """
    name: str
    type: str = "text"   # url | text | path | numeric | cell_ref — hint only
    description: str = ""
    source_hint: str = ""
    value: Optional[str] = None
    bound_at_step: Optional[int] = None
    bound_evidence: Optional[str] = None


def find_slot_refs(s: Optional[str]) -> List[str]:
    """All ``${slot_name}`` references in ``s`` (no dedup, order preserved)."""
    if not s:
        return []
    return _SLOT_REF_RE.findall(s)


def substitute_slots(
    s: Optional[str], slots: Dict[str, "VerifySpecSlot"], *, regex_escape: bool = False
) -> Tuple[Optional[str], List[str]]:
    """Return ``(substituted, unresolved)``. Each ``${name}`` whose slot
    has a non-None ``value`` is replaced by that value (optionally
    ``re.escape``'d, for spots where the substituted string is itself a
    regex — url_pattern, file_grep patterns). Unresolved refs are left
    intact in the output and accumulated in ``unresolved``."""
    if not s:
        return s, []
    unresolved: List[str] = []

    def _repl(m: "re.Match[str]") -> str:
        name = m.group(1)
        slot = slots.get(name)
        if slot is None or slot.value is None:
            if name not in unresolved:
                unresolved.append(name)
            return m.group(0)
        return re.escape(slot.value) if regex_escape else slot.value

    return _SLOT_REF_RE.sub(_repl, s), unresolved


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #

@dataclass
class ProgressLedger:
    initial_context: InitialContext
    required_outcomes: List[Outcome]                       # mutable via audit edits
    done: List[DoneEntry] = field(default_factory=list)    # legacy / append-only
    failed_paths: List[FailedPath] = field(default_factory=list)
    # Strategy texts the curator marked as ``strategy_outcome_satisfied`` —
    # high-level approaches whose primary outcome(s) are now [verified].
    # Rendered in the planner prompt as a "Completed Strategies" block so
    # the planner doesn't re-propose them on REPLAN. Each entry is a dict
    # ``{"strategy": <str>, "satisfied_outcomes": [<id>, ...]}``.
    completed_strategies: List[Dict[str, Any]] = field(default_factory=list)
    # Task-start screenshot, base64-encoded PNG. Populated by the
    # ledger initializer (see initializer.py) and re-used by the
    # DONE-time auditor for before/after comparison. Excluded from
    # ``to_dict`` because it's >100KB and would balloon traj.jsonl.
    initial_screenshot_b64: Optional[str] = None
    # Multi-app tasks only: the PhaseBoard (app-phase decomposition +
    # cross-app handoff). ``None`` for single-app tasks — every phase
    # call site is gated so single-domain behaviour is unchanged.
    # Built by init_ledger from the LLM's ``phases`` array; the agent
    # reads it off the ledger and drives phase transitions.
    phase_board: Optional[Any] = None

    # Slot table — placeholders the init LLM emitted when it didn't
    # know a concrete value (see ``Slot`` above). Keyed by slot name.
    # Populated at init time and bound incrementally by the perceiver
    # as observations expose concrete values. Empty for tasks whose
    # verify specs are fully concrete.
    slots: Dict[str, VerifySpecSlot] = field(default_factory=dict)

    # A5: task-wide working-memory Facts that don't belong to any
    # specific Outcome (env paths probed at task start, free-form
    # note_write before a specific outcome is in focus). Each Fact
    # here carries ``source_outcome_id="global"``. ``working_memory()``
    # returns these merged with per-outcome facts so the planner sees
    # one flat view. Not gated by any outcome state — global facts
    # never go stale.
    global_facts: Dict[str, Fact] = field(default_factory=dict)

    # ---- helpers ------------------------------------------------------------- #

    def _all_outcome_ids(self) -> List[str]:
        return [o.id for o in self.required_outcomes]

    def get(self, oid: str) -> Optional[Outcome]:
        """Look up an Outcome by id. O(n) — fine for n < ~20."""
        for o in self.required_outcomes:
            if o.id == oid:
                return o
        return None

    # ---- A5: working-memory Fact helpers ---------------------------------- #

    # Marker used in ``Fact.source_outcome_id`` when the value can't be
    # tied to any specific Outcome. Stored in ``ledger.global_facts``,
    # not on any Outcome.
    GLOBAL_FACT_OWNER = "global"

    def write_fact(
        self,
        name: str,
        value: Any,
        *,
        owning_outcome: Optional[str] = None,
        step_idx: int = 0,
        type_tag: Optional[str] = None,
    ) -> Fact:
        """Append/overwrite one Fact, routed by ``owning_outcome``:

          • matches a real Outcome id  → ``outcome.facts[name]`` (per
            outcome; goes stale when that outcome REVERTS)
          • None or unknown id         → ``self.global_facts[name]``
            (task-wide, never goes stale)

        Returns the stored Fact so the caller can also mirror into
        legacy ``_notebook`` (A5 dual-write phase).
        """
        target_outcome = None
        if owning_outcome is not None and owning_outcome != self.GLOBAL_FACT_OWNER:
            target_outcome = self.get(owning_outcome)
        # Cheap type inference if caller didn't tag.
        if type_tag is None:
            if isinstance(value, bool):
                type_tag = "bool"
            elif isinstance(value, int):
                type_tag = "int"
            elif isinstance(value, float):
                type_tag = "float"
            elif isinstance(value, str):
                type_tag = "str"
            elif isinstance(value, list):
                type_tag = "list[str]" if all(isinstance(x, str) for x in value) else "json"
            else:
                type_tag = "json"
        owner_tag = target_outcome.id if target_outcome is not None else self.GLOBAL_FACT_OWNER
        fact = Fact(
            name=name,
            value=value,
            type=type_tag,
            source_outcome_id=owner_tag,
            source_step=step_idx,
            last_verified_at_step=step_idx,
        )
        if target_outcome is not None:
            target_outcome.facts[name] = fact
        else:
            self.global_facts[name] = fact
        return fact

    def working_memory(self) -> Dict[str, Fact]:
        """Derived flat view of all currently-live Facts. Skips per-
        outcome facts whose source Outcome is REVERTED (liveness rule
        in fact.py). ``global_facts`` are always included since they
        belong to no specific outcome. Per-outcome entries take
        precedence on name collision (real evidence beats task-wide
        scratchpad).
        """
        out: Dict[str, Fact] = dict(self.global_facts)
        for o in self.required_outcomes:
            if o.state == OUTCOME_REVERTED:
                continue
            for name, fact in (o.facts or {}).items():
                out[name] = fact
        return out

    def failures_for(self, oid: str) -> List[FailedPath]:
        """A4: per-outcome failure view. Returns all entries from
        ``self.failed_paths`` whose ``target_outcome_id`` matches
        ``oid``. Cross-cutting entries (target_outcome_id == None) are
        NOT included — callers wanting them should iterate
        ``self.failed_paths`` directly. Used by replan-prompt builders
        that want a clean per-outcome subset.
        """
        return [
            fp for fp in self.failed_paths
            if getattr(fp, "target_outcome_id", None) == oid
        ]

    def state_of(self, oid: str) -> str:
        """State of an outcome by id; PENDING if missing (back-compat for
        callers that may reference an outcome that isn't in the ledger)."""
        o = self.get(oid)
        return o.state if o is not None else OUTCOME_PENDING

    # ---- derived views (read from per-outcome state) ------------------------ #

    def pending(self) -> List[Outcome]:
        """Outcomes that have never been verified (pure pending state).
        Disjoint from ``verified()`` and ``reverted()``."""
        return [o for o in self.required_outcomes if o.is_pending()]

    def verified(self) -> List[Outcome]:
        return [o for o in self.required_outcomes if o.is_verified()]

    def reverted(self) -> List[Outcome]:
        return [o for o in self.required_outcomes if o.is_reverted()]

    def not_verified(self) -> List[Outcome]:
        """Convenience: outcomes that BLOCK DONE — union of pending + reverted."""
        return [o for o in self.required_outcomes if not o.is_verified()]

    def _leaf_outcome_ids(self) -> List[str]:
        """Outcome ids that NO other outcome ``depends_on`` — the task's
        TERMINAL deliverables.

        A verified leaf logically proves its whole dependency chain was
        satisfied (you cannot have the final artifact without having
        done the prerequisites), so the DONE gate need only require
        leaves. Intermediate outcomes are transient MEANS — their
        CURRENT state (e.g. a helper window losing focus → an
        ``*_opened`` outcome flapping to reverted) is irrelevant once
        the agent has moved past them, and gating DONE on them only
        creates spurious rejections.

        Falls back to ALL outcome ids when the graph has no leaves
        (a depends_on cycle, or every outcome is depended-on) so the
        gate never silently accepts nothing."""
        all_ids = self._all_outcome_ids()
        depended: set = set()
        for o in self.required_outcomes:
            for dep in (getattr(o, "depends_on", None) or []):
                depended.add(dep)
        leaves = [oid for oid in all_ids if oid not in depended]
        return leaves if leaves else all_ids

    def all_verified(self, ids: List[str]) -> bool:
        return all((self.get(oid) and self.get(oid).is_verified()) for oid in ids)

    def all_done(self) -> bool:
        return self.all_verified(self._leaf_outcome_ids())

    def can_accept_done_claim(self) -> Tuple[bool, str]:
        """Gate: DONE accepted iff every LEAF outcome (terminal
        deliverable — see ``_leaf_outcome_ids``) is verified. Reverted
        leaves BLOCK DONE; intermediate outcomes are NOT gated on — a
        verified leaf already proves its dependency chain ran.

        On rejection, also bumps per-outcome DONE-rejection counters
        so the KeyNode dispatcher can detect a wrong-from-the-start
        verifier (after K rejections, falls back to LLM judging) and
        the planner's force_replan prompt can show how many times
        each outcome has been the cause."""
        leaf_ids = self._leaf_outcome_ids()
        bad: List[Outcome] = []
        for oid in leaf_ids:
            o = self.get(oid)
            if o is None or not o.is_verified():
                if o is not None:
                    bad.append(o)
                else:
                    # Missing outcome — record nothing; just fail the gate.
                    bad.append(Outcome(id=oid, description="(missing)", evidence_hint=""))
        if not bad:
            return True, ""
        for o in bad:
            o.done_rejected_count += 1
        labels = [f"{o.id}({o.state})" for o in bad]
        return False, f"{len(bad)} outcome(s) not verified: " + ", ".join(labels)

    def next_focus_outcome(self) -> Optional[Outcome]:
        """Return the FIRST outcome that is currently the agent's
        active focus — i.e., it is not yet verified AND all its
        ``depends_on`` outcomes ARE verified.

        Used by the prompt renderer to mark a single ▶ FOCUS ◀
        outcome at a time, driving the planner to attack one
        milestone in isolation rather than juggling all of them. An
        outcome with empty ``depends_on`` is always reachable (its
        dependency precondition holds vacuously), so the first such
        pending outcome wins on a fresh task.

        Returns None when all outcomes are verified (task is done by
        the gate's standard) or when the entire pending set is
        blocked by reverted dependencies (rare; typically means the
        actor undid work and hasn't redone it yet).
        """
        # All terminal deliverables verified → task is DONE-able; no
        # focus. Without this, a flapped intermediate (e.g. an
        # ``*_opened`` outcome that reverted when its window lost
        # focus) would be returned as focus and drag the planner back
        # to redo already-finished prerequisite work.
        _leaf_ids = self._leaf_outcome_ids()
        if _leaf_ids and all(
            (self.get(_lid) and self.get(_lid).is_verified())
            for _lid in _leaf_ids
        ):
            return None
        for o in self.required_outcomes:
            if o.is_verified():
                continue
            # All deps must be currently verified for this outcome to
            # be the active focus. A REVERTED dep means the planner
            # must restore it before this outcome can progress.
            deps_ok = all(
                (self.get(dep) and self.get(dep).is_verified())
                for dep in o.depends_on
            )
            if deps_ok:
                return o
        return None

    # ---- v2 entry point: timeline-driven state update ----------------------- #

    def apply_step_keynodes(self, step: StepRecord) -> None:
        """Primary v2 path: fold a StepRecord's key_nodes into per-outcome
        state. Called by the agent after the KeyNode detector returns.
        Also appends to legacy ``done`` for any outcome that flipped to
        verified, so trajectory readers see a consistent log."""
        prev_states = {o.id: o.state for o in self.required_outcomes}
        for kn in step.key_nodes:
            if not kn.enters_derivation():
                continue
            o = self.get(kn.target)
            if o is None:
                continue
            o.apply_keynode(kn, step.step_idx)

        # Append legacy DoneEntry records for any pending→verified flip.
        done_ids = {d.outcome_id for d in self.done}
        for o in self.required_outcomes:
            if (prev_states.get(o.id) != OUTCOME_VERIFIED
                    and o.is_verified()
                    and o.id not in done_ids):
                ev_text = ""
                for kn in step.key_nodes:
                    if kn.target == o.id and kn.kind == KN_OUTCOME_SATISFIED:
                        ev_text = kn.evidence
                        break
                self.done.append(DoneEntry(
                    outcome_id=o.id,
                    verified_at_step=step.step_idx,
                    verified_at_subgoal="",
                    evidence=(ev_text or "")[:240],
                ))

    # ---- legacy append APIs (kept for v1 fallback paths) -------------------- #

    def add_done(
        self,
        outcome_id: str,
        evidence: str,
        step_idx: int,
        subgoal_text: str,
    ) -> bool:
        """Legacy v1 entry point used by the old summarizer and any
        external code that imported ``ProgressLedger`` directly. Mirrors
        the v1 behavior (idempotent dedup on outcome_id) but ALSO
        synthesizes a KeyNode and folds it into the cache so the new
        derive path stays consistent."""
        if outcome_id not in {o.id for o in self.required_outcomes}:
            return False
        if outcome_id in {d.outcome_id for d in self.done}:
            return False
        # legacy append
        self.done.append(DoneEntry(
            outcome_id=outcome_id,
            verified_at_step=step_idx,
            verified_at_subgoal=(subgoal_text or "")[:240],
            evidence=(evidence or "")[:240],
        ))
        # mirror into per-outcome state so v2 derive stays consistent
        o = self.get(outcome_id)
        if o is not None:
            synth = KeyNode(
                kind=KN_OUTCOME_SATISFIED,
                target=outcome_id,
                evidence=(evidence or "")[:240],
                confidence=CONF_HIGH,
                detected_at_step=step_idx,
            )
            o.apply_keynode(synth, step_idx)
        return True

    # Cap on how many concrete actor actions we store per FailedPath.
    # Prevents unbounded growth when a strategy keeps re-triggering the
    # cadence — only the first ~12 distinct actions are kept; rendering
    # to the prompt typically shows the first 5.
    _MAX_ACTIONS_PER_FAILED_PATH = 12

    def add_failed_path(
        self,
        path: str,
        why: str,
        step_idx: int,
        subgoal_text: str,
        evidence_strength: str = "auto",
        action_sequence: Optional[List[str]] = None,
        level: str = "action",
        target_outcome_id: Optional[str] = None,
    ) -> bool:
        """Record a failed strategy or action. Returns True if a new
        entry was appended; False if the path matched an existing
        entry of the SAME level — in the False case the existing
        entry's ``observation_count`` is bumped,
        ``last_observed_at_step`` updated, and any previously unseen
        entries from ``action_sequence`` are appended (capped at
        ``_MAX_ACTIONS_PER_FAILED_PATH`` to bound prompt length).

        ``level`` is "strategy" (planner-declared <strategy> that was
        abandoned) or "action" (stuck subgoal text). Dedup is
        per-level — the same text at strategy vs action level produces
        two distinct entries, since they tell the planner different
        things.

        A4: ``target_outcome_id`` defaults to the ledger's current
        focus outcome when None — best guess at which outcome this
        dead-end was failing against. The planner prompt filters
        failures by focus outcome so noise from other-outcome attempts
        doesn't bleed in.
        """
        key = (path or "").strip().lower()
        if not key:
            return False
        # A4: default target_outcome_id to current focus if caller
        # didn't supply one.
        if target_outcome_id is None:
            try:
                _f = self.next_focus_outcome()
                target_outcome_id = _f.id if _f is not None else None
            except Exception:
                target_outcome_id = None
        new_actions = self._normalize_action_sequence(action_sequence)
        for fp in self.failed_paths:
            if ((fp.path or "").strip().lower() == key
                    and getattr(fp, "level", "action") == level):
                fp.observation_count += 1
                fp.last_observed_at_step = step_idx
                # If a dedup hit re-confirms an entry whose target was
                # previously uncategorized, fill it in now.
                if (target_outcome_id is not None
                        and getattr(fp, "target_outcome_id", None) is None):
                    fp.target_outcome_id = target_outcome_id
                if new_actions:
                    seen = {a.strip().lower() for a in fp.action_sequence}
                    for a in new_actions:
                        if a.strip().lower() in seen:
                            continue
                        fp.action_sequence.append(a)
                        seen.add(a.strip().lower())
                        if len(fp.action_sequence) >= self._MAX_ACTIONS_PER_FAILED_PATH:
                            break
                return False
        self.failed_paths.append(FailedPath(
            path=(path or "")[:240],
            why=(why or "")[:240],
            first_observed_at_step=step_idx,
            first_observed_at_subgoal=(subgoal_text or "")[:240],
            evidence_strength=evidence_strength,
            observation_count=1,
            last_observed_at_step=step_idx,
            action_sequence=new_actions,
            level=level,
            target_outcome_id=target_outcome_id,
        ))
        return True

    @classmethod
    def _normalize_action_sequence(
        cls, actions: Optional[List[str]],
    ) -> List[str]:
        """Trim each action summary to a single line + 200 chars and
        cap the list at ``_MAX_ACTIONS_PER_FAILED_PATH``. Drops empty
        / whitespace-only entries."""
        if not actions:
            return []
        out: List[str] = []
        for a in actions:
            if not isinstance(a, str):
                continue
            s = a.replace("\n", " ").strip()
            if not s:
                continue
            out.append(s[:200])
            if len(out) >= cls._MAX_ACTIONS_PER_FAILED_PATH:
                break
        return out

    # ---- planner-driven review merge (ref:N protocol) ----------------------- #
    #
    # The planner's <dead_end_review> output uses an explicit reference
    # protocol so we don't have to do fuzzy semantic matching ourselves:
    #
    #   ref:N — implies: <forward implication>
    #     Updates failed_paths[N-1]: append the implication to its
    #     `why` field. Does NOT bump observation_count — see below.
    #
    #   ref:N,M,K — implies: <implication>
    #     Same, but applies to multiple existing entries — used when
    #     the planner recognizes near-duplicate entries describing
    #     the same strategy.
    #
    #   ref:N — weak signal
    #     Acknowledges the entry but no implication to merge. No-op.
    #
    #   ref:new — <strategy text> — implies: <implication>
    #     Adds a NEW entry with evidence_strength="planner". The
    #     planner is required by the prompt to anchor this on UI
    #     evidence visible in the current screenshot.
    #
    # No LLM call here — the matching decision was already made by
    # the planner when it wrote the ref tag.
    #
    # IMPORTANT: ``observation_count`` is bumped ONLY by
    # ``add_failed_path`` dedup hits (the cadence trigger firing
    # again on the same subgoal — i.e. the strategy was actually
    # RE-ATTEMPTED). The planner mentioning an entry retrospectively
    # in <dead_end_review> ("the X strategy was wrong, that's why
    # we're using Y now") is reflective commentary, NOT a fresh
    # attempt — bumping the count there inflates the signal and
    # misleads future turns into thinking the strategy is being
    # actively re-tried when it isn't. So merge_planner_review
    # leaves observation_count alone.

    _REF_LINE_RE = re.compile(
        r'^\s*[-•*]?\s*ref:\s*'
        r'([\d,\s]+|new)'                  # group 1: indices or "new"
        r'\s*(?:—|--|–|-)\s*'              # separator
        r'(.+?)\s*$',                      # group 2: payload (rest of line)
        re.IGNORECASE,
    )

    def merge_planner_review(
        self,
        review_text: str,
        step_idx: int,
        current_subgoal: str = "",
    ) -> Tuple[int, int]:
        """Merge a parsed <dead_end_review> body back into the ledger.

        Parses each ``ref:N — ...`` line and applies the planner's
        explicitly-marked decision (which existing entry the line
        addresses, or that it's a NEW claim). No fuzzy matching — the
        planner's tag IS the matching decision.

        Returns ``(n_updated, n_added)`` — how many existing entries
        were updated (count bumped + implication appended) and how
        many new entries were added via ``ref:new``.
        """
        if not review_text:
            return 0, 0

        n_updated = 0
        n_added = 0
        for raw_line in review_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = self._REF_LINE_RE.match(line)
            if not m:
                continue
            ref_part = m.group(1).strip().lower()
            payload = m.group(2).strip()

            # Strip a leading "implies:" / "implies " label so the
            # appended why reads as a sentence, not a literal echo.
            implication = payload
            for prefix in ("implies:", "implies "):
                if implication.lower().startswith(prefix):
                    implication = implication[len(prefix):].strip()
                    break

            # Branch on ref kind
            if ref_part == "new":
                # ref:new — <strategy> — implies: <implication>
                # The strategy and implication are joined in the
                # payload by another em-dash. Split on the FIRST
                # remaining em-dash variant (or "implies:") to peel
                # them apart. If we can't split, treat the whole
                # payload as the strategy (no implication).
                strat, why = self._split_strategy_and_why(payload)
                if not strat:
                    continue
                created = self.add_failed_path(
                    path=strat,
                    why=why or "(planner self-claimed dead-end; no implication given)",
                    step_idx=step_idx,
                    subgoal_text=current_subgoal,
                    evidence_strength="planner",
                )
                if created:
                    n_added += 1
                else:
                    # Same path already existed — add_failed_path
                    # already bumped its observation_count.
                    n_updated += 1
                continue

            # ref:N or ref:N,M,K
            indices: List[int] = []
            for tok in ref_part.replace(",", " ").split():
                try:
                    indices.append(int(tok))
                except ValueError:
                    pass
            if not indices:
                continue

            is_weak = implication.lower().startswith("weak signal")

            for idx in indices:
                # 1-based index per the prompt; clamp.
                arr_idx = idx - 1
                if not (0 <= arr_idx < len(self.failed_paths)):
                    continue
                fp = self.failed_paths[arr_idx]
                # Note: do NOT bump observation_count here — that
                # field tracks RE-ATTEMPTS (cadence dedup hits in
                # add_failed_path), not retrospective mentions in
                # planner reviews. Counting both inflates the signal
                # because the planner repeatedly references old
                # entries to explain why the current direction
                # differs ("X was wrong, that's why we use Y now").
                if is_weak or not implication:
                    n_updated += 1
                    continue
                merged = (fp.why or "").rstrip()
                addition = f" | implies: {implication}"
                if addition.strip().lower() not in merged.lower():
                    fp.why = (merged + addition)[:480]
                n_updated += 1

        return n_updated, n_added

    @staticmethod
    def _split_strategy_and_why(payload: str) -> Tuple[str, str]:
        """For a ``ref:new`` line's payload, separate the strategy
        text from the implication. Accepts either an em-dash
        separator or an explicit "implies:" / "implies " label."""
        # Prefer the explicit "implies:" split — least ambiguous.
        m = re.search(r"\bimplies\b\s*:?\s*", payload, re.IGNORECASE)
        if m:
            strat = payload[:m.start()].rstrip(" —–-")
            why = payload[m.end():].strip()
            return strat, why
        # Fall back to splitting on the next em-dash variant.
        for sep in ("—", "–", "--"):
            if sep in payload:
                strat, _, why = payload.partition(sep)
                return strat.strip(), why.strip()
        # No split available — treat the whole thing as the strategy.
        return payload.strip(), ""

    # ---- rendering ---------------------------------------------------------- #

    def to_prompt_block(self, latest_snapshot: Any = None) -> str:
        """Render the planner-facing outcome view via the v2 formatter
        (verified / [REVERTED] / pending) plus the failed_paths block
        and an initial-context summary line.

        ``latest_snapshot`` is accepted for caller-API compatibility but
        is no longer rendered into the block — its content fully overlaps
        the upstream ``SCREEN SNAPSHOT`` message and re-emitting it here
        had the side-effect of making the planner over-weight a single
        subgoal-scoped verdict as task-level confirmation. Keep the
        parameter so call sites don't have to change; ignore the value.
        """
        _ = latest_snapshot  # intentionally unused — see docstring
        # initial context line
        head_lines: List[str] = []
        ctx = self.initial_context
        ctx_bits: List[str] = []
        if ctx.active_url:
            ctx_bits.append(f"active tab: {ctx.active_url}")
        if ctx.active_app:
            ctx_bits.append(f"app: {ctx.active_app}")
        if ctx.window_title:
            ctx_bits.append(f'window: "{ctx.window_title}"')
        if ctx.open_tabs and len(ctx.open_tabs) > 1:
            ctx_bits.append(f"open tabs: {len(ctx.open_tabs)}")
        if ctx_bits:
            head_lines.append("[Initial context]")
            for b in ctx_bits:
                head_lines.append(f"  • {b}")
            head_lines.append("")
        # Environment snapshot from probe + perceive_environment.
        # Lists ONLY identifiers verified to exist on the VM at task
        # start — schemas / binaries / config files / processes that
        # are likely relevant to the user's instruction. Trust these
        # over guessed names.
        if ctx.environment_probe:
            # Runtime-relevant slice only. The full probe is fed to
            # init_ledger via user_content where it informs verify-spec
            # selection. Once the spec is fixed, the verify key/path is
            # already encoded in each outcome's evidence_hint, so we drop
            # the redundant [Relevant binaries] / [Relevant GSettings
            # schemas] / [Relevant dconf paths] / [Relevant config files]
            # / app-specific config dumps from every-step rendering.
            # Kept: operation-path framing (still steers actor CLI/GUI
            # choice each turn) + [Identity] (path expansion) + [Running
            # processes] (whether the target app is alive) + [Possibly
            # useful files] (file paths the planner may decide to open
            # at any turn via ``open_app(file=...)``; not encoded into a
            # verify, so it has to survive past init_ledger).
            head_lines.append("[VM environment — observed at task start]")
            head_lines.append(
                "  Actor's operation path: prefer GUI when a native app "
                "handles the target directly (in-app preferences, right-"
                "click menus, install buttons, dialogs). Prefer CLI for "
                "editing app config files (JSON / INI / dot-files under "
                "~/.config or app data dirs), system settings via "
                "gsettings / dconf, batch file operations, text edits, "
                "pattern matching across many files."
            )
            keep_sections = {"[Identity]", "[Running processes]",
                             "[Possibly useful files]"}
            in_keep = False
            in_useful = False
            # Hint appended right after the [Possibly useful files] block
            # so the planner knows HOW to open one — open_app bypasses the
            # GUI file picker (which is where Calc/Writer tasks lose runs
            # to mis-clicks → blank-document openings).
            useful_files_hint = (
                "    ▸ to open one of these, use "
                "`open_app(app=<app>, file=<absolute path>)` — bypasses "
                "the GUI file picker (mis-clicks there commonly open a "
                "blank document instead)."
            )
            # Some perceiver runs hallucinate repeated lines inside
            # ``[Possibly useful files]`` (observed: 25× the same
            # ``Bookmarks-journal`` path on a chrome task). Dedup
            # path entries WITHIN that section while preserving order;
            # leave header / non-file lines alone.
            seen_useful: set = set()
            for ln in ctx.environment_probe.splitlines():
                stripped = ln.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    if in_useful:                # leaving the section
                        head_lines.append(useful_files_hint)
                        in_useful = False
                        seen_useful.clear()
                    in_keep = stripped in keep_sections
                    if stripped == "[Possibly useful files]":
                        in_useful = True
                        seen_useful = set()
                if in_keep:
                    if in_useful and stripped and not (
                            stripped.startswith("[") and stripped.endswith("]")):
                        if stripped in seen_useful:
                            continue
                        seen_useful.add(stripped)
                    head_lines.append(f"  {ln}")
            if in_useful:                        # section was last
                head_lines.append(useful_files_hint)
            head_lines.append("")
        body = render_outcome_view(
            self.required_outcomes, self.failed_paths,
            completed_strategies=self.completed_strategies,
        )
        # NOTE: latest_snapshot's verdict is intentionally NOT re-rendered
        # here. The full ``SCREEN SNAPSHOT`` block (composed by the
        # planner-prompt builder upstream) already carries the same
        # subgoal_check + clauses + coverage. Echoing it a second time
        # made the planner over-weight a single subgoal-scoped verdict
        # — observed: a single ✓ "tab visible" being read as task-level
        # confirmation when the actual outcome required a URL change.
        return "\n".join(head_lines) + body

    # ---- serialization ------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Don't dump the b64 screenshot to traj.jsonl — it's >100KB
        # per row and inflates trajectory dumps to MB-scale.
        d.pop("initial_screenshot_b64", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProgressLedger":
        ctx_d = d.get("initial_context", {}) or {}
        ctx = InitialContext(
            open_tabs=list(ctx_d.get("open_tabs") or []),
            active_url=ctx_d.get("active_url"),
            active_app=ctx_d.get("active_app"),
            window_title=ctx_d.get("window_title"),
            environment_probe=ctx_d.get("environment_probe"),
            useful_file_contents=ctx_d.get("useful_file_contents"),
        )
        # Outcome: tolerate persisted records that lack newer A1 state
        # fields. Field defaults handle missing keys; we only filter
        # unknown keys.
        _o_fields = set(Outcome.__dataclass_fields__.keys())
        req = [
            Outcome(**{k: v for k, v in o.items() if k in _o_fields})
            for o in d.get("required_outcomes") or []
        ]
        done = [DoneEntry(**e) for e in d.get("done") or []]
        # FailedPath: tolerate old persisted records that lack newer
        # fields (level, action_sequence, observation_count, etc.).
        # Field defaults take care of missing optional fields; we only
        # need to guard against UNKNOWN keys (none expected today).
        _fp_fields = set(FailedPath.__dataclass_fields__.keys())
        failed = [
            FailedPath(**{k: v for k, v in f.items() if k in _fp_fields})
            for f in d.get("failed_paths") or []
        ]
        # Back-compat: traj.jsonl written before A1 has an outcome_cache
        # blob alongside per-outcome bare identity. Replay tools can re-
        # hydrate state by walking the cache dict and writing it onto
        # the corresponding outcomes.
        cache_d = d.get("outcome_cache") or {}
        if cache_d:
            states = cache_d.get("states") or {}
            lss = cache_d.get("last_satisfy_step") or {}
            lis = cache_d.get("last_invalidate_step") or {}
            rc = cache_d.get("revert_count") or {}
            by_id = {o.id: o for o in req}
            for oid, st in states.items():
                o = by_id.get(oid)
                if o is None:
                    continue
                o.state = st
                if oid in lss:
                    o.last_satisfy_step = lss[oid]
                if oid in lis:
                    o.last_invalidate_step = lis[oid]
                if oid in rc:
                    o.revert_count = rc[oid]
        else:
            # Older format with only legacy done[] — synthesize verified.
            by_id = {o.id: o for o in req}
            for de in done:
                o = by_id.get(de.outcome_id)
                if o is not None:
                    o.state = OUTCOME_VERIFIED
                    o.last_satisfy_step = de.verified_at_step
        completed = []
        for cs in d.get("completed_strategies") or []:
            if isinstance(cs, dict) and cs.get("strategy"):
                completed.append({
                    "strategy": str(cs["strategy"]),
                    "satisfied_outcomes": list(cs.get("satisfied_outcomes") or []),
                })
        slots: Dict[str, VerifySpecSlot] = {}
        for sd in (d.get("slots") or {}).values() if isinstance(d.get("slots"), dict) else []:
            if isinstance(sd, dict) and sd.get("name"):
                slots[sd["name"]] = VerifySpecSlot(**{
                    k: v for k, v in sd.items()
                    if k in VerifySpecSlot.__dataclass_fields__
                })
        return cls(
            initial_context=ctx,
            required_outcomes=req,
            done=done,
            failed_paths=failed,
            completed_strategies=completed,
            slots=slots,
        )
