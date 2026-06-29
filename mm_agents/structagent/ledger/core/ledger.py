"""Progress Ledger v2 — outcome state derived from the Event Timeline.

Outcome state (verified / pending / reverted) is folded from KeyNodes the
agent feeds in. Supports REVERSION: a verified outcome flips back to
pending when its evidence is invalidated.

v1 fields/methods preserved for back-compat (``done``, ``failed_paths``,
``add_done``, ``add_failed_path``, ``can_accept_done_claim``, ``pending``,
``all_done``, ``to_prompt_block``, ``to_dict``, ``from_dict``). ``add_done``
synthesizes a KeyNode so legacy paths still update the authoritative state.
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
# --------------------------------------------------------------------------- #

def abstract_actor_response(resp: Optional[str]) -> str:
    """One-line summary of an actor response for a FailedPath's
    ``action_sequence``.

    Prefers the actor's own ``Plan:`` line (right level of abstraction for
    "what was tried"), then the ``[2-stage re-grounded]`` line, then the
    first non-empty line truncated.
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


# Stopwords for strategy-similarity matching. Drop generic English + UI
# verbs so discriminating tokens (element names, routes, method names)
# dominate the Jaccard score.
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


# Jaccard threshold for the mechanical-fallback path of
# ``strategy_similar`` (no LLM judge). Token matching false-positives on
# rewording, so prefer the ``judge=`` path; mechanical mode is for offline
# tests and as a safety net when the LLM path errors out.
STRATEGY_SIMILAR_JACCARD = 0.7


def strategy_similar(
    prior: Optional[str], new: Optional[str],
    *,
    judge: Optional[Any] = None,
    actions_under_prior: Optional[List[str]] = None,
    progress_signal: str = "unknown",
    threshold: float = STRATEGY_SIMILAR_JACCARD,
) -> bool:
    """True if the planner is still pursuing ``prior`` (SAME), False if it
    abandoned it for a structurally different approach (DIFFERENT).

    With ``judge`` (callable -> bool): LLM-backed, called as
    ``judge(prior, new, actions_under_prior, progress_signal)``.
    ``progress_signal`` is "advanced" / "stuck" / "unknown" — on "unknown"
    the judge should lean SAME to avoid false dead-ends.

    Without ``judge``: token Jaccard >= ``threshold`` (imprecise but
    dependency-free; for offline tests).

    Caller caches lookups; this does not. A raised judge is treated as
    "not similar" rather than swallowed.
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
# transition. Caller injects the LLM call so this module stays free of any
# vLLM / openai dependency.
# --------------------------------------------------------------------------- #

# How many recent actor responses to surface to the judge — enough to show
# progress vs stuck-loop, not so much it blows the few-shot window.
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
    """SAME -> True (no FailedPath); anything else -> False (switch)."""
    return "SAME" in (response or "").upper()[:30]


def judge_strategy_change(
    *,
    prior: str, new: str,
    actions_under_prior: List[str],
    progress_signal: str,
    llm_call: Any,
    cache: Optional[Dict] = None,
) -> bool:
    """LLM-backed: True if planner is still pursuing ``prior`` (SAME),
    False if it abandoned it (DIFFERENT).

    Args:
      actions_under_prior: actor responses run while ``prior`` was
        committed — evidence of whether prior was making progress.
      progress_signal: "advanced" / "stuck" / "unknown".
      llm_call: ``(prompt: str) -> str``, injected to keep this module
        free of LLM-client deps.
      cache: memoization dict keyed on (prior, new, progress_signal) so a
        changed signal re-issues the call.

    Empty prior/new -> False; identical -> True (skip LLM); a raised
    ``llm_call`` -> False (safe-side: caller records prior as a FailedPath).
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
# Strategy-change tracker — single entry point for the agent.
#
# At every plan-commit site (step==0 install + REPLAN swap) the agent packs
# its tracking fields into a ``StrategyState``, calls
# ``commit_strategy_change`` with the new <strategy>, and assigns the
# returned state back. Keeps all "did the strategy change" logic in one
# place instead of scattered in the agent.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StrategyState:
    current_strategy: Optional[str]
    started_at_step: int
    replans_within_strategy: int


def _derive_progress_signal(
    ledger: Optional["ProgressLedger"], span_start: int, step_idx: int,
) -> str:
    """Whether the prior strategy's span [span_start, step_idx] saw forward
    motion: "advanced" (an outcome satisfied within the span), "stuck" (no
    outcome satisfied), or "unknown" (no ledger)."""
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
      - Empty ``new_strategy`` (parser hiccup): treat as SAME, bump counter.
      - No prior strategy (first plan): install ``new_strategy``, start span.
      - ``strategy_similar`` SAME: reworded/narrowed same path — append a
        "same" event, bump counter.
      - DIFFERENT: switched — append a "switch" event, install
        ``new_strategy``, reset counter/span. The LLM curator turns switch
        events into strategy-level FailedPaths later, post-REPLAN.
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
        # First plan — append an "install" event marking the start of
        # the strategy chain.
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
        # Same approach reworded — append a "same" event so the curator
        # can see how long the planner has been on this strategy. Don't
        # write ledger.failed_paths here; the curator owns that decision.
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

    # Strategy switch — record the event. The LLM curator decides
    # whether/how it becomes a strategy-level FailedPath (merge, drop as
    # actor-flailing, etc.) next time it runs, post-REPLAN.
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
    """Structured verification directive attached to an Outcome. Authored by
    ``init_ledger`` at task start, consumed by the KeyNode dispatcher each
    step. Flat ``kind`` values, no nesting.

    file_grep
      Regex-match ``patterns`` against a VM file's bytes. ``proximity_lines``
      requires the patterns to occur within that many lines of each other
      (list-entry tasks where fields must belong to the same JSON object,
      e.g. a keybinding with both 'key' and 'command' set).
      Required: ``file_path``, ``patterns``.
    url_match
      Regex-match the active-tab URL from a11y. Required: ``url_pattern``.
    a11y_match
      Find an a11y row matching tag/text/state. ``text_contains`` and
      ``state_contains`` are AND'd. Required: one of ``tag``,
      ``text_contains``, ``state_contains``.

    file_grep with an unresolvable ``file_path`` (ENOENT after fallback
    search) falls back to the legacy LLM KeyNode path — avoids false
    negatives when init_ledger writes a wrong path."""

    kind: str  # "file_grep" | "url_match" | "a11y_match" | "shell_command" | "calc_verify" | "impress_verify"

    # file_grep
    file_path: Optional[str] = None
    patterns: Optional[List[str]] = None
    all_must_match: bool = True
    proximity_lines: Optional[int] = None

    # calc_verify — list of check dicts, each
    # ``{"op": <cell_value|sheet_exists|pivot_exists|cell_format|sort_order|
    # freeze_pane|duplicate_range_match>, ...kwargs}``. All must pass (AND).
    # Framework builds the python3 -c command from the dicts — the LLM no
    # longer writes Python source, so fake-import / unescaped-quote /
    # missing-PASS-token bugs are gone by construction.
    calc_checks: Optional[List[Dict[str, Any]]] = None

    # impress_verify — same shape as ``calc_checks``, via IMPRESS_VERIFY_OPS.
    impress_checks: Optional[List[Dict[str, Any]]] = None

    # writer_verify — same shape as ``calc_checks``, via WRITER_VERIFY_OPS.
    writer_checks: Optional[List[Dict[str, Any]]] = None

    # url_match
    url_pattern: Optional[str] = None

    # a11y_match
    tag: Optional[str] = None
    text_contains: Optional[List[str]] = None
    state_contains: Optional[List[str]] = None

    # shell_command — ground truth lives in a CLI tool's stdout (``which
    # spotify``, ``gsettings get ...``, etc.). Dispatcher runs ``command``
    # and passes iff exit code == ``expected_exit_code`` (default 0) AND
    # some ``expected_substring`` is in stdout (case-insensitive) AND no
    # ``forbidden_substring`` appears.
    #
    # ``command`` is an argv list, NOT a shell string — passed to the VM
    # with NO shell interpretation, so quoting / $-expansion / pipe /
    # redirect do NOT work. Use argv form, e.g.
    # ``["gsettings", "get", "org.gnome...", "idle-dim"]``.
    command: Optional[List[str]] = None
    expected_substring: Optional[List[str]] = None
    forbidden_substring: Optional[List[str]] = None
    expected_exit_code: Optional[int] = 0

    # Dual-channel safety net for a11y_match (only the LLM KeyNode path
    # reads these; file_grep / url_match are deterministic). init_ledger
    # MUST populate at least one ``visual_must_hold`` clause; a11y_match
    # outcomes lacking it are flagged and fall back to legacy a11y-only.
    #
    # ``visual_must_hold``: single-sentence visual clauses that MUST hold
    #   on the current screenshot (AND-semantics) for the outcome to be
    #   satisfied. ``visual_must_not_hold``: clauses that MUST be false.
    # A failing clause keeps the outcome at its previous state (no
    # invalidation emitted).
    visual_must_hold: Optional[List[str]] = None
    visual_must_not_hold: Optional[List[str]] = None

    def is_valid(self) -> bool:
        """Whether the spec is fillable for its kind. Used by init_ledger's
        parser to drop malformed entries rather than crash mid-verify."""
        if self.kind == "file_grep":
            return bool(self.file_path) and bool(self.patterns)
        if self.kind == "url_match":
            return bool(self.url_pattern)
        if self.kind == "a11y_match":
            # [FIX a11y_visual_only_valid] visual_must_hold alone is a valid
            # grounding for a11y_match — no static tag/text/state anchor
            # needed when the criterion is best phrased as a sentence the
            # screenshot must satisfy. Roll back: drop the visual_must_hold
            # branch.
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
            # Hard whitelist on argv[0]. Anything off-list is either a
            # forbidden pipeline (find / tar / awk / cat / grep / bash …) or
            # an unsourced path; reject so the outcome falls back to
            # verify=None and the KeyNode LLM-judge handles it.
            if self.command[0] not in (
                "which", "dpkg", "snap", "pgrep", "test", "gsettings",
                "python3",
            ):
                return False
            # ``test`` has no stdout — exit-code-only, so a substring rule is
            # meaningless. Every other binary needs at least one substring,
            # else the spec passes on exit==0 alone and never discriminates.
            if self.command[0] == "test":
                return True
            # ``python3`` only valid as ``["python3", "-c", "<expr>"]`` —
            # anything else lets the actor sneak in arbitrary code. The expr
            # must print a PASS token, so require an expected_substring.
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
    """A milestone state proving a subgoal done.

    Design rules (enforced via the init_ledger prompt; see
    ``progress_ledger/initializer.py``):

      * Each outcome is a GATING MILESTONE whose regression cascades and
        blocks downstream work. Transient UI cues ("search bar visible",
        "dialog dismissed") are NOT outcomes — they're subgoals.

      * ``evidence_hint`` anchors on the most DURABLE signal available
        (persisted file/config > URL/cookies > DOM state > UI focus).
        Lower in this hierarchy = more verified↔reverted flapping. Free
        text; the planner reads it for the target state to drive toward.

      * ``verify`` is the machine-executable counterpart. The dispatcher
        prefers it when present (deterministic), else falls back to the
        legacy LLM KeyNode path (None, or unevaluable e.g. file_grep with
        unresolvable file_path).

      * ``depends_on`` lists prerequisite outcome ids. NOT order-enforced
        (KeyNodes may verify in any sequence) — consumed by
        ``next_focus_outcome`` for single-focus prompting and by the
        renderer for structure.
    """
    id: str
    description: str
    evidence_hint: str
    depends_on: List[str] = field(default_factory=list)
    verify: Optional[VerifySpec] = None
    # Multi-app tasks only: app this outcome is verified in (e.g. "chrome",
    # "libreoffice_calc"). None for single-domain. Drives the agent's
    # ``_current_domain`` switching — domain follows the focus outcome's app.
    app: Optional[str] = None

    # ---- A1: per-outcome state (was OutcomeStateCache) ---------------- #
    # Authoritative state. PENDING → VERIFIED on outcome_satisfied, VERIFIED
    # → REVERTED on outcome_invalidated. apply_keynode() is the only mutator.
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

    # A8-V0: cached StuckDiagnosis (see stuck_diagnosis.py). Reset to None
    # on outcome_satisfied so a verified-then-reverted outcome re-diagnoses
    # next stuck cycle. Plain dict to keep asdict serialization simple.
    last_stuck_diagnosis: Optional[Dict[str, Any]] = None
    last_stuck_diagnosis_signature: Optional[str] = None

    # ---- A2: per-Outcome Evidence + Failure containers ---------------- #
    # ``evidence`` appended once per verify_with_trace by
    # bundle_assembler.build() — silent-observer, nobody reads it yet.
    # ``failures`` empty until A4 (migration out of failed_paths).
    evidence: List[Evidence] = field(default_factory=list)
    failures: List[Failure] = field(default_factory=list)

    # ---- A5: working-memory Facts owned by this Outcome --------------- #
    # ``facts``: typed captured values (structured replacement for the
    # legacy flat ``_notebook``). Populated by note_write / extract_info
    # mirror / perceiver slot binder when the producing outcome is known.
    # A5 dual-writes alongside ``_notebook``; A6 switches readers over.
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
        """True once the DONE-rejection counter hits the threshold — caller
        should drop deterministic verify and hand off to the LLM judge."""
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
        """Stash the latest verify_with_trace dict. Seeds file_baseline_size
        / file_baseline_hash on first observation so
        ``file_changed_since_baseline()`` can tell if edits reach disk."""
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
        """True if the verifier-target file size/hash differs from baseline,
        False if confirmed unchanged, None if data missing."""
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
        """Update self.state from a single KeyNode targeting this outcome."""
        if kn.kind == KN_OUTCOME_SATISFIED:
            self.state = OUTCOME_VERIFIED
            self.last_satisfy_step = step_idx
            # A8-V0: drop cached stuck diagnosis so a later revert+stuck
            # cycle diagnoses fresh against the post-revert state.
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
    """Legacy v1 record, still appended on outcome_satisfied for
    trajectory.html readers and the old ``add_done`` path. Authoritative
    state lives in per-outcome ``state``."""
    outcome_id: str
    verified_at_step: int
    verified_at_subgoal: str
    evidence: str


@dataclass
class FailedPath:
    """A dead-end record at one of two granularities (``level``).

      level == "strategy"
        ``path`` is the high-level <strategy> the planner declared (e.g.
        "Use the catalog site's main-page search"). Recorded when the
        planner SWITCHES strategy across REPLANs; future plans must not
        re-propose it.

      level == "action"
        ``path`` is the stuck subgoal text the planner kept emitting
        without progress. Recorded by the cadence trigger only when the
        strategy stayed the same — strategy presumed sound, only the
        tactical execution failed; try alternative tactics within it.

    ``action_sequence`` holds one-line summaries of the actor's actual
    attempts (evidence the path was tried). Capped at 12 for prompt size.
    """
    path: str
    why: str
    first_observed_at_step: int
    first_observed_at_subgoal: str
    # Which layer this dead-end lives at — see class docstring. Default
    # "action" for legacy persisted records (most are subgoal-text-style).
    level: str = "action"
    # Evidence provenance: "auto" (heuristic trigger) / "planner" (self-
    # claimed via <dead_end_review>) / "keynode" (invalidation chain).
    evidence_strength: str = "auto"
    # Times re-confirmed. Bumped ONLY by add_failed_path dedup hits (the
    # strategy was actually RE-ATTEMPTED). Retrospective <dead_end_review>
    # mentions do NOT bump it — see merge_planner_review.
    observation_count: int = 1
    # Last step we saw evidence of this dead-end. Defaults to
    # first_observed_at_step at construction.
    last_observed_at_step: int = 0
    # One-line summaries of actor invocations under this strategy (e.g.
    # "click(1885,140)"). Filled by the cadence trigger, extended with
    # distinct entries on dedup hits (capped at 12). Empty for planner-
    # claimed (``ref:new``) entries — the planner doesn't know the actions.
    action_sequence: List[str] = field(default_factory=list)
    # True if authored by the strong-model rescue agent. Curator's
    # drop_existing must NOT re-open these: rescue had full context and
    # deliberately declared the path dead.
    strong_model_origin: bool = False

    # A4: which Outcome this dead-end failed against. None = uncategorized
    # (legacy, or ambient actor-flailing). Set by the curator from the
    # originating StrategyEvent's focus_outcome_id (see
    # commit_strategy_change). render_outcome_view filters per-focus-outcome
    # so replanning O3 doesn't see O1/O2 noise.
    target_outcome_id: Optional[str] = None


@dataclass
class InitialContext:
    open_tabs: List[str] = field(default_factory=list)
    active_url: Optional[str] = None
    active_app: Optional[str] = None
    window_title: Optional[str] = None
    # Filtered VM-environment snapshot (≤~1.5KB) from
    # ``probes.probe_environment`` + ``perceiver.perceive_environment``.
    # Lists task-relevant identifiers (binaries, config files, GSettings
    # schemas, processes) observed on the VM — never invented. None if the
    # probe was unavailable/empty.
    environment_probe: Optional[str] = None
    # Disk-content probe of [Possibly useful files] from
    # ``file_probes.probe_useful_files``. Spreadsheets via SheetPerceiver,
    # docx/pptx as plain-text (soffice --convert-to txt). init_ledger only;
    # live UNO perceivers take over later, so not in per-turn prompts.
    useful_file_contents: Optional[str] = None


# --------------------------------------------------------------------------- #
# Slot — placeholder for a value the init_ledger LLM doesn't know yet
# --------------------------------------------------------------------------- #

# ``${slot_name}`` refs in verify-spec strings. Identifier-only names so the
# parser can't mistake regex metacharacters for a slot name when the spec is
# itself a regex (url_pattern, file_grep patterns).
_SLOT_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class VerifySpecSlot:
    """Typed placeholder for a value the init_ledger LLM can't know yet
    (runtime URL/filename, OCR-read number, ...). Outcomes reference slots
    via ``${slot_name}``; the verifier substitutes bound values and short-
    circuits to UNCERTAIN (None) while any ref is unbound — so the
    deterministic path can't rubber-stamp a vacuous pattern like ``.*``.

    Bind-once: ``value`` is set once by the perceiver on first evidence;
    later observations don't overwrite, so active-URL drift can't become
    spec drift.
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
    """Return ``(substituted, unresolved)``. Replaces each bound ``${name}``
    with its value (``re.escape``'d when the result is itself a regex).
    Unresolved refs stay intact and are collected in ``unresolved``."""
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
    # Strategies the curator marked ``strategy_outcome_satisfied`` (their
    # primary outcomes are verified). Rendered as a "Completed Strategies"
    # block so the planner doesn't re-propose them on REPLAN. Each entry:
    # ``{"strategy": <str>, "satisfied_outcomes": [<id>, ...]}``.
    completed_strategies: List[Dict[str, Any]] = field(default_factory=list)
    # Task-start screenshot (base64 PNG) from the initializer, reused by the
    # DONE-time auditor for before/after. Excluded from ``to_dict`` — >100KB
    # would balloon traj.jsonl.
    initial_screenshot_b64: Optional[str] = None
    # Multi-app tasks only: the PhaseBoard (app-phase decomposition + cross-
    # app handoff). None for single-app; every phase call site is gated so
    # single-domain behaviour is unchanged.
    phase_board: Optional[Any] = None

    # Slot table (see VerifySpecSlot above), keyed by name. Populated at
    # init, bound incrementally by the perceiver. Empty when verify specs
    # are fully concrete.
    slots: Dict[str, VerifySpecSlot] = field(default_factory=dict)

    # A5: task-wide Facts belonging to no specific Outcome (env paths,
    # note_write before a focus outcome exists). Each carries
    # ``source_outcome_id="global"`` and never goes stale.
    # ``working_memory()`` merges these with per-outcome facts.
    global_facts: Dict[str, Fact] = field(default_factory=dict)

    # ---- helpers ------------------------------------------------------------- #

    def _all_outcome_ids(self) -> List[str]:
        return [o.id for o in self.required_outcomes]

    def get(self, oid: str) -> Optional[Outcome]:
        """Look up an Outcome by id. O(n), fine for n < ~20."""
        for o in self.required_outcomes:
            if o.id == oid:
                return o
        return None

    # ---- A5: working-memory Fact helpers ---------------------------------- #

    # ``Fact.source_outcome_id`` marker for values not tied to any Outcome;
    # stored in ``global_facts``.
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
        """Append/overwrite one Fact, routed by ``owning_outcome``: a real
        outcome id -> ``outcome.facts`` (stale when it reverts); None /
        unknown -> ``global_facts`` (never stale). Returns the stored Fact
        for the caller's legacy ``_notebook`` mirror (A5 dual-write).
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
        """Flat view of currently-live Facts. Skips per-outcome facts whose
        Outcome is REVERTED; always includes ``global_facts``. Per-outcome
        entries win name collisions (real evidence beats scratchpad).
        """
        out: Dict[str, Fact] = dict(self.global_facts)
        for o in self.required_outcomes:
            if o.state == OUTCOME_REVERTED:
                continue
            for name, fact in (o.facts or {}).items():
                out[name] = fact
        return out

    def failures_for(self, oid: str) -> List[FailedPath]:
        """A4: failed_paths whose ``target_outcome_id`` == ``oid``. Excludes
        cross-cutting (None) entries — iterate ``failed_paths`` for those.
        Used by replan-prompt builders wanting a clean per-outcome subset.
        """
        return [
            fp for fp in self.failed_paths
            if getattr(fp, "target_outcome_id", None) == oid
        ]

    def state_of(self, oid: str) -> str:
        """State of an outcome by id; PENDING if missing (back-compat)."""
        o = self.get(oid)
        return o.state if o is not None else OUTCOME_PENDING

    # ---- derived views (read from per-outcome state) ------------------------ #

    def pending(self) -> List[Outcome]:
        """Never-verified outcomes. Disjoint from verified() and reverted()."""
        return [o for o in self.required_outcomes if o.is_pending()]

    def verified(self) -> List[Outcome]:
        return [o for o in self.required_outcomes if o.is_verified()]

    def reverted(self) -> List[Outcome]:
        return [o for o in self.required_outcomes if o.is_reverted()]

    def not_verified(self) -> List[Outcome]:
        """Outcomes that BLOCK DONE — pending + reverted."""
        return [o for o in self.required_outcomes if not o.is_verified()]

    def _leaf_outcome_ids(self) -> List[str]:
        """Outcome ids no other outcome ``depends_on`` — the TERMINAL
        deliverables.

        A verified leaf proves its whole dependency chain ran (no final
        artifact without prerequisites), so DONE need only gate on leaves.
        Intermediate outcomes are transient MEANS — their current state
        (e.g. an ``*_opened`` outcome flapping when its window loses focus)
        is irrelevant once passed, and gating on them only causes spurious
        rejections.

        Falls back to ALL ids when there are no leaves (depends_on cycle, or
        every outcome is depended-on) so the gate never accepts nothing."""
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
        """Gate: DONE accepted iff every LEAF outcome (see
        ``_leaf_outcome_ids``) is verified. Reverted leaves block; non-leaf
        outcomes aren't gated on.

        On rejection, bumps per-outcome DONE-rejection counters so the
        dispatcher can spot a wrong-from-the-start verifier (after K, falls
        back to LLM judging) and force_replan can show the cause counts."""
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
        """First outcome currently in focus: not yet verified AND all its
        ``depends_on`` verified.

        The renderer marks a single ▶ FOCUS ◀ outcome so the planner
        attacks one milestone at a time. Empty-``depends_on`` outcomes are
        always reachable, so the first such pending one wins on a fresh task.

        None when all leaves are verified (task done by the gate), or when
        the whole pending set is blocked by reverted deps (rare).
        """
        # All leaves verified → DONE-able, no focus. Without this, a flapped
        # intermediate (e.g. an ``*_opened`` outcome reverted on focus loss)
        # would be returned and drag the planner back to redo finished work.
        _leaf_ids = self._leaf_outcome_ids()
        if _leaf_ids and all(
            (self.get(_lid) and self.get(_lid).is_verified())
            for _lid in _leaf_ids
        ):
            return None
        for o in self.required_outcomes:
            if o.is_verified():
                continue
            # All deps must currently be verified. A reverted dep must be
            # restored before this outcome can progress.
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
        state. Also appends to legacy ``done`` for any newly-verified outcome
        so trajectory readers stay consistent."""
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
        """Legacy v1 entry point (old summarizer / external callers).
        Idempotent dedup on outcome_id, and also synthesizes a KeyNode so
        the v2 state stays consistent."""
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
        # mirror into per-outcome state for v2
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

    # Cap on actor actions stored per FailedPath — bounds growth when a
    # strategy keeps re-triggering the cadence (prompt renders ~first 5).
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
        """Record a failed strategy or action. True if a new entry was
        appended; False on a same-``level`` dedup hit (bumps
        observation_count + last_observed_at_step, merges unseen
        action_sequence entries, capped at _MAX_ACTIONS_PER_FAILED_PATH).

        ``level`` is "strategy" (abandoned <strategy>) or "action" (stuck
        subgoal text). Dedup is per-level — same text at different levels
        means different things to the planner, so two entries.

        A4: ``target_outcome_id`` defaults to the current focus outcome —
        the planner prompt filters failures by focus so other-outcome noise
        doesn't bleed in.
        """
        key = (path or "").strip().lower()
        if not key:
            return False
        # A4: default target_outcome_id to current focus.
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
                # Fill in a previously-uncategorized target on re-confirm.
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
        """Single-line, 200-char-trimmed action summaries, capped at
        ``_MAX_ACTIONS_PER_FAILED_PATH``; drops empty entries."""
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
    # <dead_end_review> uses an explicit reference protocol so we skip fuzzy
    # semantic matching:
    #   ref:N — implies: <impl>      append <impl> to failed_paths[N-1].why
    #   ref:N,M,K — implies: <impl>  same, for near-duplicate entries
    #   ref:N — weak signal          acknowledge, no-op
    #   ref:new — <strategy> — implies: <impl>
    #                                new entry, evidence_strength="planner"
    # No LLM call — the planner's ref tag IS the matching decision.
    #
    # IMPORTANT: ``observation_count`` is bumped ONLY by add_failed_path
    # dedup hits (an actual RE-ATTEMPT). A retrospective mention here ("X
    # was wrong, that's why we use Y now") is commentary, not a fresh
    # attempt — bumping it would inflate the signal. So merge leaves it.

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
        """Merge a <dead_end_review> body into the ledger by applying each
        ``ref:N — ...`` line's explicit decision (no fuzzy matching).

        Returns ``(n_updated, n_added)``.
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

            # Strip a leading "implies:" label so the appended why reads as
            # a sentence, not a literal echo.
            implication = payload
            for prefix in ("implies:", "implies "):
                if implication.lower().startswith(prefix):
                    implication = implication[len(prefix):].strip()
                    break

            # Branch on ref kind
            if ref_part == "new":
                # ref:new — <strategy> — implies: <impl>. Split strategy
                # from implication on the first em-dash / "implies:"; if no
                # split, the whole payload is the strategy.
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
                    # Existed already — add_failed_path bumped its count.
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
                # Do NOT bump observation_count here — it tracks RE-ATTEMPTS,
                # not retrospective review mentions (see block comment above).
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
        """Split a ``ref:new`` payload into (strategy, implication) on an
        "implies:" label or an em-dash separator."""
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
        """Render the planner-facing outcome view (verified / [REVERTED] /
        pending) plus the failed_paths block and an initial-context line.

        ``latest_snapshot`` is kept for API compat but ignored: it overlaps
        the upstream ``SCREEN SNAPSHOT`` message, and re-emitting it made the
        planner over-weight a subgoal-scoped verdict as task-level confirm.
        """
        _ = latest_snapshot  # intentionally unused — see docstring
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
        # Environment snapshot: identifiers verified to exist on the VM at
        # task start. Trust these over guessed names.
        if ctx.environment_probe:
            # Runtime-relevant slice only. The full probe went to init_ledger
            # for verify-spec selection; those keys/paths are now encoded in
            # each outcome's evidence_hint, so drop the redundant
            # [Relevant *] sections from per-step rendering. Kept:
            # operation-path framing (steers CLI/GUI each turn), [Identity],
            # [Running processes] (is the app alive), [Possibly useful files]
            # (paths the planner may open via open_app(file=...); not encoded
            # in any verify, so must survive past init_ledger).
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
            # Appended after [Possibly useful files] so the planner knows
            # HOW to open one — open_app bypasses the GUI file picker, where
            # Calc/Writer tasks lose runs to mis-clicks → blank documents.
            useful_files_hint = (
                "    ▸ to open one of these, use "
                "`open_app(app=<app>, file=<absolute path>)` — bypasses "
                "the GUI file picker (mis-clicks there commonly open a "
                "blank document instead)."
            )
            # Perceiver sometimes repeats lines in [Possibly useful files]
            # (observed: 25× the same Bookmarks-journal path on a chrome
            # task). Dedup path entries within the section, order-preserving;
            # leave header/non-file lines alone.
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
        # latest_snapshot's verdict is intentionally NOT re-rendered — the
        # upstream SCREEN SNAPSHOT block already carries it. Echoing it made
        # the planner over-weight a subgoal verdict (observed: a single ✓
        # "tab visible" read as task-level confirm when the outcome needed
        # a URL change).
        return "\n".join(head_lines) + body

    # ---- serialization ------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Drop the b64 screenshot — >100KB/row, inflates traj.jsonl to MBs.
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
        # Tolerate persisted records lacking newer A1 fields (defaults cover
        # missing keys); filter unknown keys.
        _o_fields = set(Outcome.__dataclass_fields__.keys())
        req = [
            Outcome(**{k: v for k, v in o.items() if k in _o_fields})
            for o in d.get("required_outcomes") or []
        ]
        done = [DoneEntry(**e) for e in d.get("done") or []]
        # Same for FailedPath: defaults cover missing fields (level,
        # action_sequence, ...); just guard against unknown keys.
        _fp_fields = set(FailedPath.__dataclass_fields__.keys())
        failed = [
            FailedPath(**{k: v for k, v in f.items() if k in _fp_fields})
            for f in d.get("failed_paths") or []
        ]
        # Back-compat: pre-A1 traj.jsonl carries an outcome_cache blob;
        # re-hydrate per-outcome state from it.
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
            # Oldest format: only legacy done[] — synthesize verified.
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
