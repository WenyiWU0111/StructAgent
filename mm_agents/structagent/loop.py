import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from mm_agents.structagent.utils.image import process_image
from mm_agents.structagent.utils.a11y import (
    extract_active_url_from_a11y as _extract_active_url_from_a11y,
    linearize_obs_a11y as _linearize_obs_a11y,
)
from mm_agents.structagent.utils.llm_client import LLMClient
from mm_agents.structagent.core.planner import Planner
from mm_agents.structagent.core.verifier import LedgerOrchestrator
from mm_agents.structagent.core.actor import Actor, ActionCompile
from mm_agents.structagent.core.actor.burst_controller import ActorBurstController
from mm_agents.structagent.core.actor.burst_state import ActorBurstState
from mm_agents.structagent.core.recovery import Recovery, RecoveryState
from mm_agents.structagent.config import CAConfig

logger = None


@dataclass
class _StepContext:
    """Per-step input context for the planner-actor pipeline.

    Populated once by ``_pa_begin_step`` (step counting, screenshot decode,
    perception, instruction augmentation) and read by every downstream phase.
    Planner OUTPUTS (plan / decision / parsed / last_subgoal_done) are NOT
    stored here — they flow as explicit return values through ``_pa_predict``
    so the per-step data pipeline stays visible at the call site.
    """
    instruction: str
    obs: dict
    env: Any = None
    wall_step_idx: Optional[int] = None
    # — filled by _pa_begin_step —
    step_idx: int = 0
    screenshot_bytes: Optional[bytes] = None
    screen_width: int = 0
    screen_height: int = 0


class StructAgent(LLMClient, ActionCompile, Actor, ActorBurstController,
                  Planner, LedgerOrchestrator, Recovery):

    def __init__(
        self,
        platform="ubuntu",
        model="qwen2.5-vl-72b-instruct",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.0,
        action_space="pyautogui",
        observation_type="screenshot",
        history_n=4,  # Number of previous interactions to include in full detail
        add_thought_prefix=False,
        planner_max_images=5,
        # ---- Actor: decomposer LLM (inline grounding) ----------
        decomposer_model=None,                # alias string; None → fall back to planner's model
        decomposer_api_url=None,              # None → fall back to planner's own model_url
        decomposer_system_path=None,          # path to A2 template .txt; None → built-in default
        verifier_model=None,                  # alias string; None → fall back to the planner model.
                                              # Splits the VERIFICATION stack (init_ledger spec authoring
                                              # + per-step outcome checks + done-auditor) onto its own
                                              # model knob; one of the 3 canonical models (planner /
                                              # verifier / grounding).
        # T3 Done-Auditor (plan Part I.2). When True, after the
        # structured ledger gate accepts a DONE, an adversarial
        # fresh-context LLM call re-checks the claim against the task
        # text + raw a11y + screenshots. Only an explicit FAIL flips the
        # acceptance to False. PARTIAL means uncertainty, not refutation.
        # Default on via the constructor; env can also enable it in
        # env-driven runs. Per-task budget caps spend.
        enable_done_auditor=True,
        done_auditor_model="vllm_qwen35-vl",
        done_auditor_max_per_task=10,
        # Stuck-diagnosis injection switch. When True, the per-outcome
        # verifier-trace render AND the framework stuck root-cause LLM
        # output are injected as a dedicated block in the force_replan
        # planner prompt. When False (default), neither runs and the
        # builder injects nothing — preserves baseline e8c4db0 behaviour
        # exactly. Opt-in via kwarg or env ``ENABLE_STUCK_DIAGNOSIS_
        # INJECTION=1`` so CI sweeps can A/B without recompiling.
        enable_stuck_diagnosis_injection=False,
        # NB: ``disable_specialized_executors`` was removed. All specialized
        # executors (LO/OS/VLC/VSCode/Thunderbird/Wiki/MultiHop/MultiApp)
        # have been deleted from the planner. Tasks now route through the
        # planner-actor GUI path uniformly.
        **_legacy_kwargs,                     # absorb any old kwargs (e.g.
                                              # disable_specialized_executors)
                                              # so existing callers don't break.
    ):
        self.platform = platform
        self.model = model
        # Verifier model: init_ledger spec authoring + per-step outcome verification
        # + the done-auditor run on THIS model. Defaults to the planner model, so a
        # single-model run is byte-identical; set it to split verification onto a
        # separate (e.g. stronger) model without touching the planner.
        self.verifier_model = verifier_model or model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.history_n = history_n  # Control how many previous interactions to include
        self.add_thought_prefix = add_thought_prefix
        self.planner_max_images = planner_max_images
        # Actor: decomposer LLM (inline grounding)
        self.decomposer_model = decomposer_model
        self.decomposer_api_url = decomposer_api_url
        self.decomposer_system_path = decomposer_system_path
        # Cache for the A2 decomposer system template (lazy-loaded on first use)
        self._decomposer_system_template: Optional[str] = None
        assert action_space in ["pyautogui"], "Invalid action space"
        assert observation_type in ["screenshot"], "Invalid observation type"
        # — config (construction-only) + per-task state (shared with reset) —
        self._init_runtime_config(
            enable_done_auditor,
            enable_stuck_diagnosis_injection, done_auditor_model,
            done_auditor_max_per_task)
        self._reset_task_state()

    def _init_runtime_config(self, enable_done_auditor,
                             enable_stuck_diagnosis_injection,
                             done_auditor_model, done_auditor_max_per_task):
        """Resolve all CONSTRUCTION-time config: runtime feature flags
        (``self.cfg`` + derived gates), fixed thresholds, and the
        done-auditor / stuck-diagnosis knobs. Set ONCE at
        construction — NOT re-run by reset() (config is process-stable;
        only per-task state resets)."""
        # memory_steps_text + osworld memory dicts removed —
        # cross-run memory retrieval gone.
        self.plan_update_interval: int = 1
        # Boundary-verify (always on): verify the current subgoal's TARGET
        # milestones when an actor burst hands control back to the planner,
        # then render the result as planner evidence.
        # Resolve ALL runtime feature flags ONCE, here, into self.cfg. Every
        # other site reads self.cfg.* — no scattered os.environ or repeated
        # from_env() in the per-step path.
        self.cfg = CAConfig.from_env()
        self._ACTOR_BURST_MAX_TURNS: int = max(1, self.cfg.actor_burst_max_turns)
        self._ACTOR_BURST_NO_EFFECT_LIMIT: int = max(
            1, self.cfg.actor_burst_no_effect_limit)
        # Feed retrieved check recipes into the boundary verifier when verifier
        # memory is enabled (this is where verifier memory now actually lands).
        self._verifier_memory_on: bool = self.cfg.verifier_experience_memory
        # Threshold: after this many consecutive REPLANs without a single
        # subgoal getting marked done, treat the task as stuck on "no-known-
        # UI-path" and force a tactic-switching replan (dispatcher Rule #6).
        self._REPLAN_WITHOUT_PROGRESS_THRESHOLD: int = 3
        # working memory: event timeline + per-turn KeyNode detector.
        # `MEMORY_VERSION=v1` env var falls back to legacy summarizer-only path.
        self._memory_version: str = self.cfg.memory_version
        if logger:
            logger.info("[Memory] version=%s", self._memory_version)
        # Perceiver feature flag (Phase: SubgoalSnapshot — see
        # docs/PROGRESS_LEDGER_V2_CHANGELOG.md §17). Default OFF for
        # zero-regression risk. Toggle with env var PERCEIVER_ENABLED=1.
        # When ON: a11y is intent-prefiltered, perceiver produces a
        # SubgoalSnapshot, planner sees snapshot block + crop image
        # instead of raw screenshot + raw a11y.
        self.use_perceiver: bool = self.cfg.perceiver
        # T3 Done-Auditor state. Constructor default is ON. The env
        # switch is still OR'd in so CI sweeps can enable it without
        # recompiling the kwarg list.
        self.enable_done_auditor: bool = (
            bool(enable_done_auditor)
            or self.cfg.done_auditor_env
        )
        # Stuck-diagnosis injection: gates BOTH the per-outcome
        # verifier-trace render AND the diagnose_stuck root-cause LLM
        # call in the force_replan path (planner.py). Default OFF →
        # neither runs, no injection, identical to baseline behaviour.
        # ON → both run and their output is bundled into
        # ``verifier_feedback`` for build_prompt_force_replan to inject.
        self.enable_stuck_diagnosis_injection: bool = (
            bool(enable_stuck_diagnosis_injection)
            or self.cfg.stuck_diagnosis_injection_env
        )
        self.done_auditor_model: str = done_auditor_model
        self.done_auditor_max_per_task: int = int(done_auditor_max_per_task)

    def _reset_task_state(self):
        """Initialize / clear ALL per-task mutable state. Called by BOTH
        __init__ and reset() so the two can never drift — a fresh agent and
        a reset agent hold byte-identical per-task state. Reads (never
        writes) the config attrs set by _init_runtime_config()."""
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.responses = []  # Store model responses
        self.screenshots = []  # Store processed screenshots
        # Planner state (plan-with-memory)
        self.current_plan: Optional[str] = None
        self.next_step_hint: Optional[str] = None
        # Planner-actor subgoal state
        self.subgoal_queue: List[str] = []
        # Phase-2 subgoal contract: each entry in subgoal_queue has a
        # parallel ``expected_post_state`` string describing what the next
        # screenshot should show after the subgoal succeeds. Populated by
        # ``_pa_parse_plan_into_subgoals`` at plan-parse time; the
        # planner's NEXT call reads these to judge ``done`` without a
        # separate check. Empty strings are allowed (legacy fallback).
        self.subgoal_expected_post_states: List[str] = []
        self.current_subgoal_expected_post_state: Optional[str] = None
        # Multi-app only: per-subgoal app tag parsed from <step app="...">.
        # Parallel to subgoal_queue. Drives _current_domain switching —
        # the execution context follows the app the PLANNER says the
        # current step runs in (not the verifier's focus-outcome app).
        self.subgoal_apps: List[Optional[str]] = []
        self.current_subgoal_app: Optional[str] = None
        # Ledger-driven subgoal completion (C / P1). Per-subgoal list
        # of outcome ids the step is supposed to advance to verified,
        # parsed from <step ... targets="X,Y">. Parallel to
        # subgoal_queue. Empty list = legacy fallback to planner's
        # self-assessment against expected_post_state.
        self.subgoal_targets: List[List[str]] = []
        self.current_subgoal_targets: List[str] = []
        # Initial-file registry: {app_id: file_path} OBSERVED at task
        # start (a wmctrl/lsof probe of the running env — see
        # open_app.build_initial_files_probe_script). Lets open_app
        # re-open the right file by path when called without one.
        # Populated once at step 0; empty for single-app tasks.
        self._initial_open_files: Dict[str, str] = {}
        self._initial_files_probed: bool = False
        self.completed_subgoals: List[str] = []
        # Subgoals that hit MAX_SUBGOAL_STEPS without finishing and were
        # force-replanned. Kept separately so neither the planner prompts
        # nor the corrector treat them as "already done" and skip their
        # re-introduction — they're failed attempts, not completions.
        self.abandoned_subgoals: List[str] = []
        self.current_subgoal: Optional[str] = None
        self.current_subgoal_step: int = 0
        # Planner's per-turn advisory message for the actor (set when
        # the planner emits <feedback_to_actor> on a done=NO assessment).
        # Cleared after one actor call; does NOT mutate current_subgoal.
        self.feedback_to_actor: Optional[str] = None
        self.current_subgoal_feedbacks: List[str] = []
        self.actor_history: List[dict] = []
        self._actor_burst_state: ActorBurstState = ActorBurstState()
        self._pending_planner_verifier_report: Optional[str] = None
        self.planner_history: List[tuple] = []
        # One-sentence label describing WHY force_replan should fire on
        # the NEXT turn. Set by the DONE-acceptance gate after a
        # rejection, or by other code paths
        # that pre-stage a known-stuck signal. Consumed by
        # ``_decide_planner_mode_and_reason`` rule #3 (which clears it
        # after reading) and surfaced in the force_replan prompt's
        # <stuck_reason>.
        self._force_replan_reason: Optional[str] = None
        # Parallel to ``_force_replan_reason``: a coarse category that
        # tags WHICH stuck pattern triggered the force_replan, so the
        # prompt builder can dispatch ONE focused plan-shape guidance
        # instead of dumping every possible category at the planner.
        # Values:
        #   "actor_failure"      — actor IMPOSSIBLE / no-bash-body (rule #2)
        #   "done_rejected"      — DONE rejected by ledger gate (rule #3)
        #   "budget_exhausted"   — subgoal step budget burned (rule #4)
        #   "queue_empty_no_done"— queue drained but task end-state unverified (rule #5)
        #   "replan_cadence"     — N consecutive REPLANs within same strategy (rule #6)
        #   None                 — no force_replan staged
        # Cleared together with ``_force_replan_reason`` whenever the
        # mode decision consumes the staged signal.
        self._force_replan_category: Optional[str] = None
        # Set at the END of an actor turn when the actor failed (returned
        # IMPOSSIBLE, FAIL, or the code-actor produced no body). Read at
        # the START of the next turn by
        # ``_decide_planner_mode_and_reason`` rule #2 to choose
        # mode="force_replan" with this string as the reason. Cleared
        # after the mode decision consumes it.
        self._pending_actor_failure_reason: Optional[str] = None
        # Counters that let us detect "planner keeps self-REPLAN-ing but
        # nothing is getting committed" — the Actor-side subgoal-step-cap
        # never fires in that case (each REPLAN resets current_subgoal_step
        # to 0), so we need a second stuck detector keyed on REPLAN cadence.
        self._replans_since_last_progress: int = 0
        self._last_search_fired_at_replan: int = -99  # so the first fire passes gate
        self._total_replan_count: int = 0
        # Fix 1 (feasibility verdict): the replan count at which we last ran
        # the INFEASIBLE judge. -999 so the first fire at replan>=K passes the
        # re-check cadence gate. Reset per task. See _maybe_feasibility_verdict.
        self._last_feasibility_check_at_replan: int = -999
        # IMPOSSIBLE handling: the actor's explicit "can't do this" must NOT be
        # routed to actor_redo (retrying the impossible, never bumping the replan
        # count → feasibility verdict starves). The flag makes the diagnose/router
        # hook skip this turn (→ normal planner force_replan, count++); the
        # consecutive count triggers an early feasibility verdict.
        self._actor_emitted_impossible: bool = False
        self._consecutive_impossible_count: int = 0
        # Read-only mirror of the above counters + the dispatcher's most
        # recent transition receipt. Refreshed once per turn in
        # ``_finalize_recovery_state`` (called immediately after the
        # mode dispatcher decides). T1 scope: ``self._*`` counters
        # remain source of truth; this is for tests + future T3
        # done-auditor escalation reads. See plan file Part I.1.
        self._recovery_state: RecoveryState = RecoveryState.bootstrap()
        # Two-layer FailedPath tracking.
        #   ``current_plan_strategy`` — the one-sentence <strategy> the
        #     planner declared at the head of its most recent committed
        #     <plan>. None until the first plan commits.
        #   ``replans_within_current_strategy`` — number of REPLANs the
        #     planner has done while keeping (a token-Jaccard equivalent of)
        #     the SAME strategy. Resets to 0 whenever the strategy changes.
        #     Rule #6 fires as an ACTION-level FailedPath only when this
        #     counter crosses _REPLAN_WITHOUT_PROGRESS_THRESHOLD (i.e. the
        #     specific click/step is stuck within an otherwise-fine
        #     strategy). When the planner SWITCHES to a structurally
        #     different strategy (Jaccard < threshold), the old strategy
        #     itself is recorded as a STRATEGY-level FailedPath instead.
        #   ``current_strategy_started_at_step`` — env-step index where
        #     ``current_plan_strategy`` was first committed; used to slice
        #     ``self.actions`` into the action_sequence summary stored on
        #     the strategy-level FailedPath when the strategy is abandoned.
        self.current_plan_strategy: Optional[str] = None
        self.replans_within_current_strategy: int = 0
        self.current_strategy_started_at_step: int = 0
        # Cache for ``_judge_strategy_similar`` — keyed on the (a, b)
        # pair (plus reverse-key check on lookup) so we issue at most
        # one LLM "is this the same approach?" call per unique pair
        # within a task.
        self._strategy_judge_cache: Dict[Tuple[str, str], bool] = {}
        # LLM-driven dead-end curator state. Strategy events accumulate
        # on each plan commit (install / same / switch); action-level
        # stuck events accumulate on each rule-#6 cadence trigger fire.
        # The curator runs after each REPLAN, looks at this full
        # history + currently-recorded failed_paths + outcome states,
        # and re-derives the authoritative ``ledger.failed_paths``
        # list. Replaces the old "every switch → write FailedPath"
        # mechanical behavior, which couldn't see across multiple
        # switches and produced fuzzy-duplicate / actor-flailing
        # entries.
        self._strategy_history: List[Any] = []  # List[StrategyEvent]
        self._action_stuck_history: List[Any] = []  # List[ActionStuckEvent]
        # Cached per-strategy notes from the most recent curator pass.
        # Keyed by strategy_text → "<verdict> — <rationale>" so the
        # timeline render can annotate each span box with the curator's
        # reading. Cleared on reset.
        self._last_curator_strategy_notes: Dict[str, str] = {}
        self.actor_subgoal_done_for_planner: bool = False
        self.actor_exhausted_for_planner: bool = False
        self.all_done_after_step: bool = False
        self.planner_response: Optional[str] = None
        self.planner_decision: Optional[str] = None
        # Full parsed planner output (Phase-2 contract: last_subgoal_done,
        # evidence, feedback_to_actor, decision, plan). Refreshed each
        # ``_pa_call_planner``; consumed by ``_pa_predict`` control flow.
        self.planner_parsed: Dict[str, Any] = {}
        # Progress ledger — structured proprioception. Init lazy (needs first obs).
        # Accumulators collect the per-step screenshots + actions until we hit
        # a subgoal boundary (CONTINUE / REPLAN / DONE / exhaust / max_steps),
        # at which point the summarizer runs and the lists reset.
        self.ledger = None                                     # ProgressLedger | None
        self._ledger_subgoal_screenshots: List[bytes] = []
        self._ledger_subgoal_actions: List[str] = []
        from mm_agents.structagent.ledger.core.timeline import TimelineEvent  # noqa: F401 (type only)
        self.timeline: List[Any] = []   # List[TimelineEvent]
        self._candidate_step: Optional[Any] = None  # transient — current turn's StepRecord
        self._ledger_subgoal_plan_step: str = ""
        self._ledger_last_done_reason: Optional[str] = None    # for HTML/log when ledger gate blocks DONE
        # Periodic-trigger state (Phase 3: simple every-N-turns summarizer).
        # We fire after every N actor turns (N = _LEDGER_FIRE_EVERY_N_TURNS)
        # plus once at planner DONE. All other boundary tags removed.
        self._ledger_turns_accumulated: int = 0
        # Window start state (captured at last fire). Used to compute the
        # observable delta (URL + a11y diff) over the firing window.
        self._ledger_url_at_window_start: Optional[str] = None
        self._ledger_a11y_at_window_start: Optional[str] = None
        # Snapshot of the obs the previous _pa_call_planner saw.
        # Used to compute a "last-1-turn delta" — the objective effect
        # of the most recent actor action — and inject it into the
        # planner's prompt so its <last_subgoal_assessment> is anchored
        # to actual observable change rather than free-form interpretation.
        self._planner_url_at_last_turn: Optional[str] = None
        self._planner_a11y_at_last_turn: Optional[str] = None
        self._current_snapshot: Optional[Any] = None  # SubgoalSnapshot when use_perceiver
        # INTENDED domain — what the planner says the next step belongs to.
        # Updated only from the planner's <step app="X"> tag (see
        # _pick_multi_app_domain). Stable across verifier flukes.
        self._current_domain: Optional[str] = None
        # PHYSICAL domain — what the OS is currently focused on, observed
        # from the a11y root every step. Decoupled from intent so a
        # mismatch is detectable and verifier scope can be gated on
        # actual UI visibility (see _observe_physical_domain).
        self._physical_domain: Optional[str] = None
        # Mismatch run length — how many consecutive steps physical has
        # disagreed with intended. Used by reconcile() to give up on the
        # auto-focus push after a few turns and demote intended to
        # physical (rather than spinning forever on a window that
        # refuses to come forward). See _reconcile_domains.
        self._domain_mismatch_steps: int = 0
        # Per-task budget — decremented on EVERY audit call (PASS too),
        # so a thrashing auditor can't burn unbounded LLM calls.
        self._done_audit_budget_remaining: int = (
            self.done_auditor_max_per_task
            if self.enable_done_auditor else 0
        )
        # Distinct from ``_done_rejected_count``: counts ONLY explicit
        # FAIL rejections by the auditor. PARTIAL is logged as
        # inconclusive. For offline analytics; the dispatcher does not
        # read this.
        self._done_audit_reject_count: int = 0
        # Set by the audit hook on explicit FAIL; consumed by
        # ``recovery.dispatcher`` Rule #3 to flip the force_replan
        # category from "done_rejected" to "done_audit_failed".
        self._force_replan_category_pending: Optional[str] = None
        # Failure-attribution v2: set by the planner-side force_replan
        # hook when the router decides ``actor_redo`` (i.e. previous
        # stuck state attributed to ACTOR with prose-level guidance).
        # Consumed + cleared (one-shot) inside actor's
        # ``_pa_call_actor_points`` on its next invocation. Schema:
        # {subgoal_id, root_cause_summary, memory_canonical_approach,
        #  divergence_description, missing_or_wrong, diagnosis}.
        self._pending_actor_recovery_payload: Optional[Dict[str, Any]] = None
        # Cache of the most recent ``InterventionDecision`` produced by
        # the v2 pre-replan hook. The planner-side prompt builder
        # (``_pa_build_subgoal_check_prompt``) consumes this on entry so
        # the v2 diagnose+router LLM call only happens ONCE per
        # force_replan turn, not twice.
        self._cached_failure_attribution_decision: Optional[Any] = None
        # Audit records from the v2 pre-replan hook's side-effect
        # executors. Populated when the router decides
        # ``verifier_override`` / ``env_intervention`` so the dump and
        # tests can read what actually happened (or what would have
        # happened in DRY-RUN). One-shot: overwritten each force_replan.
        self._last_verifier_override_audit: Optional[Dict[str, Any]] = None
        self._last_env_intervention_audit: Optional[Dict[str, Any]] = None
        # Last cli_run's stdout/stderr/returncode — captured from the
        # VM-side sentinel file ``/tmp/_last_cli_run.json`` (written by
        # ``cli_run_helpers.build_runtime_script``). cli_run uses
        # subprocess.run with capture_output=True so the planner's
        # screenshot can never see its output; injecting this dict into
        # the planner's observation block lets the planner recognize
        # cli_run successes (filepath found, command output, etc.)
        # instead of misjudging "no terminal visible" as failure.
        self._last_cli_run_output: Optional[Dict[str, Any]] = None
        # Last extract_info's per-file results. Shape:
        #   {"ok": bool, "query": str, "schema": dict,
        #    "files": [{"path": str, "ext": str, "ok": bool,
        #               "extracted": dict | None, "error": str | None,
        #               "raw_excerpt": str}, ...]}
        # Persistent KV notebook — single source of truth for any value
        # the agent has captured this task. Two write paths:
        #   (a) ``note_write(key, value)`` action — planner/decomposer
        #       explicitly save observed strings (an affiliation, a
        #       URL, a count).
        #   (b) ``extract_info`` results auto-mirror per-file entries
        #       (replaces the old per-task ``_extract_info_history``
        #       list; one notebook is rendered, not two blocks).
        # A7: ``self._notebook`` has been DELETED. Working memory now
        # lives per-Outcome in ``ledger.required_outcomes[*].facts``
        # plus ``ledger.global_facts`` for task-wide captures. The
        # rendering helper ``fact.render_working_memory`` is used by
        # both planner and decomposer prompts; goes stale automatically
        # when source Outcomes REVERT.
        # DocPerceiver / SheetPerceiver cache — populated by the
        # appropriate run_inspect() at task start + after every
        # structured-action mutation. DocPerceiver fires for writer; the
        # parallel SheetPerceiver fires for calc. Both render into the
        # SAME injection slot `_doc_inspect_block` because the planner
        # prompt has only one structure-block section. The dirty flag
        # flips True after every calc_*/impress_*/writer_* action and
        # triggers a re-probe on the NEXT _pa_predict() turn.
        self._doc_inspect_block: Optional[str] = None
        self._doc_inspect_dirty: bool = True
        # Impress: cached slide-focus list extracted once per task from
        # the task instruction (regex → LLM fallback → active slide).
        # Reused on every subsequent slide_perceiver probe so we don't
        # re-pay the focus-LLM cost each turn. Reset on every new task.
        self._impress_focus_slides: Optional[List[int]] = None
        # Task instruction cached for the focus extractor (mirrors how
        # _doc_inspect_block needs the instruction to build the block).
        self._task_instruction: Optional[str] = None
        # done_rejected accumulator for stuck detection (signal B).
        # Bumped each time the ledger gate rejects a planner DONE.
        self._done_rejected_count: int = 0
        # Text-answer benchmark (WebVoyager / Mind2Web / MMInA) detection.
        # Used to (a) augment the planner prompt so the model knows to
        # emit ``finished(answer="...")`` at task DONE and (b) scrape
        # that emit so the unified text-answer runner can pass it to
        # the right eval path (LLM judge vs gold-reference match). The
        # detection block ALSO accepts the legacy ``ANSWER(...)`` shape
        # for backward-compat with wiki_search_executor.
        self._is_text_answer_task: bool = False
        self._text_answer: Optional[str] = None
        self._text_answer_domain: Optional[str] = None
        # Back-compat aliases. Old runners and replay tools read these
        # field names; treat them as read-only mirrors of the unified
        # state above. Deletable once all readers migrate.
        self._is_mmina_task: bool = False
        self._mmina_answer: Optional[str] = None
        self._mmina_domain: Optional[str] = None
        # — per-task fields the run harness also relies on (previously
        #   created only in reset(); set here too so a fresh agent has them) —
        self.action_descriptions = []
        self.instruction = None
        self._raw_instruction = None
        self._related_apps: List[str] = []
        self._phase_start_action_idx: int = 0



    # ══════════════════════════════════════════════════════════════════════
    # Planner-Actor framework (ported from ZeroGUI training)
    # ══════════════════════════════════════════════════════════════════════

    # --- constants ---
    _MAX_ACTOR_RETRIES = 3
    _MAX_SUBGOAL_STEPS = 5
    _LEDGER_FIRE_EVERY_N_TURNS = 5      # periodic ledger summarizer cadence
    # 'verify the result' steps are usually redundant but non-harmful.
    # Only drop true no-op WAIT subgoals here.
    _SUBGOAL_SKIP_RE = re.compile(r'\bWAIT\b', re.IGNORECASE)


    # --- multi-hop detection ---

    @staticmethod
    def _is_multihop(instruction: str) -> bool:
        """Detect multi-hop MMInA tasks from instruction content.

        Multi-hop intents always start with the standard preamble:
        "For actions 'book a hotel','book a car','book a flight'..."
        and include "return the final url as the answer".

        TODO: Make this more agentic — instead of pattern matching the preamble,
        use the MetaPlanner LLM to decide if a task is multi-hop based on
        semantic understanding of the instruction.
        """
        lower = (instruction or "").lower()
        return "for actions" in lower and "return the final url" in lower




    # --- ledger helpers ------------------------------------------------- #

    def _ledger_disabled(self) -> bool:
        """Ablation switch: set env DISABLE_LEDGER=1 to turn off the
        progress-ledger pipeline entirely. Useful for {a11y on/off} ×
        {ledger on/off} 2x2 experiments without code changes."""
        return self.cfg.disable_ledger

    def _is_multi_app(self) -> bool:
        """True iff this task spans >1 application (the OSWorld
        ``multi_apps`` domain). Gates all multi-app behaviour — per-app
        outcome tags, dynamic ``_current_domain`` switching, the
        multi-app prompt additions. When False, every code path stays
        byte-identical to the single-domain behaviour."""
        return len(getattr(self, "_related_apps", None) or []) > 1

    @staticmethod
    def _vm_python_runner(env):
        """Resolve the VM-side python executor — ``run_python_script`` on
        either the real DesktopEnv (``env.controller``) or APIDesktopEnv
        (``env``). Returns None when unavailable."""
        if env is None:
            return None
        ctrl = getattr(env, "controller", None)
        if ctrl is not None and hasattr(ctrl, "run_python_script"):
            return ctrl.run_python_script
        if hasattr(env, "run_python_script"):
            return env.run_python_script
        return None

    def _activate_app_for_domain(self, env, domain: Optional[str]) -> None:
        """Bring the window for ``domain``'s application to the front via
        the ``open_app`` script. Best-effort: no-op for the ``os`` domain
        (shell work, no window to focus) and when no VM runner exists.

        Called on every multi-app ``_current_domain`` switch so the agent
        never has to juggle windows through the GUI — the framework
        focuses the right app as a side effect of the planner's phase."""
        if not domain or domain == "os":
            return
        runner = self._vm_python_runner(env)
        if runner is None:
            return
        try:
            from mm_agents.structagent.actions.app.open_app import build_open_app_script
            file_path = (getattr(self, "_initial_open_files", None)
                         or {}).get(domain)
            script = build_open_app_script(domain, file_path)
            if script:
                res = runner(script)
                from mm_agents.structagent.actions.app.active_app import (
                    app_switch_verify_enabled, parse_open_app_result,
                )
                if app_switch_verify_enabled():
                    # Closed loop (§8): read the script's focus-verify sentinel
                    # instead of fire-and-forget. ok=None (no sentinel / old VM)
                    # → unverified, behave as before.
                    parsed = parse_open_app_result(res)
                    self._last_app_switch = {"domain": domain, **parsed}
                    if logger:
                        logger.info(
                            "[MultiApp] activate %s -> ok=%s type=%s",
                            domain, parsed.get("ok"),
                            parsed.get("failure_type"))
                elif logger:
                    logger.info(
                        "[MultiApp] auto-activated %s window%s", domain,
                        f" ({file_path})" if file_path else "")
        except Exception as e:
            if logger:
                logger.warning(
                    "[MultiApp] auto-activate %s failed: %s", domain, e)

    def _phase_board(self):
        """The PhaseBoard for this task, or None. Multi-app tasks build
        one in init_ledger; single-app tasks never do. Centralizes the
        ``self.ledger.phase_board`` lookup so every phase call site is a
        clean no-op (returns None) on single-app tasks."""
        led = getattr(self, "ledger", None)
        return getattr(led, "phase_board", None) if led is not None else None

    def _pick_multi_app_domain(self) -> Tuple[Optional[str], Optional[str]]:
        """Decide the agent's INTENDED app for this turn — i.e. which app
        the planner says the next step belongs to. Pure, no side effects.

        Returns ``(domain, source)`` where ``domain`` is a member of
        ``_related_apps`` (or None when no change applies) and
        ``source`` is a short human-readable label for the log line.

        Single source of truth: the planner's ``<step app="...">`` tag,
        surfaced as ``current_subgoal_app``. Phase board / focus
        outcome / actor open_app were previously consulted as
        priorities 0/1/3, but all three are downstream of the verifier
        — when the verifier is wrong, they propagate the error into
        intent selection. Stripping them isolates "what the agent is
        TRYING to do" (planner intent) from "what the verifier thinks
        is done" (which controls phase advance / focus, but should not
        retroactively choose the active app).

        Returning ``(None, None)`` means "no planner signal yet" — the
        caller holds the previous ``_current_domain`` unchanged.
        """
        if not self._is_multi_app():
            return (None, None)
        related = self._related_apps or []
        _sg_app = getattr(self, "current_subgoal_app", None)
        if _sg_app:
            _sg_app = _sg_app.strip().lower().replace(
                " ", "_").replace("-", "_")
            _sg_app = {"terminal": "os",
                       "vscode": "vs_code"}.get(_sg_app, _sg_app)
            if _sg_app in related:
                return (_sg_app, "planner step")
        # C: planner forgot or mistyped the <step app=...> tag → fall
        # back to the PhaseBoard's active phase app. Better to follow
        # the structured phase contract than to leave _current_domain
        # stale on a turn the planner failed to tag. Only used when
        # primary signal is absent; planner's explicit tag still wins
        # when present.
        pb = self._phase_board()
        if pb is not None:
            try:
                active = pb.active()
            except Exception:
                active = None
            if active is not None and active.app in related:
                return (active.app, "phase board active (fallback)")
        return (None, None)

    def _pre_planner(self, obs: dict, step_idx: int, env: Any = None) -> None:
        """Build a candidate StepRecord from current obs, run the KeyNode
        detector, and fold key_nodes into the ledger BEFORE the planner
        prompt is built. This way the planner sees fresh outcome state
        (including [REVERTED] flags) in its read-only Progress Ledger
        block. Stores the candidate on ``self._candidate_step`` so
        ``_post_actor`` can finalize it after the actor runs.

        working-memory only: no-op when ``MEMORY_VERSION=v1`` or ledger is None /
        has no required outcomes (initializer failed)."""
        if self._memory_version == "v1":
            return
        if self.ledger is None or not self.ledger.required_outcomes:
            return
        from mm_agents.structagent.ledger.core.timeline import StepRecord, append_step
        from mm_agents.structagent.ledger.core.summarizer import compute_window_delta
        import datetime as _dt

        # Orphan-cand guard: ``_pa_predict`` has multiple early-return
        # paths (DONE accepted, DONE rejected & defer, code-actor empty
        # body, 2-stage re-grounding success, mmina answer) that skip
        # the trailing ``_post_actor`` call. The previous turn's
        # ``_pre_planner`` already folded that turn's KeyNodes into
        # ``self.ledger.outcome_cache`` (which advances state correctly),
        # but without a matching ``append_step`` call the cand never
        # makes it into ``self.timeline``. Renderers then see the
        # cache reflecting step N while the latest TimelineEvent.steps
        # still holds step N-1's KeyNodes — exactly the drift that
        # produced "[REVERTED] verified step 10, INVALIDATED step 11"
        # alongside a "✓ outcome_satisfied" KeyNode tile in the same
        # turn's render. Append the orphan now so timeline catches up
        # before we install the fresh cand for THIS step.
        stale_cand = self._candidate_step
        if stale_cand is not None and getattr(stale_cand, "step_idx", -1) < step_idx:
            try:
                append_step(self.timeline, stale_cand,
                            self.current_subgoal or "")
                if logger:
                    logger.info(
                        "[working_mem] orphan-cand recovered into timeline: "
                        "step_idx=%d (prior turn early-returned without "
                        "finalize)", stale_cand.step_idx)
            except Exception as e:
                if logger:
                    logger.warning(
                        "[working_mem] orphan-cand append failed: %s", e)
            self._candidate_step = None

        a11y = _linearize_obs_a11y(obs, max_items=300)
        # Chrome: append CHECKER-SOURCE web DOM (CSS class + ARIA state) — the
        # layer AT-SPI a11y lacks but the chrome checker actually reads (filter
        # chips, active tabs). One append HERE flows to every downstream consumer
        # of StepRecord.a11y (perceiver / planner / boundary / done-auditor).
        # Off via WEB_DOM_INJECT=0. Fails soft (CDP hiccup -> plain a11y).
        if (os.environ.get("WEB_DOM_INJECT", "1") == "1"
                and (self._current_domain or "").lower() == "chrome"
                and getattr(self, "_vm_ip", None)):
            try:
                from mm_agents.structagent.utils.web_dom import (
                    dump_tab_doms, render_web_dom)
                _web = render_web_dom(dump_tab_doms(
                    self._vm_ip, getattr(self, "_chromium_port", 9222)))
                if _web:
                    a11y = (a11y + "\n\n[WEB DOM — CSS class + ARIA state the "
                            "a11y tree lacks; the chrome checker reads THIS "
                            "layer for success]\n" + _web)
            except Exception as _e:
                if logger:
                    logger.warning("[web_dom] inject failed: %s", _e)
        url = _extract_active_url_from_a11y(a11y)
        screenshot_b64 = None
        try:
            shot = obs.get("screenshot") if isinstance(obs, dict) else None
            if shot:
                screenshot_b64 = base64.b64encode(shot).decode("ascii")
        except Exception:
            screenshot_b64 = None
        cand = StepRecord(
            step_idx=step_idx,
            timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
            screenshot_b64=screenshot_b64,
            a11y=a11y,
            url=url,
        )
        # Flatten timeline (event→steps) into one chronological StepRecord list.
        _all_steps = [s for ev in self.timeline for s in (ev.steps or [])]
        prev_step = _all_steps[-1] if _all_steps else None
        # The 3 frames just BEFORE prev (with a screenshot) → with prev+current
        # the visual judge sees up to 5 frames of the workflow, not only 2. A
        # save/print dialog confirming a file landed before it closes is
        # invisible to a 2-frame judge (task e1e75309). Slicing never raises
        # IndexError — a short/empty list just yields fewer (or zero) frames.
        _earlier_steps = [s for s in _all_steps[-4:-1]
                          if getattr(s, "screenshot_b64", None)]
        # Compute a11y delta for detector context
        a11y_delta_block = ""
        if prev_step is not None:
            try:
                wd = compute_window_delta(
                    url_before=prev_step.url, url_after=url,
                    a11y_before=prev_step.a11y, a11y_after=a11y,
                )
                a11y_delta_block = wd.to_prompt_block()
            except Exception as e:
                if logger:
                    logger.warning("[working_mem] window_delta failed: %s", e)
        # ── Perceiver FIRST (gated by PERCEIVER_ENABLED). Produces a
        # SubgoalSnapshot the planner will consume + acts as a STRONG
        # PRIOR for KeyNode below. Order: perceiver → KeyNode → ledger
        # gate (single source of truth on outcome state). See
        # docs/PROGRESS_LEDGER_V2_CHANGELOG.md §17.
        self._current_snapshot = None
        # Initial plan turn — no "last action" exists, no current_subgoal
        # has been emitted yet, no expected_post_state to score against.
        # Running the perceiver here produces meaningless "did last action
        # achieve expected state? NO" verdicts that just add noise to the
        # initial-plan prompt. Skip until step >= 1.
        if self.use_perceiver and step_idx > 0:
            try:
                from mm_agents.structagent.ledger.support.a11y_prefilter import (
                    prefilter_linearized_a11y,
                )
                from mm_agents.structagent.perception.perceiver import perceive
                from mm_agents.structagent.perception import perceiver_debug
                # Intent-prefilter a11y for the perceiver
                hints = [o.evidence_hint for o in self.ledger.required_outcomes]
                filtered_a11y = prefilter_linearized_a11y(
                    linearized_text=a11y or "",
                    subgoal_text=self.current_subgoal or "",
                    outcome_evidence_hints=hints,
                    top_n=80,
                )
                # Screen size for the perceiver (so it can return full-screen
                # bbox when uncertain). Decode PNG dims once from cand.
                screen_size: Tuple[int, int] = (1920, 1080)  # safe default
                shot_bytes = obs.get("screenshot") if isinstance(obs, dict) else None
                if shot_bytes:
                    try:
                        from PIL import Image
                        import io as _io
                        screen_size = Image.open(_io.BytesIO(shot_bytes)).size
                    except Exception:
                        pass
                last_action = (prev_step.actor_action_summary
                                if prev_step is not None else "") or ""

                # ── Debug dump: perceiver INPUT (before LLM call) ──
                results_dir = getattr(self, "_results_dir", None)
                a11y_full_rows = (a11y or "").count("\n")
                perceiver_debug.dump_perceiver_input(
                    results_dir=results_dir,
                    step_idx=step_idx,
                    subgoal_text=self.current_subgoal or "",
                    expected_post_state=self.current_subgoal_expected_post_state or "",
                    target_outcomes=self.ledger.required_outcomes,
                    last_action_summary=last_action,
                    a11y_filtered_text=filtered_a11y or "",
                    a11y_full_rows=a11y_full_rows,
                    transition_block=a11y_delta_block or "",
                    screen_size=screen_size,
                    screenshot_bytes=shot_bytes,
                )

                # raw_sink captures the perceiver's raw LLM response so
                # we can dump it even if parsing fails downstream.
                _captured_raw = {"text": None}

                def _raw_sink(text: str) -> None:
                    _captured_raw["text"] = text

                _unbound_slots = (
                    [s for s in self.ledger.slots.values() if s.value is None]
                    if self.ledger and self.ledger.slots else []
                )
                self._current_snapshot = perceive(
                    screenshot_bytes=shot_bytes or b"",
                    screen_size=screen_size,
                    a11y_text=filtered_a11y or "",
                    subgoal_text=self.current_subgoal or "",
                    expected_post_state=(
                        self.current_subgoal_expected_post_state or ""),
                    target_outcomes=self.ledger.required_outcomes,
                    last_action_summary=last_action,
                    transition_block=a11y_delta_block or "",
                    call_llm=self.call_llm,
                    model=self.verifier_model,    # outcome verification → verifier model
                    raw_sink=_raw_sink,
                    unbound_slots=_unbound_slots,
                )

                # Promote any new perceiver-resolved slot bindings into
                # the ledger. Bind-once: an already-bound slot is never
                # overwritten — protects against per-step drift (e.g. the
                # active URL changes after the value was captured).
                # Unknown slot names from the perceiver are ignored.
                if (self._current_snapshot is not None
                        and self.ledger and self.ledger.slots
                        and self._current_snapshot.slot_bindings):
                    for b in self._current_snapshot.slot_bindings:
                        slot = self.ledger.slots.get(b.name)
                        if slot is None or slot.value is not None:
                            continue
                        slot.value = b.value
                        slot.bound_at_step = step_idx
                        slot.bound_evidence = b.evidence
                        if logger:
                            logger.info(
                                "[Slot] step=%d bound %s = %r  (evidence: %s)",
                                step_idx, b.name, b.value[:80],
                                (b.evidence or "")[:80],
                            )

                # ── Debug dump: perceiver OUTPUT (raw + parsed) ──
                perceiver_debug.dump_perceiver_raw(
                    results_dir=results_dir,
                    step_idx=step_idx,
                    raw_response=_captured_raw["text"],
                )
                perceiver_debug.dump_perceiver_snapshot(
                    results_dir=results_dir,
                    step_idx=step_idx,
                    snapshot=self._current_snapshot,
                )

                if logger:
                    if self._current_snapshot is None:
                        logger.warning(
                            "[Perceiver] step=%d returned None — planner "
                            "will fall back to raw obs", step_idx)
                    else:
                        snap = self._current_snapshot
                        logger.info(
                            "[Perceiver] step=%d coverage=%s overall=%s "
                            "controls=%d bbox=%s blockers=%d",
                            step_idx, snap.coverage,
                            snap.subgoal_check.overall,
                            len(snap.relevant_controls),
                            snap.visual_focus.bbox,
                            len(snap.unexpected_blockers))
                # Persist on the StepRecord so timeline replay /
                # ledger render / curator can see it later.
                cand.snapshot = self._current_snapshot
            except Exception as e:
                if logger:
                    logger.warning("[Perceiver] step=%d failed: %s",
                                    step_idx, e)
                self._current_snapshot = None

        # ── Boundary-verify mode (always on): the per-step KeyNode poll is
        # bypassed. Outcome states are refreshed by boundary verifier reports
        # when an actor burst hands control back to the planner, plus at final
        # DONE. The StepRecord / a11y-delta / perceiver snapshot were already
        # built above (infra the planner + timeline need); only the expensive
        # per-step KeyNode detection + ledger fold is skipped.
        cand.key_nodes = []
        self._candidate_step = cand

    def _run_boundary_verify(self, outcomes, step_idx, env, *,
                             dump_name, dump_targets):
        """Shared boundary-verify core.

        Used by actor-loop-exit reports and verify-on-DONE. Builds context from
        the current candidate step, runs verify_milestone over ``outcomes``,
        marks verified ones on the ledger, and debug-dumps the verdict list.
        """
        from mm_agents.structagent.core.verifier.boundary_verify import (
            verify_milestone, VerifyContext, allowed_probes_for_domain)
        from mm_agents.structagent.ledger.core.ledger import OUTCOME_VERIFIED
        from mm_agents.structagent.ledger.support.a11y_prefilter import (
            prefilter_linearized_a11y)
        cand = self._candidate_step
        # Domain drives the per-domain trust map (allowed probes + ground-truth
        # guidance): the current (task / focused-app) domain — NOT the transient
        # _physical_domain (often unset early → chrome wrongly defaulted). Fall
        # back to _physical_domain.
        dom = self._current_domain or self._physical_domain or ""
        # Intent-prefilter the a11y to subgoal + milestone-relevant rows (same
        # processing the perceiver/planner get); raw a11y on content-heavy pages
        # is huge.
        a11y_text = prefilter_linearized_a11y(
            linearized_text=getattr(cand, "a11y", "") or "",
            subgoal_text=self.current_subgoal or "",
            outcome_evidence_hints=[getattr(o, "evidence_hint", "") for o in outcomes],
            top_n=80,
        )
        # Last few prior screenshots (oldest→newest) for workflow context — the
        # current frame is cand.screenshot_b64; add up to 4 earlier frames from
        # the timeline so the verifier sees the最近 5 张 (like the KeyNode judge).
        _all_steps = [s for ev in self.timeline for s in (ev.steps or [])]
        prior_shots = [s.screenshot_b64 for s in _all_steps[-4:]
                       if getattr(s, "screenshot_b64", None)]
        # Verifier memory: retrieve check recipes for this milestone class and fold
        # them into the verify prompt (opt-in via ENABLE_VERIFIER_EXPERIENCE_MEMORY).
        check_recipe_block = ""
        if self._verifier_memory_on:
            try:
                from mm_agents.structagent.memory.runtime.retrieval.query_check_recipes import (
                    retrieve_check_recipes, render_check_recipe_block)
                q = " ; ".join(
                    [self.current_subgoal or ""]
                    + [getattr(o, "description", "") or "" for o in outcomes]
                ).strip(" ;")
                check_recipe_block = render_check_recipe_block(
                    retrieve_check_recipes(q, domain=dom, k=3))
            except Exception as e:  # noqa: BLE001
                if logger:
                    logger.warning("[BoundaryVerify] check-recipe retrieval failed: %s", e)
        ctx = VerifyContext(
            instruction=self.instruction or "",
            domain=dom,
            subgoal_text=self.current_subgoal or "",
            expected_post_state=self.current_subgoal_expected_post_state or "",
            # actor_history entries are {"screenshot_b64": <~1MB>, "response":
            # text} — take ONLY response text. str(a) would dump the full
            # screenshot b64 as TEXT (~1M chars ≈ 260k tokens) and blow the
            # context window (observed BadRequestError 400).
            actor_history=[
                (a.get("response") if isinstance(a, dict) else str(a)) or ""
                for a in (self.actor_history or [])
            ][-8:],
            screenshot_b64=getattr(cand, "screenshot_b64", None),
            prior_screenshots_b64=prior_shots,
            a11y_text=a11y_text,
            snapshot=self._current_snapshot,
            check_recipe_block=check_recipe_block,
        )
        verdicts = verify_milestone(
            outcomes, ctx, self.call_llm, self.verifier_model,
            env=env, logger=logger)
        for v in verdicts:
            if v.verified:
                o = self.ledger.get(v.outcome_id)
                if o is not None:
                    o.state = OUTCOME_VERIFIED
                    o.last_satisfy_step = step_idx
                    authored_check = None
                    if getattr(v, "authored_check", None) is not None:
                        try:
                            from dataclasses import asdict as _asdict
                            authored_check = _asdict(v.authored_check)
                        except Exception:  # noqa: BLE001
                            authored_check = dict(
                                getattr(v.authored_check, "__dict__", {}) or {})
                    trace = {
                        "source": "boundary_verify",
                        "verified": True,
                        "method": getattr(v, "method", "") or "obs",
                        "reason": getattr(v, "reason", "") or "",
                        "evidence": getattr(v, "evidence", "") or "",
                        "confidence": (
                            "high"
                            if str(getattr(v, "method", "")).startswith("probe:")
                            else "medium"
                        ),
                        "step_idx": step_idx,
                    }
                    if authored_check is not None:
                        trace["authored_check"] = authored_check
                    if hasattr(o, "record_verifier_trace"):
                        o.record_verifier_trace(trace)
                    else:
                        o.last_verifier_trace = trace
        try:
            results_dir = getattr(self, "_results_dir", None)
            if results_dir:
                from pathlib import Path as _Path
                bd = _Path(results_dir) / "boundary_verify_debug"
                bd.mkdir(parents=True, exist_ok=True)
                (bd / dump_name).write_text(json.dumps({
                    "step_idx": step_idx,
                    "subgoal": self.current_subgoal,
                    "targets": dump_targets,
                    "allowed_probes": sorted(allowed_probes_for_domain(dom)),
                    "verdicts": [{
                        "outcome_id": v.outcome_id, "verified": v.verified,
                        "method": v.method, "reason": v.reason,
                        "evidence": v.evidence, "error": v.error,
                        "probe": (v.authored_check.kind
                                  if v.authored_check else None),
                    } for v in verdicts],
                }, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            if logger:
                logger.warning("[BoundaryVerify] debug dump failed: %s", e)
        return verdicts

    def _boundary_verify_done_leaves(self, obs: dict, step_idx: int,
                                     env: Any = None) -> None:
        """verify-on-DONE (boundary mode).

        The per-step KeyNode poll is bypassed, so run the boundary verifier on
        any still-PENDING leaf outcome at task-DONE time. Already-verified
        leaves are trusted. The DONE gate then reads the updated state.
        """
        led = self.ledger
        if led is None:
            return
        try:
            leaf_ids = set(led._leaf_outcome_ids())
        except Exception:
            leaf_ids = {o.id for o in led.required_outcomes}
        pending = [o for o in led.required_outcomes
                   if o.id in leaf_ids and not o.is_verified()]
        if not pending:
            return
        verdicts = self._run_boundary_verify(
            pending, step_idx, env,
            dump_name=f"step_{step_idx:03d}_done.json",
            dump_targets=sorted(leaf_ids))
        if logger:
            nv = sum(1 for v in verdicts if v.verified)
            logger.info(
                "[BoundaryVerify] verify-on-DONE step=%d: %d/%d pending leaf "
                "now verified", step_idx, nv, len(pending))

    def _verify_actor_loop_exit_for_planner(self, ctx, exit_reason: str) -> None:
        """Run milestone verification after actor control returns to planner.

        This replaces the old per-step KeyNode subgoal gate. It updates
        verified ledger outcomes, then writes a one-shot report for the planner.
        It never advances the subgoal, rejects the subgoal, or stages replan.
        """
        step_idx = ctx.step_idx
        env = ctx.env
        targets = list(self.current_subgoal_targets or [])
        state = getattr(self, "_actor_burst_state", None)
        low_level_status = getattr(state, "last_verdict", None) or "unknown"
        low_level_reason = getattr(state, "last_reason", None) or ""

        lines = [
            f"actor_loop_exit_reason: {exit_reason}",
            f"low_level_status: {low_level_status}",
        ]
        if low_level_reason:
            lines.append(f"low_level_reason: {low_level_reason[:500]}")
        lines.append(f"current_subgoal: {(self.current_subgoal or '')[:500]}")

        if self.ledger is None:
            lines.append("milestone_check: skipped because ledger is unavailable")
            self._pending_planner_verifier_report = "\n".join(lines)
            return
        if not targets:
            lines.append(
                "milestone_check: skipped because the current subgoal has no "
                "explicit target outcome id")
            self._pending_planner_verifier_report = "\n".join(lines)
            return

        outcomes = [o for o in self.ledger.required_outcomes if o.id in targets]
        pending = [o for o in outcomes if not o.is_verified()]
        already = [o for o in outcomes if o.is_verified()]
        if already:
            lines.append("already_verified_targets:")
            for o in already:
                lines.append(f"  - {o.id}: {getattr(o, 'description', '')[:180]}")
        if not pending:
            lines.append("milestone_check: all current targets were already verified")
            self._pending_planner_verifier_report = "\n".join(lines)
            return

        verdicts = self._run_boundary_verify(
            pending, step_idx, env,
            dump_name=f"step_{step_idx:03d}_actor_burst_exit.json",
            dump_targets=targets)

        if any(getattr(v, "error", False) for v in verdicts):
            lines.append("milestone_check: verifier unavailable or partially failed")
        verified = [v for v in verdicts if v.verified]
        unverified = [v for v in verdicts if not v.verified]
        if verified:
            lines.append("verified_targets:")
            for v in verified:
                lines.append(
                    f"  - {v.outcome_id}: {v.reason[:220]} | "
                    f"evidence: {v.evidence[:220]}")
        if unverified:
            lines.append("unverified_or_uncertain_targets:")
            for v in unverified:
                lines.append(
                    f"  - {v.outcome_id}: {v.reason[:220]} | "
                    f"evidence: {v.evidence[:220]}")
        if logger:
            nv = sum(1 for v in verdicts if v.verified)
            logger.info(
                "[BoundaryVerify] actor-loop-exit report step=%d: %d/%d "
                "pending target(s) verified", step_idx, nv, len(pending))
        self._pending_planner_verifier_report = "\n".join(lines)

    def _post_actor(
        self,
        actor_response: Optional[str],
        action_code: Optional[str],
        planner_decision: Optional[str],
        planner_done_assessment: Optional[str],
    ) -> None:
        """Finalize this turn's StepRecord (fill in actor + decision),
        close the previous TimelineEvent if appropriate (using THIS
        turn's done/decision re the prev subgoal), and append the
        candidate step to the timeline.

        working-memory only and harmless if ``_pre_planner`` was skipped."""
        if self._memory_version == "v1":
            return
        cand = self._candidate_step
        if cand is None:
            return
        from mm_agents.structagent.ledger.core.timeline import (
            append_step, close_current_event, event_outcome_for_decision,
            EV_ONGOING, EV_ABANDONED_REPLAN,
        )
        # Fill in turn-end fields
        if actor_response:
            cand.actor_action_summary = actor_response.replace("\n", " ").strip()[:240]
            cand.actor_action_raw = action_code if action_code else None
        cand.planner_decision = planner_decision
        cand.planner_done_assessment = planner_done_assessment

        # Close previous event using this turn's done/decision (which
        # refers to prev event's subgoal). Must happen BEFORE append_step
        # so the new step opens a fresh event when subgoal changes.
        prev_outcome = event_outcome_for_decision(planner_decision, planner_done_assessment)
        if prev_outcome != EV_ONGOING and self.timeline:
            # Capture the planner's <Bottleneck> from THIS turn as the
            # closing reason for the event we're about to close — but
            # only when the event is being abandoned (REPLAN). For
            # commit/done outcomes there's no failure to explain.
            close_reason = None
            if prev_outcome == EV_ABANDONED_REPLAN:
                close_reason = (self.planner_parsed or {}).get("bottleneck")
            close_current_event(self.timeline, outcome=prev_outcome,
                                at_step=max(0, cand.step_idx - 1),
                                reason=close_reason)
        append_step(self.timeline, cand, self.current_subgoal or "")
        # consume — clear so a stray reference can't leak across turns
        self._candidate_step = None

    def _pa_run_actor(self, ctx):
        """Actor phase: resolve the subgoal (+ one-shot planner hint), run the
        2-stage re-grounding fast path, call the actor, handle
        WAIT/IMPOSSIBLE/DONE/FAIL, accumulate the ledger window and finalize
        this turn's timeline StepRecord. Always returns (response, [actions])."""
        instruction, obs, env, step_idx = (
            ctx.instruction, ctx.obs, ctx.env, ctx.step_idx)
        screenshot_bytes = ctx.screenshot_bytes
        screen_width = ctx.screen_width
        screen_height = ctx.screen_height
        # 2) ACTOR: single call (retries happen across predict() calls)
        subgoal = self.current_subgoal or "Proceed with the task."
        # Phase-2: pipe the planner's advisory feedback (emitted when the
        # last_subgoal_assessment was done=NO) to the actor as a hint.
        # ONLY for this turn — not mutating current_subgoal.
        if self.feedback_to_actor:
            subgoal = (
                f"{subgoal}\n"
                f"[Planner hint from last turn — apply this on your next attempt]: "
                f"{self.feedback_to_actor}"
            )
            self.feedback_to_actor = None  # one-shot
        subgoal = self._pa_apply_actor_burst_feedback(subgoal)

        actor_subgoal_done = False
        actor_exhausted = False
        final_actions = []
        actor_response = ""

        screenshot_bytes = obs["screenshot"] if isinstance(obs.get("screenshot"), bytes) else screenshot_bytes
        img = Image.open(BytesIO(screenshot_bytes)).convert("RGB")
        img_w, img_h = img.width, img.height
        current_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")


        actor_response = self._pa_call_actor(instruction, subgoal, screenshot_bytes, (img_w, img_h))
        if logger:
            logger.info("[PA-Actor] subgoal_step=%d response=%s",
                        self.current_subgoal_step, actor_response)

        # WAIT handling
        parsed_check = self._pa_parse_actor_response(actor_response, img_w, img_h, screen_width, screen_height, env=env)
        for _wait_retry in range(3):
            if len(parsed_check) == 0:
                break
            if not (len(parsed_check) == 1 and parsed_check[0] == "WAIT"):
                break
            if env is None:
                break
            time.sleep(3)
            wait_obs, _, wait_done, _ = env.step("WAIT")
            if wait_obs is None or wait_done:
                break
            obs = wait_obs
            screenshot_bytes = wait_obs["screenshot"]
            self.actor_history.append({"screenshot_b64": current_b64, "response": actor_response})
            current_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            img = Image.open(BytesIO(screenshot_bytes)).convert("RGB")
            img_w, img_h = img.width, img.height
            actor_response = self._pa_call_actor(instruction, subgoal, screenshot_bytes, (img_w, img_h))
            parsed_check = self._pa_parse_actor_response(actor_response, img_w, img_h, screen_width, screen_height, env=env)

        # Check impossible
        if len(parsed_check) == 1 and parsed_check[0] == "IMPOSSIBLE":
            # Stage a force_replan signal for next turn — rule #2 of
            # _decide_planner_mode_and_reason will pick it up. Burst mode
            # always hands the actor failure back to the planner through
            # the actor-loop verifier report, so the legacy
            # _pending_actor_failure_reason / _actor_emitted_impossible
            # path is not taken here.
            actor_exhausted = True
            self._actor_emitted_impossible = False
            self._consecutive_impossible_count += 1

        self.actor_history.append({"screenshot_b64": current_b64, "response": actor_response})

        # Progress Ledger: accumulate (screenshot, action) into the
        # current window. Fire summarizer periodically every N actor
        # turns (plus once at DONE in the planner-decision branch above).
        # Boundary-tag-based firing (continue/replan/exhaust/max_steps)
        # has been removed — periodic fire covers the common case and
        # DONE fire covers the gate-decision case.
        self._ledger_accumulate(obs, actor_response)
        self._ledger_turns_accumulated += 1
        if (self.ledger is not None
                and self._ledger_turns_accumulated >= self._LEDGER_FIRE_EVERY_N_TURNS):
            self._ledger_fire_boundary("periodic", obs, step_idx)
            self._ledger_turns_accumulated = 0

        # Text-answer benchmarks: detect terminal sentinel in actor
        # response. Accepts BOTH new ``finished(answer="...")`` (the
        # canonical form planner instructs the model to emit) and
        # legacy ``ANSWER(...)`` (kept so wiki_search_executor and any
        # replay against MMInA-era trajectories still terminate
        # correctly). Both populate ``self._text_answer`` and mirror it
        # onto the legacy ``_mmina_answer`` alias for back-compat
        # readers.
        if (self._is_text_answer_task or self._is_mmina_task) \
                and actor_response:
            from mind2web_eval import extract_text_answer
            _ta = extract_text_answer(actor_response)
            if _ta:
                self._text_answer = _ta
                self._mmina_answer = _ta            # legacy mirror
                self.all_done_after_step = True
                if logger:
                    logger.info("[TextAnswer] sentinel detected: %s",
                                _ta[:200])
                _emit = f'finished(answer="{_ta}")'
                self.actions.append(_emit)
                return _emit, ["pyautogui.hotkey('Escape')"]

        # Actor confirmed subgoal done
        if len(parsed_check) == 1 and parsed_check[0] == "DONE":
            actor_subgoal_done = True

        # FAIL
        if len(parsed_check) == 1 and parsed_check[0] == "FAIL":
            actor_exhausted = True

        # Real actions
        if not actor_subgoal_done and not actor_exhausted:
            final_actions = parsed_check
            self._consecutive_impossible_count = 0   # a real action breaks the streak
        elif actor_subgoal_done:
            # Actor-DONE is only a claim. Emit a harmless WAIT so the runner
            # advances to a fresh observation for low-level verify.
            final_actions = ["WAIT"]
        elif actor_exhausted:
            # Actor failure is handed to the planner through the actor-loop
            # verifier report, not the legacy force_replan path.
            final_actions = ["WAIT"]

        self.current_subgoal_step += 1
        self.actor_subgoal_done_for_planner = actor_subgoal_done
        self.actor_exhausted_for_planner = actor_exhausted

        # Record action description (full response for coord extraction in 2-stage grounding)
        self.actions.append(actor_response if actor_response else "")

        # Action overlay: annotate the pre-action screenshot with the
        # decided actor action (CLICK marker, TYPE/HOTKEY/SCROLL labels)
        # and propagate the annotated frame to ALL stores so planner
        # history / actor history / ledger window / timeline / HTML
        # all show the same self-explanatory image.
        try:
            from mm_agents.structagent.utils.screenshot_annotator import annotate_screenshot_with_action
            _pre_action_bytes = obs.get("screenshot") if isinstance(obs, dict) else None
            if _pre_action_bytes and final_actions:
                _annotated_bytes = annotate_screenshot_with_action(_pre_action_bytes, final_actions)
                _annotated_b64 = base64.b64encode(_annotated_bytes).decode("ascii")
                # 1. candidate step (was set with raw bytes in _pre_planner)
                if self._candidate_step is not None:
                    self._candidate_step.screenshot_b64 = _annotated_b64
                # 2. planner_history last entry (re-process so it's
                #    sized for the planner's vision model)
                if self.planner_history:
                    feedback = self.planner_history[-1][1]
                    self.planner_history[-1] = (process_image(_annotated_bytes), feedback)
                # 3. actor_history last entry (last actor saw the raw frame;
                #    we replace with annotated for downstream replay)
                if self.actor_history:
                    self.actor_history[-1]["screenshot_b64"] = _annotated_b64
                # 4. ledger boundary-summarizer window (last appended entry)
                if self._ledger_subgoal_screenshots:
                    self._ledger_subgoal_screenshots[-1] = _annotated_bytes
        except Exception as _ann_e:
            if logger:
                logger.warning("[Annotator] failed to overlay action on screenshot: %s", _ann_e)

        self._pa_record_actor_burst_turn(
            ctx,
            actor_response=actor_response,
            final_actions=final_actions,
            actor_subgoal_done=actor_subgoal_done,
            actor_exhausted=actor_exhausted,
        )

        # finalize this turn's StepRecord and append to timeline
        _done_assess = None
        if isinstance(self.planner_response, str):
            _m = re.search(r"<done>\s*(YES|NO)\s*</done>", self.planner_response)
            _done_assess = _m.group(1) if _m else None
        _act_code = (final_actions[0] if final_actions else None)
        self._post_actor(
            actor_response=actor_response,
            action_code=_act_code,
            planner_decision=getattr(self, "planner_decision", None),
            planner_done_assessment=_done_assess,
        )

        return actor_response, final_actions

    def _pa_begin_step(self, ctx):
        """Per-step setup before the planner: cache instruction + step/
        screenshot, fetch the cli-run sentinel, resolve the domain, run the
        multi-app phase-advance / domain-switch / physical reconcile, pre-upload
        the ops lib, probe initial files, augment the instruction, re-probe the
        open document (SP/DP/Slide) and init the ledger. Populates ctx with the
        loop-locals the rest of the step needs (step_idx, screenshot_bytes,
        screen_width, screen_height) and the augmented instruction."""
        instruction = ctx.instruction
        obs = ctx.obs
        env = ctx.env
        wall_step_idx = ctx.wall_step_idx
        # Cache the task instruction so per-turn perceivers (notably
        # slide_perceiver) can extract the slide focus once.
        self._task_instruction = instruction
        # Stash the outer step counter so per-step debug dumps
        # (perceiver artifacts, planner messages) can name files correctly.
        self._wall_step_idx = wall_step_idx
        step_idx = len(self.actions)
        screenshot_bytes = obs["screenshot"]
        screen_img = Image.open(BytesIO(screenshot_bytes))
        screen_width, screen_height = screen_img.size

        # Clear DONE-rejection reason at the start of every turn so HTML /
        # logs only display it on the step where the ledger gate actually
        # rejected. Without this, ``_ledger_last_done_reason`` is set once
        # at the rejection step and persists to every subsequent step's
        # render — a stale reason appearing long after the cache state
        # has evolved.
        self._ledger_last_done_reason = None

        # cli_run / structured-action output retrieval — if the most
        # recent action was a cli_run or a calc_* / impress_* / writer_*
        # structured action, fetch ``/tmp/_last_cli_run.json`` from the
        # VM so the planner can see its stdout/stderr/exit. Without this
        # the planner only sees the post-action screenshot, which
        # doesn't surface command output (cli_run runs via
        # subprocess.run with capture_output=True; structured actions
        # exec() with stdout redirect). Stash on self for the
        # planner-prompt builder; cleared again after the planner
        # consumes it.
        self._last_cli_run_output = None
        # ``actor_history`` is cleared on every subgoal advance, but
        # ``self.actions`` is the cumulative action summary log so it
        # survives that — use it to detect whether the most recent
        # action was a cli_run / calc_* / impress_* / writer_*. The
        # structured actions are built server-side by their respective
        # ``action_to_runtime_script`` into a UNO runtime script, so
        # their stdout/stderr lands in the same ``/tmp/_last_cli_run.json``
        # sentinel and they mutate the open document — gate identically.
        last_action_summary = (self.actions[-1] if self.actions else "") or ""
        from mm_agents.structagent.actions.calc.actions import CALC_ACTION_NAMES as _CALC_NAMES
        from mm_agents.structagent.actions.impress.actions import IMPRESS_ACTION_NAMES as _IMPRESS_NAMES
        from mm_agents.structagent.actions.writer.actions import WRITER_ACTION_NAMES as _WRITER_NAMES
        last_is_calc_action = any(
            f"{n}(" in last_action_summary for n in _CALC_NAMES
        )
        last_is_impress_action = any(
            f"{n}(" in last_action_summary for n in _IMPRESS_NAMES
        )
        last_is_writer_action = any(
            f"{n}(" in last_action_summary for n in _WRITER_NAMES
        )
        is_cli_run_action = (
            "cli_run(" in last_action_summary
            or "open_app(" in last_action_summary
            or last_is_calc_action
            or last_is_impress_action
            or last_is_writer_action
        )
        if env is not None and is_cli_run_action:
            try:
                raw = env.controller.get_file("/tmp/_last_cli_run.json")
                if raw:
                    import json as _json
                    self._last_cli_run_output = _json.loads(raw.decode("utf-8", "replace"))
            except Exception as e:
                if logger:
                    logger.debug("[cli_run] failed to fetch sentinel: %s", e)
            # DocPerceiver / SheetPerceiver / SlidePerceiver: a
            # calc_* / impress_* / writer_* action may have mutated the
            # doc — mark the cached structure block stale so the next
            # turn re-probes before building the planner prompt.
            if (last_is_calc_action
                    or last_is_impress_action
                    or last_is_writer_action):
                self._doc_inspect_dirty = True

        # Extract domain from instruction prefix like "[libreoffice_writer] ..."
        # Tolerate space-separated variants like "[libreoffice calc]" —
        # OSWorld task metadata uses both; we normalize to underscore form
        # so handler routing (which keys on "libreoffice_calc" etc.) works.
        if self._current_domain is None:
            domain_m = re.match(r'^\[([^\]]+)\]', instruction or "")
            if domain_m:
                raw = domain_m.group(1).strip().lower().replace(" ", "_")
                # ``[multi_apps]`` is not a real domain — it's the tag
                # for multi-app tasks. Don't route on it; leave
                # _current_domain unset (lib_run_single sets it from
                # related_apps[0] before predict() in the normal path).
                if raw != "multi_apps":
                    self._current_domain = raw

        # MULTI-APP context switch (D+E+F+G). For a multi_apps task,
        # re-point ``_current_domain`` to the app the agent is actually
        # OPERATING IN this turn. Everything app-specific then follows:
        #   • ops-lib pre-upload below (G) re-uploads the new app's lib
        #   • the perceiver block (F) re-probes via should_probe(domain)
        #   • domain_knowledge / evidence_guide / structured-action
        #     setup (E) are all rebuilt from _current_domain each turn
        #
        # Source priority:
        #   1. PLANNER — the current subgoal's ``<step app="...">`` tag.
        #      The planner organizes a multi-app task as per-app phases
        #      and knows which app the current step runs in; this is the
        #      EXECUTION-context signal and is authoritative.
        #   2. VERIFIER (fallback) — the ledger focus-outcome's app.
        #      Used only when the planner didn't tag the step. The
        #      focus outcome is "the next deliverable to verify", which
        #      can differ from "the app being operated" (a Writer
        #      deliverable may need prerequisite work back in Calc), so
        #      it is a fallback, not the primary driver.
        #
        # Gated on _is_multi_app() — single-app tasks skip this entirely,
        # so _current_domain stays fixed exactly as before. Step 0 has no
        # subgoal/ledger yet → no switch → initial domain stays
        # related_apps[0].
        #
        # Phase advance runs FIRST: if the active phase's outcomes all
        # verified, summarize its cross-app handoff and flip to the next
        # phase, so the domain switch below reads the fresh active phase.
        if self._is_multi_app():
            try:
                from mm_agents.structagent.ledger.core.phase_board import (
                    advance_phase_if_complete,
                )
                _pb = self._phase_board()
                if _pb is not None and self.ledger is not None:
                    _ps = getattr(self, "_phase_start_action_idx", 0)
                    # Pull fresh observable state for the phase auditor
                    # (gated on enable_done_auditor — phase audit shares
                    # the same opt-in as end-of-task DoneAuditor since
                    # they are philosophically the same: adversarial
                    # recheck at a completion boundary). Screenshot is
                    # base64 of the current observation; a11y is what
                    # the perceiver last extracted; doc_inspect_block
                    # is the UNO probe for office tasks. None on any
                    # missing field — the auditor handles None
                    # gracefully and omits the section.
                    _audit_screenshot_b64 = None
                    try:
                        import base64 as _b64
                        _ss = obs.get("screenshot") if isinstance(obs, dict) else None
                        if _ss:
                            _audit_screenshot_b64 = _b64.b64encode(_ss).decode("ascii")
                    except Exception:
                        _audit_screenshot_b64 = None
                    _audit_a11y = None
                    try:
                        _audit_a11y = getattr(self, "_planner_a11y_at_last_turn", None)
                    except Exception:
                        _audit_a11y = None
                    _advanced = advance_phase_if_complete(
                        _pb,
                        verified_ids={o.id for o in self.ledger.verified()},
                        current_subgoal_app=getattr(
                            self, "current_subgoal_app", None),
                        phase_actions=list((self.actions or [])[_ps:]),
                        call_llm=self.call_llm, model=self.verifier_model,  # phase-completion check → verifier model
                        logger=logger,
                        enable_audit=getattr(
                            self, "enable_done_auditor", False),
                        current_screenshot_b64=_audit_screenshot_b64,
                        current_a11y=_audit_a11y,
                        doc_inspect_block=getattr(
                            self, "_doc_inspect_block", None),
                    )
                    if _advanced:
                        self._phase_start_action_idx = len(self.actions or [])
                    # Per-turn status line — greppable phase state in
                    # runtime.log every turn (`grep "\[PhaseBoard\]"`).
                    logger.info("[PhaseBoard] %s", _pb.summary_line())
            except Exception as _e:
                logger.warning("[Phase] advance check failed: %s", _e)
        if self._is_multi_app():
            try:
                _new_dom, _switch_src = self._pick_multi_app_domain()
                if _new_dom and _new_dom != self._current_domain:
                    logger.info(
                        "[MultiApp] domain switch %s -> %s (%s)",
                        self._current_domain, _new_dom, _switch_src,
                    )
                    self._current_domain = _new_dom
                    # Force the perceiver to re-probe for the new app.
                    self._doc_inspect_dirty = True
                    # Auto-focus the new app's window so the actor never
                    # has to switch windows through the GUI — window
                    # juggling (dock-click / Alt+Tab) is the largest
                    # source of multi-app churn. Best-effort.
                    self._activate_app_for_domain(env, _new_dom)
            except Exception as _e:
                logger.warning(
                    "[MultiApp] domain switch check failed: %s", _e)

            # R1 + R3 — observe PHYSICAL app and reconcile against
            # INTENDED. Decoupled from _current_domain (which is
            # intent). On mismatch: push physical toward intent
            # best-effort (auto-focus), and after N consecutive
            # mismatches give up and demote intent to physical so the
            # planner isn't stuck waiting for a window switch that
            # never happens (R3 demote rule).
            try:
                from mm_agents.structagent.actions.app.active_app import observe_active_app
                _a11y_raw = (obs.get("accessibility_tree")
                             if isinstance(obs, dict) else None)
                _phys_new = observe_active_app(_a11y_raw)
                # Every-step status — even when unchanged. Cheap (one log
                # line) and gives us a clear signal whether R1 is firing
                # at all. Includes a11y size so we can spot empty
                # observations.
                logger.info(
                    "[MultiApp] physical=%s intended=%s "
                    "(a11y_xml_chars=%d, mismatch_run=%d)",
                    _phys_new, self._current_domain,
                    len(_a11y_raw) if _a11y_raw else 0,
                    self._domain_mismatch_steps,
                )
                if _phys_new != self._physical_domain:
                    logger.info(
                        "[MultiApp] physical app %s -> %s",
                        self._physical_domain, _phys_new,
                    )
                self._physical_domain = _phys_new
                if (_phys_new is not None and self._current_domain is not None
                        and _phys_new != self._current_domain):
                    self._domain_mismatch_steps += 1
                    logger.info(
                        "[MultiApp] mismatch: intended=%s physical=%s "
                        "(run=%d)",
                        self._current_domain, _phys_new,
                        self._domain_mismatch_steps,
                    )
                    # Best-effort: try to bring intended app forward.
                    # _activate_app_for_domain is idempotent + silent
                    # on failure; the next turn's observation tells us
                    # if it worked.
                    self._activate_app_for_domain(env, self._current_domain)
                    from mm_agents.structagent.actions.app.active_app import (
                        app_switch_verify_enabled, switch_failure_message,
                    )
                    if app_switch_verify_enabled():
                        # Closed loop (§8): NEVER silently rewrite the agent's
                        # intent to match reality. After 2 mismatch turns with
                        # a verified-failed switch, SURFACE the failure to the
                        # planner (force_replan next turn) so the decision-
                        # maker replans around the real cause.
                        parsed = getattr(self, "_last_app_switch", None) or {}
                        if (self._domain_mismatch_steps >= 2
                                and parsed.get("domain") == self._current_domain
                                and parsed.get("ok") is False):
                            reason = switch_failure_message(
                                self._current_domain, parsed)
                            self._pending_actor_failure_reason = reason
                            logger.warning("[MultiApp] %s", reason)
                            self._domain_mismatch_steps = 0
                    elif self._domain_mismatch_steps >= 5:
                        # Legacy behaviour (flag off): after N=5 turns of failed
                        # reconciliation, accept reality: planner should work
                        # with what's actually focused rather than spinning.
                        logger.info(
                            "[MultiApp] demote intended %s -> physical %s "
                            "after %d mismatch turns",
                            self._current_domain, _phys_new,
                            self._domain_mismatch_steps,
                        )
                        self._current_domain = _phys_new
                        self._doc_inspect_dirty = True
                        self._domain_mismatch_steps = 0
                else:
                    self._domain_mismatch_steps = 0
            except Exception as _e:
                logger.warning(
                    "[MultiApp] physical observe failed: %s", _e)

        # Pre-upload the per-domain ops library to the VM as a module
        # (libreoffice_calc / libreoffice_impress). Once uploaded, every
        # downstream structured-action + verify-shell dispatch becomes a tiny
        # `from <mod> import *` shim instead of inlining ~120KB. Safe
        # no-op for non-LO domains, envs without run_python_script, or
        # repeated calls (cached by content hash).
        if env is not None and self._current_domain:
            try:
                from mm_agents.structagent.actions.uno._ops_lib_remote import maybe_upload_ops_lib
                maybe_upload_ops_lib(env, self._current_domain)
            except Exception as _e:
                logger.warning("[ops_lib_remote] pre-upload failed: %s", _e)

        # Initial-file registry (multi-app only): observe which document
        # files are open at task start, via ONE wmctrl/lsof probe of the
        # running env (a legitimate environment observation — it does NOT
        # read the task spec). Lets open_app re-open the right file by
        # path when later called without one. Runs once; gated on
        # _is_multi_app() so single-app tasks stay byte-identical.
        if (env is not None and self._is_multi_app()
                and not self._initial_files_probed):
            self._initial_files_probed = True
            try:
                from mm_agents.structagent.actions.app.open_app import (
                    build_initial_files_probe_script,
                    parse_initial_files_probe,
                )
                _runner = self._vm_python_runner(env)
                if _runner is not None:
                    _res = _runner(build_initial_files_probe_script())
                    _out = ((_res.get("output") or _res.get("message") or "")
                            if isinstance(_res, dict) else "")
                    self._initial_open_files = parse_initial_files_probe(_out)
                    logger.info(
                        "[InitialFiles] observed at task start: %s",
                        self._initial_open_files,
                    )
            except Exception as _e:
                logger.warning("[InitialFiles] probe failed: %s", _e)

        # Augment instruction once per task so the ledger initializer sees
        # date-resolved + task-template-extracted fields (e.g. "destination:
        # NYC") rather than only the raw place name ("New York"). The same
        # enriched string is cached in `self.instruction` and reused by the
        # initial planner prompt downstream.
        if getattr(self, "instruction", None) is None:
            self._raw_instruction = instruction
            instruction = self._pa_augment_instruction(instruction)
            self.instruction = instruction
        else:
            instruction = self.instruction

        # DocPerceiver / SheetPerceiver: re-probe the open document via
        # UNO if the cache is dirty (task start OR a cli_run_uno mutated
        # the doc on the previous turn). Only fires when domain is in the
        # doc-route allowlist (libreoffice_writer / libreoffice_calc).
        # Read-only — uses env.controller.run_python_script, separate
        # channel from the actor's cli_run stream so it can't corrupt
        # state.
        # Runs BEFORE `_ledger_init_if_needed` so the resulting
        # `self._doc_inspect_block` can be plumbed into init_ledger's
        # prompt for office domains (replaces the noisy linearized a11y
        # tree with the SP/DP structured block — a11y can't
        # distinguish FORMULA vs VALUE vs EMPTY cells, which is exactly
        # the signal init_ledger needs to pick the right answer cell).
        if env is not None and self._doc_inspect_dirty:
            try:
                from mm_agents.structagent.perception import doc_perceiver, sheet_perceiver, slide_perceiver
                # Pick the right probe for the current domain. Writer →
                # DocPerceiver (paragraphs/tables/cursor). Calc →
                # SheetPerceiver (sheets/cells/formulas/charts). Impress
                # → SlidePerceiver (slides/shapes/font properties). All
                # three write artifacts under <results_dir>/<probe_name>/.
                results_dir = getattr(self, "_results_dir", None)
                # Fall back to the local ``step_idx`` (= len(self.actions))
                # when the caller didn't pass ``wall_step_idx`` through
                # ``predict()``. Without this, the SP/DP-block writer's
                # ``step_idx:03d`` formatter crashes AND every downstream
                # ``step_idx > 0`` comparison raises TypeError.
                step_idx = getattr(self, "_wall_step_idx", None) or step_idx
                blk = None
                if doc_perceiver.should_probe(self._current_domain):
                    blk = doc_perceiver.run_inspect(
                        env, logger=logger,
                        results_dir=results_dir, step_idx=step_idx,
                    )
                elif sheet_perceiver.should_probe(self._current_domain):
                    blk = sheet_perceiver.run_inspect(
                        env, logger=logger,
                        results_dir=results_dir, step_idx=step_idx,
                    )
                elif slide_perceiver.should_probe(self._current_domain):
                    # Impress: split probe + render so focus_slides
                    # (the set of slide indices the task targets) can
                    # be extracted ONCE per task via a lightweight
                    # LLM call and reused on every subsequent probe.
                    # Off-focus slides collapse to a one-line summary,
                    # keeping multi-slide deck SP blocks small.
                    parsed = slide_perceiver.probe(env, logger=logger)
                    if parsed is not None:
                        if self._impress_focus_slides is None:
                            try:
                                # Adapter: slide_perceiver expects a
                                # call_llm(messages=, model=, temperature=,
                                # max_tokens=) callable; self.call_llm uses
                                # a payload dict.
                                def _focus_llm(messages, model, temperature=0.0,
                                               max_tokens=64):
                                    return self.call_llm(
                                        {"model": model, "messages": messages,
                                         "max_tokens": max_tokens,
                                         "top_p": 0.9,
                                         "temperature": temperature},
                                        model,
                                    )
                                self._impress_focus_slides = (
                                    slide_perceiver.extract_focus_slides(
                                        self._task_instruction, parsed,
                                        call_llm=_focus_llm,
                                        model=self.model,
                                        logger=logger,
                                    )
                                )
                                if logger:
                                    logger.info(
                                        "[SlidePerceiver] focus extracted: %r",
                                        self._impress_focus_slides)
                            except Exception as e:
                                if logger:
                                    logger.info(
                                        "[SlidePerceiver] focus extraction "
                                        "failed: %s", e)
                                self._impress_focus_slides = None
                        blk = slide_perceiver.render_block(
                            parsed,
                            focus_slides=self._impress_focus_slides,
                        )
                        # Persist the block to disk for offline review.
                        if results_dir is not None and step_idx is not None:
                            try:
                                import os as _os_local
                                d = _os_local.path.join(
                                    results_dir, "slide_perceiver")
                                _os_local.makedirs(d, exist_ok=True)
                                with open(_os_local.path.join(
                                        d, f"step_{step_idx:03d}_block.txt"),
                                        "w") as _f:
                                    _f.write(blk)
                            except Exception:
                                pass
                self._doc_inspect_block = blk    # None on failure → no block injected
                self._doc_inspect_dirty = False
            except Exception as e:
                if logger:
                    logger.info("[Perceiver] probe call raised: %s", e)
                self._doc_inspect_dirty = False      # don't retry-loop on permanent failures

        # Progress Ledger: initialize on the first call of this task.
        # Pass env so init_ledger can run a one-shot probe of the VM
        # (binaries, config files, GSettings schemas, processes) and
        # ground the ledger in actual environment state. For office
        # domains, also pass the SP/DP block produced above so the
        # init author has the same structured cell/paragraph data the
        # planner sees on every step.
        self._ledger_init_if_needed(instruction, obs, env=env)
        ctx.step_idx = step_idx
        ctx.screenshot_bytes = screenshot_bytes
        ctx.screen_width = screen_width
        ctx.screen_height = screen_height
        ctx.instruction = instruction

    def _pa_advance_or_finish(self, ctx, plan, planner_decision,
                              last_subgoal_done):
        """Subgoal-queue maintenance + terminal decisions: install/advance the
        queue, run the DONE-acceptance gate (+ Done-Auditor) and re-plan on
        reject, emit DONE / INFEASIBLE, and the post-REPLAN feasibility verdict.
        Returns a (response, [actions]) tuple to TERMINATE the step (DONE /
        INFEASIBLE / feasibility / DONE-rejected-defer), or None to continue."""
        instruction, obs, env, step_idx = (
            ctx.instruction, ctx.obs, ctx.env, ctx.step_idx)
        # --- subgoal queue maintenance ---
        if step_idx == 0:
            if plan is not None:
                parsed_steps = self._pa_parse_plan_into_subgoals(plan)
                self.subgoal_queue = [s["text"] for s in parsed_steps]
                self.subgoal_expected_post_states = [
                    s["expected_post_state"] for s in parsed_steps]
                self.subgoal_apps = [
                    s.get("app") for s in parsed_steps]
                self.subgoal_targets = [
                    s.get("targets") or [] for s in parsed_steps]
                self._commit_strategy_change(step_idx)
            popped, _ = self._pop_next_subgoal(step_idx=step_idx)
            self._pa_reset_actor_burst_state()
            # T4: snapshot disk plan after initial plan install +
            # first subgoal pop (so plan.md reflects the focus on disk
            # before the actor runs). See Part I.3.
            self._write_plan_snapshot(step_idx)
            if not popped:
                self.current_subgoal = "Complete the task."
                self.current_subgoal_expected_post_state = None
                self.current_subgoal_app = None
                self.current_subgoal_targets = []
        else:
            # NOTE: subgoal-step-budget stuck detection is now handled
            # UPFRONT by ``_decide_planner_mode_and_reason`` rule #4 —
            # if the budget is burned, this turn's mode is already
            # "force_replan" and the planner call above produced the
            # recovery plan in a single call (no more same-turn 2nd
            # planner call here).

            # DONE-acceptance gate (audit + cache). The ONLY trigger
            # that needs the planner's output as input — fires here,
            # AFTER the planner returned DONE. On rejection we re-call
            # the planner ONCE in force_replan mode within the same
            # turn (the only path that produces a 2nd planner call).
            if planner_decision == "DONE":
                accepted, reject_reason = self._check_done_acceptance(
                    env, obs, step_idx)
                # T3 Done-Auditor — adversarial second opinion on the
                # accepted DONE. Only fires when the structured gate
                # already passed AND the flag is on AND per-task
                # budget is non-zero. Explicit FAIL flips accepted to
                # False and stages "done_audit_failed"; PARTIAL means
                # the auditor lacked enough evidence to overturn the
                # structured gate. See plan Part I.2.
                if (accepted
                        and self.enable_done_auditor
                        and self._done_audit_budget_remaining > 0):
                    from mm_agents.structagent.core.final_audit import audit_done
                    self._done_audit_budget_remaining -= 1
                    audit_verdict, audit_reason, audit_result = audit_done(self)
                    if logger:
                        logger.info(
                            "[DoneAuditor] step=%d verdict=%s reason=%s",
                            step_idx, audit_verdict,
                            (audit_reason or "")[:200])
                    # Persist full reasoning to <results_dir>/audit_debug/
                    # for offline triage. Best-effort.
                    try:
                        results_dir = getattr(self, "_results_dir", None)
                        if results_dir:
                            from pathlib import Path as _Path
                            audit_dir = _Path(results_dir) / "audit_debug"
                            audit_dir.mkdir(parents=True, exist_ok=True)
                            ad_path = audit_dir / (
                                f"step_{step_idx:03d}_audit.json")
                            ad_path.write_text(json.dumps({
                                "step_idx": step_idx,
                                "verdict": audit_verdict,
                                "reason": audit_reason,
                                "response_text": audit_result.response_text,
                                "llm_meta": audit_result.llm_meta,
                                "error": audit_result.error,
                                "budget_remaining_after":
                                    self._done_audit_budget_remaining,
                            }, indent=2), encoding="utf-8")
                    except Exception as e:  # noqa: BLE001
                        if logger:
                            logger.warning(
                                "[DoneAuditor] audit_debug dump failed: %s", e)
                    # Auditor UNAVAILABLE ≠ refutation. ``error`` set means the
                    # call itself failed (timeout / connection / exception);
                    # ``verdict is None`` means the response carried no parseable
                    # PASS/FAIL/PARTIAL. Either way there is NO second opinion —
                    # keep the structured gate's acceptance (fail-open, same as
                    # the phase-handoff audit) instead of converting an infra
                    # blip into a forced replan of a finished task. (One run had
                    # 27 tasks' accepted DONEs refuted purely by API timeouts.)
                    _auditor_unavailable = bool(
                        getattr(audit_result, "error", None)
                        or getattr(audit_result, "verdict", None) is None)
                    if _auditor_unavailable:
                        if logger:
                            logger.info(
                                "[DoneAuditor] step=%d auditor UNAVAILABLE (%s)"
                                " — keeping gate acceptance",
                                step_idx,
                                str(audit_result.error or "unparseable verdict")[:120])
                    elif audit_verdict == "FAIL":
                        accepted = False
                        reject_reason = (
                            f"done audit refuted ({audit_verdict}): "
                            f"{audit_reason}"
                        )
                        self._force_replan_category_pending = (
                            "done_audit_failed")
                        self._done_audit_reject_count += 1
                        # Mirror into the legacy counter that
                        # _decide_planner_mode_and_reason's downstream
                        # consumers may already be reading.
                        self._done_rejected_count = (
                            getattr(self, "_done_rejected_count", 0) + 1)
                    elif audit_verdict == "PARTIAL" and logger:
                        logger.info(
                            "[DoneAuditor] step=%d PARTIAL is inconclusive; "
                            "keeping gate acceptance",
                            step_idx)
                if not accepted:
                    self._force_replan_reason = reject_reason
                    plan2, decision2 = self._pa_call_planner(
                        instruction, obs, step_idx, env=env,
                        mode="force_replan",
                    )
                    if plan2:
                        self.current_plan = plan2
                        plan = plan2
                        planner_decision = "REPLAN"
                    else:
                        # Planner gave no <plan> on the rejection retry.
                        # Re-stage the reason and defer one turn so the
                        # next turn's mode-decision rule #3 fires
                        # force_replan again (avoids re-running the
                        # actor on the now-stale current_subgoal which
                        # would silently undo work — e.g. uncheck a
                        # filter that was already toggled correctly).
                        if logger:
                            logger.info(
                                "[Planner] DONE rejected with no new "
                                "plan; deferring to next turn's "
                                "force_replan (no actor call)."
                            )
                        self._force_replan_reason = reject_reason
                        self.actions.append("WAIT (DONE rejected, deferring)")
                        return ("DONE rejected by ledger; "
                                "deferring to next turn for replan.",
                                ["WAIT"])

            if planner_decision == "DONE":
                self.all_done_after_step = True
                # Fallback: planner emits DONE without the actor ever
                # routing through the sentinel detection above. Scan
                # planner_response for the answer so the runner still
                # has something to score.
                if ((self._is_text_answer_task or self._is_mmina_task)
                        and not self._text_answer
                        and self.planner_response):
                    from mind2web_eval import extract_text_answer
                    _ta = extract_text_answer(self.planner_response)
                    if _ta:
                        self._text_answer = _ta
                        self._mmina_answer = _ta    # legacy mirror
                if logger:
                    logger.info("[PA-Planner] DONE — task complete, returning terminate action.")
                self.actions.append("DONE (planner)")
                return "Planner decided DONE — task is complete.", ["pyautogui.hotkey('Escape')"]

            elif planner_decision == "INFEASIBLE":
                # Agent's own task-level infeasibility verdict (only offered by
                # the stuck/force-replan contract after ≥2 abandoned approaches).
                # Emit a terminal FAIL action so env.action_history ends with
                # FAIL — exactly what OSWorld's "infeasible" evaluator scores 1.0.
                # Crucially we do NOT set all_done_after_step: that path ends the
                # episode WITHOUT executing any action (see lib_run_single), which
                # would drop the FAIL. Returning ["FAIL"] makes the loop run
                # env.step("FAIL"), which itself sets done=True and stops.
                if logger:
                    logger.info("[PA-Planner] INFEASIBLE — task judged impossible, emitting FAIL.")
                self.actions.append("FAIL (planner: infeasible)")
                return "Planner decided INFEASIBLE — task cannot be completed.", ["FAIL"]

            elif planner_decision == "CONTINUE":
                if last_subgoal_done is True:
                    # Advance queue: current subgoal verified done, move to next.
                    # (Ledger summarizer is now fired periodically every N
                    # actor turns + at DONE, not at every CONTINUE boundary.)
                    # Reset the "stuck on self-REPLAN without progress"
                    # counter — a freshly-completed subgoal is the signal
                    # that the task is advancing again.
                    self._replans_since_last_progress = 0
                    if self.current_subgoal:
                        self.completed_subgoals.append(self.current_subgoal)
                    popped, _skipped = self._pop_next_subgoal(step_idx=step_idx)
                    self._pa_reset_actor_burst_state()
                    if not popped:
                        # Queue empty mid-task — install end-state
                        # placeholder. The next turn's mode-decision
                        # rule #5 will fire force_replan if it's still
                        # active (i.e. this turn's actor didn't push
                        # the task to DONE).
                        self.current_subgoal = "All planned subgoals completed. Please confirm task is done or add missing steps."
                        self.current_subgoal_expected_post_state = None
                        self.current_subgoal_app = None
                        self.current_subgoal_targets = []
                    self._ledger_subgoal_plan_step = self.current_subgoal or ""
                # else: stay on the same subgoal; actor will retry with
                # the planner's feedback_to_actor hint piped in below.

            elif planner_decision == "REPLAN" and plan is not None:
                # Ledger summarizer is now fired periodically + at DONE;
                # this branch only swaps the queue.
                parsed_steps = self._pa_parse_plan_into_subgoals(plan)
                self.subgoal_queue = [s["text"] for s in parsed_steps]
                self.subgoal_expected_post_states = [
                    s["expected_post_state"] for s in parsed_steps]
                self.subgoal_apps = [
                    s.get("app") for s in parsed_steps]
                self.subgoal_targets = [
                    s.get("targets") or [] for s in parsed_steps]
                self._pop_next_subgoal(step_idx=step_idx)
                self._pa_reset_actor_burst_state()
                self._ledger_subgoal_plan_step = self.current_subgoal or ""
                # T4: snapshot disk plan after REPLAN swap. See Part I.3.
                self._write_plan_snapshot(step_idx)

                # REPLAN cadence counters — track "planner self-REPLANs
                # without committing any subgoal." The actor-side
                # subgoal-step-cap can't catch this because each REPLAN
                # resets current_subgoal_step to 0. The threshold check
                # itself + side effects (abandon + ledger.add_failed_path
                # + reason composition) live in
                # _decide_planner_mode_and_reason rule #6, fired at the
                # START of the next turn. Here we just bump the counters.
                self._total_replan_count += 1
                self._replans_since_last_progress += 1
                # Two-layer FailedPath: detect strategy switch + log
                # an event into ``self._strategy_history``. The curator
                # below decides whether/how this becomes a recorded
                # FailedPath.
                self._commit_strategy_change(step_idx)
                # LLM dead-end curator: re-derive ledger.failed_paths
                # from the FULL strategy + action-stuck history. Runs
                # once per REPLAN commit. The curator can drop /
                # merge / revise entries it previously wrote — so a
                # "different-looking" strategy that's actually the
                # same approach as one already in the list won't
                # produce a fuzzy duplicate.
                self._curate_failed_paths(step_idx)
                # Fix 1 — evidence-driven feasibility verdict. After K
                # cumulative force-replans, ask the verifier model whether the
                # task is genuinely INFEASIBLE (emit FAIL) rather than worth
                # another replan. Conservative: only a DIRECT "feature/resource
                # absent" verdict returns FAIL; everything else falls through
                # and the freshly-installed plan runs as normal.
                _fv = self._maybe_feasibility_verdict(step_idx)
                if _fv is not None:
                    return _fv
        return None

    def _pa_run_planner(self, ctx):
        """Planner phase: decide the mode (initial / force_replan /
        progress_check), run the failure-attribution v2 pre-replan hook (which
        may skip the planner call via actor_redo / verifier_override / env_
        intervention), otherwise call the planner once. Sets self.planner_parsed
        / self.feedback_to_actor and returns
        (plan, planner_decision, parsed, last_subgoal_done)."""
        instruction, obs, env, step_idx = (
            ctx.instruction, ctx.obs, ctx.env, ctx.step_idx)
        # === PLANNER: mode-driven, one call per step ====================
        # Decide UPFRONT which prompt branch this turn needs (initial /
        # force_replan / progress_check). Side effects of force_replan
        # (abandoned_subgoals append, ledger.add_failed_path, counter
        # resets) happen inside the helper so the caller doesn't need
        # to re-derive them. See _decide_planner_mode_and_reason for
        # the full rule list.
        mode, mode_reason = self._decide_planner_mode_and_reason(
            env, obs, step_idx)
        if mode == "force_replan" and mode_reason:
            # Stage for _pa_build_subgoal_check_prompt → <stuck_reason>.
            self._force_replan_reason = mode_reason
        # T1: snapshot current counters + transition receipt onto
        # ``self._recovery_state`` for tests + future T3 escalation reads.
        # Read-only mirror — does not change dispatcher behavior.
        self._finalize_recovery_state(mode, step_idx)
        if logger:
            logger.info(
                "[PA-Planner-Mode] step=%d mode=%s reason=%s",
                step_idx, mode,
                (mode_reason[:160] if mode_reason else ""))

        # ── Failure-attribution v2 pre-replan hook ────────────────────
        # When ``USE_FAILURE_ATTRIBUTION=1`` and we're about to enter
        # the planner's force_replan path, run diagnose+router FIRST.
        # If the router decides ``actor_redo``, we:
        #   • stash the prose-level recovery payload for the actor
        #   • synthesize a CONTINUE decision so the existing
        #     subgoal_queue is kept
        #   • consume the force_replan state flags
        #   • SKIP the planner LLM call entirely (one fewer LLM call,
        #     and the planner can't accidentally re-plan around an
        #     actor-side glitch)
        # Otherwise we just cache the router decision on self so the
        # planner's force_replan prompt builder reuses it instead of
        # re-calling diagnose_failure (preventing duplicate LLM calls).
        _skipped_planner = False
        # USE_FAILURE_ATTRIBUTION: off/unset = v1 (production default);
        # on = v2 LIVE — verifier_override actually flips the focus
        # outcome to VERIFIED via a synthetic KeyNode. The dry-run middle
        # tier is gone; safety gates inside the router still apply.
        _attribution_active = self.cfg.failure_attribution
        _verifier_override_live = _attribution_active
        # IMPOSSIBLE is a definitive actor verdict — skip the diagnose+route hook
        # (which would mislabel it role=actor → actor_redo, retry the impossible,
        # and never bump _total_replan_count). Consume the flag each planner turn;
        # falling through runs a normal planner force_replan (REPLAN → count++).
        _impossible_skip = getattr(self, "_actor_emitted_impossible", False)
        self._actor_emitted_impossible = False
        if (mode == "force_replan"
                and self.ledger is not None
                and getattr(self, "enable_stuck_diagnosis_injection",
                            False)
                and _attribution_active
                and not _impossible_skip):
            try:
                from mm_agents.structagent.attribution.failure_attribution import (
                    diagnose_failure,
                )
                from mm_agents.structagent.attribution.failure_attribution_router import (
                    route_failure_attribution,
                )
                _focus = self.ledger.next_focus_outcome()
                if _focus is not None:
                    _diag = diagnose_failure(
                        focus_outcome=_focus,
                        ledger=self.ledger,
                        observation=obs,
                        timeline=getattr(self, "timeline", []) or [],
                        stuck_category=getattr(self,
                                                "_force_replan_category",
                                                None) or "actor_failure",
                        stuck_reason=mode_reason,
                        call_llm=self.call_llm,
                        model=self.model,
                        results_dir=getattr(self, "_results_dir", None),
                        step_idx=getattr(self, "_wall_step_idx",
                                         0) or 0,
                        current_domain=getattr(self, "_current_domain",
                                                None),
                        last_cli_run_output=getattr(
                            self, "_last_cli_run_output", None),
                    )
                    if _diag is not None:
                        _decision = route_failure_attribution(
                            _diag,
                            focus_outcome=_focus,
                            timeline=getattr(self, "timeline", []) or [],
                        )
                        if logger:
                            logger.info(
                                "[FailureAttribution v2 pre-replan] "
                                "role=%s cat=%s router=%s reason=%r",
                                _diag.attributed_to_role,
                                _diag.category,
                                _decision.kind,
                                _decision.reason[:200],
                            )
                        if _decision.kind == "actor_redo":
                            # Pure skip — no planner LLM call this turn.
                            self._pending_actor_recovery_payload = (
                                _decision.params)
                            # Consume force_replan state so next turn's
                            # dispatcher doesn't re-trigger force_replan
                            # for the same situation.
                            self._force_replan_reason = None
                            self._force_replan_category = None
                            self._force_replan_category_pending = None
                            # Synthesize a no-op planner result so the
                            # downstream control flow sees CONTINUE +
                            # the existing subgoal_queue untouched.
                            plan = None
                            planner_decision = "CONTINUE"
                            self.planner_parsed = {
                                "decision": "CONTINUE",
                                "last_subgoal_done": False,
                                "feedback_to_actor": None,
                            }
                            self._dump_stuck_category = "actor_redo_skip"
                            self._dump_stuck_reason = _decision.reason
                            _skipped_planner = True
                        elif _decision.kind == "verifier_override":
                            # Side-effect: synthesize KN_OUTCOME_SATISFIED
                            # on the focus outcome. Always LIVE when v2
                            # is on (USE_FAILURE_ATTRIBUTION=1); router
                            # safety gates remain the only guard.
                            from mm_agents.structagent.attribution.failure_attribution_executor import (
                                apply_verifier_override,
                            )
                            _vo_audit = apply_verifier_override(
                                focus_outcome=_focus,
                                decision_params=_decision.params,
                                step_idx=getattr(self, "_wall_step_idx",
                                                 0) or 0,
                                dry_run=not _verifier_override_live,
                            )
                            self._last_verifier_override_audit = _vo_audit
                            # Consume force_replan flags regardless of
                            # apply status — even a dry-run logged event
                            # should not re-trigger force_replan next
                            # turn.
                            self._force_replan_reason = None
                            self._force_replan_category = None
                            self._force_replan_category_pending = None
                            plan = None
                            planner_decision = "CONTINUE"
                            self.planner_parsed = {
                                "decision": "CONTINUE",
                                "last_subgoal_done": (
                                    _vo_audit.get("status") == "applied"),
                                "feedback_to_actor": None,
                            }
                            self._dump_stuck_category = "verifier_override"
                            self._dump_stuck_reason = (
                                f"{_decision.reason} | audit_status="
                                f"{_vo_audit.get('status')}")
                            _skipped_planner = True
                        elif _decision.kind == "env_intervention":
                            # Side-effect: only ``wait`` is auto-executed
                            # today; other kinds fall back to planner.
                            from mm_agents.structagent.attribution.failure_attribution_executor import (
                                apply_env_intervention,
                            )
                            _ei_audit = apply_env_intervention(
                                env=env,
                                decision_params=_decision.params,
                                step_idx=getattr(self, "_wall_step_idx",
                                                 0) or 0,
                            )
                            self._last_env_intervention_audit = _ei_audit
                            if _ei_audit.get("status") == "applied":
                                self._force_replan_reason = None
                                self._force_replan_category = None
                                self._force_replan_category_pending = None
                                plan = None
                                planner_decision = "CONTINUE"
                                self.planner_parsed = {
                                    "decision": "CONTINUE",
                                    "last_subgoal_done": False,
                                    "feedback_to_actor": None,
                                }
                                self._dump_stuck_category = (
                                    "env_intervention_applied")
                                self._dump_stuck_reason = (
                                    f"{_decision.reason} | kind="
                                    f"{_ei_audit.get('recovery_kind')}")
                                _skipped_planner = True
                            else:
                                # Cache as planner_replan fallback for
                                # the v2 branch in planner.py.
                                if logger:
                                    logger.info(
                                        "[FailureAttribution v2] env_"
                                        "intervention status=%s — "
                                        "falling back to planner_replan",
                                        _ei_audit.get("status"))
                                self._cached_failure_attribution_decision = (
                                    _decision)
                        else:
                            # planner_replan / fallback_planner — cache
                            # the decision so planner's existing v2 hook
                            # reuses it (no duplicate LLM call).
                            self._cached_failure_attribution_decision = (
                                _decision)
            except Exception as _v2e:
                if logger:
                    logger.warning(
                        "[FailureAttribution v2 pre-replan] failed "
                        "(falling back to v1/planner flow): %s", _v2e)

        if _skipped_planner:
            if logger:
                logger.info(
                    "[FailureAttribution v2] PLANNER SKIPPED — actor "
                    "will consume recovery payload on its next call")
        else:
            plan, planner_decision = self._pa_call_planner(
                instruction, obs, step_idx, env=env, mode=mode)
        parsed = getattr(self, "planner_parsed", {}) or {}
        last_subgoal_done = parsed.get("last_subgoal_done")
        self.feedback_to_actor = parsed.get("feedback_to_actor") or None
        return plan, planner_decision, parsed, last_subgoal_done

    def _pa_post_planner(self, ctx, parsed, plan, planner_decision,
                         last_subgoal_done):
        """Post-planner bookkeeping: dispatch <notebook_audit> note_write calls,
        log the Bottleneck, reset coord counters on a completed subgoal, clear
        legacy actor flags, install current_plan on step0/REPLAN, and default a
        missing decision to REPLAN. Returns the (possibly updated) planner_decision."""
        step_idx = ctx.step_idx
        # Execute <notebook_audit><writes> note_write calls BEFORE the
        # actor runs this turn's first <step>. These are NOT subgoals
        # — they're host-side Notebook updates the planner declared in
        # its plan header. Running them now means the very next
        # decomposer / actor call sees them in the [Notebook] block on
        # the SAME turn (no round-trip needed). Idempotent: re-running
        # on a 2nd planner call (e.g. DONE-rejection retry) just
        # overwrites the same keys.
        _nb_writes = parsed.get("notebook_writes") or []
        if _nb_writes:
            from mm_agents.structagent.actions.handlers.notebook import handle_note_write
            n_ok, n_fail = 0, 0
            for raw_call in _nb_writes:
                try:
                    ast_call = self._pa_parse_action_ast(raw_call)
                    if (ast_call
                            and ast_call.get("function") == "note_write"
                            and isinstance(ast_call.get("args"), dict)):
                        # A7: Facts-only persistence (notebook deleted).
                        # owning_outcome defaults to current focus; routes
                        # to global_facts when no focus is set.
                        _focus = None
                        try:
                            _f = self.ledger.next_focus_outcome() if self.ledger else None
                            _focus = _f.id if _f is not None else None
                        except Exception:
                            _focus = None
                        handle_note_write(
                            ast_call["args"],
                            ledger=self.ledger,
                            logger=logger,
                            owning_outcome=_focus,
                            step_idx=getattr(self, "_wall_step_idx", 0) or 0,
                        )
                        n_ok += 1
                    else:
                        n_fail += 1
                        if logger:
                            logger.warning(
                                "[notebook_audit] unparsable / non-note_write: %r",
                                raw_call[:120])
                except Exception as e:  # noqa: BLE001
                    n_fail += 1
                    if logger:
                        logger.warning(
                            "[notebook_audit] dispatch failed for %r: %s",
                            raw_call[:120], e)
            if logger:
                logger.info("[notebook_audit] step=%d dispatched %d note_write "
                            "call(s) before actor (failures=%d)",
                            step_idx, n_ok, n_fail)

        # Visibility log: surface the planner's <Bottleneck> (its
        # one-sentence synthesis of the current blocker) in runtime.log.
        # The cadence trigger reuses this same string as the strategy
        # abstract for its failed_paths writes, so seeing it logged here
        # explains what shows up later in the ledger.
        if logger:
            bn = parsed.get("bottleneck")
            if bn:
                logger.info("[PA-Planner] Bottleneck: %s", bn[:240])

        # Consume stale legacy flags (not used by the new planner-every-step flow)
        self.actor_subgoal_done_for_planner = False
        self.actor_exhausted_for_planner = False

        # Update current_plan when step 0 or REPLAN
        if plan is not None and (step_idx == 0 or planner_decision == "REPLAN"):
            self.current_plan = plan

        if planner_decision is None and step_idx > 0:
            planner_decision = "REPLAN"
        return planner_decision

    def _pa_reset_progress_counter(self):
        """Reset the REPLAN-cadence counter when the KeyNode detector saw a
        fresh outcome-satisfied / value-committed this turn (real progress)."""
        # Progress-detection: if the KeyNode detector saw a fresh
        # outcome satisfied or value committed THIS turn, reset the
        # REPLAN-cadence counter. Without this, a turn whose actor
        # actually broke through (e.g. clicked the right button and
        # the resulting outcome got verified) can still trigger
        # rule #6 force_replan because the counter is bumped on
        # planner_decision == "REPLAN" regardless of whether the
        # underlying state actually advanced.
        try:
            cand = getattr(self, "_candidate_step", None)
            if cand is not None and getattr(cand, "key_nodes", None):
                from mm_agents.structagent.ledger.core.timeline import (
                    KN_OUTCOME_SATISFIED, KN_VALUE_COMMITTED,
                )
                progress_kinds = {KN_OUTCOME_SATISFIED, KN_VALUE_COMMITTED}
                if any(getattr(kn, "kind", None) in progress_kinds
                       for kn in cand.key_nodes):
                    if (self._replans_since_last_progress > 0
                            and logger):
                        logger.info(
                            "[Progress] KeyNode detector saw outcome/value "
                            "this turn — resetting REPLAN cadence counter "
                            "from %d to 0",
                            self._replans_since_last_progress,
                        )
                    self._replans_since_last_progress = 0
        except Exception as e:
            if logger:
                logger.warning("[Progress] reset check failed: %s", e)


    # --- main planner-actor predict ---

    def _pa_predict(self, instruction: str, obs: dict, env=None,
                    wall_step_idx: Optional[int] = None) -> Tuple[str, List[str]]:
        """Planner-actor predict: planner decides subgoal, actor executes it.
        Returns (response_text, [pyautogui_code_strings]).

        Planner runs every turn (Phase-2 unified contract — folds the
        old _pa_check_subgoal_done + _correct_plan calls into a single
        prompt). The mode (initial / force_replan / progress_check) is
        chosen UPFRONT by ``_decide_planner_mode_and_reason`` and routed
        through the right prompt builder. ~95% of turns are 1 planner
        call; the only path that produces 2 is when the planner emits
        DONE and the audit + cache-gate combination rejects it.
        """
        # 0) Per-step setup → planner-ready state (see _pa_begin_step).
        ctx = _StepContext(instruction=instruction, obs=obs, env=env,
                           wall_step_idx=wall_step_idx)
        self._pa_begin_step(ctx)

        # working memory — build candidate StepRecord and run KeyNode
        # detector BEFORE the planner sees its prompt, so the ledger's
        # to_prompt_block() reflects fresh outcome state ([verified] /
        # [REVERTED] / [pending]). No-op when MEMORY_VERSION=v1.
        self._pre_planner(ctx.obs, ctx.step_idx, env=ctx.env)

        # Reset REPLAN cadence when the KeyNode detector saw progress this turn.
        self._pa_reset_progress_counter()

        burst_decision = self._pa_actor_burst_precheck(ctx)
        if burst_decision == "continue_actor":
            self.planner_decision = "CONTINUE"
            self.planner_parsed = {
                "decision": "CONTINUE",
                "last_subgoal_done": False,
                "feedback_to_actor": None,
            }
            if logger:
                logger.info(
                    "[ActorBurst] planner skipped; actor continues subgoal")
            return self._pa_run_actor(ctx)
        if burst_decision == "actor_loop_end":
            self._verify_actor_loop_exit_for_planner(ctx, burst_decision)
            self._pa_reset_actor_burst_state()
            self._pending_actor_failure_reason = None
            self._actor_emitted_impossible = False
            self.actor_subgoal_done_for_planner = False
            self.actor_exhausted_for_planner = False

        # === PLANNER phase — mode + failure-attribution + one planner call.
        plan, planner_decision, parsed, last_subgoal_done = \
            self._pa_run_planner(ctx)

        # Post-planner bookkeeping (notebook writes, current_plan, etc.).
        planner_decision = self._pa_post_planner(
            ctx, parsed, plan, planner_decision, last_subgoal_done)

        # 1) Advance the subgoal queue / handle terminal decisions. Returns a
        #    (response, actions) tuple to end the step, else None to run the actor.
        finish = self._pa_advance_or_finish(
            ctx, plan, planner_decision, last_subgoal_done)
        if finish is not None:
            return finish

        # 2) ACTOR phase — subgoal execution; always returns (response, actions).
        return self._pa_run_actor(ctx)

    def _maybe_feasibility_verdict(self, step_idx):
        """Fix 1 — at >= K cumulative force-replans, ask the verifier model
        whether the task is genuinely INFEASIBLE. Returns the ``_pa_predict``
        ``(reason, ["FAIL"])`` tuple to emit a terminal FAIL, or ``None`` to
        keep replanning.

        Gated behind ENABLE_FEASIBILITY_VERDICT. Conservative by construction:
        any non-INFEASIBLE verdict (incl. parse-fail / error) → None. Re-checks
        every FEASIBILITY_RECHECK_EVERY replans so a task that accumulates more
        evidence later still gets judged, while bounding the call count.

        The brief is built from the live runtime.log via the SAME extractor the
        replay validation used — see attribution/feasibility_verdict.py's
        CONSISTENCY CONTRACT. Emitting ["FAIL"] mirrors the planner-INFEASIBLE
        dispatch: env.step("FAIL") sets done and makes action_history end with
        FAIL, which is exactly what OSWorld's infeasible evaluator scores 1.0.
        """
        if not self.cfg.feasibility_verdict:
            return None
        if self.ledger is None:
            return None
        try:
            K = self.cfg.feasibility_verdict_k
            recheck_every = self.cfg.feasibility_recheck_every
        except ValueError:
            K, recheck_every = 8, 6
        # Trigger at K cumulative replans OR N consecutive actor-IMPOSSIBLEs
        # (early-out for tasks the actor keeps declaring undoable).
        if (self._total_replan_count < K
                and self._consecutive_impossible_count < 3):
            return None
        if (self._total_replan_count
                - self._last_feasibility_check_at_replan) < recheck_every:
            return None
        results_dir = getattr(self, "_results_dir", None)
        if not results_dir:
            return None
        try:
            log_text = open(os.path.join(results_dir, "runtime.log"),
                            errors="replace").read()
        except OSError:
            return None
        # Mark BEFORE the call so a judge crash still advances the cadence.
        self._last_feasibility_check_at_replan = self._total_replan_count
        import logging
        _log = logging.getLogger("desktopenv.qwen25vl_agent")
        from mm_agents.structagent.attribution.feasibility_verdict import (
            extract_brief_from_log, feasibility_verdict, INFEASIBLE,
        )
        brief = extract_brief_from_log(log_text, self._total_replan_count)
        instruction = (getattr(self, "_raw_instruction", None)
                       or getattr(self, "instruction", None) or "")
        verdict, reason = feasibility_verdict(
            self.call_llm, self.verifier_model, instruction, brief, logger=_log,
        )
        if verdict != INFEASIBLE:
            return None
        _log.info("[FeasibilityVerdict] INFEASIBLE at replan=%d — emitting "
                  "FAIL. reason=%r", self._total_replan_count, reason[:200])
        self.actions.append("FAIL (feasibility judge: infeasible)")
        return (f"Feasibility judge decided INFEASIBLE after "
                f"{self._total_replan_count} replans — {reason}", ["FAIL"])

    def predict(self, instruction: str, obs: Dict, env=None,
                wall_step_idx: Optional[int] = None) -> List:
        """Predict the next action(s) based on the current observation.

        Thin entry point — delegates to ``_pa_predict``. ``wall_step_idx``
        is the OUTER step counter from lib_run_single (env.step calls so
        far). Currently unused inside ``_pa_predict`` — kept on the
        signature for callers in lib_run_single and for any future
        wall-step-keyed logic.
        """
        return self._pa_predict(instruction, obs, env=env,
                                wall_step_idx=wall_step_idx)

    def reset(self, _logger=None):
        global logger
        logger = (_logger if _logger is not None else
                  logging.getLogger("desktopenv.qwen25vl_agent"))

        self._reset_task_state()
