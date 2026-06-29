"""Recovery: planner mode decision, FailedPath curator, strategy tracking.

The cluster of methods that watch for stuck-signals (no progress, DONE
rejection, action repetition, REPLAN cadence) and decide what to do —
either pop the next subgoal, force a planner replan with a typed
stuck_reason, or run the LLM curator that re-derives ``ledger.failed_paths``
from the strategy + action stuck history.

Folded into ``StructAgent`` via inheritance.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from mm_agents.structagent.core.recovery.transitions import Transition


logger = logging.getLogger("desktopenv.qwen25vl_agent_planner")


class Recovery:
    """Planner-mode decisions + curator + strategy tracking."""

    def _decide_planner_mode_and_reason(
        self, env, obs: dict, step_idx: int,
    ) -> Tuple[str, Optional[str]]:
        """Return ``(mode, reason)`` for THIS turn's planner call.

        Called once at the top of ``_pa_predict`` after ``_pre_planner``.
        Reads (and consumes) ``self._pending_actor_failure_reason``,
        ``self._force_replan_reason``, the REPLAN-cadence counters, and
        the current subgoal-step budget; performs the rule-#4/#5/#6 side
        effects inline so the caller doesn't need to re-derive them."""
        # Rule #1: first turn — no prior state to react to.
        if step_idx == 0:
            return "initial", None

        # Rule #2: actor reported failure on the previous turn
        # (IMPOSSIBLE / FAIL / no-bash-body). Has highest priority among
        # the next-turn signals because the actor's own dead-end is the
        # most concrete "this plan is stuck" evidence we have.
        pending = getattr(self, "_pending_actor_failure_reason", None)
        if pending:
            self._pending_actor_failure_reason = None  # consume
            self._force_replan_category = "actor_failure"
            return "force_replan", pending

        # Rule #3: a force_replan reason is pre-staged on
        # ``self._force_replan_reason``. Source: the DONE-acceptance gate
        # rejected DONE last turn but the in-turn force_replan retry
        # returned no <plan>, so the reason was re-staged for the next
        # turn (the "defer" path).
        #
        # T3 done-auditor (plan Part I.2) reuses the same staging
        # channel but writes ``_force_replan_category_pending`` to flip
        # the category from "done_rejected" to "done_audit_failed" — so
        # the planner's force_replan prompt knows the rejection came
        # from the adversarial auditor (different remediation hint)
        # not from the structured ledger gate.
        staged_reason = getattr(self, "_force_replan_reason", None)
        if staged_reason:
            self._force_replan_reason = None  # consume
            pending_category = getattr(
                self, "_force_replan_category_pending", None)
            if pending_category:
                self._force_replan_category_pending = None  # consume
                self._force_replan_category = pending_category
            else:
                self._force_replan_category = "done_rejected"
            return "force_replan", staged_reason

        # Rule #4: subgoal step budget exhausted without the planner ever
        # judging done=YES. Burns through 5 actor attempts on the same
        # subgoal — the strategy is wrong, not the execution. Abandon
        # the subgoal so the force_replan prompt sees it in
        # <abandoned_subgoals>.
        last_subgoal_done = (self.planner_parsed or {}).get("last_subgoal_done") \
            if hasattr(self, "planner_parsed") else None
        if (self.current_subgoal_step >= self._MAX_SUBGOAL_STEPS
                and last_subgoal_done is not True):
            if self.current_subgoal:
                self.abandoned_subgoals.append(self.current_subgoal)
            self._force_replan_category = "budget_exhausted"
            return "force_replan", (
                f"subgoal step budget exhausted ({self.current_subgoal_step}"
                f" / {self._MAX_SUBGOAL_STEPS}) without the planner ever "
                f"emitting done=YES; the repeated attempts on the same "
                f"subgoal produced no verified progress"
            )

        # Rule #5: subgoal queue is empty BUT the task end-state has not
        # been verified DONE. Last turn's CONTINUE drained the queue;
        # this turn the planner needs to either emit DONE (which the
        # cache gate then validates) or REPLAN with a closing step.
        if (not self.subgoal_queue
                and self.current_subgoal
                and "All planned subgoals completed" in (self.current_subgoal or "")):
            self._force_replan_category = "queue_empty_no_done"
            return "force_replan", (
                "the subgoal queue is empty (all planned subgoals marked "
                "completed) but the task's overall end-state has not yet "
                "been verified"
            )

        # Rule #6: ACTION-level cadence trigger — planner has been
        # REPLAN-ing within the SAME strategy for too long without
        # progress. The strategy itself is presumed sound (otherwise
        # the planner would have switched <strategy>, which is already
        # captured by the strategy-LEVEL FailedPath emitted by
        # ``_commit_strategy_change`` at REPLAN commit time). This
        # trigger records the stuck SUBGOAL text as an action-level
        # dead-end so the planner explores alternative TACTICS within
        # the same strategy (different element, keyboard shortcut,
        # alternate coords).
        #
        # The two gates that must both fire:
        #   - replans_within_current_strategy >= threshold
        #     (planner persisted on the SAME strategy across the stuck
        #     replans — without this, strategy-switches that happen to
        #     coincide with no-progress would falsely fire action-level)
        #   - gap_since_last_fire >= threshold
        #     (don't re-fire on consecutive turns; need fresh evidence)
        from mm_agents.structagent.ledger.core.ledger import abstract_actor_response
        from mm_agents.structagent.ledger.core.exploration import ActionStuckEvent
        gap_since_last_fire = (self._total_replan_count
                               - self._last_search_fired_at_replan)
        if (self.replans_within_current_strategy
                >= self._REPLAN_WITHOUT_PROGRESS_THRESHOLD
                and gap_since_last_fire
                >= self._REPLAN_WITHOUT_PROGRESS_THRESHOLD):
            stuck_subgoal = (self.current_subgoal or "").strip()
            if stuck_subgoal:
                if (not self.abandoned_subgoals
                        or self.abandoned_subgoals[-1] != stuck_subgoal):
                    self.abandoned_subgoals.append(stuck_subgoal)
                # Append an ActionStuckEvent — the LLM curator decides
                # whether this becomes an action-level FailedPath
                # (after seeing it alongside the strategy history +
                # currently-recorded entries). DO NOT write
                # ledger.add_failed_path directly anymore.
                span_actions = list(
                    self.actions[self.current_strategy_started_at_step:]
                )
                action_summaries = [
                    abstract_actor_response(a) for a in span_actions
                ][-6:]
                evt = ActionStuckEvent(
                    step_idx=step_idx,
                    stuck_subgoal=stuck_subgoal,
                    replans_within_strategy=self.replans_within_current_strategy,
                    actions_during_subgoal=action_summaries,
                    current_strategy=self.current_plan_strategy,
                )
                self._attach_curator_evidence(evt)
                self._action_stuck_history.append(evt)
            self._last_search_fired_at_replan = self._total_replan_count
            self._force_replan_category = "replan_cadence"
            return "force_replan", (
                f"You have re-emitted the subgoal "
                f"'{stuck_subgoal}' across "
                f"{self.replans_within_current_strategy} consecutive "
                f"REPLANs within the same strategy without the "
                f"expected post-state being reached. The specific "
                f"tactical step is stuck — DO NOT re-emit the same "
                f"subgoal as-is. Either: (a) try a different tactic "
                f"under the SAME strategy (alternate element name, "
                f"keyboard shortcut, alt coords), or (b) abandon the "
                f"strategy entirely (declare a new <strategy> in the "
                f"new <plan>) — try a direct URL, top-level menu, etc."
            )

        # Default: normal turn — let the planner do its CONTINUE / REPLAN
        # / DONE dispatch on the prior subgoal's outcome.
        return "progress_check", None

    def _finalize_recovery_state(self, mode: str, step_idx: int) -> None:
        """Refresh ``self._recovery_state`` to reflect the transition
        the dispatcher just decided + current counter snapshot.

        Called once per turn, immediately after
        ``_decide_planner_mode_and_reason`` returns. ``mode`` is the
        mode string the dispatcher returned ("initial" /
        "force_replan" / "progress_check"); when it's "force_replan"
        the concrete stuck category was stored on
        ``self._force_replan_category`` by the matching rule.

        T1 scope: read-only mirror. ``self._*`` counters remain source
        of truth; this method snapshots them onto ``self._recovery_state``
        so tests and T3 escalation reads can use a single field.
        See plan file Part I.1.
        """
        if mode == "force_replan":
            category = getattr(self, "_force_replan_category", None)
            if category is None:
                # Shouldn't happen — the dispatcher always sets the
                # category before returning "force_replan". Fall back
                # to keeping the prior receipt rather than crashing.
                return
            transition: Transition = category  # type: ignore[assignment]
        else:
            # "initial" / "progress_check" — both are valid Transition values.
            transition = mode  # type: ignore[assignment]

        self._recovery_state = self._recovery_state.with_transition(
            transition,
            step_idx=step_idx,
            replans_since_last_progress=getattr(
                self, "_replans_since_last_progress", 0),
            last_search_fired_at_replan=getattr(
                self, "_last_search_fired_at_replan", -99),
            total_replan_count=getattr(
                self, "_total_replan_count", 0),
            done_rejected_count=getattr(
                self, "_done_rejected_count", 0),
            replans_within_current_strategy=getattr(
                self, "replans_within_current_strategy", 0),
        )

    def _pop_next_subgoal(self, *, step_idx: int) -> Tuple[bool, int]:
        """Pop the next subgoal from the queue and install it as current.

        Subgoal advancement is driven SOLELY by the planner's
        ``<done>YES</done>`` output: the runtime never inspects ledger
        outcome state to second-guess that decision. The planner sees
        ledger state in its own prompt and decides accordingly.

        Returns ``(popped, skipped)`` — ``skipped`` is kept in the
        signature for caller-API compatibility but is always 0 now
        (no chain-skip path).

        Updates ``current_subgoal`` / ``current_subgoal_expected_post_state``
        / ``current_subgoal_app`` / ``current_subgoal_targets`` in place.
        """
        if not self.subgoal_queue:
            self.current_subgoal = None
            self.current_subgoal_expected_post_state = None
            self.current_subgoal_app = None
            self.current_subgoal_targets = []
            return False, 0
        self.current_subgoal = self.subgoal_queue.pop(0)
        self.current_subgoal_expected_post_state = (
            self.subgoal_expected_post_states.pop(0)
            if self.subgoal_expected_post_states else None)
        self.current_subgoal_app = (
            self.subgoal_apps.pop(0)
            if self.subgoal_apps else None)
        self.current_subgoal_targets = (
            self.subgoal_targets.pop(0)
            if self.subgoal_targets else [])
        self.current_subgoal_step = 0
        self.current_subgoal_feedbacks = []
        self.actor_history = []
        return True, 0

    def _judge_strategy_similar(
        self, prior: str, new: str,
        actions_under_prior: List[str], progress_signal: str,
    ) -> bool:
        """Thin wrapper: pack an LLM-call lambda + the per-task cache,
        delegate to ledger module's ``judge_strategy_change`` (which
        owns the prompt + the SAME/DIFFERENT contract)."""
        from mm_agents.structagent.ledger.core.ledger import judge_strategy_change

        def _llm_call(prompt: str) -> str:
            return self.call_llm(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt}
                    ]}],
                    "max_tokens": 5,
                    "temperature": 0.0,
                    "top_p": self.top_p,
                },
                self.model,
            ) or ""

        return judge_strategy_change(
            prior=prior, new=new,
            actions_under_prior=actions_under_prior,
            progress_signal=progress_signal,
            llm_call=_llm_call,
            cache=self._strategy_judge_cache,
        )

    def _commit_strategy_change(self, step_idx: int) -> None:
        """Thin wrapper: pack agent state, delegate to ledger module's
        ``commit_strategy_change`` (with the LLM judge), assign result
        back. Called from each plan-commit site (step==0 install +
        REPLAN swap)."""
        from mm_agents.structagent.ledger.core.ledger import (
            StrategyState, commit_strategy_change,
        )
        new_strategy = (self.planner_parsed or {}).get("plan_strategy")
        prior_len = len(self._strategy_history)
        result = commit_strategy_change(
            ledger=self.ledger,
            new_strategy=new_strategy,
            prior=StrategyState(
                current_strategy=self.current_plan_strategy,
                started_at_step=self.current_strategy_started_at_step,
                replans_within_strategy=self.replans_within_current_strategy,
            ),
            actions=self.actions,
            current_subgoal=self.current_subgoal,
            step_idx=step_idx,
            judge=self._judge_strategy_similar,
            strategy_history=self._strategy_history,
        )
        # If commit appended a new event, attach perceiver/screenshot
        # evidence so the curator can later judge tactical-vs-strategic.
        if len(self._strategy_history) > prior_len:
            self._attach_curator_evidence(self._strategy_history[-1])
        self.current_plan_strategy = result.current_strategy
        self.current_strategy_started_at_step = result.started_at_step
        self.replans_within_current_strategy = result.replans_within_strategy

    def _attach_curator_evidence(self, evt) -> None:
        """Populate the ``_EvidenceFields`` on a freshly-created
        StrategyEvent / ActionStuckEvent from the agent's current
        perceiver + screenshot state. Curator uses these to decide
        whether a failed strategy span was tactical (target was on
        screen, click missed) or strategic (target never appeared).

        Safe no-op when perceiver / candidate step are unavailable.
        """
        cand = getattr(self, "_candidate_step", None)
        if cand is not None and getattr(cand, "screenshot_b64", None):
            evt.last_screenshot_b64 = cand.screenshot_b64
        snap = getattr(self, "_current_snapshot", None)
        if snap is None:
            return
        try:
            sc = getattr(snap, "subgoal_check", None)
            if sc is not None:
                evt.last_perceiver_overall = getattr(sc, "overall", None)
            rc = list(getattr(snap, "relevant_controls", []) or [])
            blockers = list(getattr(snap, "unexpected_blockers", []) or [])
            evt.last_perceiver_controls_count = len(rc)
            evt.last_perceiver_blockers = len(blockers)
            # Top-10 relevant controls (perceiver already pre-filtered
            # to ≤ 8 in normal mode; cap at 10 for safety) — text-form
            # so curator can grep for the target subgoal element name
            # without needing the screenshot.
            lines = []
            for c in rc[:10]:
                role = getattr(c, "role", "?") or "?"
                label = (getattr(c, "label", "") or "")[:80]
                state = ",".join(getattr(c, "state", []) or [])
                val = (getattr(c, "value", "") or "")[:40]
                lines.append(
                    f"  {role} '{label}'  state=[{state}]"
                    + (f"  value={val!r}" if val else "")
                )
            if lines:
                evt.last_a11y_top_k = "\n".join(lines)
        except Exception:
            # Non-fatal — curator will fall back to text-only judgment
            pass

    def _curate_failed_paths(self, step_idx: int) -> None:
        """Run the LLM dead-end curator and replace
        ``ledger.failed_paths`` with the curated list.

        Called after every REPLAN commit (and after a rule-#6 cadence
        fire). The curator sees the FULL strategy + action-stuck
        history + current outcome state + currently-recorded
        ``failed_paths``, and re-derives the authoritative list — it
        can add, drop, merge, or revise entries.

        On any failure (LLM error, JSON parse fail) the existing
        ``failed_paths`` are preserved unchanged.
        """
        if self.ledger is None or not self.ledger.required_outcomes:
            return
        from mm_agents.structagent.ledger.core.exploration import (
            curate_failed_paths,
        )

        def _llm_call(arg) -> str:
            """Accept either a prompt string (legacy) or a list of
            messages (multimodal). ``curate_failed_paths`` now
            prefers the messages form so it can attach screenshots
            as image_url parts for tactical-vs-strategic judgment."""
            if isinstance(arg, list):
                messages = arg
            else:
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": arg}
                ]}]
            return self.call_llm(
                {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": 1500,
                    "temperature": 0.0,
                    "top_p": self.top_p,
                },
                self.model,
            ) or ""

        outcome_states = {o.id: o.state for o in self.ledger.required_outcomes}
        analysis_sink: List[Dict[str, Any]] = []
        new_failed_paths = curate_failed_paths(
            instruction=getattr(self, "instruction", "") or "",
            required_outcomes=list(self.ledger.required_outcomes),
            outcome_states=outcome_states,
            strategy_events=list(self._strategy_history),
            action_stuck_events=list(self._action_stuck_history),
            current_failed_paths=list(self.ledger.failed_paths),
            llm_call=_llm_call,
            current_step=step_idx,
            analysis_sink=analysis_sink,
        )
        if new_failed_paths is None:
            if logger:
                logger.warning(
                    "[Curator] returned None — preserving existing "
                    "failed_paths (%d entries)",
                    len(self.ledger.failed_paths))
            return
        # Cache per-strategy notes for the timeline render. Each entry
        # is "<verdict> — <rationale>" keyed by strategy_text. Replaces
        # any prior cache (curator runs after every REPLAN, so stale
        # notes are simply overwritten with the freshest verdict).
        new_notes: Dict[str, str] = {}
        # Collect ``strategy_outcome_satisfied`` strategies — the curator
        # marks these when a strategy's primary outcomes are now [verified]
        # so the planner doesn't re-propose them on REPLAN. Cross-check
        # that each claimed satisfied_outcome is actually [verified] in
        # ``outcome_cache`` before persisting; the LLM curator can
        # mis-classify, and we don't want a stale "completed" tag to
        # mask a real pending outcome from the planner.
        new_completed_strategies: List[Dict[str, Any]] = []
        for a in analysis_sink:
            if a.get("kind") == "strategy":
                strat = (a.get("strategy_text") or "").strip()
                verdict = a.get("verdict", "?")
                rationale = (a.get("rationale", "") or "").strip()[:240]
                if strat:
                    note = f"{verdict} — {rationale}" if rationale else verdict
                    new_notes[strat] = note
                if verdict == "strategy_outcome_satisfied" and strat:
                    raw_outs = a.get("satisfied_outcomes") or []
                    if isinstance(raw_outs, list):
                        verified_outs = [
                            oid for oid in raw_outs
                            if isinstance(oid, str)
                            and outcome_states.get(oid) == "verified"
                        ]
                        if verified_outs:
                            new_completed_strategies.append({
                                "strategy": strat,
                                "satisfied_outcomes": verified_outs,
                            })
        if new_notes:
            self._last_curator_strategy_notes = new_notes
        # Replace (not merge): the curator re-derives the authoritative
        # list every pass, just like failed_paths.
        self.ledger.completed_strategies = new_completed_strategies

        if logger:
            n_strat = sum(1 for fp in new_failed_paths
                          if fp.level == "strategy")
            n_act = len(new_failed_paths) - n_strat
            logger.info(
                "[Curator] failed_paths re-derived: %d total "
                "(%d strategy, %d action) — was %d entries",
                len(new_failed_paths), n_strat, n_act,
                len(self.ledger.failed_paths))
            for a in analysis_sink:
                kind = a.get("kind", "?")
                verdict = a.get("verdict", "?")
                rationale = (a.get("rationale", "") or "")[:160]
                if kind == "strategy":
                    label = (a.get("strategy_text", "") or "?")[:60]
                    merge = a.get("merge_target_path") or ""
                    merge_str = f" → merge into '{merge[:60]}'" if merge else ""
                    logger.info(
                        "[Curator]   strategy '%s': %s%s — %s",
                        label, verdict, merge_str, rationale)
                elif kind == "action":
                    pattern = (a.get("pattern", "") or "?")[:60]
                    under = (a.get("occurred_under_strategy", "") or "?")[:40]
                    logger.info(
                        "[Curator]   action '%s' (under '%s'): %s — %s",
                        pattern, under, verdict, rationale)
                else:
                    ref = (a.get("event_ref", "") or "?")[:60]
                    logger.info(
                        "[Curator]   %s: %s — %s",
                        ref, verdict, rationale)
        self.ledger.failed_paths = new_failed_paths

