"""LedgerOrchestrator: ledger init / accumulate / fire-boundary / DONE-acceptance.

The cluster of methods that read the observation, dump it into the
ledger, fire the periodic summarizer / KeyNode detector, and gate
planner DONE on the typed ledger state.

Folded into ``StructAgent`` via inheritance.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from mm_agents.structagent.utils.a11y import (
    extract_active_url_from_a11y as _extract_active_url_from_a11y,
    linearize_obs_a11y as _linearize_obs_a11y,
)


logger = logging.getLogger("desktopenv.qwen25vl_agent_planner")


class LedgerOrchestrator:
    """Outcome-ledger orchestration on the agent."""

    # ----- _dump_environment_probe --------------------------------------------------
    def _dump_environment_probe(
        self,
        raw: Optional[str],
        digested: Optional[str],
        useful_file_contents: Optional[str] = None,
    ) -> None:
        """Persist init_ledger's once-per-task probe outputs (raw bash
        stdout + perceiver digestion + disk-content probe of useful
        files) under ``<results_dir>/perceiver_debug/`` for offline
        inspection."""
        try:
            from mm_agents.structagent.perception import perceiver_debug
            perceiver_debug.dump_environment_probe(
                results_dir=getattr(self, "_results_dir", None),
                raw_probe=raw,
                digested=digested,
                useful_file_contents=useful_file_contents,
            )
        except Exception as e:
            if logger:
                logger.info("[Probe-Dump] failed: %s", e)

    # ----- _dump_init_ledger_prompt --------------------------------------------------
    def _dump_init_ledger_prompt(
        self,
        messages: List[Dict[str, Any]],
        attempt: int,
        raw_response: str,
    ) -> None:
        """Persist the exact init_ledger LLM messages + the raw response
        for offline inspection. Goes under
        ``<results_dir>/keynode_debug/init_ledger_prompt_attempt_N.json``.

        Heavy multimodal pieces (b64 PNG screenshots) inflate the dump
        by ~150KB each; we strip them to URL-shape markers so the file
        stays readable. Text content + the system prompt are preserved
        verbatim so the user can audit exactly what the author LLM saw.
        """
        try:
            import json as _json
            import os as _os
            results_dir = getattr(self, "_results_dir", None)
            if not results_dir:
                return
            out_dir = _os.path.join(results_dir, "keynode_debug")
            _os.makedirs(out_dir, exist_ok=True)
            # Strip image data from message content for readability,
            # mirroring the planner_messages dump convention.
            def _summarize(content):
                if isinstance(content, str):
                    return {"type": "text", "len_chars": len(content),
                            "text": content}
                if isinstance(content, list):
                    parts = []
                    for p in content:
                        if isinstance(p, dict):
                            if p.get("type") == "image_url" or "image_url" in p:
                                url = (p.get("image_url") or {}).get("url", "")
                                parts.append({
                                    "type": "image",
                                    "url_prefix": url[:30],
                                    "url_chars": len(url),
                                })
                            elif "text" in p:
                                t = p.get("text") or ""
                                parts.append({"type": "text",
                                              "len_chars": len(t),
                                              "text": t})
                            else:
                                parts.append({"type": "?", "keys": list(p.keys())})
                        else:
                            parts.append({"type": "?", "repr": repr(p)[:120]})
                    return {"type": "multipart", "parts": parts}
                return {"type": "?", "repr": repr(content)[:120]}

            stripped = []
            for m in messages:
                stripped.append({
                    "role": m.get("role", "?"),
                    **_summarize(m.get("content")),
                })
            out_path = _os.path.join(
                out_dir, f"init_ledger_prompt_attempt_{attempt}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                _json.dump({
                    "attempt": attempt,
                    "messages": stripped,
                    "raw_response": raw_response,
                }, f, ensure_ascii=False, indent=2)
                f.write("\n")
        except Exception as e:
            if logger:
                logger.info("[InitLedgerPromptDump] failed: %s", e)

    # ----- _ledger_init_if_needed --------------------------------------------------
    def _ledger_init_if_needed(
        self, instruction: str, obs: dict, env: Any = None,
    ) -> None:
        """Lazy-init the progress ledger on the very first predict call."""
        if self._ledger_disabled() or self.ledger is not None:
            return
        try:
            from mm_agents.structagent.ledger import init_ledger
            a11y = _linearize_obs_a11y(obs, max_items=120)
            active_url = _extract_active_url_from_a11y(a11y)
            # For office domains (libreoffice_calc / libreoffice_writer /
            # libreoffice_impress) the SP/DP structured block is far
            # more informative than the linearized a11y tree for picking
            # the right answer cell / paragraph. Pass the structured
            # block in lieu of a11y when it's available. We still
            # extract `active_url` from a11y (the doc URL parsing path
            # is unaffected).
            office_doms = {"libreoffice_calc", "libreoffice_writer",
                           "libreoffice_impress"}
            cur_dom = getattr(self, "_current_domain", None)
            doc_inspect_block = getattr(self, "_doc_inspect_block", None)
            if cur_dom in office_doms and doc_inspect_block:
                init_a11y = None
            else:
                init_a11y = a11y
            self.ledger = init_ledger(
                instruction=instruction,
                initial_obs=obs,
                call_llm=self.call_llm,
                model=self.verifier_model,        # verification stack runs on the verifier model
                a11y_text=init_a11y,
                active_url=active_url,
                domain=cur_dom,
                env=env,  # for environment probe (binaries, schemas, ...)
                on_probe_dump=self._dump_environment_probe,
                on_prompt_dump=self._dump_init_ledger_prompt,
                doc_inspect_block=doc_inspect_block,
                related_apps=getattr(self, "_related_apps", None),
            )
            if logger and self.ledger is not None:
                logger.info(
                    "[Ledger] initialized: %d required outcomes, active_url=%s",
                    len(self.ledger.required_outcomes),
                    self.ledger.initial_context.active_url,
                )
                # Multi-app: surface that this is a multi-app task and
                # the per-app outcome breakdown, so runtime.log makes
                # the multi-app structure visible at a glance.
                if self._is_multi_app():
                    _by_app = [
                        f"{getattr(o, 'app', None)}:{o.id}"
                        for o in self.ledger.required_outcomes
                    ]
                    logger.info(
                        "[MultiApp] task spans %s | outcomes by app: %s",
                        self._related_apps, _by_app,
                    )
            # Seed the window-start snapshot so the first periodic fire
            # can compute a real delta (instead of falling back to empty).
            self._ledger_url_at_window_start = active_url
            self._ledger_a11y_at_window_start = a11y
            self._ledger_turns_accumulated = 0
        except Exception as e:
            if logger:
                logger.warning("[Ledger] init failed: %s", e)
            self.ledger = None

    # ----- _ledger_accumulate --------------------------------------------------
    def _ledger_accumulate(self, obs: Optional[dict], action_text: Optional[str]) -> None:
        """Record a per-step (screenshot, action) pair into the current
        subgoal window. Called once per actor-emit during a subgoal."""
        if self.ledger is None:
            return
        if isinstance(obs, dict):
            shot = obs.get("screenshot")
            if isinstance(shot, (bytes, bytearray)):
                self._ledger_subgoal_screenshots.append(bytes(shot))
        if action_text:
            self._ledger_subgoal_actions.append(str(action_text)[:500])

    # ----- _ledger_fire_boundary --------------------------------------------------
    def _ledger_fire_boundary(
        self,
        boundary_type: str,
        obs: Optional[dict],
        step_idx: int,
    ) -> None:
        """Run the summarizer at a subgoal boundary, then reset the window.

        ``boundary_type`` ∈ {"periodic", "done"} (Phase-3 simplified trigger).

        Also computes an objective ``WindowDelta`` (URL change + a11y node
        additions / removals / state flips) between the snapshot captured
        at the last fire and the current observation, then injects it into
        the summarizer prompt so the LLM reasons from the delta rather
        than re-interpreting raw screenshots.

        No-op if ledger missing or the window is empty.
        """
        if self.ledger is None:
            return
        if not self._ledger_subgoal_screenshots and not self._ledger_subgoal_actions:
            # nothing accumulated; still advance plan_step
            self._ledger_subgoal_plan_step = self.current_subgoal or ""
            return
        a11y_after = _linearize_obs_a11y(obs, max_items=300) if obs else None
        url_after = _extract_active_url_from_a11y(a11y_after)
        # Compute the hard-tier delta BEFORE the LLM call so it lands in
        # the prompt. If the window-start snapshot wasn't captured
        # (first fire of the task), delta falls back to "no diff".
        try:
            from mm_agents.structagent.ledger.core.summarizer import compute_window_delta, update_ledger
            delta = compute_window_delta(
                url_before=self._ledger_url_at_window_start,
                url_after=url_after,
                a11y_before=self._ledger_a11y_at_window_start,
                a11y_after=a11y_after,
            )
            # v2: pass the recent timeline window so the failed-path
            # summarizer can use [Timeline signals] (subgoal recurrence,
            # outcome-KeyNode counts) instead of re-interpreting raw
            # screenshots. v1 mode: timeline is empty list; signals
            # block degenerates to empty.
            tl_window = (list(self.timeline[-8:])
                         if getattr(self, "timeline", None) else None)
            update_ledger(
                self.ledger,
                subgoal_screenshots=list(self._ledger_subgoal_screenshots),
                subgoal_actions=list(self._ledger_subgoal_actions),
                subgoal_plan_step=self._ledger_subgoal_plan_step,
                boundary_type=boundary_type,
                latest_a11y=a11y_after,
                latest_url=url_after,
                step_idx=step_idx,
                call_llm=self.call_llm,
                model=self.model,
                window_delta=delta,
                timeline_window=tl_window,
            )
        except Exception as e:
            if logger:
                logger.warning("[Ledger] summarizer boundary=%s failed: %s",
                               boundary_type, e)
        finally:
            # Reset window: drain buffers, reset plan-step text, and
            # snapshot the new window-start state for the next fire.
            self._ledger_subgoal_screenshots.clear()
            self._ledger_subgoal_actions.clear()
            self._ledger_subgoal_plan_step = self.current_subgoal or ""
            self._ledger_url_at_window_start = url_after
            self._ledger_a11y_at_window_start = a11y_after

    # --- v2 working-memory helpers (Event Timeline + KeyNode detector) ---

    # ----- _check_done_acceptance --------------------------------------------------
    def _check_done_acceptance(
        self, env, obs: dict, step_idx: int,
    ) -> Tuple[bool, Optional[str]]:
        """Run audit + cache gate on the planner's DONE claim.

        Returns ``(accepted, reject_reason)``. When accepted the caller
        terminates the task; when rejected the caller stages a 2nd
        planner call in force_replan mode with ``reject_reason`` as
        the stuck signal."""
        # Fire the ledger summarizer so DONE-time evidence is fresh.
        self._ledger_fire_boundary("done", obs, step_idx)
        if self.ledger is None:
            return True, None

        # verify-on-DONE (boundary mode): the per-step KeyNode poll is bypassed.
        # Actor-loop-exit reports refresh subgoal targets during execution; at
        # DONE time, verify any still-PENDING leaf so the DONE gate reads a real
        # check rather than a stale never-checked "pending".
        try:
            self._boundary_verify_done_leaves(obs, step_idx, env=env)
        except Exception as e:  # noqa: BLE001
            if logger:
                logger.warning("[BoundaryVerify] verify-on-DONE failed: %s", e)

        # Layer 2: cache gate (audit ABSTAIN, disabled, or unavailable).
        ok, reason = self.ledger.can_accept_done_claim()
        if ok:
            return True, None
        self._ledger_last_done_reason = reason
        # Bump cumulative DONE-rejection count — read by the recovery
        # dispatcher's stuck detector.
        self._done_rejected_count = getattr(self, "_done_rejected_count", 0) + 1
        if logger:
            logger.warning("[Ledger] DONE rejected: %s", reason)
        return False, (
            f"the progress ledger rejected a DONE claim — {reason}"
        )
