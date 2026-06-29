"""Controller mixin for the VLM-supervised actor burst loop."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from mm_agents.structagent.core.verifier.low_level_verify import (
    LowLevelVerifyContext,
    verify_actor_turn,
)
from mm_agents.structagent.utils.image import process_image


logger = logging.getLogger("desktopenv.qwen25vl_agent")


class ActorBurstController:
    """Small control layer between planner and actor.

    A burst spans multiple OSWorld ``predict`` calls. The actor emits one action
    batch, the runner executes it, and the next call verifies the before/after
    screenshots before deciding whether planner control is needed again.
    """

    def _pa_reset_actor_burst_state(self) -> None:
        if getattr(self, "_actor_burst_state", None) is not None:
            self._actor_burst_state.reset()

    def _pa_record_actor_burst_turn(
        self,
        ctx,
        *,
        actor_response: str,
        final_actions: List[str],
        actor_subgoal_done: bool,
        actor_exhausted: bool,
    ) -> None:
        subgoal = self.current_subgoal
        if not subgoal:
            self._pa_reset_actor_burst_state()
            return
        state = self._actor_burst_state
        state.start_if_needed(subgoal)
        terminal = None
        if actor_subgoal_done:
            terminal = "DONE"
        elif actor_exhausted:
            terminal = "EXHAUSTED"
        before = ctx.obs.get("screenshot") if isinstance(ctx.obs, dict) else None
        if isinstance(before, bytes):
            before_b64 = process_image(before)
        else:
            before_b64 = ctx.screenshot_bytes
            if isinstance(before_b64, bytes):
                before_b64 = process_image(before_b64)
        if not before_b64:
            return
        state.record_turn(
            before_screenshot_b64=before_b64,
            actor_response=actor_response or "",
            actions=final_actions or [],
            actor_terminal=terminal,
        )

    def _pa_actor_burst_precheck(self, ctx) -> Optional[str]:
        state = self._actor_burst_state
        if not state.active or not state.pending_verification:
            return None
        if state.subgoal != self.current_subgoal:
            state.reset()
            return None
        if state.last_actor_terminal == "EXHAUSTED":
            state.apply_verdict(
                status="stuck",
                reason="actor returned IMPOSSIBLE/FAIL for this subgoal",
            )
            return "actor_loop_end"
        after = ctx.obs.get("screenshot") if isinstance(ctx.obs, dict) else None
        if not isinstance(after, bytes) or not state.last_before_screenshot_b64:
            return None
        after_b64 = process_image(after)
        recent_turns = list(getattr(state, "verified_turns", []) or [])[-3:]
        prior_screenshots = [
            str(t.get("after_screenshot_b64") or "")
            for t in recent_turns
            if t.get("after_screenshot_b64")
        ]
        recent_actor_actions = [
            self._pa_render_actor_burst_history_line(t)
            for t in recent_turns
        ]

        verdict = verify_actor_turn(
            LowLevelVerifyContext(
                instruction=ctx.instruction,
                subgoal=state.subgoal or "",
                expected_post_state=self.current_subgoal_expected_post_state,
                actor_response=state.last_actor_response or "",
                actions=state.last_actions or [],
                before_screenshot_b64=state.last_before_screenshot_b64,
                after_screenshot_b64=after_b64,
                prior_screenshots_b64=prior_screenshots,
                recent_actor_actions=recent_actor_actions,
                actor_terminal=state.last_actor_terminal,
                prior_feedback=state.feedback_to_actor,
            ),
            call_llm=self.call_llm,
            model=self.verifier_model,
            logger=logger,
        )
        state.apply_verdict(
            status=verdict.status,
            reason=verdict.reason,
            feedback_to_actor=verdict.feedback_to_actor,
            after_screenshot_b64=after_b64,
        )
        self._pa_dump_actor_burst_debug(ctx, verdict)

        if verdict.status == "complete":
            return "actor_loop_end"
        if verdict.status in {"off_track", "stuck"}:
            return "actor_loop_end"
        if state.turn_idx >= self._ACTOR_BURST_MAX_TURNS:
            state.apply_verdict(
                status="stuck",
                reason=(
                    f"actor burst reached max turns ({state.turn_idx}) "
                    "without completing the current subgoal"
                ),
            )
            return "actor_loop_end"
        if (verdict.status == "no_effect"
                and state.no_effect_count >= self._ACTOR_BURST_NO_EFFECT_LIMIT):
            state.apply_verdict(
                status="stuck",
                reason=(
                    f"actor burst saw {state.no_effect_count} consecutive "
                    "no_effect verifier verdicts"
                ),
            )
            return "actor_loop_end"
        return "continue_actor"

    @staticmethod
    def _pa_render_actor_burst_history_line(turn: dict) -> str:
        actions = turn.get("actions") or []
        action_text = " | ".join(str(a).replace("\n", " ")[:220]
                                 for a in actions[:3])
        response = str(turn.get("actor_response") or "").replace("\n", " ")
        status = str(turn.get("verdict_status") or "")
        reason = str(turn.get("verdict_reason") or "").replace("\n", " ")
        parts = [f"turn {turn.get('turn_idx')}"]
        if action_text:
            parts.append(f"actions: {action_text[:500]}")
        elif response:
            parts.append(f"actor: {response[:300]}")
        if status:
            parts.append(f"verifier: {status} {reason[:240]}".strip())
        return " ; ".join(parts)

    def _pa_apply_actor_burst_feedback(self, subgoal: str) -> str:
        state = self._actor_burst_state
        feedback = state.feedback_to_actor if state.active else None
        if not feedback:
            return subgoal
        state.feedback_to_actor = None
        return (
            f"{subgoal}\n"
            f"[Low-level verifier feedback from the previous actor turn]: "
            f"{feedback}"
        )

    def _pa_dump_actor_burst_debug(self, ctx, verdict) -> None:
        # Always dump burst verify traces (gated only by results_dir).
        results_dir = getattr(self, "_results_dir", None)
        if not results_dir:
            return
        try:
            state = self._actor_burst_state
            d = Path(results_dir) / "actor_burst_debug"
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"step_{ctx.step_idx:03d}_turn_{state.turn_idx:02d}.json"
            path.write_text(json.dumps({
                "step_idx": ctx.step_idx,
                "subgoal": state.subgoal,
                "expected_post_state": self.current_subgoal_expected_post_state,
                "turn_idx": state.turn_idx,
                "no_effect_count": state.no_effect_count,
                "actor_response": state.last_actor_response,
                "actions": state.last_actions,
                "actor_terminal": state.last_actor_terminal,
                "recent_history_turns": [
                    {
                        k: v for k, v in t.items()
                        if k != "after_screenshot_b64"
                    }
                    for t in list(getattr(state, "verified_turns", []) or [])[-3:]
                ],
                "verdict": {
                    "status": verdict.status,
                    "reason": verdict.reason,
                    "evidence": verdict.evidence,
                    "feedback_to_actor": verdict.feedback_to_actor,
                    "error": verdict.error,
                    "raw": verdict.raw,
                },
            }, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("[ActorBurst] debug dump failed: %s", e)
