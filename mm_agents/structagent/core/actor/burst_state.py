"""Small state container for supervised actor bursts.

The OSWorld runner executes actions outside the agent, so a "burst" spans
multiple predict calls: one call emits an actor action, the next call observes
its effect and verifies whether the same subgoal should keep control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ActorBurstState:
    """Mutable per-subgoal state for the VLM-supervised actor loop."""

    active: bool = False
    subgoal: Optional[str] = None
    turn_idx: int = 0
    no_effect_count: int = 0
    feedback_to_actor: Optional[str] = None
    last_before_screenshot_b64: Optional[str] = None
    last_actor_response: Optional[str] = None
    last_actions: List[str] = field(default_factory=list)
    last_actor_terminal: Optional[str] = None
    pending_verification: bool = False
    last_verdict: Optional[str] = None
    last_reason: Optional[str] = None
    verified_turns: List[Dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.active = False
        self.subgoal = None
        self.turn_idx = 0
        self.no_effect_count = 0
        self.feedback_to_actor = None
        self.last_before_screenshot_b64 = None
        self.last_actor_response = None
        self.last_actions = []
        self.last_actor_terminal = None
        self.pending_verification = False
        self.last_verdict = None
        self.last_reason = None
        self.verified_turns = []

    def start_if_needed(self, subgoal: Optional[str]) -> None:
        if not subgoal:
            self.reset()
            return
        if not self.active or self.subgoal != subgoal:
            self.reset()
            self.active = True
            self.subgoal = subgoal

    def record_turn(
        self,
        *,
        before_screenshot_b64: str,
        actor_response: str,
        actions: List[str],
        actor_terminal: Optional[str] = None,
    ) -> None:
        self.active = True
        self.last_before_screenshot_b64 = before_screenshot_b64
        self.last_actor_response = actor_response or ""
        self.last_actions = list(actions or [])
        self.last_actor_terminal = actor_terminal
        self.pending_verification = True
        self.turn_idx += 1

    def apply_verdict(
        self,
        *,
        status: str,
        reason: str = "",
        feedback_to_actor: Optional[str] = None,
        after_screenshot_b64: Optional[str] = None,
    ) -> None:
        self.last_verdict = status
        self.last_reason = reason
        self.pending_verification = False
        if after_screenshot_b64:
            self.verified_turns.append({
                "turn_idx": self.turn_idx,
                "after_screenshot_b64": after_screenshot_b64,
                "actor_response": self.last_actor_response or "",
                "actions": list(self.last_actions or []),
                "actor_terminal": self.last_actor_terminal,
                "verdict_status": status,
                "verdict_reason": reason or "",
            })
            self.verified_turns = self.verified_turns[-6:]
        if status == "no_effect":
            self.no_effect_count += 1
        elif status in {"continue", "complete"}:
            self.no_effect_count = 0
        if feedback_to_actor:
            self.feedback_to_actor = feedback_to_actor
