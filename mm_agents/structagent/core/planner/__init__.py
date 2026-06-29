"""Planner subsystem.

- ``planner.Planner`` — planner mixin (LLM decomposition, CONTINUE/REPLAN/DONE
  dispatch, subgoal queue), folded into ``StructAgent`` via inheritance.
- ``prompts`` — prompt builders + format/quality rules for ``Planner._pa_build_*``.
- ``debug`` — optional per-turn HTML dump of planner state.
- ``plan_store`` (T4) — on-disk markdown plan per run, so the planner reads its
  own prior commitments from a stable file instead of re-deriving the DAG each
  turn. See plan file Part I.3.

Re-exports ``Planner`` so existing imports survive the file-to-folder move.
"""

from mm_agents.structagent.core.planner.planner import Planner

__all__ = ["Planner"]
