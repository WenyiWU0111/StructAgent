"""Plans subsystem.

- ``planner.Planner`` — the planner mixin (LLM-driven decomposition,
  CONTINUE / REPLAN / DONE dispatch, subgoal queue management).
  Folded into ``StructAgent`` via inheritance.
- ``prompts`` — prompt builders + format/quality rules used by
  ``Planner._pa_build_*`` methods. Was ``core/planner_prompts.py``.
- ``debug`` — optional HTML dump of planner state per turn. Was
  ``core/planner_debug.py``.
- ``plan_store`` (T4) — on-disk markdown plan file per run, so the
  planner sees its own prior commitments via a stable file rather
  than re-deriving from in-memory DAG every turn. See plan file
  Part I.3.

Re-exports ``Planner`` so existing
``from mm_agents.structagent.core.planner import Planner`` keeps
working after the file-to-folder restructure (covered by a shim at
``core/planner.py``-as-was; new code should import from
``core.planner``).
"""

from mm_agents.structagent.core.planner.planner import Planner

__all__ = ["Planner"]
