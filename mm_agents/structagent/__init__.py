"""StructAgent — a verifier-centered planner / actor / verifier loop with a
typed, evidence-backed outcome ledger (the task *State*).

Package layout — ROLES (what the agent does each step) live under ``core/``;
SUBSYSTEMS (the State / tools the roles operate on) are top-level siblings:

    structagent/
      ├── loop.py              — StructAgent class (public entry point)
      ├── _paths.py            — repo-root / standard-dir resolution
      │
      ├── core/                — ROLES (mixins composed into the loop)
      │     ├── planner/         decide the next subgoal
      │     ├── actor/           execute (decompose + ground + compile)
      │     ├── verifier/        check milestones; emit evidence events
      │     ├── recovery/        stuck detection + replan routing
      │     └── final_audit/     DONE-time holistic gate (DONE auditor)
      │
      ├── ledger/              — SUBSYSTEM: milestone progress STATE
      ├── perception/          — SUBSYSTEM: screen → structured snapshot
      ├── attribution/         — SUBSYSTEM: failure diagnosis + routing
      ├── memory/              — SUBSYSTEM: experience mining + retrieval
      ├── actions/             — SUBSYSTEM: typed app/office operations
      ├── domain/              — per-app knowledge + evidence guides
      └── utils/               — llm_client / image / a11y helpers

Public usage:

    from mm_agents.structagent import StructAgent
    agent = StructAgent(model="vllm_qwen35-vl")
    actions = agent.predict(instruction, obs)
"""

from .loop import StructAgent  # noqa: F401

__all__ = ["StructAgent"]
