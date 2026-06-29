"""Typed source of truth for Causal-Agent runtime flags.

Flags are env-driven; ``CAConfig.from_env()`` returns an immutable snapshot,
resolved once at agent construction so the per-step path reads ``self.cfg.*``
instead of scattered ``os.environ`` lookups.

Boolean truthiness is uniform: ``{1,true,yes,on}`` (case-insensitive) → True,
else False (incl. unset). Exceptions: ``perceiver`` (int), ``memory_version``
(string), feasibility knobs (ints).

Deliberately out of scope (read at point of use, not feature flags):
secrets (OPENROUTER_API_KEY / DASHSCOPE_*, kept out of logged config), paths
(see ``_paths.py``), model names / offline-pipeline knobs, and retrieval
singletons (MEMORY_BANK_VERSION / MEMORY_TOP_K, owned by the memory subsystem).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY = ("1", "true", "yes", "on")


def _flag(name: str) -> bool:
    """Uniform boolean env flag: ``{1,true,yes,on}`` → True, else False."""
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _int_flag(name: str, default: int) -> int:
    """Integer env flag; ``default`` on unset / non-numeric."""
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class CAConfig:
    """Immutable snapshot of runtime feature flags. Build via :meth:`from_env`,
    not directly, so the env stays the single source of truth."""

    # — ledger / verification —
    disable_ledger: bool              # DISABLE_LEDGER — turn the ledger off entirely
    app_switch_verify: bool           # ENABLE_APP_SWITCH_VERIFY (default off)
    file_probe: bool                  # OSWORLD_FILE_PROBE — disk-content probe at init
    failure_attribution: bool         # USE_FAILURE_ATTRIBUTION — v2 LIVE override
    feasibility_verdict: bool         # ENABLE_FEASIBILITY_VERDICT — Fix 1: judge INFEASIBLE at replan>=K
    feasibility_verdict_k: int        # FEASIBILITY_VERDICT_K — consecutive replans before judging INFEASIBLE
    feasibility_recheck_every: int    # FEASIBILITY_RECHECK_EVERY — re-judge cadence (in replans)
    domain_coding_gate: bool          # ENABLE_DOMAIN_CODING_GATE — GUI-only domains (chrome/gimp/thunderbird): no cli_run/edit_json, decomposer gets domain knowledge, verifier/memory prefer GUI
    actor_burst_max_turns: int        # ACTOR_BURST_MAX_TURNS — max actor turns per subgoal before force_replan
    actor_burst_no_effect_limit: int  # ACTOR_BURST_NO_EFFECT_LIMIT — consecutive no_effect verdicts before force_replan

    # — done-auditor / stuck-diagnosis: ENV PART ONLY —
    #   loop.py OR's these with the matching ctor kwarg: effective = kwarg or env.
    done_auditor_env: bool                 # ENABLE_DONE_AUDITOR
    stuck_diagnosis_injection_env: bool    # ENABLE_STUCK_DIAGNOSIS_INJECTION

    # — perception (int-valued, kept historical) —
    perceiver: bool                   # PERCEIVER_ENABLED — bool(int(...))

    # — memory —
    planner_experience_memory: bool   # ENABLE_PLANNER_EXPERIENCE_MEMORY
    verifier_experience_memory: bool  # ENABLE_VERIFIER_EXPERIENCE_MEMORY
    memory_version: str               # MEMORY_VERSION (default "v2")

    @classmethod
    def from_env(cls) -> "CAConfig":
        return cls(
            disable_ledger=_flag("DISABLE_LEDGER"),
            app_switch_verify=_flag("ENABLE_APP_SWITCH_VERIFY"),
            file_probe=_flag("OSWORLD_FILE_PROBE"),
            failure_attribution=_flag("USE_FAILURE_ATTRIBUTION"),
            feasibility_verdict=_flag("ENABLE_FEASIBILITY_VERDICT"),
            feasibility_verdict_k=_int_flag("FEASIBILITY_VERDICT_K", 8),
            feasibility_recheck_every=_int_flag("FEASIBILITY_RECHECK_EVERY", 6),
            domain_coding_gate=_flag("ENABLE_DOMAIN_CODING_GATE"),
            actor_burst_max_turns=_int_flag("ACTOR_BURST_MAX_TURNS", 5),
            actor_burst_no_effect_limit=_int_flag("ACTOR_BURST_NO_EFFECT_LIMIT", 2),
            done_auditor_env=_flag("ENABLE_DONE_AUDITOR"),
            stuck_diagnosis_injection_env=_flag("ENABLE_STUCK_DIAGNOSIS_INJECTION"),
            perceiver=bool(int(os.environ.get("PERCEIVER_ENABLED", "0") or "0")),
            planner_experience_memory=_flag("ENABLE_PLANNER_EXPERIENCE_MEMORY"),
            verifier_experience_memory=_flag("ENABLE_VERIFIER_EXPERIENCE_MEMORY"),
            memory_version=(
                os.environ.get("MEMORY_VERSION", "v2").strip().lower() or "v2"
            ),
        )
