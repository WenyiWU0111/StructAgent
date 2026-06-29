"""Single source of truth for repository + standard-directory paths.

Why this exists
---------------
Hardcoding ``/path/to/StructAgent`` throughout 40+ files makes the
repo non-portable: a fresh ``git clone`` to any other directory breaks
imports immediately. This module derives the repo root from its own
location, with optional env-var overrides for non-standard deployments
(Docker mounts, training cluster paths).

Lives under ``mm_agents/structagent/`` — the canonical home for this
project's first-party Python code. (``memory/`` has since migrated in as
``structagent/memory/``; ``mm_agents/rollout/`` predates the split and
may migrate similarly.)

Usage
-----
::

    from mm_agents.structagent._paths import REPO_ROOT, RESULTS_DIR, EVAL_EXAMPLES

    config = REPO_ROOT / "evaluation_examples" / "test_nogdrive.json"
    out = RESULTS_DIR / "my_rollout" / "out.json"

Env overrides
-------------
Set ``OSWORLD_REPO_ROOT`` or ``OSWORLD_RESULTS_DIR`` to relocate either
without touching code (useful when results live on a different mount):

::

    OSWORLD_RESULTS_DIR=/mnt/big_disk/osworld_results python ...
"""
from __future__ import annotations

import os
from pathlib import Path

# This file lives at  <REPO_ROOT>/mm_agents/structagent/_paths.py
# So REPO_ROOT = the dir three levels up.
_DEFAULT_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

REPO_ROOT: Path = Path(
    os.environ.get("OSWORLD_REPO_ROOT") or _DEFAULT_REPO_ROOT
).resolve()

# Standard subdirectories. Override individually via env vars when needed
# (e.g. results dir on a bigger volume).
RESULTS_DIR: Path = Path(
    os.environ.get("OSWORLD_RESULTS_DIR") or (REPO_ROOT / "results")
).resolve()

EVAL_EXAMPLES: Path = REPO_ROOT / "evaluation_examples" / "examples"
ROLLOUT_RESULTS_DIR: Path = RESULTS_DIR / "pyautogui" / "screenshot"
ROLLOUT_MANIFEST_DIR: Path = RESULTS_DIR / "rollouts" / "_manifest"
PLANNER_EXPERIENCE_DIR: Path = RESULTS_DIR / "planner_experience"
VERIFIER_MEMORY_DIR: Path = RESULTS_DIR / "verifier_memory"
SUCCESSFUL_LEDGERS_DIR: Path = RESULTS_DIR / "successful_ledgers"
EXTERNAL_TRAJECTORIES_DIR: Path = RESULTS_DIR / "external_trajectories"


__all__ = [
    "REPO_ROOT",
    "RESULTS_DIR",
    "EVAL_EXAMPLES",
    "ROLLOUT_RESULTS_DIR",
    "ROLLOUT_MANIFEST_DIR",
    "PLANNER_EXPERIENCE_DIR",
    "VERIFIER_MEMORY_DIR",
    "SUCCESSFUL_LEDGERS_DIR",
    "EXTERNAL_TRAJECTORIES_DIR",
]
