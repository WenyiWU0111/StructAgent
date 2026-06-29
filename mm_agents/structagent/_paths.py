"""Single source of truth for repo + standard-directory paths.

Repo root is derived from this file's location so a fresh clone to any
directory works without code edits. Override ``OSWORLD_REPO_ROOT`` /
``OSWORLD_RESULTS_DIR`` to relocate (e.g. results on a different mount).
"""
from __future__ import annotations

import os
from pathlib import Path

# <REPO_ROOT>/mm_agents/structagent/_paths.py — root is three levels up.
_DEFAULT_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

REPO_ROOT: Path = Path(
    os.environ.get("OSWORLD_REPO_ROOT") or _DEFAULT_REPO_ROOT
).resolve()

# Standard subdirectories. Override RESULTS_DIR via env when it lives elsewhere.
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
