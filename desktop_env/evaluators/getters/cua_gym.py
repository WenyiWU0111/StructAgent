"""CUA-Gym expected-state getter.

OSWorld's evaluator dispatch resolves ``evaluator.expected.type`` to a
``get_<type>(env, config)`` getter. For CUA-Gym we don't need a getter that
queries the VM — the metric (``cua_gym_reward_py``) just needs the
``{reward_py_path, verify_func?, task_dir?}`` spec carried in the task config
itself. This getter is a transparent pass-through.
"""
from __future__ import annotations

from typing import Any, Dict


def get_cua_gym_reward_spec(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """No-op: echo the config back as the expected_state."""
    return config
