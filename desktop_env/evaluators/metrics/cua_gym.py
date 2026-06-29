"""CUA-Gym reward.py bridge — OSWorld evaluator metric.

CUA-Gym ships a per-task ``reward.py`` that exposes a ``verify_task(file_path)``
function returning a [0, 1] float, plus ``WORKDIR`` / ``TASK_ID`` module-level
constants pointing at the file the agent is expected to produce on the VM.

This metric wraps that into the OSWorld dispatch convention:

  evaluator: {
    func:   "cua_gym_reward_py",
    result: {type: "vm_file", path: "/home/user/<TASK_ID>.<ext>", dest: ...},
    expected: {type: "cua_gym_reward_spec",
               reward_py_path: <abs path>,
               verify_func:    "verify_task" | null,
               task_dir:       <abs path>},
  }

The `result` getter (existing ``get_vm_file``) pulls the file from the VM
to a local cache path. The `expected` getter (new ``get_cua_gym_reward_spec``)
just echoes its config — it carries the spec the metric needs to invoke
reward.py.

We import reward.py via ``importlib.util.spec_from_file_location`` to keep
module names unique across the 10,910-task corpus (re-using ``reward`` as
the module name across tasks would lead to import-cache collisions).

Failures (file missing, import error, missing verify_task, exception in
verify_task) → score 0.0 with a logged reason. Designed to NEVER raise
through the OSWorld evaluator dispatcher.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
import traceback
from typing import Any, Dict, Optional

logger = logging.getLogger("desktopenv.metrics.cua_gym")


def _import_reward_module(reward_py_path: str, task_id_hint: str = ""):
    """Load reward.py as a unique sys.modules entry. task_id_hint disambiguates
    the module name so 10k tasks can be loaded over the lifetime of a worker
    without cache poisoning."""
    if not os.path.isfile(reward_py_path):
        return None, f"reward.py not found at {reward_py_path}"
    # Use a unique module name keyed on path to avoid sys.modules collisions
    module_name = f"_cua_gym_reward_{abs(hash(reward_py_path)):x}"
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, reward_py_path)
        if spec is None or spec.loader is None:
            return None, f"spec_from_file_location returned None for {reward_py_path}"
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as e:
        return None, f"import of reward.py crashed: {e}\n{traceback.format_exc()[:500]}"


def cua_gym_reward_py(
    result: Optional[str],
    expected: Optional[Dict[str, Any]] = None,
    **options,
) -> float:
    """Run a CUA-Gym task's reward.py to score the agent's VM-side output.

    Args:
        result: local path to the file pulled from the VM (from get_vm_file).
                None when the VM file pull failed.
        expected: {"reward_py_path", "verify_func"?, "task_dir"?} — config
                  from get_cua_gym_reward_spec.
        options: ignored.

    Returns: float in [0.0, 1.0]. 0.0 on any failure path.
    """
    if not expected or not isinstance(expected, dict):
        logger.warning("cua_gym_reward_py: missing/invalid `expected` "
                       "spec; got %r", expected)
        return 0.0
    reward_py_path = expected.get("reward_py_path")
    verify_func_name = expected.get("verify_func") or "verify_task"
    if not reward_py_path:
        logger.warning("cua_gym_reward_py: expected.reward_py_path missing")
        return 0.0

    if result is None:
        logger.warning(
            "cua_gym_reward_py: VM file pull returned None (no file at "
            "/home/user/<TASK_ID>.<ext>?). reward_py=%s", reward_py_path)
        return 0.0
    if not os.path.isfile(result):
        logger.warning(
            "cua_gym_reward_py: result path does not exist on host: %s", result)
        return 0.0

    mod, err = _import_reward_module(reward_py_path)
    if mod is None:
        logger.warning("cua_gym_reward_py: %s", err)
        return 0.0

    # Resolve the verify function. Most reward.py expose `verify_task`; a
    # minority use a task-specific name (e.g. verify_resume_merge). The
    # to_osworld_example pre-extraction may set verify_func in expected.
    fn = getattr(mod, verify_func_name, None)
    if fn is None:
        # Fallback: pick the first function whose name starts with "verify".
        for k in dir(mod):
            if k.startswith("verify") and callable(getattr(mod, k)):
                fn = getattr(mod, k)
                verify_func_name = k
                break
    if fn is None:
        logger.warning(
            "cua_gym_reward_py: no verify_* callable in %s", reward_py_path)
        return 0.0

    try:
        score = fn(result)
    except TypeError:
        # Some reward.py have verify_task() with no args (relies on module
        # globals to locate the file). Retry with no args.
        try:
            score = fn()
        except Exception as e:
            logger.warning("cua_gym_reward_py: %s() crashed: %s",
                           verify_func_name, e)
            return 0.0
    except Exception as e:
        logger.warning("cua_gym_reward_py: %s(%s) crashed: %s",
                       verify_func_name, result, e)
        return 0.0

    try:
        score_f = float(score)
    except (TypeError, ValueError):
        logger.warning("cua_gym_reward_py: non-numeric return from %s: %r",
                       verify_func_name, score)
        return 0.0

    # Clamp to [0, 1] for safety — some reward.py return raw component sums.
    return max(0.0, min(1.0, score_f))
