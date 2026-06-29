"""TrajectoryHandle + iterators over internal trajectories.

Walks trajectory directories with consistent file-loading semantics
(used by the miners).

Internal trajectory layout (per task):
    results/pyautogui/screenshot/<run_tag>/<domain>/<task_id>/
        traj.jsonl               # per-step actions + responses
        traj.pretty.json         # pretty version
        trajectory.html          # human-readable with base64-embedded screenshots
        result.txt               # final score (0.0 or 1.0)
        runtime.log
        recording.mp4
        keynode_debug/
            step_NNN_keynode_input.json
            step_NNN_keynode_verdicts.json
            step_NNN_keynode_llm_messages.json    # only if took_llm_path
            step_NNN_keynode_llm_raw.txt          # only if took_llm_path
        perceiver_debug/

Step-numbering alignment (verified on multiple tasks):
    keynode_debug filename step_NNN     ↔ traj.jsonl unique step_num = NNN
    keynode_debug internal step_idx     = NNN - 1  (0-indexed)
    trajectory.html <h3>Step X</h3>     ↔ traj.jsonl row (X-1)  [1-indexed labels]

    For a KeyNode call at step_num=NNN, the *observation state* it saw equals
    the screenshot taken at the FIRST traj row whose step_num > NNN. That's
    "Step (row_idx+1)" in trajectory.html (1-indexed). For the last step,
    fall back to the final screenshot.

Internal success ledger:
    results/successful_ledgers/<domain>/<task_id>.json
    results/successful_ledgers/_index.jsonl  # registry of all successful task_ids
"""
from __future__ import annotations
from mm_agents.structagent._paths import REPO_ROOT

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

SUCCESSFUL_LEDGERS_DIR = REPO_ROOT / "results/successful_ledgers"
SUCCESSFUL_INDEX = SUCCESSFUL_LEDGERS_DIR / "_index.jsonl"
PYAUTOGUI_ROOT = REPO_ROOT / "results/pyautogui/screenshot"

_STEP_RE = re.compile(r"step_(\d{3})_keynode_verdicts\.json$")


@dataclass
class TrajectoryHandle:
    """One trajectory's location + helpers."""

    task_id: str
    domain: str
    source_run: Path            # screenshot/<run_tag>/<domain>/<task_id>/
    source_model_version: str   # e.g. "vllm_claude-opus_planner_..._v0"
    ledger_path: Optional[Path] = None   # set only on success
    score: Optional[float] = None        # 1.0 success, 0.0 fail

    # ---------- existence checks ----------

    def has_keynode_debug(self) -> bool:
        return (self.source_run / "keynode_debug").is_dir()

    def has_traj_jsonl(self) -> bool:
        return (self.source_run / "traj.jsonl").is_file()

    def has_ledger(self) -> bool:
        return self.ledger_path is not None and self.ledger_path.is_file()

    # ---------- loaders ----------

    def load_traj_jsonl(self) -> List[dict]:
        p = self.source_run / "traj.jsonl"
        if not p.is_file():
            return []
        out = []
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def load_ledger(self) -> Optional[dict]:
        if not self.has_ledger():
            return None
        return json.loads(self.ledger_path.read_text())

    def list_keynode_verdict_paths(self) -> List[Path]:
        """Sorted list of step_NNN_keynode_verdicts.json paths."""
        kd = self.source_run / "keynode_debug"
        if not kd.is_dir():
            return []
        out = []
        for p in kd.iterdir():
            if _STEP_RE.search(p.name):
                out.append(p)
        return sorted(out, key=lambda p: int(_STEP_RE.search(p.name).group(1)))

    def load_keynode_verdict(self, step_idx: int) -> Optional[dict]:
        p = self.source_run / "keynode_debug" / f"step_{step_idx:03d}_keynode_verdicts.json"
        if not p.is_file():
            return None
        return json.loads(p.read_text())

    def load_keynode_input(self, step_idx: int) -> Optional[dict]:
        p = self.source_run / "keynode_debug" / f"step_{step_idx:03d}_keynode_input.json"
        if not p.is_file():
            return None
        return json.loads(p.read_text())

    def load_keynode_llm_raw(self, step_idx: int) -> Optional[str]:
        p = self.source_run / "keynode_debug" / f"step_{step_idx:03d}_keynode_llm_raw.txt"
        if not p.is_file():
            return None
        return p.read_text()

    def load_keynode_llm_messages(self, step_idx: int) -> Optional[dict]:
        p = self.source_run / "keynode_debug" / f"step_{step_idx:03d}_keynode_llm_messages.json"
        if not p.is_file():
            return None
        return json.loads(p.read_text())

    def read_result_score(self) -> Optional[float]:
        p = self.source_run / "result.txt"
        if not p.is_file():
            return None
        try:
            return float(p.read_text().strip())
        except (ValueError, OSError):
            return None

    def last_traj_step(self) -> Optional[dict]:
        traj = self.load_traj_jsonl()
        return traj[-1] if traj else None

    def last_traj_steps(self, n: int = 3) -> List[dict]:
        traj = self.load_traj_jsonl()
        return traj[-n:] if traj else []

    def agent_emitted_done(self) -> bool:
        """True if last traj step shows the agent self-declared DONE."""
        last = self.last_traj_step()
        if not last:
            return False
        if last.get("done") is True:
            return True
        action = last.get("action")
        if isinstance(action, dict):
            return action.get("action_type") == "DONE"
        if isinstance(action, str):
            return "DONE" in action[:50]
        return False

    # ---------- HTML screenshot extraction + alignment ----------

    def html_screenshots_b64(self) -> List[str]:
        """trajectory.html → list of base64 PNGs, one per <h3>Step N</h3>
        block (index 0 = Step 1). [] if the file is missing."""
        if hasattr(self, "_html_screenshots_cache"):
            return self._html_screenshots_cache
        p = self.source_run / "trajectory.html"
        if not p.is_file():
            self._html_screenshots_cache = []
            return self._html_screenshots_cache
        html = p.read_text(encoding="utf-8", errors="replace")
        # <h3>Step N</h3> immediately followed by <img class='screenshot' ...>
        matches = re.findall(
            r"<h3>Step (\d+)</h3>\s*<img class='screenshot' src='data:image/png;base64,([A-Za-z0-9+/=]+)'",
            html,
        )
        matches_int = sorted(((int(n), b64) for n, b64 in matches), key=lambda x: x[0])
        self._html_screenshots_cache = [b64 for _, b64 in matches_int]
        return self._html_screenshots_cache

    def keynode_step_to_html_idx(self, keynode_step_num: int) -> Optional[int]:
        """keynode step_NNN (1-indexed filename) → 0-indexed screenshot list
        index for the state KeyNode observed.

        Rule: the state at step_num=NNN is the screenshot at the FIRST traj row
        with step_num > NNN; that row's list index is row_idx. None if no later
        row exists (last step) — caller falls back to the last screenshot.
        """
        traj = self.load_traj_jsonl()
        for i, row in enumerate(traj):
            sn = row.get("step_num")
            if sn is None:
                continue
            if sn > keynode_step_num:
                return i
        return None

    def get_keynode_observation_screenshot(self, keynode_step_num: int) -> Optional[str]:
        """Base64 PNG of the screenshot KeyNode observed for step_num=NNN.
        Falls back to the last screenshot for the final step; None if none.
        """
        screenshots = self.html_screenshots_b64()
        if not screenshots:
            return None
        idx = self.keynode_step_to_html_idx(keynode_step_num)
        if idx is None or idx >= len(screenshots):
            return screenshots[-1]
        return screenshots[idx]


# ---------- iterators ----------

def iter_successful_trajectories() -> Iterator[TrajectoryHandle]:
    """Yield a handle per row in successful_ledgers/_index.jsonl."""
    if not SUCCESSFUL_INDEX.is_file():
        return
    with SUCCESSFUL_INDEX.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = row["task_id"]
            domain = row["domain"]
            source_run = Path(row["source_run"])
            ledger_path = SUCCESSFUL_LEDGERS_DIR / domain / f"{task_id}.json"
            yield TrajectoryHandle(
                task_id=task_id,
                domain=domain,
                source_run=source_run,
                source_model_version=row.get("source_model_version", ""),
                ledger_path=ledger_path if ledger_path.is_file() else None,
                score=float(row.get("score", 1.0)),
            )


def iter_pyautogui_runs() -> Iterator[Path]:
    """All <run_tag> dirs under results/pyautogui/screenshot/."""
    if not PYAUTOGUI_ROOT.is_dir():
        return
    for p in PYAUTOGUI_ROOT.iterdir():
        if p.is_dir() and not p.name.startswith("_") and p.name != "curated":
            yield p


def iter_all_pyautogui_tasks() -> Iterator[TrajectoryHandle]:
    """Every <run_tag>/<domain>/<task_id>/ — success AND failure, no score
    filtering (caller decides)."""
    for run_dir in iter_pyautogui_runs():
        run_tag = run_dir.name
        for domain_dir in run_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            for task_dir in domain_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                task_id = task_dir.name
                if not _is_uuid_like(task_id):
                    continue
                yield TrajectoryHandle(
                    task_id=task_id,
                    domain=domain,
                    source_run=task_dir,
                    source_model_version=run_tag,
                    ledger_path=None,
                    score=None,
                )


def _is_uuid_like(s: str) -> bool:
    return len(s) == 36 and s.count("-") == 4


def iter_internal_false_done_trajectories() -> Iterator[TrajectoryHandle]:
    """Internal fail-with-DONE trajectories (from scan_internal_failures.py).
    Handles have score=0.0 and ledger_path=None (failures aren't in
    _index.jsonl) — use load_keynode_input, not load_ledger."""
    audit_path = REPO_ROOT / "results/verifier_memory/_audit/internal_failures_audit.json"
    if not audit_path.is_file():
        return
    data = json.loads(audit_path.read_text())
    for task_id, info in (data.get("chosen_per_task") or {}).items():
        yield TrajectoryHandle(
            task_id=task_id,
            domain=info["domain"],
            source_run=Path(info["source_run"]),
            source_model_version=info["source_model_version"],
            ledger_path=None,
            score=info.get("score", 0.0),
        )


def terminal_screenshot_b64(handle: TrajectoryHandle) -> Optional[str]:
    """Last HTML screenshot — the terminal state the agent ended in."""
    screenshots = handle.html_screenshots_b64()
    return screenshots[-1] if screenshots else None


def latest_outcomes_status(handle: TrajectoryHandle) -> List[dict]:
    """Outcomes status from the last keynode_verdict file:
    list of {outcome_id, verify_kind, evidence, ...}."""
    vps = handle.list_keynode_verdict_paths()
    if not vps:
        return []
    try:
        last = json.loads(vps[-1].read_text())
    except (json.JSONDecodeError, OSError):
        return []
    out = []
    for po in last.get("per_outcome", []):
        emitted = po.get("emitted_keynode") or {}
        out.append({
            "outcome_id": po.get("outcome_id"),
            "verify_kind": po.get("verify_kind"),
            "previous_state": po.get("previous_state"),
            "emitted_kind": emitted.get("kind"),  # outcome_satisfied / outcome_blocked / None
            "evidence": emitted.get("evidence"),
            "confidence": emitted.get("confidence"),
        })
    return out
