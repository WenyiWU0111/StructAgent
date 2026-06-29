"""Unified loaders for the memory mining pipeline.

Three data sources, one common record shape, one downstream pipeline:

  results/planner_experience/<domain>/<tid>.json   ← internal (from rollouts)
  results/agentnet_normalized/<aid>.json           ← AgentNet Ubuntu, normalized
  results/mind2web_normalized/<aid>.json           ← Multimodal-Mind2Web, normalized

Each loader yields ``UnifiedRecord`` so the pooling / clustering / polishing
stages don't care about origin. task_ids are hashed to opaque 12-char SHA1
prefixes and ``source_label`` uses neutral descriptors, so provenance can't be
read off the output.

source_label scheme (see SourceLabel below):
  internal rollout / AgentNet Ubuntu → "linux_desktop_trajectories"
  AgentNet Win/Mac                   → "cross_os_desktop_trajectories" (defer)
  Multimodal-Mind2Web                → "web_navigation_trajectories"
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[5]  # repo root; [4] would stop at mm_agents/
INTERNAL_DIR = REPO / "results" / "planner_experience"
AGENTNET_DIR = REPO / "results" / "agentnet_normalized"
MIND2WEB_DIR = REPO / "results" / "mind2web_normalized"
MIND2WEB_AUG_DIR = MIND2WEB_DIR / "_augmented"


# ─── Source labelling ────────────────────────────────────────────────────────
class SourceLabel:
    """Neutral source labels for output. Multiple sources may collapse into one
    label by design — see module docstring."""
    LINUX_DESKTOP = "linux_desktop_trajectories"
    WEB_NAVIGATION = "web_navigation_trajectories"


def _hash_id(raw: str, length: int = 12) -> str:
    """SHA1 truncated to ``length`` hex chars. Anonymises task_ids so output
    can't be traced to a source dataset by ID format."""
    return hashlib.sha1((raw or "").encode("utf-8")).hexdigest()[:length]


# ─── Common record ──────────────────────────────────────────────────────────
@dataclass
class UnifiedRecord:
    """Common shape consumed by downstream pooling / clustering / polishing.

    Fields:
      task_id:      Original source task_id (UUID / annotation_id). Never
                    overwritten — the identity used for pipeline keying and
                    joins back to source data. task_hash is SHA1(task_id)[:12],
                    used only at output-packaging time to anonymise the release.
      domain:       Framework-normalised domain (chrome / libreoffice_calc /
                    vs_code / multi_apps / ...). Scopes clustering.
      source_label: Neutral label (SourceLabel.*) for output provenance.
      key_actions:  List of {subgoal_idx, action, tool_or_widget} dicts.
      outcome:      {completed, final_observation_excerpt, final_action,
                    failure_modes, recovery_attempts}.
      task_completed: True/False for L_v3 bucketing; None for internal entries
                    that don't break out true/false.
      raw_meta:     Source-specific debug metadata; never written to v3 outputs.
    """
    task_id: str
    task_hash: str
    instruction: str
    domain: str
    source_label: str
    subgoals: List[str] = field(default_factory=list)
    key_actions: List[Dict[str, Any]] = field(default_factory=list)
    outcome: Dict[str, Any] = field(default_factory=dict)
    pitfalls: List[str] = field(default_factory=list)
    task_completed: Optional[bool] = None
    raw_meta: Dict[str, Any] = field(default_factory=dict)


# ─── Domain normalisation ───────────────────────────────────────────────────
_AGENTNET_DOMAIN_MAP = {
    # AgentNet uses CamelCase; framework uses snake_case
    "Chrome": "chrome",
    "VScode": "vs_code",
    "VLC": "vlc",
    "Gimp": "gimp",
    "MultiApp": "multi_apps",
    "OS": "os",
    "Thunderbird": "thunderbird",
    "error_correction": "multi_apps",  # heterogeneous; multi-app is closest
    "libreoffice_calc": "libreoffice_calc",
    "libreoffice_impress": "libreoffice_impress",
    "libreoffice_writer": "libreoffice_writer",
}


def _normalise_agentnet_domain(raw: str) -> str:
    return _AGENTNET_DOMAIN_MAP.get(raw or "", (raw or "").lower())


# ─── Loaders ────────────────────────────────────────────────────────────────
def _safe_load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_internal() -> Iterator[UnifiedRecord]:
    """Load mined internal trajectories.

    The miner (memory/mine_planner_experience) writes per-task records under
    results/planner_experience/<domain>/<task_id>.json, each with a ``judgment``
    block: {overall_strategy, core_productive_segments, segment_judgments[
    {seg_id, classification, key_subgoal, key_action, recovery_lesson,
    anti_pattern, ...}], task_class_signature, global_pitfalls}.

    PRODUCTIVE segments → subgoals + key_actions; pitfalls = global_pitfalls +
    anti_patterns; recovery_attempts = recovery_lessons.
    """
    if not INTERNAL_DIR.exists():
        logger.warning("internal dir missing: %s", INTERNAL_DIR)
        return
    for domain_dir in sorted(INTERNAL_DIR.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith("_"):
            continue
        for fp in sorted(domain_dir.glob("*.json")):
            d = _safe_load(fp)
            if not d:
                continue
            j = d.get("judgment") or {}
            seg_js = j.get("segment_judgments") if isinstance(j, dict) else None
            if not isinstance(seg_js, list) or not seg_js:
                continue
            subgoals: List[str] = []
            key_actions: List[Dict[str, Any]] = []
            recovery_attempts: List[str] = []
            extra_pitfalls: List[str] = []
            for sj in seg_js:
                if not isinstance(sj, dict):
                    continue
                if sj.get("classification") != "PRODUCTIVE":
                    # still harvest recovery / anti-pattern signal from these
                    if sj.get("anti_pattern"):
                        extra_pitfalls.append(sj["anti_pattern"])
                    if sj.get("recovery_lesson"):
                        recovery_attempts.append(sj["recovery_lesson"])
                    continue
                if sj.get("key_subgoal"):
                    subgoals.append(sj["key_subgoal"])
                    if sj.get("key_action"):
                        key_actions.append({
                            "subgoal_idx": len(subgoals) - 1,
                            "action": sj["key_action"],
                            "tool_or_widget": "",
                        })
                if sj.get("anti_pattern"):
                    extra_pitfalls.append(sj["anti_pattern"])
                if sj.get("recovery_lesson"):
                    recovery_attempts.append(sj["recovery_lesson"])
            if not subgoals:
                continue
            pitfalls = list(j.get("global_pitfalls") or []) + extra_pitfalls
            score = d.get("score", 0)
            completed = (score >= 1.0) if score is not None else True
            raw_tid = d.get("task_id") or fp.stem
            yield UnifiedRecord(
                task_id=raw_tid,
                task_hash=_hash_id(raw_tid),
                instruction=d.get("task_text") or "",
                domain=domain_dir.name,
                source_label=SourceLabel.LINUX_DESKTOP,
                subgoals=subgoals,
                key_actions=key_actions,
                outcome={
                    "completed": completed,
                    "final_observation_excerpt": "",
                    "final_action": "",
                    "failure_modes": [],
                    "recovery_attempts": recovery_attempts,
                },
                pitfalls=pitfalls,
                task_completed=completed,
                raw_meta={"_source": "internal",
                          "_task_class": j.get("task_class_signature"),
                          "_path": str(fp.relative_to(REPO))},
            )


def load_agentnet() -> Iterator[UnifiedRecord]:
    """Load AgentNet normalized records (one json per task).

    Uses ``natural_language_task`` as the canonical instruction — the other
    fields don't capture full task scope (``instruction`` is just a topic line,
    ``subgoals[0]`` is "Launch X").
    """
    if not AGENTNET_DIR.exists():
        logger.warning("AgentNet dir missing: %s", AGENTNET_DIR)
        return
    for fp in sorted(AGENTNET_DIR.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        d = _safe_load(fp)
        if not d:
            continue
        parsed = d.get("parsed") or {}
        if not parsed or parsed.get("_unparsed"):
            continue
        raw_tid = d.get("task_id") or fp.stem
        instruction = (d.get("natural_language_task")
                       or d.get("instruction")
                       or "")
        yield UnifiedRecord(
            task_id=raw_tid,
            task_hash=_hash_id(raw_tid),
            instruction=instruction,
            domain=_normalise_agentnet_domain(d.get("domain") or ""),
            source_label=SourceLabel.LINUX_DESKTOP,
            subgoals=list(parsed.get("subgoals") or []),
            key_actions=list(parsed.get("key_actions") or []),
            outcome=parsed.get("outcome") or {},
            pitfalls=list(parsed.get("pitfalls") or []),
            task_completed=d.get("task_completed"),
            raw_meta={"_source": "agentnet",
                      "_path": str(fp.relative_to(REPO))},
        )


def load_mind2web() -> Iterator[UnifiedRecord]:
    """Load Multimodal-Mind2Web normalized records (one json per task).

    The normalized file only carries ``parsed.{subgoals,key_actions,...}``; the
    instruction + website + domain/subdomain live in the matching
    ``_augmented/<task_id>.json``, which we side-load.
    """
    if not MIND2WEB_DIR.exists():
        logger.warning("Mind2Web dir missing: %s", MIND2WEB_DIR)
        return
    for fp in sorted(MIND2WEB_DIR.glob("*.json")):
        if fp.name.startswith("_") or fp.parent.name == "_augmented":
            continue
        d = _safe_load(fp)
        if not d:
            continue
        parsed = d.get("parsed") or {}
        if not parsed or parsed.get("_unparsed"):
            continue
        raw_tid = d.get("task_id") or fp.stem

        # Side-load augmented intermediate for instruction + website meta.
        aug_fp = MIND2WEB_AUG_DIR / f"{raw_tid}.json"
        aug = _safe_load(aug_fp) or {}

        yield UnifiedRecord(
            task_id=raw_tid,
            task_hash=_hash_id(raw_tid),
            instruction=aug.get("instruction") or d.get("instruction") or "",
            domain="chrome",   # all Mind2Web entries are web → chrome domain
            source_label=SourceLabel.WEB_NAVIGATION,
            subgoals=list(parsed.get("subgoals") or []),
            key_actions=list(parsed.get("key_actions") or []),
            outcome=parsed.get("outcome") or {},
            pitfalls=list(parsed.get("pitfalls") or []),
            task_completed=True,   # all Mind2Web rows are annotated success
            raw_meta={"_source": "mind2web",
                      "_path": str(fp.relative_to(REPO)),
                      "website": aug.get("mind2web_website"),
                      "mind2web_domain": aug.get("mind2web_domain"),
                      "mind2web_subdomain": aug.get("mind2web_subdomain")},
        )


# ─── Pool ───────────────────────────────────────────────────────────────────
def pool_all_sources() -> List[UnifiedRecord]:
    """Concatenate all three sources, undefined order. Caller shuffles."""
    all_records: List[UnifiedRecord] = []
    for name, fn in (("internal", load_internal),
                     ("agentnet", load_agentnet),
                     ("mind2web", load_mind2web)):
        before = len(all_records)
        for rec in fn():
            all_records.append(rec)
        logger.info("loaded %d records from %s (total now %d)",
                    len(all_records) - before, name, len(all_records))
    return all_records


def summarise_pool(records: List[UnifiedRecord]) -> Dict[str, Any]:
    """Per-domain / source-label / completed counts. Sanity-check before
    clustering."""
    from collections import Counter
    by_dom = Counter(r.domain for r in records)
    by_label = Counter(r.source_label for r in records)
    by_completed = Counter(str(r.task_completed) for r in records)
    return {
        "total": len(records),
        "per_domain": dict(by_dom.most_common()),
        "per_source_label": dict(by_label),
        "per_task_completed": dict(by_completed),
        "per_source_raw": dict(Counter(
            r.raw_meta.get("_source") for r in records
        )),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    pool = pool_all_sources()
    print(json.dumps(summarise_pool(pool), indent=2))
