"""Failure Attribution v2 — causal-flavoured root-cause diagnosis with
multi-layer-memory critique.

WHY a v2 of stuck_diagnosis
---------------------------
The v1 module (`stuck_diagnosis.py`) does a structurally sound job
producing one-line root cause + a 4-category tag + a concrete hint.
Examining real dumps across multiple runs surfaced four recurring
failure modes:

1. **Goal vs evidence confusion.** verify spec's `evidence_hint`
   (e.g. "Layer Properties dialog shows height=512") gets read AS the
   goal — so the LLM proposes "open the dialog" instead of "resize the
   layer".
2. **Zero memory awareness.** The diagnose sees the trajectory but
   doesn't know the memory bank ALREADY has a canonical approach for
   this exact subgoal — e.g. an L1 entry "GIMP: Layer > Scale Layer
   resizes ONE layer; Image > Scale Image resizes the canvas; common
   conflation" would have pinpointed the failure instantly.
3. **Category drift without evidence.** The "no-flip-without-evidence"
   prompt rule mitigates but doesn't eliminate this; the LLM still
   flips tactical → strategic across replans on the same focus.
4. **Role attribution is collapsed into category.** v1 has
   `tactical/strategic/info/unclear` (HOW to fix) but no
   `planner/actor/verifier/env` (WHO is responsible). The two
   dimensions are orthogonal and both matter for routing.

v2's response shape
-------------------
Forces the LLM to walk a five-phase reasoning trail with fields
ordered as RECALL → OBSERVE → CRITIQUE → ATTRIBUTE → ACT, so
conclusions come AFTER evidence anchoring:

    {
      "memory_recall": {
        "canonical_approach": "<1 sentence what memory says>",
        "memory_referenced": [{"layer","id","score","why_relevant"}]
      },
      "observation_summary": "...",
      "evidence_quotes":   ["...", "..."],
      "memory_critique":   {"memory_followed", "divergence_description"},
      "root_cause_summary": "...",
      "attributed_to_role": "planner|actor|verifier|env|unclear",
      "category":           "tactical|strategic|info_gap|unclear",
      "tactical_subkind":   "wrong_target|target_missing|...|null",
      "missing_or_wrong":   [...],
      "confidence":         "high|medium|low",
      "next_action_hint":   "<one concrete action signature>"
    }

The role × category combinations encode where the loop broke:
  • planner+tactical   — chose subtly wrong subgoal under right plan
  • planner+strategic  — chose wrong overall plan
  • actor+tactical     — execution off under correct subgoal
  • verifier+tactical  — false-rejected a real completion
  • verifier+strategic — verify spec doesn't match actual goal
  • env+tactical       — race/dialog/modal interrupted
  • env+strategic      — environment fundamentally blocks this path

Env-gated rollout
-----------------
Set ``USE_FAILURE_ATTRIBUTION=1`` to route the recovery hook here. v1
remains live as the default for safe A/B comparison; the replay tool
in ``scripts/dev_tools/replay_failure_attribution.py`` runs v2 against
all historic stuck_diagnosis dumps for side-by-side eval.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# Reuse v1's snapshot-building helpers — they're stable and battle-tested.
# Only the *system prompt* + *output schema* + *memory injection* differ.
from mm_agents.structagent.attribution.stuck_diagnosis import (
    _render_goal_section,
    _render_available_actions_section,
    _render_cli_run_section,
    _render_action_rollup_section,
    _render_observation_section,
    _render_recent_attempts_section,
    _render_verifier_trace_section,
    _render_recovery_trigger_section,
    _render_prior_diagnoses_section,
    _screenshot_data_url_from_observation,
    _MAX_RECENT_STEPS,
    _digest,
)


_SYSTEM_PROMPT = """You are a root-cause failure-attribution analyst for a GUI agent stuck on a single goal. You produce ONE JSON object — nothing else. No prose before or after. No markdown headings. No "Phase 1" labels. Just the JSON.

JSON field requirements (fill in this exact order so the conclusion is grounded in earlier evidence):

1. memory_recall.canonical_approach: one sentence summarising what RETRIEVED MEMORY recommends for this goal. If the memory block is empty / "no relevant memory found", use "<no memory>".
2. memory_recall.memory_referenced: array of 1-4 memory entries you actually relied on. Each entry: {"layer":"L1|L2|L3a|L3c", "id":"<verbatim id from memory block>", "score": <verbatim score>, "why_relevant":"<one clause>"}. Empty array if no memory was relevant.
3. observation_summary: 2 sentences MAX — what the agent did, what the current state actually shows.
4. evidence_quotes: array of 2-4 verbatim strings from the snapshot (an a11y row, a verifier-trace line, a CLI stderr line, a RECENT[N] action line). Format each as "<source>: <quoted text>". Every claim downstream MUST be anchored to one of these.
5. memory_critique.memory_followed: one of "true" / "false" / "partial" / "n/a". Use "n/a" only when memory_referenced is empty.
6. memory_critique.divergence_description: 1-2 sentences naming exactly WHERE the agent's path deviates from memory's canonical approach. Be specific. If memory was followed correctly but task still failed, say so — that's a strong attribution signal toward verifier/env.
7. root_cause_summary: 2-3 sentences pinpointing why stuck RIGHT NOW. Must reference fields 4 and 6.
8. attributed_to_role: PICK ONE.
   - "planner"  — wrong subgoal / wrong plan / mis-read the spec
   - "actor"    — execution off under correct plan (wrong widget, bad CLI args)
   - "verifier" — false-rejected a real completion OR verify spec mismatched to goal
   - "env"      — race / dialog / modal / permission / state issue
   - "unclear"  — insufficient signal
9. category: PICK ONE (orthogonal to role).
   - "tactical"  — same strategy, different tactic
   - "strategic" — strategy can't satisfy spec; switch approach category
   - "info_gap"  — a value the spec needs is unknown to the agent
   - "unclear"   — insufficient signal
10. tactical_subkind: PICK ONE only when category="tactical": "wrong_target" / "target_missing" / "target_blocked" / "blocking_state" / "focus_lost" / "execution_error". Otherwise null.
11. missing_or_wrong: array of 1-4 specific items (e.g. ["Layer Properties dialog not visible"]).
12. confidence: "high" / "medium" / "low".
13. next_action_hint: ONE concrete action by exact signature from AVAILABLE ACTIONS. Examples: hotkey(key='escape') | cli_run('pkill -f chrome; sleep 2') | impress_set_font(slide=1, shape='Title', font_name='Arial'). Vague phrases are REJECTED. MUST NOT exactly match any coord/signature listed in ACTION ROLLUP or RECENT ATTEMPTS — repeating a tried-and-failed signature regresses the loop. If you cannot name a fresh action, set category="unclear".
14. verifier_override_evidence: array of verbatim snapshot quotes that prove the underlying state IS ALREADY satisfied, ONLY when attributed_to_role="verifier". Each quote must come literally from CURRENT OBSERVATION / SCREENSHOT text / window title / CLI stdout. Empty array [] when role != "verifier" or when you cannot defend an override.
15. env_recovery_kind: ONE of "wait" | "focus" | "recovery_script" | "retry", ONLY when attributed_to_role="env". null otherwise.
16. env_recovery_params: small object with kind-specific keys, ONLY when env_recovery_kind is set. Examples:
   - kind="wait"             → {"ms": 5000}
   - kind="focus"            → {"target_widget": "<a11y label>"}
   - kind="recovery_script"  → {"command": "pkill -f gimp; sleep 2; gimp &"}
   - kind="retry"            → {"action": "<the exact action to retry>"}
   Empty object {} otherwise.

Hard rules:
- Never invent a UI state the snapshot does not contain. Every downstream claim must point to one entry in evidence_quotes.
- READ THE WINDOW TITLE LITERALLY. If the title indicates the operation is ALREADY completed (e.g. "Indexed color" already in title when goal is "set indexed mode"; file shown saved when goal is "save"), strongly prefer attributed_to_role="verifier" + populate verifier_override_evidence with the title quote. The repeated failed clicks are then the SYMPTOM, not the cause — the spec is mismatched.
- When PRIOR DIAGNOSES is non-empty: either build on the prior OR explicitly say why category is changing with new evidence. A category flip without new evidence_quote = hallucination — set category="unclear" instead.
- When ACTION ROLLUP shows ⚠ STRATEGIC CANDIDATE (same signature ≥3 tries no positive delta): strongly prefer category="strategic" AND attributed_to_role="planner" — the planner keeps prescribing the same dead-end. Choosing tactical requires citing a fresh evidence_quote.
- When memory_followed="true" but task still failed: attributed_to_role should usually be "verifier" or "env" (memory worked but stuck → spec mismatch or environment).

EXACT JSON SHAPE (start your response with `{` and end with `}` — nothing else):
{"memory_recall":{"canonical_approach":"...","memory_referenced":[{"layer":"...","id":"...","score":0.0,"why_relevant":"..."}]},"observation_summary":"...","evidence_quotes":["...","..."],"memory_critique":{"memory_followed":"true|false|partial|n/a","divergence_description":"..."},"root_cause_summary":"...","attributed_to_role":"planner|actor|verifier|env|unclear","category":"tactical|strategic|info_gap|unclear","tactical_subkind":"wrong_target|target_missing|target_blocked|blocking_state|focus_lost|execution_error|null","missing_or_wrong":["..."],"confidence":"high|medium|low","next_action_hint":"...","verifier_override_evidence":[],"env_recovery_kind":null,"env_recovery_params":{}}
"""


# ───────────────────────────────────────────────────────────────────────
# Dataclass
# ───────────────────────────────────────────────────────────────────────


_VALID_ROLES = {"planner", "actor", "verifier", "env", "unclear"}
_VALID_CATEGORIES = {"tactical", "strategic", "info_gap", "unclear"}
_VALID_SUBKINDS = {
    "wrong_target", "target_missing", "target_blocked",
    "blocking_state", "focus_lost", "execution_error",
}


_VALID_ENV_RECOVERY_KINDS = {"wait", "focus", "recovery_script", "retry"}


@dataclass
class FailureAttribution:
    # ── 1. RECALL ──
    memory_recall_canonical: str = ""
    memory_referenced: List[Dict[str, Any]] = field(default_factory=list)
    # ── 2. OBSERVE ──
    observation_summary: str = ""
    evidence_quotes: List[str] = field(default_factory=list)
    # ── 3. CRITIQUE ──
    memory_followed: str = "n/a"
    divergence_description: str = ""
    # ── 4. ATTRIBUTE ──
    root_cause_summary: str = ""
    attributed_to_role: str = "unclear"
    category: str = "unclear"
    tactical_subkind: Optional[str] = None
    missing_or_wrong: List[str] = field(default_factory=list)
    confidence: str = "low"
    # ── 5. ACT ──
    next_action_hint: str = ""
    # ── 6. ROLE-SPECIFIC ROUTING PAYLOAD ──
    # Only meaningful when role matches; empty/None otherwise.
    # The router downstream picks the right payload for the chosen route.
    verifier_override_evidence: List[str] = field(default_factory=list)
    # filled when attributed_to_role == "verifier" + the LLM judges
    # the underlying state IS already satisfied. Each entry is a
    # verbatim quote from the snapshot (window title / a11y row / CLI
    # stdout) that supports the override claim. Empty list means the
    # LLM did not produce a defensible override.
    env_recovery_kind: Optional[str] = None
    # "wait" | "focus" | "recovery_script" | "retry"
    env_recovery_params: Dict[str, Any] = field(default_factory=dict)
    # e.g. {"ms": 5000} for wait; {"target_widget": "..."} for focus;
    # {"command": "pkill -f chrome; sleep 2; chrome &"} for recovery_script

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_recall": {
                "canonical_approach": self.memory_recall_canonical,
                "memory_referenced": list(self.memory_referenced),
            },
            "observation_summary": self.observation_summary,
            "evidence_quotes": list(self.evidence_quotes),
            "memory_critique": {
                "memory_followed": self.memory_followed,
                "divergence_description": self.divergence_description,
            },
            "root_cause_summary": self.root_cause_summary,
            "attributed_to_role": self.attributed_to_role,
            "category": self.category,
            "tactical_subkind": self.tactical_subkind,
            "missing_or_wrong": list(self.missing_or_wrong),
            "confidence": self.confidence,
            "next_action_hint": self.next_action_hint,
            "verifier_override_evidence": list(
                self.verifier_override_evidence),
            "env_recovery_kind": self.env_recovery_kind,
            "env_recovery_params": dict(self.env_recovery_params),
        }


def _parse_response(raw: str) -> Optional[FailureAttribution]:
    """Extract the JSON object from the model response + validate enums.
    Returns None on hard parse failure."""
    if not raw:
        return None
    txt = raw.strip()
    # Strip code-fence if model emitted one despite instructions
    if txt.startswith("```"):
        m = re.search(r"```(?:json)?\s*\n(.*?)\n```", txt, re.DOTALL)
        if m:
            txt = m.group(1)
    # Otherwise take the first { ... } object (greedy match for braces)
    if not txt.startswith("{"):
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return None
        txt = m.group(0)
    try:
        d = json.loads(txt)
    except json.JSONDecodeError as e:
        logger.warning("failure_attribution parse error: %s; raw_head=%r",
                       e, raw[:200])
        return None
    if not isinstance(d, dict):
        return None

    recall = d.get("memory_recall") or {}
    critique = d.get("memory_critique") or {}

    role = (d.get("attributed_to_role") or "unclear").strip().lower()
    if role not in _VALID_ROLES:
        role = "unclear"
    cat = (d.get("category") or "unclear").strip().lower()
    if cat not in _VALID_CATEGORIES:
        cat = "unclear"
    sub = d.get("tactical_subkind")
    sub_norm: Optional[str] = None
    if isinstance(sub, str):
        s = sub.strip().lower()
        if s in _VALID_SUBKINDS:
            sub_norm = s
    if cat != "tactical":
        sub_norm = None
    conf = (d.get("confidence") or "low").strip().lower()
    if conf not in ("high", "medium", "low"):
        conf = "low"
    mf = (critique.get("memory_followed") or "n/a").strip().lower()
    if mf not in ("true", "false", "partial", "n/a"):
        mf = "n/a"

    # Role-specific routing payloads (Phase B).
    voe = [str(x) for x in (d.get("verifier_override_evidence") or [])]
    erk = d.get("env_recovery_kind")
    erk_norm: Optional[str] = None
    if isinstance(erk, str):
        s = erk.strip().lower()
        if s in _VALID_ENV_RECOVERY_KINDS:
            erk_norm = s
    erp = d.get("env_recovery_params")
    erp_norm: Dict[str, Any] = erp if isinstance(erp, dict) else {}

    return FailureAttribution(
        memory_recall_canonical=str(recall.get("canonical_approach") or ""),
        memory_referenced=list(recall.get("memory_referenced") or []),
        observation_summary=str(d.get("observation_summary") or ""),
        evidence_quotes=[str(q) for q in (d.get("evidence_quotes") or [])],
        memory_followed=mf,
        divergence_description=str(critique.get("divergence_description") or ""),
        root_cause_summary=str(d.get("root_cause_summary") or ""),
        attributed_to_role=role,
        category=cat,
        tactical_subkind=sub_norm,
        missing_or_wrong=[str(x) for x in (d.get("missing_or_wrong") or [])],
        confidence=conf,
        next_action_hint=str(d.get("next_action_hint") or ""),
        verifier_override_evidence=voe,
        env_recovery_kind=erk_norm,
        env_recovery_params=erp_norm,
    )


# ───────────────────────────────────────────────────────────────────────
# Memory retrieval block builder
# ───────────────────────────────────────────────────────────────────────


def _build_retrieved_memory_block(
    focus_outcome: Any,
    current_domain: Optional[str],
    *,
    k_l1: int = 3,
    k_l3a_rule: int = 4,
) -> str:
    """Pull v3 memory relevant to the current goal/subgoal and render
    as a prompt block. Returns "" when v3 indexes aren't built or the
    retriever can't load (caller will see "no relevant memory" in the
    rendered block).

    Query = focus_outcome's description + the most recent subgoal text
    available on the outcome — concatenated so MiniLM ranks for both
    the WHAT (description) and the HOW (subgoal phrasing)."""
    try:
        from mm_agents.structagent.memory.runtime.retrieval.query_multi_layer_memory \
            import _get_singleton_retriever
    except Exception:
        return ""
    try:
        retriever = _get_singleton_retriever()
    except Exception as e:
        logger.info("failure_attribution: v3 retriever unavailable: %s", e)
        return ""

    desc = str(getattr(focus_outcome, "description", "") or "")
    subgoal = ""
    for attr in ("current_subgoal", "subgoal_text", "subgoal"):
        v = getattr(focus_outcome, attr, None)
        if v:
            subgoal = str(v)
            break
    query = f"{desc}" + (f" | {subgoal}" if subgoal else "")
    if not query.strip():
        return ""

    dom = (current_domain or "").strip() or "unknown"
    try:
        ret_subgoal = retriever.retrieve_for_subgoal_transition(
            query, dom, k_l1=k_l1, k_l3a_rule=k_l3a_rule,
        )
        ret_start = retriever.retrieve_for_task_start(
            query, dom, k_l2=1, k_l3a_cluster=1,
        )
    except Exception as e:
        logger.info("failure_attribution: v3 retrieval failed: %s", e)
        return ""

    lines: List[str] = []

    def add_l1(hits):
        if not hits:
            return
        lines.append("## L1 typical_actions (per-domain) — top "
                     f"{len(hits)}:")
        for h in hits:
            p = h.payload
            lines.append(
                f"  ▸ id={p.get('action_id')} score={h.score:.3f} "
                f"approach={p.get('approach')}")
            lines.append(
                f"     subgoal: {(p.get('subgoal_pattern') or '')[:140]}")
            lines.append(
                f"     when_to_use: {(p.get('when_to_use') or '')[:200]}")
            for t in (p.get("typical_actions") or [])[:3]:
                lines.append(f"       - {str(t)[:180]}")
            for pit in (p.get("common_pitfalls") or [])[:2]:
                lines.append(
                    f"       ⚠ {(pit.get('pitfall') or '')[:160]}")
            lines.append("")

    def add_l3a_rule(hits):
        if not hits:
            return
        lines.append(f"## L3a rules — top {len(hits)}:")
        for h in hits:
            p = h.payload
            rule_text = (p.get("rule") or "")[:200]
            lines.append(f"  ▸ id={p.get('cluster_id')} score={h.score:.3f}")
            lines.append(f"     rule: {rule_text}")
            applies = (p.get("applies_when") or "")[:160]
            if applies:
                lines.append(f"     applies_when: {applies}")
            lines.append("")

    def add_l2(hits):
        if not hits:
            return
        lines.append(f"## L2 plan_template — top {len(hits)}:")
        for h in hits:
            p = h.payload
            lines.append(
                f"  ▸ id={p.get('skill_id')} score={h.score:.3f} "
                f"task_class={(p.get('task_class') or '')[:80]}")
            for step in (p.get("plan_template") or [])[:4]:
                lines.append(
                    f"     - {(step.get('subgoal') or '')[:140]}")
            lines.append("")

    def add_l3a_cluster(hits):
        if not hits:
            return
        lines.append(f"## L3a cluster summary — top {len(hits)}:")
        for h in hits:
            p = h.payload
            lines.append(
                f"  ▸ id={p.get('cluster_id')} score={h.score:.3f}")
            lines.append(
                f"     summary: {(p.get('cluster_summary') or '')[:200]}")
            for r in (p.get("common_rules") or [])[:3]:
                lines.append(f"       rule: {(r.get('rule') or '')[:160]}")
            lines.append("")

    def add_l3c(d):
        if not d:
            return
        lines.append(f"## L3c domain invariants for `{dom}`:")
        for inv in (d.get("domain_invariants") or [])[:5]:
            lines.append(
                f"  - {(inv.get('invariant') or '')[:200]}")
        for sc in (d.get("common_shortcuts") or [])[:4]:
            lines.append(
                f"  - shortcut '{sc.get('shortcut','')[:30]}': "
                f"{(sc.get('purpose') or '')[:120]}")
        lines.append("")

    add_l1(ret_subgoal.get("l1_actions") or [])
    add_l3a_rule(ret_subgoal.get("l3a_rules") or [])
    add_l2(ret_start.get("l2_plans") or [])
    add_l3a_cluster(ret_start.get("l3a_clusters") or [])
    add_l3c(ret_start.get("l3c_domain") or {})

    block = "\n".join(lines).strip()
    if not block:
        return ""
    return f"RETRIEVED MEMORY for the current goal " \
           f"(query={query[:80]!r}, domain={dom}):\n\n{block}"


# ───────────────────────────────────────────────────────────────────────
# Main entry point — drop-in shape-compatible with v1's diagnose_stuck
# ───────────────────────────────────────────────────────────────────────


def diagnose_failure(
    *,
    focus_outcome: Any,
    ledger: Any,
    observation: Dict[str, Any],
    timeline: List[Any],
    call_llm: Callable,
    model: Optional[str],
    stuck_category: Optional[str] = None,
    stuck_reason: Optional[str] = None,
    results_dir: Optional[str] = None,
    step_idx: int = 0,
    current_domain: Optional[str] = None,
    last_cli_run_output: Optional[Dict[str, Any]] = None,
) -> Optional[FailureAttribution]:
    """v2 failure-attribution analysis. Same input contract as v1's
    ``diagnose_stuck`` so the planner-side caller switch is a one-liner."""
    if focus_outcome is None:
        return None

    # ── Build snapshot sections (reuse v1 helpers) ──
    goal_section = _render_goal_section(focus_outcome)
    actions_section = _render_available_actions_section(current_domain)
    cli_section = _render_cli_run_section(last_cli_run_output)
    prior_diag_section = _render_prior_diagnoses_section(focus_outcome)
    rollup_section = _render_action_rollup_section(
        timeline or [], focus_outcome.id,
    )
    obs_section = _render_observation_section(observation or {})
    attempts_section = _render_recent_attempts_section(
        timeline or [], focus_outcome.id,
    )
    trace_section = _render_verifier_trace_section(focus_outcome)
    recovery_section = _render_recovery_trigger_section(
        stuck_category, stuck_reason,
    )

    # ── NEW: retrieved memory block (v3) ──
    memory_block = _build_retrieved_memory_block(
        focus_outcome, current_domain,
    ) or ("RETRIEVED MEMORY: <no relevant memory found OR v3 index "
          "unavailable>")

    text_pre = (
        "GOAL (the outcome currently being attempted):\n"
        f"{goal_section}\n\n"
        f"{memory_block}\n\n"
        "RECOVERY TRIGGER (which 6-rule condition fired this turn):\n"
        f"{recovery_section}\n\n"
        "PRIOR DIAGNOSES ON THIS FOCUS (last 3, oldest first — read "
        "BEFORE classifying so you don't flip-flop):\n"
        f"{prior_diag_section}\n\n"
        "ACTION ROLLUP (recent actions on this focus grouped by literal "
        "signature; ≥3 tries with no positive delta = STRATEGIC "
        "EXHAUSTION):\n"
        f"{rollup_section}\n\n"
        "AVAILABLE ACTIONS (catalogue — your next_action_hint must name "
        "one of these by its exact signature):\n"
        f"{actions_section}\n\n"
        "LAST CLI_RUN / OPEN_APP RESULT (most recent shell/script the "
        "actor executed):\n"
        f"{cli_section}\n\n"
        "CURRENT SCREENSHOT — primary visual evidence:"
    )
    text_post = (
        "CURRENT OBSERVATION (a11y signals; supplementary to the screenshot):\n"
        f"{obs_section}\n\n"
        f"RECENT ATTEMPTS (last {_MAX_RECENT_STEPS} steps):\n"
        f"{attempts_section}\n\n"
        "LAST VERIFIER TRACE on this outcome (why verify currently returns False):\n"
        f"{trace_section}\n"
    )

    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": text_pre},
    ]
    screenshot_data_url = _screenshot_data_url_from_observation(observation)
    if screenshot_data_url:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": screenshot_data_url},
        })
    user_content.append({"type": "text", "text": text_post})

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # ── Cache by (focus_id, action-digest, obs-digest) — same key as v1
    #    but in a separate slot on the focus outcome so v1 and v2 caches
    #    don't collide during A/B comparison.
    actions_digest = _digest(rollup_section + attempts_section)
    obs_digest = _digest(obs_section)
    sig = f"{focus_outcome.id}|{actions_digest}|{obs_digest}"

    cached_sig = getattr(focus_outcome, "last_failure_attribution_signature",
                         None)
    cached_dict = getattr(focus_outcome, "last_failure_attribution", None)

    diag: Optional[FailureAttribution] = None
    cache_status = "miss"

    if cached_sig == sig and cached_dict is not None:
        # Parse cached dict back into the dataclass.
        try:
            diag = FailureAttribution(
                memory_recall_canonical=(cached_dict.get(
                    "memory_recall", {}).get("canonical_approach") or ""),
                memory_referenced=list(cached_dict.get(
                    "memory_recall", {}).get("memory_referenced") or []),
                observation_summary=cached_dict.get(
                    "observation_summary") or "",
                evidence_quotes=list(cached_dict.get(
                    "evidence_quotes") or []),
                memory_followed=(cached_dict.get(
                    "memory_critique", {}).get("memory_followed") or "n/a"),
                divergence_description=(cached_dict.get(
                    "memory_critique", {}).get(
                        "divergence_description") or ""),
                root_cause_summary=cached_dict.get(
                    "root_cause_summary") or "",
                attributed_to_role=cached_dict.get(
                    "attributed_to_role") or "unclear",
                category=cached_dict.get("category") or "unclear",
                tactical_subkind=cached_dict.get("tactical_subkind"),
                missing_or_wrong=list(cached_dict.get(
                    "missing_or_wrong") or []),
                confidence=cached_dict.get("confidence") or "low",
                next_action_hint=cached_dict.get("next_action_hint") or "",
            )
            cache_status = "hit"
        except Exception:
            diag = None
            cache_status = "miss"

    raw = ""
    if diag is None:
        try:
            # call_llm signature is (payload, model) — payload is the request dict
            # (same shape as stuck_diagnosis.py's call). The old keyword call
            # call_llm(messages=..., model=...) raised TypeError on EVERY invocation,
            # silently disabling failure-attribution v2 for whole runs (the warning
            # below ate it), so the verifier_override/router never fired.
            raw = call_llm({"model": model, "messages": messages}, model) or ""
        except Exception as e:
            logger.warning("failure_attribution LLM call failed: %s", e)
            raw = ""
        diag = _parse_response(raw)

    # ── Persist prompt + response for debug (same dir naming as v1) ──
    if results_dir:
        try:
            slug = re.sub(r"[^a-z0-9]+", "_",
                          (str(focus_outcome.id) or "").lower()).strip("_")
            sub = Path(results_dir) / "failure_attribution"
            sub.mkdir(parents=True, exist_ok=True)
            tag = (stuck_category or "stuck").lower()
            fp = sub / f"step_{step_idx:04d}_{tag}_{slug}.json"
            payload: Dict[str, Any] = {
                "step_idx": step_idx,
                "focus_outcome_id": str(focus_outcome.id),
                "signature": sig,
                "cache_status": cache_status,
                "prompt_messages": _redact_image_urls(messages),
                "raw_response": raw,
                "parsed": diag.to_dict() if diag else None,
            }
            fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.info("failure_attribution debug-dump failed: %s", e)

    # ── Update focus outcome cache ──
    if diag is not None:
        try:
            focus_outcome.last_failure_attribution_signature = sig
            focus_outcome.last_failure_attribution = diag.to_dict()
        except Exception:
            pass

    return diag


def _redact_image_urls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace base64 image_url payloads with a placeholder so dumps
    stay small + diffable. Lossless for everything except the actual
    screenshot bytes (which are also dumped separately as PNG)."""
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        c = m.get("content")
        if isinstance(c, list):
            new_c = []
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    new_c.append({"type": "image_url",
                                  "image_url": {"url": "[redacted]"}})
                else:
                    new_c.append(p)
            out.append({**m, "content": new_c})
        else:
            out.append(m)
    return out


# ───────────────────────────────────────────────────────────────────────
# Prompt block renderer (consumed by planner force_replan path)
# ───────────────────────────────────────────────────────────────────────


def render_failure_attribution(
    diag: FailureAttribution, *, focus_id: str
) -> str:
    """Render the attribution as a compact block for injection into the
    next force_replan prompt. Mirrors v1's render_stuck_diagnosis shape
    so the consuming prompt template sees a familiar block."""
    lines = [
        f"[Failure attribution for {focus_id}]",
        f"  attributed_to_role: {diag.attributed_to_role}",
        f"  category: {diag.category}"
        + (f" / {diag.tactical_subkind}" if diag.tactical_subkind else ""),
        f"  confidence: {diag.confidence}",
        f"  root_cause: {diag.root_cause_summary[:400]}",
    ]
    if diag.memory_recall_canonical and diag.memory_recall_canonical != "<no memory>":
        lines.append(f"  memory_canonical: {diag.memory_recall_canonical[:300]}")
        lines.append(f"  memory_followed: {diag.memory_followed}")
        if diag.divergence_description:
            lines.append(
                f"  divergence: {diag.divergence_description[:300]}")
    if diag.missing_or_wrong:
        lines.append("  missing_or_wrong:")
        for item in diag.missing_or_wrong[:4]:
            lines.append(f"    - {str(item)[:200]}")
    if diag.evidence_quotes:
        lines.append("  key_evidence:")
        for q in diag.evidence_quotes[:3]:
            lines.append(f"    - {str(q)[:200]}")
    if diag.next_action_hint:
        lines.append(f"  next_action_hint: {diag.next_action_hint[:300]}")
    return "\n".join(lines)
