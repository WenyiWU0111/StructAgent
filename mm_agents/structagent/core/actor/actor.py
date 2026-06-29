"""Actor: turn one subgoal into one Thought/Action emission.

Decomposer LLM with inline grounding. Structured-action catalogues
(calc_*/impress_*/writer_*) are injected for LibreOffice domains. Output is
UI-TARS Thought/Action format so the shared parser handles it.

``Actor`` is mixed into ``StructAgent`` via inheritance; methods read
``self.decomposer_model`` / ``self.actor_history`` / ``self._current_domain``
/ ``self.ledger`` off the composed instance.
"""

import logging
import os
import re
from typing import List, Optional, Tuple

from mm_agents import model_endpoints as ME
import openai


_logger = logging.getLogger("desktopenv.qwen25vl_agent_planner")


class Actor:
    """Actor: decomposer LLM with inline grounding."""

    _FROZEN_MARKER_OPEN_RE_STATIC = re.compile(
        r"\{\{#\s*FROZEN\s+[a-zA-Z_][a-zA-Z0-9_]*\s*#\}\}\n?"
    )
    _FROZEN_MARKER_CLOSE_RE_STATIC = re.compile(r"\{\{#\s*END\s+FROZEN\s*#\}\}\n?")

    # Domain coding-gate: the cli_run/edit_json "coding route" of the
    # decomposer prompt is wrapped in {{# CODING_ROUTE #}} ... {{# END
    # CODING_ROUTE #}}. For GUI-only domains (and when the gate is on) the
    # whole block is stripped so the decomposer never learns those actions
    # exist; otherwise only the marker lines go. See domain.GUI_ONLY_DOMAINS.
    _CODING_ROUTE_BLOCK_RE = re.compile(
        r"\{\{#\s*CODING_ROUTE\s*#\}\}.*?\{\{#\s*END\s+CODING_ROUTE\s*#\}\}\n?",
        re.DOTALL)
    _CODING_ROUTE_MARKER_RE = re.compile(
        r"\{\{#\s*(?:END\s+)?CODING_ROUTE\s*#\}\}\n?")

    def _load_decomposer_system_template(self) -> str:
        """Decomposer template with ``{{# FROZEN ... #}}`` markers stripped
        (cached). Uses bundled ``DECOMPOSER_SYSTEM_TEMPLATE`` unless
        ``decomposer_system_path`` points at an external file."""
        if self._decomposer_system_template is not None:
            return self._decomposer_system_template
        if self.decomposer_system_path:
            with open(self.decomposer_system_path, "r", encoding="utf-8") as f:
                raw = f.read()
        else:
            from mm_agents.structagent.core.actor.prompts import (
                DECOMPOSER_SYSTEM_TEMPLATE,
            )
            raw = DECOMPOSER_SYSTEM_TEMPLATE
        raw = self._FROZEN_MARKER_OPEN_RE_STATIC.sub("", raw)
        raw = self._FROZEN_MARKER_CLOSE_RE_STATIC.sub("", raw)
        self._decomposer_system_template = raw
        return raw

    def _pa_call_actor_points(self, instruction: str, subgoal: str,
                                screenshot_bytes: bytes,
                                screen_size: Tuple[int, int]) -> str:
        """Inline-grounding actor path. Returns UI-TARS Thought/Action text
        so the downstream parser sees a consistent shape."""
        from mm_agents.structagent.core.actor.decomposer_actor import (
            call_decomposer_actor, INLINE_GROUNDING_INSTRUCTION,
        )

        dec_alias = self.decomposer_model or self.model
        if not dec_alias:
            raise RuntimeError(
                "the decomposer actor requires decomposer_model "
                "(or planner model attribute to fall back to)"
            )
        # Alias -> served-model-name map, mirroring the planner's
        # _call_llm_vllm so the decomposer also recognises OpenRouter-hosted
        # models on top of the local vLLM set. Keep in sync with _call_llm_vllm.
        dec_key = dec_alias.split("_")[-1] if "_" in dec_alias else dec_alias
        dec_model = ME.model_name(dec_key) if ME.is_known(dec_key) else dec_alias
        dec_url = ((ME.server_url(dec_key) if ME.is_known(dec_key) else None)
                   or self.decomposer_api_url
                   or getattr(self, "model_url", None))
        if not dec_url:
            raise RuntimeError(
                "decomposer needs decomposer_api_url, a known decomposer_model, "
                "or the planner's model_url")
        dec_url = dec_url if dec_url.endswith("/v1") else dec_url.rstrip("/") + "/v1"

        # Fill template context. str.replace, NOT .format(): the template has
        # literal JSON examples that .format() would read as placeholders.
        tpl = self._load_decomposer_system_template()
        completed_str = "\n".join(f"  ✓ {s}" for s in self.completed_subgoals) or "  (none)"
        remaining_str = "\n".join(f"  {i+2}. {s}"
                                    for i, s in enumerate(self.subgoal_queue)) or "  (none)"
        rendered = (tpl.replace("{subgoal}", subgoal or "")
                       .replace("{completed_subgoals}", completed_str)
                       .replace("{remaining_subgoals}", remaining_str))

        # Domain coding-gate: GUI-only domains silently lose the
        # cli_run/edit_json coding route (strip the whole {{# CODING_ROUTE #}}
        # block). Other domains keep it (strip only the marker lines). Gate
        # off -> always keep.
        from mm_agents.structagent.config import CAConfig as _CAC
        from mm_agents.structagent.domain import is_gui_only as _is_gui_only
        if _CAC.from_env().domain_coding_gate and _is_gui_only(
                getattr(self, "_current_domain", None)):
            rendered = self._CODING_ROUTE_BLOCK_RE.sub("", rendered)
        else:
            rendered = self._CODING_ROUTE_MARKER_RE.sub("", rendered)

        # Inject the structured-action catalogue for the active LibreOffice
        # domain (all doc mutations route through calc_*/impress_*/writer_*).
        dom = getattr(self, "_current_domain", None)
        if (dom or "").lower() in ("libreoffice_calc",
                                   "libreoffice_impress"):
            try:
                if (dom or "").lower() == "libreoffice_impress":
                    from mm_agents.structagent.actions.impress.actions import (
                        action_schema_block, IMPRESS_ACTION_NAMES as _AN)
                    _an_label = "IMPRESS"
                else:
                    from mm_agents.structagent.actions.calc.actions import (
                        action_schema_block, CALC_ACTION_NAMES as _AN)
                    _an_label = "CALC"
                _cab = action_schema_block()
                if _cab:
                    rendered = rendered + "\n\n" + _cab
                    _logger.info(
                        "[DomainCtx] actor prompt: domain=%s | "
                        "%s structured actions injected (%d ops: %s)",
                        dom, _an_label, len(_AN),
                        ", ".join(sorted(_AN)),
                    )
            except Exception as e:
                _logger.info(
                    "[PA-Actor(Points)] calc-action-schema inject failed: %s", e,
                )
        elif (dom or "").lower() == "libreoffice_writer":
            try:
                from mm_agents.structagent.actions.writer.actions import (
                    action_schema_block, WRITER_ACTION_NAMES as _WAN)
                _wab = action_schema_block()
                if _wab:
                    rendered = rendered + "\n\n" + _wab
                    _logger.info(
                        "[DomainCtx] actor prompt: domain=%s | "
                        "WRITER structured actions injected (%d ops: %s)",
                        dom, len(_WAN), ", ".join(sorted(_WAN)),
                    )
            except Exception as e:
                _logger.info(
                    "[PA-Actor(Points)] writer-action-schema inject "
                    "failed: %s", e)
        else:
            _logger.info(
                "[DomainCtx] actor prompt: domain=%s | NO structured "
                "actions (GUI-only context)",
                dom,
            )


        # Actor experience memory (L1 subgoal-level). Retrieves past
        # successful action recipes matching this subgoal and appends a
        # typical_actions + gotchas block to the prompt. Render returns ""
        # on a miss (safe no-op). Also dumps block+meta to
        # <results_dir>/planner_memory_debug/.
        from mm_agents.structagent.config import CAConfig
        if CAConfig.from_env().planner_experience_memory and subgoal:
            try:
                from mm_agents.structagent.memory.runtime.injection.planner_prompt_injection import (
                    render_for_actor_subgoal, dump_memory_debug,
                    format_log_summary,
                )
                l1_block, _l1_meta = render_for_actor_subgoal(
                    subgoal,
                    instruction,
                    domain=getattr(self, "_current_domain", "") or "",
                )
                if l1_block.strip():
                    rendered = rendered + "\n\n" + l1_block
                _logger.info(
                    "[PlannerMemory] actor injected (%d chars) "
                    "version=%s domain=%s subgoal=%r mode=%s n_hits=%s",
                    len(l1_block),
                    _l1_meta.get("memory_version", "v2"),
                    _l1_meta.get("domain") or "-",
                    subgoal[:60],
                    _l1_meta.get("mode"),
                    _l1_meta.get("n_total_hits",
                                 len(_l1_meta.get("hits") or [])),
                )
                _logger.info(
                    "[PlannerMemory] %s",
                    format_log_summary(_l1_meta),
                )
                _stage = f"actor_l1_step{len(self.completed_subgoals):03d}"
                dump_memory_debug(
                    getattr(self, "_results_dir", None),
                    stage_tag=_stage,
                    block=l1_block,
                    meta=_l1_meta,
                    subgoal=subgoal,
                )
            except Exception as _l1e:
                _logger.info(
                    "[PlannerMemory] L1 retrieval failed (ignored): %s",
                    _l1e,
                )

        # Failure-attribution actor recovery block: when the planner blamed
        # the stuck state on ROLE=actor + ``actor_redo``, it stashed recovery
        # guidance on ``self._pending_actor_recovery_payload``. Inject + clear
        # here so the actor sees it exactly once. Consumer is unconditional:
        # if there's a payload, use it.
        _actor_recov_payload = getattr(
            self, "_pending_actor_recovery_payload", None)
        if _actor_recov_payload:
            try:
                from mm_agents.structagent.memory.runtime.recovery.recovery_block import (
                    render_actor_recovery_block,
                )
                _recovery_block = render_actor_recovery_block(
                    _actor_recov_payload)
                rendered = rendered + "\n\n" + _recovery_block
                _logger.info(
                    "[FailureAttribution] actor recovery block injected "
                    "(%d chars) subgoal=%r — payload subgoal_id=%r "
                    "rcs=%d canonical=%d",
                    len(_recovery_block), (subgoal or "")[:60],
                    _actor_recov_payload.get("subgoal_id"),
                    len(_actor_recov_payload.get("root_cause_summary") or ""),
                    len(_actor_recov_payload.get("memory_canonical_approach") or ""),
                )
            except Exception as _recov_e:
                _logger.info(
                    "[FailureAttribution] recovery-block render failed "
                    "(ignored): %s", _recov_e)
            finally:
                # One-shot: clear so the next subgoal doesn't re-receive it.
                self._pending_actor_recovery_payload = None

        # Directive that makes the decomposer emit a 0-1000 ``point`` (plus
        # start/end/anchor_point for drag/scroll) per spatial action;
        # decomposer_actor reads these coordinates directly.
        rendered = rendered + "\n\n" + INLINE_GROUNDING_INSTRUCTION

        # Apply online debug runtime prompt patch (actor target)
        rendered = self._compose_system_prompt(rendered, role="actor")

        _logger.info("[PA-Actor] decomposer=%s (alias=%s) @ %s",
                     dec_model, dec_alias, dec_url)
        # OpenRouter needs the real key; local vLLM accepts "EMPTY".
        if "openrouter.ai" in dec_url:
            dec_api_key = os.environ.get("OPENROUTER_API_KEY") or "EMPTY"
        else:
            dec_api_key = "EMPTY"
        dec_client = openai.OpenAI(base_url=dec_url, api_key=dec_api_key)
        # Env-perceiver file inventory, passed verbatim so extract_info can't
        # invent filenames from a pattern.
        _useful_paths: Optional[List[str]] = None
        try:
            if self.ledger is not None and self.ledger.initial_context is not None:
                from mm_agents.structagent.ledger.core.probes import (
                    extract_useful_paths,
                )
                _useful_paths = extract_useful_paths(
                    self.ledger.initial_context.environment_probe
                ) or None
        except Exception:  # noqa: BLE001
            _useful_paths = None
        return call_decomposer_actor(
            instruction=instruction, subgoal=subgoal,
            screenshot_bytes=screenshot_bytes, screen_size=screen_size,
            decomposer_client=dec_client,
            decomposer_model=dec_model,
            decomposer_system_template=rendered,
            actor_history=self.actor_history,
            doc_inspect_block=getattr(self, "_doc_inspect_block", None),
            last_cli_run_output=getattr(self, "_last_cli_run_output", None),
            useful_paths=_useful_paths,
            working_memory_text=self._build_working_memory_text(),
            task_domain=self._current_domain,
        )

    def _build_working_memory_text(self) -> Optional[str]:
        """Render the per-outcome Fact view for the decomposer prompt.
        Returns None when no Facts are live so the caller skips an empty
        block."""
        try:
            from mm_agents.structagent.ledger.core.records import (
                render_working_memory,
            )
            if self.ledger is None:
                return None
            _focus = self.ledger.next_focus_outcome()
            return render_working_memory(
                self.ledger.working_memory(),
                focus_outcome_id=_focus.id if _focus is not None else None,
            )
        except Exception:
            return None

    def _pa_call_actor(self, instruction: str, subgoal: str, screenshot_bytes: bytes,
                       screen_size: Tuple[int, int]) -> str:
        """One subgoal -> one Thought/Action emission via the inline-grounding
        decomposer."""
        return self._pa_call_actor_points(instruction, subgoal,
                                          screenshot_bytes, screen_size)
