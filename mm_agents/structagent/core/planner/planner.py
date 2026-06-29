"""Planner: build prompts, call LLM, parse the planner contract.

Methods folded into ``StructAgent`` via inheritance. They access
``self.model`` / ``self.actor_history`` / ``self.subgoal_queue`` /
``self.timeline`` / ``self.ledger`` / ``self._notebook`` on the composed
instance.

Includes:
  * instruction augmentation (resolve dates, append [Task Requirements])
  * initial-plan / subgoal-check / progress-check / force-replan prompt builders
  * the planner LLM call orchestrator with retry + re-author logic
  * response parsers (planner-contract XML + plan-into-subgoals)
"""

import base64
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from mm_agents.structagent.utils.image import process_image
from mm_agents.structagent.utils.a11y import (
    extract_active_url_from_a11y as _extract_active_url_from_a11y,
    linearize_obs_a11y as _linearize_obs_a11y,
)


logger = logging.getLogger("desktopenv.qwen25vl_agent_planner")


# ---------------------------------------------------------------------------
# LAYERED PROMPT ARCHITECTURE
#
# The progress_check / force_replan planner prompt is assembled as 5
# topically scoped LAYERS instead of a flat list of overlapping blocks.
# Each FACT has ONE owner LAYER; sibling LAYERs reference the owner
# rather than restating. When two LAYERs would disagree (e.g. Required
# Outcomes says "verified" but the subgoal queue is non-empty), the
# CONFLICT RESOLUTION table in LAYER 5 names the winner.
#
#   LAYER 1  Task constraint        — instruction + initial context +
#                                     [Required Outcomes] (what success
#                                     looks like)
#   LAYER 2  Strategy history       — past abandoned strategies +
#                                     [Failed paths] / dead-end patterns
#                                     (what NOT to repeat)
#   LAYER 3  Subgoal state          — current plan steps + which subgoal
#                                     is active + queue + abandoned +
#                                     working-memory Facts (where you
#                                     are in the plan)
#   LAYER 4  This turn's evidence   — K screenshots + perceiver SCREEN
#                                     SNAPSHOT (or a11y delta) + recent
#                                     actor actions + last cli_run +
#                                     stuck signals
#   LAYER 5  Decision contract      — assessment + decision rubric +
#                                     output format + CONFLICT
#                                     RESOLUTION table
#
# The LAYER guide below is concatenated onto the SYSTEM prompt so the
# planner reads the structure before any user message arrives.
# ---------------------------------------------------------------------------

_LAYERED_PROMPT_GUIDE = """\
The user messages below are organized into 5 LAYERS. Read them top-down.

  [LAYER 1 — Task constraint]      what success looks like
  [LAYER 2 — Strategy history]     past strategies + dead-ends to avoid
  [LAYER 3 — Subgoal state]        where you are in the plan right now
  [LAYER 4 — Turn evidence]        this turn's screen + perceiver + actor
  [LAYER 5 — Decision contract]    your output rubric + conflict rules

AUTHORITY MAP — when a question arises, look at the named layer:
  • "Is the TASK done?"                 → LAYER 1 [Required Outcomes],
                                          cross-checked with LAYER 3
                                          queue per LAYER 5 rule (6).
  • "Is the current PLAN on track?"     → LAYER 3 plan structure
                                          (✓ done / ▶ active / · pending)
                                          + abandoned list.
  • "Did the CURRENT subgoal just       → LAYER 4 perceiver SCREEN
     complete?"                            SNAPSHOT subgoal_check.
  • "What is the canonical current      → LAYER 3 ▶ active step.
     subgoal text + expected post-          (LAYER 4 SCREEN SNAPSHOT may
     state?"                                quote them as the scope of
                                            its verdict — that quote is
                                            a scope label, not the
                                            canonical fact.)
  • "What should I NOT try again?"      → LAYER 2 strategies + LAYER 1
                                          dead-end patterns.

When two LAYERS disagree (e.g. LAYER 1 says N/N verified but LAYER 3
queue is non-empty), the CONFLICT RESOLUTION table at the bottom of
LAYER 5 names which wins. Read that table before emitting <decision>."""


# Conflict-resolution table appended to LAYER 5. Kept as a constant so
# additions ride with code review rather than getting embedded in a
# prompt-building f-string with the rest of the contract.
_CONFLICT_RESOLUTION_TABLE = """\
CONFLICT RESOLUTION — when two LAYERS disagree, the rule below picks
the winner. Apply these BEFORE emitting <decision>.

  (a) LAYER 1 [Required Outcomes] says "N/N verified" vs.
      LAYER 3 "Remaining subgoals" is non-empty
        → trust LAYER 3. The verify spec for the "verified" outcome
          was too weak. DONE is FORBIDDEN; emit REPLAN whose first
          step targets whatever the unfinished subgoal in LAYER 3 was
          actually trying to produce. This is LAYER 5 DONE rule (6).

  (b) LAYER 4 perceiver "Overall: YES" on the current subgoal vs.
      LAYER 4 stuck signal "N turns no verifiable progress"
        → trust the stuck signal. The perceiver verdict matched a
          NON-UNIQUE feature (e.g. a nav link that exists on every
          sub-page). Treat <done> as NO and emit REPLAN.

  (c) LAYER 2 past abandoned strategy text ≈ your proposed new
      <strategy> text
        → forbidden unless you explicitly declare in <observations>
          what is structurally different (different element,
          different access path, different command form).

  (d) LAYER 4 perceiver "Coverage: complete" vs. your <observations>
      saying scroll is needed to see the target
        → trust the scroll requirement. Perceiver coverage is a
          heuristic on the rendered viewport, not a proof that
          off-screen content is irrelevant."""


class Planner:
    """Planner role: prompt + LLM call + parse."""

    # --- planner prompt builders ---

    def _pa_augment_instruction(self, instruction: str) -> str:
        """Resolve relative dates and append a [Task Requirements] block.

        Called once per task at the top of `_pa_predict` — before ledger
        init — so both the ledger initializer and the initial planner
        prompt observe the same enriched instruction (e.g. a task that
        said "New York" gains a `destination: NYC` line the ledger can
        cite as the evidence-bearing target).
        """
        from mm_agents.structagent.utils.helpers import resolve_relative_dates, extract_task_fields

        def _call_llm(messages, max_tokens=200):
            return self.call_llm(
                {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                 "top_p": 0.9, "temperature": 0.0},
                self.model,
            )
        instruction = resolve_relative_dates(instruction, call_llm_fn=_call_llm)
        task_fields = extract_task_fields(instruction, _call_llm)
        if task_fields:
            instruction = instruction + task_fields
        if logger:
            logger.info("[PA-Planner] instruction: %s", instruction)
        return instruction

    def _pa_build_initial_plan_prompt(
        self, instruction: str,
        retrieved_knowledge: Optional[str] = None,
    ) -> str:
        # Instruction is already augmented (date resolution + task_fields)
        # by `_pa_augment_instruction`, called upstream in `_pa_predict`.
        self.instruction = instruction
        from mm_agents.structagent.core.planner.prompts import _plan_format_block, _plan_quality_rules
        teacher_demo_block = self._init_plan_teacher_demo_block(instruction)
        retrieved_block = self._init_plan_retrieved_block(retrieved_knowledge)
        domain_block = self._init_plan_domain_block()
        planner_memory_block = self._init_plan_memory_block(instruction)
        rule_3c = self._init_plan_rule_3c()

        # Synthesized-tool hint: same env-gating as the decomposer
        # injection in actor.py. We split the hint into two pieces:
        #
        #   - synth_priority_block: a HIGH-PRIORITY block rendered ABOVE
        #     the retrieved-experience blocks (L2 plan templates + L1
        #     subgoal memory). Past trajectories were all GUI flows, so
        #     when a synth tool exists the retrieved memory anchors the
        #     planner on the multi-step click recipe — we need to break
        #     that anchor BEFORE the planner reads the GUI plans.
        #
        #   - rule_3d: a one-liner stub inside the rule list, just so
        #     the actor's-tools enumeration stays internally consistent
        #     ("(a) GUI / (b) cli_run / (c) <domain>_* / (d) synth").
        #
        # Implemented per [[feedback-no-task-specifics-in-shared-prompts]]:
        # only abstract tool names + abstract when_to_use go in here —
        # no per-task literal values that would leak the eval answer.
        synth_priority_block = ""
        rule_3d = ""

        prompt = f"""You are a task planning assistant. Given the current task and the current screen, generate a concise and actionable step-by-step plan.

Current Task:
{instruction}
{synth_priority_block}
{retrieved_block}{teacher_demo_block}{domain_block}{planner_memory_block}
Based on the current task and the screenshot above, generate a step-by-step plan that:
1. Is concise and actionable, no more than 5-10 steps.
2. Breaks down the task into clear, sequential steps.
3. Each step is executable by one of the actor's tools:
   (a) GUI actions (click / type / hotkey / scroll) — default for most tasks;
   (b) cli_run — one-shot bash on the VM (sed, gsettings, dconf, pip, file copy / chmod, launch a GUI app);
{rule_3c}
{rule_3d}
4. Trust what you see on screen and adapt accordingly when UI differs from your expectations.
5. Remember to open needed applications and tabs before at the beginning of the plan!

Additional strict requirements for your plan:
- You MUST use EXACT DATE in your plan, you can use the format {{Year}}-{{Month0D}}-{{Day0D}} for the date.

{_plan_quality_rules(domain=getattr(self, "_current_domain", None))}

Output format — use this exact XML. Every <step> MUST have BOTH <text> and <expected_post_state>:

<plan>
{_plan_format_block(domain=getattr(self, "_current_domain", None))}
...
</plan>"""
        # Text-answer tasks (WebVoyager / Mind2Web / MMInA): the
        # ONLY verifiable artifact is the agent's final textual answer
        # plus what's on screen. The planner must end its plan with a
        # finished(answer="...") sentinel so the runner can score.
        if self._is_text_answer_task or self._is_mmina_task:
            prompt += """

IMPORTANT: This is a web-based task. Your plan MUST end with a final
step that emits the sentinel:

  finished(answer="<your synthesised answer>")

Rules for the answer text:
  - Concise but informative (1-2 sentences max). It is the textual
    proof that the task was completed (e.g. the course name + provider,
    the price you found, a one-line summary of what was set).
  - If the task only asks for an action (no question), emit a brief
    confirmation, e.g. finished(answer="Tab closed and bookmark added").
  - For QA tasks with a short factual answer (e.g. a Wikipedia
    lookup), emit just the value: finished(answer="42") or
    finished(answer="Albert Einstein").
  - Do NOT call finished(answer=...) until the screen shows the task
    actually succeeded — premature emission terminates the run.
  - Output finished(answer=\"...\") AS PART OF YOUR Thought/Action turn,
    not in a separate channel.

CROSS-SITE NAVIGATION (web tasks only):
The actor has a dedicated ``navigate(url='https://...')`` action that
drives the browser to a URL in ONE reliable step via Chrome DevTools —
no address-bar click + type required, no risk of the vision model
mis-grounding the address bar. When your plan needs the agent to jump
to a specific known URL (Wikipedia article, Trip.com, YouTube,
Eventbrite, …), word the subgoal as

  "Navigate to https://www.wikipedia.org/wiki/Foo"

rather than

  "Click the address bar and type 'wikipedia.org/wiki/Foo' then Enter"

The decomposer will translate the former into a single
``navigate(url=...)`` call. Reserve the click-and-type form ONLY for
in-page links / search fields where there's no canonical URL to jump
to. Multi-hop tasks (MMInA compare / multi567 / multipro / normal)
typically need 2-4 navigate steps — plan them explicitly so the
decomposer doesn't waste turns trying to click around."""
        return prompt

    def _init_plan_teacher_demo_block(self, instruction):
        """Teacher-demo injection block.

        This was an offline-training-only feature (few-shot teacher demos
        mined from rollouts) and is not part of the released agent. It is
        kept as a no-op so the planner prompt template stays stable.
        """
        return ""

    def _init_plan_retrieved_block(self, retrieved_knowledge):
        """Render the optional retrieved-knowledge (Google-search) block."""
        retrieved_block = ""
        if retrieved_knowledge and retrieved_knowledge.strip():
            retrieved_block = (
                "\nRETRIEVED INFORMATION (from a Google search by another "
                "agent — may be useful, may be partly outdated; verify "
                "each step against the screenshot before committing to "
                "it; if a retrieved step does not match what's actually "
                "visible, ignore it):\n"
                f"{retrieved_knowledge.strip()}\n"
            )
        return retrieved_block

    def _init_plan_domain_block(self):
        """Per-domain knowledge + data-layout scaffold + live doc structure."""
        # Domain-specific knowledge (injected ONLY when current task's
        # domain has a knowledge file under mm_agents/domain_knowledge/).
        # No bleed across domains. Source: <domain>.md — high-level
        # framing of the whole domain, always injected when the file
        # exists. (Per-operation UNO snippet retrieval was retired —
        # all three LibreOffice domains now route through the
        # structured calc_* / impress_* / writer_* action catalogues,
        # whose schema rides in the action-schema block instead.)
        domain_block = ""
        try:
            from mm_agents.structagent.domain import get_domain_knowledge
            cur_domain = getattr(self, "_current_domain", None)
            parts = []
            if self._is_multi_app():
                # Multi-app: the INITIAL plan covers the whole cross-app
                # task. Tell the planner it is multi-app + which apps,
                # and give every related app's knowledge (each headed).
                # Per-step prompts later narrow to the current app.
                parts.append(
                    "\nMULTI-APP TASK — this task spans these "
                    f"applications: {self._related_apps}. Plan it as "
                    "per-app phases: contiguous segments of work inside "
                    "ONE app, in execution order; keep each <step> "
                    "within a single app.\n"
                    "REQUIRED: every <step> MUST carry an ``app`` "
                    "attribute naming the application that step runs "
                    "in — written EXACTLY as one of "
                    f"{self._related_apps}. Format: "
                    "`<step app=\"<application>\">`. This drives the "
                    "agent's per-step context (which app's actions, "
                    "domain knowledge and UNO snippets are loaded), so "
                    "an accurate tag is essential — tag it with the app "
                    "the step physically operates in, e.g. a step that "
                    "copies data from a spreadsheet is "
                    "`app=\"libreoffice_calc\"` even if the data is "
                    "ultimately destined for a document.\n"
                )
                for _dk_app in (self._related_apps or []):
                    _dk = get_domain_knowledge(_dk_app)
                    if _dk:
                        parts.append(
                            f"\n### Domain-specific knowledge for "
                            f"`{_dk_app}` ###\n{_dk}\n"
                        )
            else:
                dk = get_domain_knowledge(cur_domain)
                if dk:
                    parts.append(
                        f"\nDomain-specific knowledge for `{cur_domain}` "
                        f"(applies only to this task's domain — use this "
                        f"when planning):\n{dk}\n"
                    )
            # Phase B: structured Q1-Q4 data_layout reasoning slot for
            # Calc. Forces the planner to answer row-semantics / task-
            # pattern / output-target / lookup-spec BEFORE writing
            # <strategy> / <step>s, so the subgoal text speaks the
            # calc_* vocabulary (e.g. "Use calc_fill_column on F10:F23
            # with template ...") instead of GUI shape ("click F10,
            # type formula, drag fill handle"). The actor's translation
            # is then a one-shot JSON construction.
            # MULTI-APP: the initial plan covers the WHOLE cross-app
            # task, but ``cur_domain`` is only related_apps[0]. Gating
            # the calc/impress data_layout scaffold on
            # ``cur_domain == "libreoffice_calc"`` therefore SKIPPED it
            # whenever the task's first app isn't calc — so the planner
            # decomposed the Calc phase into GUI-shape subgoals ("click
            # cell A1") instead of calc_* vocabulary. Inject the
            # scaffold for EVERY office app the task spans. Single-app:
            # ``_dl_apps == {cur_domain}`` → byte-identical to before.
            _dl_apps = {
                (a or "").lower()
                for a in (self._related_apps if self._is_multi_app()
                          else [cur_domain])
            }
            if "libreoffice_calc" in _dl_apps:
                try:
                    from mm_agents.structagent.actions.calc.actions import data_layout_planner_block
                    parts.append("\n" + data_layout_planner_block() + "\n")
                except Exception as e:
                    if logger:
                        logger.info("[PA-Planner] data_layout_planner_block "
                                    "load failed: %s", e)
            if "libreoffice_impress" in _dl_apps:
                try:
                    from mm_agents.structagent.actions.impress.actions import (
                        data_layout_planner_block as
                        impress_data_layout_planner_block,
                    )
                    parts.append("\n" + impress_data_layout_planner_block() + "\n")
                except Exception as e:
                    if logger:
                        logger.info("[PA-Planner] impress data_layout_planner_block "
                                    "load failed: %s", e)
            # DocPerceiver block — the framework-side UNO probe's
            # rendered structure (paragraph list with style/font/etc +
            # tables + view-cursor). Refreshed at task start and after
            # every structured-action mutation. Lets the planner pick concrete
            # paragraph indices for ops like text_to_table / line_spacing
            # instead of having the actor guess from a screenshot.
            doc_block = getattr(self, "_doc_inspect_block", None)
            if doc_block:
                parts.append(
                    "\nLive document structure (auto-refreshed; use this "
                    "to pick concrete paragraph indices and avoid having "
                    "the actor count paragraphs from a screenshot):\n"
                    + doc_block + "\n"
                )
            domain_block = "".join(parts)
            if logger:
                _md_apps = (self._related_apps if self._is_multi_app()
                            else [cur_domain])
                _md_injected = [a for a in (_md_apps or [])
                                if get_domain_knowledge(a)]
                logger.info(
                    "[DomainCtx] initial-plan prompt: current_domain=%s "
                    "multi_app=%s | domain_knowledge(.md)=%s | "
                    "data_layout=%s | doc_structure=%s",
                    cur_domain, self._is_multi_app(), _md_injected,
                    "yes" if (cur_domain or "").lower() in (
                        "libreoffice_calc", "libreoffice_impress") else "no",
                    "yes" if getattr(self, "_doc_inspect_block", None)
                    else "no",
                )
        except Exception as e:
            if logger:
                logger.info("[PA-Planner] domain_knowledge load failed: %s", e)
        return domain_block

    def _init_plan_memory_block(self, instruction):
        """L2 planner-experience memory block (ENABLE_PLANNER_EXPERIENCE_MEMORY)."""
        # Planner experience memory (L2 task-level skills) — gated on
        # ENABLE_PLANNER_EXPERIENCE_MEMORY. Retrieves past successful
        # plan templates whose when_to_use matches this task, renders a
        # block to splice into the initial-plan prompt. Render layer
        # returns ("", meta_with_OOD_mode) on miss → safe no-op.
        # Also dumps block+meta to <results_dir>/planner_memory_debug/
        # for post-hoc forensics.
        planner_memory_block = ""
        from mm_agents.structagent.config import CAConfig
        if CAConfig.from_env().planner_experience_memory:
            try:
                from mm_agents.structagent.memory.runtime.injection.planner_prompt_injection import (
                    render_for_planner_initial, dump_memory_debug,
                    format_log_summary,
                )
                planner_memory_block, _pmem_meta = render_for_planner_initial(
                    instruction,
                    domain=getattr(self, "_current_domain", "") or "",
                )
                if logger:
                    logger.info(
                        "[PlannerMemory] initial-plan injected (%d chars) "
                        "version=%s domain=%s mode=%s n_hits=%s",
                        len(planner_memory_block),
                        _pmem_meta.get("memory_version", "v2"),
                        _pmem_meta.get("domain") or "-",
                        _pmem_meta.get("mode"),
                        _pmem_meta.get("n_total_hits",
                                       len(_pmem_meta.get("hits") or [])),
                    )
                    logger.info(
                        "[PlannerMemory] %s",
                        format_log_summary(_pmem_meta),
                    )
                dump_memory_debug(
                    getattr(self, "_results_dir", None),
                    stage_tag="init_l2",
                    block=planner_memory_block,
                    meta=_pmem_meta,
                )
            except Exception as _pmem_e:
                if logger:
                    logger.info(
                        "[PlannerMemory] L2 retrieval failed (ignored): %s",
                        _pmem_e,
                    )
                planner_memory_block = ""
        return planner_memory_block

    def _init_plan_rule_3c(self):
        """Domain-aware canonical structured-action rule (calc_*/impress_*/writer_*)."""
        # Rule-3(c) is domain-aware: each LibreOffice domain gets the
        # canonical-path bullet for its own structured-action set
        # (calc_* / impress_* / writer_*) so the planner's subgoals
        # speak that vocabulary instead of GUI shape.
        _RULE_3C_CALC = (
            "   (c) calc_* JSON action — the CANONICAL path for "
            "libreoffice_calc spreadsheet edits. Full action schema "
            "is in the Domain-specific knowledge block + the "
            "DATA_LAYOUT block above. Each <step>'s <text> SHOULD "
            "name the action + kwargs (e.g. \"Use calc_fill_column "
            "on Sheet1 col F rows 10..23 with formula_template "
            "``=VLOOKUP(E{r};$D$2:$E$7;2;1)``\"). The actor builds "
            "the JSON deterministically — you never plan Python "
            "source."
        )
        _RULE_3C_IMPRESS = (
            "   (c) impress_* JSON action — the CANONICAL path for "
            "libreoffice_impress slide edits (text / font / shape / "
            "image / chart / table / background / notes / transition "
            "/ export / app setting). Full action schema is in the "
            "Domain-specific knowledge block + the DATA_LAYOUT "
            "block above. Each <step>'s <text> SHOULD name the "
            "action + kwargs (e.g. \"Use impress_set_text_color with "
            "slide=1, placeholder='title', color='#FF0000'\"). The "
            "actor builds the JSON deterministically — you never "
            "plan Python source. Prefer impress_* over GUI clicks "
            "whenever a structured action covers the operation."
        )
        _RULE_3C_WRITER = (
            "   (c) writer_* JSON action — the CANONICAL path for "
            "libreoffice_writer document edits (paragraph / character "
            "format, structure, content, find-replace, image, export). "
            "Full action schema is in the Domain-specific knowledge "
            "block above. Each <step>'s <text> SHOULD name the action "
            "+ kwargs (e.g. \"Use writer_alignment with "
            "alignment='center', target_indices=[0]\"). The actor "
            "builds the JSON deterministically — you never plan Python "
            "source. Prefer writer_* over GUI clicks whenever a "
            "structured action covers the operation."
        )
        _cur_dom_lc = (getattr(self, "_current_domain", None) or "").lower()
        # MULTI-APP: the initial plan covers the whole cross-app task,
        # so the planner needs the canonical-path rule for EVERY office
        # app it spans — not just related_apps[0]. Gating on a single
        # _current_domain left the planner planning, say, the Calc
        # phase with only the Writer rule → GUI-shape subgoals.
        if self._is_multi_app():
            _apps = {(a or "").lower()
                     for a in (self._related_apps or [])}
            _frags = []
            if "libreoffice_calc" in _apps:
                _frags.append(_RULE_3C_CALC)
            if "libreoffice_impress" in _apps:
                _frags.append(_RULE_3C_IMPRESS)
            if "libreoffice_writer" in _apps:
                _frags.append(_RULE_3C_WRITER)
            rule_3c = "\n".join(_frags)  # empty when no LibreOffice app
        elif _cur_dom_lc == "libreoffice_calc":
            rule_3c = _RULE_3C_CALC
        elif _cur_dom_lc == "libreoffice_impress":
            rule_3c = _RULE_3C_IMPRESS
        elif _cur_dom_lc == "libreoffice_writer":
            rule_3c = _RULE_3C_WRITER
        else:
            # chrome / os / vlc / gimp / thunderbird / vs_code etc. —
            # no LibreOffice structured-action set is available; the
            # planner stays on the (a) GUI + (b) cli_run menu.
            rule_3c = ""
        return rule_3c


    def _pa_build_subgoal_check_prompt(self, instruction: str,
                                       obs: Optional[dict] = None,
                                       env=None,
                                       force_replan: bool = False) -> str:
        """Dispatch to situation-specific prompt builder in planner_prompts.py.

        ``obs`` and ``env`` are threaded down so stuck retrieval can pull
        the actual error evidence when ``force_replan`` is set.

        ``force_replan``: passed by ``_pa_call_planner`` when the upfront
        mode decision selected ``mode="force_replan"``. The reason that
        triggered the mode is read from ``self._force_replan_reason``
        (staged by ``_decide_planner_mode_and_reason`` before this is
        called) and surfaced in <stuck_reason>."""
        from mm_agents.structagent.core.planner.prompts import (
            build_prompt_actor_exhausted,
            build_prompt_all_subgoals_done, build_prompt_subgoal_done,
            build_prompt_progress_check,
        )

        common = dict(
            instruction=instruction,
            current_plan=self.current_plan,
            completed_subgoals=self.completed_subgoals,
            current_subgoal=self.current_subgoal,
            current_subgoal_step=self.current_subgoal_step,
            subgoal_queue=self.subgoal_queue,
            # Domain steers the rule-#4 action menu (which structured
            # action set — calc_* / impress_* / writer_*) in
            # ``_plan_quality_rules``. Without this, situational REPLAN
            # prompts can't name the right structured-action vocabulary
            # and regress the subgoals to GUI shape.
            domain=getattr(self, "_current_domain", None),
        )

        # DocPerceiver block — prepend to whatever prompt the dispatcher
        # returns, so initial / progress-check / force-replan / actor-
        # exhausted etc. ALL see the live document structure. Mirrors
        # the injection in _pa_build_initial_plan_prompt at line ~819.
        doc_block_prefix = ""
        doc_block = getattr(self, "_doc_inspect_block", None)
        if doc_block:
            doc_block_prefix = (
                "\nLive document structure (auto-refreshed; use this to "
                "pick concrete paragraph indices and avoid having the "
                "actor count paragraphs from a screenshot):\n"
                + doc_block + "\n\n"
            )

        # Multi-app per-step: inject ALL related_apps' knowledge.
        #
        # [FIX multi_app_per_turn_full_dk] Previously narrowed to just
        # _current_domain. That dropped sibling-app idioms when the
        # current phase needed a feature owned by another related app
        # (e.g. e135df7c — chrome/thunderbird-tagged phase needed
        # ``calc_export_file`` HTML export, but per-turn loaded only
        # thunderbird.md → planner went down the GUI File→Export rabbit
        # hole). The initial plan already loads everything; per-turn
        # should match. Single-app tasks unchanged. Roll back by
        # restoring ``_dk = get_domain_knowledge(_ps_dom)`` (single
        # current domain) — both branches are kept side by side here.
        if self._is_multi_app():
            try:
                from mm_agents.structagent.domain import get_domain_knowledge
                _ps_dom = getattr(self, "_current_domain", None)
                _dk_parts = []
                _ps_dom_lower = (_ps_dom or "").lower().strip()
                _related = [a for a in (self._related_apps or [])
                            if a and str(a).strip()]
                _seen = set()
                # current_domain first (so planner reads "in-context"
                # knowledge before sibling-app idioms)
                _order = ([_ps_dom_lower] if _ps_dom_lower else []) + [
                    str(a).lower().strip() for a in _related
                    if str(a).lower().strip() != _ps_dom_lower]
                for _dk_app in _order:
                    if not _dk_app or _dk_app in _seen:
                        continue
                    _seen.add(_dk_app)
                    _block = get_domain_knowledge(_dk_app)
                    if _block:
                        _dk_parts.append((_dk_app, _block))
                _dk = "\n\n".join(
                    f"### Domain-specific knowledge for `{app}` ###\n{body}"
                    for app, body in _dk_parts
                ) if _dk_parts else ""
                doc_block_prefix = (
                    "\nMULTI-APP TASK — applications: "
                    f"{self._related_apps}. REQUIRED: every <step> in "
                    "your plan MUST carry an ``app`` attribute naming "
                    "the application that step runs in — written "
                    f"EXACTLY as one of {self._related_apps}. Format: "
                    "`<step app=\"<application>\">`. Tag it with the "
                    "app the step physically operates in.\n\n"
                    + doc_block_prefix
                )
                if _dk:
                    # [FIX multi_app_per_turn_full_dk] _dk now already
                    # contains its own per-app ``### Domain-specific
                    # knowledge for `<app>` ###`` headers (one section
                    # per related app), so no outer wrapper needed.
                    # Roll back: replace this block with the old
                    # single-section "for the app you are currently in"
                    # wrapper if reverting the per-turn full-DK fix.
                    doc_block_prefix = (
                        f"\n{_dk}\n\nThe `{self._current_domain}` "
                        "section is the app the current phase operates "
                        "in; the other sections describe sibling apps "
                        "your plan may need to reach into (e.g. produce "
                        "an artifact in app A to feed phase B in app "
                        "B).\n\n" + doc_block_prefix
                    )
                # PHASE BOARD — per-app phase decomposition + the
                # cross-app handoff. Prepended FIRST so it frames the
                # whole prompt: which phase the planner is in, what the
                # upstream phase actually produced, plan only the active
                # phase. None on single-app or init-failure → skipped.
                _pb = self._phase_board()
                if _pb is not None:
                    try:
                        _pb_block = _pb.render_block()
                    except Exception:
                        _pb_block = ""
                    if _pb_block:
                        doc_block_prefix = (
                            "\n" + _pb_block + "\n\nPlan ONLY the steps "
                            "for the [active] phase above — do NOT plan "
                            "ahead into [pending] phases (a later phase "
                            "is planned when it becomes active). Every "
                            "<step> stays within the active phase's "
                            "app. The active phase's `receives:` line "
                            "is the concrete artifact the upstream "
                            "phase produced — rely on it, do not "
                            "re-derive it.\n\n"
                            + doc_block_prefix
                        )
                # R3 — domain mismatch notice. The OS-focused window
                # disagrees with the planner's intended domain. Tell
                # the planner explicitly so it can either issue an
                # ``open_app(<intended>)`` step or adapt to work in the
                # currently-focused app. Surfaced for >0 mismatch
                # turns; after 5 the agent auto-demotes intended so
                # this block stops appearing.
                if (self._is_multi_app()
                        and self._physical_domain is not None
                        and self._current_domain is not None
                        and self._physical_domain != self._current_domain):
                    _mm_block = (
                        f"⚠ WINDOW FOCUS MISMATCH\n"
                        f"   intended (per your last <step app=...>): "
                        f"{self._current_domain}\n"
                        f"   physical (OS focus right now): "
                        f"{self._physical_domain}\n"
                        f"   The framework already attempted auto-focus "
                        f"{self._domain_mismatch_steps}× and it hasn't taken. "
                        f"Either (a) emit a step that explicitly switches "
                        f"window (e.g. open_app('{self._current_domain}') "
                        f"or click the window in the dock), OR (b) replan "
                        f"to work within `{self._physical_domain}` if that "
                        f"is acceptable for the active phase.\n"
                    )
                    doc_block_prefix = "\n" + _mm_block + "\n" + doc_block_prefix
                if logger:
                    _loaded_names = [a for a, _ in _dk_parts]
                    logger.info(
                        "[DomainCtx] per-step planner prompt: "
                        "current_domain=%s | domain_knowledge(.md)=%s",
                        self._current_domain,
                        ("%s (%dch total)" % (_loaded_names, len(_dk)))
                        if _dk else "no",
                    )
            except Exception as e:
                if logger:
                    logger.info("[PA-Planner-Situational] multi-app "
                                "domain_knowledge load failed: %s", e)

        # Phase B (situational planners): same data_layout reasoning
        # slot the initial-plan prompt gets. Without this, REPLAN-class
        # prompts (force_replan / progress_check / actor_exhausted) drop
        # the Q1-Q4 chain and regress to GUI-shape <step>s — which
        # defeats the calc_* migration on the very subgoal the actor
        # was stuck on. Prepended after the doc_block so the planner
        # reads structure first, then the answer template.
        data_layout_prefix = ""
        _cur_dom = (getattr(self, "_current_domain", None) or "").lower()
        if _cur_dom == "libreoffice_calc":
            try:
                from mm_agents.structagent.actions.calc.actions import data_layout_planner_block
                data_layout_prefix = data_layout_planner_block() + "\n\n"
            except Exception as e:
                if logger:
                    logger.info("[PA-Planner-Situational] "
                                "data_layout_planner_block load failed: %s", e)
        elif _cur_dom == "libreoffice_impress":
            try:
                from mm_agents.structagent.actions.impress.actions import (
                    data_layout_planner_block as
                    impress_data_layout_planner_block,
                )
                data_layout_prefix = (
                    impress_data_layout_planner_block() + "\n\n")
            except Exception as e:
                if logger:
                    logger.info("[PA-Planner-Situational] impress "
                                "data_layout_planner_block load failed: %s", e)

        if force_replan:
            return self._build_subgoal_check_force_replan(
                common, doc_block_prefix, data_layout_prefix, obs, env,
                instruction)
        elif self.actor_exhausted_for_planner:
            actor_actions = list(self.actions[-6:]) if self.actions else []
            return doc_block_prefix + data_layout_prefix + build_prompt_actor_exhausted(**common, actor_actions=actor_actions)
        elif not self.subgoal_queue:
            logger.info("[PA-Planner] All subgoals completed: %s", self.subgoal_queue)
            return doc_block_prefix + data_layout_prefix + build_prompt_all_subgoals_done(**common)
        elif self.actor_subgoal_done_for_planner:
            return doc_block_prefix + data_layout_prefix + build_prompt_subgoal_done(**common)
        else:
            return doc_block_prefix + data_layout_prefix + build_prompt_progress_check(**common)

    def _build_subgoal_check_force_replan(self, common, doc_block_prefix,
                                          data_layout_prefix, obs, env,
                                          instruction):
        """Build the force_replan situational prompt: failure-attribution
        routing, verifier feedback, stuck diagnosis and L1/L2 replan memory,
        then delegate to build_prompt_force_replan. Extracted verbatim from
        the force_replan branch of _pa_build_subgoal_check_prompt."""
        from mm_agents.structagent.core.planner.prompts import build_prompt_force_replan
        # Rich failure context for the stuck-mode REPLAN. These go into
        # <stuck_analysis> so the planner reasons over concrete
        # evidence instead of re-inventing the same-theme plan.
        recent_actions = list(self.actions[-6:]) if self.actions else []
        abandoned = list(self.abandoned_subgoals[-6:]) if getattr(self, "abandoned_subgoals", None) else []
        stuck_reason = getattr(self, "_force_replan_reason", None) \
            or "actor reported IMPOSSIBLE on the current subgoal"
        stuck_category = getattr(self, "_force_replan_category", None) \
            or "actor_failure"
        # consume so the next turn's dispatcher doesn't stale-read it
        self._force_replan_reason = None
        self._force_replan_category = None
        # A8 debug: stash for the planner-call HTML dump downstream.
        # _pa_call_planner reads (and clears) these around its LLM
        # invocation so the HTML pages can show which recovery rule
        # triggered this force_replan.
        self._dump_stuck_category = stuck_category
        self._dump_stuck_reason = stuck_reason

        # External-knowledge retrieval via isolated headless Chrome.
        # Stuck-recovery search is now agentic: when the planner
        # decides it needs an external fact for THIS replan, it will
        # emit a <search> tag and the tool-loop driver fetches it.
        # The pre-built prompt no longer carries a retrieved_knowledge
        # block.

        # Verifier feedback block: render a concrete diagnostic
        # (patterns checked, per-pattern hits, file size + change
        # status, file content preview, LLM root-cause analysis)
        # for every non-verified outcome. Used to be gated on
        # ``stuck_category == "done_rejected"`` only, but now fires
        # on every force_replan because:
        #   • replan_cadence / actor_failure stuck reasons can
        #     ALSO have unsatisfied outcomes whose feedback the
        #     planner needs to see
        #   • the LLM diagnoser caches per outcome trace signature,
        #     so repeated force_replans without state change are
        #     effectively free
        verifier_fb_block: Optional[str] = None
        bad_outcomes_for_reauthor: List[Any] = []
        if self.ledger is not None:
            try:
                from mm_agents.structagent.ledger.core.timeline import (
                    render_verifier_feedback,
                )
                bad_outcomes = [
                    o for o in self.ledger.required_outcomes
                    if not o.is_verified()
                ]
                blocks = []
                for o in bad_outcomes[:3]:   # cap to 3 to keep prompt bounded
                    b = render_verifier_feedback(
                        o,
                        call_llm=self.call_llm, model=self.model,
                    )
                    if b:
                        blocks.append(b)
                if blocks:
                    verifier_fb_block = "\n\n".join(blocks)
                # Capture for the reauthor pass below — render_verifier_feedback
                # has now updated verifier_mismatch_strikes via the diagnoser.
                bad_outcomes_for_reauthor = bad_outcomes[:3]
            except Exception as e:
                if logger:
                    logger.info(
                        "[PA-Planner] verifier feedback render failed: %s", e)

        # G1+G2+G4 gate: when the diagnoser has consecutively classified
        # an outcome's failures as ``verifier_mismatch`` (and we haven't
        # already patched it), invoke the ledger-init authoring LLM to
        # rewrite the verify spec. The init-author has the full prompt
        # (KNOWN APP CONFIG PATHS, pattern tips, dual-channel rules), so
        # it produces accurate replacements; we feed it the diagnoser's
        # feedback as additional context.
        if self.ledger is not None and bad_outcomes_for_reauthor:
            try:
                from mm_agents.structagent.ledger.core.initializer import (
                    apply_reauthor_pass,
                )
                a11y_for_reauthor = _linearize_obs_a11y(obs, max_items=120)
                apply_reauthor_pass(
                    ledger=self.ledger,
                    instruction=instruction,
                    obs=obs,
                    bad_outcomes=bad_outcomes_for_reauthor,
                    call_llm=self.call_llm,
                    model=self.model,
                    a11y_text=a11y_for_reauthor,
                    active_url=_extract_active_url_from_a11y(a11y_for_reauthor),
                    domain=getattr(self, "_current_domain", None),
                )
            except Exception as e:
                if logger:
                    logger.info(
                        "[PA-Planner] reauthor pass failed: %s", e)

        # A8-V0: LLM-based stuck diagnosis. Fires on EVERY force_replan
        # (no extra "min replans" gate — the 6-rule recovery cascade
        # is already the conservative gate; if it decided to fire
        # force_replan, the agent is stuck enough to warrant a focused
        # root-cause synthesis). Duplicate calls on unchanged state
        # are de-duped by signature-based cache in diagnose_stuck.
        # The recovery's ``stuck_category`` + ``stuck_reason`` are
        # passed through so the LLM can scope its diagnosis (e.g.
        # replan_cadence → focus tactical not strategic; done_rejected
        # → focus on the pending outcome's verifier trace). Full
        # prompt + response persist to ``{results_dir}/stuck_diagnosis/``
        # for offline debug.
        #
        # Gated by ``self.enable_stuck_diagnosis_injection`` — when
        # the flag is OFF (default) skip the LLM call entirely AND
        # pass ``verifier_feedback=None`` to the builder so the
        # prompt-side injection is also skipped (preserves baseline
        # e8c4db0 behaviour exactly).
        if (self.ledger is not None
                and getattr(self, "enable_stuck_diagnosis_injection",
                            False)):
            # Switch between v1 stuck_diagnosis (the original 4-cat
            # diagnose) and v2 failure_attribution (5-phase analysis
            # + role attribution + router). Default v1 preserves
            # production behaviour exactly; set
            # USE_FAILURE_ATTRIBUTION=1 to opt into v2 LIVE
            # (verifier_override actually flips outcomes).
            from mm_agents.structagent.config import CAConfig
            _attribution_active = CAConfig.from_env().failure_attribution
            try:
                focus = self.ledger.next_focus_outcome()
                if focus is None:
                    pass
                elif _attribution_active:
                    # v2 path. The agent-side pre-replan hook (see
                    # ``_pa_predict`` in agent.py) ALREADY ran
                    # diagnose+router and either:
                    #   (a) cached the InterventionDecision on
                    #       ``self._cached_failure_attribution_decision``
                    #       — we consume it here, no duplicate
                    #       LLM call; or
                    #   (b) decided ``actor_redo`` and short-
                    #       circuited the planner entirely — in
                    #       which case this code path doesn't run
                    #       at all (mode wouldn't be force_replan).
                    # If neither cache nor short-circuit (e.g. the
                    # pre-hook crashed), we fall through to a fresh
                    # diagnose_failure call so the planner still
                    # gets actionable analysis.
                    from mm_agents.structagent.memory.runtime.recovery.recovery_block import (
                        render_planner_recovery_block,
                    )
                    decision = getattr(
                        self, "_cached_failure_attribution_decision",
                        None)
                    if decision is not None:
                        # consume the cache
                        self._cached_failure_attribution_decision = None
                        if logger:
                            logger.info(
                                "[PA-Planner] consuming cached v2 "
                                "decision: kind=%s reason=%r",
                                decision.kind,
                                (decision.reason or "")[:200])
                    else:
                        from mm_agents.structagent.attribution.failure_attribution import (
                            diagnose_failure,
                        )
                        from mm_agents.structagent.attribution.failure_attribution_router import (
                            route_failure_attribution,
                        )
                        diag = diagnose_failure(
                            focus_outcome=focus,
                            ledger=self.ledger,
                            observation=obs,
                            timeline=getattr(self, "timeline", []) or [],
                            stuck_category=stuck_category,
                            stuck_reason=stuck_reason,
                            call_llm=self.call_llm,
                            model=self.model,
                            results_dir=getattr(self, "_results_dir", None),
                            step_idx=getattr(self, "_wall_step_idx", 0) or 0,
                            current_domain=getattr(self, "_current_domain", None),
                            last_cli_run_output=getattr(self, "_last_cli_run_output", None),
                        )
                        decision = (
                            route_failure_attribution(
                                diag, focus_outcome=focus,
                                timeline=getattr(self, "timeline",
                                                 []) or [])
                            if diag is not None else None)
                        if decision is not None and logger:
                            logger.info(
                                "[PA-Planner] failure_attribution v2 "
                                "fresh-diagnose (no cache): "
                                "role=%s cat=%s router=%s",
                                diag.attributed_to_role,
                                diag.category,
                                decision.kind,
                            )
                    if decision is not None:
                        # The planner sees the analysis as a block
                        # so it can keep the same plan (when role=
                        # actor/env/verifier) or replan deeply
                        # (when role=planner/strategic). The actor
                        # stash already happened in agent.py if
                        # actor_redo (and in that case mode wouldn't
                        # be force_replan, so this code path is for
                        # planner_replan / fallback_planner only).
                        block = render_planner_recovery_block(
                            decision.params, focus_id=focus.id)
                        if verifier_fb_block:
                            verifier_fb_block = (
                                verifier_fb_block + "\n\n" + block)
                        else:
                            verifier_fb_block = block
                else:
                    # v1 path (unchanged).
                    from mm_agents.structagent.attribution.stuck_diagnosis import (
                        diagnose_stuck,
                        render_stuck_diagnosis,
                    )
                    stuck_diag = diagnose_stuck(
                        focus_outcome=focus,
                        ledger=self.ledger,
                        observation=obs,
                        timeline=getattr(self, "timeline", []) or [],
                        stuck_category=stuck_category,
                        stuck_reason=stuck_reason,
                        call_llm=self.call_llm,
                        model=self.model,
                        results_dir=getattr(self, "_results_dir", None),
                        step_idx=getattr(self, "_wall_step_idx", 0) or 0,
                        current_domain=getattr(self, "_current_domain", None),
                        last_cli_run_output=getattr(self, "_last_cli_run_output", None),
                    )
                    if stuck_diag is not None:
                        block = render_stuck_diagnosis(
                            stuck_diag, focus_id=focus.id,
                        )
                        if verifier_fb_block:
                            verifier_fb_block = (
                                verifier_fb_block + "\n\n" + block
                            )
                        else:
                            verifier_fb_block = block
            except Exception as e:
                if logger:
                    logger.info(
                        "[PA-Planner] %s hook failed: %s",
                        "failure_attribution" if _attribution_active else "stuck_diagnosis",
                        e)

        # Gate the injection at the builder boundary too — when the
        # flag is OFF, pass ``verifier_feedback=None`` even if
        # ``render_verifier_feedback`` populated ``verifier_fb_block``
        # for the reauthor side-effects above. This way the planner
        # prompt is byte-identical to baseline when the flag is off.
        _vfb_to_inject = (
            verifier_fb_block
            if getattr(self, "enable_stuck_diagnosis_injection",
                       False)
            else None
        )

        # B2.A: at force_replan, surface corpus L2 alternative
        # strategies + L1 alternative actions as a sibling block to
        # the diagnosis. Gated on ENABLE_PLANNER_EXPERIENCE_MEMORY
        # (same flag as the initial-plan L2 injection and the actor
        # L1 injection). Render layer returns ("", meta_with_OOD)
        # on miss → safe no-op. Failure is logged + swallowed so
        # retrieval issues NEVER break the force_replan path.
        _l1_replan_block = ""
        _l2_replan_block = ""
        from mm_agents.structagent.config import CAConfig
        if CAConfig.from_env().planner_experience_memory:
            try:
                from mm_agents.structagent.memory.runtime.injection.replan_memory_injection import (
                    render_l1_alternatives_for_replan,
                    render_l2_alternatives_for_replan,
                )
                from mm_agents.structagent.memory.runtime.injection.planner_prompt_injection import (
                    dump_memory_debug, format_log_summary,
                )
                _replan_dom = getattr(self, "_current_domain", "") or ""
                _l1_replan_block, _l1_meta = render_l1_alternatives_for_replan(
                    self.current_subgoal or "", domain=_replan_dom)
                _l2_replan_block, _l2_meta = render_l2_alternatives_for_replan(
                    instruction or "", domain=_replan_dom)
                if logger:
                    logger.info(
                        "[PlannerMemory] force_replan L1-alt injected "
                        "(%d chars) %s",
                        len(_l1_replan_block),
                        format_log_summary(_l1_meta),
                    )
                    logger.info(
                        "[PlannerMemory] force_replan L2-alt injected "
                        "(%d chars) %s",
                        len(_l2_replan_block),
                        format_log_summary(_l2_meta),
                    )
                _rd = getattr(self, "_results_dir", None)
                _wstep = getattr(self, "_wall_step_idx", 0) or 0
                dump_memory_debug(
                    _rd,
                    stage_tag=f"force_replan_step{_wstep:03d}_l1_alt",
                    block=_l1_replan_block,
                    meta=_l1_meta,
                    subgoal=self.current_subgoal,
                )
                dump_memory_debug(
                    _rd,
                    stage_tag=f"force_replan_step{_wstep:03d}_l2_alt",
                    block=_l2_replan_block,
                    meta=_l2_meta,
                )
            except Exception as _replan_mem_e:
                if logger:
                    logger.info(
                        "[PlannerMemory] force_replan L1/L2 retrieval "
                        "failed (ignored): %s", _replan_mem_e,
                    )
                _l1_replan_block = ""
                _l2_replan_block = ""

        return doc_block_prefix + data_layout_prefix + build_prompt_force_replan(
            **common,
            recent_actions=recent_actions,
            abandoned_subgoals=abandoned,
            stuck_reason=stuck_reason,
            stuck_category=stuck_category,
            retrieved_knowledge=None,
            verifier_feedback=_vfb_to_inject,
            l1_alternatives_block=_l1_replan_block,
            l2_alternatives_block=_l2_replan_block,
        )


    def _pa_check_subgoal_done(self, subgoal: str, obs: dict) -> Tuple[bool, Optional[str]]:
        """Ask the planner model whether the current subgoal has been completed.
        If NO, provides a refined subgoal with feedback for the actor.

        Returns: (is_done, refined_subgoal_or_None)
            - (True, None) if subgoal is done
            - (False, "refined subgoal text") if not done, with corrective feedback
            - (False, None) on error
        """
        if not self.actor_history:
            return False, None

        # Build messages with recent screenshots and actions for this subgoal
        messages = [
            {"role": "system", "content": [{"type": "text", "text":
                "You are a GUI task evaluator. Given screenshots and actions taken for a specific subgoal, "
                "determine whether the subgoal has been **fully and correctly committed** in the LATEST screenshot. "
                "Glyphs appearing in a field or a partial visual change are NOT by themselves proof of completion. "
                "If not committed, provide a more detailed subgoal with specific feedback, but DO NOT change the original subgoal.\n\n"
                "COMMIT-STATE CHECKLIST — go through these before answering YES:\n\n"
                "1. Typeahead / autocomplete fields (airport code, city, email recipient, stock ticker, "
                "   search-with-suggestions): if a suggestion dropdown / popover is visible below or next to "
                "   the field, the typed value is NOT committed yet — the user/agent still has to click one "
                "   of the suggestions. Answer NO and refine the subgoal to 'click the <visible suggestion row> "
                "   in the open dropdown to commit <value>'.\n"
                "2. Plain text inputs: the typed value should be persisted in the field AND no IME composition "
                "   caret / selection highlight should remain. A blue-highlighted field containing the new text "
                "   may still be uncommitted.\n"
                "3. Checkboxes: must show the checked glyph (✓, filled square, colored box), NOT just a click "
                "   ripple near the label. If only the label was highlighted but the box is still empty, answer "
                "   NO and refine to 'click the actual checkbox square to the left of the <label>'.\n"
                "4. Radio buttons: the target option should show the filled dot; the previously-selected option "
                "   should be empty. If both look empty or the old one is still filled, answer NO.\n"
                "5. Date / time pickers: the selected date must appear in the field's display text AND the picker "
                "   panel should be closed. If the panel is still open, the date is not committed yet.\n"
                "6. Dropdowns / selects: the dropdown must be CLOSED and the chosen option shown in the collapsed "
                "   field display. An open dropdown with an item merely hovered is NOT committed.\n"
                "7. Dialogs / modals (confirm, save, delete): completed only when the dialog is DISMISSED and the "
                "   underlying UI reflects the committed state, not merely when the confirm button was clicked.\n"
                "8. Page / route transitions (new URL, list update, search results): verify the NEW content is "
                "   actually rendered, not just a loading spinner.\n\n"
                "When uncertain, err on the side of NO: a false-NO costs one extra actor turn; a false-YES closes "
                "the subgoal prematurely and forfeits the rest of the task."
            }]},
        ]
        user_content = []

        # Include recent actor history (screenshots + actions for this subgoal)
        history_to_show = self.actor_history[-3:]
        for entry in history_to_show:
            b64 = entry.get("screenshot_b64", "")
            resp = entry.get("response", "")
            if b64:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            if resp:
                user_content.append({"type": "text", "text": f"Actor action: {resp[:500]}"})

        # Add current screenshot
        screenshot_bytes = obs.get("screenshot")
        if isinstance(screenshot_bytes, bytes):
            current_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{current_b64}"}})

        # Attach a11y tree of the current screenshot — critical for the
        # radio/checkbox checklist rules in the system prompt, which can
        # now be answered from st:checked / st:selected state instead of
        # pixel inspection.
        a11y_text = _linearize_obs_a11y(obs)
        if a11y_text:
            user_content.append({"type": "text", "text": (
                "Accessibility tree of the LATEST screenshot above "
                "(tab-separated: tag / text / state). Only true-valued "
                "state flags appear (e.g. 'checked,enabled' on a radio-button "
                "means it is the selected option; others listed with just "
                "'enabled' are not selected). For radio/checkbox/focus "
                "verification, read the state column rather than the image.\n\n"
                f"{a11y_text}"
            )})

        # Progress Ledger: read-only structured state from a separate observer.
        if self.ledger is not None:
            user_content.append({"type": "text", "text": (
                "Structured progress ledger (authoritative — produced by a "
                "separate observer, not by your reasoning):\n\n"
                f"{self.ledger.to_prompt_block()}"
            )})

        user_content.append({"type": "text", "text": f"""The above screenshots (oldest to latest) show the result of actions taken for this subgoal:

Subgoal: {subgoal}

Based on the LATEST screenshot, has this subgoal been visually completed?

If YES: answer "YES" only.
If NO: answer in this exact format:
NO
<refined_subgoal>A more specific version of the subgoal based on what you see in the screenshot. Describe exactly what the actor should do differently, including the correct UI element to target. But DO NOT change the intent of the original subgoal.</refined_subgoal>"""})

        messages.append({"role": "user", "content": user_content})

        try:
            response = self.call_llm(
                {"model": self.model, "messages": messages, "max_tokens": 500, "top_p": self.top_p, "temperature": 0.0},
                self.model,
            )
            answer = (response or "").strip()
            # Strip thinking content. Qwen3.5-VL emits free-form reasoning
            # followed by </think> and then the answer (no opening tag).
            # Drop everything up to and including the LAST </think>; also
            # handle the older fully-wrapped <think>...</think> form.
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            _thk = answer.rfind("</think>")
            if _thk >= 0:
                answer = answer[_thk + len("</think>"):].lstrip()
            is_done = answer.upper().startswith("YES")

            refined_subgoal = None
            if not is_done:
                # Extract refined subgoal from response
                m = re.search(r'<refined_subgoal>(.*?)</refined_subgoal>', answer, re.DOTALL)
                if m:
                    refined_subgoal = m.group(1).strip()

            if logger:
                logger.info("[PA-Planner] Subgoal done check: %s → done=%s, refined=%s",
                            answer, is_done, refined_subgoal if refined_subgoal else "None")
            return is_done, refined_subgoal
        except Exception as e:
            if logger:
                logger.warning("[PA-Planner] Subgoal done check failed: %s", e)
            return False, None

    # --- plan corrector ---

    def _correct_plan(self, plan: str) -> str:
        """Review and correct plan using recent screenshots."""
        from mm_agents.structagent.utils.helpers import correct_plan
        screenshots = [img_b64 for img_b64, _ in self.planner_history[-3:]] if self.planner_history else []
        if not screenshots:
            if logger:
                logger.info("[PA-Corrector] No screenshots available, skipping correction")
            return plan
        if logger:
            logger.info("[PA-Corrector] Correcting plan with %d screenshots...", len(screenshots))
        def _call(messages, max_tokens=1500):
            return self.call_llm(
                {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                 "top_p": 0.9, "temperature": 0.0},
                self.model)
        ledger_block = None
        if getattr(self, "ledger", None) is not None:
            try:
                ledger_block = self.ledger.to_prompt_block()
            except Exception:
                ledger_block = None
        corrected = correct_plan(
            self.instruction, plan, screenshots, _call,
            completed_subgoals=self.completed_subgoals,
            ledger_block=ledger_block,
        )
        if corrected:
            if logger:
                logger.info("[PA-Corrector] Plan corrected:\n%s", corrected)
            return corrected
        else:
            if logger:
                logger.info("[PA-Corrector] Plan unchanged (no corrections needed)")
            return plan



    # --- planner response parsing ---

    @staticmethod
    def _pa_parse_planner_response(response: str) -> Dict[str, Any]:
        """Parse the Phase-2 unified planner output contract.

        Expected tags (any may be missing; code is tolerant). The contract
        was re-ordered to enforce a forward chain of thought:
        observations → assessment → review → bottleneck → decision →
        plan/hint. The old <feedback_to_actor> field (emitted INSIDE
        <last_subgoal_assessment>) was renamed to <actor_hint> and moved
        AFTER all reasoning so the model can no longer pre-commit to a
        scroll-loop strategy. The <dead_end_review> block was also
        removed — planner-driven review confused "I'm proposing X"
        with "X failed", polluting failed_paths with strategies the
        planner was about to TRY rather than ones that had FAILED.

          <observations>...</observations>
          <last_subgoal_assessment>
            <done>YES|NO</done>
            <evidence>...</evidence>
          </last_subgoal_assessment>
          <Bottleneck>one sentence</Bottleneck>
          <decision>DONE|CONTINUE|REPLAN</decision>
          <plan>
            <strategy>...</strategy>                       (NEW: high-level approach)
            <step>...</step>
            ...
          </plan>                                          (REPLAN only)
          <actor_hint>...</actor_hint>                     (done=NO only)
          <root_cause>...</root_cause>                     (actor_exhausted path)

        Returns a dict with:
          plan (str | None)               raw <plan>..</plan> body for legacy callers
          plan_strategy (str | None)      NEW: <strategy> text from inside <plan>;
                                          drives strategy-level dead-end tracking
          decision (str | None)           DONE / CONTINUE / REPLAN
          last_subgoal_done (bool | None) parsed from <done>
          evidence (str | None)
          observations (str | None)
          bottleneck (str | None)
          feedback_to_actor (str | None)  populated from <actor_hint> (new) OR
                                          <feedback_to_actor> (back-compat)
        """
        out: Dict[str, Any] = {
            "plan": None, "plan_strategy": None, "decision": None,
            "last_subgoal_done": None, "evidence": None,
            "observations": None,
            "bottleneck": None,
            "feedback_to_actor": None,
        }
        if not (response and response.strip()):
            return out
        # Qwen3.5 thinking-mode strip: drop everything up to and
        # including the LAST </think>. With QWEN35_THINKING=1 the model
        # emits free-form reasoning + </think> + structured answer; the
        # reasoning can contain text that confuses tag parsers below.
        # No-op when thinking is off.
        _thk = response.rfind("</think>")
        if _thk >= 0:
            response = response[_thk + len("</think>"):].lstrip()

        # --- observations (description-only block at the top) ---
        obs_m = re.search(r"<observations>\s*([\s\S]*?)\s*</observations>",
                          response, re.IGNORECASE)
        if obs_m:
            out["observations"] = obs_m.group(1).strip()

        # --- last_subgoal_assessment (no longer carries feedback_to_actor) ---
        ass_m = re.search(
            r"<last_subgoal_assessment>([\s\S]*?)</last_subgoal_assessment>",
            response, re.IGNORECASE)
        if ass_m:
            body = ass_m.group(1)
            done_m = re.search(r"<done>\s*([\s\S]*?)\s*</done>", body, re.IGNORECASE)
            if done_m:
                v = done_m.group(1).strip().upper()
                if v.startswith("YES"):
                    out["last_subgoal_done"] = True
                elif v.startswith("NO"):
                    out["last_subgoal_done"] = False
            ev_m = re.search(r"<evidence>\s*([\s\S]*?)\s*</evidence>", body, re.IGNORECASE)
            if ev_m:
                out["evidence"] = ev_m.group(1).strip()
            # Back-compat: older planner checkpoints still emit
            # <feedback_to_actor> inside the assessment. Honor it as a
            # fallback when the new <actor_hint> tag is absent below.
            fb_m = re.search(r"<feedback_to_actor>\s*([\s\S]*?)\s*</feedback_to_actor>",
                             body, re.IGNORECASE)
            if fb_m:
                fb = fb_m.group(1).strip()
                if fb and fb.lower() not in ("(omit when done=yes)", "omitted", "n/a", "none"):
                    out["feedback_to_actor"] = fb

        # NOTE: <dead_end_review> block was removed from the contract —
        # see docstring above for rationale. Old planner checkpoints
        # that still emit it are silently ignored; failed_paths is
        # written exclusively by the cadence trigger.

        # --- Bottleneck (synthesis sentence — drives decision) ---
        bn_m = re.search(r"<Bottleneck>\s*([\s\S]*?)\s*</Bottleneck>",
                         response, re.IGNORECASE)
        if bn_m:
            bn = bn_m.group(1).strip()
            if bn and bn.lower() not in ("none", "n/a", "(omit)"):
                out["bottleneck"] = bn

        # --- plan (raw XML body between <plan>..</plan>) ---
        plan_m = re.search(r"<plan>\s*([\s\S]*?)\s*</plan>", response, re.IGNORECASE)
        if plan_m:
            out["plan"] = plan_m.group(1).strip()
            # Extract <strategy> from inside the plan body. The contract
            # asks the planner to declare ONE strategy per plan as the
            # high-level approach this plan implements; downstream
            # bookkeeping uses it to detect strategy-level abandonment
            # (planner switches strategy across REPLANs ⇒ old strategy
            # was abandoned and gets recorded as a strategy-level dead-end).
            strat_m = re.search(
                r"<strategy>\s*([\s\S]*?)\s*</strategy>",
                out["plan"], re.IGNORECASE)
            if strat_m:
                strat = strat_m.group(1).strip()
                # Drop placeholder / instruction echoes that small
                # models occasionally copy verbatim from the prompt.
                if strat and "Strategy ≠ tactic" not in strat \
                        and not strat.lower().startswith("one sentence describing"):
                    out["plan_strategy"] = strat[:300]

        # --- decision ---
        decision_m = re.search(r"<decision>\s*([\s\S]*?)\s*</decision>", response, re.IGNORECASE)
        if decision_m:
            raw = decision_m.group(1).strip().upper()
            if "REPLAN" in raw:
                out["decision"] = "REPLAN"
            elif "DONE" in raw:
                out["decision"] = "DONE"
            elif "CONTINUE" in raw:
                out["decision"] = "CONTINUE"
            elif "INFEASIBLE" in raw:
                # Agent's own task-level infeasibility verdict (only offered in
                # the stuck/force-replan contract after ≥2 abandoned approaches).
                # Drives a terminal FAIL in the agent dispatch. REPLAN is checked
                # first above, so any ambiguous "INFEASIBLE … REPLAN" favours
                # retrying, never failing — the conservative default.
                out["decision"] = "INFEASIBLE"

        # root_cause → REPLAN (actor_exhausted prompt shape)
        if re.search(r"<root_cause>", response, re.IGNORECASE):
            out["decision"] = "REPLAN"

        # --- actor_hint (new contract) — preferred over the back-compat
        # <feedback_to_actor> read above. The hint replaces feedback_to_actor
        # at the prompt level but we keep the agent-side field name
        # ``feedback_to_actor`` to avoid downstream churn. ---
        ah_m = re.search(r"<actor_hint>\s*([\s\S]*?)\s*</actor_hint>",
                         response, re.IGNORECASE)
        if ah_m:
            ah = ah_m.group(1).strip()
            # Drop instruction echoes / placeholders.
            if (ah
                    and ah.lower() not in ("(omit)", "omitted", "n/a", "none")
                    and "Emit ONLY if" not in ah
                    and "HARD CONSTRAINT" not in ah):
                out["feedback_to_actor"] = ah  # NEW value wins over back-compat

        # Phase-2 contract: <plan> is emitted ONLY when decision=REPLAN.
        # Some planner checkpoints still emit a <plan> next to CONTINUE
        # (restating what's already in the queue); honor the explicit
        # <decision> tag and DROP the stray plan so the queue isn't
        # needlessly replaced. Only infer REPLAN when decision is absent.
        if out["plan"] is not None and out["decision"] in ("CONTINUE", "INFEASIBLE"):
            out["plan"] = None
        if out["plan"] is not None and out["decision"] is None:
            out["decision"] = "REPLAN"
            # ``decision_inferred`` lets the caller distinguish "LLM
            # explicitly said REPLAN" from "LLM emitted a plan with no
            # <decision> tag and we filled in REPLAN" — the latter is
            # benign at step 0 (mode=initial never asks for a decision)
            # but the planner-cadence counters / log labels should not
            # treat it as a real REPLAN event.
            out["decision_inferred"] = True

        # Contract guard: actor_hint should only carry a value when
        # last_subgoal_done is False. If the planner emitted <actor_hint>
        # alongside done=YES (small-model contract leak), drop it — the
        # subgoal succeeded so no hint applies.
        if out["last_subgoal_done"] is True:
            out["feedback_to_actor"] = None

        ans_m = re.search(r'ANSWER\((.+?)\)', response, re.IGNORECASE | re.DOTALL)
        if ans_m and out["decision"] is None:
            out["decision"] = "DONE"

        # --- notebook_audit (host-only Notebook writes) ---
        # Planner contract: every <plan> begins with a <notebook_audit>
        # carrying <needs_writes>YES|NO</needs_writes> + optional
        # <writes> block with one note_write(key=..., value=...) call
        # per line. Framework executes these BEFORE the actor runs the
        # first GUI/calc <step>. Parser is tolerant — when the block
        # is absent (older planner output) we just skip the dispatch.
        out["notebook_writes"] = []
        au_m = re.search(
            r"<notebook_audit>\s*([\s\S]*?)\s*</notebook_audit>",
            response, re.IGNORECASE)
        if au_m:
            body = au_m.group(1)
            needs_m = re.search(r"<needs_writes>\s*([\s\S]*?)\s*</needs_writes>",
                                body, re.IGNORECASE)
            needs_yes = bool(needs_m and needs_m.group(1).strip().upper().startswith("YES"))
            writes_m = re.search(r"<writes>\s*([\s\S]*?)\s*</writes>",
                                 body, re.IGNORECASE)
            if needs_yes and writes_m:
                # Each non-empty line should be a note_write(...) call.
                for line in writes_m.group(1).splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    # Drop any leading bullet / numbering noise.
                    line = re.sub(r"^[-*•\d.\)\s]+", "", line)
                    if line.lower().startswith("note_write"):
                        out["notebook_writes"].append(line)
        return out

    @classmethod
    def _pa_parse_plan_into_subgoals(cls, plan_text: str) -> List[Dict[str, Any]]:
        """Parse a plan body into ``[{"text": ..., "expected_post_state": ...}, ...]``.

        Accepts two formats:

          A) New XML-structured plan (Phase 2 contract):
             <step>
               <text>Click the X button.</text>
               <expected_post_state>Dropdown menu labeled Y is visible.</expected_post_state>
             </step>

          B) Legacy numbered-list plan (for back-compat with any planner
             model call that still emits the old format).

        Format A is tolerant of common planner-side XML emit bugs:

          (1) Missing ``</step>`` closure — the planner sometimes opens
              two ``<step>`` blocks back-to-back without closing the
              first. We split on the OPENING ``<step>`` tag instead of
              requiring a balanced ``<step>...</step>`` regex match.

          (2) ``</text>`` typo as ``</expected_post_state>`` close —
              planner often writes ``</text>`` to close the
              ``<expected_post_state>`` tag. We accept ``</text>`` as a
              tolerated closing tag for that field.

          (3) Missing tag closures fall back to peeking the NEXT opening
              tag (``<step>``, ``<expected_post_state>``, ``</plan>``)
              as the implicit terminator.

        Without these tolerances, Format A would silently emit zero
        steps on a malformed plan, falling through to Format B's
        "every non-empty line is a subgoal" branch and pushing
        XML-tag-literal lines (``<step>``, ``<expected_post_state>...``)
        into ``subgoal_queue`` — which is what bricked the carl
        trajectory's later steps. Format B is now also XML-aware and
        skips lines starting with ``<``.

        Note: the legacy ``<script_body>`` tag is now defunct. If a stale
        plan still contains ``<script_body>``, we silently strip it and
        use ``<text>``.
        """
        subs: List[Dict[str, str]] = []
        if not plan_text:
            return subs

        # --- Format A: split on <step> openings (tolerant of missing closures) ---
        # ``re.split`` with a CAPTURING group on the OPENING tag yields:
        #   [<prefix>, <step-tag-1>, body_1, <step-tag-2>, body_2, ...]
        # Each body runs from after its opening <step> to right before
        # the NEXT opening <step> or end-of-text — which means a missing
        # </step> can no longer drop a step on the floor. The captured
        # tag is kept so a multi-app ``<step app="...">`` attribute can
        # be parsed (drives the agent's _current_domain switch).
        parts = re.split(
            r"(<step\b[^>]*>)", plan_text, flags=re.IGNORECASE,
        )
        step_tags = parts[1::2] if len(parts) > 1 else []
        step_bodies = parts[2::2] if len(parts) > 1 else []
        for _si, body in enumerate(step_bodies):
            step_tag = step_tags[_si] if _si < len(step_tags) else ""
            # Multi-app: parse the per-step app tag, e.g.
            # <step app="libreoffice_calc">. Normalized to lowercase
            # underscore form; None when absent (single-app plans).
            _app_m = re.search(
                r'\bapp\s*=\s*["\']([^"\']+)["\']',
                step_tag, re.IGNORECASE)
            step_app = (_app_m.group(1).strip().lower().replace(" ", "_")
                        if _app_m else None)
            # Ledger-driven subgoal completion (C / P1). The planner
            # declares which outcome id this step is supposed to flip
            # to verified, e.g. <step app="thunderbird"
            # target="bills_folder_selected">. Framework reads ledger
            # state to decide subgoal-done instead of relying on the
            # planner's self-assessment against an unobservable
            # expected_post_state (the failure mode where "URL is in
            # clipboard" loops forever because clipboard isn't
            # visible). Empty when absent — falls back to legacy
            # post-state self-check. Accept ``target`` (singular,
            # canonical) and ``targets`` (legacy/alias).
            _targets_m = re.search(
                r'\btargets?\s*=\s*["\']([^"\']*)["\']',
                step_tag, re.IGNORECASE)
            step_targets: List[str] = []
            if _targets_m:
                for t in _targets_m.group(1).split(","):
                    t = t.strip()
                    if t:
                        # Normalize like outcome ids: snake_case
                        t = re.sub(r"[^a-zA-Z0-9_]+", "_", t).strip("_").lower()
                        if t and t not in step_targets:
                            step_targets.append(t)
            # Trim trailing </step>, </plan>, or any leaked </text> /
            # </expected_post_state> at the very end so they don't end
            # up inside the captured fields.
            body_trimmed = re.sub(
                r"\s*(?:</step>|</plan>)[\s\S]*$", "",
                body, flags=re.IGNORECASE,
            )

            # <text> ... terminator. Terminator: </text>, OR the next
            # opening tag (<expected_post_state>, <step>, <plan>), OR
            # </plan>. Lookahead lets us peek without consuming.
            text_m = re.search(
                r"<text>\s*([\s\S]*?)\s*"
                r"(?:</text>"
                r"|(?=<expected_post_state\b)"
                r"|(?=<step\b)"
                r"|(?=</plan>)"
                r"|$)",
                body_trimmed, re.IGNORECASE,
            )
            text = text_m.group(1).strip() if text_m else ""

            # <expected_post_state> ... terminator. ALSO accepts </text>
            # as the closing tag — that's the typo the planner emits in
            # ~all plans we've sampled.
            eps_m = re.search(
                r"<expected_post_state>\s*([\s\S]*?)\s*"
                r"(?:</expected_post_state>"
                r"|</text>"
                r"|(?=<step\b)"
                r"|(?=</plan>)"
                r"|$)",
                body_trimmed, re.IGNORECASE,
            )
            eps = eps_m.group(1).strip() if eps_m else ""

            # Defensive: strip any straggling XML tags that snuck into
            # the captured fields (e.g. when planner inlined a stray
            # tag mid-sentence).
            text = re.sub(r"<[^>]+>", "", text).strip()
            eps = re.sub(r"<[^>]+>", "", eps).strip()

            if text and not cls._SUBGOAL_SKIP_RE.search(text):
                subs.append({"text": text, "expected_post_state": eps,
                             "app": step_app, "targets": step_targets})

        # If the planner emitted ANY Format-A <step> opening, commit to
        # Format A — even if some/all blocks lacked <text>. Falling
        # through to Format-B's "every non-empty line is a subgoal"
        # catch-all is what produces the XML-noise-as-subgoal bug.
        if step_bodies:
            return subs

        # --- Format B: numbered list fallback ---
        for line in plan_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Defensive: never accept a bare XML-tag-literal line as a
            # subgoal. If we got here, Format A found no <step> opening,
            # but the plan may still contain straggler tags from a
            # half-emitted XML response — skip them outright.
            if line.startswith("<") or line.startswith("</"):
                continue
            m = re.match(r'^(?:step\s*)?\d+[\.\):\-]\s*(.+)$', line, re.IGNORECASE)
            if m:
                text = m.group(1).strip()
                if not cls._SUBGOAL_SKIP_RE.search(text):
                    subs.append({"text": text, "expected_post_state": "",
                                 "app": None, "targets": []})
        if not subs:
            for line in plan_text.strip().splitlines():
                text = line.strip()
                if not text:
                    continue
                if text.startswith("<") or text.startswith("</"):
                    continue
                if not cls._SUBGOAL_SKIP_RE.search(text):
                    subs.append({"text": text, "expected_post_state": "",
                                 "app": None, "targets": []})
        return subs

    # --- LAYER 3 / LAYER 4 renderers (see _LAYERED_PROMPT_GUIDE) ---

    def _render_layer3_subgoal_state(self) -> str:
        """LAYER 3 — subgoal state. Sole owner of:
          • current plan steps (✓ done / ▶ active / · pending)
          • current_subgoal text + expected_post_state
          • subgoal_queue / completed / abandoned
          • working-memory Facts produced by past subgoals
        """
        lines: List[str] = ["[LAYER 3 — Subgoal state]", ""]

        # Plan structure: completed → active → remaining.
        completed = list(getattr(self, "completed_subgoals", []) or [])
        current = getattr(self, "current_subgoal", None) or ""
        cur_step = getattr(self, "current_subgoal_step", 0)
        queue = list(getattr(self, "subgoal_queue", []) or [])
        eps = getattr(self, "current_subgoal_expected_post_state", None) or "(not specified)"
        n_total = len(completed) + (1 if current else 0) + len(queue)
        lines.append(f"Plan structure ({n_total} step(s) total):")
        for i, sg in enumerate(completed, 1):
            lines.append(f"  ✓ step {i} — {sg[:140]}")
        if current:
            idx = len(completed) + 1
            lines.append(
                f"  ▶ step {idx} — {current[:140]}  (CURRENT — "
                f"on this subgoal for {cur_step} actor turn(s))")
            lines.append(f"      expected post-state: {eps[:240]}")
        for i, sg in enumerate(queue, start=len(completed) + 2):
            lines.append(f"  · step {i} — {sg[:140]}")
        lines.append(f"Remaining-subgoals count: {len(queue)}")

        # Abandoned subgoals — pulled from agent state if tracked.
        abandoned = list(getattr(self, "abandoned_subgoals", []) or [])
        if abandoned:
            lines.append("")
            lines.append(
                "Abandoned subgoals (DO NOT re-issue these — they "
                "burned their step budget without verifying):")
            for sg in abandoned[-6:]:
                lines.append(f"  • {sg[:200]}")

        verifier_report = getattr(
            self, "_pending_planner_verifier_report", None)
        if verifier_report:
            lines.append("")
            lines.append("[Verifier report after actor loop]")
            lines.append(
                "This is evidence from a separate verifier after the actor "
                "stopped its low-level loop. Use it to decide whether to "
                "continue, revise the subgoal, or replan; it is not an "
                "automatic runtime decision.")
            lines.append(verifier_report)

        # Working memory (Facts captured so far by completed subgoals).
        if self.ledger is not None:
            try:
                from mm_agents.structagent.ledger.core.records import (
                    render_working_memory as _render_facts,
                )
                _focus = self.ledger.next_focus_outcome()
                _focus_id = _focus.id if _focus is not None else None
                facts_block = _render_facts(
                    self.ledger.working_memory(),
                    focus_outcome_id=_focus_id,
                )
                if facts_block:
                    lines.append("")
                    lines.append(facts_block)
            except Exception as e:
                if logger:
                    logger.info("[LAYER 3] facts render failed: %s", e)

        # Judge guidance — kept here (LAYER 3 owns the subgoal/eps the
        # judgment is scoped to). The actual judgment goes in LAYER 5's
        # <last_subgoal_assessment> output tag.
        lines.append("")
        lines.append(
            "When you fill <last_subgoal_assessment> in your output, "
            "judge <done> against the CURRENT subgoal's expected_post_"
            "state above — read evidence from the source that can "
            "actually SEE that state:")
        lines.append(
            "  • structured doc edits (alignment/font/color/text/size) "
            "→ STRUCTURE block in LAYER 4 (live UNO re-probe each turn)")
        lines.append("  • browser state → a11y tree + URL in LAYER 4")
        lines.append("  • shell / file results → last cli_run output in LAYER 4")
        lines.append("  • dialogs / navigation / highlight → screenshot in LAYER 4")
        return "\n".join(lines)

    def _render_layer4_evidence_text(
        self,
        *,
        snapshot: Any,
        a11y_text: Optional[str],
        last_turn_delta_block: Optional[str],
        last_cli_run: Optional[dict],
        recent_actor_actions: List[str],
        stuck_category: Optional[str],
        stuck_reason: Optional[str],
    ) -> str:
        """LAYER 4 — turn evidence text block. Images attached separately
        upstream (K-frame history attach). Sole owner of:
          • perceiver SCREEN SNAPSHOT  (perceiver path)
          • a11y tree + last-turn delta (raw-obs path)
          • recent actor actions on this subgoal (last 6, newest-first)
          • last cli_run output
          • stuck signal (force_replan category + reason)
        """
        lines: List[str] = ["[LAYER 4 — This turn's evidence]", ""]
        lines.append(
            "Up to 3 screenshots are attached separately ABOVE this "
            "block (oldest → newest). The text below summarises what the "
            "perceiver / a11y tree / last cli_run / actor recently produced.")

        if snapshot is not None:
            from mm_agents.structagent.perception.perceiver import to_prompt_block
            lines.append("")
            lines.append(
                "SCREEN SNAPSHOT (perceiver-distilled, subgoal-conditioned) — "
                "trust the subgoal_check verdicts (each comes with concrete "
                "evidence). If coverage='partial' be conservative; verify "
                "before committing major decisions.")
            lines.append("")
            lines.append(to_prompt_block(snapshot))
        elif a11y_text:
            from mm_agents.structagent.ledger.support.a11y_legend import with_legend
            lines.append("")
            lines.append(
                "Accessibility tree of the current screen — authoritative "
                "source for UI state (fall back to visual inspection only "
                "for layout / image / color).")
            lines.append("")
            lines.append(with_legend(a11y_text, header="a11y rows (tag\ttext\tstate):"))
            if last_turn_delta_block:
                lines.append("")
                lines.append(
                    "Last-1-turn observable delta — what the actor's most "
                    "recent action ACTUALLY changed on screen (a11y diff). "
                    "If the delta does not show the change required by the "
                    "current subgoal's expected_post_state in LAYER 3, the "
                    "answer is <done>NO</done> regardless of what the "
                    "screenshot looks like.")
                lines.append("")
                lines.append(last_turn_delta_block)

        if recent_actor_actions:
            lines.append("")
            lines.append("Recent actor actions (most recent first):")
            for i, a in enumerate(reversed(recent_actor_actions[-6:])):
                lines.append(f"  -{i+1}: {(a or '').strip()[:240]}")

        if last_cli_run:
            cmd = (last_cli_run.get("command") or "")[:400]
            cmd_mode = last_cli_run.get("mode") or "?"
            rc = last_cli_run.get("returncode")
            stdout = (last_cli_run.get("stdout") or "")[:2000]
            stderr = (last_cli_run.get("stderr") or "")[:800]
            lines.append("")
            lines.append(
                "Last cli_run output — the previous turn ran a non-interactive "
                "shell command; the screenshot does NOT show its output. "
                "This block IS the empirical result; trust it over the "
                "screenshot when judging whether the command succeeded.")
            lines.append(f"  command:    {cmd}")
            lines.append(f"  mode:       {cmd_mode}")
            lines.append(f"  returncode: {rc}")
            lines.append(f"  stdout:     {stdout or '(empty)'}")
            lines.append(f"  stderr:     {stderr or '(empty)'}")

        if stuck_category or stuck_reason:
            lines.append("")
            lines.append("Stuck signal (this turn is force_replan):")
            if stuck_category:
                lines.append(f"  category: {stuck_category}")
            if stuck_reason:
                lines.append(f"  reason:   {stuck_reason[:400]}")

        return "\n".join(lines)

    # --- planner call (uses _call_llm_vllm via call_llm) ---

    def _pa_call_planner(self, instruction: str, obs: dict, step_idx: int,
                         env=None,
                         mode: str = "progress_check"
                         ) -> Tuple[Optional[str], Optional[str]]:
        """Call planner model. Returns (plan_text_or_None, decision_or_None).

        ``env`` is the OSWorld DesktopEnv handle, threaded down so the
        stuck-recovery search can read error evidence directly via
        ``env.controller``.

        ``mode`` selects which planner prompt to build:
          - "initial"        : step==0 — build_prompt_initial_plan
          - "force_replan"   : known-stuck → build_prompt_force_replan
            with stuck context. The reason is read from
            ``self._force_replan_reason`` (staged by
            ``_decide_planner_mode_and_reason`` or by
            ``_check_done_acceptance`` when DONE was rejected).
          - "progress_check" : default — _pa_build_subgoal_check_prompt
            does its normal CONTINUE/REPLAN/DONE dispatch.
        """
        screenshot_bytes = obs["screenshot"]
        processed_image = process_image(screenshot_bytes)

        # Update planner history
        if self.planner_history and self.planner_history[-1][1] is None and self.current_subgoal_feedbacks:
            self.planner_history[-1] = (self.planner_history[-1][0], self.current_subgoal_feedbacks[-1])
        self.planner_history.append((processed_image, None))
        if len(self.planner_history) > self.planner_max_images:
            self.planner_history = self.planner_history[-self.planner_max_images:]

        if mode == "initial":
            system_content = self._compose_system_prompt(
                "You are a helpful assistant that generates concise, "
                "actionable step-by-step plans for desktop GUI tasks. "
                "You can see the current screen state to make informed plans."
            )
            planning_prompt = self._pa_build_initial_plan_prompt(
                instruction, retrieved_knowledge=None,
            )
            k = self.planner_max_images
        else:
            # The list of decisions the planner is allowed to emit this
            # turn (CONTINUE / REPLAN / DONE) is determined by which
            # situation builder LAYER 5 dispatches to — progress_check
            # disallows DONE, force_replan disallows CONTINUE,
            # done_rejected disallows both DONE and CONTINUE, etc. The
            # system prompt does NOT enumerate decisions; it points the
            # planner at LAYER 5's "Decision semantics" block, which is
            # built dynamically with the right allow_* flags.
            system_content = self._compose_system_prompt(
                "You are the planner for a desktop GUI task agent. Each "
                "turn you read the LAYERED context below and emit the "
                "output tags described in LAYER 5's \"Output format\" + "
                "\"Decision semantics\" blocks. The exact set of decisions "
                "available THIS turn is whichever bullets appear under "
                "\"Decision semantics — pick exactly one\" in LAYER 5; "
                "do not invent a decision that is not listed there.\n\n"
                + _LAYERED_PROMPT_GUIDE
            )
            planning_prompt = self._pa_build_subgoal_check_prompt(
                instruction, obs=obs, env=env,
                force_replan=(mode == "force_replan"),
            )
            k = 3

        entries = self.planner_history[-k:] if k > 0 and len(self.planner_history) > k else list(self.planner_history)
        a11y_text = _linearize_obs_a11y(obs)
        url_after = _extract_active_url_from_a11y(a11y_text)
        snap = getattr(self, "_current_snapshot", None)

        messages = [{"role": "system", "content": system_content}]

        messages = self._assemble_planner_messages(
            mode, planning_prompt, entries, snap, url_after,
            a11y_text, messages)

        # Debug dump: persist what the planner LLM is about to see.
        # Active only when use_perceiver is on (otherwise the planner is
        # in pure raw-obs mode and existing dumps already cover it).
        if getattr(self, "use_perceiver", False):
            try:
                from mm_agents.structagent.perception import perceiver_debug
                perceiver_debug.dump_planner_messages(
                    results_dir=getattr(self, "_results_dir", None),
                    step_idx=step_idx,
                    messages=messages,
                )
            except Exception as e:
                if logger:
                    logger.info("[planner-msg-dump] failed: %s", e)

        result = self._planner_call_retry_parse(messages, step_idx, mode)
        if mode != "initial":
            self._pending_planner_verifier_report = None
        return result

    def _assemble_planner_messages(self, mode, planning_prompt, entries,
                                   snap, url_after, a11y_text, messages):
        """Assemble the planner LLM message payload: initial flat path or
        the LAYERED (1-5) progress_check/force_replan path. Returns the
        completed messages list. Extracted verbatim from _pa_call_planner."""

        # ============================================================
        # INITIAL-PLAN PATH — legacy flat assembly. The layered
        # structure below only restructures the progress_check /
        # force_replan turns where the cross-block contradiction bugs
        # (e.g. "Required Outcomes N/N verified" vs. "Remaining
        # subgoals non-empty") actually manifested.
        # ============================================================
        if mode == "initial":
            for past_img_b64, _ in entries[:-1]:
                past_url = (past_img_b64 if past_img_b64.startswith("data:image")
                            else f"data:image/png;base64,{past_img_b64}")
                messages.append({"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": past_url}},
                ]})
            if entries:
                last_img_b64, _ = entries[-1]
                url = (last_img_b64 if last_img_b64.startswith("data:image")
                        else f"data:image/png;base64,{last_img_b64}")
                messages.append({"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                ]})
            self._planner_url_at_last_turn = url_after
            self._planner_a11y_at_last_turn = a11y_text
            messages.append({"role": "user", "content": [
                {"type": "text", "text": planning_prompt}]})
        else:
            # ============================================================
            # LAYERED PATH (progress_check / force_replan / actor_exhausted /
            # all_subgoals_done / subgoal_done — whichever was dispatched
            # inside _pa_build_subgoal_check_prompt). The dispatcher
            # already selected the right LAYER 5 contract; LAYERS 1-4
            # are mode-agnostic.
            # ============================================================

            # ---------- LAYER 1 — Task constraint ----------
            if self.ledger is not None:
                messages.append({"role": "user", "content": [{"type": "text", "text": (
                    "[LAYER 1 — Task constraint] — what success looks like.\n"
                    "Authoritative record of the task instruction, the "
                    "initial-context probe, and the per-outcome verify "
                    "ledger. The verify spec for each outcome was authored "
                    "by a separate observer and may be weaker than what "
                    "the task literally requires — when in doubt apply "
                    "LAYER 5 rule (6) / CONFLICT RESOLUTION (a).\n\n"
                    f"{self.ledger.to_prompt_block()}"
                )}]})

            # ---------- LAYER 2 — Strategy history ----------
            v2_active = (self._memory_version != "v1"
                         and getattr(self, "timeline", None)
                         and len(self.timeline) > 0
                         and getattr(self, "_strategy_history", None))
            if v2_active:
                from mm_agents.structagent.ledger.core.timeline import (
                    render_timeline_grouped_by_strategy,
                )
                tl_text = render_timeline_grouped_by_strategy(
                    self.timeline,
                    self._strategy_history,
                    n_recent_spans=3,
                    bullet_cap_per_span=5,
                    curator_strategy_notes=getattr(
                        self, "_last_curator_strategy_notes", None),
                )
                if tl_text:
                    messages.append({"role": "user", "content": [{"type": "text", "text": (
                        "[LAYER 2 — Strategy history] — high-level "
                        "approaches the planner has ALREADY tried and "
                        "abandoned, plus the curator's verdict where "
                        "available. Do NOT loop back to any of these "
                        "without first declaring in <observations> what "
                        "is structurally different about your new "
                        "approach (CONFLICT RESOLUTION (c)).\n\n"
                        f"{tl_text}"
                    )}]})

            # ---------- LAYER 3 — Subgoal state ----------
            messages.append({"role": "user", "content": [{"type": "text", "text": (
                self._render_layer3_subgoal_state()
            )}]})

            # ---------- LAYER 4a — image attachments ----------
            # K screenshots (oldest → newest). Image messages are
            # wordless — they go BEFORE the LAYER 4 text block so the
            # text can reference "the latest screenshot above".
            for past_img_b64, _ in entries[:-1]:
                past_url = (past_img_b64 if past_img_b64.startswith("data:image")
                            else f"data:image/png;base64,{past_img_b64}")
                messages.append({"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": past_url}},
                ]})
            if entries:
                last_img_b64, _ = entries[-1]
                url = (last_img_b64 if last_img_b64.startswith("data:image")
                        else f"data:image/png;base64,{last_img_b64}")
                messages.append({"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                ]})

            # ---------- LAYER 4b — evidence text ----------
            last_turn_delta_block: Optional[str] = None
            if snap is None and (self._planner_a11y_at_last_turn is not None
                                  or self._planner_url_at_last_turn is not None):
                try:
                    from mm_agents.structagent.ledger.core.summarizer import compute_window_delta
                    last_turn_delta = compute_window_delta(
                        url_before=self._planner_url_at_last_turn,
                        url_after=url_after,
                        a11y_before=self._planner_a11y_at_last_turn,
                        a11y_after=a11y_text,
                    )
                    last_turn_delta_block = last_turn_delta.to_prompt_block()
                except Exception as e:
                    if logger:
                        logger.warning("[Planner] last-turn delta failed: %s", e)

            recent_actor_actions = list(self.actions[-6:] if getattr(self, "actions", None) else [])
            messages.append({"role": "user", "content": [{"type": "text", "text": (
                self._render_layer4_evidence_text(
                    snapshot=snap,
                    a11y_text=a11y_text if snap is None else None,
                    last_turn_delta_block=last_turn_delta_block,
                    last_cli_run=getattr(self, "_last_cli_run_output", None),
                    recent_actor_actions=recent_actor_actions,
                    stuck_category=getattr(self, "_dump_stuck_category", None),
                    stuck_reason=getattr(self, "_dump_stuck_reason", None),
                )
            )}]})

            # Snapshot a11y/url for next turn's delta computation.
            self._planner_url_at_last_turn = url_after
            self._planner_a11y_at_last_turn = a11y_text

            # ---------- LAYER 5 — Decision contract ----------
            # planning_prompt already encodes which decisions are
            # allowed this turn (via the situation-specific builder's
            # allow_done / allow_continue / allow_replan kwargs to
            # _output_contract). The dead-end mandatory check + the
            # conflict-resolution table are appended afterward.
            layer5_parts: List[str] = [
                "[LAYER 5 — Decision contract] — your output rubric, "
                "decision rules, and CONFLICT RESOLUTION table. Apply "
                "the rules below using only the facts in LAYERS 1-4. "
                "The \"Decision semantics — pick exactly one\" section "
                "below lists which decisions are emittable THIS turn.",
                planning_prompt,
            ]
            if self.ledger is not None and len(self.ledger.failed_paths) > 0:
                layer5_parts.append(
                    "MANDATORY DEAD-END CHECK — before emitting <plan>:\n"
                    "  For EACH entry in [Failed paths] (LAYER 1), write "
                    "one sentence stating why your next proposed step is "
                    "NOT a repeat of that dead-end. If you cannot state a "
                    "clear structural difference, you MUST pick a "
                    "different approach.\n"
                    "  If a dead-end entry says an outcome was blocked by "
                    "a specific cause (e.g. 'Dates are Required error', "
                    "'CAPTCHA', 'validation banner'), your next step MUST "
                    "address that root cause FIRST — not retry the "
                    "blocked action."
                )
            layer5_parts.append(_CONFLICT_RESOLUTION_TABLE)
            messages.append({"role": "user", "content": [{"type": "text", "text": (
                "\n\n".join(layer5_parts)
            )}]})
        return messages

    def _planner_call_retry_parse(self, messages, step_idx, mode):
        """Call the planner model with up to 3 attempts, parse each response,
        stash self.planner_* state, and return (plan, decision). Extracted
        verbatim from _pa_call_planner."""
        def _planner_call_once(_msgs):
            return self.call_llm(
                {"model": self.model, "messages": _msgs,
                 "max_tokens": 2000, "top_p": self.top_p,
                 # bBoN diversity knob. ``self.temperature`` is 0.0 everywhere
                 # except live best-of-N rollouts (run_bbon.py sets ~0.7), so at
                 # the default this is byte-identical to before. The planner
                 # decision is bBoN's main diversity source: different plans ->
                 # different trajectories to choose between.
                 "temperature": self.temperature},
                self.model,
            ) or ""

        planner_response = None
        parsed: Dict[str, Any] = {"plan": None, "decision": None,
                                   "last_subgoal_done": None,
                                   "evidence": None, "feedback_to_actor": None}
        for _attempt in range(3):
            import time as _time
            _t0 = _time.time()
            planner_response = _planner_call_once(messages)
            _elapsed = _time.time() - _t0
            parsed = self._pa_parse_planner_response(planner_response or "")
            # A8 debug — dump every planner LLM call (input + response) to
            # ``{results_dir}/planner_debug/`` as a self-contained HTML
            # page. Index updated on each dump so the debug tree is usable
            # mid-run. Failures here never raise.
            try:
                from mm_agents.structagent.core.planner.debug import (
                    dump_planner_call as _dump_planner_call,
                )
                _dump_planner_call(
                    results_dir=getattr(self, "_results_dir", None),
                    step_idx=step_idx,
                    mode=mode,
                    attempt=_attempt,
                    messages=messages,
                    response=planner_response,
                    metadata={
                        "stuck_category": getattr(self, "_dump_stuck_category", None),
                        "stuck_reason": getattr(self, "_dump_stuck_reason", None),
                        "parsed_decision": parsed.get("decision"),
                        "parsed_plan_head": (parsed.get("plan") or "")[:2000],
                        "elapsed_s": _elapsed,
                    },
                )
            except Exception as e:
                if logger:
                    logger.info("[planner_debug] dump call failed: %s", e)
            if step_idx == 0:
                if parsed["plan"] is not None:
                    break
            elif parsed["decision"] == "REPLAN":
                if parsed["plan"] is not None:
                    break
            elif parsed["decision"] is not None:
                break
        # A8 debug: clear the stuck_category stash so the next planner
        # turn (which may be progress_check / initial / etc.) starts
        # without stale recovery context bleeding into its HTML dump.
        self._dump_stuck_category = None
        self._dump_stuck_reason = None
        self.planner_response = planner_response
        # At step 0 (mode=initial), the LLM is asked for a fresh <plan>
        # with no <decision> tag. The parser fills in decision="REPLAN"
        # as a fallback (see ``_pa_parse_planner_response``), but that
        # is a parser-only inference, NOT a real REPLAN event. Blank it
        # so the runtime log, trajectory HTML, and REPLAN-cadence
        # counters (rule #6 of ``_decide_planner_mode_and_reason``) don't
        # treat the initial plan emission as a back-tracking REPLAN.
        if step_idx == 0 and parsed.get("decision_inferred"):
            parsed["decision"] = None
        self.planner_decision = parsed["decision"]
        # Expose the full parsed record for _pa_predict's control flow.
        self.planner_parsed = parsed
        if logger:
            logger.info("[PA-Planner] step=%d response=%s", step_idx, (planner_response or ""))
        return parsed["plan"], parsed["decision"]


    # T4: write the disk plan snapshot after every plan commit.
    # See plan file Part I.3.
    def _write_plan_snapshot(self, step_idx: int) -> None:
        """Render the current Outcome DAG + subgoal state as markdown
        and write to ``<results_dir>/plan.md`` (atomic).

        Called from each plan-commit site (step==0 install + REPLAN
        swap). Best-effort: any failure is logged and swallowed —
        plan_store is a debug artifact, not in the control path.
        """
        from mm_agents.structagent.core.planner.plan_store import (
            render_plan_md, write_plan,
        )
        ledger = getattr(self, "ledger", None)
        if ledger is None:
            return
        results_dir = getattr(self, "_results_dir", None)
        if not results_dir:
            return
        try:
            focus = ledger.next_focus_outcome() if hasattr(
                ledger, "next_focus_outcome") else None
            focus_id = focus.id if focus is not None else None
            md = render_plan_md(
                ledger,
                instruction=getattr(self, "instruction", "") or "",
                current_focus_id=focus_id,
                current_subgoal=getattr(self, "current_subgoal", None),
                step_idx=step_idx,
                # Planner-side state — the per-turn subgoal decomposition
                # the planner emitted via <plan>. Distinct from the
                # ledger's Outcome DAG (milestones). Without these, the
                # disk plan.md only showed milestones and looked
                # incomplete during triage.
                plan_strategy=getattr(self, "current_plan_strategy", None),
                subgoal_queue=list(
                    getattr(self, "subgoal_queue", []) or []),
                subgoal_expected_post_states=list(
                    getattr(self, "subgoal_expected_post_states", []) or []),
                abandoned_subgoals=list(
                    getattr(self, "abandoned_subgoals", []) or []),
            )
            # step_idx pinned so the per-step history file at
            # plans/step_NNNN.md preserves this revision (diffable
            # vs prior REPLAN snapshots). plan.md still gets the
            # latest. See plan file Part I.3.
            write_plan(results_dir, md, step_idx=step_idx)
        except Exception as e:
            if logger:
                logger.warning(
                    "[plan_store] snapshot write failed at step %d: %s",
                    step_idx, e)
