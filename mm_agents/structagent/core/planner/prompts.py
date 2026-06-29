"""Per-situation prompt builders for the planner-actor agent.

One focused builder per planner decision situation instead of one
monolithic prompt.
"""

from typing import List, Optional



def _build_common_context(
    instruction: str,
    current_plan: Optional[str],
    completed_subgoals: List[str],
    current_subgoal: Optional[str],
    current_subgoal_step: int,
    subgoal_queue: List[str],
    domain: Optional[str] = None,
) -> str:
    """LAYER 5 prompt context: task instruction + read-only current plan.

    Subgoal progress, recent actions, and the Facts block belong to
    LAYER 3 and are rendered upstream; LAYER 5 references them, not
    restates them. The subgoal kwargs are unused here, kept only so the
    5 builders share one signature.
    """
    _ = (completed_subgoals, current_subgoal,
         current_subgoal_step, subgoal_queue)  # LAYER 3 owns these
    return f"""Task: {instruction}

Current Plan (read-only — already committed to the runtime):
{current_plan or '(no plan yet)'}"""


# --------------------------------------------------------------------------- #
# Unified output-contract helpers (Phase 2: subgoal contract + folded check)
# --------------------------------------------------------------------------- #

def _plan_format_block(domain: Optional[str] = None) -> str:
    """<plan> format: one <strategy> + one or more <step>s.

    The <strategy> drives dead-end tracking: REPLANning with a different
    strategy auto-records the old one as a failed path; same strategy is
    treated as tactical refinement, not abandonment. Strategy is the
    route/method class, not a tactic or a state.
    """
    calc_data_layout_tag = ""
    calc_step_qs = ""
    synth_step_qs = ""
    synth_eligibility_tag = ""
    dom_lower = (domain or "").lower()
    if dom_lower == "libreoffice_calc":
        calc_data_layout_tag = (
            "  <data_layout>\n"
            "    <q1_row_semantics>...</q1_row_semantics>\n"
            "  </data_layout>\n"
        )
        calc_step_qs = (
            "    <q2_task_pattern>(letter) calc_*_action for THIS step</q2_task_pattern>\n"
            "    <q3_output_target>sheet + range for THIS step's output</q3_output_target>\n"
            "    <q4_action_spec>pattern-specific kwargs for THIS step (see DATA_LAYOUT block above)</q4_action_spec>\n"
        )
    elif dom_lower == "libreoffice_impress":
        calc_data_layout_tag = (
            "  <data_layout>\n"
            "    <q1_directives>\n"
            "      PART 1 — slide inventory (1 line per relevant slide).\n"
            "      PART 2 — directives (D1..DN, one per row, each is\n"
            "      verb / property / verbatim target_phrase from the\n"
            "      instruction). One directive ⇒ exactly one <step>.\n"
            "    </q1_directives>\n"
            "  </data_layout>\n"
        )
        calc_step_qs = (
            "    <q2_task_pattern>(letter) impress_*_action for THIS step</q2_task_pattern>\n"
            "    <q3_target_binding>translate THIS directive's target_phrase from Q1 PART 2 into one concrete selector (placeholder / shape / shape_index / scope) + the property name being set</q3_target_binding>\n"
            "    <q4_action_spec>pattern-specific kwargs for THIS step (see DATA_LAYOUT block above)</q4_action_spec>\n"
        )

    dl_header = ""
    if calc_data_layout_tag:
        dl_header = (
            " (libreoffice_calc / libreoffice_impress tasks MUST prepend "
            "a <data_layout> block carrying Q1 only — Q2/Q3/Q4 live "
            "INSIDE each <step> so every action runs its emission "
            "protocol independently; see DATA_LAYOUT block above)"
        )
    elif synth_eligibility_tag:
        dl_header = (
            " (this task's domain has synth-tool shortcuts active — "
            "MUST prepend a <shortcut_eligibility> block deciding "
            "per-tool applies=yes|no with reason, BEFORE <strategy>. "
            "Strategy MUST reflect any yes decisions)"
        )
    return f"""Plan MUST start with one <strategy> tag, then one or more <step>s{dl_header}:
{calc_data_layout_tag}{synth_eligibility_tag}  <strategy>One sentence describing the HIGH-LEVEL APPROACH this plan implements — i.e. the path / method / route used to reach the goal. Strategy ≠ tactic. GOOD: "Use the catalog site's main-page search box to query 'lamp'", "Navigate via top nav Products link", "Use direct URL navigation". BAD: "Click magnifying glass" (that's a tactic), "The lamp listing is loaded" (that's a state).</strategy>
  <step target="outcome_id">
{calc_step_qs}{synth_step_qs}    <text>One-line subgoal text that a GUI actor can complete in 1-3 actions.</text>
    <expected_post_state>Concrete visible evidence on screen after success — e.g. "URL changes to /results", "rename text input with editable cursor appears on 'New folder' label", "toggle labeled 'Do Not Track' is in the ON position". Describe what would convince a human observer that THIS specific step succeeded — NOT the whole task.</expected_post_state>
  </step>

★ ``target`` (REQUIRED on every <step>): the SINGLE outcome id from
  the [Required Outcomes] block that THIS step is meant to flip to
  [verified]. The runtime uses ledger outcome state — not your
  ``expected_post_state`` self-check — to decide when to advance to
  the next step. This avoids self-loops on unobservable post-states
  (e.g. "URL is on clipboard" can never be visually verified, so a
  self-check loops forever).

  RULES:
  • Use an id that LITERALLY appears in [Required Outcomes]. Do NOT
    invent ids; match snake_case exactly.
  • One step ⇒ exactly ONE target outcome. If your micro-action
    would actually drive two outcomes at once, name the most
    DIRECTLY responsible one and let the other verify on its own
    next turn (the verifier checks all outcomes every turn).
  • If a step is a pure micro-action with NO directly corresponding
    outcome (e.g. "scroll down to see more rows"), set target="" —
    runtime falls back to your ``expected_post_state`` self-check.
  • Do NOT emit a step whose target is ALREADY [verified] in this
    turn's ledger — that step is obsolete. Skip straight to the
    first outcome still pending."""


def _plan_quality_rules(domain: Optional[str] = None) -> str:
    """Corrector rules merged into every plan-generating prompt, so the
    plan comes out pre-corrected and no separate `_correct_plan` VLM
    call is needed.

    ``domain`` picks the rule-#4 action vocabulary: LibreOffice domains
    get the matching structured-action set (calc_* / impress_* /
    writer_*, the canonical path since raw UNO Python isn't available);
    everything else gets GUI + cli_run only.
    """
    if (domain or "").lower() == "libreoffice_calc":
        rule_4 = ('4. Each step must be ONE of: (a) a concrete action a '
                  'GUI actor completes in 1-3 low-level actions (click, '
                  'type, scroll); OR (b) a single ``cli_run`` shell '
                  'command on the VM (sed, gsettings, dconf, pip, file '
                  'copy, app launch); OR (c) a single ``calc_*`` JSON '
                  'action against the OPEN spreadsheet — the CANONICAL '
                  'path for libreoffice_calc. Full action schema is in '
                  'the domain knowledge / data_layout block above; the '
                  '<text> SHOULD name the action + kwargs (e.g. "Use '
                  'calc_fill_column on Sheet1 col F rows 10..23 with '
                  'formula_template ``=VLOOKUP(E{r};$D$2:$E$7;2;1)``"); '
                  'the actor builds the JSON deterministically.')
    elif (domain or "").lower() == "libreoffice_impress":
        rule_4 = ('4. Each step must be ONE of: (a) a concrete action a '
                  'GUI actor completes in 1-3 low-level actions (click, '
                  'type, scroll); OR (b) a single ``cli_run`` shell '
                  'command on the VM (sed, gsettings, dconf, pip, file '
                  'copy, app launch); OR (c) a single ``impress_*`` JSON '
                  'action against the OPEN presentation — the CANONICAL '
                  'path for libreoffice_impress. Full action schema is '
                  'in the domain knowledge / data_layout block above; '
                  'the <text> SHOULD name the action + kwargs (e.g. '
                  '"Use impress_set_text_color with slide=1, '
                  'placeholder=\'title\', color=\'#FF0000\'"); the '
                  'actor builds the JSON deterministically. PREFER '
                  'impress_* over GUI clicks whenever a structured '
                  'action covers the operation (set text, font, color, '
                  'paragraph alignment, insert shape / image / chart / '
                  'table, background, notes, transition, slide '
                  'structural ops, export).')
    elif (domain or "").lower() == "libreoffice_writer":
        rule_4 = ('4. Each step must be ONE of: (a) a concrete action a '
                  'GUI actor completes in 1-3 low-level actions (click, '
                  'type, scroll); OR (b) a single ``cli_run`` shell '
                  'command on the VM (sed, gsettings, dconf, pip, file '
                  'copy, app launch); OR (c) a single ``writer_*`` JSON '
                  'action against the OPEN document — the CANONICAL '
                  'path for libreoffice_writer. Full action schema is '
                  'in the domain knowledge block above; the <text> '
                  'SHOULD name the action + kwargs (e.g. "Use '
                  'writer_alignment with alignment=\'center\', '
                  'target_indices=[0]"); the actor builds the JSON '
                  'deterministically. PREFER writer_* over GUI clicks '
                  'whenever a structured action covers the operation '
                  '(paragraph / character format, structure, content, '
                  'find-replace, image, export).')
    else:
        rule_4 = ('4. Each step must be ONE of: (a) a concrete action a '
                  'GUI actor completes in 1-3 low-level actions (click, '
                  'type, scroll); OR (b) a single ``cli_run`` shell '
                  'command on the VM (sed, gsettings, dconf, pip, file '
                  'copy, app launch).')
    return f"""Plan quality rules — apply these while writing the <plan>:
1. MERGE sequential interactions into a single step: "Click field" + "Type text" → "Click the field and type 'text'"; "Click dropdown" + "Select option" → "Click the dropdown and select 'option'".
2. Omit unnecessary verification / confirmation steps ("Wait for page to load", "Verify the result", "Confirm the action").
{rule_4}
4. If the latest screenshot shows any popup / modal / cookie banner / overlay that blocks the main UI, your step 1 MUST be to dismiss it before the task-specific steps.
5. DO NOT re-introduce steps already marked as completed.
6. DEFAULT VALUE PRESERVATION. When a form / dialog field is pre-populated (by the browser, OS, or upstream context) with a value that satisfies the task, DO NOT overwrite it. Type into a field only when (a) the instruction explicitly asks for a specific value, OR (b) the field is empty AND must be filled to proceed.
7. PLAN MUST MATCH OUTCOME STATE. Cross-reference EVERY `<step>` you write against the [Required Outcomes] block in your context:
   • DO NOT emit any step whose effect would re-produce an outcome already marked **[verified]** — that outcome is done; redoing it wastes turns AND can REVERT it (e.g. typing a different value into a cell that already has the right value, or re-creating a resource and triggering a name-collision error that breaks the verified state).
   • Your FIRST `<step>` MUST target the ▶ FOCUS ◀ outcome's evidence (the hint line shows what to produce).
   • Subsequent steps may target [REVERTED] / [pending] / [blocked] outcomes in dependency order. Skip any outcome whose row begins with `[verified]`.
   • Your `<strategy>` text must describe ONLY the remaining work; do not re-mention already-verified outcomes.
   • If your screenshot read disagrees with [Required Outcomes], TRUST [Required Outcomes] — its [verified] states come from deterministic verifies that ran THIS turn.
8. NO BLIND CONCRETE ANCHORS. Until the target app's state (structured block or screenshot) is actually in your context, write each step SEMANTICALLY — describe WHAT to act on and WHY, not the exact range / coordinate / count / literal value. Guessed locators become frozen wrong targets that contaminate every later step.
   GOOD <text>: "Copy the header row plus the data row(s) the task asks for."
   BAD  <text>: "Copy range A1:D10." (the locator is a pre-observation guess)
   Once the relevant state IS in context, concrete anchors are EXPECTED — read them verbatim, never from assumption."""


def _assessment_block(domain: Optional[str] = None) -> str:
    """Domain-agnostic ``<last_subgoal_assessment>`` skeleton.

    Per-domain evidence-channel rules come from
    ``get_evidence_guide`` and are embedded in ``<evidence>``. This
    block is emitted every planner turn, so those rules must ride along
    here — they can't live in the step-0-only ``<domain>.md``.
    """
    try:
        from mm_agents.structagent.domain import get_evidence_guide
        guide = get_evidence_guide(domain)
    except Exception:
        guide = ""
    guide_block = f"\n{guide}\n" if guide else ""
    return f"""<last_subgoal_assessment>
  <evidence>One short sentence citing CONCRETE evidence that the previous subgoal's expected_post_state is — or is not — satisfied. Read it from the channel appropriate to this domain (guide below). A [verified] outcome from the ledger is authoritative ONLY when its verify kind is deterministic (file_grep / url_match / shell_command / calc_verify / impress_verify — each re-runs a real check against live state every turn); an a11y_match-verified outcome comes from a vision-LLM judge that CAN hallucinate — independently confirm before relying on it. Cite the specific element / row / structure-field you read — never paraphrase intent.
{guide_block}  </evidence>
  <subgoal_anchor_check>One short sentence: if the current subgoal's text names a concrete anchor (range / index / count / name / coordinate / literal value) that the observed state — structured block or screenshot — CONTRADICTS, the OBSERVED STATE is authoritative. Such an anchor was likely guessed before the app was observed and is stale. In that case do NOT fault the actor for a perception-grounded result that matches reality: judge done-ness against the subgoal's SEMANTIC intent plus the observed state, treat that result as correct even when it differs from the subgoal's literal anchor, and note the stale anchor so the next plan can correct it. If the subgoal has no concrete anchor, or its anchor agrees with the observed state, write "no stale anchor".</subgoal_anchor_check>
  <contradiction_check>One short sentence: scan your own <evidence> above for any phrase that LITERALLY contradicts the expected_post_state — e.g. a Save / Apply / Confirm / Set-as-default action still pending or unclicked, a value still showing its PRE-change state, an input typed but not yet committed, or a validation / "Required" error referencing an unfilled field. If any such phrase appears in your evidence, you MUST set <done>NO</done>.</contradiction_check>
  <done>YES or NO</done>
</last_subgoal_assessment>

"""



def _output_contract(
    include_assessment: bool = True,
    allow_continue: bool = True,
    allow_done: bool = True,
    allow_replan: bool = True,
    allow_infeasible: bool = False,
    require_plan_on_replan: bool = True,
    domain: Optional[str] = None,
) -> str:
    """Unified output-format block appended to every planner prompt.

    Block order is observation → assessment → blocker → decision →
    action, each consuming the prior blocks. This keeps the planner from
    committing to a strategy that its own later reasoning contradicts —
    the failure mode of the removed <feedback_to_actor> / <dead_end_review>
    fields. <plan> is REPLAN-only; <actor_hint> is done=NO-only and must
    not repeat a [Failed paths] strategy.
    """
    decisions = []
    if allow_done:       decisions.append("DONE")
    if allow_continue:   decisions.append("CONTINUE")
    if allow_replan:     decisions.append("REPLAN")
    if allow_infeasible: decisions.append("INFEASIBLE")
    decisions_str = " | ".join(decisions)

    assessment_block = _assessment_block(domain) if include_assessment else ""

    plan_block = ""
    if require_plan_on_replan:
        plan_block = f"""

<plan>
{_plan_format_block(domain=domain)}
</plan>
[plan-emission rule — re-stated]: EMIT the <plan>...</plan> block ABOVE
ONLY when <decision> is REPLAN. When <decision> is CONTINUE or DONE,
SKIP the block ENTIRELY — do NOT output the opening <plan> tag, do NOT
output a <strategy>, do NOT output any <step>. The runtime ignores any
<plan> emitted alongside CONTINUE/DONE; emitting one wastes tokens and
confuses trajectory rendering. The current plan is already committed
in your read-only context — DO NOT re-emit it."""

    # <dead_end_review> was removed: the planner conflated "I'm
    # proposing X" with "X failed" (same representation, opposite
    # intent), logging untried strategies as dead-ends and inflating
    # observation_count. failed_paths now flows planner→Bottleneck→
    # cadence trigger (rule #6)→failed_paths instead.
    dead_end_block = """<Bottleneck>ONE sentence. Synthesize <observations> +
the [Failed paths] block in your context (if any) into the SINGLE
concrete blocker right now. Examples: "Color filter is not surfaced
anywhere on the category landing page", "the filter sidebar collapses
every time the actor clicks a product tile". If you are about to emit
DONE or CONTINUE-with-done=YES, write "none — task progressing".
</Bottleneck>"""

    actor_hint_block = """

<actor_hint>
  Emit ONLY if <done>NO</done> in <last_subgoal_assessment> (i.e. actor
  failed to make the subgoal visible-progress).

  ONE sentence of concrete guidance for the actor's next attempt — applies
  whether <decision> is CONTINUE (actor retries same subgoal) OR REPLAN
  (actor will work on the new plan's first subgoal; hint still useful for
  cross-cutting execution advice like "the typed text is in the box —
  press Enter to submit it").

  HARD CONSTRAINT: the hint MUST NOT match any strategy already listed
  in the [Failed paths] block in your context above. If your only
  available hint would repeat one of those failed strategies AND no
  fresh non-dead-end hint exists, you MUST switch <decision> to REPLAN
  AND omit this tag — the new <plan> is the recovery, not a hint.
</actor_hint>"""

    # Build the decision-semantics bullets dynamically so disallowed
    # decisions don't appear at all. Previously the DONE bullet showed
    # even with allow_done=False and the planner emitted DONE anyway,
    # causing DONE→rejected→defer→DONE loops past the ledger gate.
    decision_semantics_lines = ["Decision semantics — pick exactly one:"]
    if allow_done:
        decision_semantics_lines.append(
            "- DONE — claim the WHOLE task instruction is satisfied. Runtime EFFECT: terminates immediately at end-of-turn — no more actor steps, no chance to fix anything afterwards. By emitting DONE you are asserting, point by point, that ALL of these hold RIGHT NOW:\n"
            "    (1) every entry in the [Required Outcomes] block is [verified] — Summary reads \"N/N verified\", zero [pending] / [blocked] / [REVERTED] entries. Necessary but NOT sufficient: init_ledger may have written a verify spec that is structurally WEAKER than what the task literally requires (an a11y node that exists on a page BEFORE the user interacts with it can mark an \"X has been done\" outcome verified the moment the page loads). Cross-check against (3) and (6) before trusting this.\n"
            "    (2) <done>YES</done> in the assessment block (current subgoal verified by the latest evidence).\n"
            "    (3) The \"Remaining subgoals\" list in your context is empty.\n"
            "    (4) <Bottleneck> reads \"none — task progressing\".\n"
            "    (5) Your <observations> does NOT mention \"next step\" / \"remaining\" / \"still need to\" / \"should now\" — any such phrase contradicts the DONE claim.\n"
            "    (6) HARD CONTRADICTION GUARD — if [Required Outcomes] says N/N verified BUT \"Remaining subgoals\" is non-empty OR \"Abandoned subgoals\" lists a step that the plan still expected to run, the verify spec is too weak. DONE is FORBIDDEN; emit REPLAN whose first step targets whatever the unfinished subgoal was actually trying to produce (the URL change, the file write, the visible result) — do NOT trust the ledger's \"verified\" mark in that contradiction.\n"
            "  If ANY of (1)-(6) fails, DONE is FORBIDDEN. When (1) fails (non-verified outcomes remain), emit REPLAN whose first step produces evidence for the first ▶ FOCUS ◀ outcome — that pending outcome is exactly what is NOT yet done."
        )
    if allow_continue:
        decision_semantics_lines.append(
            "- CONTINUE — the current plan + subgoal queue stays intact. Two branches based on <done>:\n"
            "    (a) <done>YES</done> AND \"Remaining subgoals\" is non-empty → runtime POPS the next subgoal off the queue and the actor works on THAT next turn. Omit <actor_hint> (it would attach to the wrong subgoal).\n"
            "    (b) <done>NO</done> AND the strategy is still viable (no relevant [Failed paths] entry matches your intended next move) AND you can give a non-dead-end <actor_hint> → runtime RE-RUNS the actor on the SAME current subgoal with your hint attached.\n"
            "  In both branches another planner turn fires after the next actor batch — you will see updated state and can re-decide."
        )
    if allow_replan:
        decision_semantics_lines.append(
            "- REPLAN — current trajectory is dead: either (i) the current subgoal is blocked AND every hint you would give matches a dead-end in [Failed paths], or (ii) the plan itself is wrong even before hitting a dead-end (wrong app, wrong order, missing prerequisite). Runtime EFFECT: REPLACES the entire plan + subgoal queue with the fresh <plan> you emit, then runs the actor on the NEW first subgoal next turn. Pair with <done>NO</done>; the new <plan> must be consistent with what <Bottleneck> says is blocking. Optionally emit <actor_hint> ONLY for cross-cutting execution advice that survives the plan change (e.g. \"the typed text is already in the box — press Enter to submit it\")."
        )
    if allow_infeasible:
        decision_semantics_lines.append(
            "- INFEASIBLE — declare the WHOLE task genuinely impossible in this environment, emitting a terminal FAIL. Runtime EFFECT: same finality as DONE — the episode ends immediately — but it records the task as un-completable. This is a LAST RESORT, never an escape from a task that is merely hard, slow, tedious, or unfamiliar.\n"
            "  WHY a task can be GENUINELY infeasible (a real property of the task, NOT your failure to execute) — some tasks are designed to be impossible and the CORRECT answer is to recognise that and FAIL:\n"
            "    • Feature does not exist in THIS version: the menu item / setting / option / capability the instruction needs is absent — removed, renamed away, gated behind a build this install doesn't have, or it was never a real feature. App/website versions drift; an instruction written against another version can reference something that simply isn't here.\n"
            "    • Deliberately blocked by the environment: the control is permanently disabled / greyed-out by policy, a file or record the task needs was removed, or it requires an account / permission / network resource that is not available here.\n"
            "    • Self-contradictory or logically impossible: the request demands mutually exclusive conditions, or an object / state that cannot exist.\n"
            "  Emit INFEASIBLE when BOTH hold:\n"
            "    (1) You have already tried at least TWO structurally DIFFERENT approaches (visible in [Failed paths] / the \"Abandoned subgoals\" list) and every one dead-ended for the SAME root reason above.\n"
            "    (2) <Bottleneck> names one of the genuine causes above (missing feature / blocked-by-design / contradiction) — NOT a search failure. \"I can't find it yet\", \"the page is slow\", \"the click missed\", \"I'm not sure how\" are NOT infeasibility — they are REPLAN.\n"
            "  ABSENCE IS PROOF, not doubt: when you have checked the canonical locations where the feature/setting would live (its settings section, the documented menu path, the relevant dialog) and it is NOT there — and your abandoned attempts dead-ended on that same absence — that satisfies (1)+(2): declare INFEASIBLE. Do NOT invent ever-more-exotic places to keep searching for a control this environment does not have; \"maybe it exists somewhere I haven't looked\" is wishful thinking, not a fresh approach.\n"
            "  A fresh approach justifies REPLAN over INFEASIBLE only if it is structurally different from every abandoned one AND plausibly BYPASSES the named root cause. Re-searching yet another menu for an absent feature, retrying the same page, or cosmetic variations of a failed attempt do NOT count.\n"
            "  When the blocker is your own execution (an element you can see but cannot hit, a flaky page, a missed click) — emit REPLAN, never INFEASIBLE. Wrongly failing a doable task forfeits it entirely; that is far worse than spending more steps. Pair with <done>NO</done> and OMIT the <plan> block (you are declaring there is no recovery plan)."
        )
    decision_semantics_block = "\n".join(decision_semantics_lines)

    return f"""Output format — emit these tags in EXACTLY this order:

<observations>
  <visible>CURRENT SCREEN — the app / window / page in focus (URL + title for a browser; active slide/sheet/document region for an office app), plus the INTERACTIVE elements you can act on right now: buttons, menu items, dropdowns, inputs, toggles, links. List them with their CURRENT state (e.g. "dropdown 'Option A' set to 'Standard'", "button 'Apply' enabled"). Description only; no judgment.</visible>
  <unexplored>Content the task may need that is NOT in the current view. For scrolling UIs (web pages, long documents) this is below-the-fold regions — fill it in ONLY IF the task target is NOT among the interactive elements you just listed in &lt;visible&gt;. If the next move operates an element ALREADY in &lt;visible&gt;, or a STRUCTURE block in your context already enumerates the whole document, write "none — task target is in current view".</unexplored>
</observations>

{assessment_block}{dead_end_block}

<decision>{decisions_str}</decision>{plan_block}{actor_hint_block}

{decision_semantics_block}

CRITICAL tag-emission rules — violations break the runtime:
- ★ <plan> ONLY when <decision> is REPLAN. ON CONTINUE OR DONE you MUST OMIT the entire <plan>...</plan> block — no opening tag, no <strategy>, no <step>. The current plan is already in your read-only context; re-emitting it is forbidden and ignored. If you find yourself wanting to refine the plan mid-CONTINUE, that means your decision should be REPLAN, NOT CONTINUE.
- <actor_hint> ONLY when <done>NO</done>. NEVER on done=YES (subgoal already done — no hint needed). Allowed on either CONTINUE or REPLAN.
- <Bottleneck> always emitted (1 sentence). On DONE/CONTINUE+done=YES it must say "none — task progressing"."""


# ---------------------------------------------------------------------------
# Situation 1: Actor reported subgoal done (finished()). Focus: verify the
#   claim, then check whether the NEXT subgoal is still valid.
# ---------------------------------------------------------------------------

def build_prompt_subgoal_done(
    instruction: str,
    current_plan: Optional[str],
    completed_subgoals: List[str],
    current_subgoal: Optional[str],
    current_subgoal_step: int,
    subgoal_queue: List[str],
    domain: Optional[str] = None,
) -> str:
    """LAYER 5 decision contract for the subgoal-done turn.

    Subgoals live in LAYER 3; this just emits the eval rubric + output
    contract.
    """
    _ = (completed_subgoals, current_subgoal, current_subgoal_step,
         subgoal_queue)
    ctx = _build_common_context(instruction, current_plan, completed_subgoals,
                                current_subgoal, current_subgoal_step, subgoal_queue)

    eval_block = (
            "The actor REPORTED the current subgoal is finished — but a self-report is a CLAIM, "
            "NOT proof. First VERIFY the claim against the evidence appropriate to this domain "
            "(see the evidence guide inside the assessment block below — STRUCTURE block for "
            "structural document edits, a11y tree + screenshot for web/app state, command output "
            "for shell). Typed glyphs in a field, a partial visual change, or a dialog that merely "
            "opened are NOT by themselves proof the subgoal's target end-state was committed:\n"
            "0. VERIFY THE CLAIM — is the current subgoal's expected end-state ACTUALLY present in the evidence right now? If it is NOT, the actor's finished() was premature → assess done=NO and give an <actor_hint> naming exactly what is still missing. done=NO keeps the agent on the SAME subgoal (it is retried, NOT advanced) — stop here, do not run the checks below.\n"
            "1. OVERALL TASK CHECK — ONLY if step 0 confirmed the subgoal is genuinely complete (done=YES): is the Task's requested end-state ALREADY reached (named element created, value set, toggle flipped)? Check the [Required Outcomes] block in LAYER 1: if everything is [verified] AND LAYER 3 \"Remaining subgoals\" is empty → output DONE. Re-read LAYER 5 rule (6) before doing so.\n"
            "2. Does the current state match what the NEXT subgoal in LAYER 3 expects? (e.g. if the next subgoal says \"click X in the dropdown\", is that dropdown actually open?)\n"
            "3. Has the state changed in a way that makes remaining subgoals obsolete or incorrect (a dialog closed, navigated away, unexpected state)?\n"
            "4. Are any remaining subgoals no longer needed because they were accomplished as a side effect?"
        )

    return f"""{ctx}

{eval_block}

If step 0 found the subgoal NOT actually complete → assess done=NO + output CONTINUE; because done=NO, the SAME subgoal is retried (it is NOT advanced).
If the Task is already fully achieved AND LAYER 5 rule (6) is satisfied → output DONE.
If the subgoal is genuinely complete (done=YES) and the next subgoal is still valid → output CONTINUE to advance to it.
If the remaining plan no longer fits → output REPLAN with corrected steps.

{_plan_quality_rules(domain=domain)}

{_output_contract(include_assessment=True, domain=domain)}"""



# ---------------------------------------------------------------------------
# Situation 3: Actor exhausted (failed after max retries)
# ---------------------------------------------------------------------------

def build_prompt_actor_exhausted(
    instruction: str,
    current_plan: Optional[str],
    completed_subgoals: List[str],
    current_subgoal: Optional[str],
    current_subgoal_step: int,
    subgoal_queue: List[str],
    actor_actions: List[str] = None,
    domain: Optional[str] = None,
) -> str:
    """LAYER 5 decision contract for the actor-exhausted turn.

    Failed subgoal, completed list, recent actions, and stuck signal
    are rendered upstream in LAYER 3 / LAYER 4.
    """
    _ = (completed_subgoals, current_subgoal, current_subgoal_step,
         subgoal_queue, actor_actions)
    ctx = _build_common_context(instruction, current_plan, completed_subgoals,
                                current_subgoal, current_subgoal_step, subgoal_queue)

    diagnose_block = (
            "The actor FAILED to complete the current subgoal after multiple attempts (see LAYER 3 for which subgoal, LAYER 4 for the recent actor actions that failed). DIAGNOSE THE FAILURE, reading from the evidence source "
            "appropriate to this domain:\n"
            "1. For GUI actions — did anything change after each attempt? If NOTHING changed across multiple clicks, the target is likely GREYED OUT / DISABLED or the click missed. For structured actions (calc_* / impress_* / writer_*) — read the action's output block (returncode / error message) for the concrete failure.\n"
            "2. If a control appears greyed out / disabled, a PREREQUISITE was not met (text not selected, no item chosen, wrong mode, required field empty).\n"
            "3. Could a previously \"completed\" subgoal (see LAYER 3 Completed list) have ACTUALLY FAILED? (a drag-select that highlighted nothing, a click that missed, a navigation that didn't land, a structured edit that errored.)\n"
            "4. Is the actor targeting the wrong element? Compare what it acted on against what your context shows is actually there."
        )

    return f"""{ctx}

{diagnose_block}
You MUST output REPLAN. Your new plan MAY redo previously "completed" subgoals if they likely failed.

{_plan_quality_rules(domain=domain)}

Output format — emit these in order:
<last_subgoal_assessment>
  <done>NO</done>
  <evidence>One short sentence naming the concrete evidence (screen element, a11y row, STRUCTURE-block field, or command output) that shows the failed subgoal did not commit.</evidence>
</last_subgoal_assessment>
<root_cause>Why did the actor fail? Identify the specific reason (disabled element, missing prerequisite, wrong target, structured-action error, etc.). If a prerequisite was missing, identify which completed subgoal likely failed and needs to be redone.</root_cause>
<observations>
  <visible>Current screen — the app / window / page in focus plus INTERACTIVE elements with their state (dropdowns / buttons / inputs / toggles).</visible>
  <unexplored>Off-view regions ONLY if the task target is NOT among the visible interactive elements; otherwise write "none — task target is in current view".</unexplored>
</observations>
<decision>REPLAN</decision>
<plan>
{_plan_format_block(domain=domain)}
</plan>
<actor_hint>(optional) ONE sentence of concrete execution guidance for the actor's next attempt. Omit if the only available hint would repeat a strategy already in [Failed paths].</actor_hint>
<Bottleneck>What specific blocker the new plan addresses.</Bottleneck>"""


# ---------------------------------------------------------------------------
# Situation 4: All planned subgoals completed
# ---------------------------------------------------------------------------

def build_prompt_all_subgoals_done(
    instruction: str,
    current_plan: Optional[str],
    completed_subgoals: List[str],
    current_subgoal: Optional[str],
    current_subgoal_step: int,
    subgoal_queue: List[str],
    domain: Optional[str] = None,
) -> str:
    """LAYER 5 decision contract for the all-subgoals-done turn.

    Completed list is in LAYER 3, instruction in LAYER 1; this just
    emits the verification rubric + output contract.
    """
    _ = (completed_subgoals, current_subgoal, current_subgoal_step,
         subgoal_queue, current_plan)

    check_block = (
            "All planned subgoals have been verified as completed (see LAYER 3 Completed list). Now carefully check the task end-state against the LAYER 1 Required Outcomes + the literal Task instruction, "
            "reading from the evidence source appropriate to this domain "
            "(STRUCTURE block for document edits, a11y tree + screenshot for "
            "web/app state, command output for shell):\n\n"
            "1. For EACH requirement in the Task (LAYER 1), which completed subgoal (LAYER 3) fulfilled it? Is any requirement NOT addressed?\n"
            "2. Are there any:\n"
            "   - pending confirmation actions? (a \"Save\" / \"OK\" / \"Done\" / \"Apply\" / \"Set as default\" / \"Confirm\" control still visible and unclicked — note: structured document edits persist on their own and need no Save button)\n"
            "   - open dialogs or pending actions?\n"
            "   - error messages or warnings?\n"
            "3. Does the final state match what the task asked for?"
        )

    return f"""Task: {instruction}

{check_block}

If ALL requirements are confirmed met AND no confirmation actions remain AND LAYER 5 rule (6) holds → output DONE.
If ANY requirement is missing or a confirmation action is still pending → output REPLAN with the missing steps.
{_plan_quality_rules(domain=domain)}

{_output_contract(include_assessment=False, allow_continue=False, domain=domain)}"""


# ---------------------------------------------------------------------------
# Situation 5: Force replan (actor reported <Impossible>)
# ---------------------------------------------------------------------------

_STUCK_CATEGORY_GUIDANCE = {
    "done_rejected": (
        "STUCK CATEGORY: DONE rejected by ledger gate. The path you're "
        "on is LARGELY RIGHT — most required outcomes are already verified; "
        "only a SPECIFIC final outcome is missing. Look at the [Required "
        "Outcomes] block above for the [pending] outcome's evidence_hint, "
        "and emit a SHORT plan (often 1 step) that directly produces that "
        "evidence in the CURRENT view / on the CURRENT file.\n"
        "  • DO NOT propose a different path / entry point.\n"
        "  • DO NOT re-open a file or re-navigate to a view you are "
        "already on.\n"
        "  • DO NOT invent a new \"blocker\" (e.g. an unrelated menu / "
        "extra dialog / nested sub-tab) when the actual missing piece "
        "is a direct interaction with what's already visible.\n"
        "  • Map the pending outcome's evidence_hint to the smallest "
        "concrete action that produces it on the current state — the "
        "verifier looks for that specific evidence, not for ceremony "
        "around it."
    ),
    "actor_failure": (
        "STUCK CATEGORY: actor IMPOSSIBLE / clicks miss / no observable "
        "state change. The current path's entry point is broken — the "
        "target element may not exist where assumed, may be occluded, "
        "may be in a disabled state, or the wrong window may be active. "
        "Emit a STRUCTURALLY DIFFERENT plan:\n"
        "  • a different element identifier (alternate name / role / "
        "tag) that points at the same logical target\n"
        "  • a different access path (menu vs toolbar vs shortcut vs "
        "address-bar/command-palette/CLI vs context menu)\n"
        "  • for CLI / shell tasks: a different command form or "
        "different flags / quoting\n"
        "  • NOT a re-wording of the same action on the same element."
    ),
    "replan_cadence": (
        "STUCK CATEGORY: same strategy across N consecutive REPLANs. "
        "The high-level approach is fine but the SPECIFIC TACTICAL STEP "
        "is stuck. Emit an alternate tactic WITHIN the same strategy:\n"
        "  • a different element identifier on the same view (alternate "
        "name, differently-labeled affordance pointing at the same "
        "target)\n"
        "  • alternate coordinates / scroll-into-view first / focus the "
        "right window first\n"
        "  • keyboard shortcut targeting the same goal\n"
        "  • for CLI / shell tasks: same command intent, different "
        "syntax (e.g. switch from globbing to ``find -name``)\n"
        "  • DO NOT abandon the strategy unless every same-strategy "
        "tactic was already tried."
    ),
    "budget_exhausted": (
        "STUCK CATEGORY: subgoal step budget burned without verifiable "
        "progress. The current subgoal text + current strategy together "
        "are not producing the expected post-state. Either:\n"
        "  (a) keep the strategy but emit a different concrete subgoal "
        "(different element / tactic / command form), or\n"
        "  (b) abandon the strategy entirely if every tactic was tried.\n"
        "Inspect [Failed paths] and ``Abandoned subgoals`` to avoid "
        "re-emitting any of them."
    ),
    "queue_empty_no_done": (
        "STUCK CATEGORY: subgoal queue drained but task end-state NOT "
        "verified. The original plan finished its steps but didn't "
        "produce every required outcome. Look at the [Required Outcomes] "
        "block above for what's still [pending], and emit a CLOSING plan "
        "(usually 1-2 steps) that produces those outcomes from the "
        "current state. Do NOT restart the task; do NOT re-do "
        "already-verified outcomes."
    ),
}


def build_prompt_force_replan(
    instruction: str,
    current_plan: Optional[str],
    completed_subgoals: List[str],
    current_subgoal: Optional[str],
    current_subgoal_step: int,
    subgoal_queue: List[str],
    recent_actions: Optional[List[str]] = None,
    abandoned_subgoals: Optional[List[str]] = None,
    stuck_reason: str = "actor reported IMPOSSIBLE on the current subgoal",
    stuck_category: Optional[str] = None,
    retrieved_knowledge: Optional[str] = None,
    verifier_feedback: Optional[str] = None,
    domain: Optional[str] = None,
    l2_alternatives_block: Optional[str] = None,
    l1_alternatives_block: Optional[str] = None,
) -> str:
    """Force a REPLAN with rich failure context. Triggered when stuck:
    actor <Impossible>, DONE rejected by the ledger gate, subgoal budget
    exhausted, or repeated empty-delta actions / same-theme REPLANs.

    Asks for ONE new plan (not K candidates) but dumps the full failure
    context and forces a <stuck_analysis> block before the plan, so the
    model sees all evidence before picking the next approach.
    """
    # LAYER 5 (decision contract) only. LAYER 3/4 (planner.py) render the
    # subgoal lists, recent actions, and this turn's stuck signal +
    # retrieved_knowledge. The unused args stay for caller-API stability.
    #
    # ``verifier_feedback``: per-outcome verifier traces + framework
    # root-cause hypothesis, bundled at the force_replan dispatch site.
    # Gated by ``enable_stuck_diagnosis_injection`` (default OFF) — caller
    # passes None when off, keeping baseline behaviour byte-for-byte.
    # When non-empty it renders before <stuck_analysis> so the planner
    # sees the framework's hypothesis first.
    _ = (completed_subgoals, current_subgoal, current_subgoal_step,
         subgoal_queue, recent_actions, abandoned_subgoals,
         retrieved_knowledge)
    ctx = _build_common_context(instruction, current_plan, completed_subgoals,
                                current_subgoal, current_subgoal_step, subgoal_queue)

    verifier_feedback_section = ""
    if verifier_feedback and verifier_feedback.strip():
        verifier_feedback_section = (
            "========== FRAMEWORK VERIFIER + STUCK DIAGNOSIS ==========\n"
            "Per-outcome verifier traces and a framework root-cause LLM\n"
            "hypothesis follow. Use the DIAG rows (when present) as\n"
            "MECHANICAL ground truth — they name the actual cell /\n"
            "URL / a11y row the verify checked and what it saw vs\n"
            "expected. Use the ``Suggested next move`` as a starting\n"
            "point for your replan; if you disagree, explain why in\n"
            "<stuck_analysis>.\n\n"
            + verifier_feedback.strip() + "\n"
            "==========================================================\n"
        )

    # B2.A: corpus-retrieved alternatives (L1 = action variants for the
    # failing subgoal, L2 = alternative strategies for the task class).
    # Rendered upstream by planner.py, env-gated by
    # ENABLE_PLANNER_EXPERIENCE_MEMORY; empty on OOD retrieval or flag
    # off. Spliced after the diagnosis block to read in narrative order:
    # diagnose → L1 tactical alternatives → L2 strategic alternatives.
    memory_alternatives_section = ""
    _mem_parts: List[str] = []
    if l1_alternatives_block and l1_alternatives_block.strip():
        _mem_parts.append(l1_alternatives_block.strip())
    if l2_alternatives_block and l2_alternatives_block.strip():
        _mem_parts.append(l2_alternatives_block.strip())
    if _mem_parts:
        memory_alternatives_section = "\n\n".join(_mem_parts) + "\n"

    analysis_block = (
        "<stuck_analysis>\n"
        "  <assumption_wrong>One sentence. Which specific assumption about the UI / navigation did you make that the evidence contradicts? (e.g. \"I assumed 'Do Not Track' toggle lives directly under Privacy, but after scrolling 20× it is clearly not there — it must be nested under a sub-section\".)</assumption_wrong>\n"
        "  <why_recent_actions_failed>One sentence referencing the last actor action(s) + the observable delta: did the clicks land? did the URL change? did any relevant a11y element appear?</why_recent_actions_failed>\n"
        "</stuck_analysis>"
    )
    plan_constraints = (
        "Universal plan rules (apply on top of the STUCK CATEGORY "
        "guidance above):\n"
        "  • Do NOT re-word any entry in [Failed paths] (LAYER 1) or "
        "any subgoal listed under \"Abandoned subgoals\" (LAYER 3).\n"
        "  • First step must be concrete enough for the actor to "
        "execute in 1-3 low-level actions."
    )

    # Gate INFEASIBLE conservatively: offer it only after ≥2 distinct
    # approaches were abandoned (real effort spent), and never on a
    # done_rejected (that's a completion dispute, not impossibility).
    # When not offered, the contract omits INFEASIBLE so the planner
    # can't fail-early.
    allow_infeasible = (stuck_category != "done_rejected"
                        and len(abandoned_subgoals or []) >= 2)

    return f"""{ctx}
{verifier_feedback_section}
{memory_alternatives_section}
Before you write a new plan, you MUST diagnose. The stuck signal
(category + reason), recent actor actions, and abandoned subgoals are
in LAYER 4 / LAYER 3 above. Dead-end strategies are in LAYER 2. Use
them — do NOT repeat any of them.

{analysis_block}

{_STUCK_CATEGORY_GUIDANCE.get(stuck_category or "actor_failure", _STUCK_CATEGORY_GUIDANCE["actor_failure"])}

{_plan_quality_rules(domain=domain)}

{plan_constraints}

{(
    "DONE is NOT a valid decision this turn — the ledger gate just "
    "rejected a DONE claim because non-verified outcomes remain. "
    "You MUST emit REPLAN with a step that produces the missing "
    "outcome's evidence (see the [Required Outcomes] block in LAYER 1 "
    "for the ▶ FOCUS ◀ outcome's hint + last verify stdout)."
) if stuck_category == "done_rejected" else (
    "If the Task's end-state is ALREADY visible (on the latest screenshot or via terminal verify output) despite the stuck signal → output DONE instead, but ONLY if the [Required Outcomes] block in LAYER 1 shows ALL outcomes as [verified]; if any [pending] / [blocked] / [REVERTED] entry remains, emit REPLAN."
)}

{_output_contract(include_assessment=True, allow_continue=False, allow_done=(stuck_category != "done_rejected"), allow_infeasible=allow_infeasible, domain=domain)}"""


# ---------------------------------------------------------------------------
# Situation 6: Normal progress check (stuck too long / mid-subgoal)
# ---------------------------------------------------------------------------

def build_prompt_progress_check(
    instruction: str,
    current_plan: Optional[str],
    completed_subgoals: List[str],
    current_subgoal: Optional[str],
    current_subgoal_step: int,
    subgoal_queue: List[str],
    domain: Optional[str] = None,
) -> str:
    """Normal mid-subgoal evaluation — LAYER 5 decision contract only.

    Current subgoal, queue, completed/abandoned lists, and step counter
    live in LAYER 3 (rendered upstream); this just emits the decision
    rules + output contract.
    """
    _ = (completed_subgoals, current_subgoal, current_subgoal_step,
         subgoal_queue)
    ctx = _build_common_context(instruction, current_plan, completed_subgoals,
                                current_subgoal, current_subgoal_step, subgoal_queue)

    check_block = (
            "Evaluate progress, reading from the evidence source appropriate "
            "to this domain (see the evidence guide inside the assessment "
            "block below):\n"
            "1. Has the state moved TOWARD the current subgoal (see LAYER 3 for what subgoal is current, LAYER 4 for what the actor just produced)? Judge from the RIGHT source — for a structural document edit, the STRUCTURE block re-probe, NOT a screenshot diff (a correct structured-action edit can leave the screenshot visually unchanged, or the edited slide/sheet may not be on screen); for web/app state, the a11y tree + screenshot.\n"
            "2. Is there an unexpected popup, error dialog, or loading state blocking progress?\n"
            "3. Has the subgoal been achieved even though the actor didn't explicitly report it?\n"
            "4. Do the elements in your context match what the plan expects? Read actual labels / field names carefully.\n"
            "Only when the right evidence shows genuinely NO progress across multiple attempts is the current approach not working → REPLAN."
        )

    return f"""{ctx}

{check_block}
If the subgoal was achieved but task has more to do → output CONTINUE.
If progress is stalled or blocked → output REPLAN with a different approach.

{_plan_quality_rules(domain=domain)}

{_output_contract(include_assessment=True, allow_done=False, domain=domain)}"""
