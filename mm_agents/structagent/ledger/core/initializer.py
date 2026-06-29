"""Ledger initialization: one LLM call at task start.

Input: instruction + initial observation (screenshot + a11y tree + URL).
Output: a seeded ``ProgressLedger`` with ``initial_context`` and 2-6
``required_outcomes``.

Constraint: target derivation must NOT use ``example["config"]`` or
``example["evaluator"]`` — only the instruction and current screen, the
same information a human has on first glance.
"""
from __future__ import annotations
import base64
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from mm_agents.structagent.ledger.core.ledger import (
    InitialContext,
    Outcome,
    ProgressLedger,
    VerifySpec,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """You initialize a structured progress ledger for a GUI
automation agent. The agent runs inside an OSWorld benchmark VM. The
following baseline VM configuration is the same in every task — treat
it as known environment fact, not as part of any single task:

  • Linux user: ``user`` (HOME=/home/user). ``~`` expands to
    ``/home/user``.
  • Default sudo password for ``user`` is ``password``.
  • Open a terminal with ``ctrl+alt+t`` (gnome-terminal).

You see the user's task instruction and the agent's initial screen.
Your job is to read them and output:

  1. initial_context — what the agent is starting with (app, active URL,
     window title, visible open tabs). Read these off the screen and the
     provided a11y tree. If you cannot read a field, leave it null.

  2. required_outcomes — a small ORDERED chain of GATING MILESTONES that
     the agent must pass through to complete the task. NOT a checklist
     of every screen interaction; the planner breaks down the
     micro-steps. Each outcome has:
       - id: short stable snake_case identifier
       - description: one natural-language sentence
       - evidence_hint: concrete observable (see priority list below)
       - depends_on: list of outcome ids that must hold first; use
         this to express ordering ("filter requires being in the
         right section first")

╭─ MILESTONE vs WAYPOINT — the most important distinction ─────────╮
│                                                                   │
│ Test:  "If this state regressed, would the AGENT need to redo    │
│         OTHER work too — i.e., does it CASCADE downstream?"      │
│                                                                   │
│   YES  →  MILESTONE   (include as outcome)                        │
│   NO   →  WAYPOINT    (DROP — let the planner handle as subgoal) │
│                                                                   │
│ Examples:                                                         │
│   • "Navigate to Community section" — milestone                  │
│       (must be on /community to filter Latest; regression          │
│        cascades — you'd have to re-navigate AND re-filter)       │
│   • "Filter by Latest"               — milestone                  │
│       (sort order gates which thread you'd open; regression       │
│        cascades — you'd open the wrong thread)                   │
│   • "Open thread with most replies"  — milestone (final goal)    │
│   • "Search bar visible / focused"   — WAYPOINT                  │
│       (transient UI cue; doesn't gate downstream — search can     │
│        succeed regardless of current focus state)                 │
│   • "Privacy consent dialog dismissed" — WAYPOINT                │
│       (the underlying cookie/pref persists; if dialog re-appears, │
│        actor just dismisses again — doesn't undo other work)      │
│                                                                   │
│ Aim for 1–4 milestones. ONE is fine when the task is direct.     │
│ But every IMPERATIVE VERB PHRASE in the instruction must map     │
│ to AT LEAST ONE outcome. Do NOT collapse two verbs onto one      │
│ outcome unless they target the SAME artifact.                    │
│                                                                   │
│ ★ ENUMERATE — ONE OUTCOME PER CONCRETE DELIVERABLE ★              │
│                                                                   │
│ When the instruction names N specific deliverables (whether       │
│ literally enumerated — "the 4 years 2019, 2020, 2021, 2022" —    │
│ or implied with a fixed count — "the latest 5 emails", "three    │
│ machine learning conferences"), write ONE outcome per item, with │
│ the item's identifying token IN THE OUTCOME ID. Examples:        │
│   • Task "5 latest emails → report" →                            │
│       email_1_row_in_report, email_2_row_in_report, ...,         │
│       email_5_row_in_report                                       │
│   • Task "awards for 2019-2022" →                                │
│       award_2019_row, award_2020_row,                            │
│       award_2021_row, award_2022_row                             │
│   • Task "one researcher's row into a spreadsheet" →             │
│       researcher_row_appended (singular — one item)              │
│                                                                   │
│ ✗ FORBIDDEN — bulk outcome NAMES that hide multi-item work:       │
│       all_X_added         X_data_populated      X_extracted       │
│       X_collected         X_filled              X_rows_written    │
│       all_emails_added    award_data_entered    contacts_imported │
│ These wrap N items behind one verify, so the planner can't tell  │
│ "3/N done" from "0/N done". If you'd write `all_*` or `*_data`,  │
│ split into N specifically-named outcomes instead.                │
│                                                                   │
│ If the collection size is GENUINELY unknown at init (e.g. "all   │
│ papers in the folder" where you can't see the folder yet) and    │
│ no count appears in the instruction, fall back to ONE outcome     │
│ whose verify checks a file-level signal (xlsx exists with the    │
│ right columns + non-empty body, etc.). Still avoid `all_*` /      │
│ `*_data` names — pick something specific like                    │
│ `result_xlsx_populated_from_papers_folder`.                       │
╰───────────────────────────────────────────────────────────────────╯
<!-- [FIX imperative_coverage_bias] Added verb-coverage on top of
     milestone count. Old text was "Err on FEWER outcomes; the
     planner will fill in the gaps" — that caused init_ledger to
     silently drop critical imperatives when their verify was hard
     (move files into folder, edit code before running, …). Roll
     back by deleting the second sentence above. -->

╭─ EVIDENCE PRIORITY — write evidence_hint anchored on durable signal ╮
│                                                                       │
│ Rank evidence types from most-durable (top) to most-fragile           │
│ (bottom). ALWAYS use the highest-ranking one available; the lower     │
│ you go, the more the KeyNode detector will flap the outcome between   │
│ verified ↔ reverted just because the actor scrolled or navigated.    │
│                                                                       │
│   1. Persisted file / config / DB content                             │
│         e.g. "~/.config/X.cfg contains 'foo=bar'"                    │
│              "sqlite > SELECT name FROM users WHERE..."               │
│                                                                       │
│   2. URL path / query string                                          │
│         e.g. "URL contains '/community/latest'"                       │
│              "URL has '?sort=latest&category=X'"                     │
│                                                                       │
│   3. Stable DOM state (radio.checked, select.value, form values)     │
│         e.g. "radio-button 'Open NTP' has state=checked,enabled"     │
│              "input field labeled 'Site URL' has value 'X'"          │
│                                                                       │
│   4. AVOID — transient UI cues (currently visible / focused /         │
│      hovered / highlighted); these flap with every navigation.        │
│         e.g. AVOID "search bar is focused"                            │
│         e.g. AVOID "no dialog visible right now"                      │
│                                                                       │
│ GROUNDEDNESS OVERRIDE: rank (1) "Persisted file content" applies      │
│ only when you can QUOTE the exact key / pattern from the probe        │
│ output or domain knowledge block. If the target key name would be a   │
│ guess from training memory (no verbatim hit anywhere in your input),  │
│ DROP to rank (3) stable DOM/a11y state on the GUI control — a         │
│ guessed file pattern is false precision: it will pass only when the   │
│ actor literally echoes the guessed string into the file, which        │
│ diverges from what the app actually writes via the GUI.               │
│                                                                       │
│ NO SPECULATION in evidence_hint text. Never write "(likely X)" /      │
│ "(probably Y)" / "(perhaps Z)" / "(default is W)" for a value you     │
│ have not seen on screen / in the probe / in the instruction. Use a    │
│ placeholder like ``<default filename>`` / ``<chosen value>`` so the   │
│ planner knows to READ the value, not type a guessed string. A guessed │
│ literal in evidence_hint reads to the planner as authoritative and    │
│ leads to "rename to X" subgoals that overwrite the app's real default.│
╰───────────────────────────────────────────────────────────────────────╯

╭─ SUBSTRING-ANCHOR HAZARD ─────────────────────────────────────────────╮
│                                                                       │
│ Substring / partial-pattern anchors PASS too easily when the actor    │
│ lands a wrong-target match (a different column, a different page,     │
│ a different value). Whenever possible prefer value-level anchors:     │
│                                                                       │
│   - cells: ``expected_value`` (compares the cell's actual computed    │
│     value) over ``expected_formula_contains`` — "contains <fn>"       │
│     lets a formula written on the wrong cells PASS while gold         │
│     checks the right ones. ``expected_value`` reads the live cell     │
│     value (works on both formula and literal cells), so it locks      │
│     the check to a specific numeric result.                           │
│   - URLs: full path + query parameters over a single keyword.         │
│   - a11y: ``state_contains`` (checked / selected) over text-only      │
│     match for binary-state controls.                                  │
│                                                                       │
│ When the value is only knowable at runtime, declare a slot and use    │
│ ``${name}`` substitution — see SLOTS section below. Do NOT default    │
│ to substring "just to be safe" — substring is the source of false     │
│ DONEs, not safer.                                                     │
╰───────────────────────────────────────────────────────────────────────╯

╭─ ATOMIC OUTCOMES — split independently-verifiable dimensions ─────────╮
│                                                                        │
│ Test: can each dimension's state be observed INDEPENDENTLY (separate   │
│ a11y nodes / URL params / chip texts)? If YES, each one MUST be its    │
│ own outcome. Don't bundle multiple independent settings into one       │
│ catch-all outcome.                                                    │
│                                                                        │
│ Why: a bundled outcome flaps verified ↔ reverted every time ANY      │
│ dimension changes — the KeyNode detector can't tell "user adjusted    │
│ one knob" apart from "whole config is broken", so the planner re-      │
│ does the entire setup instead of fixing the one drifted dimension.    │
│                                                                        │
│ Exception: a single durable signal that covers them atomically         │
│ (e.g. URL has '?type=A&active=true' where both params always co-set)   │
│ may stay as one outcome.                                              │
│                                                                        │
│ Example (illustrative — different domain):                            │
│   Task: "configure a product search to in-stock, free-ship, 4-star"   │
│   BAD:  one outcome "search_configured" — evidence "URL has stock,    │
│         shipping, AND rating params"                                  │
│   GOOD: three outcomes, all depends_on the parent search outcome —    │
│         "stock_filter_instock"   → URL has 'stock=1' OR chip 'In stock'│
│         "ship_filter_free"       → URL has 'shipping=free'            │
│         "rating_filter_4star"    → URL has rating param OR chip on    │
│                                                                        │
╰────────────────────────────────────────────────────────────────────────╯

╭─ MULTI-STEP INSTRUCTIONS — every imperative verb is its own outcome ─╮
│                                                                        │
│ When the instruction chains mutations with "Next" / "Then" / "And" /   │
│ "After", each sub-clause is an INDEPENDENT outcome — even if an        │
│ earlier clause looks like setup or the file already shows a partial    │
│ example. Failure mode: collapse "fill X. Next compute Y" into one     │
│ outcome on Y; verify on Y passes mechanically (formula present, blank │
│ operands), eval scores 0 because X is still empty.                    │
╰────────────────────────────────────────────────────────────────────────╯

╭─ ACTION COMPLETION — verify the POST-action state, not pre-action ────╮
│                                                                        │
│ For each action verb in the instruction (submit / send / save / open  │
│ / click / create / book / apply / remove), the outcome MUST verify    │
│ the action is COMPLETE — not just that the agent could perform it.    │
│                                                                        │
│   BAD  "submit button is visible"      ← pre-action state             │
│   GOOD "form is submitted: URL changed to /confirmation OR a success  │
│         toast/heading is present"                                     │
│                                                                        │
│   BAD  "settings page link visible"    ← pre-navigation               │
│   GOOD "settings page is open: URL is chrome://settings"              │
│                                                                        │
│   BAD  "send button is enabled"        ← pre-action state             │
│   GOOD "email is sent: appears in Sent folder OR URL navigated away   │
│         from /compose OR 'Message sent' confirmation present"         │
╰────────────────────────────────────────────────────────────────────────╯

Hard constraints:
  • Do NOT include intermediate path steps as outcomes — those are
    the planner's job to sequence. Each outcome should pass the
    cascade test above.
  • Do NOT assume evaluator internals. Only describe what you can
    observe on a future screen / a11y / URL.
  • Do NOT mark any outcome as already-done in this initial output;
    done-status is updated later by the KeyNode detector.
  • depends_on entries MUST reference ids of OTHER outcomes in this
    same list. Don't reference outcomes you haven't listed.

VERIFY DIRECTIVE — alongside ``evidence_hint`` (free text for the
planner), each outcome carries a structured ``verify`` object that the
KeyNode detector executes deterministically every step. Choose one of
THREE kinds based on where the outcome's success state actually lives:

  kind="file_grep" — the outcome's success persists to a known file
       on disk (settings/keybindings/preferences/script source).
       Required: ``file_path`` (absolute or ~-anchored, MUST be a
       regular FILE — not a directory; file_grep on a directory
       returns no verdict. To check file existence at a path, use
       ``shell_command`` shape (4) ``["test", "-f", "<path>"]``
       instead.) and
       ``patterns`` (list of regex strings; all must match by default).
       Optional: ``proximity_lines`` — when set, all patterns must
       co-occur within an N-line sliding window. Use 3 for keybindings
       (same-entry "key" + "command" sit 1 line apart in pretty-printed
       JSON, so a 3-line window catches them while EXCLUDING the next
       entry's fields). Don't use values > 4 — too loose to discriminate
       neighboring entries.

       Pattern-writing tips:
         • Anchor on the value, not just the key:
             GOOD:  '"reportMissingImports"\\s*:\\s*"none"'
             BAD:   '"reportMissingImports"'   ← matches even when
                                                  value is "warning"
         • Boolean values: ``"foo"\\s*:\\s*false`` (no quotes around
           false).
         • Don't use jq syntax — file_grep only supports regex.
         • NESTED-OBJECT TARGETS: when the task target's parent key
           conceptually owns a sub-dict (e.g. `python.analysis.
           diagnosticSeverityOverrides.<rule> = <severity>`), one
           leaf-only pattern accepts BOTH the correct nested form
           `{"parent": {"leaf": v}}` AND the flat-dot-key form
           `{"parent.leaf": v}` — but evaluators usually only accept
           nested. To discriminate, emit TWO patterns + proximity_lines=4:
              ['"<parent>"\\s*:\\s*\\{',  '"<leaf>"\\s*:\\s*"<v>"']
           Skip this when the target is a leaf primitive
           (`editor.fontSize = 14`, `debug.X = false`) — single
           pattern is fine there.

  kind="url_match" — the outcome is a URL state (browser navigation,
       chrome:// page open).
       Required: ``url_pattern`` (regex against the active tab URL).

  kind="shell_command" — RESTRICTED USE. Only when the task instruction
       LITERALLY names a CLI binary, process, or package, and the
       outcome is "this name is reachable / running / installed". The
       command's every argv token must be COPIED VERBATIM from the
       instruction (or be a fixed CLI verb like ``which`` / ``pgrep``
       / ``dpkg``). NEVER invent identifiers (no schema names, no
       config keys, no file paths) — if the parameter isn't in the
       instruction text, do NOT use shell_command.
       Required:
         • ``command``: argv list (each token a separate string —
            e.g. ``["which", "spotify"]``, NOT a single shell string).
            No shell metacharacters / pipes / redirects — the executor
            does NOT spawn a shell.
         • At least one of ``expected_substring`` (case-insensitive
            substrings, ANY one is enough) OR ``forbidden_substring``
            (none of these may appear in stdout).
            RULE: each ``forbidden_substring`` MUST be disjoint from
            every ``expected_substring`` (no overlap, no substring
            relation). Overlapping specs are unsatisfiable — the same
            stdout cannot both contain and not contain the same token.
            RULE: ``expected_substring`` / ``forbidden_substring`` are
            LITERAL substring matches (case-insensitive). They do NOT
            support glob (``*`` ``?``) or regex syntax. To match
            ".ipynb" files in output, use ``expected=['.ipynb']`` —
            NOT ``expected=['*.ipynb']`` (the latter searches for the
            literal ``*`` character which is absent from filename
            output). Same applies to wildcards like ``*.gz``, ``*.py``.
       Optional:
         • ``expected_exit_code`` (default 0).

       The ONLY six task shapes that qualify:
         (1) Binary on PATH (task says "install <bin>" / "make <bin>
              callable"):
                ``["which", "<bin-from-instruction>"]``
                expected_substring: ["<bin-from-instruction>"]
                forbidden_substring: ["not found"]
         (2) Package installed by name (task names the .deb / snap
              package literally):
                ``["dpkg", "-s", "<pkg>"]`` →
                expected ["Status: install ok installed"]
              OR ``["snap", "list", "<pkg>"]`` → forbidden ["error:"]
         (3) Process running by literal name:
                ``["pgrep", "-f", "<process-name-from-instruction>"]``
                expected_exit_code: 0
                expected_substring: ["<process-name>"]
         (4) File exists at known path (task says "save / create /
              generate <file> at <absolute-path>", or produces a
              media file the evaluator compares perceptually):
                ``["test", "-f", "<absolute-path>"]``     (exists)
                ``["test", "-s", "<absolute-path>"]``     (non-empty)
                expected_exit_code: 0
                expected_substring / forbidden_substring: OMIT — test
                produces no stdout; the verdict is exit-code only.
         (4b) File created in a directory with UNKNOWN exact filename
              (task names a target <directory> and a target <extension>
              but the filename is app-derived / "default"). Pick the
              newest matching file:
                ``["sh", "-c", "ls -1t <dir>/*.<ext> 2>/dev/null | head -1"]``
                expected_exit_code: 0
                expected_substring: ["."  + "<ext>"]
              (The substring just confirms ``ls`` returned a non-empty
              line of the right type. Prefer this over ``test -f``
              whenever the filename is implicit.) HARD RULE: for ANY
              outcome whose verb is "save / saved / export / exported
              / download / downloaded / write to / written to" and
              whose object is a file on disk, kind MUST be
              ``shell_command`` (shape 4 or 4b). NEVER ``a11y_match``.
              GUI cues like "dialog closed" / "page visible again" are
              false-positives: dialogs close on cancel, on disk-full,
              and on collision auto-rename — none of which mean the
              file the task asked for is on disk.
         (5) GNOME / Ubuntu system setting state — ONLY when the
              perceiver block above lists the schema verbatim under
              ``[Relevant GSettings schemas]``. The schema name and
              key MUST be copied character-for-character from the
              perceiver output. NEVER write a schema name the
              perceiver did not list:
                ``["gsettings", "get", "<schema-from-perceiver>", "<key-from-perceiver>"]``
                expected_substring: ["<substring-of-expected-value>"]
                  (gsettings prints the value with surrounding quotes
                   / type wrapper — use a substring that survives
                   that formatting, e.g. ``"true"`` / ``"/home/user/.../foo.png"``)
              For relocatable schemas the perceiver prints the
              ``:<dconf-path>`` suffix as part of the schema name —
              keep that suffix verbatim or the call fails.
         (6) Office-document structural property. The shape depends on
              the domain:

              (6a) libreoffice_calc — DO NOT use kind="shell_command"
                   here. Emit kind="calc_verify" with a structured
                   ``calc_checks`` list instead. The framework builds
                   the Python verify body deterministically from the
                   flat JSON, so the LLM never writes Python source —
                   no string escaping, no PASS-token plumbing, no
                   ``from uno_helper import`` hallucinations. Full
                   schema + examples are in the "CALC_VERIFY" block
                   injected into the user message below; refer to
                   that for op names + kwargs.

              (6b) libreoffice_impress — DO NOT use kind="shell_command"
                   here. Emit kind="impress_verify" with a structured
                   ``impress_checks`` list instead, the same way calc
                   uses ``calc_verify``. The framework builds the
                   Python verify body from the flat JSON (op name +
                   kwargs) and runs it through the pre-uploaded
                   ``IMPRESS_OPS_LIBRARY`` on the VM — no python-pptx
                   (not installed), no invented ``uno_helper`` (does
                   not exist), no string-escaping headaches. Full
                   schema + examples are in the "IMPRESS_VERIFY"
                   block injected into the user message below.

              (6c) libreoffice_writer — DO NOT use kind="shell_command"
                   with a hand-written ``python3 -c`` (python-docx /
                   raw UNO). Emit kind="writer_verify" with a
                   structured ``writer_checks`` list, the same way
                   calc uses ``calc_verify``. The framework builds the
                   verify body from the flat JSON and runs it through
                   the pre-uploaded ``WRITER_OPS_LIBRARY`` on the VM —
                   no python-docx (stale-disk vs LO-memory mismatch),
                   no invented UNO API. Full schema + examples are in
                   the "WRITER_VERIFY" block injected into the user
                   message below.

       DO NOT use shell_command for ANY of these — they require
       knowing identifiers that are NOT in the instruction OR the
       perceiver output:
         • GSettings reads whose schema does NOT appear in the
           perceiver's ``[Relevant GSettings schemas]`` block — that
           is unsourced and almost always a hallucinated schema name.
         • dconf reads (always — the perceiver doesn't ship a
           shape for these).
         • find / tar / awk pipelines (multi-step shell logic).
         • Anything that needs a UUID, profile path, or config key
           you didn't see verbatim in the instruction or perceiver
           output.

       For those, prefer a11y_match (LLM judges screen state) or
       file_grep when an actual config file path is known.

  kind="a11y_match" — the post-action state is observed at the GUI
       control's commit point. Right when EITHER:
         (a) the success is genuinely transient (panel open, picker
             visible, dropdown selected) with no persistent backing,
             OR
         (b) the success persists to a config file BUT the exact
             target key NAME is not visible verbatim in the probe
             output or domain knowledge block — guessing a key name
             from training memory leads to a verify that passes only
             when the actor literally echoes that guessed key into
             the file, which diverges from what the app writes via
             the GUI.
       This is the RIGHT choice for preference-toggle tasks
       (Tools/Edit → Preferences, Account Settings, …) when the
       target key isn't grounded — don't fall back to file_grep with
       a guessed key just because a config file exists.
       Required: at least one of ``tag``, ``text_contains`` (list of
       substrings), ``state_contains`` (list of a11y state flags like
       "checked", "selected").

       TOGGLE / RADIO / CHECKBOX outcomes — when the task is to enable
       / disable / select a control with a binary state,
       ``state_contains`` is MANDATORY (``["checked"]`` for toggles,
       ``["selected"]`` for radio / menu items). Text alone matches
       whether the control is on or off; a toggle labelled e.g.
       "<feature> enabled" is present in the a11y tree regardless of
       state. The dual-channel ``visual_must_hold`` below is a
       separate safety net, not a substitute for the state anchor.

       MANDATORY for every a11y_match outcome — write the dual-channel
       safety net so a single transient a11y cue (e.g. a row inside a
       picker dialog, a temporary tooltip) cannot trigger a false
       satisfied verdict:

         "visual_must_hold":  list of 1-3 single-sentence visual
                              clauses that MUST hold true on the
                              current screenshot. AND-semantics.
                              Each clause should DISAMBIGUATE which
                              UI region the a11y row belongs to,
                              OR confirm the action actually committed
                              (e.g. "no dialog visible above the
                              editor"; "the toolbar is back to its
                              default state with no overlay").
                              Avoid vague phrases that just paraphrase
                              the a11y constraint.

         "visual_must_not_hold": list of 0-2 single-sentence visual
                              clauses that MUST be FALSE. Required
                              for any "commit-by-dismiss" task —
                              i.e., where the action only counts when
                              an intermediate overlay disappears
                              (modal dialog, popover, command palette,
                              dropdown). The first clause should name
                              that intermediate overlay so its
                              continued presence rules out success.

       Generic shape (NOT a real task — placeholders only):

         outcome "<some_committed_state>":
           visual_must_hold: [
             "The widget that signals success is in its NORMAL screen
              region (e.g. main editor / sidebar / status bar), NOT
              inside a transient overlay.",
             "The intermediate overlay used to trigger the action
              (dialog / popover / palette) is no longer visible."
           ]
           visual_must_not_hold: [
             "An intermediate overlay (dialog with confirm/cancel
              buttons, dropdown, command palette) is still visible
              above the main UI."
           ]

KNOWN APP CONFIG PATHS — when the task is about one of these apps,
PREFER the listed path over guessing.

  vs_code:
    user_settings = ~/.config/Code/User/settings.json
    keybindings   = ~/.config/Code/User/keybindings.json
  google-chrome:
    NEVER use file_grep against ~/.config/google-chrome/* — Chrome
    owns these files live; they aren't flushed to disk until Chrome
    exits or settles, so file_grep FAILs even when the actor flipped
    the right toggle. Use a11y_match for toggle / setting state, and
    url_match for navigation.
  libreoffice:
    config        = ~/.config/libreoffice/4/user/registrymodifications.xcu
  filesystem (general):
    use the literal absolute path the task names (e.g. /tmp/test.py).

╭─ SLOTS — placeholders for values you DO NOT know yet ────────────────╮
│                                                                       │
│ Some verify values are revealed only by running the task — anything   │
│ the agent must READ or COMPUTE before it can be checked (a value      │
│ pulled from a page, a filename chosen at runtime by another app,      │
│ a clipboard payload, a number an evaluator paste-buffers).            │
│                                                                       │
│ When you would otherwise write a generic catch-all pattern (``.*``,   │
│ ``.+``, an empty string, a vacuous keyword like "<the value>" or     │
│ "<extracted X>"), DO NOT. Vacuous patterns false-positive the         │
│ deterministic verifier on any observable input and let the agent      │
│ declare DONE on nothing.                                              │
│                                                                       │
│ Instead:                                                              │
│   1. Declare the unknown in the top-level ``slots`` array with a      │
│      short snake_case name, a coarse type, and a one-line             │
│      description + ``source_hint`` telling the runtime perceiver      │
│      where to look. Types: url | text | path | numeric | cell_ref.    │
│   2. Reference it inside the relevant verify field as ``${name}``.    │
│      The runtime substitutes the bound value at verify time, and      │
│      defers the check to the LLM judge while the slot is unbound      │
│      (so deterministic never rubber-stamps).                          │
│                                                                       │
│ DO NOT declare a slot just because the value is "specific". Only      │
│ when the value CANNOT be known until the agent has explored. Concrete │
│ keys named in the instruction or visible in the perceiver block stay  │
│ as literal verify values.                                             │
│                                                                       │
│ Generic shape (substitute with the abstraction appropriate to your    │
│ task — NEVER copy these names verbatim):                              │
│                                                                       │
│   slots: [{                                                           │
│     "name": "<runtime_value>", "type": "<url|text|path|...>",         │
│     "description": "<what the value is>",                             │
│     "source_hint": "<which app surface exposes it>"                   │
│   }]                                                                  │
│   outcome.verify:                                                     │
│     {"kind": "url_match", "url_pattern": "^${<runtime_value>}$"}      │
│  OR {"kind": "file_grep", "patterns": ["^${<runtime_value>}$"]}       │
│  OR {"kind": "a11y_match", "text_contains": ["${<runtime_value>}"]}   │
│  OR (inside impress_verify / calc_verify / writer_verify checks):     │
│     {"kind": "impress_verify",                                        │
│      "impress_checks": [{                                              │
│        "op": "slide_notes", "slide": 1,                               │
│        "expected_substring": "${<runtime_value>}"                     │
│      }]}                                                               │
│                                                                       │
│ Refs inside regex fields (``patterns``, ``url_pattern``) are          │
│ ``re.escape``'d on substitution — bind values verbatim, do not        │
│ pre-escape.                                                           │
│                                                                       │
│ ★ HARD BAN — bracket-style placeholders. NEVER write:                 │
│     "[Planned remarks for Slide N]"   ✗ (square brackets + caps)     │
│     "<value>"  /  "<expected text>"   ✗ (angle brackets)             │
│     "{{ slot_name }}"                 ✗ (curly braces)               │
│ The verifier greps these LITERALLY and they never match — the         │
│ outcome would stay pending forever. The ONLY supported "I don't       │
│ know this yet" syntax is ``${slot_name}`` with the slot declared      │
│ in the top-level ``slots`` array. If you don't know what to put       │
│ AND can't declare a slot for it (because no observable source for     │
│ the value exists), OMIT the verify field — the outcome will fall      │
│ back to LLM judge against the screenshot.                             │
╰───────────────────────────────────────────────────────────────────────╯

╭─ VALUE PROVENANCE — classify every literal verify value KNOWN/UNKNOWN ─╮
│                                                                        │
│ Before you finalize, walk EVERY concrete literal you put in a verify   │
│ spec (a ``patterns`` entry, ``expected_substring``, ``url_pattern``, a │
│ ``calc_checks`` value, ...) and classify it:                          │
│                                                                        │
│   KNOWN   — the EXACT bytes appear in the instruction or in the        │
│             accessibility tree / perceiver block you were given. You   │
│             are copying observed bytes, not recalling them.            │
│                                                                        │
│   UNKNOWN — you are reconstructing the value from memory or            │
│             expectation rather than copying it, OR the form an app     │
│             STORES / outputs may differ from the form you can see      │
│             (apps reformat, expand, truncate, or relabel values when   │
│             they persist them). A value generated, read, or computed   │
│             only once the task runs is UNKNOWN. When unsure whether    │
│             the stored/checked bytes equal what you observed → UNKNOWN.│
│                                                                        │
│   An UNKNOWN value MUST become a ``${slot}`` or the verify field is    │
│   OMITTED — NEVER a hardcoded literal. Prefer the most STABLE KNOWN    │
│   signal available (an identifier copied verbatim from the            │
│   instruction or perceiver) over a fragile value you are guessing the  │
│   exact bytes of.                                                     │
│                                                                        │
│ Emit the classification in the top-level ``value_provenance`` array    │
│ (schema below). No value you mark UNKNOWN may appear as a literal      │
│ anywhere in your verify specs — the framework rejects that.           │
╰────────────────────────────────────────────────────────────────────────╯

Output ONLY a JSON object in this exact schema:

{
  "initial_context": {
    "open_tabs":    ["<url>", ...],
    "active_url":   "<url or null>",
    "active_app":   "<app name or null>",
    "window_title": "<title or null>"
  },
  "value_provenance": [
    {
      "value":  "<each literal you placed in a verify spec, verbatim>",
      "status": "KNOWN" | "UNKNOWN",
      "why":    "<one line: where you copied it from (KNOWN), or why its exact bytes are runtime-derived (UNKNOWN)>"
    },
    ...
  ],
  "slots": [
    {
      "name":         "<snake_case>",
      "type":         "url" | "text" | "path" | "numeric" | "cell_ref",
      "description":  "<what the value is, in one sentence>",
      "source_hint":  "<where the perceiver should look to read it>"
    },
    ...
  ],
  "required_outcomes": [
    {
      "id": "<snake_case>",
      "description": "<one sentence>",
      "evidence_hint": "<concrete observable, prefer durable signal>",
      "depends_on": ["<other_outcome_id>", ...],
      "verify": {
        "kind": "file_grep" | "url_match" | "a11y_match" | "shell_command" | "calc_verify" | "impress_verify" | "writer_verify",
        "file_path": "<for file_grep, else omit>",
        "patterns": ["<regex>", ...],
        "all_must_match": true,
        "proximity_lines": null,
        "url_pattern": "<for url_match, else omit>",
        "tag": "<for a11y_match, else omit>",
        "text_contains": ["<substring>", ...],
        "state_contains": ["<state flag>", ...],
        "visual_must_hold":     ["<single-sentence clause>", ...],
        "visual_must_not_hold": ["<single-sentence clause>", ...],
        "command":              ["<argv>", ...],
        "expected_substring":   ["<substring>", ...],
        "forbidden_substring":  ["<substring>", ...],
        "expected_exit_code":   0,
        "calc_checks":          [ {"op": "<op>", ...kwargs}, ... ],
        "impress_checks":       [ {"op": "<op>", ...kwargs}, ... ],
        "writer_checks":        [ {"op": "<op>", ...kwargs}, ... ]
      }
    },
    ...
  ]
}

Worked example — task: "Find the discussion in the Community section,
under Latest category, with the most replies."

{
  "initial_context": { "active_url": "site.com/", "active_app": "chrome",
                        "window_title": "Site Home", "open_tabs": [] },
  "required_outcomes": [
    {
      "id": "on_community_section",
      "description": "Browser is on the Community section",
      "evidence_hint": "URL path contains '/community' or '/discussions'",
      "depends_on": [],
      "verify": {
        "kind": "url_match",
        "url_pattern": "/(community|discussions)"
      }
    },
    {
      "id": "latest_filter_active",
      "description": "The Latest category filter is selected",
      "evidence_hint": "URL contains 'sort=latest' OR a11y tree has tab 'Latest' with state=selected",
      "depends_on": ["on_community_section"],
      "verify": {
        "kind": "url_match",
        "url_pattern": "sort=latest"
      }
    },
    {
      "id": "top_thread_opened",
      "description": "A thread is opened",
      "evidence_hint": "URL matches a thread path (e.g. /thread/<id>) AND the page shows posts",
      "depends_on": ["latest_filter_active"],
      "verify": {
        "kind": "url_match",
        "url_pattern": "/thread/\\\\d+"
      }
    }
  ]
}

Worked example — task: "modify VS Code settings to disable error
reporting for Python missing imports."

{
  "initial_context": { "active_app": "vscode", "active_url": null,
                        "window_title": "Visual Studio Code", "open_tabs": [] },
  "required_outcomes": [
    {
      "id": "missing_imports_disabled",
      "description": "settings.json sets reportMissingImports to 'none'",
      "evidence_hint": "~/.config/Code/User/settings.json contains python.analysis.diagnosticSeverityOverrides.reportMissingImports = 'none'",
      "depends_on": [],
      "verify": {
        "kind": "file_grep",
        "file_path": "~/.config/Code/User/settings.json",
        "patterns": ["\\"reportMissingImports\\"\\\\s*:\\\\s*\\"none\\""]
      }
    }
  ]
}

Worked example for kind="file_grep" on keybindings.json — shape
"add or override a key binding" (generic placeholder, NOT a real
task — substitute `<key-from-instruction>`, `<command-from-
instruction>`, `<when-from-instruction>` with the EXACT values
named in the user's instruction when you author the spec; never
carry over the placeholder values):

{
  "required_outcomes": [
    {
      "id": "<keybinding_set>",
      "description": "<one-sentence description naming the exact key+command+when from the instruction>",
      "evidence_hint": "<observable: keybindings.json has an entry whose key=<key-from-instruction>, command=<command-from-instruction>, when=<when-from-instruction>>",
      "depends_on": [],
      "verify": {
        "kind": "file_grep",
        "file_path": "~/.config/Code/User/keybindings.json",
        "patterns": [
          "\\"key\\"\\\\s*:\\\\s*\\"<key-from-instruction, regex-escaped>\\"",
          "\\"command\\"\\\\s*:\\\\s*\\"<command-from-instruction, regex-escaped>\\"",
          "\\"when\\"\\\\s*:\\\\s*\\"<when-clause-from-instruction, regex-escaped>\\""
        ],
        "proximity_lines": 3
      }
    }
  ]
}

The `when` pattern is REQUIRED whenever the instruction names a
specific focus context for the binding — "from / in / while focused
on the terminal" implies a `when` clause naming that context (e.g.
the terminal context key). Omitting the `when` pattern lets the
actor satisfy the verify with a no-`when` entry that the evaluator
will reject. Drop the `when` pattern ONLY when the instruction
explicitly says the binding should fire globally / everywhere.

When the instruction asks to REMOVE a default binding rather than
add a new one, the `command` value carries a leading `-` (e.g. the
override is `-<original-command>`). The `key` in that entry is
still the SAME key as the binding being removed — do NOT substitute
a different key. The `when` clause must match the original
binding's `when`.

Worked example for kind="shell_command" — shape (1) "binary on PATH"
(generic placeholder, NOT a real task — instruction names the binary):

{
  "required_outcomes": [
    {
      "id": "<binary_present>",
      "description": "<binary-from-instruction> is reachable from the user shell",
      "evidence_hint": "`which <binary-from-instruction>` prints a path",
      "depends_on": [],
      "verify": {
        "kind": "shell_command",
        "command": ["which", "<binary-from-instruction>"],
        "expected_substring": ["<binary-from-instruction>"],
        "forbidden_substring": ["not found"]
      }
    }
  ]
}
"""


# --------------------------------------------------------------------------- #
# LLM call
# --------------------------------------------------------------------------- #

# Appended to the init prompt in boundary mode: author milestone DESCRIPTIONS
# only; the verify METHOD is authored later at the boundary (boundary_verify.py).
# Keeps all milestone-selection rules, drops verify-spec/value_provenance/slots.
# init_ledger also forces verify=None, so this is belt-and-suspenders.
_DESC_ONLY_OVERRIDE = (
    "\n\n════════ OUTPUT OVERRIDE — DESCRIPTIONS-ONLY MODE ════════\n"
    "For THIS run the verification METHOD is authored LATER — when each "
    "milestone is claimed done, with full execution context — NOT now. So:\n"
    "  • OMIT the \"verify\" field on every outcome.\n"
    "  • Do NOT emit a \"value_provenance\" array or a \"slots\" array.\n"
    "  • Ignore every instruction above about verify-spec kinds, evidence "
    "durability for the verifier, value provenance, and slots.\n"
    "Everything ELSE stands unchanged: milestone-vs-waypoint selection, one "
    "outcome per concrete deliverable, depends_on ordering, and (multi-app "
    "tasks only) the per-outcome \"app\"/\"phase\" tags and the top-level "
    "\"phases\" array. Keep evidence_hint as a concrete observable — WHERE to "
    "look / what success looks like — so the later verifier knows where to "
    "check.\n"
    "Each required_outcomes entry = { id, description, evidence_hint, "
    "depends_on [, app, phase for multi-app] }. Return the JSON ledger seed "
    "object and nothing else."
)


def init_ledger(
    instruction: str,
    initial_obs: Dict[str, Any],
    call_llm: Callable,
    model: Optional[str] = None,
    *,
    a11y_text: Optional[str] = None,
    active_url: Optional[str] = None,
    domain: Optional[str] = None,
    env: Any = None,
    on_probe_dump: Optional[Callable[
        [Optional[str], Optional[str], Optional[str]], None]] = None,
    on_prompt_dump: Optional[Callable[
        [List[Dict[str, Any]], int, str], None]] = None,
    doc_inspect_block: Optional[str] = None,
    related_apps: Optional[List[str]] = None,
) -> ProgressLedger:
    """Build a fresh :class:`ProgressLedger` for a new task.

    Args:
        instruction:   raw task instruction.
        initial_obs:   first observation (must have ``screenshot``; may have
                        ``accessibility_tree``).
        call_llm:      callable compatible with the planner's
                        ``self.call_llm(payload, model)``.
        model:         model name; ``None`` lets the caller default elsewhere.
        a11y_text:     pre-linearized a11y tree (optional; avoids re-linearizing).
        active_url:    pre-extracted active tab URL (optional).

    Returns a :class:`ProgressLedger` with ``initial_context`` and
    ``required_outcomes``; ``done``/``failed_paths`` empty. On parse failure,
    returns an empty-outcome ledger with the known ``initial_context`` — caller
    should tolerate a best-effort ledger rather than crash.
    """
    screenshot_bytes = initial_obs.get("screenshot") if isinstance(initial_obs, dict) else None

    user_content: List[Dict[str, Any]] = []
    initial_b64: Optional[str] = None

    if screenshot_bytes:
        try:
            initial_b64 = base64.b64encode(screenshot_bytes).decode("ascii")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{initial_b64}"},
            })
        except Exception as e:
            logger.warning("[LedgerInit] failed to encode screenshot: %s", e)

    # VM environment probe + per-task digestion. Plain-text inventory of
    # binaries / config files / GSettings schemas / processes that actually
    # exist on the VM, so the LLM grounds identifiers instead of inventing
    # them. Skipped silently when env is missing or probe fails.
    environment_probe_text: Optional[str] = None
    raw_probe: Optional[str] = None
    # ─────────────────────────────────────────────────────────────────
    # [FIX env_probe_unconditional] Two layers, both ALWAYS run.
    #   ① probe_environment — VM-side bash inventory (~5s): Desktop/
    #     Documents/Downloads listings, binaries, gsettings, procs. Raw
    #     artifact saved to environment_probe_raw.txt.
    #   ② perceive_environment — host-side LLM compression of ① into
    #     digest sections ([Identity] / [Possibly useful files] / ...).
    #     No VM interaction, so it can't cause /run_python timeouts.
    #     [Possibly useful files] is key for multi-app office tasks where
    #     the open document is paired with other files on disk.
    #
    # The heavyweight file_probes.probe_useful_files step (③, reads file
    # CONTENT via soffice→csv) caused VM contention before; stays gated off
    # by default via OSWORLD_FILE_PROBE below. Roll back to the old
    # domain-skip behavior by restoring ``_skip_probe_for_office``.
    try:
        from mm_agents.structagent.ledger.core.probes import probe_environment
        raw_probe = probe_environment(env, domain=domain)
    except Exception as e:
        logger.info("[LedgerInit] env probe (raw) failed: %s", e)
        raw_probe = None
    if raw_probe:
        try:
            from mm_agents.structagent.perception.perceiver import perceive_environment
            environment_probe_text = perceive_environment(
                instruction=instruction,
                probe_text=raw_probe,
                call_llm=call_llm,
                model=model,
                domain=domain,
            )
        except Exception as e:
            logger.info("[LedgerInit] env probe (digest) failed: %s", e)
            environment_probe_text = None
    # ──── end FIX env_probe_unconditional ────
    # Disk-content probe — read on-disk contents of each [Possibly useful
    # files] entry (soffice --convert-to + stdlib csv/text). Without it,
    # init_ledger / the verify-author guess cell positions (placeholder
    # a1_refs like "new_day_cell") or the task shape. None on any failure.
    # DISABLED — implicated in env-side timeouts during a batch run (root
    # cause unconfirmed: VM contention / env API health / call cadence).
    # Re-enable after diagnosis; flip via the env var to A/B without churn.
    useful_file_contents: Optional[str] = None
    from mm_agents.structagent.config import CAConfig
    if CAConfig.from_env().file_probe:
        try:
            from mm_agents.structagent.ledger.core.probes import (
                extract_useful_paths, probe_useful_files,
            )
            _useful_paths = extract_useful_paths(environment_probe_text)
            if _useful_paths:
                useful_file_contents = probe_useful_files(env, _useful_paths)
                logger.info(
                    "[LedgerInit] file_probes: %d path(s) → %s",
                    len(_useful_paths),
                    ("%d chars" % len(useful_file_contents))
                    if useful_file_contents else "no block",
                )
        except Exception as e:
            logger.info("[LedgerInit] file_probes failed: %s", e)

    if on_probe_dump is not None:
        try:
            on_probe_dump(raw_probe, environment_probe_text,
                          useful_file_contents)
        except Exception as e:
            logger.info("[LedgerInit] on_probe_dump raised: %s", e)

    text_parts: List[str] = [f"Task instruction:\n{instruction.strip()}"]
    # Multi-app tasks (OSWorld multi_apps): tell the author to tag every
    # outcome with the app whose state proves it (drives per-app context
    # switching). Single-app tasks skip this → byte-identical prompt.
    _ma_apps = [a for a in (related_apps or []) if a and str(a).strip()]
    if len(_ma_apps) > 1:
        text_parts.append(
            "MULTI-APP TASK — this task spans these applications:\n"
            f"  {_ma_apps}\n"
            "EXTEND the output schema with TWO additions, authored in "
            "this order:\n"
            "\n"
            "(1) FIRST, a top-level \"phases\" array. Decompose the task "
            "into app PHASES — one phase per application, in execution "
            "order. A phase is the coarse unit \"work in app X toward "
            "goal G, then hand an artifact to the next app\". Each phase "
            "object:\n"
            "    {\n"
            "      \"app\":  \"<one app from the list above>\",\n"
            "      \"goal\": \"<one SEMANTIC sentence — what to "
            "accomplish in this app. NO concrete anchors: no cell "
            "ranges, indices, counts, row/col numbers, or literal "
            "content values — the app is not observed yet, anchors "
            "would be guesses>\",\n"
            "      \"planned_handoff_in\":  \"<what this phase receives "
            "from the previous phase — semantic; null for phase 1>\",\n"
            "      \"planned_handoff_out\": \"<what artifact this phase "
            "produces for the next phase — semantic; null for the last "
            "phase>\"\n"
            "    }\n"
            "  • One phase per DISTINCT app stretch. If the task "
            "returns to an app later, that is a separate phase.\n"
            "  • The app list may include an app the task does NOT "
            "use — only emit phases for apps genuinely needed.\n"
            "\n"
            "(2) THEN \"required_outcomes\" as specified — each outcome "
            "ALSO carries:\n"
            "    \"app\":   the ONE app whose state proves it (verify "
            "location: file on disk → \"os\"; spreadsheet cell → "
            "\"libreoffice_calc\"; browser URL → \"chrome\"; etc.),\n"
            "    \"phase\": the 1-based index into the \"phases\" array "
            "of the phase this outcome belongs to.\n"
            "  • Every outcome must name a valid phase index. Outcomes "
            "cluster by phase and roughly in execution order.\n"
            "  • A phase may legitimately have zero outcomes (e.g. a "
            "pure navigation phase) — that is allowed."
        )
    if doc_inspect_block:
        # Structured perceiver block (SheetPerceiver / DocPerceiver):
        # aligned cell types + formulas + head rows (Calc), paragraph list +
        # styles (Writer). Authoritative for document structure. When present
        # we skip the a11y tree below — it flattens cells/paragraphs into
        # lines that can't tell FORMULA from VALUE from EMPTY, the exact
        # signal needed to pick the answer cell.
        text_parts.append(
            "Live document structure (authoritative — produced by a "
            "framework-side UNO probe of the actual open document):\n"
            + doc_inspect_block
        )
    elif a11y_text:
        text_parts.append(
            "Accessibility tree of the initial screen (aux; "
            "tab / text / state):\n" + a11y_text
        )
    if active_url:
        text_parts.append(f"Active tab URL (aux): {active_url}")
    if environment_probe_text:
        text_parts.append(
            "VM environment snapshot. Tool inventory, NOT a path "
            "recommendation; don't bias toward CLI because more CLI "
            "items are listed.\n"
            "  Actor's operation path: prefer GUI when a native app "
            "handles the target directly (in-app preferences, right-"
            "click menus, install buttons, dialogs). Prefer CLI for "
            "editing app config files (JSON / INI / dot-files under "
            "~/.config or app data dirs), system settings via "
            "gsettings / dconf, batch file ops, text edits, pattern "
            "matching across many files.\n"
            "  Verification (your verify spec): regardless of operation "
            "path, prefer CLI / file checks for accuracy — only fall "
            "back to a11y_match when no CLI/file mirror exists.\n\n"
            + environment_probe_text
        )
    if useful_file_contents:
        text_parts.append(
            "Useful file contents (on-disk snapshot of files surfaced "
            "in [Possibly useful files]). Read these BEFORE writing "
            "verify specs or planning concrete operations: the cell "
            "addresses / paragraph indices / slide numbers below are "
            "AUTHORITATIVE — use them verbatim instead of inventing "
            "placeholders. Spreadsheet blocks are rendered in the same "
            "format as the per-turn SheetPerceiver, so the same "
            "column-letter / row-number coordinate system applies.\n\n"
            + useful_file_contents
        )
    # Domain-specific knowledge — only for the matching domain (empty for
    # unknown/mismatched, so it never bleeds across).
    if domain:
        try:
            from mm_agents.structagent.domain import get_verifier_knowledge
            if len(_ma_apps) > 1:
                # Multi-app plans the whole cross-app task — inject every
                # related app's knowledge, each headed.
                for _dk_app in _ma_apps:
                    dk = get_verifier_knowledge(_dk_app)
                    if dk:
                        text_parts.append(
                            f"### Domain-specific knowledge for "
                            f"`{_dk_app}` ###\n{dk}"
                        )
            else:
                dk = get_verifier_knowledge(domain)
                if dk:
                    text_parts.append(
                        f"Domain-specific knowledge for `{domain}` "
                        f"(applies ONLY to this task's domain):\n{dk}"
                    )
            # The three LibreOffice domains use structured calc_verify /
            # impress_verify / writer_verify kinds; the framework builds the
            # UNO body deterministically, so no raw-UNO template here. The
            # per-domain VERIFY schema block is injected separately below.
        except Exception as e:
            logger.info("[LedgerInit] domain_knowledge load failed: %s", e)

    # Verifier experience memory: boundary-mode t=0 emits descriptions only
    # (no VerifySpec), so the spec-authoring recipe block isn't injected here.
    # ENABLE_VERIFIER_EXPERIENCE_MEMORY instead feeds the boundary verifier
    # retrieved CHECK RECIPES at the boundary (see query_check_recipes).

    # Phase B: Q1-Q4 reasoning block for Calc. WORKBOOK STRUCTURE + the
    # Q1-Q4 STOP chain in domain knowledge give the input + question shape;
    # data_layout forces an answer slot in the output so reasoning is visible
    # and verify-spec building has a grounded column/row anchor to copy.
    # Phase C: calc_verify / impress_verify schema block — framework builds
    # the python3 -c body server-side, so the LLM never writes verify Python
    # (kills the ``from uno_helper import`` hallucination class).
    #
    # MULTI-APP: a multi_apps task plans across EVERY related app, but
    # ``domain`` is only related_apps[0]. Gating on domain=="libreoffice_calc"
    # skipped these blocks when the first app isn't calc — directive 6a/6b
    # then told the LLM to emit calc_verify/impress_verify without showing the
    # schema, so it fell back to a broken shell_command UNO one-liner. Inject
    # for every office app the task spans. Single-app: _verify_apps==[domain]
    # → byte-identical to before.
    _verify_apps = {
        (a or "").lower()
        for a in (_ma_apps if len(_ma_apps) > 1 else [domain])
    }
    if "libreoffice_calc" in _verify_apps:
        try:
            from mm_agents.structagent.actions.calc.actions import (
                data_layout_init_block, calc_verify_schema_block,
            )
            text_parts.append(data_layout_init_block())
            text_parts.append(calc_verify_schema_block())
        except Exception as e:
            logger.info("[LedgerInit] data_layout / calc_verify "
                        "block load failed: %s", e)
    if "libreoffice_impress" in _verify_apps:
        try:
            from mm_agents.structagent.actions.impress.actions import (
                data_layout_init_block as impress_data_layout_init_block,
                impress_verify_schema_block,
            )
            text_parts.append(impress_data_layout_init_block())
            text_parts.append(impress_verify_schema_block())
        except Exception as e:
            logger.info("[LedgerInit] impress data_layout / verify "
                        "block load failed: %s", e)
    if "libreoffice_writer" in _verify_apps:
        try:
            from mm_agents.structagent.actions.writer.actions import (
                writer_verify_schema_block,
            )
            text_parts.append(writer_verify_schema_block())
        except Exception as e:
            logger.info("[LedgerInit] writer_verify block load "
                        "failed: %s", e)

    # Multi-app JSON schema update — placed LAST so it's the authoritative
    # output template. The office data_layout block above ships its own
    # "JSON schema update" that omits ``phases``; without this final schema
    # the LLM follows that one and drops the phase board. Restates the full
    # shape with ``phases`` and shows ``data_layout`` coexisting.
    if len(_ma_apps) > 1:
        text_parts.append(
            "JSON schema update for MULTI-APP tasks (authoritative — "
            "this is the final output shape):\n"
            "\n"
            "{\n"
            "  \"initial_context\": { ... },\n"
            "  \"phases\": [                       // REQUIRED for "
            "multi-app — the app-level decomposition\n"
            "    {\n"
            "      \"app\":  \"<one related app>\",\n"
            "      \"goal\": \"<one semantic sentence — NO concrete "
            "anchors>\",\n"
            "      \"planned_handoff_in\":  \"<from previous phase; "
            "null for phase 1>\",\n"
            "      \"planned_handoff_out\": \"<to next phase; null for "
            "the last phase>\"\n"
            "    },\n"
            "    ...                              // one per app phase, "
            "in execution order\n"
            "  ],\n"
            "  \"data_layout\": { ... },           // office tasks "
            "ONLY — keep it AND \"phases\"; they do not replace each "
            "other\n"
            "  \"required_outcomes\": [\n"
            "    { ..., \"app\": \"<app>\", \"phase\": <1-based phase "
            "index> },\n"
            "    ...\n"
            "  ]\n"
            "}\n"
            "\n"
            "\"phases\" is authored FIRST, before \"required_outcomes\". "
            "Every outcome carries its \"phase\" index. Emit \"phases\" "
            "even when you also emit \"data_layout\"."
        )
    text_parts.append(
        "Return the JSON ledger seed object and nothing else."
    )
    user_content.append({"type": "text", "text": "\n\n".join(text_parts)})

    # Boundary-verify mode: author descriptions only at t=0; verify METHOD
    # comes later at the boundary. Append override here + force verify=None below.
    user_content.append({"type": "text", "text": _DESC_ONLY_OVERRIDE})

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    def _call_once(_messages: List[Dict[str, Any]]) -> str:
        """Single LLM invocation. Honors QWEN35_THINKING: when on, the model
        walks the verify-kind decision tree before emitting JSON, which cuts
        the invented-key bug class. ``_parse_seed_json`` strips any
        ``<think>...</think>`` prefix."""
        try:
            return call_llm(
                {"model": model, "messages": _messages,
                 "max_tokens": 5000, "temperature": 0.0, "top_p": 0.9},
                model,
            ) or ""
        except Exception as e:
            logger.warning("[LedgerInit] LLM call failed: %s", e)
            return ""

    # Up to 3 attempts. temperature=0.0 still varies on OpenRouter qwen3.5-9b
    # (same input → different outcome counts/keys across runs). Retry on zero
    # outcomes / parse failure — cheap, avoids an empty ledger from one bad
    # sample.
    parsed: Optional[Dict[str, Any]] = None
    outcomes: List[Outcome] = []
    raw: Optional[str] = None
    for attempt in range(1, 4):
        raw_attempt = _call_once(messages)
        # Dump prompt + raw response for offline audit of why the author LLM
        # produced a given outcome/verifier shape.
        if on_prompt_dump is not None:
            try:
                on_prompt_dump(messages, attempt, raw_attempt or "")
            except Exception as e:
                logger.info("[LedgerInit] on_prompt_dump raised: %s", e)
        parsed_attempt = _parse_seed_json(raw_attempt) if raw_attempt else None
        # On parse failure (e.g. model nests `expected_substring` inside the
        # `command` array), one LLM repair pass usually rescues the response —
        # cheaper than discarding the attempt.
        if parsed_attempt is None and raw_attempt:
            try:
                from mm_agents.structagent.ledger.support.json_repair import repair_json_via_llm
                fixed = repair_json_via_llm(
                    raw_attempt, call_llm=call_llm, model=model, logger=logger,
                )
                if fixed:
                    parsed_attempt = _parse_seed_json(fixed)
                    if parsed_attempt is not None:
                        logger.info(
                            "[LedgerInit] attempt %d: parsed AFTER json_repair pass",
                            attempt)
            except Exception as e:
                logger.info("[LedgerInit] json_repair raised: %s", e)
        outcomes_attempt = _build_outcomes(parsed_attempt)
        if outcomes_attempt:
            # VALUE PROVENANCE: a value the model marked UNKNOWN must not stay
            # a hardcoded literal in a verify spec (should be ${slot} or
            # omitted). Push back with the offenders and retry — uses the
            # model's own classification, so no per-task knowledge needed.
            _viol = _value_provenance_violations(parsed_attempt)
            if _viol and attempt < 3:
                logger.warning(
                    "[LedgerInit] attempt %d: %d UNKNOWN value(s) still "
                    "hardcoded in verify specs %s — pushing back",
                    attempt, len(_viol), _viol)
                messages = list(messages) + [{
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        "Your value_provenance marked these values UNKNOWN, but "
                        "you still wrote them as literals inside verify specs: "
                        + json.dumps(_viol, ensure_ascii=False)
                        + ". An UNKNOWN value cannot be a hardcoded literal — "
                        "for each, either replace it with a ${slot} (declared in "
                        "the top-level slots array) or OMIT that verify field. "
                        "Re-emit the COMPLETE corrected JSON ledger object.")}]}]
                continue
            raw, parsed, outcomes = raw_attempt, parsed_attempt, outcomes_attempt
            if attempt > 1:
                logger.info("[LedgerInit] succeeded on attempt %d "
                            "(%d outcomes)", attempt, len(outcomes))
            break
        logger.info("[LedgerInit] attempt %d yielded 0 outcomes; "
                    "retrying" if attempt < 3 else
                    "[LedgerInit] attempt %d yielded 0 outcomes; giving up",
                    attempt)

    # Descriptions-only (boundary mode): verify METHOD is authored at the
    # boundary. Force verify=None even if the LLM emitted one despite the
    # override, and skip repair-or-drop (it would otherwise rewrite/drop every
    # spec-less outcome — exactly the t=0 specs we're deferring).
    for _o in outcomes:
        _o.verify = None

    ctx = _build_context(parsed, a11y_text=a11y_text, active_url=active_url)
    if environment_probe_text:
        ctx.environment_probe = environment_probe_text
    if useful_file_contents:
        ctx.useful_file_contents = useful_file_contents

    # Multi-app: build the PhaseBoard from the LLM's ``phases`` array and
    # assign each surviving outcome to its phase. Single-app has no ``phases``
    # → from_init_payload returns None → downstream phase call sites no-op.
    phase_board = None
    if len(_ma_apps) > 1:
        try:
            from mm_agents.structagent.ledger.core.phase_board import PhaseBoard
            phase_board = PhaseBoard.from_init_payload(
                (parsed or {}).get("phases"))
            if phase_board is not None:
                # id → 1-based phase index. Match by id, not position:
                # fact-check/repair may have dropped or reordered outcomes.
                _id2phase: Dict[str, int] = {}
                for _o in (parsed or {}).get("required_outcomes", []) or []:
                    if isinstance(_o, dict) and _o.get("id") is not None:
                        try:
                            _id2phase[str(_o["id"])] = int(_o.get("phase"))
                        except (TypeError, ValueError):
                            pass
                _unplaced = []                     # outcomes with no phase
                for _oc in outcomes:
                    _pi = _id2phase.get(_oc.id)
                    if _pi is not None and 1 <= _pi <= len(phase_board.phases):
                        phase_board.phases[_pi - 1].outcome_ids.append(_oc.id)
                    else:
                        _unplaced.append(_oc)      # collected; not surfaced
                logger.info(
                    "[LedgerInit] PhaseBoard: %d phases — %s%s",
                    len(phase_board.phases),
                    [f"{p.app}({len(p.outcome_ids)}oc)"
                     for p in phase_board.phases],
                    (f" | {len(_unplaced)} UNPLACED" if _unplaced else ""))
        except Exception as e:
            logger.info("[LedgerInit] PhaseBoard build failed: %s", e)
            phase_board = None

    slots = _build_slots(parsed)
    ledger = ProgressLedger(
        initial_context=ctx,
        required_outcomes=outcomes,
        done=[],
        failed_paths=[],
        initial_screenshot_b64=initial_b64,
        phase_board=phase_board,
        slots=slots,
    )
    logger.info(
        "[LedgerInit] seeded ledger: %d outcomes, %d slot(s), app=%s active_url=%s",
        len(outcomes), len(slots), ctx.active_app, ctx.active_url,
    )
    if slots:
        logger.info(
            "[LedgerInit] slots: %s",
            ", ".join(f"{n}({s.type})" for n, s in slots.items()),
        )
    return ledger


# --------------------------------------------------------------------------- #
# JSON parsing helpers
# --------------------------------------------------------------------------- #

def _value_provenance_violations(parsed: Optional[Dict[str, Any]]) -> List[str]:
    """Return UNKNOWN-classified values still hardcoded as literals in some
    verify spec (each should be a ``${slot}`` or omitted). Empty = clean or no
    value_provenance.

    Enforces the model's own ``value_provenance`` verdict, so no per-task
    knowledge needed. Correctly slotted values appear as ``${slot}`` and don't
    trip this. Short values (<=3 chars) skipped to avoid coincidental matches.
    """
    if not isinstance(parsed, dict):
        return []
    prov = parsed.get("value_provenance")
    if not isinstance(prov, list):
        return []
    unknown_vals = []
    for e in prov:
        if not isinstance(e, dict):
            continue
        if str(e.get("status", "")).strip().upper() != "UNKNOWN":
            continue
        val = str(e.get("value", "")).strip()
        if len(val) > 3 and "${" not in val:
            unknown_vals.append(val)
    if not unknown_vals:
        return []
    specs_blob = " ".join(
        json.dumps(o.get("verify") or {}, ensure_ascii=False)
        for o in (parsed.get("required_outcomes") or [])
        if isinstance(o, dict))
    return [uv for uv in unknown_vals if uv in specs_blob]


def _parse_seed_json(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON object from an LLM response, tolerating prose fences.

    Uses ``json.JSONDecoder.raw_decode``, not brace-counting: raw_decode is
    JSON-aware (tracks strings + escapes), so it stops at the right outer brace
    even when string values contain escaped curly braces (e.g. regex ``"\\{"``).
    The old balance-counter silently dropped any outcome whose verify.patterns
    held an escaped brace, returning ``None`` → empty ledger.
    """
    if not raw:
        return None
    s = raw.strip()

    # Strip Qwen3.5-VL thinking prefix. With QWEN35_THINKING=1 the model emits
    # ``<reasoning>...</think>\n{JSON}``; without stripping, find-first-`{`
    # could land on a brace inside the reasoning block and raw_decode fails.
    think_end = s.rfind("</think>")
    if think_end >= 0:
        s = s[think_end + len("</think>"):].lstrip()

    # strip fenced code blocks
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1)

    start = s.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(s, idx=start)
        return obj if isinstance(obj, dict) else None
    except Exception as e:
        logger.warning("[LedgerInit] JSON parse error: %s", e)
        return None


def _build_context(
    parsed: Optional[Dict[str, Any]],
    *,
    a11y_text: Optional[str],
    active_url: Optional[str],
) -> InitialContext:
    ctx_d = (parsed or {}).get("initial_context") or {}
    # Prefer LLM-parsed, fall back to caller hints.
    return InitialContext(
        open_tabs=list(ctx_d.get("open_tabs") or []),
        active_url=(ctx_d.get("active_url") or active_url),
        active_app=ctx_d.get("active_app"),
        window_title=ctx_d.get("window_title"),
    )


def _parse_verify_spec(raw: Any) -> Optional["VerifySpec"]:
    """Parse a verify dict from LLM output into a VerifySpec.

    Returns None on missing/malformed/unknown-kind input. Invalid specs are
    dropped silently — the outcome falls back to the legacy LLM KeyNode path."""
    from mm_agents.structagent.ledger.core.ledger import VerifySpec
    if not isinstance(raw, dict):
        return None
    kind = (raw.get("kind") or "").strip().lower()
    if kind not in ("file_grep", "url_match", "a11y_match",
                    "shell_command", "calc_verify", "impress_verify",
                    "writer_verify"):
        return None
    # Cap proximity_lines at 4 to avoid cross-entry false positives in
    # list-style files (keybindings.json): same-entry "key"+"command" sit ~1
    # line apart, so 3 catches them; 5+ swallows the next entry's fields.
    prox_raw = raw.get("proximity_lines")
    if isinstance(prox_raw, int) and prox_raw > 0:
        prox = min(prox_raw, 4)
    else:
        prox = None
    # Dual-channel safety net (KeyNode LLM path, a11y_match outcomes). Lists of
    # free-form single-sentence clauses; non-string elements dropped.
    must_hold = [
        s for s in (raw.get("visual_must_hold") or [])
        if isinstance(s, str) and s.strip()
    ] or None
    must_not = [
        s for s in (raw.get("visual_must_not_hold") or [])
        if isinstance(s, str) and s.strip()
    ] or None
    cmd = raw.get("command")
    cmd_argv = (
        [str(x) for x in cmd if isinstance(x, (str, int, float))] or None
        if isinstance(cmd, list) else None
    )
    exit_raw = raw.get("expected_exit_code", 0)
    if isinstance(exit_raw, bool):
        # bool is an int subclass; treat True/False as bogus.
        exit_code: Optional[int] = 0
    elif isinstance(exit_raw, int):
        exit_code = exit_raw
    else:
        exit_code = 0
    calc_checks_raw = raw.get("calc_checks")
    calc_checks = (
        [c for c in calc_checks_raw if isinstance(c, dict)]
        if isinstance(calc_checks_raw, list) else None
    ) or None
    impress_checks_raw = raw.get("impress_checks")
    impress_checks = (
        [c for c in impress_checks_raw if isinstance(c, dict)]
        if isinstance(impress_checks_raw, list) else None
    ) or None
    writer_checks_raw = raw.get("writer_checks")
    writer_checks = (
        [c for c in writer_checks_raw if isinstance(c, dict)]
        if isinstance(writer_checks_raw, list) else None
    ) or None
    spec = VerifySpec(
        kind=kind,
        file_path=raw.get("file_path") if isinstance(raw.get("file_path"), str) else None,
        patterns=[p for p in (raw.get("patterns") or []) if isinstance(p, str)] or None,
        all_must_match=bool(raw.get("all_must_match", True)),
        proximity_lines=prox,
        calc_checks=calc_checks,
        impress_checks=impress_checks,
        writer_checks=writer_checks,
        url_pattern=raw.get("url_pattern") if isinstance(raw.get("url_pattern"), str) else None,
        tag=raw.get("tag") if isinstance(raw.get("tag"), str) else None,
        text_contains=[t for t in (raw.get("text_contains") or []) if isinstance(t, str)] or None,
        state_contains=[s for s in (raw.get("state_contains") or []) if isinstance(s, str)] or None,
        visual_must_hold=must_hold,
        visual_must_not_hold=must_not,
        command=cmd_argv,
        expected_substring=[
            s for s in (raw.get("expected_substring") or [])
            if isinstance(s, str)
        ] or None,
        forbidden_substring=[
            s for s in (raw.get("forbidden_substring") or [])
            if isinstance(s, str)
        ] or None,
        expected_exit_code=exit_code,
    )
    if not spec.is_valid():
        return None
    # Soft-validate the dual-channel net on a11y_match. Missing
    # visual_must_hold is logged, not fatal — falls back to a11y-only judging.
    if kind == "a11y_match" and not must_hold:
        logger.warning(
            "[LedgerInit] a11y_match outcome missing visual_must_hold "
            "(dual-channel safety net not provided; falling back to "
            "a11y-only LLM judging which is prone to false positives "
            "from transient overlay rows)"
        )
    return spec


def _repair_or_drop_unverifiable_outcomes(
    outcomes: List[Outcome],
    *,
    instruction: str,
    initial_obs: Optional[Dict[str, Any]],
    call_llm: Optional[Callable],
    model: Optional[str],
    a11y_text: Optional[str] = None,
    active_url: Optional[str] = None,
    domain: Optional[str] = None,
) -> List[Outcome]:
    """Guarantee every shipped outcome has a RUNNABLE verify.

    An outcome with ``verify=None`` (LLM emitted none, or one failing
    ``is_valid()``) can never be verified — deterministic path has nothing to
    run, soft-detector can't observe states like "data on clipboard". It sits
    ``pending`` and traps ``next_focus_outcome`` (focus never advances, its
    downstream never focusable). The leaf-gate shields DONE, but the focus trap
    and misleading prompt remain.

    Per outcome without a valid verify:
      1. REPAIR — one ``reauthor_outcome_verifier`` call (injects the app's
         structured verify schema, e.g. calc_verify incl. clipboard_has_content).
      2. If repair fails: KEEP it with ``verify=None`` (leaf or intermediate).
         [FIX keep_unverifiable_outcomes] We used to drop intermediates here on
         "a downstream leaf proves the intermediate ran". That broke 82e3c869
         (put images in folder → zip): "images_moved" had no verify, was
         dropped, "zip exists" passed on an EMPTY zip → DONE on nothing.
         Keeping it leaves a visible "no_verify" entry so the agent sees the
         imperative exists, and the soft-detector can still judge from
         screenshots. Roll back by restoring the prior drop+re-wire branch.

    Does NOT rely on the LLM — repair is best-effort, soft-detector is the
    backstop. Outcomes that all carry valid verifies return unchanged.
    """
    if not outcomes:
        return outcomes

    def _has_verify(o: Outcome) -> bool:
        v = getattr(o, "verify", None)
        try:
            return v is not None and v.is_valid()
        except Exception:
            return False

    bad = [o for o in outcomes if not _has_verify(o)]
    if not bad:
        return outcomes

    depended: set = set()
    for o in outcomes:
        for d in (getattr(o, "depends_on", None) or []):
            depended.add(d)

    dropped: Dict[str, List[str]] = {}   # dropped id -> its depends_on
    for o in bad:
        new_spec = None
        if call_llm is not None:
            try:
                new_spec = reauthor_outcome_verifier(
                    instruction, o,
                    "This outcome was emitted with NO usable verify "
                    "spec. Author a valid, runnable verify — use the "
                    "structured verify kind for the outcome's app.",
                    None, initial_obs, call_llm, model,
                    a11y_text=a11y_text, active_url=active_url,
                    domain=domain,
                )
            except Exception as e:
                logger.warning("[LedgerInit] verify-repair reauthor "
                               "raised for %s: %s", o.id, e)
        if new_spec is not None:
            try:
                _ok = new_spec.is_valid()
            except Exception:
                _ok = False
            if _ok:
                o.verify = new_spec
                logger.info("[LedgerInit] repaired null verify: "
                            "outcome=%s -> kind=%s", o.id, new_spec.kind)
                continue
        # Repair failed — [FIX a11y_fallback_verify] Synthesize an a11y_match
        # verify grounded on a single visual clause (the evidence_hint
        # verbatim) for KeyNode's LLM to judge against the screenshot. No
        # keyword extraction: the description IS the judge prompt. Better than
        # inert ``verify=None`` and more honest than a fabricated shell verify.
        # Roll back to the prior ``kept with verify=None`` warning one-liner.
        _role = "INTERMEDIATE" if o.id in depended else "LEAF"
        _hint = (getattr(o, "evidence_hint", None) or "").strip() \
                or (getattr(o, "description", None) or "").strip()
        if _hint:
            o.verify = VerifySpec(
                kind="a11y_match",
                visual_must_hold=[_hint],
            )
            logger.info(
                "[LedgerInit] %s outcome=%s — synthesized a11y_match "
                "fallback (visual_must_hold=<evidence_hint>)",
                _role, o.id)
        else:
            logger.warning(
                "[LedgerInit] %s outcome=%s has no runnable verify, "
                "repair failed, and evidence_hint is empty — kept "
                "with verify=None", _role, o.id)

    return outcomes


_SLOT_TYPES = {"url", "text", "path", "numeric", "cell_ref"}


def _build_slots(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse the top-level ``slots`` array into ``{name: Slot}``.

    Drops malformed entries silently — an unknown ``${name}`` is treated as
    unresolved (returns None), so a missing slot degrades to the LLM judge path
    rather than crashing. Unknown ``type`` coerces to "text" (most permissive)."""
    from mm_agents.structagent.ledger.core.ledger import VerifySpecSlot
    raw_list = (parsed or {}).get("slots") or []
    out: Dict[str, VerifySpecSlot] = {}
    if not isinstance(raw_list, list):
        return out
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_")
        if not name or name in out:
            continue
        slot_type = (item.get("type") or "text").strip().lower()
        if slot_type not in _SLOT_TYPES:
            slot_type = "text"
        out[name] = VerifySpecSlot(
            name=name,
            type=slot_type,
            description=(item.get("description") or "").strip()[:240],
            source_hint=(item.get("source_hint") or "").strip()[:240],
        )
    return out


def _build_outcomes(parsed: Optional[Dict[str, Any]]) -> List[Outcome]:
    raw_list = (parsed or {}).get("required_outcomes") or []
    out: List[Outcome] = []
    seen_ids = set()
    # Two-pass: collect normalized ids first so depends_on can be validated
    # against the real id set (the LLM emits semantic-looking deps that don't
    # match the normalized id; keep only deps that resolve).
    parsed_items: List[Dict[str, Any]] = []
    for item in raw_list[:8]:  # hard cap
        if not isinstance(item, dict):
            continue
        oid = (item.get("id") or "").strip()
        desc = (item.get("description") or "").strip()
        hint = (item.get("evidence_hint") or "").strip()
        if not oid or not desc or not hint:
            continue
        oid = re.sub(r"[^a-zA-Z0-9_]+", "_", oid).strip("_").lower()
        if not oid or oid in seen_ids:
            continue
        seen_ids.add(oid)
        parsed_items.append({"id": oid[:60], "desc": desc[:240], "hint": hint[:240], "raw": item})

    valid_ids = {p["id"] for p in parsed_items}
    for p in parsed_items:
        # depends_on: ids that must be in valid_ids. Drop self-refs and unknown
        # ids silently — safe even if the LLM hallucinates a dep on a sibling
        # that didn't parse.
        deps_raw = p["raw"].get("depends_on") or []
        if not isinstance(deps_raw, list):
            deps_raw = []
        deps: List[str] = []
        for d in deps_raw:
            if not isinstance(d, str):
                continue
            d_norm = re.sub(r"[^a-zA-Z0-9_]+", "_", d.strip()).strip("_").lower()
            if d_norm and d_norm != p["id"] and d_norm in valid_ids and d_norm not in deps:
                deps.append(d_norm)
        verify = _parse_verify_spec(p["raw"].get("verify"))
        # Multi-app: author tags each outcome with the app whose state proves
        # it. Normalized to the underscore form used for domain routing.
        # Absent/non-string (single-app) → None.
        app_raw = p["raw"].get("app")
        app = None
        if isinstance(app_raw, str) and app_raw.strip():
            app = app_raw.strip().lower().replace(" ", "_").replace("-", "_")
        out.append(Outcome(
            id=p["id"],
            description=p["desc"],
            evidence_hint=p["hint"],
            depends_on=deps,
            verify=verify,
            app=app,
        ))
    return out


# --------------------------------------------------------------------------- #
# Reauthor — single-outcome verify spec rewrite
# --------------------------------------------------------------------------- #

_REAUTHOR_PREAMBLE = """REAUTHOR MODE — the previous verify spec for one
outcome has been rejected by the deterministic verifier on every recent
run, AND an independent diagnoser has classified the failure as
"verifier_mismatch" (i.e. the actor has substantially completed the
work but the verify spec doesn't recognise it).

Your job: REWRITE the verify spec so it correctly captures the
outcome's success state. The full authoring rules above still apply
(KNOWN APP CONFIG PATHS, pattern-writing tips, dual-channel gates for
a11y_match, action-completion rule). You MAY change the ``kind`` if the
diagnosis shows the original kind was wrong (e.g. file_grep →
a11y_match when the state lives only in the UI; a11y_match →
url_match when it's actually a URL). Keep the outcome ``id``,
``description`` and ``depends_on`` stable.

CRITICAL — STRUCTURED VERIFY KINDS ARE STABLE.
When the existing spec is ``calc_verify`` or ``impress_verify``,
DO NOT downgrade to ``shell_command``. The framework generates these
specs deterministically from a fixed op+kwargs schema, so the spec
itself is never wrong — a repeated FAIL means the actor has not yet
reached the target state, not that the patterns are off. Rewriting
``impress_verify`` as ``shell_command`` with python-pptx (not
installed on the VM) or an invented ``uno_helper`` import is a
strict regression: the new spec ALWAYS errors and the outcome can
never satisfy. Stay on ``calc_verify`` / ``impress_verify`` and
adjust the ``calc_checks`` / ``impress_checks`` list if (and only
if) the schema (op name, kwargs) genuinely needs to change to
capture the success state. Refer to the CALC_VERIFY / IMPRESS_VERIFY
block injected into the user message for valid op names + kwargs.

Output format: the SAME JSON schema as the init call, but with
EXACTLY ONE entry in ``required_outcomes`` (the rewritten outcome).
You can leave ``initial_context`` empty (e.g. all-null fields); only
``required_outcomes[0].verify`` is consumed by the caller. Do NOT add
new outcomes or rename the id."""


def _format_verify_for_reauthor(spec: Optional[VerifySpec]) -> str:
    """Render a VerifySpec back to a JSON-ish dict the LLM can read."""
    if spec is None:
        return "(no previous verify spec — original outcome had none)"
    fields: Dict[str, Any] = {"kind": spec.kind}
    if spec.kind == "file_grep":
        fields["file_path"] = spec.file_path
        fields["patterns"] = spec.patterns or []
        fields["all_must_match"] = spec.all_must_match
        if spec.proximity_lines is not None:
            fields["proximity_lines"] = spec.proximity_lines
    elif spec.kind == "url_match":
        fields["url_pattern"] = spec.url_pattern
    elif spec.kind == "a11y_match":
        if spec.tag:
            fields["tag"] = spec.tag
        if spec.text_contains:
            fields["text_contains"] = spec.text_contains
        if spec.state_contains:
            fields["state_contains"] = spec.state_contains
        if spec.visual_must_hold:
            fields["visual_must_hold"] = spec.visual_must_hold
        if spec.visual_must_not_hold:
            fields["visual_must_not_hold"] = spec.visual_must_not_hold
    elif spec.kind == "shell_command":
        fields["command"] = list(spec.command or [])
        if spec.expected_substring:
            fields["expected_substring"] = list(spec.expected_substring)
        if spec.forbidden_substring:
            fields["forbidden_substring"] = list(spec.forbidden_substring)
        if spec.expected_exit_code is not None:
            fields["expected_exit_code"] = spec.expected_exit_code
    return json.dumps(fields, indent=2, ensure_ascii=False)


def _format_trace_for_reauthor(trace: Optional[Dict[str, Any]]) -> str:
    """Compact summary of the last failing verifier trace (with file-content
    preview when available) — same info the diagnoser saw."""
    if not isinstance(trace, dict) or not trace:
        return "(no trace available)"
    kind = trace.get("kind") or "?"
    parts: List[str] = [f"kind: {kind}"]
    if kind == "file_grep":
        parts.append(f"resolved_path: {trace.get('resolved_path', '?')}")
        parts.append(f"file_size_bytes: {trace.get('file_size_bytes', '?')}")
        for entry in (trace.get("per_pattern_matches") or []):
            ml = entry.get("matched_lines") or []
            parts.append(
                f"pattern {entry.get('pattern', '?')!r}: {len(ml)} match(es)"
            )
        head = (trace.get("file_content_head") or "")[:1200]
        if head:
            parts.append("")
            parts.append("Current file content (first 1200 chars):")
            parts.append("```")
            parts.append(head)
            parts.append("```")
    elif kind == "url_match":
        parts.append(f"url_extracted: {trace.get('url_extracted', '(none)')}")
    elif kind == "a11y_match":
        parts.append(f"rows_scanned: {trace.get('rows_scanned', 0)}")
        parts.append(f"matched_row: {trace.get('matched_row') or '(none)'}")
    elif kind == "shell_command":
        parts.append(f"command: {trace.get('command') or []}")
        parts.append(f"exit_code: {trace.get('exit_code')}")
        if trace.get("matched_substring"):
            parts.append(f"matched_substring: {trace['matched_substring']!r}")
        if trace.get("verdict_reason"):
            parts.append(f"verdict_reason: {trace['verdict_reason']}")
        head = (trace.get("stdout_head") or "")[:1200]
        if head:
            parts.append("")
            parts.append("Command stdout (first 1200 chars, stderr merged):")
            parts.append("```")
            parts.append(head)
            parts.append("```")
    return "\n".join(parts)


def reauthor_outcome_verifier(
    instruction: str,
    outcome: Outcome,
    diagnosis_text: str,
    last_trace: Optional[Dict[str, Any]],
    initial_obs: Optional[Dict[str, Any]],
    call_llm: Callable,
    model: Optional[str] = None,
    *,
    a11y_text: Optional[str] = None,
    active_url: Optional[str] = None,
    domain: Optional[str] = None,
) -> Optional[VerifySpec]:
    """Re-author the ``verify`` spec for one outcome whose deterministic
    verifier keeps rejecting DONE despite (per the diagnoser) the work being
    substantially complete.

    Reuses :data:`_SYSTEM_PROMPT` for the full authoring rules, plus a
    ``REAUTHOR MODE`` preamble constraining it to one outcome. Returns a fresh
    :class:`VerifySpec`, or ``None`` on LLM/parse/validity failure (caller
    typically leaves the existing spec as a no-op patch).
    """
    user_content: List[Dict[str, Any]] = []

    screenshot_bytes = (
        initial_obs.get("screenshot")
        if isinstance(initial_obs, dict) else None
    )
    if screenshot_bytes:
        try:
            b64 = base64.b64encode(screenshot_bytes).decode("ascii")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        except Exception as e:
            logger.warning("[ReauthorVerifier] screenshot encode failed: %s", e)

    text_parts: List[str] = [
        f"Task instruction:\n{instruction.strip()}",
        "",
        f"Outcome to re-author (keep id stable):",
        f"  id:            {outcome.id}",
        f"  description:   {outcome.description}",
        f"  evidence_hint: {outcome.evidence_hint}",
        f"  depends_on:    {list(outcome.depends_on or [])}",
        "",
        "Previous verify spec (rejected by deterministic verifier):",
        _format_verify_for_reauthor(outcome.verify),
        "",
        "Diagnoser feedback (independent LLM, classified as "
        "verifier_mismatch):",
        diagnosis_text.strip() or "(no diagnosis text)",
        "",
        "Last failing verifier trace:",
        _format_trace_for_reauthor(last_trace),
    ]
    if a11y_text:
        text_parts.append("")
        text_parts.append(
            "Accessibility tree of the current screen (aux):\n" + a11y_text
        )
    if active_url:
        text_parts.append(f"\nActive tab URL (aux): {active_url}")
    # Reauthor targets ONE outcome, so the domain is that outcome's own ``app``
    # tag — NOT ``domain`` (only related_apps[0]). Without this, re-authoring a
    # calc outcome on a writer-first task got the writer schema (or none) and
    # the LLM fell back to a broken shell_command UNO.
    _reauthor_dom = (getattr(outcome, "app", None) or domain or "")
    if _reauthor_dom:
        try:
            from mm_agents.structagent.domain import get_verifier_knowledge
            dk = get_verifier_knowledge(_reauthor_dom)
            if dk:
                text_parts.append(
                    f"\nDomain-specific knowledge for `{_reauthor_dom}` "
                    f"(applies to this outcome's app):\n{dk}"
                )
        except Exception as e:
            logger.info("[ReauthorVerifier] domain_knowledge load failed: %s", e)
    # Inject the domain's structured-verify schema so the reauthor LLM keeps
    # emitting calc_verify / impress_verify instead of downgrading to
    # shell_command + python-pptx (uninstalled) or an invented uno_helper
    # import. init_ledger already injects these; reauthor was missing them.
    dom_lc = _reauthor_dom.lower()
    if dom_lc == "libreoffice_calc":
        try:
            from mm_agents.structagent.actions.calc.actions import calc_verify_schema_block
            text_parts.append(calc_verify_schema_block())
        except Exception as e:
            logger.info(
                "[ReauthorVerifier] calc_verify schema load failed: %s", e)
    elif dom_lc == "libreoffice_impress":
        try:
            from mm_agents.structagent.actions.impress.actions import impress_verify_schema_block
            text_parts.append(impress_verify_schema_block())
        except Exception as e:
            logger.info(
                "[ReauthorVerifier] impress_verify schema load failed: %s", e)
    elif dom_lc == "libreoffice_writer":
        try:
            from mm_agents.structagent.actions.writer.actions import writer_verify_schema_block
            text_parts.append(writer_verify_schema_block())
        except Exception as e:
            logger.info(
                "[ReauthorVerifier] writer_verify schema load failed: %s", e)
    text_parts.append(
        "\nReturn the JSON object (same schema as init_ledger) with "
        "EXACTLY one entry in required_outcomes — the rewritten "
        f"outcome `{outcome.id}`. Nothing else."
    )
    user_content.append({"type": "text", "text": "\n\n".join(text_parts)})

    sys_prompt = _SYSTEM_PROMPT + "\n\n" + _REAUTHOR_PREAMBLE
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        raw = call_llm(
            {"model": model, "messages": messages,
             "max_tokens": 1500, "temperature": 0.0, "top_p": 0.9},
            model,
        ) or ""
    except Exception as e:
        logger.warning("[ReauthorVerifier] LLM call failed: %s", e)
        raw = ""

    parsed = _parse_seed_json(raw) if raw else None
    if not parsed:
        logger.info(
            "[ReauthorVerifier] no parseable JSON from LLM for outcome=%s",
            outcome.id,
        )
        return None
    raw_list = parsed.get("required_outcomes") or []
    if not isinstance(raw_list, list) or not raw_list:
        logger.info(
            "[ReauthorVerifier] empty required_outcomes for outcome=%s",
            outcome.id,
        )
        return None
    # Pick the entry whose id matches; fall back to the first if the LLM
    # renamed it (call site keeps the original id — only the verify changes).
    target = None
    for item in raw_list:
        if isinstance(item, dict) and (item.get("id") or "").strip().lower() == outcome.id.lower():
            target = item
            break
    if target is None and isinstance(raw_list[0], dict):
        target = raw_list[0]
    if not isinstance(target, dict):
        return None
    new_spec = _parse_verify_spec(target.get("verify"))
    if new_spec is None:
        logger.info(
            "[ReauthorVerifier] LLM produced invalid verify for outcome=%s",
            outcome.id,
        )
        return None
    logger.info(
        "[ReauthorVerifier] outcome=%s: kind %s -> %s",
        outcome.id,
        getattr(outcome.verify, "kind", "(none)"),
        new_spec.kind,
    )
    return new_spec


def apply_reauthor_pass(
    *,
    ledger: "ProgressLedger",
    instruction: str,
    obs: Dict[str, Any],
    bad_outcomes: List[Outcome],
    call_llm: Callable,
    model: Optional[str] = None,
    a11y_text: Optional[str] = None,
    active_url: Optional[str] = None,
    domain: Optional[str] = None,
) -> None:
    """Rewrite the ``verify`` spec of each non-verified outcome whose diagnoser
    has consecutively flagged ``verifier_mismatch``, via
    :func:`reauthor_outcome_verifier`.

    Gates (all must hold per outcome):
      • G1 ``verifier_mismatch_strikes >= VERIFIER_REAUTHOR_MIN_STRIKES``
      • G2 ``done_rejected_count       >= VERIFIER_REAUTHOR_MIN_DONE_REJECTIONS``
      • G4 ``patch_applied            <  VERIFIER_REAUTHOR_MAX_PATCHES``

    On success: mutates ``outcome.verify`` in place, bumps ``patch_applied``,
    resets ``verifier_mismatch_strikes``, clears ``last_diagnosis_signature``
    so the next force-replan re-diagnoses against the new spec. Failures leave
    state untouched.
    """
    from mm_agents.structagent.ledger.core.timeline import (
        VERIFIER_REAUTHOR_MIN_STRIKES,
        VERIFIER_REAUTHOR_MIN_DONE_REJECTIONS,
        VERIFIER_REAUTHOR_MAX_PATCHES,
    )
    # State lives on each Outcome directly (was OutcomeStateCache).
    candidates: List[Outcome] = []
    for o in bad_outcomes:
        cause = o.last_diagnosis_cause or ""
        strikes = o.verifier_mismatch_strikes
        rejections = o.done_rejected_count
        patches = o.patch_applied
        if cause != "verifier_mismatch":
            continue
        if strikes < VERIFIER_REAUTHOR_MIN_STRIKES:
            continue
        if rejections < VERIFIER_REAUTHOR_MIN_DONE_REJECTIONS:
            continue
        if patches >= VERIFIER_REAUTHOR_MAX_PATCHES:
            continue
        candidates.append(o)
    if not candidates:
        return

    for o in candidates:
        last_trace = dict(o.last_verifier_trace or {})
        last_diag = o.last_diagnosis or ""
        try:
            new_spec = reauthor_outcome_verifier(
                instruction=instruction,
                outcome=o,
                diagnosis_text=last_diag,
                last_trace=last_trace,
                initial_obs=obs,
                call_llm=call_llm,
                model=model,
                a11y_text=a11y_text,
                active_url=active_url,
                domain=domain,
            )
        except Exception as e:
            logger.info("[Reauthor] outcome=%s call failed: %s", o.id, e)
            continue
        if new_spec is None:
            logger.info(
                "[Reauthor] outcome=%s: LLM returned no usable spec", o.id)
            continue
        old_kind = getattr(o.verify, "kind", "(none)") if o.verify else "(none)"
        o.verify = new_spec
        o.patch_applied += 1
        o.verifier_mismatch_strikes = 0
        o.last_diagnosis_signature = None
        logger.info(
            "[Reauthor] outcome=%s: verify rewritten %s -> %s (patches=%d)",
            o.id, old_kind, new_spec.kind, o.patch_applied,
        )
