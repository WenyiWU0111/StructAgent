"""Structured Calc action handlers — flat JSON kwargs, no nested Python.

Small decomposer models (qwen3.5-9b) reliably fail when asked to emit
Python source as a string inside JSON: bracket-mismatch in nested
lists, ``, print(`` instead of ``)\\n``, missing ``=`` prefix on
formulas, hallucinated imports, etc. Each failure is a different
emission bug; rule-based repair cannot exhaustively cover them.

Root fix: stop asking the actor to write Python for Calc. Define a
small set of structured ``calc_*`` actions whose payload is a flat
JSON object with named kwargs. The actor emits the JSON; the
framework deterministically builds the equivalent ``op_*`` call. The
Python source is generated server-side — guaranteed syntactically
correct.

This module centralises three slices:

  * :data:`CALC_ACTION_NAMES` — membership gate for "is this one of
    mine?" Single source of truth for both dispatchers.
  * :func:`step_to_wire` — convert a decomposer's JSON action dict to
    an AST-parseable ``calc_xxx(arg=val, ...)`` wire string. Called
    from ``decomposer_actor._step_to_uitars_line``.
  * :func:`action_to_runtime_script` — given an ``action_type`` and
    parsed kwargs, build the wrapped runtime script (boilerplate +
    op library + actor body) ready to ``exec`` on the VM. Called
    from ``qwen25vl_agent_planner._pa_action_to_pyautogui_code``.

Adding a new ``calc_*`` action means: add the name to
:data:`CALC_ACTION_NAMES`, add a ``_wire_xxx`` validator returning a
wire string, add a ``_build_xxx`` Python-code generator. Both
dispatchers find the handlers automatically via the action_type →
function mapping at the bottom of this file.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ====================================================================
# Membership set + schema injection
# ====================================================================

CALC_ACTION_NAMES: frozenset = frozenset({
    "calc_set_cell",
    "calc_set_range",
    "calc_fill_column",
    "calc_fill_row",
    "calc_fill_blanks",
    "calc_split_text",
    "calc_new_sheet",
    "calc_rename_sheet",
    "calc_pivot_table",
    "calc_sort_range",
    "calc_freeze",
    "calc_format_number",
    "calc_duplicate_range",
    "calc_copy_to_clipboard",
    "calc_hide_range",
    "calc_insert_chart",
    "calc_export_file",
    "calc_set_color",
    "calc_merge_cells",
    "calc_data_validation",
    "calc_reorder_columns",
    "calc_delete_columns",
    "calc_delete_rows",
})


# ====================================================================
# calc_verify — structured JSON verify checks
# ====================================================================
#
# Companion to ``CALC_ACTION_NAMES`` for the verify side. Each
# ``calc_checks`` item is ``{"op": <op>, ...kwargs}``. Kwargs are
# passed through 1:1 to the matching ``verify_*`` helper in
# ``calc_ops_lib.py`` (no renaming, no validation magic — what you
# write is what gets called). The framework builds the python3 -c
# body server-side so the init_ledger LLM never writes Python.
#
# ``CALC_VERIFY_OPS`` is the canonical op → helper mapping; new ops
# get added here and the rest (validator, prompt schema, verifier
# dispatch) follows automatically.

CALC_VERIFY_OPS: Dict[str, str] = {
    "sheet_exists":      "verify_sheet_exists",
    "cell_value":        "verify_cell_value",
    "pivot_exists":      "verify_pivot_exists",
    "cell_format":       "verify_cell_format",
    "sort_order":        "verify_sort_order",
    "freeze_pane":       "verify_freeze_pane",
    "duplicate_range_match":  "verify_duplicate_range_match",
    "row_hidden":        "verify_row_hidden",
    "col_hidden":        "verify_col_hidden",
    "chart_exists":      "verify_chart_exists",
    "file_exists":       "verify_file_exists",
    "cell_color":        "verify_cell_color",
    "cell_merged":       "verify_cell_merged",
    "data_validation":   "verify_data_validation",
    "clipboard_has_content": "verify_clipboard_has_content",
}

# Required kwargs per op. Optional ones are not enforced at validation
# time — the underlying ``verify_*`` helper applies its own defaults.
_CALC_VERIFY_REQUIRED_KWARGS: Dict[str, Tuple[str, ...]] = {
    "sheet_exists":      ("name",),
    "cell_value":        ("sheet", "a1_ref"),
    "pivot_exists":      ("on_sheet",),
    "cell_format":       ("sheet", "range_ref", "format_string"),
    "sort_order":        ("sheet", "range_ref", "sort_col_within_range"),
    "freeze_pane":       (),
    "duplicate_range_match":  ("source_sheet", "source_range",
                          "dest_sheet", "dest_anchor"),
    "row_hidden":        ("sheet", "rows"),
    "col_hidden":        ("sheet", "cols"),
    "chart_exists":      ("on_sheet",),
    "file_exists":       ("path",),
    "cell_color":        ("sheet", "a1_ref"),
    "cell_merged":       ("sheet", "a1_ref"),
    "data_validation":   ("sheet", "range_ref"),
    "clipboard_has_content": (),
}

# For cell_value, exactly one of these expectation kwargs MUST appear
# (anything else is a no-op check). The validator enforces this.
_CELL_VALUE_EXPECT_KEYS: Tuple[str, ...] = (
    "expected_value", "expected_substring",
    "expected_formula", "expected_formula_contains",
    "expected_formula_regex",
)


def validate_calc_verify_check(chk: Dict[str, Any]) -> bool:
    """True iff ``chk`` is a well-formed calc_verify check item.

    Used by ``VerifySpec.is_valid`` so malformed checks drop the whole
    spec (outcome falls back to LLM KeyNode). Validation is structural
    only — value types / sheet existence are checked at run time inside
    the verify_* helper.
    """
    op = chk.get("op")
    if op not in CALC_VERIFY_OPS:
        return False
    for k in _CALC_VERIFY_REQUIRED_KWARGS[op]:
        if k not in chk or chk[k] in (None, ""):
            return False
    if op == "cell_value":
        if not any(chk.get(k) is not None for k in _CELL_VALUE_EXPECT_KEYS):
            return False
    return True


def _normalize_lo_format_string(s):
    """Strip spurious ``\\$`` / ``\\[`` / ``\\]`` escapes from a LO
    number-format string.

    init_ledger writes calc_verify specs as JSON. When the LLM is asked
    to embed a locale-prefix like ``[$-419]0.0`` inside a JSON string
    value, it frequently emits ``[\\$-419]0.0`` — defensively escaping
    ``$`` as if the string were a regex / shell / template literal.
    LO's ``queryKey`` sees the literal backslash, does not recognize the
    string, and returns ``-1`` — so ``verify_cell_format`` permanently
    FAILs even when the actor applied the correct format. Strip those
    three escapes (none have valid meaning in a LO locale prefix) so
    the LLM's hallucinated escape no longer locks the verify.
    """
    if not isinstance(s, str):
        return s
    return s.replace("\\$", "$").replace("\\[", "[").replace("\\]", "]")


def build_calc_verify_python_body(checks: List[Dict[str, Any]]) -> str:
    """Return the python3 -c body string for a list of calc_verify
    checks. ``all`` semantics — the body prints ``PASS`` iff every
    check returns truthy; otherwise prints ``FAIL: check[<i>] (<op>)``
    naming the first failing check.

    The boilerplate (UNO connect block + Calc helper definitions) is
    NOT included here — ``_maybe_wrap_calc_verify_command`` in
    verifiers.py prepends it via cli_run_uno_helpers.build_runtime_script.
    """
    if not checks:
        return "print('FAIL: no checks')"
    lines: List[str] = ["_results = []"]
    for chk in checks:
        op = chk["op"]
        fn = CALC_VERIFY_OPS[op]
        kwargs = {k: v for k, v in chk.items() if k != "op"}
        if op == "cell_format":
            kwargs = dict(kwargs)
            kwargs["format_string"] = _normalize_lo_format_string(
                kwargs.get("format_string"))
        call = _kw_call(fn, kwargs)
        lines.append(f"_results.append(bool({call}))")
    lines.append("_ok = all(_results)")
    lines.append(
        "if _ok:\n    print('PASS')\nelse:\n"
        "    _i = _results.index(False)\n"
        "    print('FAIL: check[' + str(_i) + '] ' + "
        "{idx_to_op})".format(
            idx_to_op="repr(_OPS[_i])",
        )
    )
    # _OPS list precedes the body so the error message can name which op
    ops_repr = [chk["op"] for chk in checks]
    return "_OPS = " + repr(ops_repr) + "\n" + "\n".join(lines)


def calc_verify_schema_block() -> str:
    """Prompt block describing the structured ``calc_verify`` kind for
    init_ledger. Injected into the init_ledger user content ONLY for
    libreoffice_calc tasks (Writer / Impress still use the python3 -c
    shape since their verify needs raw UNO; Calc has the structured
    helpers).
    """
    return _CALC_VERIFY_SCHEMA_BLOCK


_CALC_VERIFY_SCHEMA_BLOCK = """\
CALC_VERIFY — structured verify spec for libreoffice_calc

Calc tasks MUST use ``kind="calc_verify"`` (NOT ``kind="shell_command"``
with embedded ``python3 -c``). The framework builds the verifier
subprocess from flat JSON; you never write Python, never write argv,
never worry about quote escaping or PASS-token plumbing.

Shape:

  "verify": {
    "kind": "calc_verify",
    "calc_checks": [
      {"op": "<op-name>", ...kwargs},
      ...
    ]
  }

Multiple checks AND together — the outcome verifies iff EVERY check
returns truthy. Failure message names the first failing check + its
``op`` so the trace tells you which check broke.

# CRITICAL — do NOT hallucinate precise anchors

Every check below has REQUIRED structural kwargs and OPTIONAL
expected-value anchors (``expected_value`` / ``expected_substring``
/ ``expected_first_value`` / ``expected_last_value`` / ``title`` /
``expected_series_count`` / ``expected_rows`` / ``expected_cols``
/ ``allowed_values`` …). An anchor permanently FAILs the verify
if its value doesn't match the actual post-action cell state.

Rule: emit an anchor ONLY when you can read its exact value either
(a) VERBATIM from the task instruction text (e.g. "set cell to 42"
⇒ ``expected_value``=42; "title 'Q1 sales'" ⇒ ``title``="Q1 sales"),
or (b) VERBATIM from the SP block as plain text/number (not a
formatted date, not a scientific-notation float, not a truncated
sample row from a `(N rows omitted)` table).

Otherwise OMIT the anchor — the structural / monotonic-only check
alone passes when the agent does the right thing. A hallucinated
anchor (e.g. ``expected_first_value=42160`` invented from a date-
serial column; ``title="Sales by Date"`` invented for a task that
names no title; ``expected_series_count=35`` invented from data-
point count) GUARANTEES task failure even when the agent's work
is correct.

# Ops

* ``sheet_exists`` — sheet present in the workbook
  {"op":"sheet_exists", "name":<sheet-name>}

* ``cell_value`` — one cell against expectations. Supply EXACTLY ONE
  expectation kwarg:
  {"op":"cell_value", "sheet":<sheet>, "a1_ref":<single-cell-a1>,
   "expected_value":<int|float|str>}          // value compare
  {"op":"cell_value", "sheet":<sheet>, "a1_ref":<a1>,
   "expected_substring":<text>}               // text contains
  {"op":"cell_value", "sheet":<sheet>, "a1_ref":<a1>,
   "expected_formula":<full =formula>}        // exact formula match
  {"op":"cell_value", "sheet":<sheet>, "a1_ref":<a1>,
   "expected_formula_contains":<plain substring of formula>}
  {"op":"cell_value", "sheet":<sheet>, "a1_ref":<a1>,
   "expected_formula_regex":<regex matched with re.search>}

* ``pivot_exists`` — at least one pivot on ``on_sheet`` (optionally
  named)
  {"op":"pivot_exists", "on_sheet":<sheet>, "pivot_name":<optional>}

* ``cell_format`` — every non-empty cell in range has the format
  {"op":"cell_format", "sheet":<sheet>, "range_ref":<a1-range>,
   "format_string":<excel-format-string>}
  ``format_string`` is consumed verbatim by LO's ``queryKey``. Write the
  bytes EXACTLY as LO accepts them — do NOT backslash-escape ``$``,
  ``[``, or ``]`` inside the JSON. A locale prefix is written as
  ``[$-419]`` (raw ``$``), NOT ``[\$-419]``. A literal ``\`` before
  one of those chars makes ``queryKey`` return -1 and the verify will
  permanently FAIL.
  Concrete examples — copy these byte-for-byte:
    "[$-419]0.0"        Russian locale, 1 decimal (renders 0.1 → "0,1")
    "[$-419]#,##0.0"    Russian locale with thousands grouping
    "[$-407]0.00"       German locale, 2 decimals
    "0.0,,\" M\""       millions scale, " M" suffix (1500000 → "1.5 M")
    "0.0,,,\" B\""      billions scale, " B" suffix (1.5e9 → "1.5 B")
    "0.00%"             percent, 2 decimals
    "0.00E+00"          scientific, 2 decimals

* ``sort_order`` — data column monotonic
  {"op":"sort_order", "sheet":<sheet>, "range_ref":<a1-range>,
   "sort_col_within_range":<0-based int>,
   "ascending":true, "contains_header":true}

* ``freeze_pane`` — workbook has freeze in effect
  {"op":"freeze_pane"}

* ``duplicate_range_match`` — spot-check that source cells landed at dest
  {"op":"duplicate_range_match", "source_sheet":<sh>, "source_range":<a1>,
   "dest_sheet":<sh>, "dest_anchor":<single-cell-a1>,
   "n_rows_to_check":3}

* ``clipboard_has_content`` — the system clipboard is non-empty. The
  companion verify for a ``calc_copy_to_clipboard`` action: confirms
  the copy actually populated the clipboard (the clipboard is
  transient, so this is the only way to verify the copy short of
  pasting). No kwargs.
  {"op":"clipboard_has_content"}

* ``row_hidden`` / ``col_hidden`` — every listed row / column matches
  the expected hidden state. Pass ``hidden=true`` (default) for "row
  is hidden", ``hidden=false`` for "row is visible".
  {"op":"row_hidden", "sheet":<sh>, "rows":[3, 6, 8],
   "hidden":true}
  {"op":"col_hidden", "sheet":<sh>, "cols":["A", "C"],
   "hidden":true}

* ``chart_exists`` — at least one embedded chart matches on
  ``on_sheet``. Optional filters: ``name`` (chart name),
  ``chart_type`` ("column"/"bar"/"line"/"pie"/"scatter"/"area"),
  ``title`` (substring — ONLY emit when the task instruction
  literally names the chart title; if the task does NOT name a
  title, OMIT this filter — hallucinated titles lock verify in
  permanent FAIL while the actor picks its own equally arbitrary
  string), ``expected_series_count`` (int — DataSeries count on
  the chart).
  ``expected_series_count`` — OMIT BY DEFAULT. Emit ONLY when the
  task wording literally names the series count ("three line
  charts" → 3; "two bars" → 2; "a chart with bar AND line series"
  → 2). For the common shape "create a line/bar chart with X on
  X-axis and Y on Y-axis", DO NOT emit it — the LLM cannot
  reliably distinguish series count from data-point count, and a
  wrong value (e.g. 35 when the chart has 1 series of 35 points)
  permanently FAILs verify. The form (2) emission protocol in
  the planner already constrains series count by construction;
  this verify filter is a fallback for tasks where the count is
  explicit.
  {"op":"chart_exists", "on_sheet":<sh>, "chart_type":"line"}
  {"op":"chart_exists", "on_sheet":<sh>, "chart_type":"bar",
   "title":"<substring from task wording>"}

* ``file_exists`` — a file at ``path`` exists on the VM with size at
  least ``min_bytes`` (default 1). Pair this with ``calc_export_file``
  to confirm the PDF / CSV got written.
  {"op":"file_exists", "path":<absolute path on VM>}
  {"op":"file_exists", "path":<absolute path>, "min_bytes":<int>}

* ``cell_color`` — verify ``CellBackColor`` / ``CharColor`` / bold /
  italic. Each kwarg is checked only when present.
  Single-cell form (``where_formula`` omitted):
    {"op":"cell_color", "sheet":<sh>, "a1_ref":<single cell>,
     "bgcolor":<"#RRGGBB">}
    {"op":"cell_color", "sheet":<sh>, "a1_ref":<single cell>,
     "font_color":<"#RRGGBB">, "bold":<bool>}
  Range + predicate form — REQUIRED for tasks that paired
  ``calc_set_color`` with ``where_formula``. Pass the SAME range +
  predicate the op used; the verify checks every cell in the range
  where the predicate is True carries the expected colour. Picking
  an arbitrary single cell for a per-cell predicate task usually
  lands on one that does NOT satisfy the predicate, locking the
  verify in a permanent FAIL even when the agent ran the op
  correctly. ``where_formula`` MUST be set whenever the task is
  "highlight cells matching <X>" (weekends, max value, blanks,
  threshold, etc.):
    {"op":"cell_color", "sheet":<sh>, "a1_ref":<multi-cell range>,
     "bgcolor":<"#RRGGBB">,
     "where_formula":"WEEKDAY({cell};2)>=6"}      # weekends
    {"op":"cell_color", "sheet":<sh>, "a1_ref":<col range>,
     "font_color":<"#RRGGBB">,
     "where_formula":"{cell}=MAX($<col>$<r1>:$<col>$<rN>)"}  # highest
    {"op":"cell_color", "sheet":<sh>, "a1_ref":<col range>,
     "font_color":<"#RRGGBB">,
     "where_formula":"{cell}=MIN($<col>$<r1>:$<col>$<rN>)"}  # lowest
  Optional ``check_unmatched_unstyled":true`` ALSO requires that
  cells where the predicate is False do NOT carry the colour —
  catches the "agent coloured the whole range" failure mode.

* ``cell_merged`` — the cell at ``a1_ref`` is the anchor of a merged
  region (i.e. ``IsMerged`` is True). Pair with ``cell_value`` to
  confirm the merged content as well.
  {"op":"cell_merged", "sheet":<sh>, "a1_ref":<anchor a1>}

* ``data_validation`` — the range has a data-validation rule. When
  ``allowed_values`` is supplied, the rule's Formula1 must encode
  the same set (order-insensitive).
  {"op":"data_validation", "sheet":<sh>, "range_ref":<a1 range>,
   "allowed_values":[<value>, <value>, ...]}

* ``sort_order`` — already documented. ``expected_first_value`` /
  ``expected_last_value`` are OPTIONAL anchors — OMIT them unless
  the exact value is visible verbatim in the SP block AND the
  data type is plain numeric / string. Date columns store as
  serial numbers (the SP shows them as ``4.226e+04`` or formatted
  text — neither is the true cell value), and a hallucinated
  anchor permanently FAILs verify even when the agent sorts
  correctly. Default to the monotonic-only form:
  {"op":"sort_order", "sheet":<sh>, "range_ref":<a1>,
   "sort_col_within_range":<0-based int>, "ascending":<bool>}
  Add anchors only when the column is plain numeric / text and
  the SP shows the post-sort first/last cell verbatim:
  {"op":"sort_order", "sheet":<sh>, "range_ref":<a1>,
   "sort_col_within_range":<0-based int>, "ascending":<bool>,
   "expected_first_value":<value visible in SP>}

* ``freeze_pane`` — already documented; pass ``expected_rows`` and/or
  ``expected_cols`` for an exact-position check. Without them, any
  freeze counts.
  {"op":"freeze_pane", "expected_rows":<int>, "expected_cols":<int>}

* ``duplicate_range_match`` — already documented; pass ``transposed":true``
  when verifying a ``calc_duplicate_range(transpose=true)`` operation.

# Patterns — match the task to one shape

* fill column (a / c / d): two cell_value checks — first row + last
  row — with ``expected_formula_contains`` naming the function the
  task uses (``VLOOKUP``, ``CONCATENATE``, ``SUM``, etc.). DO NOT use
  ``expected_value`` for formula columns — that would force a specific
  literal which depends on the exact source data and breaks if the
  evaluator's data differs slightly.

* pivot table: chain ``sheet_exists`` for the new sheet (if any) +
  ``pivot_exists`` on that sheet. The pivot row/data fields are
  internal; init_ledger checks structural presence, not config.

* sort: one ``sort_order`` check + ideally one ``cell_value`` against
  the new first row.

* freeze: one ``freeze_pane`` check.

* new sheet / rename: one ``sheet_exists`` check on the expected
  final name.

* copy range: one ``duplicate_range_match`` over the first few rows.

* number format: one ``cell_format`` over the affected range.

* chart: one ``chart_exists`` check with ``chart_type``. Add
  ``title`` substring ONLY when the task instruction literally
  names a chart title. Add ``expected_series_count`` ONLY when the
  task literally names a series count ("three line charts" → 3;
  "two bars" → 2). For the common "line/bar chart with X on
  X-axis and Y on Y-axis" wording where neither title nor count
  is stated, the verify is just ``{"op":"chart_exists",
  "on_sheet":<sh>, "chart_type":<type>}`` — series-count and
  title correctness is enforced upstream by the planner's form
  (2) emission protocol, not by re-deriving values init_ledger
  cannot read from the SP. NEVER substitute ``sheet_exists`` for
  ``chart_exists`` on a chart task — ``sheet_exists`` passes
  whenever the data sheet is present and gives the agent a
  false-positive.

# Examples — pick the substring that ACTUALLY appears in YOUR formula

  VLOOKUP grade-scale (template uses VLOOKUP) → substring "VLOOKUP":
    {"op":"cell_value", "a1_ref":"F10", "expected_formula_contains":"VLOOKUP"}

  Cross-sheet year-over-year (template `=(Sheet1.B{r+1}-Sheet1.B{r})/Sheet1.B{r}*100`)
  → substring "Sheet1.B" (the cross-sheet ref) — NOT "VLOOKUP":
    {"op":"cell_value", "a1_ref":"B2", "expected_formula_contains":"Sheet1.B"}

  Row aggregate (template `=SUM(B2:B11)`) → substring "SUM":
    {"op":"cell_value", "a1_ref":"B12", "expected_formula_contains":"SUM"}

  Header-tagged concat (template `=$A$1&": "&FIXED(A{r},2)&...`)
  → substring "FIXED" or "$A$1":
    {"op":"cell_value", "a1_ref":"E2", "expected_formula_contains":"FIXED"}

# Example — pivot table on a new sheet

  "verify": {
    "kind": "calc_verify",
    "calc_checks": [
      {"op":"sheet_exists", "name":"PivotSheet"},
      {"op":"pivot_exists", "on_sheet":"PivotSheet"}
    ]
  }

Do NOT emit ``kind="shell_command"`` with a python3 -c body for
libreoffice_calc tasks — that path is deprecated for Calc. The
``calc_verify`` kind covers every Calc evaluator shape this benchmark
uses.
"""


def action_schema_block() -> str:
    """Return the prompt block describing every ``calc_*`` action.
    Injected into the actor's system prompt only for the
    ``libreoffice_calc`` domain — Writer / Impress / browser actors
    aren't cluttered with these.

    The block is the SINGLE SOURCE OF TRUTH the actor reads — Calc
    tasks never go through ``cli_run_uno`` from the actor's side.
    """
    return _SCHEMA_BLOCK


def data_layout_init_block() -> str:
    """Append-only schema requirement for init_ledger when the task
    domain is libreoffice_calc.

    The domain knowledge block (libreoffice_calc.md) defines a
    Q1-Q4 STOP chain — row semantics / task pattern / output target
    / lookup spec — that anchors verify spec choices to the actual
    workbook structure. Without an enforced output slot, the LLM
    skips the chain and produces verify specs against guessed
    coordinates (e.g. F23 expected '32' when 32 is column E's mark
    value, not column F's message). Forcing a structured
    ``data_layout`` field BEFORE ``required_outcomes`` makes the
    reasoning visible and gives downstream verify-spec building a
    grounded anchor to copy column letters from.
    """
    return _DATA_LAYOUT_INIT_BLOCK


def data_layout_planner_block() -> str:
    """Append-only requirement for the planner's <plan> emission when
    the task domain is libreoffice_calc.

    Mirrors :func:`data_layout_init_block` for the planner side. The
    planner currently emits GUI-shape subgoals ("Click F10, type
    formula, press Enter, drag fill handle") even when the actor's
    cleanest path is one ``calc_fill_column`` JSON. Forcing a
    ``<data_layout>`` block before ``<strategy>`` makes the planner
    pick the calc_* action up front and write the subgoal in that
    vocabulary, so the actor doesn't have to reverse-engineer GUI
    text back to a structured action.
    """
    return _DATA_LAYOUT_PLANNER_BLOCK


_DATA_LAYOUT_INIT_BLOCK = """\
DATA_LAYOUT — REQUIRED FOR libreoffice_calc TASKS

Before writing ``required_outcomes``, output a ``data_layout`` object
answering the Q0-Q4 STOP chain from the domain knowledge block above.
Each answer must be grounded in the WORKBOOK STRUCTURE / SheetPerceiver
block — not training memory.

  q0_subtasks: list of sub-tasks the instruction asks for.
       **RULE: one calc_* action ⇒ one q0 entry.** Bundling two
       actions into a single subtask collapses them into ONE q4
       reasoning block; the second action's kwargs end up
       improvised, skipping the pattern's emission protocol. Split
       at every conjunction ("Next" / "Then" / "And then" / "After"
       / "Also" / ";"). The only exception is pattern (f) cross-
       totals (column total + row total), where one subtask spans
       both actions and q4 splits internally.
       Bundle into ONE sub-task only when the SAME range receives
       multiple atomic settings at once ("Merge A1:C1 with blue
       fill, bold white text" → 1 sub-task). Use ``depends_on`` to
       chain when a later sub-task reads from an earlier one.

  Then answer Q1-Q4 SEPARATELY for EACH sub-task in q0_subtasks
  (one entry in ``subtasks_reasoning`` per q0 item, in order).
  Within each sub-task's answers:

  q1_row_semantics: one sentence. What does ONE row of the relevant
       sheet represent? Read it off the ``hdr:`` line + sample data
       rows. Be concrete (e.g. "Sheet1 row = one student; D = mark
       threshold, E = grade label"). ALSO scan the SP block for any
       data-area row whose leftmost-column label is
       ``Total`` / ``Sum`` / ``Subtotal`` / ``Grand total`` — those
       are NOT data rows, they are row-aggregate destinations. List
       their row indices alongside the row semantics so Q2 / Q3
       account for both axes.

  q2_task_pattern: pick ONE of (a / b / b' / c / d / e / f / g / h)
       and name the calc_* action(s) it maps to. (a) per-row formula
       in a new COLUMN (each row has its own formula) →
       calc_fill_column; per-column formula in a new ROW (each column
       has its own formula; "new row called Total / Growth / Average")
       → calc_fill_row — DO NOT use calc_fill_column for a new-row
       task, it walks DOWN a column and creates wrong cells.
       (b) multi-column block → calc_set_range;
       (b') group-by → calc_pivot_table; (c) cross-table lookup
       (VLOOKUP) → calc_fill_column with a VLOOKUP template;
       (d) delimiter text split → calc_split_text; (e) structural →
       calc_sort_range / calc_freeze / calc_format_number /
       calc_new_sheet / calc_rename_sheet / calc_duplicate_range /
       calc_set_cell / calc_hide_range / calc_set_color /
       calc_merge_cells / calc_data_validation /
       calc_reorder_columns / calc_delete_columns /
       calc_delete_rows; (f) compound — TWO actions needed:
       cross-totals (column total AND row total) →
       calc_fill_column + calc_set_range; container + content (new
       sheet + content) → calc_new_sheet + calc_set_range /
       calc_pivot_table; (g) chart insertion → calc_insert_chart
       (single step; default placement is on the data sheet — omit
       ``sheet=`` or set sheet=data_sheet unless the task explicitly
       names a different destination sheet); (h) file
       export → calc_export_file (single step, format inferred
       from path extension; supports PDF fit-to-N-pages scaling).
       PRECEDENCE — chart keyword (``chart``, ``graph``, ``plot``,
       ``bar chart``, ``column chart``, ``line chart``, ``pie
       chart``) → (g); export keyword (``export to PDF``, ``save
       as CSV``, ``export to csv``, ``fit on one page``, ``fit to
       page``) → (h). Both win over (b') even when ``for each
       <X>`` phrasing appears — those describe the chart axis /
       sheet contents, not a pivot row grouping.
       STRUCTURAL (e) PRECEDENCE — when the task names a
       VISUAL / STRUCTURAL change (background fill, font color,
       merge, dropdown picker, column reorder, transpose,
       hide), pattern is (e) with the matching calc_* action.
       Do NOT fall back to a generic calc_set_range — the
       structural action exists for a reason and the verifier
       has a specific op to confirm it.

  q3_output_target: WHICH column / range gets filled? Cite the SHEET
       NAME + COLUMN LETTER + ROW RANGE, derived from the SP block's
       ``used=A1:<col><N>`` line. If multiple columns share the SAME
       header (e.g. two ``Officer Name`` columns), pick the EMPTY one
       (data rows show ``·`` in the SP block). Do NOT target a column
       that already has populated values.

  q4_action_spec: pattern-specific spec for the calc_* action your
       plan will dispatch. The shape switches on the q2 pattern letter
       — fill the slots for your pattern, leave others out. EVERY
       slot MUST be filled (empty list ``[]`` for unused fields, NOT
       omitted), so each placement decision is forced rather than
       defaulted. Re-read the instruction word-by-word while filling
       — the trigger phrases below catch easy-to-miss wording (e.g.
       "as column headers" vs "as row labels" picks column_fields vs
       row_fields for pivots).

       (a) per-row formula → calc_fill_column:
            target_col=<letter>;
            formula_template=<template with {r} for cell refs OR
                              {seq} for 1-based sequence values>;
            first_row=<int>; last_row=<int>
            Use {seq} when the literal output depends on position
            ("No. {seq}" → "No. 1", "No. 2", …). Use {r} only for
            cell references in real formulas.

       (b) multi-column block → calc_set_range:
            start_a1=<anchor>; values_shape=<rows>x<cols>;
            header_row=[<col1-header>, <col2-header>, ...]

       (b') group-by → calc_pivot_table:
            row_fields=[<header>...]; column_fields=[<header>...];
            filter_fields=[<header>...]; data_fields=[[<header>,<agg>]...]
            DECISION RULE — read the instruction:
              * "as the row labels", "for each <X>", "<X> in rows" →
                <X> goes in row_fields
              * "as the column headers", "as columns", "<X> as columns" →
                <X> goes in column_fields
              * default to row_fields when wording is ambiguous
            ALL FOUR field arrays MUST be listed (``[]`` for unused).

       (c) cross-table lookup (VLOOKUP) → calc_fill_column with VLOOKUP:
            key_col=<letter>; value_col=<letter>; lookup_range=<a:b>;
            match_type=<0|1>
            KEY column holds values that match what you query with.
            VALUE column holds what the formula returns. ``lookup_range``
            SPANS BOTH.
            ``match_type=0`` (exact): keys are discrete labels (product
            names, branch names, student IDs).
            ``match_type=1`` (approx): keys are range thresholds
            (0, 30, 60, 80, 90 for grade boundaries — a score of 42
            won't match any exactly; formula returns the row of the
            largest key ≤ 42).

       (d) delimiter text split → calc_split_text:
            src_col=<letter>; out_cols=[<letter>, ...];
            delimiter=<char or string, default " ">;
            last_takes_rest=<true|false, default true>;
            first_row=<int>; last_row=<int>
            HARD RULE: ``out_cols`` MUST list ONE letter per column
            named in the instruction's fill target list ("split into
            First Name, Last Name and Rank" ⇒ 3 letters, NOT 2).
            ``last_takes_rest=true`` is the safe default — handles
            multi-word last segments (e.g. multi-word job titles)
            without changing anything else.
            Use the formula-template fallback (2× calc_fill_column
            with REGEX / LEFT / MID / RIGHT) ONLY when the split is
            NOT by a fixed delimiter — e.g. extract a phone number by
            pattern, take first 5 chars by position.

       (e) structural op → one of the structural calc_* actions:
            action=<calc_sort_range|calc_freeze|calc_format_number|
                   calc_new_sheet|calc_rename_sheet|calc_duplicate_range|
                   calc_set_cell>;
            kwargs=<flat key=value list of the action's required kwargs
                    from the Calc structured actions block>

       (g) chart insertion → calc_insert_chart (single step). Two
            data-feed forms — pick by counting series in the gold
            chart:
            FORM (1) wide table (N series): data_range=<A1 range>.
            FORM (2) single summary row/column (1 series) OR
            non-adjacent X/Y axes: categories=<A1 range> +
            values=[<A1 range>, ...] (one entry per series).
            For form (2), ``categories`` and each ``values`` entry
            MUST share shape and length — values entry is a ROW on
            cols X..Y ⇒ categories is the HEADER ROW on cols X..Y;
            values entry is a COLUMN on rows R1..R2 ⇒ categories is
            the LEFTMOST LABEL COLUMN on rows R1..R2. Mismatching
            orientation (column categories with a row values) gives
            the wrong x-axis labels.
            sheet=<placement sheet — DEFAULTS to data_sheet; only
                  set to a DIFFERENT sheet when the task explicitly
                  says "in a new sheet called X" / "on Sheet2">;
            data_sheet=<source data sheet>;
            chart_type=<column|bar|line|pie|scatter|area>;
            title=<chart title string>
            One step covers everything; DO NOT chain a calc_new_sheet
            before this. Phrasing like "for each <X>" in the
            instruction names the CATEGORY axis of the chart, not a
            pivot grouping — pattern is (g), not (b'). Multiple
            charts on one sheet: omit ``name=`` from BOTH calls
            (framework auto-picks Chart1, Chart2, ...) so the second
            call doesn't overwrite the first.

       (h) file export (PDF / CSV) → calc_export_file (single step):
            path=<absolute path ending in .pdf or .csv>;
            fit_width_pages=<int | omit>;     # PDF only
            fit_height_pages=<int | omit>;    # PDF only
            sheet=<sheet name | omit>;        # CSV only, defaults to active
            delimiter=<char | omit>;          # CSV only, defaults to ","
            COMPOSE ``path`` from the instruction: <directory> +
            <basename> + <.pdf|.csv>. Read directory + basename
            rule from the wording; ``path`` ends in EXACTLY ONE
            extension. Init_ledger verify is ``file_exists`` with
            the same ``path``.

If any of Q1-Q4 cannot be answered concretely from the SP block, the
field's value should start with ``UNSURE:`` followed by what's missing
(e.g. ``UNSURE: SP block has no `used=` line for Sheet2``). Don't
guess.

The ``data_layout`` object goes IMMEDIATELY after ``initial_context``
and BEFORE ``required_outcomes``. ``required_outcomes`` MUST contain
ONE outcome per ``q0_subtasks`` entry (1:1, same order); each
outcome's verify spec uses the column letters / row ranges named in
that sub-task's ``q3_output_target`` verbatim. Use ``depends_on`` to
chain when a later sub-task reads from an earlier one (e.g. a concat
column depends on the source columns being filled first).

JSON schema update for libreoffice_calc tasks:

{
  "initial_context": { ... },
  "data_layout": {
    "q0_subtasks": ["<short imperative phrase>", ...],
    "subtasks_reasoning": [
      {
        "q1_row_semantics": "<one sentence>",
        "q2_task_pattern":  "<letter + calc_* action name>",
        "q3_output_target": "<sheet + column letter + row range>",
        "q4_action_spec":   "<pattern-specific slots>"
      },
      ...                          // one per q0_subtasks entry
    ]
  },
  "required_outcomes": [ ... ]      // one per q0_subtasks entry
}
"""


_DATA_LAYOUT_PLANNER_BLOCK = """\
DATA_LAYOUT — REQUIRED FOR libreoffice_calc TASKS

For libreoffice_calc tasks, the planner MUST emit a ``<data_layout>``
block as the FIRST child of ``<plan>``, before ``<strategy>``. The
fields are the Q1-Q4 STOP chain from the domain knowledge block
above; each answer must be grounded in the SheetPerceiver / WORKBOOK
STRUCTURE block, not training memory.

**Grain of each question** — Q1 is sheet-level (one answer per
plan, lives in ``<data_layout>``). Q2 / Q3 / Q4 are PER-ACTION:
each ``<step>`` carries its OWN ``<q2_task_pattern>`` +
``<q3_output_target>`` + ``<q4_action_spec>`` so every action runs
through its pattern's emission protocol independently. Bundling Q2
("(e) calc_fill_row + (g) calc_insert_chart") collapses both into
one reasoning block; the second action's kwargs end up improvised
inside ``<text>`` and skip the protocol (notably the Form (2)
chart trim+count check). One step ⇒ one Q2 ⇒ one Q3 ⇒ one Q4.

  <q1_row_semantics> — (in ``<data_layout>``) what ONE row of the
                       relevant sheet represents
  <q2_task_pattern>  — (in each ``<step>``) letter (a / b / b' /
                       c / d / e / f / g / h) + the ONE calc_*
                       action name for THIS step. e.g. ``(c)
                       calc_fill_column with VLOOKUP``; ``(g)
                       calc_insert_chart`` for chart tasks; ``(h)
                       calc_export_file`` for PDF/CSV export.
                       PRECEDENCE — chart keyword (``chart``,
                       ``graph``, ``plot``) → (g); export keyword
                       (``export to PDF``, ``save as CSV``, ``fit
                       on one page``) → (h). Both win over (b')
                       even when ``for each <X>`` phrasing appears.
  <q3_output_target> — (in each ``<step>``) sheet + column letter
                       + row range for THIS step's output; for
                       duplicate headers, name the EMPTY column
                       (SP shows ``·``)
  <q4_action_spec>   — (in each ``<step>``) pattern-specific
                       kwargs for the calc_* action this step
                       dispatches. EVERY slot of the pattern's
                       spec MUST be filled (use empty array
                       ``[]`` for unused fields, NEVER omit), so
                       each placement decision is forced rather
                       than defaulted. The spec slots per pattern:

    (a) calc_fill_column (per-row formula or position-indexed value):
        ``target_col=<letter>; formula_template=<{r} for cell refs OR
        {seq} for 1-based sequence>; first_row=<int>; last_row=<int>``
        {seq}-based literals ("No. {seq}") avoid the off-by-first_row
        bug that {r}-based literals ("No. {r}") have.
        For a NEW ROW filled with one formula per column ("new row
        called Total / Growth / Average"; row totals filling B6:E6,
        B13:E13, B20:E20 with SUM across that row) → use
        ``calc_fill_row`` (NOT calc_fill_column emitting B6/B13/B20
        single cells with self-referencing formulas):
        ``row=<int>; first_col=<letter>; last_col=<letter>;
        formula_template=<{c} for current col letter, {prev_c} for
        previous col, {seq} for 1-based>; label=<optional, written
        one column LEFT of first_col>``. ONE calc_fill_row per Total
        row writes the whole B..E range in one step.
        ROW-DEPENDS-ON-EARLIER-ROW RULE — when a later subtask writes
        a SECOND summary row (Growth / change / ratio / pct) whose
        values are derived from an EARLIER summary row produced by a
        prior subtask, the derived row's ``formula_template`` anchors
        its row references on the EARLIER summary row's index — NOT
        on the raw-data first row. Re-read q0 before filling q4: if
        a Total / Sum / Aggregate row was just emitted at row R, the
        derived row's template is
        ``=({c}<R>-{prev_c}<R>)/{prev_c}<R>`` (or analogous), with
        <R> = the aggregate row's index. Defaulting <R> to 2 (or
        whatever the data start row is) silently feeds raw rows into
        the derived row instead of the aggregated values — a hard-
        to-spot bug because Calc still computes a value, just the
        wrong one.

    (b) calc_set_range (multi-column block):
        ``start_a1=<anchor>; values_shape=<rows>x<cols>;
        header_row=[<col1-header>, ...]``

    (b') calc_pivot_table (group-by):
        ``row_fields=[<header>...]; column_fields=[<header>...];
        filter_fields=[<header>...]; data_fields=[[<header>,<agg>]...]``
        DECISION RULE — re-read the task instruction word-by-word:
          * "as the row labels", "for each <X>" → <X> ⇒ row_fields
          * "as the column headers", "as columns", "<X> as columns"
            → <X> ⇒ column_fields
          * default to row_fields when wording is genuinely ambiguous
        ALL FOUR fields MUST appear (``[]`` for unused).

    (c) calc_fill_column + VLOOKUP (cross-table lookup):
        ``key_col=<letter>; value_col=<letter>;
        lookup_range=<a:b spans both cols>; match_type=<0|1>``
        match=0 for discrete labels (product / branch names);
        match=1 for range thresholds (0/30/60/80/90 grade boundaries).

    (d) calc_split_text (delimiter text split):
        ``src_col=<letter>; out_cols=[<letter>...];
        delimiter=<char|str, default " ">;
        last_takes_rest=<true|false, default true>;
        first_row=<int>; last_row=<int>``
        HARD RULE: ``out_cols`` length MUST equal the number of column
        names in the instruction's fill list ("split into First Name,
        Last Name and Rank" ⇒ out_cols=["B","C","D"], NOT ["B","C"]).
        Formula fallback (calc_fill_column with REGEX/LEFT/MID/RIGHT)
        is ONLY for non-delimiter extraction (pattern match, fixed
        position).

    (e) structural calc_* action:
        ``action=<calc_sort_range|calc_freeze|calc_format_number|
                  calc_new_sheet|calc_rename_sheet|calc_duplicate_range|
                  calc_set_cell|calc_hide_range|calc_set_color|
                  calc_merge_cells|calc_data_validation|
                  calc_reorder_columns|calc_delete_columns|
                  calc_delete_rows>;
        kwargs=<flat key=value pairs of that action's required kwargs>``
        Quick map from task wording to the right structural action:
          - "highlight / background color / font color / bold" →
            calc_set_color (bgcolor / font_color / bold; "#RRGGBB")
          - "merge cells <A1:Cn>" → calc_merge_cells with optional
            ``value=<text to put in the merged anchor>``
          - "dropdown / data validation / allowed values" →
            calc_data_validation (allowed_values=[...]; type="list")
          - "reorder / move columns to <X, Y, Z>" →
            calc_reorder_columns (new_order=[...]) — accepts
            column letters OR header strings
          - "transpose <range> to <anchor>" → calc_duplicate_range with
            ``transpose=true`` (same source / dest semantics)

    (g) calc_insert_chart (single-step chart insertion):
        Two data-feed forms — pick by counting how many series the
        gold chart has:
        FORM (1) wide table chart (N series): ``data_range=<A1
        range — pick by INTENT; ONE series per non-header row>``.
        FORM (2) single summary row/column chart (1 series) OR
        non-adjacent X/Y axes: ``categories=<A1 range>;
        values=[<A1 range>, ...]`` — one entry per series.
        Form (2) emission protocol — derive categories from the
        task wording + SP, NOT from a "categories = col A" prior:
         HARD CONSTRAINTS (check BOTH before emitting):
         1. ``count(categories) == count(values[i])`` — off-by-one
            (extra leftmost-label cell or extra header cell) is
            REJECT, recompute the range.
         2. ``categories`` and ``values[i]`` live on DIFFERENT rows
            (when each is a row) or DIFFERENT columns (when each is
            a column). Same row / same column → REJECT. Pointing
            categories at the same row as values makes the x-axis
            labels be the data values themselves — semantically
            empty. Categories ALWAYS lives in a separate header /
            label row or column.
         (i)   read the instruction for the x-axis LABEL NAME
               ("x-axis is <X>", "for each <X>", "<X> on X-axis");
         (ii)  find which SP cells hold those labels (usually the
               hdr row OR the leftmost data column) — cite range;
         (iii) find which SP cells hold the y-axis data — cite
               range;
         (iv)  trim (ii) to cover EXACTLY the same cols (or rows)
               as (iii). If (iii) spans cols X..Y, (ii) spans cols
               X..Y — not col A..Y, not col X..Z. Then emit
               categories=(ii), values=[(iii)].
         (v)   shape match: row values ⇒ row categories; column
               values ⇒ column categories.
        If (i)'s label name is NOT the leftmost-column header,
        categories almost certainly comes from row 1 — do NOT
        default to col A. See domain knowledge ``Choosing
        data_range precisely``.
        Shared: ``sheet=<placement — DEFAULT data_sheet; only
        different when the instruction explicitly names "in a new
        sheet called X" / "on Sheet2">; data_sheet=<source>;
        chart_type=<column|bar|line|pie|scatter|area>;
        title=<chart title>``.
        DO NOT chain a ``calc_new_sheet`` before this. Picking an
        arbitrary chart-title-like sheet name (e.g. ``sheet=
        "2019 Monthly Costs"``) auto-creates an empty side-sheet —
        the ``chart_exists`` verify usually targets the data sheet
        and will permanently FAIL.

    (h) calc_export_file (single-step PDF / CSV export):
        ``path=<absolute path ending in .pdf or .csv>;
        fit_width_pages=<1 if "fit to one page wide" else omit>;
        fit_height_pages=<1 if "fit to one page tall" else omit>;
        sheet=<sheet name if instruction names a specific sheet
        else omit>``
        Path comes from the instruction: stem of the open xlsx +
        ``.pdf`` / ``.csv``, anchored at ``<home>`` when the
        instruction says "in my home directory" / "under home".
        Read the xlsx stem from the SP block's window title line
        — DO NOT guess. NO ``calc_new_sheet`` / no Save-As GUI
        chain — one ``calc_export_file`` action handles the
        entire flow.

After ``<data_layout>`` (which carries only Q1), emit
``<strategy>`` then one or more ``<step>`` blocks. EACH ``<step>``
carries its own ``<q2_task_pattern>`` + ``<q3_output_target>`` +
``<q4_action_spec>`` + ``<text>`` + ``<expected_post_state>``. The
``<text>`` MUST carry EVERY kwarg from this step's
``<q4_action_spec>`` verbatim — same kwargs, same column letters,
same A1 ranges. The actor decomposer only reads ``<text>``; if q4
wrote ``categories=<X>, values=[<Y>]`` and the text only says
"create a chart of the <row>", the actor LOSES those ranges and
re-guesses (usually wrong). Spell out every kwarg.

Each ``<step>`` should be phrased in the calc_* vocabulary so the
actor's translation is a one-shot lookup. Examples:

  Pivot with category as COLUMN headers (instruction says "Foo as the
  column headers"):
    <step>
      <q2_task_pattern>(b') calc_pivot_table</q2_task_pattern>
      <q3_output_target>Sheet2 anchored at A1</q3_output_target>
      <q4_action_spec>row_fields=[]; column_fields=["Foo"];
        filter_fields=[]; data_fields=[["Bar","SUM"]]</q4_action_spec>
      <text>Use calc_pivot_table with source_sheet=Sheet1,
source_range=A1:G2001, dest_sheet=Sheet2, dest_anchor=A1,
column_fields=["Foo"], data_fields=[["Bar","SUM"]].</text>
    </step>

  Per-row VLOOKUP (approx match on thresholds):
    <step>
      <q2_task_pattern>(c) calc_fill_column with VLOOKUP</q2_task_pattern>
      <q3_output_target>Sheet1 col F rows 10..23</q3_output_target>
      <q4_action_spec>key_col=E; value_col=F; lookup_range=$D$2:$E$7;
        match_type=1</q4_action_spec>
      <text>Use calc_fill_column on Sheet1 column F rows 10..23 with
formula_template ``=VLOOKUP(E{r};$D$2:$E$7;2;1)``.</text>
    </step>

  Chart — Form (1), whole table (N series):
    <step>
      <q2_task_pattern>(g) calc_insert_chart (form 1)</q2_task_pattern>
      <q3_output_target>chart on <placement-sheet></q3_output_target>
      <q4_action_spec>sheet=<placement>; data_sheet=<source>;
        data_range=<full A1 incl header>; chart_type=<type>;
        title=<T></q4_action_spec>
      <text>Use calc_insert_chart with sheet=<placement>,
data_sheet=<source>, data_range=<A1 range>, chart_type=<type>,
title=<T>.</text>
    </step>

  Chart — Form (2), single summary row / column (1 series):
    <step>
      <q2_task_pattern>(g) calc_insert_chart (form 2)</q2_task_pattern>
      <q3_output_target>chart on <placement-sheet></q3_output_target>
      <q4_action_spec>sheet=<placement>; data_sheet=<source>;
        categories=<A1 range>; values=[<A1 range>];
        chart_type=<type>; title=<T></q4_action_spec>
      <text>Use calc_insert_chart with sheet=<placement>,
data_sheet=<source>, categories=<A1 range>, values=[<A1 range>],
chart_type=<type>, title=<T>.</text>
    </step>
    Note: DO NOT add ``data_range=`` alongside ``categories``+
    ``values`` — those two forms are mutually exclusive; the
    structured-action wire layer will drop the redundant
    ``data_range`` silently but it noise-pollutes the trace.
    Note: ONE step; do NOT emit a separate ``calc_new_sheet``
    before it. ``sheet=`` defaults to the data sheet — only set
    differently when the instruction names a destination
    (``"in a new sheet named X"``).

NOT: ``Click F10, type =VLOOKUP(...), press Enter, drag fill
handle...`` — GUI shape forces the actor to reverse-engineer intent
back into a calc_fill_column action and often produces wrong GUI
coordinates.

ALSO NOT: raw UNO Python or any python-docx-style file edit — there
is no raw-UNO escape hatch for libreoffice_calc. ALWAYS phrase
spreadsheet mutations as a calc_* action by name (calc_fill_column /
calc_set_cell / calc_set_range / calc_new_sheet / calc_rename_sheet /
calc_pivot_table / calc_sort_range / calc_freeze / calc_format_number /
calc_duplicate_range / calc_copy_to_clipboard / calc_split_text / calc_hide_range /
calc_insert_chart / calc_export_file).

The ``<data_layout>`` block goes inside ``<plan>``. When the
``<decision>`` is CONTINUE or DONE (no replan needed), ``<plan>``
itself is omitted — ``<data_layout>`` is omitted with it.

Output skeleton for libreoffice_calc — note Q2 / Q3 / Q4 live
INSIDE each ``<step>``, NOT inside ``<data_layout>``:

<plan>
  <data_layout>
    <q1_row_semantics>...</q1_row_semantics>
  </data_layout>
  <strategy>...</strategy>
  <step>
    <q2_task_pattern>...</q2_task_pattern>
    <q3_output_target>...</q3_output_target>
    <q4_action_spec>...</q4_action_spec>
    <text>...</text>
    <expected_post_state>...</expected_post_state>
  </step>
</plan>
"""


_SCHEMA_BLOCK = """\
Calc structured actions — the framework builds the underlying Python
deterministically from your flat JSON kwargs, so you never write
Python source. Brackets, escapes, newlines: all handled for you.

# Common conventions

* `sheet`            : sheet name (string, from SP `Sheet[<name>]` line)
* `a1_ref` / range   : A1 notation ("A1", "B2:D10", "F2:F12")
* Multi-arg formulas use `;` as argument separator
  (e.g. `=VLOOKUP(E2;$A$2:$B$7;2;0)`), NOT comma
* Cross-sheet refs   : dot `.`, NOT bang `!`
  (e.g. `=SUM('Retail Price'.B2:B23)`)

# Actions

* calc_fill_column — fill ONE column with a per-row formula or value
  {"action":"calc_fill_column",
   "sheet":<sheet>, "column":<col-letter>,
   "first_row":<int>, "last_row":<int>,
   "formula_template":<template using {r} and/or {seq}>,
   "header":<optional string written one row above first_row>}
  Placeholders (substituted per row server-side):
    {r}      — sheet row index (2, 3, …). Use to build CELL REFERENCES,
               e.g. `=VLOOKUP(E{r};$D$2:$E$7;2;1)` or `=A{r}-B{r}`.
    {r+N},   — sheet row index OFFSET by integer N. Use for cross-row
    {r-N}      formulas referencing the row above / below, e.g.
               year-on-year change `=(B{r+1}-B{r})/B{r}` or
               previous-row diff `=B{r}-B{r-1}`. The bare `{r-1}`
               literal does NOT auto-expand without these brackets.
    {seq}    — 1-based sequence from first_row (1, 2, 3, …). Use when
               the literal value depends on POSITION rather than row,
               e.g. `"No. {seq}"` → "No. 1", "No. 2", … (NEVER
               `"No. {r}"`, which produces "No. 2".."No. <last_row>"
               — off by first_row-1).

* calc_fill_blanks — forward-fill EMPTY cells in a range from an
  adjacent direction. The op walks each line in the chosen direction
  and carries the last non-empty value into subsequent empty cells.
  Use this instead of `calc_fill_column` with a `=<col>{r}` template
  (which self-references and produces a circular reference).
  Non-empty cells in the range are NOT touched.
  {"action":"calc_fill_blanks", "sheet":<sheet>, "range_ref":<a1>,
   "source":<"above"|"below"|"left"|"right">}
  - `source="above"` (default): each empty cell takes the value from
    the closest non-empty cell ABOVE it in the same column (pandas
    forward-fill).
  - `source="below"` / `"left"` / `"right"`: same direction analogue.
  Verify with `cell_value` checks on a sample of previously-blank
  cells.

* calc_fill_row — fill ONE row with a per-column formula or value
  (the row-direction analogue of calc_fill_column; use for "Total /
  Growth / Average row across columns" tasks)
  {"action":"calc_fill_row",
   "sheet":<sheet>, "row":<int>,
   "first_col":<col-letter>, "last_col":<col-letter>,
   "formula_template":<template using {c}, {prev_c}, and/or {seq}>,
   "label":<optional string written one column LEFT of first_col>}
  Placeholders (substituted per cell server-side):
    {c}      — current column letter (B, C, D, …). Use to build cell
               refs, e.g. `=SUM({c}2:{c}11)` for a monthly total row.
    {prev_c} — previous column letter (one to the LEFT of {c}). Use
               for cross-column growth, e.g. `=({c}12-{prev_c}12)/{prev_c}12`.
               When the template uses {prev_c}, the FIRST column in
               the range is left blank automatically — matches the
               "Jan growth = blank" shape of growth rows.
    {seq}    — 1-based sequence from first_col (1, 2, 3, …).

* calc_set_cell — write a SINGLE cell
  {"action":"calc_set_cell", "sheet":<sheet>, "a1_ref":<single-cell-a1>,
   "value":<literal>}     // OR
  {"action":"calc_set_cell", "sheet":<sheet>, "a1_ref":<single-cell-a1>,
   "formula":<formula starting with `=`>}

* calc_set_range — write a 2D rectangular block (multi-column writes)
  {"action":"calc_set_range", "sheet":<sheet>, "start_a1":<anchor>,
   "values":[[<row1-col1>,<row1-col2>], [<row2-col1>,<row2-col2>], ...]}
  Cell typing per element: number → numeric; str starting with `=` →
  formula; other str → text; null → cell left UNCHANGED.

* calc_split_text — split each row's source cell by a delimiter into
  N destination columns. Framework reads the source cell, applies
  Python ``str.split``, writes each piece to the corresponding
  out_col. No nested FIND/MID formulas needed.
  {"action":"calc_split_text", "sheet":<sheet>, "src_col":<letter>,
   "first_row":<int>, "last_row":<int>,
   "out_cols":[<letter>, <letter>, ...],
   "delimiter":<single-char or string, default " ">,
   "last_takes_rest":<bool, default true>}
  - ``last_takes_rest=true`` (default): when the source has MORE
    delimiters than ``len(out_cols)-1``, the LAST out_col gets the
    remaining tail joined back with delimiter — so multi-word
    last-segments (e.g. "Senior Vice President" as Rank) land
    correctly in the final column.
  - ``last_takes_rest=false``: extra pieces are discarded.
  Source column is NOT modified — preserves "don't touch the original
  data" task constraint automatically.

* calc_new_sheet — create a new sheet (blank or copy of existing)
  {"action":"calc_new_sheet", "name":<new-sheet>,
   "position":<"end"|"start"|int|"before:<name>"|"after:<name>">,
   "source_sheet":<optional sheet to copy from; omit for blank>}

* calc_rename_sheet — rename an existing sheet
  {"action":"calc_rename_sheet", "current_name":<current>,
   "new_name":<new>}

* calc_pivot_table — build a pivot table summarising a source range
  {"action":"calc_pivot_table",
   "source_sheet":<src-sheet>, "source_range":<a1 incl header>,
   "dest_sheet":<dest-sheet>, "dest_anchor":"A1",
   "row_fields":[<header-string>, ...],
   "column_fields":[<header-string>, ...],   // optional
   "filter_fields":[<header-string>, ...],   // optional
   "data_fields":[[<header>,<"SUM"|"AVERAGE"|"COUNT"|"MAX"|"MIN"|"PRODUCT">], ...],
   "pivot_name":<name>}
  Source MUST include the header row (field names are auto-derived
  from row 0). `data_fields` is list-of-2-element-lists in JSON;
  framework converts to tuples for the op call.

* calc_sort_range — sort a range by one or more columns
  {"action":"calc_sort_range", "sheet":<sheet>, "range_ref":<a1>,
   "sort_columns":[[<col-index-0-based>,<bool-ascending>], ...],
   "contains_header":<bool>}

* calc_freeze — freeze top rows / left columns
  {"action":"calc_freeze", "n_cols":<int>, "n_rows":<int>}

* calc_format_number — apply Excel-style number format to a range
  {"action":"calc_format_number", "sheet":<sheet>, "range_ref":<a1>,
   "format_string":<excel-format-string>,
   "locale":<optional [lang, country, variant] list>}

* calc_duplicate_range — duplicate a cell range to ANOTHER LOCATION
  WITHIN the spreadsheet (sheet → sheet), preserving cell types
  {"action":"calc_duplicate_range",
   "source_sheet":<src>, "source_range":<a1>,
   "dest_sheet":<dst>, "dest_anchor":<a1>}
  Destination sheet MUST already exist; chain calc_new_sheet first
  if not. This does NOT touch the clipboard — never use it for a
  cross-app copy/paste (use calc_copy_to_clipboard for that).

* calc_copy_to_clipboard — select an EXACT cell range and copy it to
  the CLIPBOARD, ready to paste into another application
  {"action":"calc_copy_to_clipboard", "sheet":<sheet>, "range_ref":<a1>}
  This SELECTS AND COPIES in ONE action — after it runs the data IS
  on the clipboard. Do NOT follow it with a separate copy / Ctrl+C /
  calc_duplicate_range step. The next step is to switch to the
  target app and paste. Use this — NOT a GUI drag-select — whenever
  the task moves Calc data into another app. It selects PRECISELY
  ``range_ref`` (no extra rows / columns).
  RANGE — the #1 mistake: a copied table needs its HEADER ROW.
  Start the range at row 1 (the header). Copying a data row at
  sheet-row N means the header PLUS that row → range_ref
  "A1:<lastcol>N", not "A<N>:<lastcol>N" (which drops the header).
  Include only the row(s) the task names. Optional "copy":false
  selects without copying (rare).

* calc_hide_range — hide entire rows and/or columns (View-only;
  source cells are NOT modified, satisfies "do not delete data"
  constraints)
  {"action":"calc_hide_range", "sheet":<sheet>,
   "rows":[<1-based row number>, ...],
   "cols":[<column letter or 0-based index>, ...]}
  Provide either ``rows``, ``cols``, or both (at least one
  required). Use for "hide rows where X" / "hide column Y" tasks.

* calc_insert_chart — insert an embedded chart. ONE-SHOT operation:
  the chart references ``data_sheet`` directly — DO NOT copy data to
  the destination sheet first. Two data-feed forms (pick ONE):

  Form (1): single contiguous range. Best when the chart visualises
  a whole table.
  {"action":"calc_insert_chart",
   "sheet":<placement — DEFAULTS to data_sheet>,
   "data_sheet":<source sheet>,
   "data_range":<contiguous A1 range over the source area>,
   "chart_type":<"column"|"bar"|"line"|"pie"|"scatter"|"area">,
   "title":<chart title>,                  // optional
   "anchor":<single-cell A1 for chart top-left>}  // default "A1"

  Form (2): explicit categories + values list. REQUIRED when:
    - x-axis label column and y-axis value column are NON-ADJACENT
      ("X-axis is <colA>, Y-axis is <colB>" tasks);
    - the chart shows ONLY a single summary row / column whose
      header row supplies category labels.
  Use form (2) instead of an over-wide form (1) — picking a wide
  ``data_range`` when only one row/column is wanted yields N
  spurious extra series that the eval rejects.
  {"action":"calc_insert_chart",
   "sheet":<placement>, "data_sheet":<source>,
   "categories":<A1 range for x-axis / category labels>,
   "values":[<A1 range>, ...],   // one entry per series
   "chart_type":<type>, "title":<title>}

  Form (2) emission protocol — derive categories from the task +
  SP, NOT from a default "categories = leftmost label column"
  prior:
   HARD CONSTRAINT (verify BEFORE emitting): the cell count of
   ``categories`` MUST EQUAL the cell count of every ``values[i]``.
   Off by one — typically caused by extending categories into the
   leftmost-label cell or the header cell — is REJECT; recompute
   the range, don't emit.
   1. Parse the chart instruction for the x-axis LABEL NAME
      (whatever noun the instruction binds to the x-axis: e.g.
      "x-axis is <X>", "for each <X>", "<X> on the X-axis").
   2. Find which SP cells contain those labels — almost always
      the ``hdr:`` row OR the leftmost data column. Cite the A1
      range.
   3. Find which SP cells hold the y-axis data (summary row,
      summary column, or the column the instruction names).
   4. TRIM step 2's range to cover EXACTLY the same cols (or
      rows) as step 3's range. If step 3 spans cols X..Y, step 2
      spans cols X..Y — not col A..Y if A isn't part of X..Y,
      not col X..Z.
   5. Emit ``categories=<trimmed step 2>``, ``values=[<step 3>]``.
      SAME shape: row values ⇒ row categories; column values ⇒
      column categories.
  If step 1's label name is not the leftmost-column header,
  categories almost certainly comes from row 1 — DO NOT default
  to col A in that case.

  Form (1) and (2) are mutually exclusive — pass ONE shape per call.

  Other kwargs (shared):
   "subtype":<"clustered"|"stacked"|"percent">,
   "name":<chart name — OMIT for multiple charts on one sheet so the
           framework auto-picks Chart1/Chart2/...; same name twice
           overwrites>,
   "column_headers":<true|false>,    // form (1) only, default true
   "row_headers":<true|false>}       // form (1) only, default true

  The evaluator compares chart series formulas as a dict key —
  series count must match the gold. Form (2) is the safest way to
  force series count = len(values); form (1) emits one series per
  non-header row in ``data_range``. Picking an arbitrary sheet
  name auto-creates an empty side-sheet and the ``chart_exists``
  verify on the data sheet then FAILs.

* calc_export_file — export the OPEN document to PDF / CSV / HTML via
  UNO ``storeToURL``. ONE-SHOT replacement for the unreliable GUI menu
  path (File → Export → pick filter → dropdown → click Save):
  {"action":"calc_export_file",
   "path":<absolute path on VM ending in .pdf / .csv / .html>,
   "format":<"pdf"|"csv"|"html"|"auto">,  // default "auto" — infer from path
   // PDF-specific (ignored for CSV)
   "fit_width_pages":<int>,         // 1 = "fit to 1 page wide"
   "fit_height_pages":<int>,        // 1 = "fit to 1 page tall"
   // CSV-specific (ignored for PDF)
   "sheet":<sheet name>,            // default = active sheet
   "delimiter":<single char>,       // default ","
   "encoding":<"utf-8"|"ascii"|"cp1252">,  // default "utf-8"
   "save_formulas":<bool>}          // default false (values "as shown")
  COMPOSE ``path`` FROM THE INSTRUCTION. Read the directory (e.g.
  home directory, current directory, a named path) and the desired
  basename from the task wording, then pick the extension matching
  ``format``. The basename comes from the instruction's filename
  rule (e.g. "same name as the spreadsheet" → use the open file's
  basename WITHOUT its source extension; "rename to <Y>" → use Y).
  The op silently strips a stray source extension if you pass
  ``<base>.xlsx.pdf`` — but it is cleanest to drop the ``.xlsx``
  yourself. Pair with ``file_exists`` verify on the same ``path``.

* calc_set_color — set cell background, font color, bold, italic on
  a range. Replaces the GUI Format → Cells → Background / Font path.
  {"action":"calc_set_color",
   "sheet":<sheet>, "range_ref":<a1 range>,
   "bgcolor":<"#RRGGBB" string>,    // optional
   "font_color":<"#RRGGBB" string>, // optional
   "bold":<bool>,                   // optional
   "italic":<bool>,                 // optional
   "where_formula":<formula str>}   // optional — apply colour ONLY
                                    //   to cells where this LO Calc
                                    //   predicate evaluates TRUE.
                                    //   Use ``{cell}`` for the
                                    //   per-cell A1 ref and ``{r}``
                                    //   for the 1-based row number.
  Pass at least one of bgcolor / font_color / bold / italic. Use
  ``where_formula`` for "highlight cells matching condition" tasks
  (the natural Excel "Conditional Formatting" pattern). LO Calc
  formula syntax — ``;`` between multi-arg function args, NOT ``,``:
    "WEEKDAY({cell};2)>=6"     # Sat/Sun cells in a date column
    "MOD({r};2)=0"             # alternate rows
    "{cell}=MAX($B$2:$B$25)"   # the cell with the max value
  Empty cells are skipped by default (``skip_empty=true``) — required
  because LO evaluates ``WEEKDAY(<empty>;2)`` as ``6`` (1899-12-30 =
  Saturday) and would otherwise colour every trailing blank row. To
  COLOUR blank cells on purpose (predicate ``ISBLANK({cell})``), pass
  ``skip_empty=false`` to override.
  The op evaluates the formula via a scratch column far from the
  used area and cleans it up after — the only cells touched are
  inside ``range_ref``. Verify with ``cell_color`` on a known-true
  representative cell.

* calc_merge_cells — merge a rectangular A1 range into one visual
  cell, optionally writing a value to the anchor first.
  {"action":"calc_merge_cells",
   "sheet":<sheet>, "range_ref":<a1 multi-cell range>,
   "value":<string>}                // optional, written into anchor
                                    //   BEFORE merge — order matters
  Verify with ``cell_merged`` on the anchor cell (top-left of the
  range, e.g. "A1" for range "A1:C1").

* calc_data_validation — attach a data-validation rule (most often a
  list-of-allowed-values dropdown) to a range.
  {"action":"calc_data_validation",
   "sheet":<sheet>, "range_ref":<a1 range, often a whole column>,
   "allowed_values":[<str>, <str>, ...], // required for type="list"
   "type":<"list"|"any"|"whole"|"decimal"|"date"|"time"|
            "text_len"|"custom">,        // default "list"
   "show_dropdown":<bool>}               // default true
  ``allowed_values`` are auto-quoted and joined into LO's Formula1.
  Verify with ``data_validation`` (same kwargs).

* calc_reorder_columns — rearrange the columns of a data block by
  column letter OR by header name, leaving the rest of the sheet
  untouched. Replaces the cut/insert column GUI dance.
  {"action":"calc_reorder_columns",
   "sheet":<sheet>,
   "new_order":[<letter or header string>, ...],
                                   // e.g. ["A","C","B","D"] OR
                                   //      ["Date","First","Last"]
   "first_row":<int>,              // optional, defaults to row 2
                                   //   (data below header)
   "last_row":<int>}               // optional, defaults to last used
  Output columns are written starting at column A. Verify with
  ``cell_value`` on row 2 / row last of each renamed column.

* calc_delete_columns — PERMANENTLY remove entire columns (cells gone,
  columns to the right shift left to close the gap). This is DELETE,
  not hide — use it only when the task says "delete / remove column X"
  (for "hide column X" use calc_hide_range instead).
  {"action":"calc_delete_columns", "sheet":<sheet>,
   "cols":[<column letter or 0-based index>, ...]}
  Non-contiguous selections are fine (removed in descending order).
  Verify with ``cell_value`` on the cells that should now hold the
  shifted-left data.

* calc_delete_rows — PERMANENTLY remove entire rows (cells gone, rows
  below shift up to close the gap). DELETE, not hide — use only for
  "delete / remove row N" / "delete the empty rows"; for "hide" use
  calc_hide_range.
  {"action":"calc_delete_rows", "sheet":<sheet>,
   "rows":[<1-based row number>, ...]}
  Non-contiguous selections are fine (removed in descending order).

# How to pick

Per-row formula column → calc_fill_column.
Single cell write       → calc_set_cell.
Multi-column block      → calc_set_range.
Delimiter text split    → calc_split_text (default for "split by space/comma
                          into N columns" — N out_cols, no formula writing).
Group-by pivot          → calc_pivot_table.
Sheet management        → calc_new_sheet / calc_rename_sheet.
Range structural        → calc_sort_range / calc_freeze /
                          calc_format_number / calc_duplicate_range.
Hide rows / columns     → calc_hide_range ("hide rows where X" or
                          "hide column Y"; pass rows=[…] / cols=[…]).
Delete rows / columns   → calc_delete_rows / calc_delete_columns
                          ("delete / remove row N or column X"; the
                          cells are removed and the sheet reflows —
                          NOT the same as hide).
Insert chart            → calc_insert_chart (ONE step — destination
                          sheet is auto-created; chart references
                          data_sheet directly, no copy needed).
Export to PDF / CSV     → calc_export_file (ONE step — path picks the
                          format via extension; fit-to-page lives on
                          the action, no Page Style navigation needed).
Cell color / bold       → calc_set_color (bgcolor / font_color / bold;
                          replaces Format → Cells GUI).
Merge cells             → calc_merge_cells (with optional ``value``
                          to write into the anchor in the same step).
Data-validation dropdown → calc_data_validation (allowed_values=[...]
                          for the dropdown list).
Reorder / move columns  → calc_reorder_columns (new_order accepts
                          column letters OR header names).
Transpose paste         → calc_duplicate_range with ``transpose=true``.

If none of the above fits — that means the task is GUI-only
(opening Tools→Preferences, etc.). Use the click / type / hotkey
actions defined in the FROZEN block.
"""


# ====================================================================
# step → wire format  (decomposer_actor side)
# ====================================================================

def step_to_wire(step: Dict[str, Any]) -> Optional[str]:
    """Convert a decomposer JSON action dict to AST-parseable wire
    string. Returns ``None`` on validation failure — caller logs."""
    action = step.get("action")
    handler = _WIRE_HANDLERS.get(action)
    if handler is None:
        return None
    return handler(step)


def _kw_wire(action_name: str, kwargs: Dict[str, Any]) -> str:
    """Render kwargs to ``action_name(k=v, k=v, ...)`` wire format.

    Skips None values (treated as "not provided"). Uses ``repr`` for
    JSON-safe scalars + collections — the downstream
    ``ast.literal_eval`` reverses it exactly. Returns a single line."""
    parts = []
    for k, v in kwargs.items():
        if v is None:
            continue
        parts.append(f"{k}={v!r}")
    return f"{action_name}({', '.join(parts)})"


def _wire_fill_column(step):
    sheet = step.get("sheet")
    col = step.get("column")
    first_row = step.get("first_row")
    last_row = step.get("last_row")
    formula_template = step.get("formula_template", "")
    header = step.get("header")
    if not (sheet and col and first_row and last_row
            and isinstance(formula_template, str) and formula_template):
        return None
    try:
        first_row = int(first_row); last_row = int(last_row)
    except (TypeError, ValueError):
        return None
    return _kw_wire("calc_fill_column", {
        "sheet": sheet, "column": col,
        "first_row": first_row, "last_row": last_row,
        "formula_template": formula_template,
        "header": header if isinstance(header, str) and header else None,
    })


def _wire_fill_row(step):
    sheet = step.get("sheet")
    row = step.get("row")
    first_col = step.get("first_col")
    last_col = step.get("last_col")
    formula_template = step.get("formula_template", "")
    label = step.get("label")
    if not (sheet and row and first_col and last_col
            and isinstance(formula_template, str) and formula_template):
        return None
    try:
        row = int(row)
    except (TypeError, ValueError):
        return None
    return _kw_wire("calc_fill_row", {
        "sheet": sheet, "row": row,
        "first_col": str(first_col), "last_col": str(last_col),
        "formula_template": formula_template,
        "label": label if isinstance(label, str) and label else None,
    })


def _wire_set_cell(step):
    sheet = step.get("sheet")
    a1_ref = step.get("a1_ref")
    value = step.get("value")
    formula = step.get("formula")
    if not (sheet and a1_ref):
        return None
    # Exactly one of value / formula
    if (value is None) == (formula is None):
        return None
    return _kw_wire("calc_set_cell", {
        "sheet": sheet, "a1_ref": a1_ref,
        "value": value, "formula": formula,
    })


def _wire_set_range(step):
    sheet = step.get("sheet")
    start_a1 = step.get("start_a1")
    values = step.get("values")
    if not (sheet and start_a1 and isinstance(values, list) and values):
        return None
    if not all(isinstance(r, list) for r in values):
        return None
    return _kw_wire("calc_set_range", {
        "sheet": sheet, "start_a1": start_a1, "values": values,
    })


def _wire_new_sheet(step):
    name = step.get("name")
    position = step.get("position", "end")
    source_sheet = step.get("source_sheet")
    if not name:
        return None
    return _kw_wire("calc_new_sheet", {
        "name": name,
        "position": position,
        "source_sheet": source_sheet,
    })


def _wire_rename_sheet(step):
    current = step.get("current_name")
    new = step.get("new_name")
    if not (current and new):
        return None
    return _kw_wire("calc_rename_sheet", {
        "current_name": current, "new_name": new,
    })


def _wire_pivot_table(step):
    src_sheet = step.get("source_sheet")
    src_range = step.get("source_range")
    dst_sheet = step.get("dest_sheet")
    dst_anchor = step.get("dest_anchor", "A1")
    row_fields = step.get("row_fields", [])
    col_fields = step.get("column_fields", [])
    filter_fields = step.get("filter_fields", [])
    data_fields = step.get("data_fields", [])
    pivot_name = step.get("pivot_name", "PivotTable1")
    if not (src_sheet and src_range and dst_sheet
            and isinstance(row_fields, list)
            and isinstance(col_fields, list)
            and isinstance(filter_fields, list)
            and isinstance(data_fields, list) and data_fields):
        return None
    return _kw_wire("calc_pivot_table", {
        "source_sheet": src_sheet, "source_range": src_range,
        "dest_sheet": dst_sheet, "dest_anchor": dst_anchor,
        "row_fields": row_fields, "column_fields": col_fields,
        "filter_fields": filter_fields, "data_fields": data_fields,
        "pivot_name": pivot_name,
    })


def _wire_sort_range(step):
    sheet = step.get("sheet")
    range_ref = step.get("range_ref")
    sort_columns = step.get("sort_columns")
    contains_header = step.get("contains_header", True)
    if not (sheet and range_ref
            and isinstance(sort_columns, list) and sort_columns):
        return None
    return _kw_wire("calc_sort_range", {
        "sheet": sheet, "range_ref": range_ref,
        "sort_columns": sort_columns,
        "contains_header": bool(contains_header),
    })


def _wire_freeze(step):
    n_cols = step.get("n_cols", 0)
    n_rows = step.get("n_rows", 0)
    try:
        n_cols = int(n_cols); n_rows = int(n_rows)
    except (TypeError, ValueError):
        return None
    if n_cols == 0 and n_rows == 0:
        return None
    return _kw_wire("calc_freeze", {
        "n_cols": n_cols, "n_rows": n_rows,
    })


def _wire_format_number(step):
    sheet = step.get("sheet")
    range_ref = step.get("range_ref")
    format_string = step.get("format_string")
    locale = step.get("locale")
    if not (sheet and range_ref and format_string):
        return None
    return _kw_wire("calc_format_number", {
        "sheet": sheet, "range_ref": range_ref,
        "format_string": format_string,
        "locale": locale,
    })


def _wire_duplicate_range(step):
    src_sheet = step.get("source_sheet")
    src_range = step.get("source_range")
    dst_sheet = step.get("dest_sheet")
    dst_anchor = step.get("dest_anchor")
    if not (src_sheet and src_range and dst_sheet and dst_anchor):
        return None
    kwargs = {
        "source_sheet": src_sheet, "source_range": src_range,
        "dest_sheet": dst_sheet, "dest_anchor": dst_anchor,
    }
    if step.get("transpose") is not None:
        kwargs["transpose"] = bool(step["transpose"])
    return _kw_wire("calc_duplicate_range", kwargs)


def _wire_copy_to_clipboard(step):
    sheet = step.get("sheet")
    range_ref = step.get("range_ref") or step.get("range")
    if not (sheet and range_ref):
        return None
    kwargs = {"sheet": sheet, "range_ref": range_ref}
    if step.get("copy") is not None:
        kwargs["copy"] = bool(step["copy"])
    return _kw_wire("calc_copy_to_clipboard", kwargs)


def _wire_set_color(step):
    sheet = step.get("sheet")
    range_ref = step.get("range_ref")
    if not (sheet and range_ref):
        return None
    kwargs = {"sheet": sheet, "range_ref": range_ref}
    for k in ("bgcolor", "font_color", "bold", "italic",
              "where_formula", "skip_empty"):
        if step.get(k) is not None:
            kwargs[k] = step[k]
    has_attr = any(k in kwargs for k in
                   ("bgcolor", "font_color", "bold", "italic"))
    if not has_attr:
        return None
    return _kw_wire("calc_set_color", kwargs)


def _wire_merge_cells(step):
    sheet = step.get("sheet")
    range_ref = step.get("range_ref")
    if not (sheet and range_ref):
        return None
    kwargs = {"sheet": sheet, "range_ref": range_ref}
    if step.get("value") is not None:
        kwargs["value"] = step["value"]
    return _kw_wire("calc_merge_cells", kwargs)


def _wire_data_validation(step):
    sheet = step.get("sheet")
    range_ref = step.get("range_ref")
    if not (sheet and range_ref):
        return None
    allowed = step.get("allowed_values")
    formula = step.get("formula")
    if not (allowed or formula):
        return None
    kwargs = {"sheet": sheet, "range_ref": range_ref}
    if allowed is not None:
        kwargs["allowed_values"] = list(allowed)
    if formula is not None:
        kwargs["formula"] = formula
    for k in ("type", "show_dropdown"):
        if step.get(k) is not None:
            kwargs[k] = step[k]
    return _kw_wire("calc_data_validation", kwargs)


def _wire_reorder_columns(step):
    sheet = step.get("sheet")
    new_order = step.get("new_order")
    if not sheet or not isinstance(new_order, (list, tuple)) or not new_order:
        return None
    kwargs = {"sheet": sheet, "new_order": list(new_order)}
    for k in ("first_row", "last_row"):
        if step.get(k) is not None:
            kwargs[k] = step[k]
    return _kw_wire("calc_reorder_columns", kwargs)


def _wire_delete_columns(step):
    sheet = step.get("sheet")
    cols = step.get("cols")
    if not sheet or not isinstance(cols, (list, tuple)) or not cols:
        return None
    return _kw_wire("calc_delete_columns",
                    {"sheet": sheet, "cols": list(cols)})


def _wire_delete_rows(step):
    sheet = step.get("sheet")
    rows = step.get("rows")
    if not sheet or not isinstance(rows, (list, tuple)) or not rows:
        return None
    return _kw_wire("calc_delete_rows",
                    {"sheet": sheet, "rows": list(rows)})


def _wire_hide_range(step):
    sheet = step.get("sheet")
    rows = step.get("rows")
    cols = step.get("cols")
    if not sheet or not (rows or cols):
        return None
    kwargs = {"sheet": sheet}
    if rows: kwargs["rows"] = list(rows)
    if cols: kwargs["cols"] = list(cols)
    return _kw_wire("calc_hide_range", kwargs)


def _wire_insert_chart(step):
    sheet = step.get("sheet")
    data_sheet = step.get("data_sheet") or sheet
    data_range = step.get("data_range")
    categories = step.get("categories")
    values = step.get("values")
    # Two forms accepted (see op_insert_chart):
    #   (1) data_range — single contiguous A1 range
    #   (2) categories (optional) + values (list of A1 ranges)
    if not sheet:
        return None
    if not data_range and not values:
        return None
    kwargs = {"sheet": sheet, "data_sheet": data_sheet}
    # Same precedence rule as build layer: form (2) wins.
    if values is not None:
        if isinstance(values, str):
            values = [values]
        kwargs["values"] = list(values)
        if categories is not None:
            kwargs["categories"] = categories
    elif data_range:
        kwargs["data_range"] = data_range
    for k in ("chart_type", "subtype", "title", "name", "anchor",
              "width_cm", "height_cm", "column_headers", "row_headers"):
        if step.get(k) is not None:
            kwargs[k] = step[k]
    return _kw_wire("calc_insert_chart", kwargs)


def _wire_export_file(step):
    path = step.get("path")
    if not path:
        return None
    kwargs = {"path": path}
    for k in ("format", "fit_width_pages", "fit_height_pages",
              "sheet", "delimiter", "quote", "encoding",
              "quote_all", "save_formulas"):
        if step.get(k) is not None:
            kwargs[k] = step[k]
    return _kw_wire("calc_export_file", kwargs)


def _wire_split_text(step):
    sheet = step.get("sheet")
    src_col = step.get("src_col")
    out_cols = step.get("out_cols")
    try:
        first_row = int(step.get("first_row"))
        last_row = int(step.get("last_row"))
    except (TypeError, ValueError):
        return None
    if not (sheet and src_col
            and isinstance(out_cols, (list, tuple)) and out_cols):
        return None
    kwargs = {
        "sheet": sheet, "src_col": src_col,
        "first_row": first_row, "last_row": last_row,
        "out_cols": list(out_cols),
    }
    if step.get("delimiter") is not None:
        kwargs["delimiter"] = step["delimiter"]
    if step.get("last_takes_rest") is not None:
        kwargs["last_takes_rest"] = bool(step["last_takes_rest"])
    return _kw_wire("calc_split_text", kwargs)


def _wire_fill_blanks(step):
    sheet = step.get("sheet")
    range_ref = step.get("range_ref")
    source = step.get("source", "above")
    if not (sheet and range_ref):
        return None
    return _kw_wire("calc_fill_blanks", {
        "sheet": sheet, "range_ref": range_ref,
        "source": str(source).lower() if source else "above",
    })


_WIRE_HANDLERS = {
    "calc_fill_column":   _wire_fill_column,
    "calc_fill_row":      _wire_fill_row,
    "calc_fill_blanks":   _wire_fill_blanks,
    "calc_set_cell":      _wire_set_cell,
    "calc_set_range":     _wire_set_range,
    "calc_split_text":    _wire_split_text,
    "calc_new_sheet":     _wire_new_sheet,
    "calc_rename_sheet":  _wire_rename_sheet,
    "calc_pivot_table":   _wire_pivot_table,
    "calc_sort_range":    _wire_sort_range,
    "calc_freeze":        _wire_freeze,
    "calc_format_number": _wire_format_number,
    "calc_duplicate_range":    _wire_duplicate_range,
    "calc_copy_to_clipboard":  _wire_copy_to_clipboard,
    "calc_hide_range":    _wire_hide_range,
    "calc_insert_chart":  _wire_insert_chart,
    "calc_export_file":   _wire_export_file,
    "calc_set_color":     _wire_set_color,
    "calc_merge_cells":   _wire_merge_cells,
    "calc_data_validation": _wire_data_validation,
    "calc_reorder_columns": _wire_reorder_columns,
    "calc_delete_columns": _wire_delete_columns,
    "calc_delete_rows":   _wire_delete_rows,
}


# ====================================================================
# action_type → runtime script  (planner-side dispatcher)
# ====================================================================

def action_to_runtime_script(
    action_type: str,
    action_inputs: Dict[str, Any],
    *,
    logger: Optional[Any] = None,
) -> Optional[str]:
    """Build the wrapped Python runtime script for a parsed ``calc_*``
    action. Returns the script ready for ``exec`` on the VM, or
    ``None`` on validation failure."""
    builder = _BUILD_HANDLERS.get(action_type)
    if builder is None:
        return None
    return builder(action_inputs, logger)


def _kw_call(op_name: str, kwargs: Dict[str, Any]) -> str:
    """Render ``op_name(k=v, k=v, ...)`` Python call. Skips None
    values so optional op-kwargs default. Uses ``repr`` for all
    values so the generated source is syntactically valid."""
    parts = []
    for k, v in kwargs.items():
        if v is None:
            continue
        parts.append(f"{k}={v!r}")
    return f"{op_name}({', '.join(parts)})"


def _emit_script(call_lines: List[str], domain: Optional[str],
                 logger: Optional[Any]) -> Optional[str]:
    """Append ``print('PASS')`` and run through the wrapper."""
    # cli_run_uno_helpers lives in the sibling ``uno`` package (the single
    # canonical copy; verifiers.py imports the same absolute path). The
    # bare ``.cli_run_uno_helpers`` here looked for a calc-local module that
    # never existed → ModuleNotFoundError at first new-sheet action.
    from ..uno.cli_run_uno_helpers import build_runtime_script
    code = "\n".join(list(call_lines) + ["print('PASS')"])
    try:
        # Ops library/setup must match the ACTION's domain (calc), not the
        # active-window __current_domain (cli_run_uno binds ``doc`` by service
        # type; focus is irrelevant). See impress/actions for the bug context.
        return build_runtime_script(code, domain="libreoffice_calc")
    except ValueError as e:
        if logger:
            logger.warning("[calc action] build failed: %s", e)
        return None


def _build_fill_blanks(inputs, logger):
    sheet = inputs.get("sheet")
    range_ref = inputs.get("range_ref")
    source = inputs.get("source") or "above"
    domain = inputs.get("__current_domain")
    if not (sheet and range_ref):
        if logger:
            logger.warning("[calc_fill_blanks] missing sheet/range_ref")
        return None
    if str(source).lower() not in ("above", "below", "left", "right"):
        if logger:
            logger.warning("[calc_fill_blanks] invalid source=%r", source)
        return None
    return _emit_script([_kw_call("op_fill_blanks", {
        "sheet": sheet, "range_ref": range_ref,
        "source": str(source).lower(),
    })], domain, logger)


def _build_fill_row(inputs, logger):
    sheet = inputs.get("sheet")
    first_col = inputs.get("first_col")
    last_col = inputs.get("last_col")
    try:
        row = int(inputs.get("row", 0))
    except (TypeError, ValueError):
        if logger:
            logger.warning("[calc_fill_row] non-int row: %r", inputs)
        return None
    formula_template = inputs.get("formula_template", "")
    label = inputs.get("label")
    domain = inputs.get("__current_domain")
    if not (sheet and first_col and last_col and row and formula_template):
        if logger:
            logger.warning(
                "[calc_fill_row] missing params: sheet=%r row=%r "
                "first_col=%r last_col=%r tpl=%r",
                sheet, row, first_col, last_col,
                (formula_template or "")[:60])
        return None
    # Convert column letters to 0-based indices for iteration.
    def _letter_to_idx(s):
        s = str(s).strip().upper()
        n = 0
        for ch in s:
            if not ("A" <= ch <= "Z"):
                raise ValueError(f"invalid column letter {s!r}")
            n = n * 26 + (ord(ch) - ord("A") + 1)
        return n - 1
    def _idx_to_letter(n):
        n = int(n)
        s = ""
        n += 1
        while n > 0:
            n, rem = divmod(n - 1, 26)
            s = chr(ord("A") + rem) + s
        return s
    try:
        first_idx = _letter_to_idx(first_col)
        last_idx = _letter_to_idx(last_col)
    except ValueError as e:
        if logger:
            logger.warning("[calc_fill_row] %s", e)
        return None
    if first_idx > last_idx:
        if logger:
            logger.warning("[calc_fill_row] first_col %r > last_col %r",
                           first_col, last_col)
        return None
    # Substitute placeholders per cell. Mirrors calc_fill_column's
    # ``{r}`` / ``{seq}`` but with COLUMN semantics:
    #   {c}      — current column letter (B, C, D, …). Use in cell refs
    #              for SUM-down or other vertical aggregates.
    #   {prev_c} — previous column letter (one to the LEFT of {c}).
    #              Use for growth-style cross-column formulas like
    #              ``=({c}12-{prev_c}12)/{prev_c}12``. When applied to
    #              the FIRST column in the range, the template is
    #              skipped — the cell is left empty (matches the
    #              "Jan growth = blank" / "first month has no prev"
    #              shape of growth rows).
    #   {seq}    — 1-based sequence number from first_col (1, 2, 3, …).
    row_values = []
    for k, idx in enumerate(range(first_idx, last_idx + 1)):
        c_letter = _idx_to_letter(idx)
        if "{prev_c}" in formula_template:
            if idx == first_idx:
                row_values.append(None)
                continue
            prev_letter = _idx_to_letter(idx - 1)
            cell_text = formula_template.replace(
                "{c}", c_letter).replace(
                "{prev_c}", prev_letter).replace(
                "{seq}", str(k + 1))
        else:
            cell_text = formula_template.replace(
                "{c}", c_letter).replace(
                "{seq}", str(k + 1))
        row_values.append(cell_text)
    # Mirror calc_fill_column's auto `=` prefix for missing-equals
    # formulas — same detection (formula markers present, no leading
    # `=`). Applies element-wise; None entries (e.g. {prev_c} skip
    # for first col) are left untouched.
    import re as _re_r
    _FORMULA_HINT_RE_R = _re_r.compile(
        r"[+\-*/&^()]|"
        r"[A-Za-z_][A-Za-z0-9_]*\([^)]*\)|"
        r"\b[A-Za-z_][A-Za-z0-9_ ]*\.[A-Z]+\d"
    )
    for i, f in enumerate(row_values):
        if isinstance(f, str) and f and not f.startswith("="):
            if _FORMULA_HINT_RE_R.search(f):
                row_values[i] = "=" + f
                if logger and i == 0:
                    logger.info(
                        "[calc_fill_row] auto-prefixed `=` to "
                        "formula_template: %r", formula_template[:80])
    lines = []
    if label and isinstance(label, str):
        # Label sits one column to the LEFT of first_col. If first_col
        # is already 'A', the planner needs to embed the label inside
        # the formula_template / values list instead — skip the side
        # write rather than overwrite some unrelated cell.
        if first_idx >= 1:
            lbl_col = _idx_to_letter(first_idx - 1)
            lines.append(_kw_call("op_set_cell", {
                "sheet": sheet,
                "a1_ref": f"{lbl_col}{row}",
                "value": label,
            }))
    start_a1 = f"{first_col}{row}"
    lines.append(_kw_call("op_set_range_values", {
        "sheet": sheet, "start_a1": start_a1, "values": [row_values],
    }))
    return _emit_script(lines, domain, logger)


def _build_fill_column(inputs, logger):
    sheet = inputs.get("sheet")
    col = inputs.get("column")
    try:
        first_row = int(inputs.get("first_row", 0))
        last_row = int(inputs.get("last_row", 0))
    except (TypeError, ValueError):
        if logger:
            logger.warning("[calc_fill_column] non-int row params: %r", inputs)
        return None
    formula_template = inputs.get("formula_template", "")
    header = inputs.get("header")
    domain = inputs.get("__current_domain")
    if not (sheet and col and first_row and last_row and formula_template):
        if logger:
            logger.warning(
                "[calc_fill_column] missing params: sheet=%r col=%r "
                "first_row=%r last_row=%r tpl=%r",
                sheet, col, first_row, last_row,
                (formula_template or "")[:60])
        return None
    if first_row > last_row:
        if logger:
            logger.warning("[calc_fill_column] first_row %d > last_row %d",
                           first_row, last_row)
        return None
    # Substitute placeholders so a template can use whichever
    # semantics it needs (or both):
    #   {r}        — sheet row index. Use to build CELL REFERENCES
    #                (``=VLOOKUP(E{r};$D$2:$E$7;2;1)``).
    #   {r+N}/{r-N} — row index OFFSET by N (positive int). Use for
    #                cross-row formulas like year-on-year growth:
    #                ``=(B{r+1}-B{r})/B{r}``. Substitution is done
    #                BEFORE the bare ``{r}`` replacement so the
    #                offset form isn't mangled.
    #   {seq}      — 1-based sequence number from ``first_row``
    #                (1, 2, 3, ...). Use when the LITERAL VALUE
    #                depends on position (``"No. {seq}"`` →
    #                "No. 1", "No. 2", ...).
    import re as _re
    _offset_re = _re.compile(r"\{r([+-])(\d+)\}")
    def _expand(tpl, r):
        def _sub(m):
            sign, n = m.group(1), int(m.group(2))
            return str(r + n if sign == "+" else r - n)
        # Offset form FIRST so a literal `{r}` inside an offset like
        # `{r-1}` isn't replaced too early.
        tpl = _offset_re.sub(_sub, tpl)
        return tpl.replace("{r}", str(r)).replace(
            "{seq}", str(r - first_row + 1))
    formulas = [
        _expand(formula_template, r)
        for r in range(first_row, last_row + 1)
    ]
    # Auto-prefix `=` when the template OBVIOUSLY emits a formula but
    # the actor forgot the leading equals sign. ``op_set_range_values``
    # uses ``startswith("=")`` to decide formula vs literal text — a
    # `(A{r}-B{r-1})/B{r-1}` template without `=` lands as TEXT and
    # the `expected_formula_contains` verify reports got_type='TEXT'.
    # Detection: not already a formula AND contains a formula marker
    # (arithmetic / function call / cross-sheet ref). Pure-text
    # templates like `"No. {seq}"` / `"Item_{seq}"` carry no markers
    # and stay untouched.
    _FORMULA_HINT_RE = _re.compile(
        r"[+\-*/&^()]|"                              # arithmetic / concat / paren
        r"[A-Za-z_][A-Za-z0-9_]*\([^)]*\)|"          # function call e.g. SUM(...)
        r"\b[A-Za-z_][A-Za-z0-9_ ]*\.[A-Z]+\d"       # cross-sheet ref Sheet1.B2
    )
    for i, f in enumerate(formulas):
        if isinstance(f, str) and f and not f.startswith("="):
            if _FORMULA_HINT_RE.search(f):
                formulas[i] = "=" + f
                if logger and i == 0:
                    logger.info(
                        "[calc_fill_column] auto-prefixed `=` to "
                        "formula_template (looked like a formula but "
                        "leading `=` was missing): %r",
                        formula_template[:80])
    values = [[f] for f in formulas]
    start_a1 = f"{col}{first_row}"
    lines = []
    if header and isinstance(header, str):
        hdr_row = first_row - 1
        if hdr_row >= 1:
            lines.append(_kw_call("op_set_cell", {
                "sheet": sheet,
                "a1_ref": f"{col}{hdr_row}",
                "value": header,
            }))
    lines.append(_kw_call("op_set_range_values", {
        "sheet": sheet, "start_a1": start_a1, "values": values,
    }))
    return _emit_script(lines, domain, logger)


def _build_set_cell(inputs, logger):
    sheet = inputs.get("sheet")
    a1_ref = inputs.get("a1_ref")
    value = inputs.get("value")
    formula = inputs.get("formula")
    domain = inputs.get("__current_domain")
    if not (sheet and a1_ref):
        if logger:
            logger.warning("[calc_set_cell] missing sheet/a1_ref")
        return None
    if (value is None) == (formula is None):
        if logger:
            logger.warning(
                "[calc_set_cell] need exactly one of value / formula")
        return None
    kw = {"sheet": sheet, "a1_ref": a1_ref}
    if formula is not None:
        kw["formula"] = formula
    else:
        kw["value"] = value
    return _emit_script([_kw_call("op_set_cell", kw)], domain, logger)


def _build_set_range(inputs, logger):
    sheet = inputs.get("sheet")
    start_a1 = inputs.get("start_a1")
    values = inputs.get("values")
    domain = inputs.get("__current_domain")
    if not (sheet and start_a1 and isinstance(values, list) and values):
        if logger:
            logger.warning(
                "[calc_set_range] missing/invalid sheet/start_a1/values")
        return None
    if not all(isinstance(r, list) for r in values):
        if logger:
            logger.warning("[calc_set_range] values must be list-of-lists")
        return None
    return _emit_script([_kw_call("op_set_range_values", {
        "sheet": sheet, "start_a1": start_a1, "values": values,
    })], domain, logger)


def _build_new_sheet(inputs, logger):
    name = inputs.get("name")
    position = inputs.get("position", "end")
    source_sheet = inputs.get("source_sheet")
    domain = inputs.get("__current_domain")
    if not name:
        if logger:
            logger.warning("[calc_new_sheet] missing name")
        return None
    kw = {"name": name, "position": position}
    if source_sheet:
        kw["source_sheet"] = source_sheet
    return _emit_script([_kw_call("op_new_sheet", kw)], domain, logger)


def _build_rename_sheet(inputs, logger):
    current = inputs.get("current_name")
    new = inputs.get("new_name")
    domain = inputs.get("__current_domain")
    if not (current and new):
        if logger:
            logger.warning(
                "[calc_rename_sheet] missing current_name/new_name")
        return None
    return _emit_script([_kw_call("op_rename_sheet", {
        "current_name": current, "new_name": new,
    })], domain, logger)


def _build_pivot_table(inputs, logger):
    src_sheet = inputs.get("source_sheet")
    src_range = inputs.get("source_range")
    dst_sheet = inputs.get("dest_sheet")
    dst_anchor = inputs.get("dest_anchor", "A1")
    row_fields = inputs.get("row_fields", []) or []
    col_fields = inputs.get("column_fields", []) or []
    filter_fields = inputs.get("filter_fields", []) or []
    data_fields = inputs.get("data_fields", []) or []
    pivot_name = inputs.get("pivot_name", "PivotTable1")
    domain = inputs.get("__current_domain")
    if not (src_sheet and src_range and dst_sheet
            and isinstance(data_fields, list) and data_fields):
        if logger:
            logger.warning(
                "[calc_pivot_table] missing source / dest / data_fields")
        return None
    # JSON list-of-2-lists → tuple for the op call
    try:
        data_tuples = [tuple(df) for df in data_fields
                       if isinstance(df, (list, tuple)) and len(df) == 2]
    except Exception:
        if logger:
            logger.warning("[calc_pivot_table] bad data_fields: %r",
                           data_fields)
        return None
    if not data_tuples:
        if logger:
            logger.warning(
                "[calc_pivot_table] no valid (header, func) data_fields entries")
        return None
    return _emit_script([_kw_call("op_pivot_table", {
        "source_sheet": src_sheet, "source_range": src_range,
        "dest_sheet": dst_sheet, "dest_anchor": dst_anchor,
        "row_fields": list(row_fields),
        "column_fields": list(col_fields),
        "filter_fields": list(filter_fields),
        "data_fields": data_tuples,
        "pivot_name": pivot_name,
    })], domain, logger)


def _build_sort_range(inputs, logger):
    sheet = inputs.get("sheet")
    range_ref = inputs.get("range_ref")
    sort_columns = inputs.get("sort_columns")
    contains_header = inputs.get("contains_header", True)
    domain = inputs.get("__current_domain")
    if not (sheet and range_ref and isinstance(sort_columns, list)
            and sort_columns):
        if logger:
            logger.warning(
                "[calc_sort_range] missing sheet / range_ref / sort_columns")
        return None
    # JSON list-of-2-lists → tuple
    try:
        sort_tuples = [(int(sc[0]), bool(sc[1])) for sc in sort_columns
                       if isinstance(sc, (list, tuple)) and len(sc) == 2]
    except Exception:
        if logger:
            logger.warning("[calc_sort_range] bad sort_columns: %r",
                           sort_columns)
        return None
    if not sort_tuples:
        if logger:
            logger.warning("[calc_sort_range] no valid sort_columns entries")
        return None
    return _emit_script([_kw_call("op_sort_range", {
        "sheet": sheet, "range_ref": range_ref,
        "sort_columns": sort_tuples,
        "contains_header": bool(contains_header),
    })], domain, logger)


def _build_freeze(inputs, logger):
    try:
        n_cols = int(inputs.get("n_cols", 0))
        n_rows = int(inputs.get("n_rows", 0))
    except (TypeError, ValueError):
        if logger:
            logger.warning("[calc_freeze] non-int n_cols/n_rows: %r", inputs)
        return None
    domain = inputs.get("__current_domain")
    if n_cols == 0 and n_rows == 0:
        if logger:
            logger.warning("[calc_freeze] both n_cols and n_rows are 0 — "
                           "nothing to freeze")
        return None
    return _emit_script([_kw_call("op_freeze_pane", {
        "n_cols": n_cols, "n_rows": n_rows,
    })], domain, logger)


def _build_format_number(inputs, logger):
    sheet = inputs.get("sheet")
    range_ref = inputs.get("range_ref")
    format_string = _normalize_lo_format_string(inputs.get("format_string"))
    locale = inputs.get("locale")
    domain = inputs.get("__current_domain")
    if not (sheet and range_ref and format_string):
        if logger:
            logger.warning(
                "[calc_format_number] missing sheet/range_ref/format_string")
        return None
    kw = {"sheet": sheet, "range_ref": range_ref,
          "format_string": format_string}
    if locale is not None:
        # Locale arrives as JSON list [lang, country, variant]; the op
        # expects a tuple. Convert if list-shaped.
        if isinstance(locale, list) and len(locale) == 3:
            locale = tuple(locale)
        kw["locale"] = locale
    return _emit_script([_kw_call("op_format_number_range", kw)],
                        domain, logger)


def _build_duplicate_range(inputs, logger):
    src_sheet = inputs.get("source_sheet")
    src_range = inputs.get("source_range")
    dst_sheet = inputs.get("dest_sheet")
    dst_anchor = inputs.get("dest_anchor")
    domain = inputs.get("__current_domain")
    if not (src_sheet and src_range and dst_sheet and dst_anchor):
        if logger:
            logger.warning("[calc_duplicate_range] missing required params")
        return None
    kwargs = {
        "source_sheet": src_sheet, "source_range": src_range,
        "dest_sheet": dst_sheet, "dest_anchor": dst_anchor,
    }
    if inputs.get("transpose") is not None:
        kwargs["transpose"] = bool(inputs["transpose"])
    return _emit_script([_kw_call("op_duplicate_range", kwargs)], domain, logger)


def _build_copy_to_clipboard(inputs, logger):
    sheet = inputs.get("sheet")
    range_ref = inputs.get("range_ref") or inputs.get("range")
    domain = inputs.get("__current_domain")
    if not (sheet and range_ref):
        if logger:
            logger.warning("[calc_copy_to_clipboard] missing sheet/range_ref")
        return None
    kwargs = {"sheet": sheet, "range_ref": range_ref}
    if inputs.get("copy") is not None:
        kwargs["copy"] = bool(inputs["copy"])
    return _emit_script([_kw_call("op_copy_to_clipboard", kwargs)], domain, logger)


def _build_set_color(inputs, logger):
    sheet = inputs.get("sheet")
    range_ref = inputs.get("range_ref")
    domain = inputs.get("__current_domain")
    if not (sheet and range_ref):
        if logger:
            logger.warning("[calc_set_color] missing sheet / range_ref")
        return None
    kwargs = {"sheet": sheet, "range_ref": range_ref}
    for k in ("bgcolor", "font_color", "bold", "italic",
              "where_formula", "skip_empty"):
        if inputs.get(k) is not None:
            kwargs[k] = inputs[k]
    has_attr = any(k in kwargs for k in
                   ("bgcolor", "font_color", "bold", "italic"))
    if not has_attr:
        if logger:
            logger.warning(
                "[calc_set_color] no attributes — pass at least one of "
                "bgcolor / font_color / bold / italic")
        return None
    return _emit_script([_kw_call("op_set_color", kwargs)], domain, logger)


def _build_merge_cells(inputs, logger):
    sheet = inputs.get("sheet")
    range_ref = inputs.get("range_ref")
    domain = inputs.get("__current_domain")
    if not (sheet and range_ref):
        if logger:
            logger.warning("[calc_merge_cells] missing sheet / range_ref")
        return None
    kwargs = {"sheet": sheet, "range_ref": range_ref}
    if inputs.get("value") is not None:
        kwargs["value"] = inputs["value"]
    return _emit_script([_kw_call("op_merge_cells", kwargs)], domain, logger)


def _build_data_validation(inputs, logger):
    sheet = inputs.get("sheet")
    range_ref = inputs.get("range_ref")
    domain = inputs.get("__current_domain")
    if not (sheet and range_ref):
        if logger:
            logger.warning("[calc_data_validation] missing sheet / range_ref")
        return None
    allowed = inputs.get("allowed_values")
    formula = inputs.get("formula")
    if not (allowed or formula):
        if logger:
            logger.warning(
                "[calc_data_validation] need allowed_values or formula")
        return None
    kwargs = {"sheet": sheet, "range_ref": range_ref}
    if allowed is not None:
        kwargs["allowed_values"] = list(allowed)
    if formula is not None:
        kwargs["formula"] = formula
    for k in ("type", "show_dropdown"):
        if inputs.get(k) is not None:
            kwargs[k] = inputs[k]
    return _emit_script([_kw_call("op_data_validation", kwargs)],
                         domain, logger)


def _build_reorder_columns(inputs, logger):
    sheet = inputs.get("sheet")
    new_order = inputs.get("new_order")
    domain = inputs.get("__current_domain")
    if not sheet:
        if logger:
            logger.warning("[calc_reorder_columns] missing sheet")
        return None
    if not isinstance(new_order, (list, tuple)) or not new_order:
        if logger:
            logger.warning(
                "[calc_reorder_columns] new_order must be a non-empty list "
                "of column letters or header names")
        return None
    kwargs = {"sheet": sheet, "new_order": list(new_order)}
    for k in ("first_row", "last_row"):
        if inputs.get(k) is not None:
            kwargs[k] = inputs[k]
    return _emit_script([_kw_call("op_reorder_columns", kwargs)],
                         domain, logger)


def _build_delete_columns(inputs, logger):
    sheet = inputs.get("sheet")
    cols = inputs.get("cols")
    domain = inputs.get("__current_domain")
    if not sheet:
        if logger:
            logger.warning("[calc_delete_columns] missing sheet")
        return None
    if not isinstance(cols, (list, tuple)) or not cols:
        if logger:
            logger.warning(
                "[calc_delete_columns] cols must be a non-empty list of "
                "column letters or 0-based indices")
        return None
    return _emit_script(
        [_kw_call("op_delete_columns", {"sheet": sheet, "cols": list(cols)})],
        domain, logger)


def _build_delete_rows(inputs, logger):
    sheet = inputs.get("sheet")
    rows = inputs.get("rows")
    domain = inputs.get("__current_domain")
    if not sheet:
        if logger:
            logger.warning("[calc_delete_rows] missing sheet")
        return None
    if not isinstance(rows, (list, tuple)) or not rows:
        if logger:
            logger.warning(
                "[calc_delete_rows] rows must be a non-empty list of "
                "1-based row numbers")
        return None
    return _emit_script(
        [_kw_call("op_delete_rows", {"sheet": sheet, "rows": list(rows)})],
        domain, logger)


def _build_hide_range(inputs, logger):
    sheet = inputs.get("sheet")
    rows = inputs.get("rows")
    cols = inputs.get("cols")
    domain = inputs.get("__current_domain")
    if not sheet:
        if logger:
            logger.warning("[calc_hide_range] missing sheet")
        return None
    if not (rows or cols):
        if logger:
            logger.warning("[calc_hide_range] need rows and/or cols")
        return None
    kwargs = {"sheet": sheet}
    if rows: kwargs["rows"] = list(rows)
    if cols: kwargs["cols"] = list(cols)
    return _emit_script([_kw_call("op_hide_range", kwargs)], domain, logger)


def _build_insert_chart(inputs, logger):
    sheet = inputs.get("sheet")
    data_sheet = inputs.get("data_sheet") or sheet
    data_range = inputs.get("data_range")
    categories = inputs.get("categories")
    values = inputs.get("values")
    domain = inputs.get("__current_domain")
    if not sheet or (not data_range and not values):
        if logger:
            logger.warning(
                "[calc_insert_chart] need sheet + (data_range OR values); "
                "sheet=%r data_range=%r values=%r",
                sheet, data_range, values)
        return None
    kwargs = {"sheet": sheet, "data_sheet": data_sheet}
    # If both forms were emitted, prefer form (2) — the actor explicitly
    # naming categories+values is the more careful signal. Drop the
    # redundant data_range silently so the op call doesn't ValueError.
    if values is not None:
        if isinstance(values, str):
            values = [values]
        kwargs["values"] = list(values)
        if categories is not None:
            kwargs["categories"] = categories
        if logger and data_range:
            logger.info(
                "[calc_insert_chart] dropped redundant data_range=%r "
                "(form 2 categories+values takes precedence)",
                data_range)
    elif data_range:
        kwargs["data_range"] = data_range
    for k in ("chart_type", "subtype", "title", "name", "anchor",
              "width_cm", "height_cm", "column_headers", "row_headers"):
        if inputs.get(k) is not None:
            kwargs[k] = inputs[k]
    return _emit_script([_kw_call("op_insert_chart", kwargs)],
                         domain, logger)


def _build_export_file(inputs, logger):
    path = inputs.get("path")
    domain = inputs.get("__current_domain")
    if not path:
        if logger:
            logger.warning("[calc_export_file] missing required: path=%r", path)
        return None
    kwargs = {"path": path}
    for k in ("format", "fit_width_pages", "fit_height_pages",
              "sheet", "delimiter", "quote", "encoding",
              "quote_all", "save_formulas"):
        if inputs.get(k) is not None:
            kwargs[k] = inputs[k]
    return _emit_script([_kw_call("op_export_file", kwargs)],
                         domain, logger)


def _build_split_text(inputs, logger):
    sheet = inputs.get("sheet")
    src_col = inputs.get("src_col")
    out_cols = inputs.get("out_cols")
    try:
        first_row = int(inputs.get("first_row"))
        last_row = int(inputs.get("last_row"))
    except (TypeError, ValueError):
        if logger:
            logger.warning("[calc_split_text] non-int row params: %r", inputs)
        return None
    if not (sheet and src_col
            and isinstance(out_cols, (list, tuple)) and out_cols):
        if logger:
            logger.warning("[calc_split_text] missing sheet / src_col / out_cols")
        return None
    if not (last_row >= first_row >= 1):
        if logger:
            logger.warning("[calc_split_text] bad row range: %r-%r",
                           first_row, last_row)
        return None
    domain = inputs.get("__current_domain")
    kwargs = {
        "sheet": sheet, "src_col": src_col,
        "first_row": first_row, "last_row": last_row,
        "out_cols": list(out_cols),
    }
    if inputs.get("delimiter") is not None:
        kwargs["delimiter"] = inputs["delimiter"]
    if inputs.get("last_takes_rest") is not None:
        kwargs["last_takes_rest"] = bool(inputs["last_takes_rest"])
    return _emit_script([_kw_call("op_split_text", kwargs)], domain, logger)


_BUILD_HANDLERS = {
    "calc_fill_column":   _build_fill_column,
    "calc_fill_row":      _build_fill_row,
    "calc_fill_blanks":   _build_fill_blanks,
    "calc_set_cell":      _build_set_cell,
    "calc_set_range":     _build_set_range,
    "calc_split_text":    _build_split_text,
    "calc_new_sheet":     _build_new_sheet,
    "calc_rename_sheet":  _build_rename_sheet,
    "calc_pivot_table":   _build_pivot_table,
    "calc_sort_range":    _build_sort_range,
    "calc_freeze":        _build_freeze,
    "calc_format_number": _build_format_number,
    "calc_duplicate_range":    _build_duplicate_range,
    "calc_copy_to_clipboard":  _build_copy_to_clipboard,
    "calc_hide_range":    _build_hide_range,
    "calc_insert_chart":  _build_insert_chart,
    "calc_export_file":   _build_export_file,
    "calc_set_color":     _build_set_color,
    "calc_merge_cells":   _build_merge_cells,
    "calc_data_validation": _build_data_validation,
    "calc_reorder_columns": _build_reorder_columns,
    "calc_delete_columns": _build_delete_columns,
    "calc_delete_rows":   _build_delete_rows,
}
