# LibreOffice Calc domain knowledge

This block is injected into the planner and init_ledger prompts ONLY
when the task's domain prefix is `[libreoffice_calc]`. It does NOT
apply to other domains.

## Path: structured `calc_*` JSON actions only

Calc operations go through structured **`calc_*` actions** (flat
JSON kwargs). The framework deterministically builds the equivalent
Python from the kwargs — actors never write Python source, planners
never plan Python source. Available calc actions cover sheet
management (`calc_new_sheet`, `calc_rename_sheet`, `calc_duplicate_range`),
cell writes (`calc_set_cell`, `calc_set_range`, `calc_fill_column`),
analysis (`calc_pivot_table`, `calc_sort_range`), and presentation
(`calc_freeze`, `calc_format_number`). Full schemas are listed in the
"Calc structured actions" block injected into the actor's prompt.

GUI (click / type / hotkey) is reserved for tasks no `calc_*` action
covers: data validation dropdown, cell background color, opening
Tools→Preferences. (Chart insertion and PDF/CSV export are covered
by `calc_insert_chart` / `calc_export_file` — see pattern (g) / (h).)

<!-- [FIX cli_spreadsheet_knowledge] Added so that planner/actor don't
     try to byte-cat binary spreadsheets when the task explicitly says
     "command line" (xlsx/ods are zip containers — ``cat`` produces
     garbage). General; doesn't hint at any specific task answer.
     Roll back by deleting this commented block. -->
## Shell-level operations (`.xlsx` / `.ods` are BINARY zip containers)

When the task explicitly mandates the **command line** (and only then
— prefer `calc_*` structured actions otherwise), remember:

* `.xlsx`, `.xlsm`, `.ods`, `.xls` are **binary** archives. Never
  `cat`, `head`, `paste`, `cut`, `grep`, `awk`, or `sed` them
  directly — you'll get random bytes from the zip headers. Convert
  first.
* Conversion: `soffice --headless --convert-to csv <file> --outdir <dir>`
  writes `<basename>.csv` next to the input. Works for any LO-readable
  spreadsheet. For PDF/HTML pick `pdf` / `html` instead.
  - If a Calc instance is already running with the file open, add a
    private user profile to avoid lock contention:
    `-env:UserInstallation=file:///tmp/_tmpprof_<n>/`.
* Column operations on the resulting CSVs: `paste`, `cut`,
  `awk -F,`. Row-wise concat of two single-column files into one
  column = `paste -d '' a.csv b.csv` (joins each row's strings end-
  to-end with no delimiter).
* Open the result in Calc **from the terminal** when the task says
  "open from terminal" — that's what `libreoffice --calc <file>` is
  for. The trailing `&` is fine; launch it from the terminal session
  so the process stays attached to that session's tty.

## STOP — answer these BEFORE deciding outcomes / plan

Tasks fail when the model jumps from the instruction text to a
formula without checking how the workbook is actually laid out.
Before generating outcomes (init_ledger) or planning code (planner),
answer these FOUR questions OUT LOUD in your reasoning, using the
WORKBOOK STRUCTURE / SheetPerceiver block as the source of truth.

**Q1 — Row semantics**:  what does ONE row of each sheet represent?
  (one customer? one transaction? one month? one threshold?)
  Read it off the `hdr:` line + sample data rows. Be concrete:
  "Sheet1 row = one sales rep; B..G = months Jan..Jun".

**Q2 — Task pattern**:  pick ONE.
  (a) per-row formula column → `calc_fill_column`
  (b) multi-column block write (header + per-row values + literal
       text columns) → `calc_set_range`
  (b') group-by aggregation → `calc_pivot_table`
  (c) cross-table lookup (VLOOKUP) → `calc_fill_column` with a
       VLOOKUP formula template
  (d) text split / extract (LEFT / MID / RIGHT / FIND) → either two
       parallel `calc_fill_column` calls or one `calc_set_range`
  (e) structural op → `calc_sort_range` / `calc_freeze` /
       `calc_format_number` / `calc_new_sheet` / `calc_rename_sheet`
       / `calc_duplicate_range` / `calc_set_cell` / `calc_hide_range` /
       `calc_set_color` / `calc_merge_cells` / `calc_data_validation`
       / `calc_reorder_columns`
  (g) chart insertion → `calc_insert_chart`
  (h) file export (PDF / CSV) → `calc_export_file`

  PRECEDENCE — if the instruction names a chart/graph/plot
  ("clustered column chart", "bar chart of <X>", "line chart"),
  pattern is (g). Phrases like "for each <X>" describe the chart's
  category axis, NOT a pivot row grouping — do NOT fall through to
  (b'). One `calc_insert_chart` step is enough; do NOT chain a
  `calc_new_sheet` before it. **DEFAULT `sheet=` to the data sheet
  (same as `data_sheet`)** — the chart lives on the data sheet
  unless the task EXPLICITLY says "in a new sheet" / "on Sheet2" /
  names a destination sheet. Picking an arbitrary chart-title-like
  sheet name (e.g. `sheet="2019 Monthly Costs"`) auto-creates an
  empty side-sheet and places the chart there instead of on the data
  sheet the task expects. For multiple charts on one sheet, omit
  `name=` entirely on every call — the framework auto-picks
  Chart1/Chart2/... so the second call doesn't overwrite the first.

  PRECEDENCE — when the instruction produces an export FILE on
  disk (PDF / CSV / "fit to N pages" scaling), pattern is (h).
  Do NOT navigate File→Export / Save As dialogs by clicking;
  `calc_export_file` does the storeToURL in one step and accepts
  `fit_width_pages` / `fit_height_pages` for PDF scaling. Compose
  the output path from the instruction's directory + basename
  rules; the action takes one absolute `path` ending in `.pdf` or
  `.csv`.

  STRUCTURAL (e) MAP — pick the matching action by feature name:
    * "highlight / color / fill / font color / bold" →
      `calc_set_color` (`bgcolor` / `font_color` are `"#RRGGBB"`)
    * "merge cells" → `calc_merge_cells` (optional `value=` writes
      into the anchor BEFORE merge — one step does both)
    * "dropdown / data validation / allowed values" →
      `calc_data_validation` (`allowed_values=[...]`)
    * "reorder / move columns to <X>, <Y>, <Z>" →
      `calc_reorder_columns` (`new_order=[...]` — letters OR header
      names accepted)
    * "transpose <range>" → `calc_duplicate_range` with `transpose=true`
    * "freeze the range <X>:<Y>" → `calc_freeze` — the range names
      what should STAY VISIBLE while scrolling, so the freeze must
      cover EVERY row and column the range spans. Set
      `n_rows = <last row of Y>` and
      `n_cols = <last column index of Y>` (A=1, B=2, C=3, ...).
      The first scrollable cell is then immediately past the
      range's bottom-right corner. Examples:
        - `A1:<col><n>` → `n_rows=n, n_cols=<index of col>`
        - `A1:An`       → `n_rows=n, n_cols=1`
        - `A1:B2`       → `n_rows=2, n_cols=2`

**Q3 — Output target**:  WHICH column / range gets filled?
  * Single-table task with one obvious new column: column letter =
    last_used + 1 from the SP `used=A1:<col><N>` line.
  * If multiple columns share the SAME header (e.g. two "Officer
    Name" columns visible in `hdr:`), pick the EMPTY one — the
    column whose data rows show `·` (empty) in the SP block. Do NOT
    target a column that already has populated values.
  * `<N>` for row count: derive from SP `used=A1:<col><N>`. Don't
    guess.

**Q4 — Lookup spec (Q2 pattern (c) only)**:
  * KEY column = the column in the lookup table that holds values
    matching what you'll query with. VALUE column = the column
    whose value the formula returns. `lookup_range` MUST SPAN BOTH
    (e.g. `D:E` when keys are in D and values in E, NEVER just `E`).
  * Match type — fourth VLOOKUP arg:
      - `;0` (exact) when the key column holds DISCRETE labels
        (product names, branch names, student IDs).
      - `;1` (approx) when the key column holds RANGE THRESHOLDS
        (0, 30, 60, 80, 90 for grade boundaries — student score 42
        won't match any of these exactly; the formula must return
        the row of the largest key ≤ 42).

If any of Q1-Q4 cannot be answered concretely from the SP block,
re-read the block. Don't proceed on a guess.


## Formula syntax + Number-format syntax

See the shared domain file — those sections are factored out
because the planner and the verify-spec author must emit the
exact same strings (any drift breaks `verify_cell_format`).

## Worked examples — match the task to one pattern

### (a) per-row formula column

Task: "Add a Profit column = Sales - COGS for each week"
SP says: Sheet1 used=A1:E20, hdr: 'Week','Sales','COGS','Margin','Notes'

Q1: row = one week.
Q2: per-row formula → calc_fill_column.
Q3: new col F (last_used + 1 = E + 1), rows 2..20.

```json
{"action":"calc_fill_column",
 "sheet":<sheet>,
 "column":<new-col>,
 "first_row":<first-data>,
 "last_row":<N>,
 "formula_template":"=<col-a>{r}-<col-b>{r}",
 "header":<Header>}
```

`{r}` is the SHEET ROW INDEX, substituted per row
(`first_row..last_row`). Use it inside cell references in real
formulas. All `<...>` are read from the SP block — `<N>` from
`used=A1:<col><N>`, the column letters from the `hdr:` line.

**Cross-row offsets — `{r+N}` / `{r-N}`** for formulas that reference
the row above or below (year-over-year change, running diff, etc.):
`formula_template="=(B{r+1}-B{r})/B{r}"`. The bare literal `{r-1}`
without the brackets does NOT auto-expand.

**Cross-sheet refs — qualify cells with `<source-sheet>.` when the
target sheet differs from the source.** Filling Sheet2 with a
template `"=(B{r+1}-B{r})/B{r}"` references Sheet2's own B column
(empty) and produces zeros. Write
`"=(Sheet1.B{r+1}-Sheet1.B{r})/Sheet1.B{r}"` instead so the formula
reads from the data sheet. Use `.` not `!`; quote sheet names with
spaces in `'…'`.

**"Percentage" tasks — write the RAW RATIO + apply `0.00%` format;
NEVER multiply by 100 inside the formula.** When the instruction
says "percentage X" / "X as percent" / "percentage change", emit
the formula as a plain ratio (e.g.
`"=(Sheet1.B{r+1}-Sheet1.B{r})/Sheet1.B{r}"`) and pair it with a
`calc_format_number` step setting `format_string="0.00%"` over the
same range. A percentage cell stores the raw fraction as its
underlying value (e.g. 0.1015), not the `*100` form (10.149); both
render as "10.15%" but the stored value differs — emit the fraction
and let the % number-format display it.

**Position-indexed labels — use `{seq}`, NOT `{r}`**:

Task: "Fill the Seq No. column with values 'No. 1', 'No. 2', …"
SP says: Sheet1 used=A1:E29, hdr: ..., target col B, data rows 2..29.

```json
{"action":"calc_fill_column",
 "sheet":<sheet>, "column":"B",
 "first_row":2, "last_row":29,
 "formula_template":"No. {seq}",
 "header":"Seq No."}
```

`{seq}` is the 1-BASED SEQUENCE NUMBER from `first_row` (1, 2, 3,
…). Use whenever the literal value depends on POSITION rather than
on a cell reference — sequence numbers, "Row 1 / Row 2 / …" labels,
"Item 1" identifiers, etc. **Common bug**: `formula_template="No. {r}"`
produces "No. 2" .. "No. <last_row>" instead of "No. 1" .. "No.
<N-first_row+1>", because `{r}` is the sheet row, not the sequence.

**Header-tagged concatenation** — when a task asks for a column whose
text reads `"<HeaderA>: <A>, <HeaderB>: <B>, ..."` (each value
prefixed by its column header), build it with absolute header refs
+ `&` concat + `FIXED(cell, N)` for decimal control:

```json
{"action":"calc_fill_column",
 "sheet":<sheet>, "column":<new-col>,
 "first_row":<first-data>, "last_row":<N>,
 "formula_template":"=$A$1&\": \"&FIXED(A{r},2)&\", \"&$B$1&\": \"&FIXED(B{r},2)",
 "header":<new-header>}
```

Key pieces:
  * `$A$1` / `$B$1` / … reference each header cell with an ABSOLUTE
    row (the `$` before `1`) so the same template line works for
    every data row.
  * `FIXED(<cell>, <decimals>)` renders the numeric value with the
    requested decimal places as TEXT — Excel's concat operator `&`
    on a raw number renders with full machine precision, so the
    resulting text shows far more digits than requested.
  * `": "` and `", "` literals provide the label-separator and the
    pair-separator respectively. Naive `&" "&` (single space) drops
    the header tag entirely and produces the wrong output string.

**Per-column formula ROW — use `calc_fill_row`, NOT `calc_fill_column`.**
For "new row called Total / Growth / Average" tasks, the formula
varies per COLUMN, not per row. Use the row-direction analogue:

```json
{"action":"calc_fill_row",
 "sheet":<sheet>, "row":<new-row>,
 "first_col":<col-letter>, "last_col":<col-letter>,
 "formula_template":"=SUM({c}<first-data>:{c}<N>)",
 "label":"Total"}
```

`{c}` is the current column letter; `{prev_c}` is the previous one
(for growth-style `=({c}<R>-{prev_c}<R>)/{prev_c}<R>`, the first
cell in the range is left blank automatically). `label` writes one
cell to the LEFT of `first_col`. Picking `calc_fill_column` for a
row task fills DOWN a column instead and corrupts the data layout.

**Chained summary rows — second row's formula references the
FIRST summary row's index, not the raw-data start row.** When a
task creates TWO summary rows in sequence (e.g. an aggregate row
followed by a derived row whose values depend on it), the second
row's `formula_template` row-anchor must match the FIRST summary
row's index, not the raw-data first row. Substitute the actual
row indices from the SP block — do NOT default to the data start
row out of habit. Re-read your q4 spec before emitting: the row
number in the derived row's formula equals the row number the
aggregate step wrote, not 2 / first_data_row.

**Forward-fill BLANK cells from an adjacent direction** — use
`calc_fill_blanks(source=...)`, NOT a formula. A naive
`calc_fill_column` template `'=<col>{r}'` self-references the cell
it's written into and produces a circular reference (cell shows 0
or the header text). Iterating "above" / "below" / "left" / "right"
is a server-side decision (the op walks each line and carries the
last non-empty value forward); the actor can't precompute a `values`
matrix because the SP block elides middle rows.

```json
{"action":"calc_fill_blanks",
 "sheet":<sheet>, "range_ref":<a1>,
 "source":"above"|"below"|"left"|"right"}
```

Non-empty cells in the range stay untouched — the op only writes
into cells that were empty before, so "don't touch irrelevant
regions" constraints are satisfied automatically.

### (b) multi-column block — categories as HEADERS in a new sheet

Task: "Show total <metric> for all <categories> in a new sheet
       with two headers <CategoryHeader> and <TotalHeader>".

Q1: each source row is one record; CATEGORIES live in the source's
    HEADER ROW (each category is its own column).
Q2: multi-column write (category name + total formula per row).
Q3: dest sheet, with one output row per source category column.
    Col A of dest = category names. Col B of dest = SUM formula
    over that one source column.

Plan: chain two actions — `calc_new_sheet` then `calc_set_range`.

```json
{"action":"calc_new_sheet", "name":<dest-sheet>, "position":"end"}
```
```json
{"action":"calc_set_range",
 "sheet":<dest-sheet>,
 "start_a1":"A1",
 "values":[
   [<CategoryHeader>, <TotalHeader>],
   [<category-name-1>, "=SUM(<src-sheet>.<col-1><first-data>:<col-1><N>)"],
   [<category-name-2>, "=SUM(<src-sheet>.<col-2><first-data>:<col-2><N>)"],
   ...
 ]}
```

`<category-name-i>` is the i-th source-header string from the SP
`hdr:` line; `<col-i>` is the source column letter at the same
position. Build both lists from the SP — never from training memory.

### (b') group-by aggregation — categories in a source COLUMN

Task: "Pivot table summarising <metric> by <category>"
SP signal: the category column has type `S` (string) with repeated
values across many rows.

**Row-vs-column decision — read the instruction BEFORE picking
the JSON**:

| Instruction wording                                  | Use                              |
|------------------------------------------------------|----------------------------------|
| "as the row labels", "for each <X>", "<X> in rows"  | `row_fields:[<X>]`               |
| "as the column headers", "as columns", "<X> as columns" | `column_fields:[<X>]`         |
| no row/column wording (default)                      | `row_fields:[<X>]`               |

Always also list the OTHER three field arrays as `[]` so the
placement is explicit and the model can't silently default.

Worked example (category in rows — the default):

```json
{"action":"calc_new_sheet", "name":<dest-sheet>, "position":"end"}
```
```json
{"action":"calc_pivot_table",
 "source_sheet":<src-sheet>,
 "source_range":"A1:<lastcol><N>",
 "dest_sheet":<dest-sheet>,
 "dest_anchor":"A1",
 "row_fields":[<CategoryHeader>],
 "column_fields":[],
 "filter_fields":[],
 "data_fields":[[<MetricHeader>, "SUM"]],
 "pivot_name":"PivotTable1"}
```

Worked example (category as column headers — instruction says
"<category> as the column headers"):

```json
{"action":"calc_pivot_table",
 "source_sheet":<src-sheet>,
 "source_range":"A1:<lastcol><N>",
 "dest_sheet":<dest-sheet>,
 "dest_anchor":"A1",
 "row_fields":[],
 "column_fields":[<CategoryHeader>],
 "filter_fields":[],
 "data_fields":[[<MetricHeader>, "SUM"]],
 "pivot_name":"PivotTable1"}
```

### (c) VLOOKUP — one table holds KEYs, another holds VALUEs

**When to reach for VLOOKUP** — any of these signals:
* Instruction wording: "lookup table", "look up", "match",
  "according to", "based on the <X> table/sheet", "from the <X>
  sheet", "fill ... according to ...".
* Layout: the source row holds only an IDENTIFIER (string column)
  and the attribute the task asks about lives in a SEPARATE table
  — another sheet, OR a small reference block on the same sheet.

The formula has shape:
`=VLOOKUP(<key-cell>;<lookup-range>;<col-in-lookup>;<match-type>)`

`<lookup-range>` is:
* `'<OtherSheet>'.<key-col>:<value-col>` — cross-sheet (dot `.`).
* `$<key-col>$<start>:$<value-col>$<end>` — same-sheet reference;
  use `$` to anchor.

**Example 1 — pure text lookup (no math)**:

```json
{"action":"calc_fill_column",
 "sheet":<sheet>,
 "column":<target-col>,
 "first_row":<first-data>,
 "last_row":<N>,
 "formula_template":"=VLOOKUP(<key-col>{r};$<lookup-key-col>$<key-start>:$<lookup-val-col>$<key-end>;2;0)"}
```

**Example 2 — VLOOKUP combined with arithmetic**:

```json
{"action":"calc_fill_column",
 "sheet":<sheet>,
 "column":<new-col>,
 "first_row":<first-data>,
 "last_row":<N>,
 "formula_template":"=VLOOKUP(<key-col>{r};'<OtherSheet>'.A:B;2;0)*<q1-col>{r}*(1-<q2-col>{r})",
 "header":<MetricHeader>}
```

**Example 3 — grade scale (approx match)**:
The key column holds range thresholds (0, 30, 60, ...), not exact
labels. Use match type `1`, not `0`.

```json
{"action":"calc_fill_column",
 "sheet":<sheet>,
 "column":<grade-col>,
 "first_row":<first-data>,
 "last_row":<N>,
 "formula_template":"=VLOOKUP(<mark-col>{r};$<scale-key-col>$<scale-start>:$<scale-val-col>$<scale-end>;2;1)"}
```

### (d) text split / extract

**Primary path — delimiter split → `calc_split_text`** (one action, any
number of output columns, no formula writing):

Task shape: "Split <SourceHeader> by <delimiter> into <Out1Header>,
<Out2Header>[, <Out3Header>...] columns".

Count the output column names in the instruction — `out_cols` length
MUST match. "First Name, Last Name AND Rank" ⇒ THREE letters.

```json
{"action":"calc_split_text",
 "sheet":<sheet>, "src_col":<src-letter>,
 "first_row":<first-data>, "last_row":<N>,
 "out_cols":[<out1-letter>, <out2-letter>, <out3-letter>],
 "delimiter":" ",
 "last_takes_rest":true}
```

`last_takes_rest:true` (default) handles multi-word last segments
(e.g. "Senior Vice President" stays whole in the Rank column). Set
`false` only when the task explicitly requires exactly N tokens.

**Fallback — pattern/position extraction → `calc_fill_column`**:

Only when the split is NOT a fixed delimiter — e.g. extract a phone
number by regex, take first 5 characters by position, parse a date
component. Use one `calc_fill_column` per output column with the
needed formula. LibreOffice supports `=REGEX(<src>{r};"<pattern>";"$N")`
which is usually cleaner than nested LEFT / MID / FIND.

```json
{"action":"calc_fill_column",
 "sheet":<sheet>, "column":<out-col>,
 "first_row":<first-data>, "last_row":<N>,
 "formula_template":"=REGEX(<src-col>{r};\"<pattern with capture group>\";\"$1\")",
 "header":<OutHeader>}
```

### (e) structural ops

Direct one-to-one mapping — no formula reasoning needed:

* Sort: `calc_sort_range`
* Freeze: `calc_freeze`
* Format: `calc_format_number`
* Sheet new: `calc_new_sheet`
* Sheet rename: `calc_rename_sheet`
* Sheet copy: `calc_new_sheet` with `source_sheet`
* Range copy WITHIN the spreadsheet (sheet → sheet, e.g. duplicate a
  block to another sheet, transpose-paste): `calc_duplicate_range`.
  This writes cell values to another SHEET — it does **NOT** touch
  the clipboard. NEVER use it to "copy to the clipboard" / for a
  cross-app paste (`dest_sheet` must be a real sheet — there is no
  "clipboard" sheet).
* Copy a range to the CLIPBOARD (Calc → ANOTHER APP):
  `calc_copy_to_clipboard` with `sheet` + `range_ref`. Use this for
  MULTI-APP tasks that move Calc data into another application.
  **`calc_copy_to_clipboard` selects AND copies in ONE action** —
  after it runs the data IS on the clipboard. Do NOT plan a separate
  "copy" / "Ctrl+C" / `calc_duplicate_range` step after it — that is
  the #1 mistake and wastes many steps. The next step is: switch to
  the target app (`open_app`) and paste.
  RANGE: a copied table needs its HEADER ROW. Start the range at
  row 1 (the header) and include ONLY the data row(s) the task
  names — header + one data row at sheet-row N → `A1:<col>N`, not
  `A<N>:<col>N` (drops the header), not a wider span (pulls in
  unwanted rows). Never select the whole used area.
* Single-cell write: `calc_set_cell`
* Hide rows / columns: `calc_hide_range` with `rows=[…]` and/or
  `cols=[…]`. Use whenever the task says "hide" / "do not show" /
  "make invisible" specific rows or columns (NOT "filter" — filter
  is a separate Calc feature). Pair with `row_hidden` / `col_hidden`
  verify ops to confirm the rows/cols ended up hidden after save.
* Insert chart: `calc_insert_chart` — ONE-STEP operation. Set
  `sheet`=destination (DEFAULT: same as `data_sheet` — put the chart
  on the data sheet unless the task explicitly says "in a new sheet
  called X"; specifying a new name auto-creates that sheet), 
  `data_sheet`=source,
  `data_range` (incl. headers, no sheet prefix needed),
  `chart_type` ("column" for vertical bars, "bar" for horizontal,
  "line" / "pie" / "scatter" / "area"), `title`. NEVER chain a
  `calc_new_sheet` or `calc_duplicate_range` before this — the chart
  references the source data directly across sheets. Pair with
  `chart_exists` verify op (filter by `title` + `chart_type`; other
  kwargs are silently ignored).
  Multiple charts on one sheet: omit `name=` from BOTH calls — the
  framework auto-picks non-colliding names (Chart1, Chart2, ...).
  Setting the same `name=` twice (or relying on a single default
  name) overwrites the first chart with the second.

  **Picking the data-feed form** — `calc_insert_chart` accepts ONE
  of two shapes:
    Form (1): `data_range=<contiguous A1 range>` — LO makes ONE
      series per non-header row inside that range. Good for "chart
      the whole table" cases.
    Form (2): `categories=<A1 range>` + `values=[<A1 range>, ...]`
      — explicit axis ranges, ONE entry in `values` per series.
      Good for "chart of a single summary row / single column"
      AND for "x-axis is <colA>, y-axis is <colB>" cases where
      the two columns are NOT adjacent.
  Pick by counting the series the chart should have:
  - "show the totals/results of <Total row>" — ONE series. Form (2)
    with `values=[<the Total row>]` and `categories=<header row>`.
  - "show the growth row" — ONE series. Form (2), same idea.
  - "X-axis is <colA>, Y-axis is <colB>" — ONE series. Form (2)
    with `categories=<colA range>` and `values=[<colB range>]`.
  - "show <data> for each <X>" — N series. Form (1) with the whole
    table, OR Form (2) with one entry per series in `values`.
  Form (1) with an over-wide range (e.g. the whole table when the
  task only wants the summary row) silently produces extra series;
  the chart visually looks right but carries extra / incorrect series
  formulas. When in doubt, prefer Form (2)
  — it forces you to enumerate the series count explicitly.

  **Form (2) emission protocol** — DO NOT default to "categories =
  leftmost label column". That prior is right for one-bar-per-data-
  row charts (Form 1) but WRONG for summary-row / single-column
  / non-adjacent-axis charts. Derive `categories` from the task
  wording + SP, not from a generic chart prior:

   **HARD CONSTRAINT** (verify BEFORE emitting): the cell count of
   `categories` MUST EQUAL the cell count of every `values[i]`.
   Off by one — typically caused by extending `categories` into
   the leftmost-label cell or the header cell — is REJECT;
   recompute the range, don't emit. Count the cells, don't trust
   intuition.

   1. Parse the chart instruction for the x-axis LABEL NAME
      ("x-axis is <X>", "for each <X>", "<X> on the X-axis").
      The label name is whatever noun the instruction binds to
      the x-axis.
   2. Find WHICH SP cells contain those labels. Almost always
      either the `hdr:` line (cells in row 1) or the leftmost
      data column. Cite the exact A1 range.
   3. Find WHICH SP cells hold the y-axis data (the summary row,
      summary column, or specific column the instruction names).
   4. **TRIM** step 2's range to cover EXACTLY the same columns
      (or rows) as step 3's range. If step 3 spans cols X..Y,
      step 2 spans cols X..Y — not col A..Y if A isn't part of
      X..Y, not col X..Z.
   5. Emit `categories=<trimmed step 2>`, `values=[<step 3>]`.
      SAME shape: row values ⇒ row categories; column values ⇒
      column categories.
  If step 1's label name is not the leftmost-column header (e.g.
  months / years / dates rather than the per-row identifier),
  categories almost certainly comes from row 1, not col A —
  explicitly cross-check step 1 against the label cell range you
  would otherwise default to.
* Cell formatting (color / bold): `calc_set_color` — replaces
  Format → Cells → Background / Font GUI. Pass `sheet`, `range_ref`,
  and any of `bgcolor`, `font_color` (both `"#RRGGBB"`), `bold`,
  `italic` (booleans). The op sets only the kwargs you supply.
  Use whenever the task says "highlight" / "color the cell" /
  "make bold" / "white text on blue fill". Pair with `cell_color`
  verify op on a representative cell from the range.
  CONDITIONAL highlighting (the Excel "Conditional Formatting"
  pattern): pass `where_formula` to colour ONLY cells where a
  predicate is true. Use `{cell}` for the per-cell A1 ref and
  `{r}` for the 1-based row. LO Calc formula syntax — `;` between
  multi-arg function args, NOT `,`:
    `WEEKDAY({cell};2)>=6` → Sat / Sun in a date column
    `MOD({r};2)=0` → alternate rows
    `{cell}=MAX($B$2:$B$25)` → max-value cell
  Empty cells are skipped by default (`skip_empty=true`) — without
  this, `WEEKDAY(<empty>;2)` returns `6` (LO treats blank as
  1899-12-30 = Saturday) and trailing blank rows leak red. Pass
  `skip_empty=false` ONLY when the predicate explicitly targets
  blanks (e.g. `ISBLANK({cell})`).
  Required when the task is "highlight cells matching <condition>" —
  do NOT colour the whole range; that's a different (and wrong)
  outcome. The op evaluates the formula via a scratch column far
  from the data and cleans it up; only cells inside `range_ref`
  are touched.
* Merge cells: `calc_merge_cells` — merge a rectangular A1 range
  (`A1:C1`, `A2:B2`, etc) into one anchor cell. Optional `value`
  is written into the anchor BEFORE the merge, so "merge A1:C1
  and write 'Header'" is one step (do NOT chain `calc_set_cell`
  before the merge; the order matters). Pair with `cell_merged`
  verify op on the anchor cell.
* Data validation (dropdown): `calc_data_validation` — attach a
  list-of-allowed-values rule to a range. Required: `sheet`,
  `range_ref`, `allowed_values=[...]`. Optional `type` defaults
  to `"list"` (the dropdown form); `show_dropdown` defaults to
  true (renders the picker arrow). Replaces the Data → Validity
  GUI dance. Pair with `data_validation` verify op (same
  `allowed_values` set, order-insensitive).
* Reorder columns: `calc_reorder_columns` — rearrange the columns
  of a contiguous data block by column letter OR header name.
  Required: `sheet`, `new_order` (list — items can be column
  letters like `"A"` / `"AA"`, OR header strings looked up
  against row 1). Output goes to columns `A..`. Use for "reorder
  columns to <X>, <Y>, <Z>" instructions. Pair with `cell_value`
  verifies on the first / last row of the renamed columns.
* Transpose paste: `calc_duplicate_range` with `transpose=true` — same
  source / dest semantics as a straight copy, but writes source
  `(r,c)` to destination `(c,r)`. Pair with `duplicate_range_match`
  verify op with `transposed=true`.
* Export PDF / CSV / HTML: `calc_export_file` — ONE-STEP operation
  that replaces the GUI menu chain (File → Export → pick filter →
  dropdown → click Save). Required: `path` (absolute, ends in
  `.pdf` / `.csv` / `.html` — `format` is inferred). Optional PDF-only:
  `fit_width_pages` / `fit_height_pages` (apply
  `PageStyle.ScaleToPagesX/Y` before storeToURL — use `1` for
  fit-to-one-page-wide / tall requests). Optional CSV-only:
  `sheet` (defaults to active), `delimiter`, `encoding`,
  `save_formulas` (default `false` — writes evaluated values,
  not formulas). Compose `path` from the instruction: pick the
  directory and the output basename from the wording, then
  append `.pdf` or `.csv`. When the basename should match the
  open file, read it from the SP block's window title
  (`<X>.xlsx - LibreOffice Calc` → use `<X>`, drop the `.xlsx`)
  — do NOT append the new extension after `.xlsx`. The op
  defensively strips a stray source extension if you slip up,
  but it is cleanest to drop it yourself. Pair with
  `file_exists` verify op on the same `path`.

