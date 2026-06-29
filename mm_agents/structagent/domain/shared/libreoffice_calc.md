# LibreOffice Calc — shared formula + format-string conventions

This block is read by BOTH the planner (so the formula / format
string it emits in `calc_set_cell` / `calc_format_number` is the
canonical one) AND the init_ledger verifier-author (so the
`expected_formula` / `format_string` in the verify spec is the same
byte-for-byte string). Any drift between the two breaks `verify_cell_format`
and `verify_cell_value` — that's why this content is shared rather
than duplicated.

## Formula syntax

Every formula MUST start with `=` and use `;` between arguments
(VLOOKUP / IF / SUMIF / COUNTIF / MATCH / INDEX / DATEDIF / CONCAT /
LEFT / MID / FIND ...): `=VLOOKUP(E2;$A$2:$B$7;2;0)`. Single-arg
functions (`=SUM(A2:A11)`) have no separator and are unaffected.

Cross-sheet refs use `.` not `!`:
`=VLOOKUP(A2; Sheet1.A1:B10; 2; FALSE)`. Wrap sheet names containing
spaces in single quotes: `'Retail Price'.A:B`.

Date cells store a NUMERIC serial; `=TODAY()`, `=DATEDIF(...)`,
`=B2+30` all work on the serial.

## Number-format syntax (`calc_format_number` `format_string`)

LO uses Excel-style format codes. Patterns the benchmark commonly
asks for:

* `0.00` — fixed 2 decimals. (`0` = digit always shown; `#` =
  digit shown only if non-zero.)
* `0.0` — fixed 1 decimal.
* `[$-<LCID>]<format>` — apply the format under a different LOCALE.
  The `<LCID>` is the Microsoft locale ID in hex. Use this when the
  task asks for a different decimal/thousand separator (most users
  outside en-US use comma as decimal). Useful IDs:
    - `[$-419]` → Russian (`,` decimal, ` ` thousands)
    - `[$-407]` → German   (`,` decimal, `.` thousands)
    - `[$-40C]` → French   (`,` decimal, ` ` thousands)
    - `[$-410]` → Italian  (`,` decimal, `.` thousands)
  Example: format `[$-419]0.0` makes `0.1` render as `0,1` even when
  the workbook locale is en-US. Verify via the rendered CSV, not
  via `getString()` on the live LO instance (the live value can
  echo the source locale; the EXPORT honours the format-string
  locale).
* `0.0,` — magnitude scaling. Each trailing `,` divides the value
  by 1 000 BEFORE display. `0.0,,` = millions, `0.0,,,` = billions.
* `0.0,," M"` — magnitude scale + literal unit suffix. ONE space,
  INSIDE the quoted literal. Outer-space (`,, "M"`) or both-sides
  (`,, " M"`) render the wrong number of spaces, not the requested format.
  In an init_ledger JSON verify spec, escape the inner `"` as `\"`:
    raw  (in a Python action) : `0.0,," M"`
    JSON (in a verify spec)   : `"0.0,,\" M\""`
* `[RED]0.0;[BLUE]-0.0` — positive;negative split. Less common in
  this benchmark but the same `;` syntax lets you carve out custom
  rendering per sign.

LO parses these as Excel codes — no `\` escaping for the comma or
the quoted-literal text itself. The ONE escape you still owe is the
JSON-string escape on inner `"` when the format goes into a JSON
verify spec: a raw `0.0,," M"` becomes the JSON value `"0.0,,\" M\""`.
Common task
mappings — emit these EXACT strings byte-for-byte, both in the
planner's `calc_format_number` action AND in init_ledger's
`cell_format` verify spec (the two MUST agree, otherwise the
strict format-key compare in `verify_cell_format` fails):

  - "Set the decimal separator to comma" → format
    `[$-419]0.<n>` (or any comma-decimal locale) on the affected
    range; pick `<n>` to match the source's decimal precision.
  - "Show values in millions, rounded to one decimal, with a space
    before M" → format `0.0,," M"`     (space inside the quotes only)
  - "Show values in billions, rounded to one decimal, with a space
    before B" → format `0.0,,," B"`    (THREE commas + " B")

WHEN THE TASK PUTS THE FORMATTED VIEW IN A NEW COLUMN — e.g.
"display column A's values in column B with format X, and in
column C with format Y" — the new columns must FIRST contain the
underlying numeric values (formula `=A{r}` or a copy of the
values) before `calc_format_number` is applied. Otherwise the
format paints empty cells and nothing renders. The plan needs
**TWO actions per output column**:

  - `calc_fill_column(sheet, column=B, first_row=<r1>,
     last_row=<rN>, formula_template="=A{r}",
     header=<existing B header>)` — populate B with values
  - `calc_format_number(sheet, range_ref=B<r1>:B<rN>,
     format_string=<the magnitude/unit format>)` — apply display

Repeat for C with the other format. Keep the source column A
untouched.

UNIT AWARENESS — read column headers before composing a formula.
When a SOURCE column's header already ends in `(%)` / `(percent)` /
`(rate)`, its values are ALREADY in percent units (e.g. `11.25` =
11.25%). Do NOT multiply by 100 again when deriving another rate
from that column — the header on the output ("Period Rate (%)" /
"Growth (%)") describes the same unit, not a conversion. The
canonical "period rate" formula is `=A{r}/B{r}` (annual % divided
by number of periods), NOT `=A{r}/B{r}*100` — the latter inflates
every result by a factor of 100 — the wrong stored value (it should
stay the raw fraction). Same rule for "growth %" / "change %" / "share
of total %" formulas derived from already-% inputs.

