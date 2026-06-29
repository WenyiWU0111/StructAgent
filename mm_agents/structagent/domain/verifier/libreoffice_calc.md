# LibreOffice Calc domain knowledge

## Verify guidance — for init_ledger

Verify spec uses `kind="calc_verify"` with a flat `calc_checks` list.
The framework builds the Python verify body deterministically — you
never write Python source, never escape quotes, never plumb PASS
tokens. Full schema + op list lives in the "CALC_VERIFY" block in
the init_ledger user message; the shape is:

```json
"verify": {
  "kind": "calc_verify",
  "calc_checks": [
    {"op": "<op>", ...kwargs},
    ...
  ]
}
```

Ops available:
  sheet_exists, cell_value, pivot_exists, cell_format, sort_order,
  freeze_pane, duplicate_range_match.

For "calculate <X> per row" / "fill column X" verifies, emit TWO
`cell_value` checks — first AND last expected data cell — with
`expected_formula_contains` naming the function the formula must
use. For group-by / pivot, chain `sheet_exists` (if a new sheet)
+ `pivot_exists`. Do NOT use `kind="shell_command"` with python3
-c for Calc tasks — that path is deprecated.

For `calc_split_text` outcomes, the cells get **plain text** (not
formulas) — use `expected_substring=<token visible in the SP block
for that row>` on a sample first/last cell, NOT
`expected_formula_contains`. Example: source row 3 shows
`'Blake Dreary CEO'`; verify `B3` with `expected_substring="Blake"`,
`C3` with `expected_substring="Dreary"`, `D3` with
`expected_substring="CEO"`.

NEVER write a verify that PASSES on an UNCHANGED workbook. e.g.
`{"op":"cell_value","a1_ref":"D2","expected_value":""}` on a fill
target — D is already empty at task start, so the outcome flips to
verified before the actor acts and the planner DONEs while the task is still unsolved.
If the instruction names a fill target, the outcome's verify MUST
check that content IS PRESENT (`expected_formula_contains`,
`expected_value` non-empty), never that it stays empty/absent.
If unsure what content to expect, DROP the outcome rather than
write a negative one.

When a task writes to MULTIPLE INDEPENDENT LOCATIONS (multiple
columns, multiple sheets, several non-adjacent ranges, both row and
column totals, etc.), the verify MUST sample at least one cell PER
LOCATION. A verify that only covers one location flips to
satisfied after the actor finishes half the work — planner emits
DONE, the other locations stay unfilled, and the task fails.

## Container + data → TWO outcomes

When the instruction is "Create / Add a <X> to <show | summarize |
display> <Y>", the real end-state is the DATA (Y), not the container (X) — verify Y's content, not just that X exists.
Emit two outcomes: `<x>_created` (shell exists) + `<y>_filled`
(data rows non-empty).

## Auto-activate after mutation

After any `calc_*` action that creates a new sheet or expands an
existing sheet's used range, the framework switches the view to
that sheet before the next screenshot. The post-action screenshot
reflects the change automatically — no extra GUI step needed.
