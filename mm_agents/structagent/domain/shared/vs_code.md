# VS Code — shared config file conventions

Both the planner (when emitting a keybinding / settings edit) and the init_ledger verifier-author (when writing a file_grep against these files) must agree on the exact shape of entries — `key`, `command`, `when`, the negative-override `-` prefix, the `**/` glob form for `files.exclude`. Factored here to avoid drift.

## Keybindings.json — "remove" / "disable" idiom

The standard VS Code way to "remove" a default key binding is NOT to
delete an entry from any file. It is to APPEND a new entry whose
`command` field starts with a leading `-` ("negative override"). The
`when` clause must match the original binding's `when`.

Example — to "remove" the default `ctrl+f → list.find` binding (which
fires when the Explorer tree has focus), append:

```json
{
  "key": "ctrl+f",
  "command": "-list.find",
  "when": "listFocus && listSupportsFind"
}
```

## Adding a new keybinding

To bind `<key>` to `<command>` (with optional `<when>`), append a normal
(non-negated) entry to keybindings.json. Don't use `-` prefix.

## Keybinding `when` clauses

When a task says a shortcut should fire FROM / IN / WHILE focused on a
specific UI region (terminal, editor, debug console, sidebar, panel),
the keybinding entry should include a `when` field naming that region's
context key. Common values: `terminalFocus`, `editorTextFocus`,
`debugConsoleFocus`, `sideBarFocus`, `panelFocus`.

When the task requires a `when` clause, the file_grep verify patterns
MUST include `when` as a third pattern alongside key and command —
otherwise a no-when entry can satisfy a verify the task actually
needs a specific when for.

## `files.exclude` glob patterns — `**/` prefix

VS Code's `files.exclude` and `search.exclude` glob keys use `**/<name>`
to match nested occurrences. The plain `<name>` form only matches
top-level entries.