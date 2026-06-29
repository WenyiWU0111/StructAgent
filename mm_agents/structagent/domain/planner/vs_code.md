# VS Code domain knowledge

This block is injected into the planner and init_ledger prompts ONLY when
the task's domain prefix is `[vscode]` or `[vs_code]`. It does NOT apply
to other domains (chrome, libreoffice, etc.).

The OSWorld VM baseline (Linux user, sudo password, `ctrl+alt+t`
terminal shortcut) and the actual environment snapshot (real config
file paths, available binaries, existing workspace files) are provided
elsewhere — do NOT duplicate them here. This file documents only the
VS Code conventions a reader can find in the public VS Code docs.

## Config files

* User settings live in JSONC (JSON with `//` line comments and
  `/* */` block comments). On a fresh install the file may contain
  only a comment stub; tooling must strip comments before `json.loads`.
* Keybindings live in JSONC as an array of binding entry objects.
* Multi-folder workspaces live in a `*.code-workspace` file — strict
  JSON, NOT JSONC.

## Workspace files (`*.code-workspace`)

A workspace's shape is an object with a `folders` array (and optionally
`settings`):

```json
{
  "folders": [
    {"path": "<dir-A>"},
    {"path": "<dir-B>"}
  ],
  "settings": {}
}
```

Each folder entry's `path` is RELATIVE to the workspace file's
directory unless it starts with `/`. To "add a folder X to a workspace"
the standard VS Code idiom is to APPEND `{"path": "X"}` to the
`folders` array — this is exactly what the GUI's "Add Folder to
Workspace" command does.

When the task names a folder by absolute path but that folder is a
sibling of the workspace file (same parent directory), the canonical
form stored in the workspace file is the BARE folder name (e.g.
`{"path": "<name>"}`), NOT the absolute path. The GUI's "Add Folder
to Workspace" writes the relative form. file_grep verify patterns
for these entries must match the bare-name form — patterns matching
the absolute path will both miss the GUI's output and push the actor
to write a form VS Code itself does not use.

## Settings.json key form (primitive vs object)

A SETTING whose value is a **primitive** (string, number, bool) — write
as a flat dot-key:

```json
{"editor.fontSize": 14, "files.autoSave": "afterDelay", "debug.X": false}
```

A SETTING whose value is itself an **object/dict** (a "rules map" or
"overrides" group) — MUST be written as a nested object:

```json
{
  "<parent.path>": {
    "<child_key>": "<value>",
    ...
  }
}
```

Writing the over-flat form `{"<parent.path>.<child_key>": "<value>"}`
is NOT equivalent — VS Code's settings schema requires nested objects
for object-valued keys.


## JSON config edits — DO NOT GUI-type into a JSON literal

VS Code task instructions frequently target `settings.json`,
`keybindings.json`, workspace `*.code-workspace` files. Each is a
JSON literal where partial-content overwrite (a misplaced `Ctrl+A`
+ type) corrupts the buffer.

Do NOT plan a GUI keystroke sequence for the actual content change.
The LAST step of the plan (the one that applies the change) must be
written as a HIGH-LEVEL FILE-EDIT INTENT, and the actor's decomposer
will emit a deterministic JSON edit.

  GOOD `<text>`: "Modify ~/.config/Code/User/settings.json to set <key> to <value>." (intent only)
  GOOD `<text>`: "Append a keybinding entry to ~/.config/Code/User/keybindings.json with key=<K>, command=<C>, when=<W>." (one entry, fields named explicitly)
  BAD  `<text>`: "Press Ctrl+A, type the JSON entry, then Ctrl+S." (the ctrl+a+type pattern destroys the buffer)
  BAD  `<text>`: "Click into the editor and type the new key-value at the end." (free-form typing into a JSON literal — error-prone)

The earlier GUI steps (open the file in the editor so the user can
see it) stay as-is — only the actual content-change step uses the
file-edit intent style.

## Don't edit settings via GUI ctrl+a + type

The Settings UI's text editor accepts ctrl+a + type, but typing a JSON
fragment like `"X": "Y"` into a buffer that ctrl+a just selected
overwrites the WHOLE file — you'll usually end up with an invalid JSON
fragment as the file body. Use the structured `edit_json` action
instead; it does load → mutate → dump and never produces invalid JSON.

## Multi-line source edits (indent / dedent / comment ranges)

For a task that modifies a contiguous line range in a source file —
"indent lines A to B by one tab", "comment out lines A to B", "add a
prefix to each of these lines" — prefer a single `cli_run` with `sed`
over GUI navigation:

* indent lines A..B by one tab: `sed -i '<A>,<B> s/^/\t/' <file>`
* dedent one tab: `sed -i '<A>,<B> s/^\t//' <file>`
* comment out (Python): `sed -i '<A>,<B> s/^/# /' <file>`

Manual GUI shift+down + tab is fragile: cursor placement, focus loss,
multi-cursor state and editor scroll all conspire against it, and a
single misclick replaces the selection.

file_grep verify for this kind of task must check EACH modified line
individually (one pattern per line number's expected content), not a
single regex like `^\t` that any single tab-indented line satisfies —
otherwise the verify passes after editing only one line out of N.

## Instructions with "complete / fill in / fix / implement"

When the user's wording requires CHANGING the source file before
running it ("complete the code", "fill in the function", "fix the
bug", "implement the missing helper", "resolve the TODO"), the
operation is a TWO-step pipeline — modify, then run — NOT a one-
step "just run it":

1. READ the target file first (`cat <path>` is one cli_run line). Don't
   guess what's incomplete. Common markers the instruction's wording
   maps to: literal `TODO`, `FIXME`, `pass  # ...`, `# implement here`,
   an empty function body, an obvious placeholder constant.
2. EDIT to resolve every such marker (a single `sed` / `cli_run` heredoc
   that overwrites the relevant line range is cleanest — see the
   "multi-line source edits" section above). Don't merge multiple TODOs
   into a single hand-typed rewrite of the whole file from VS Code's
   keyboard — that's where typos and indent-loss kill the run.
3. ONLY THEN run the file and capture output (`python3 <path> > <log>`).

