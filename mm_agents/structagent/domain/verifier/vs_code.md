# VS Code — verifier knowledge

## Choosing verify kind

Decide by what you can name and where the success state actually
lives:

* **Preference / keybinding task AND you can name the exact key
  or value** (the key appears verbatim in the probe's surfaced
  VS Code config, OR the instruction itself gives a uniquely-
  identifying string like a folder name, command name, keybinding
  label) → `file_grep` on `settings.json` / `keybindings.json` /
  `.code-workspace` with that key or value as the pattern.
  file_grep is the strongest available verify here.

* **Preference / keybinding task AND you CANNOT name the key**
  (the key would be a guess from training memory — no probe hit,
  no documented mapping above) → `a11y_match` on the Settings UI
  control's post-commit state:
  - checkbox → `tag=checkbox`, `text=[<control label>]`, `state=[checked|unchecked]`
  - dropdown / enum → `tag=combobox`, `text=[<chosen value>]`
  - text / number → `tag=entry`, `text=[<typed value>]`

* **Open-file / open-folder / open-workspace task** — the success
  state is the IDE's loaded state, not a config file. VS Code's
  workspaceStorage paths are hash-named and effectively opaque, so
  guessing them is unreliable. Use `a11y_match` on the persistent UI
  evidence:
  - opened folder → `tag=tree`, `text=[<folder name>]` in the
                    Explorer sidebar root
  - opened file   → `tag=tab`,  `text=[<filename>]` in the editor
                    tab bar
  - opened workspace → `tag=window`, `text=[<workspace name>]` in
                       the title bar

Guessed-key file_grep is the failure mode to avoid: a pattern like
`"<plausible.key>"\s*:\s*true` matches only when the actor literally
echoes that exact string into the file, which is NOT how the GUI
writes — verify passes only on a wrong-content edit, while the real
key is never set.

### Init_ledger verify — DO NOT use mere file-existence

For "complete the code & save output to log.txt" the verify SHOULD
include BOTH of:

* a **code-correctness** guard on the source — at minimum:
  `kind: shell_command`, `command: ["bash","-c","! grep -E 'TODO|FIXME|<placeholder>' /path/to/file.py"]`,
  `expected_exit_code: 0`. This catches the "agent ran the file
  without completing it" failure mode (DONE accepted, log.txt exists,
  but its content is from the unfixed buggy program).
* a **content** guard on the output: `file_grep` for a value the
  instruction names verbatim (e.g. instruction quotes an expected
  number), OR at the very least `kind: shell_command`, `command:
  ["test","-s","/path/to/log.txt"]` (non-empty file). Never just
  `test -f <log>` — file existence is the WEAKEST possible verify
  and is what allowed buggy-output runs to score DONE.

These are general code-task verify rules — they apply to any "edit
source then run" task across vscode / os / data-science domains, NOT
just one specific script.
