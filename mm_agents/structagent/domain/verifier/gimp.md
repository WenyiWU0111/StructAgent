# GIMP domain knowledge

## Verify kind for preference toggles — `a11y_match` is the DEFAULT

For tasks that toggle a GIMP preference (anything reachable through
`Edit → Preferences`), the DEFAULT verify kind is `a11y_match` on the
GUI control's post-commit state, NOT `file_grep` on `gimprc`.

Use `file_grep` on `gimprc` ONLY when the exact target S-expression
key appears verbatim in the probe's surfaced gimprc content. If the
key is not visible in the probe, do not write file_grep with a
guessed key — use `a11y_match`.

Map preference controls to `a11y_match` shapes:

* checkbox setting → `tag=checkbox`, `text=[<control label>]`,
                     `state=[checked]` (or `unchecked`)
* dropdown / mode  → `tag=combobox`, `text=[<chosen value>]`
* slider / spin    → `tag=spinbutton`, `text=[<value>]`
* color-picker     → `tag=button`, `text=[<hex or label>]`

GIMP flushes the real entry to `gimprc` when the Preferences dialog is
OK'd, so the on-disk config reflects the change once confirmed — your
verify can target the persisted result without naming the exact key.

## Tasks that produce an image file

For tasks whose output is a `.png` / `.jpg` / `.xcf` / etc. on disk
whose correctness is a perceptual image property (structural
similarity, brightness, saturation, contrast, palette):

* The strongest local verify is "file exists and is non-empty":
  `shell_command ["test", "-s", "<absolute-path>"]`.
* The perceptual judgment cannot be replicated by a local verify spec
  — verify the file exists and is non-empty, not its pixel content. Do
  NOT add `expected_substring` patterns that pretend to.
* The actor MUST commit each edit via the dialog's OK / Apply button
  (not Cancel, not close-via-X) AND save via `File → Export As`. A
  dialog left open or pending at DONE means the edit didn't commit
  and the saved file is unchanged — verify ONLY after both happen.
