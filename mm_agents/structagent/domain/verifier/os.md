# OS / Ubuntu domain knowledge

## General CLI hygiene for verify specs

* Paths in argv: do NOT rely on tilde expansion. The framework now
  expands a leading `~/` automatically, but writing `/home/user/<…>`
  literally is clearer and works on all dispatch paths.
* For file-existence checks where the content matters (output of a
  computation), `test -f <path>` is the WEAKEST verify — prefer
  `test -s <path>` (non-empty) or a `file_grep` on a known value.
* `gsettings list-keys <schema>` enumerates available keys; if a
  schema appears in `[Relevant GSettings schemas]` but the key you
  want isn't listed under it, the key does NOT exist in this
  GNOME version. Choose a different mechanism.
