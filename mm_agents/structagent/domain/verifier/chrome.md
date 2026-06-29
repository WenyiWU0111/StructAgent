# Chrome domain knowledge

## Verify kind — `a11y_match`, `url_match`, and (CRITICAL) `shell_command` for saved files

* For tasks that set a browser/page preference, the default verify is
  `a11y_match` on the control's post-commit state — map checkbox →
  `tag=checkbox state=[checked]`, dropdown → `tag=combobox
  text=[<chosen value>]`, text field → `tag=entry text=[<typed value>]`.
* For tasks whose end-state is a navigation (a specific page open, a
  query issued), prefer `url_match` on the resulting URL substring —
  it is deterministic, unlike the vision-LLM-judged `a11y_match`.
* For tasks whose end-state is **a file on disk** ("save the page as
  PDF to ~/Desktop", "download the attachment", "export bookmarks to
  ~/bookmarks.html"): the verify MUST be `shell_command` against the
  filesystem (`test -f <path>`, `ls <path>`, `find ~/Desktop -name
  '*.pdf' -newer ...`, `file <path>` to check MIME). NEVER mark such
  an outcome satisfied based on GUI cues alone — "Save dialog closed"
  / "Replace dialog dismissed" / "page is visible again" are the
  classic false-positives. A save dialog can close without writing the
  file (cancel, disk full, name collision auto-renamed), or write a
  *different* file than the task asked for (Chrome auto-renames on
  collision: `webpage.pdf` → `webpage (1).pdf`). The deterministic
  shell check is the only ground truth.
