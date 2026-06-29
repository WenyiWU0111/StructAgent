# Thunderbird domain knowledge

This block is injected into the planner and init_ledger prompts ONLY when
the task's domain prefix is `[thunderbird]`. It does NOT apply to other
domains.

## Default execution route: GUI + a11y

For Thunderbird tasks, prefer GUI navigation driven by the a11y tree.
The a11y tree exposes Thunderbird controls with stable tag / text /
state triples (`checkbox`, `combobox`, `entry`, `menu-item`, `tab`,
`tree-item`); a11y matches are robust across the small visual
differences between Thunderbird versions, and pixel coords are not.

Most Thunderbird settings live behind one of these GUI entry points,
reachable from the main window:

* `Edit → Account Settings` — per-identity options (display name,
  email, signature, reply-to, default quoting / composition
  behavior), per-server options (sync, filtering, biff), and the
  Message Filters editor.
* `Edit → Preferences` — global / cross-account preferences
  (composition, viewing, network, advanced).
* `View → Folders` / per-message context menus — folder pane mode
  (all / unread / favorites / unified), per-message and per-folder
  actions.

Preference values commit when the enclosing dialog is dismissed,
i.e. one GUI sequence per setting.

