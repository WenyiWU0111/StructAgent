# Thunderbird domain knowledge

## Verify kind — `a11y_match` is the DEFAULT

For tasks that toggle or set a Thunderbird preference, the DEFAULT
verify kind is `a11y_match` on the GUI control's post-commit state.

Map preference controls to `a11y_match` shapes:

* checkbox setting   → `tag=checkbox`, `text=[<control label>]`,
                       `state=[checked]` (or `unchecked`)
* dropdown setting   → `tag=combobox`, `text=[<chosen value>]`
* text-field         → `tag=entry`,    `text=[<typed value>]`
* selected list item → `tag=tree-item` or `list-item`,
                       `text=[<item label>]`, `state=[selected]`

## In-app `about:` tabs

Several Thunderbird pages open as TABS in the main window (not as
separate dialogs):

* `about:profiles` — Profile Manager inside the running app
  (different from the profile-chooser dialog on launch).
* `about:config` — raw preference editor.
* `about:preferences` — settings UI.

When a task asks to "open the X page / tabpage in Thunderbird", the
target is one of these `about:` URLs. The visible tab label is the
literal `about:<name>` string — verify with
`a11y_match tag=tab text_contains=["about:<name>"]`, not a guessed
friendly label.
