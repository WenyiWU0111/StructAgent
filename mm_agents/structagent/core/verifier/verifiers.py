"""Deterministic verifiers consumed by the KeyNode dispatcher.

Each ``run_*`` returns one of:
    True   — outcome SATISFIED. Trust deterministically; no LLM call needed.
    False  — outcome NOT-SATISFIED. The control / file / URL was observable
             AND the spec did not match. Trust this verdict (don't fall
             back to LLM), so the planner correctly stays in pending.
    None   — UNCERTAIN. The spec couldn't be evaluated (path unresolved,
             file unreadable, URL not extractable, a11y row not visible
             in current frame). Caller should fall back to the legacy
             LLM KeyNode path so we don't false-negative on transient
             observation gaps.

The dispatcher (key_node_detector._build_messages caller) treats:
    True  → emit outcome_satisfied  (skip LLM)
    False → emit nothing  (outcome stays pending; planner will replan)
    None  → run the original LLM KeyNode call for this outcome
"""
from __future__ import annotations
import hashlib
import logging
import os
import re
import shlex
from typing import Any, Dict, List, Optional, Tuple

from mm_agents.structagent.ledger.core.ledger import (
    VerifySpec, VerifySpecSlot, substitute_slots,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Slot resolution (substitute ${name} placeholders before deterministic verify)
# --------------------------------------------------------------------------- #

# Which VerifySpec fields are themselves regex (so substituted values must
# be ``re.escape``'d) vs plain text (substituted verbatim). Fields not in
# either set are not slot-substituted.
_SPEC_REGEX_FIELDS = ("url_pattern",)                       # str fields
_SPEC_REGEX_LIST_FIELDS = ("patterns",)                     # List[str] fields
_SPEC_TEXT_FIELDS = ("file_path", "tag")                    # str
_SPEC_TEXT_LIST_FIELDS = (
    "text_contains", "state_contains",
    "expected_substring", "forbidden_substring",
    "command",
)                                                            # List[str]


# Bracket-style placeholders the init LLM sometimes emits when it
# doesn't actually know a verify value (a sibling failure mode to the
# slot system). These get literally grepped by the verifier and never
# match, so the outcome stays pending forever. Detected at spec
# load / verify time and treated as "spec authoring failure → fall back
# to LLM judge".
#
# Patterns (kept conservative to avoid eating legit content):
#   [Capital Words inside square brackets]       (≥1 word, ≥2 letters)
#   <camelCase or snake_case in angle brackets>  (excluding HTML-ish "</...>")
#   {{ template-style }} double-curly placeholders
#   [TODO] / [FIXME] / [Placeholder] / etc. — common author markers
_PLACEHOLDER_RES = [
    re.compile(r"\[[A-Z][\w\- ]{1,80}\]"),        # [Planned remarks for Slide 1]
    re.compile(r"<[a-zA-Z_][\w\- ]{1,40}>"),       # <value>, <slide_text>
    re.compile(r"\{\{[\w\- ]{1,40}\}\}"),          # {{TODO}}, {{slot_name}}
    re.compile(r"\b\[(?:TODO|FIXME|PLACEHOLDER|YOUR_TEXT|HERE|EXAMPLE|"
               r"VALUE|TEXT|CONTENT)\b[^\]]*\]", re.IGNORECASE),
]


def _has_bracket_placeholder(s: Optional[str]) -> bool:
    """True if `s` contains a bracket-style placeholder the init LLM
    forgot to instantiate (vs a proper ``${slot}`` slot reference,
    which is the supported way to express 'I don't know this yet')."""
    if not s or not isinstance(s, str):
        return False
    return any(p.search(s) for p in _PLACEHOLDER_RES)


def _scan_placeholders_anywhere(spec: VerifySpec) -> List[Tuple[str, str]]:
    """Walk every string-valued field of a VerifySpec (including nested
    *_checks list-of-dicts) and report any bracket-placeholder strings
    found. Returns a list of ``(field_path, value)`` tuples for log /
    trace; empty list means the spec is clean.
    """
    findings: List[Tuple[str, str]] = []

    def _walk_str(path: str, v: str):
        if _has_bracket_placeholder(v):
            findings.append((path, v[:120]))

    def _walk(path: str, v: Any):
        if isinstance(v, str):
            _walk_str(path, v)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                _walk(f"{path}[{i}]", x)
        elif isinstance(v, dict):
            for k, x in v.items():
                _walk(f"{path}.{k}", x)

    for fname in ("url_pattern", "file_path", "tag"):
        _walk(fname, getattr(spec, fname, None))
    for fname in ("patterns", "text_contains", "state_contains",
                  "expected_substring", "forbidden_substring", "command"):
        _walk(fname, getattr(spec, fname, None))
    for fname in ("calc_checks", "impress_checks", "writer_checks"):
        _walk(fname, getattr(spec, fname, None))
    return findings


def _resolve_spec_slots(
    spec: VerifySpec, slots: Optional[Dict[str, VerifySpecSlot]]
) -> Tuple[VerifySpec, List[str]]:
    """Return a (possibly new) VerifySpec with all bound ``${name}``
    references replaced by their values, plus the list of unresolved
    slot names found anywhere in the spec.

    Regex-bearing fields use ``re.escape`` on substituted values so a
    bound URL like ``https://aws.amazon.com/agreement?id=1`` doesn't get
    re-interpreted as regex metacharacters.

    When ``slots`` is empty or None: scans for ``${...}`` refs (in case
    init emitted placeholders without registering the slot — that's a
    spec-author bug and we surface it as ``unresolved``) but performs
    no substitution. Returns spec unchanged in that case.

    Recurses into ``calc_checks`` / ``impress_checks`` / ``writer_checks``
    (each a list of nested-dict ops). All string values inside those
    dicts get slot-substituted; non-string values pass through. This
    lets the init LLM write ``${slide_1_notes_content}`` inside e.g.
    ``{"op": "slide_notes", "slide": 1, "expected_substring":
    "${slide_1_notes_content}"}`` and have it resolved at verify time.
    """
    if not spec:
        return spec, []
    slots = slots or {}
    unresolved_all: List[str] = []
    changed: Dict[str, Any] = {}

    def _take(s: Optional[str], regex: bool) -> Optional[str]:
        new, ur = substitute_slots(s, slots, regex_escape=regex)
        for n in ur:
            if n not in unresolved_all:
                unresolved_all.append(n)
        return new

    def _take_list(xs: Optional[List[str]], regex: bool) -> Optional[List[str]]:
        if xs is None:
            return None
        return [_take(x, regex) for x in xs]

    def _take_check_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute ``${slot}`` refs in every string value of a
        check-op dict. Recurses into nested lists/dicts for forms like
        ``{"row_fields": ["${col_name}"]}``. Numbers / bools / None
        pass through unchanged."""
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, str):
                out[k] = _take(v, regex=False)
            elif isinstance(v, list):
                out[k] = [
                    _take(x, regex=False) if isinstance(x, str)
                    else _take_check_dict(x) if isinstance(x, dict)
                    else x
                    for x in v
                ]
            elif isinstance(v, dict):
                out[k] = _take_check_dict(v)
            else:
                out[k] = v
        return out

    for fname in _SPEC_REGEX_FIELDS:
        old = getattr(spec, fname, None)
        new = _take(old, regex=True)
        if new != old:
            changed[fname] = new
    for fname in _SPEC_REGEX_LIST_FIELDS:
        old = getattr(spec, fname, None)
        new = _take_list(old, regex=True)
        if new != old:
            changed[fname] = new
    for fname in _SPEC_TEXT_FIELDS:
        old = getattr(spec, fname, None)
        new = _take(old, regex=False)
        if new != old:
            changed[fname] = new
    for fname in _SPEC_TEXT_LIST_FIELDS:
        old = getattr(spec, fname, None)
        new = _take_list(old, regex=False)
        if new != old:
            changed[fname] = new
    # Recurse into the structured-check lists. Each item is a dict
    # authored by the init LLM (op + kwargs); strings inside may carry
    # ${slot} refs the same way top-level fields do.
    for fname in ("calc_checks", "impress_checks", "writer_checks"):
        old = getattr(spec, fname, None)
        if old is None:
            continue
        new = [
            _take_check_dict(item) if isinstance(item, dict) else item
            for item in old
        ]
        if new != old:
            changed[fname] = new

    if not changed:
        return spec, unresolved_all
    from dataclasses import replace
    return replace(spec, **changed), unresolved_all


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #

# When init_ledger writes a wrong path (e.g. typo in /home/<user>/), try
# the app's canonical paths instead. Keys are detected from the literal
# path the spec gave OR from the [bracket-prefix] of the instruction.
APP_PATH_FALLBACKS = {
    "vs_code": [
        "~/.config/Code/User/settings.json",
        "~/.config/Code/User/keybindings.json",
    ],
    "google-chrome": [
        "~/.config/google-chrome/Default/Preferences",
        "~/.config/google-chrome/Default/Bookmarks",
        "~/.config/google-chrome/Local State",
    ],
    "thunderbird": [
        "~/.thunderbird/*.default/prefs.js",
        "~/.thunderbird/*.default-default/prefs.js",
        "~/.thunderbird/*.default-release/prefs.js",
    ],
    "libreoffice": [
        "~/.config/libreoffice/4/user/registrymodifications.xcu",
    ],
}


def _detect_app(spec_path: str, instruction: str) -> Optional[str]:
    """Heuristic: which app does this verify belong to?"""
    p = (spec_path or "").lower()
    i = (instruction or "").lower()
    if "/code/" in p or "/.config/code" in p or "[vscode]" in i or "vs code" in i or "vscode" in i:
        return "vs_code"
    if "/google-chrome/" in p or "[chrome]" in i:
        return "google-chrome"
    if "/.thunderbird/" in p or "[thunderbird]" in i:
        return "thunderbird"
    if "/libreoffice/" in p or "libreoffice" in i:
        return "libreoffice"
    return None


def _glob_first(env: Any, pattern: str) -> Optional[str]:
    """Use VM-side bash to expand a glob; return first match or None.

    The pattern is forwarded to /bin/bash -lc which performs both ``~``
    expansion and globbing on the VM. We must NOT expanduser on the
    host — host ``$HOME`` is the developer machine, but the VM user is
    always ``user`` (OSWorld convention). Host-side expansion would
    yield e.g. ``/home/user/.config/...`` which doesn't exist on the VM.
    """
    if not env or not getattr(env, "controller", None):
        return None
    if not hasattr(env.controller, "run_bash_script"):
        return None
    try:
        result = env.controller.run_bash_script(
            f"ls -1 {pattern} 2>/dev/null | head -n 1",
            timeout=5,
        )
    except Exception:
        return None
    if not result or result.get("status") != "success":
        return None
    out = (result.get("output") or "").strip()
    return out if out else None


def _last_resort_find(env: Any, basename: str) -> Optional[str]:
    """Final fallback: ``find /home/user -name <basename>`` on the VM.

    Used when the spec_path and the curated APP_PATH_FALLBACKS all miss.
    Catches cases where init_ledger wrote a near-correct path (e.g.
    a typo or a profile-randomized subdir not in our fallback list)
    and the file actually lives somewhere else under /home/user.

    We restrict to /home/user (not /) to keep the find cheap and avoid
    accidentally matching files under /usr/share/code or
    /opt/google/chrome that happen to share the basename.
    """
    if not env or not getattr(env, "controller", None):
        return None
    if not hasattr(env.controller, "run_bash_script"):
        return None
    if not basename or "/" in basename:
        return None
    try:
        result = env.controller.run_bash_script(
            f"find /home/user -maxdepth 8 -name {basename!r} -type f "
            f"2>/dev/null | head -n 1",
            timeout=10,
        )
    except Exception:
        return None
    if not result or result.get("status") != "success":
        return None
    out = (result.get("output") or "").strip()
    return out if out else None


def resolve_file_path(
    spec_path: str, env: Any, instruction: str = "",
    trace_attempts: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Find the actual file on the VM. Returns absolute path or None.

    Path strings are passed through to the VM ``~`` and untouched —
    the VM's HTTP server expands ``~`` (its $HOME = /home/user). NEVER
    call ``os.path.expanduser`` on the host: host $HOME points to the
    developer's home, not the VM's, so host expansion produces
    /home/<dev>/... which is invalid on the VM.

    Resolution order:
      1. Literal spec path — VM /file endpoint expands ~ then probes.
      2. If spec contains glob (*), VM-side bash glob.
      3. APP_PATH_FALLBACKS list — try each candidate path on the VM,
         pick one whose basename matches the spec's.
      4. Last-resort find /home/user -name <basename> on the VM.

    ``trace_attempts`` (when provided) is appended-to with one entry per
    attempt: {"strategy", "path", "result"}. Used by the dispatcher to
    surface verifier reasoning in keynode_debug for offline inspection.
    """
    if not spec_path:
        return None
    target = spec_path  # NO host-side expanduser
    attempts = trace_attempts if trace_attempts is not None else []

    # 1. Literal path probe (VM expands ~)
    if "*" not in target:
        try:
            content = env.controller.get_file(target)
            attempts.append({
                "strategy": "literal",
                "path": target,
                "result": "found" if content is not None else "not found",
            })
            if content is not None:
                return target
        except Exception as e:
            attempts.append({
                "strategy": "literal", "path": target,
                "result": f"error: {e}",
            })

    # 2. Spec already had a glob — expand it on VM
    if "*" in target:
        resolved = _glob_first(env, target)
        attempts.append({
            "strategy": "vm_glob",
            "path": target,
            "result": resolved or "no match",
        })
        if resolved:
            return resolved

    # 3. App fallback list
    app = _detect_app(spec_path, instruction)
    spec_basename = os.path.basename(target).split("*")[0]
    if app and app in APP_PATH_FALLBACKS:
        for candidate in APP_PATH_FALLBACKS[app]:
            cand = candidate  # leave ~ for VM
            if "*" in cand:
                resolved = _glob_first(env, cand)
                if not resolved:
                    attempts.append({
                        "strategy": f"app_fallback({app})",
                        "path": cand, "result": "no match",
                    })
                    continue
            else:
                try:
                    found = env.controller.get_file(cand) is not None
                except Exception:
                    found = False
                if not found:
                    attempts.append({
                        "strategy": f"app_fallback({app})",
                        "path": cand, "result": "not found",
                    })
                    continue
                resolved = cand
            cand_basename = os.path.basename(resolved)
            attempts.append({
                "strategy": f"app_fallback({app})",
                "path": cand, "result": f"found at {resolved}",
            })
            if (cand_basename == spec_basename
                    or (spec_basename and spec_basename in cand_basename)):
                return resolved

    # 4. Last-resort find — only if we have a usable basename
    if spec_basename and "*" not in spec_basename:
        resolved = _last_resort_find(env, spec_basename)
        attempts.append({
            "strategy": "last_resort_find",
            "path": f"find /home/user -name {spec_basename}",
            "result": resolved or "no match",
        })
        if resolved:
            return resolved
    return None


# --------------------------------------------------------------------------- #
# file_grep
# --------------------------------------------------------------------------- #

def _check_patterns(text: str, patterns: List[str], all_must_match: bool,
                     proximity_lines: Optional[int]) -> bool:
    """Apply a list of regex patterns to text under AND/OR + optional
    proximity constraint."""
    try:
        compiled = [re.compile(p, re.MULTILINE) for p in patterns]
    except re.error:
        return False
    if proximity_lines is None:
        if all_must_match:
            return all(c.search(text) for c in compiled)
        return any(c.search(text) for c in compiled)
    # proximity: all patterns must co-occur within a sliding window of
    # exactly ``proximity_lines`` consecutive lines (1-indexed count, so
    # proximity_lines=3 means a 3-line window — typical for same-entry
    # "key" + "command" co-occurrence in keybindings.json).
    lines = text.split("\n")
    n = len(lines)
    win = max(1, proximity_lines)
    for i in range(n):
        window = "\n".join(lines[i:i + win])
        if all(c.search(window) for c in compiled):
            return True
    return False


def run_file_grep(spec: VerifySpec, env: Any, instruction: str = "") -> Optional[bool]:
    """Thin wrapper around the traced version — kept for unit tests +
    callers that don't need the trace dict."""
    verdict, _ = _run_file_grep_traced(spec, env, instruction)
    return verdict


# --------------------------------------------------------------------------- #
# url_match
# --------------------------------------------------------------------------- #

# Browser address-bar a11y rows. Chrome / Chromium expose the active URL
# in a single, identifiable row, e.g.:
#
#   entry\tAddress and search bar (https://...)\tfocused,enabled
#   combo-box\tAddress and search bar (chrome://settings/...)\tfocused
#   entry\tURL (https://...)\tfocused,enabled     ← Firefox / chrome variants
#
# We require BOTH conditions:
#   (1) tag is one of the editable-text-control kinds, AND
#   (2) text contains a URL-bearing-control label ("Address...bar" /
#       "URL" / "Location") plus a parenthesized URL.
#
# Older heuristic ("first https? row in the dump") false-positives any
# time another app is active (e.g. Thunderbird email body shows literal
# URL strings inside <link> rows) — see 58565672 regression where the
# url_match verifier matched an email-body URL against itself.
_ADDRESS_BAR_TAGS = ("entry", "combo-box", "text-field", "textbox")
_ADDRESS_BAR_LABEL_RE = re.compile(
    r"(?:Address(?:\s+and\s+search)?\s+bar|^URL\b|^Location\b|Location bar)",
    re.IGNORECASE,
)
_ADDRESS_BAR_URL_RE = re.compile(r"\((?P<url>(?:https?|chrome|about|file)[^)\s]+)\)")
_BROWSER_PROTOCOL_PREFIXES = ("http://", "https://", "chrome://", "about:", "file://")


def _extract_active_url(a11y_text: str) -> Optional[str]:
    """Pull the active tab URL from a11y. Strict address-bar scope only.

    Returns the URL string when a browser address-bar row is present, or
    None otherwise (NEVER falls back to "any http row in the dump" —
    that's the bug that lets Thunderbird email-body links satisfy a
    Chrome url_match verify). Returning None routes the outcome to the
    LLM judge path, which is far less likely to false-positive across
    apps.
    """
    if not a11y_text:
        return None
    for line in a11y_text.split("\n"):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        tag = parts[0].strip().lower()
        text = parts[1]
        if tag not in _ADDRESS_BAR_TAGS:
            continue
        if not _ADDRESS_BAR_LABEL_RE.search(text):
            continue
        # First check parenthesized form: 'Address... (https://...)'
        m = _ADDRESS_BAR_URL_RE.search(text)
        if m:
            return m.group("url")
        # Some browsers put the URL after the label without parens.
        for tok in text.split():
            if tok.startswith(_BROWSER_PROTOCOL_PREFIXES):
                return tok
    return None


def run_url_match(spec: VerifySpec, a11y_text: str) -> Optional[bool]:
    verdict, _ = _run_url_match_traced(spec, a11y_text)
    return verdict


# --------------------------------------------------------------------------- #
# a11y_match
# --------------------------------------------------------------------------- #

def run_a11y_match(spec: VerifySpec, a11y_text: str) -> Optional[bool]:
    """Thin wrapper around the traced version. See ``_run_a11y_match_traced``
    for semantics."""
    verdict, _ = _run_a11y_match_traced(spec, a11y_text)
    return verdict


# --------------------------------------------------------------------------- #
# Top-level dispatch
# --------------------------------------------------------------------------- #

def run_verify(spec: VerifySpec, env: Any, a11y_text: str = "",
                instruction: str = "") -> Optional[bool]:
    """Run a verify spec deterministically. See module docstring for the
    True/False/None contract."""
    verdict, _ = verify_with_trace(spec, env, a11y_text, instruction)
    return verdict


def verify_with_trace(
    spec: VerifySpec, env: Any, a11y_text: str = "",
    instruction: str = "",
    *,
    slots: Optional[Dict[str, VerifySpecSlot]] = None,
) -> tuple:
    """Like ``run_verify`` but also returns a debug trace dict describing
    what the verifier did:

      file_grep: {kind, spec_path, resolved_path, file_size_bytes,
                   patterns_compiled_ok, per_pattern_matches:[{pattern,
                   matched_lines:[(line_no, snippet)]}], proximity_used,
                   verdict_reason}
      url_match: {kind, url_extracted, regex, verdict_reason}
      a11y_match: {kind, matched_row:str|null, candidate_count,
                    verdict_reason}

    Used by ``key_node_debug`` to persist per-step KeyNode dispatcher
    decisions for offline inspection. Returns ``(verdict, trace_dict)``."""
    if spec is None or not spec.is_valid():
        return None, {"kind": "invalid", "verdict_reason": "spec is None or invalid"}

    # Slot resolution — substitute ``${name}`` references with bound values
    # (or short-circuit to UNCERTAIN if anything is unbound). Outcomes that
    # don't use slots are unaffected: _resolve_spec_slots returns the spec
    # unchanged when no ``${...}`` refs are present.
    spec, unresolved = _resolve_spec_slots(spec, slots)
    if unresolved:
        return None, {
            "kind": spec.kind,
            "verdict_reason": (
                f"unresolved slot(s): {unresolved} — deterministic verify "
                "deferred until the perceiver binds these values"
            ),
            "unresolved_slots": unresolved,
        }
    # Bracket-style placeholder sweep — catches init-LLM authoring
    # failures where the model wrote ``[Planned remarks for Slide 1]`` /
    # ``<value>`` / ``{{TODO}}`` in a verify value, expecting the
    # framework to fill it in (but only ``${slot}`` is supported). Without
    # this check, the verifier literally greps for that string forever
    # and never matches. We short-circuit to None (uncertain) so the
    # outcome routes to LLM judge instead of staying pending in a loop.
    bracket_finds = _scan_placeholders_anywhere(spec)
    if bracket_finds:
        return None, {
            "kind": spec.kind,
            "verdict_reason": (
                f"spec contains bracket-style placeholder(s) the init "
                f"LLM forgot to instantiate: {bracket_finds[:3]} — "
                "deterministic verify cannot succeed (literal grep); "
                "deferring to LLM judge. To carry a runtime value, use "
                "${slot_name} syntax with the slot declared in the "
                "top-level slots[] array."
            ),
            "bracket_placeholders": bracket_finds,
        }

    if spec.kind == "file_grep":
        return _run_file_grep_traced(spec, env, instruction)
    if spec.kind == "url_match":
        return _run_url_match_traced(spec, a11y_text)
    if spec.kind == "a11y_match":
        return _run_a11y_match_traced(spec, a11y_text)
    if spec.kind == "shell_command":
        return _run_shell_command_traced(spec, env)
    if spec.kind == "calc_verify":
        return _run_calc_verify_traced(spec, env)
    if spec.kind == "impress_verify":
        return _run_impress_verify_traced(spec, env)
    if spec.kind == "writer_verify":
        return _run_writer_verify_traced(spec, env)
    return None, {"kind": spec.kind, "verdict_reason": "unknown kind"}


def _run_impress_verify_traced(spec, env):
    """Impress counterpart of _run_calc_verify_traced. Builds the
    python3 -c body from spec.impress_checks via build_impress_verify_python_body,
    then dispatches through the shared shell-command runner."""
    from mm_agents.structagent.actions.impress.actions import build_impress_verify_python_body
    body = build_impress_verify_python_body(spec.impress_checks or [])
    synth = VerifySpec(
        kind="shell_command",
        command=["python3", "-c",
                 _wrap_verify_body(body, "libreoffice_impress", env)],
        expected_substring=["PASS"],
        expected_exit_code=0,
    )
    return _run_shell_command_traced(synth, env)


def _wrap_verify_body(body: str, domain: str, env: Any = None) -> str:
    """Wrap a structured-verify python body with the cli_run_uno
    runtime for ``domain`` (connect + setup + ops library). The domain
    is KNOWN here — no function-name scanning, so it never drifts when
    a new verify op is added. The wrapped script carries the sentinel
    so ``_maybe_wrap_calc_verify_command`` won't re-wrap it.

    Before building, ensure the domain's ops-library module is present
    on the VM. ``build_runtime_script`` emits a tiny
    ``from _<domain>_ops_lib_<hash> import *`` shim when the process
    cache says the module was uploaded — but an OSWorld snapshot reset
    between tasks wipes ``/tmp``, so that cached module can be gone.
    The actor path re-checks + re-uploads at every predict() entry;
    the verify path did NOT, so its shim imported a missing module →
    ``ModuleNotFoundError`` → EVERY structured verify of that domain
    hard-failed. ``maybe_upload_ops_lib`` re-verifies the /tmp file and
    re-uploads under the current content hash when missing; if it
    can't (no runner), ``get_uploaded_module`` returns None and the
    wrapper safely inlines the full library instead."""
    if env is not None:
        try:
            from mm_agents.structagent.actions.uno._ops_lib_remote import maybe_upload_ops_lib
            maybe_upload_ops_lib(env, domain)
        except Exception:
            pass
    from mm_agents.structagent.actions.uno.cli_run_uno_helpers import build_runtime_script
    return build_runtime_script(body, domain=domain)


def _run_writer_verify_traced(spec, env):
    """Writer counterpart of _run_calc_verify_traced. Builds the
    python3 -c body from spec.writer_checks, wraps it with the Writer
    cli_run_uno boilerplate (connect + setup + WRITER_OPS_LIBRARY) for
    the KNOWN domain, then dispatches through the shell-command runner."""
    from mm_agents.structagent.actions.writer.actions import build_writer_verify_python_body
    body = build_writer_verify_python_body(spec.writer_checks or [])
    synth = VerifySpec(
        kind="shell_command",
        command=["python3", "-c",
                 _wrap_verify_body(body, "libreoffice_writer", env)],
        expected_substring=["PASS"],
        expected_exit_code=0,
    )
    return _run_shell_command_traced(synth, env)


def _run_calc_verify_traced(spec: VerifySpec, env: Any):
    """Build the python3 -c command from ``spec.calc_checks`` and
    dispatch to the shared shell_command runner.

    The structured ``calc_checks`` list lets the init_ledger LLM avoid
    writing Python source for verify bodies — the framework deterministically
    constructs the function calls + the PASS/FAIL print. Boilerplate
    (UNO connect + helper definitions) is the same as for the legacy
    python3 -c Calc path: ``_maybe_wrap_calc_verify_command`` triggers
    on the ``verify_*`` references in the body and prepends the
    runtime.
    """
    from mm_agents.structagent.actions.calc.actions import build_calc_verify_python_body
    body = build_calc_verify_python_body(spec.calc_checks or [])
    synth = VerifySpec(
        kind="shell_command",
        command=["python3", "-c",
                 _wrap_verify_body(body, "libreoffice_calc", env)],
        expected_substring=["PASS"],
        expected_exit_code=0,
    )
    return _run_shell_command_traced(synth, env)


def _run_file_grep_traced(spec: VerifySpec, env: Any, instruction: str):
    """Match patterns in a VM file. Deterministic text/regex match with
    optional proximity + per-app path fallbacks; no LLM involvement."""
    trace: dict = {
        "kind": "file_grep",
        "spec_path": spec.file_path,
        "resolved_path": None,
        "file_size_bytes": None,
        "patterns": list(spec.patterns or []),
        "all_must_match": spec.all_must_match,
        "proximity_lines": spec.proximity_lines,
        "fallback_attempts": [],
        "per_pattern_matches": [],
        "verdict_reason": "",
    }
    if not spec.file_path or not spec.patterns:
        trace["verdict_reason"] = "missing file_path or patterns"
        return None, trace
    real_path = resolve_file_path(
        spec.file_path, env, instruction,
        trace_attempts=trace["fallback_attempts"],
    )
    trace["resolved_path"] = real_path
    if real_path is None:
        # File does not exist anywhere on the VM (literal + curated fallbacks
        # + last-resort find all missed). For "task: write to file" outcomes
        # this is a CONFIRMED NOT-SATISFIED — a non-existent file cannot
        # contain the patterns. We return False (not None) so the dispatcher
        # trusts the verdict and does not fall back to the LLM, which would
        # otherwise be tricked by transient UI cues (search-box echoes,
        # record-mode keystroke previews) into a false-positive satisfaction.
        trace["verdict_reason"] = (
            "file does not exist on VM (after literal + app fallback + "
            "last-resort find); treating as NOT-SATISFIED"
        )
        return False, trace
    try:
        content_bytes = env.controller.get_file(real_path)
    except Exception as e:
        trace["verdict_reason"] = f"get_file raised: {e}"
        return None, trace
    if content_bytes is None:
        trace["verdict_reason"] = "get_file returned None despite resolved path"
        return None, trace
    trace["file_size_bytes"] = len(content_bytes)
    # Cheap content hash so OutcomeStateCache can detect whether the
    # file changed since the first verifier observation. Used to feed
    # the planner's force_replan prompt with a "file modified since
    # task start: yes/no" diagnostic line.
    trace["file_content_hash"] = hashlib.sha256(content_bytes).hexdigest()[:16]
    # File content preview — first 800 chars after decode. Lets the
    # LLM-driven diagnoser (and the planner) see what the actor
    # actually wrote vs what the verifier expects.
    try:
        _head = content_bytes[:1500].decode("utf-8", errors="replace")
        trace["file_content_head"] = _head[:800]
    except Exception:
        trace["file_content_head"] = ""
    try:
        text = content_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        trace["verdict_reason"] = f"decode failed: {e}"
        return None, trace
    # Per-pattern match record (which lines hit) — for debug visibility.
    lines = text.split("\n")
    per_pat = []
    for p in spec.patterns:
        try:
            cre = re.compile(p, re.MULTILINE)
        except re.error as e:
            per_pat.append({"pattern": p, "compile_error": str(e), "matched_lines": []})
            continue
        matched = []
        for i, ln in enumerate(lines):
            if cre.search(ln):
                matched.append({"line_no": i + 1, "snippet": ln[:200]})
                if len(matched) >= 5:
                    break
        per_pat.append({"pattern": p, "matched_lines": matched})
    trace["per_pattern_matches"] = per_pat
    verdict = _check_patterns(text, spec.patterns, spec.all_must_match,
                              spec.proximity_lines)
    trace["verdict_reason"] = (
        "all patterns matched"
        + (" with proximity" if spec.proximity_lines and verdict else "")
        if verdict else "patterns missed (or proximity not satisfied)"
    )
    return verdict, trace


def _run_url_match_traced(spec: VerifySpec, a11y_text: str):
    trace: dict = {
        "kind": "url_match",
        "url_pattern": spec.url_pattern,
        "url_extracted": None,
        "verdict_reason": "",
    }
    if not spec.url_pattern:
        trace["verdict_reason"] = "missing url_pattern"
        return None, trace
    url = _extract_active_url(a11y_text)
    trace["url_extracted"] = url
    if url is None:
        trace["verdict_reason"] = "no URL found in a11y"
        return None, trace
    try:
        verdict = bool(re.search(spec.url_pattern, url))
    except re.error as e:
        trace["verdict_reason"] = f"regex compile error: {e}"
        return False, trace
    trace["verdict_reason"] = "URL matched pattern" if verdict else "URL did not match"
    return verdict, trace


def _run_a11y_match_traced(spec: VerifySpec, a11y_text: str):
    trace: dict = {
        "kind": "a11y_match",
        "tag": spec.tag,
        "text_contains": list(spec.text_contains or []),
        "state_contains": list(spec.state_contains or []),
        "matched_row": None,
        "rows_scanned": 0,
        "verdict_reason": "",
    }
    if not (spec.tag or spec.text_contains or spec.state_contains):
        trace["verdict_reason"] = "no constraint specified"
        return None, trace
    if not a11y_text or not a11y_text.strip():
        trace["verdict_reason"] = "a11y_text empty"
        return None, trace
    rows_scanned = 0
    for line in a11y_text.split("\n"):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rows_scanned += 1
        tag, text, state = parts[0], parts[1], parts[2]
        if spec.tag and spec.tag.lower() not in tag.lower():
            continue
        if spec.text_contains:
            if not all(s.lower() in text.lower() for s in spec.text_contains):
                continue
        if spec.state_contains:
            state_l = state.lower()
            if not all(s.lower() in state_l for s in spec.state_contains):
                continue
        trace["matched_row"] = line[:240]
        trace["rows_scanned"] = rows_scanned
        trace["verdict_reason"] = "row matched all constraints"
        return True, trace
    trace["rows_scanned"] = rows_scanned
    trace["verdict_reason"] = f"no row matched after scanning {rows_scanned} rows"
    return False, trace


_CALC_LIB_FUNC_NAMES = (
    # Mutation ops
    "op_set_cell", "op_set_range_values", "op_new_sheet", "op_rename_sheet",
    "op_pivot_table", "op_format_number_range", "op_sort_range",
    "op_freeze_pane", "op_duplicate_range",
    "op_split_text", "op_hide_range", "op_insert_chart",
    "op_export_file",
    "op_set_color", "op_merge_cells", "op_data_validation",
    "op_reorder_columns", "op_copy_to_clipboard",
    # Verify ops
    "verify_sheet_exists", "verify_cell_value", "verify_pivot_exists",
    "verify_cell_format", "verify_sort_order", "verify_freeze_pane",
    "verify_duplicate_range_match",
    "verify_row_hidden", "verify_col_hidden", "verify_chart_exists",
    "verify_file_exists",
    "verify_cell_color", "verify_cell_merged", "verify_data_validation",
    "verify_clipboard_has_content",
)


def _references_calc_ops_lib(code: str) -> bool:
    """True iff the verify code uses any Calc-library function name."""
    if not code:
        return False
    return any(fn in code for fn in _CALC_LIB_FUNC_NAMES)


_IMPRESS_LIB_FUNC_NAMES = (
    # Mutation ops
    "op_impress_set_text", "op_impress_set_font",
    "op_impress_set_paragraph", "op_impress_set_text_color",
    "op_impress_new_slide", "op_impress_delete_slide",
    "op_impress_duplicate_slide", "op_impress_move_slide",
    "op_impress_set_slide_orientation", "op_impress_set_slide_size",
    "op_impress_set_notes", "op_impress_set_transition",
    "op_impress_set_background",
    "op_impress_insert_shape", "op_impress_insert_image",
    "op_impress_insert_chart", "op_impress_insert_table",
    "op_impress_set_master_color", "op_impress_set_app_setting",
    "op_impress_export_file",
    # Verify ops
    "verify_impress_slide_count", "verify_impress_slide_exists_at_index",
    "verify_impress_shape_text", "verify_impress_font_props",
    "verify_impress_paragraph_props", "verify_impress_shape_exists",
    "verify_impress_slide_background", "verify_impress_slide_notes",
    "verify_impress_slide_orientation", "verify_impress_slide_transition",
    "verify_impress_chart_exists", "verify_impress_table_exists",
    "verify_impress_image_dims", "verify_impress_master_color",
    "verify_impress_app_setting", "verify_impress_file_exists",
)


def _references_impress_ops_lib(code: str) -> bool:
    """True iff the verify code uses any Impress-library function name."""
    if not code:
        return False
    return any(fn in code for fn in _IMPRESS_LIB_FUNC_NAMES)


# Writer ops-library function names — keep in sync with writer_ops_lib.py.
_WRITER_LIB_FUNC_NAMES = (
    "op_paste_at",
    "verify_table_count", "verify_table_after_heading",
)


def _references_writer_ops_lib(code: str) -> bool:
    """True iff the verify code uses any Writer-library function name."""
    if not code:
        return False
    return any(fn in code for fn in _WRITER_LIB_FUNC_NAMES)


def _maybe_wrap_calc_verify_command(cmd):
    """When the argv looks like ``['python3', '-c', '<code>']`` AND the
    code references a Calc or Impress library function, replace the
    code with the full wrapped runtime script (connect block + domain
    setup + ops library + the original code as the body).

    Pass-through for non-python3 verifiers, non-LO-lib verifiers
    (plain bash test commands, file_grep, etc.), and any code that
    doesn't actually use the library.
    """
    if not isinstance(cmd, list) or len(cmd) != 3:
        return cmd
    if cmd[0] != "python3" or cmd[1] != "-c":
        return cmd
    code = cmd[2]
    if not isinstance(code, str):
        return cmd
    # Idempotency: a structured verify (_run_calc/writer/impress_verify
    # _traced) already wraps the body with build_runtime_script for its
    # KNOWN domain. Re-wrapping would nest the script and break it.
    # Detect the wrapper's sentinel and pass through.
    try:
        from mm_agents.structagent.actions.uno.cli_run_uno_helpers import WRAPPED_SENTINEL
        if code.lstrip().startswith(WRAPPED_SENTINEL):
            return cmd
    except Exception:
        pass
    if _references_calc_ops_lib(code):
        domain = "libreoffice_calc"
    elif _references_impress_ops_lib(code):
        domain = "libreoffice_impress"
    elif _references_writer_ops_lib(code):
        domain = "libreoffice_writer"
    else:
        return cmd
    try:
        from mm_agents.structagent.actions.uno.cli_run_uno_helpers import build_runtime_script
        wrapped = build_runtime_script(code, domain=domain)
    except Exception:
        return cmd
    return ["python3", "-c", wrapped]


_SCRIPT_EXT_RE = re.compile(r"\.(py|js|rb|sh|pl|ts)$", re.IGNORECASE)


def _sanitize_pgrep_self_match(cmd: Any) -> Any:
    """Fix a ``pgrep -f <name>`` process check that matches non-target
    processes — chiefly the agent's OWN machinery.

    ``pgrep -f`` matches against the FULL command line of every
    process. That is wrong for an "is application X running" check:

      * it self-matches the verifier's own wrapper
        ``bash -c '... pgrep -f X ...'`` (cmdline contains X), and
      * worse, EVERY ``cli_run_uno`` script the agent runs is a
        ``python3 -c '<...>'`` whose source embeds the UNO connect
        boilerplate (``subprocess.Popen(['soffice', '--accept=…'])``,
        ``StarOffice.ComponentContext``) — so ``pgrep -f soffice``
        matches every in-flight calc/writer/impress verify or action.

    Observed: a "kill LibreOffice" task whose ``pgrep -f soffice``
    verify reported 152 matches (151 of them the agent's own
    ``python3 -c`` UNO scripts) and never passed for 15 steps while
    real LibreOffice was dead by step 3.

    The bracket trick (``[s]office``) does NOT help — it still matches
    the literal ``soffice`` inside those script bodies. The only
    correct check matches the process NAME (``comm``), not the
    cmdline: drop ``-f`` so ``pgrep`` matches the executable basename.
    ``pgrep soffice`` → 1 match (the real ``soffice.bin``); the
    ``python3`` UNO scripts have ``comm=python3`` and no longer match.

    ``-f`` is kept only when the pattern looks like it genuinely needs
    cmdline matching — a path, a pattern with whitespace, or a script
    filename (``foo.py``) whose host process ``comm`` is the
    interpreter, not the script.
    """
    if not isinstance(cmd, list):
        return cmd

    def _needs_f(pat: str) -> bool:
        # The pattern truly needs full-cmdline matching when it is not
        # a bare process name: a path, whitespace, or a script file.
        return ("/" in pat or any(c.isspace() for c in pat)
                or bool(_SCRIPT_EXT_RE.search(pat)))

    out = list(cmd)
    # argv form: ["pgrep", "-f", PAT, ...]  → drop "-f" for a bare name
    for i in range(len(out) - 2):
        if (out[i] == "pgrep" and out[i + 1] == "-f"
                and isinstance(out[i + 2], str) and out[i + 2]
                and not _needs_f(out[i + 2])):
            del out[i + 1]   # remove the "-f" flag
            break
    # string form: a `bash -c "... pgrep -f PAT ..."` token
    for i, tok in enumerate(out):
        if isinstance(tok, str) and "pgrep -f " in tok:
            out[i] = re.sub(
                r'pgrep\s+-f\s+([A-Za-z0-9][\w.-]*)',
                lambda m: ('pgrep ' + m.group(1)) if not _needs_f(m.group(1))
                else m.group(0),
                tok)
    return out


def _run_shell_command_traced(spec: VerifySpec, env: Any):
    """Execute ``spec.command`` (argv list) on the VM via
    ``env.controller.run_bash_script`` and check the result.

    Verdict True iff:
      • the command finished (status="success"), AND
      • exit code matches ``spec.expected_exit_code`` (default 0), AND
      • if ``expected_substring`` set, AT LEAST ONE substring (case-
        insensitive) appears in stdout, AND
      • if ``forbidden_substring`` set, NONE of them appear in stdout.

    Returns ``(None, trace)`` when the env has no executor (allows the
    KeyNode dispatcher to fall back to the LLM judge path)."""
    # If this is a Calc verify that uses our ops library, pre-wrap the
    # python3 -c body with the Calc cli_run_uno boilerplate so the
    # library functions become available in the verifier subprocess.
    # See ``_maybe_wrap_calc_verify_command`` for the trigger rules.
    wrapped_cmd = _maybe_wrap_calc_verify_command(list(spec.command or []))
    # Defuse a self-matching ``pgrep -f <name>`` process check (the
    # verifier's own wrapper shell otherwise matches itself).
    wrapped_cmd = _sanitize_pgrep_self_match(wrapped_cmd)
    trace: dict = {
        "kind": "shell_command",
        "command": list(spec.command or []),
        "expected_exit_code": spec.expected_exit_code,
        "expected_substring": list(spec.expected_substring or []),
        "forbidden_substring": list(spec.forbidden_substring or []),
        "exit_code": None,
        "stdout_head": None,
        "verdict_reason": "",
    }
    if not spec.command:
        trace["verdict_reason"] = "command is empty"
        return None, trace
    if not env or not getattr(env, "controller", None) \
            or not hasattr(env.controller, "run_bash_script"):
        trace["verdict_reason"] = "env has no run_bash_script executor"
        return None, trace

    # shlex-quote each argv element + redirect stderr into stdout, then
    # echo the trailing ``__EXIT__ $?`` marker so we can recover the
    # exit code from the runner's combined-stdout return value.
    #
    # Post-failure diagnostic block: when exit != 0, run a tiny read-only
    # context probe so the verifier trace surfaces enough info for the
    # downstream LLM diagnoser to identify root cause itself (rather
    # than us hand-coding every failure-mode heuristic). Probes are
    # generic — pwd / echo / ls on the last positional argv token —
    # which produce meaningful stderr ("No such file or directory") for
    # path-related commands (test / which / similar) and harmless noise
    # for non-path commands (the LLM will ignore it).
    # Use the (possibly wrapped) command — `wrapped_cmd` is identical
    # to spec.command for non-Calc-lib verifiers; for Calc-lib verifiers
    # it has the python3 -c body replaced with the runtime-wrapped
    # script.
    # ─────────────────────────────────────────────────────────────────
    # [FIX shell_verify_tilde] LLM-authored verify specs frequently use
    # ``~`` or ``~/path`` in argv (it's "shell-flavored" by reading,
    # which is how the LLM thinks). But shell_command verifies dispatch
    # argv ELEMENT-BY-ELEMENT through ``shlex.quote`` → each path lands
    # inside single-quotes, where bash does NOT perform tilde expansion
    # — the cmd then asserts on a literal ``~/...`` directory that
    # doesn't exist, the outcome stays pending forever, and the planner
    # falls into infinite REPLAN.
    #
    # Worse: when the verify is "file exists?" and reports MISSING,
    # the planner trusts it and recreates the file from scratch,
    # OVERWRITING the real downloaded artifact (observed on
    # 3680a5ee — agent wrote fake text into binary xlsx/ods files).
    #
    # Fix: before shlex.quoting, replace a leading ``~`` / ``~/`` with
    # the absolute home directory. OSWorld VMs always use
    # ``/home/user``; we hardcode it as a known constant rather than
    # round-tripping through the controller (would need a synchronous
    # probe per verify call).
    #
    # To roll back: revert this block to the original
    # ``shlex.quote(str(a))`` one-liner above.
    _VM_HOME = "/home/user"
    def _expand_argv_tilde(a):
        s = str(a)
        if s == "~":
            return _VM_HOME
        if s.startswith("~/"):
            return _VM_HOME + s[1:]
        return s
    expanded_cmd = [_expand_argv_tilde(a) for a in wrapped_cmd]
    quoted = " ".join(shlex.quote(str(a)) for a in expanded_cmd)
    last_arg = ""
    for a in reversed(expanded_cmd):
        s = str(a)
        if s and not s.startswith("-"):
            last_arg = s
            break
    quoted_last = shlex.quote(last_arg) if last_arg else ""
    # ──── end FIX shell_verify_tilde ────
    # The DIAG block is a debug helper for path-style argv (e.g.
    # ``["test","-f","/path"]``): on non-zero exit, it echoes pwd +
    # ls's the last argv token to surface "no such file or directory"
    # context. For argv that EMBED CODE in the last token
    # (``["python3","-c","<36KB UNO wrapper>"]`` — every calc_verify
    # path), the DIAG block inlines that ~36KB code three more times
    # (echo + ls + dirname), inflating the bash script to ~4× its
    # natural size. Empirically this 144KB bash payload correlates
    # with 110s HTTP-timeouts on the env API's ``/execute`` path —
    # likely subprocess argv parsing / shell-meta scanning on the
    # huge ls argument blocks the server for the full timeout window.
    # Suppress DIAG whenever the last token is code-like (multi-line,
    # or too large to be a path).
    looks_like_code = (
        len(last_arg) > 512
        or "\n" in last_arg
        or last_arg.startswith("import ")
        or last_arg.startswith("_OPS ")
    )
    if quoted_last and not looks_like_code:
        diag_block = (
            '\nif [ "$__EC" -ne "0" ]; then\n'
            '  echo "__DIAG_START__"\n'
            '  echo "pwd: $(pwd)"\n'
            f'  echo "(echo last_arg): " {quoted_last}\n'
            f'  ls -la -- {quoted_last} 2>&1 | head -3 || true\n'
            f'  ls -la -- "$(dirname -- {quoted_last})" 2>&1 | head -8 || true\n'
            '  echo "__DIAG_END__"\nfi'
        )
    else:
        diag_block = ""
    script = f'{quoted} 2>&1\n__EC=$?\necho "__EXIT__ $__EC"{diag_block}'

    # ── Verbose trace logging — captures every layer of the verify
    # subprocess dispatch so a 110s HTTP hang (or any other failure
    # mode) leaves a self-describing audit trail in runtime.log:
    #
    #   • pre-call:  argv head, script head + tail, full script byte
    #                length, spec kind, the wall-clock start time;
    #   • on success: exit code + elapsed + stdout/stderr head;
    #   • on raise:   exception class + elapsed + script dumped to
    #                 /tmp for offline inspection;
    #   • on non-success-status: status + error + elapsed + script
    #                            dumped (helps attribute HTTP
    #                            timeouts to specific verifies).
    import time as _t
    import os as _os
    _t0 = _t.time()
    cmd_head = " ".join(str(a) for a in (spec.command or []))[:200]
    script_head = script.split("\n", 1)[0][:240]
    script_tail = script[-240:] if len(script) > 240 else ""
    logger.info(
        "[ShellCmd] dispatching run_bash_script "
        "(kind=%s, script_bytes=%d, argv_head=%r, "
        "script_head=%r, script_tail=%r)",
        spec.kind, len(script), cmd_head, script_head, script_tail,
    )

    def _dump_failed_script(reason_tag: str) -> Optional[str]:
        """Save the failing script under /tmp/_verify_fail_<ts>.sh
        so the exact bytes sent to bash can be inspected post-hoc.
        Returns the path (or None if dump itself fails).
        """
        try:
            path = (f"/tmp/_verify_fail_{int(_t0)}_"
                    f"{reason_tag}_{_os.getpid()}.sh")
            with open(path, "w") as _f:
                _f.write(script)
            return path
        except Exception:
            return None

    try:
        result = env.controller.run_bash_script(script, timeout=10)
    except Exception as e:
        _elapsed = _t.time() - _t0
        _dump_path = _dump_failed_script("raise")
        logger.info(
            "[ShellCmd] run_bash_script RAISED after %.1fs: "
            "exc=%s msg=%r (script dumped to %s)",
            _elapsed, type(e).__name__, str(e)[:200], _dump_path,
        )
        trace["verdict_reason"] = f"run_bash_script raised: {e}"
        return None, trace
    _elapsed = _t.time() - _t0
    if not result or result.get("status") != "success":
        _dump_path = _dump_failed_script("status")
        logger.info(
            "[ShellCmd] run_bash_script FAILED after %.1fs: "
            "status=%s err=%r (script dumped to %s)",
            _elapsed,
            (result.get('status') if result else None),
            (result.get('error') or '')[:240],
            _dump_path,
        )
        trace["verdict_reason"] = (
            f"runner status != success: {result.get('status') if result else None}"
        )
        return None, trace
    _stdout_preview = ((result.get("output") or "")[:200]
                       .replace("\n", " ⏎ "))
    logger.info(
        "[ShellCmd] run_bash_script OK after %.1fs: "
        "returncode=%s stdout_head=%r",
        _elapsed, result.get("returncode"), _stdout_preview,
    )
    raw_out = (result.get("output") or "")
    # Strip off the trailing ``__EXIT__ <N>`` marker. Treat as exit 0
    # if the marker is absent (shouldn't happen but be safe).
    exit_code: Optional[int] = 0
    stdout = raw_out
    m = re.search(r"^__EXIT__ (\d+)\s*$", raw_out, re.MULTILINE)
    if m:
        try:
            exit_code = int(m.group(1))
        except Exception:
            exit_code = 0
        stdout = raw_out[:m.start()].rstrip("\n")
        # Parse the post-failure diagnostic block (only present when
        # exit != 0 and we asked for it).
        tail = raw_out[m.end():]
        diag_m = re.search(
            r"__DIAG_START__\n(.*?)\n__DIAG_END__",
            tail, re.DOTALL,
        )
        if diag_m:
            trace["diagnostic"] = diag_m.group(1).strip()[:1500]
    trace["exit_code"] = exit_code
    trace["stdout_head"] = stdout[:600]

    expected_exit = spec.expected_exit_code if spec.expected_exit_code is not None else 0
    if exit_code != expected_exit:
        trace["verdict_reason"] = (
            f"exit_code {exit_code} != expected {expected_exit}"
        )
        return False, trace
    stdout_l = stdout.lower()
    # Forbidden first — any hit is an instant FAIL even if expected also hit.
    for s in (spec.forbidden_substring or []):
        if s and s.lower() in stdout_l:
            trace["verdict_reason"] = f"forbidden substring matched: {s!r}"
            return False, trace
    if spec.expected_substring:
        hit = next(
            (s for s in spec.expected_substring
             if s and s.lower() in stdout_l),
            None,
        )
        if hit is None:
            trace["verdict_reason"] = "no expected substring matched stdout"
            return False, trace
        trace["matched_substring"] = hit
    trace["verdict_reason"] = "exit code + substring rules satisfied"
    return True, trace
