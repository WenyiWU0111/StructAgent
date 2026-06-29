"""Deterministic verifiers for the KeyNode dispatcher.

Each ``run_*`` returns:
    True   — SATISFIED. Trust it; skip the LLM.
    False  — NOT-SATISFIED. Control/file/URL was observable and didn't match;
             trust it (outcome stays pending → planner replans).
    None   — UNCERTAIN. Spec couldn't be evaluated (path unresolved, file
             unreadable, URL not extractable, a11y row not in frame). Caller
             falls back to the LLM KeyNode path to avoid false-negatives on
             transient observation gaps.

The dispatcher maps True → outcome_satisfied, False → nothing (stays pending),
None → the original LLM KeyNode call.
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

# Regex-bearing fields need ``re.escape`` on substituted values; text fields
# are substituted verbatim. Fields in neither set aren't slot-substituted.
_SPEC_REGEX_FIELDS = ("url_pattern",)                       # str fields
_SPEC_REGEX_LIST_FIELDS = ("patterns",)                     # List[str] fields
_SPEC_TEXT_FIELDS = ("file_path", "tag")                    # str
_SPEC_TEXT_LIST_FIELDS = (
    "text_contains", "state_contains",
    "expected_substring", "forbidden_substring",
    "command",
)                                                            # List[str]


# Bracket-style placeholders the init LLM emits when it doesn't know a verify
# value (sibling failure mode to the slot system). These get grepped literally
# and never match → outcome pending forever. Caught at verify time and treated
# as a spec-authoring failure → fall back to LLM judge.
# Patterns kept conservative to avoid eating legit content.
_PLACEHOLDER_RES = [
    re.compile(r"\[[A-Z][\w\- ]{1,80}\]"),        # [Planned remarks for Slide 1]
    re.compile(r"<[a-zA-Z_][\w\- ]{1,40}>"),       # <value>, <slide_text>
    re.compile(r"\{\{[\w\- ]{1,40}\}\}"),          # {{TODO}}, {{slot_name}}
    re.compile(r"\b\[(?:TODO|FIXME|PLACEHOLDER|YOUR_TEXT|HERE|EXAMPLE|"
               r"VALUE|TEXT|CONTENT)\b[^\]]*\]", re.IGNORECASE),
]


def _has_bracket_placeholder(s: Optional[str]) -> bool:
    """True if `s` has a bracket-style placeholder the init LLM forgot to
    instantiate (vs a proper ``${slot}`` ref, the supported way to say
    'I don't know this yet')."""
    if not s or not isinstance(s, str):
        return False
    return any(p.search(s) for p in _PLACEHOLDER_RES)


def _scan_placeholders_anywhere(spec: VerifySpec) -> List[Tuple[str, str]]:
    """Scan every string field (including nested *_checks dicts) for
    bracket placeholders. Returns ``(field_path, value)`` tuples; empty
    means clean."""
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
    """Substitute bound ``${name}`` refs in the spec; return the (possibly
    new) spec plus any unresolved slot names found anywhere in it.

    Regex-bearing fields get ``re.escape`` on substituted values so a bound
    URL doesn't get re-read as regex metacharacters.

    With empty/None ``slots``: still scans for ``${...}`` refs (an unregistered
    ref is a spec-author bug, surfaced as ``unresolved``) but substitutes
    nothing. Recurses into calc/impress/writer_checks (lists of op dicts):
    string values get substituted, non-strings pass through.
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
        """Substitute ``${slot}`` refs in every string value of a check-op
        dict; recurse into nested lists/dicts. Non-strings pass through."""
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
    # Structured-check lists: each item is an init-LLM op dict (op + kwargs)
    # whose strings may carry ${slot} refs like top-level fields.
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

# Canonical paths to try when init_ledger writes a wrong path (e.g. typo in
# /home/<user>/). App is detected from the spec path or the instruction's
# [bracket-prefix].
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
    """Heuristic: which app does this verify belong to."""
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
    """Expand a glob VM-side via bash; return first match or None.

    bash does ``~`` expansion and globbing on the VM. NEVER expanduser on
    the host — host $HOME is the dev machine, but the VM user is always
    ``user`` (OSWorld convention), so host expansion points at a path that
    doesn't exist on the VM.
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

    For when spec_path and APP_PATH_FALLBACKS all miss — e.g. init_ledger
    wrote a near-correct path (typo or profile-randomized subdir) and the
    file lives elsewhere under /home/user. Restricted to /home/user (not /)
    to keep find cheap and avoid matching same-named files under
    /usr/share/code or /opt/google/chrome.
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

    Paths pass through with ``~`` intact — the VM HTTP server expands it
    ($HOME = /home/user). NEVER expanduser on the host (host $HOME is the
    dev's, producing /home/<dev>/... invalid on the VM).

    Resolution order:
      1. Literal spec path (VM /file endpoint expands ~ then probes).
      2. Glob in spec (*) → VM-side bash glob.
      3. APP_PATH_FALLBACKS — pick a candidate whose basename matches spec's.
      4. Last-resort find /home/user -name <basename>.

    ``trace_attempts``, if given, gets one {"strategy","path","result"} entry
    per attempt for keynode_debug inspection.
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
    """Apply regex patterns to text under AND/OR + optional proximity."""
    try:
        compiled = [re.compile(p, re.MULTILINE) for p in patterns]
    except re.error:
        return False
    if proximity_lines is None:
        if all_must_match:
            return all(c.search(text) for c in compiled)
        return any(c.search(text) for c in compiled)
    # Proximity: all patterns must co-occur within a sliding window of
    # ``proximity_lines`` consecutive lines (e.g. 3 = same keybindings.json
    # entry's "key" + "command").
    lines = text.split("\n")
    n = len(lines)
    win = max(1, proximity_lines)
    for i in range(n):
        window = "\n".join(lines[i:i + win])
        if all(c.search(window) for c in compiled):
            return True
    return False


def run_file_grep(spec: VerifySpec, env: Any, instruction: str = "") -> Optional[bool]:
    """Wrapper around the traced version for callers that don't need the trace."""
    verdict, _ = _run_file_grep_traced(spec, env, instruction)
    return verdict


# --------------------------------------------------------------------------- #
# url_match
# --------------------------------------------------------------------------- #

# Browser address-bar a11y rows. Chrome/Chromium expose the active URL in one
# identifiable row, e.g.:
#   entry\tAddress and search bar (https://...)\tfocused,enabled
#   combo-box\tAddress and search bar (chrome://settings/...)\tfocused
#   entry\tURL (https://...)\tfocused,enabled     ← Firefox/chrome variants
#
# Require BOTH: tag is an editable-text-control kind, AND text has a
# URL-bearing label ("Address...bar"/"URL"/"Location") plus a parenthesized
# URL. The older "first https? row in the dump" heuristic false-positives
# when another app is active (Thunderbird email-body <link> rows) — see the
# 58565672 regression where url_match matched an email-body URL against itself.
_ADDRESS_BAR_TAGS = ("entry", "combo-box", "text-field", "textbox")
_ADDRESS_BAR_LABEL_RE = re.compile(
    r"(?:Address(?:\s+and\s+search)?\s+bar|^URL\b|^Location\b|Location bar)",
    re.IGNORECASE,
)
_ADDRESS_BAR_URL_RE = re.compile(r"\((?P<url>(?:https?|chrome|about|file)[^)\s]+)\)")
_BROWSER_PROTOCOL_PREFIXES = ("http://", "https://", "chrome://", "about:", "file://")


def _extract_active_url(a11y_text: str) -> Optional[str]:
    """Pull the active tab URL from a11y, strict address-bar scope only.

    Returns the URL only when a browser address-bar row is present, else None
    (NEVER falls back to "any http row" — that's the bug that lets Thunderbird
    email-body links satisfy a Chrome url_match). None routes to the LLM judge,
    which is far less likely to false-positive across apps.
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
        # Parenthesized form: 'Address... (https://...)'
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
    """Wrapper around the traced version. See ``_run_a11y_match_traced``."""
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
    """Like ``run_verify`` but also returns a debug trace dict:

      file_grep: {kind, spec_path, resolved_path, file_size_bytes,
                   per_pattern_matches, proximity_used, verdict_reason}
      url_match: {kind, url_extracted, regex, verdict_reason}
      a11y_match: {kind, matched_row, candidate_count, verdict_reason}

    Used by ``key_node_debug`` to persist per-step dispatcher decisions.
    Returns ``(verdict, trace_dict)``."""
    if spec is None or not spec.is_valid():
        return None, {"kind": "invalid", "verdict_reason": "spec is None or invalid"}

    # Substitute ``${name}`` refs with bound values, or short-circuit to
    # UNCERTAIN if any are unbound. No-op when the spec has no ``${...}`` refs.
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
    # Bracket-placeholder sweep — catches init-LLM specs like
    # ``[Planned remarks for Slide 1]`` / ``<value>`` / ``{{TODO}}`` that
    # expect framework fill-in (only ``${slot}`` is supported). Otherwise the
    # verifier greps that literal forever. Short-circuit to None → LLM judge.
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
    """Impress counterpart of _run_calc_verify_traced: build the python3 -c
    body from spec.impress_checks, then run via the shell-command runner."""
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
    """Wrap a structured-verify python body with the cli_run_uno runtime for
    ``domain`` (connect + setup + ops library). Domain is KNOWN here (no
    function-name scanning), and the wrapped script carries the sentinel so
    ``_maybe_wrap_calc_verify_command`` won't re-wrap it.

    First re-upload the domain's ops-lib module if missing: build_runtime_script
    emits a ``from _<domain>_ops_lib_<hash> import *`` shim when the process
    cache says it was uploaded, but an OSWorld snapshot reset wipes /tmp between
    tasks. The actor re-checks every predict(); the verify path used not to, so
    its shim imported a gone module → ModuleNotFoundError → every structured
    verify hard-failed. maybe_upload_ops_lib re-verifies + re-uploads; if it
    can't (no runner), get_uploaded_module returns None and the wrapper inlines
    the full library instead."""
    if env is not None:
        try:
            from mm_agents.structagent.actions.uno._ops_lib_remote import maybe_upload_ops_lib
            maybe_upload_ops_lib(env, domain)
        except Exception:
            pass
    from mm_agents.structagent.actions.uno.cli_run_uno_helpers import build_runtime_script
    return build_runtime_script(body, domain=domain)


def _run_writer_verify_traced(spec, env):
    """Writer counterpart of _run_calc_verify_traced: build the python3 -c body
    from spec.writer_checks, wrap with the Writer cli_run_uno boilerplate, then
    run via the shell-command runner."""
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
    """Build the python3 -c command from ``spec.calc_checks`` and run via the
    shell_command runner.

    The structured calc_checks list lets the init_ledger LLM skip writing
    Python verify bodies — the framework builds the function calls + PASS/FAIL
    print deterministically. _maybe_wrap_calc_verify_command then sees the
    ``verify_*`` refs and prepends the UNO runtime boilerplate.
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
    """Match patterns in a VM file. Deterministic text/regex with optional
    proximity + per-app path fallbacks; no LLM."""
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
        # File missing everywhere on the VM. For "write to file" outcomes this
        # is a CONFIRMED NOT-SATISFIED — a non-existent file can't hold the
        # patterns. Return False (not None) so the dispatcher trusts it and
        # doesn't fall back to the LLM, which transient UI cues (search-box
        # echoes, record-mode keystroke previews) can trick into a false PASS.
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
    # Content hash so OutcomeStateCache can tell if the file changed since the
    # first observation (feeds the planner's "file modified since start?" line).
    trace["file_content_hash"] = hashlib.sha256(content_bytes).hexdigest()[:16]
    # Content preview (first 800 chars) so the LLM diagnoser/planner can see
    # what the actor wrote vs what the verifier expects.
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
    # Per-pattern match record (which lines hit) for debug visibility.
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
    """True iff the verify code references a Calc-library function."""
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
    """True iff the verify code references an Impress-library function."""
    if not code:
        return False
    return any(fn in code for fn in _IMPRESS_LIB_FUNC_NAMES)


# Writer ops-library function names — keep in sync with writer_ops_lib.py.
_WRITER_LIB_FUNC_NAMES = (
    "op_paste_at",
    "verify_table_count", "verify_table_after_heading",
)


def _references_writer_ops_lib(code: str) -> bool:
    """True iff the verify code references a Writer-library function."""
    if not code:
        return False
    return any(fn in code for fn in _WRITER_LIB_FUNC_NAMES)


def _maybe_wrap_calc_verify_command(cmd):
    """If argv is ``['python3', '-c', '<code>']`` and the code references a
    Calc/Impress/Writer library function, replace the code with the wrapped
    runtime script (connect + domain setup + ops library + original body).

    Pass-through otherwise (non-python3, plain bash, file_grep, code that
    doesn't use the library).
    """
    if not isinstance(cmd, list) or len(cmd) != 3:
        return cmd
    if cmd[0] != "python3" or cmd[1] != "-c":
        return cmd
    code = cmd[2]
    if not isinstance(code, str):
        return cmd
    # Idempotency: structured verifies already wrap the body via
    # build_runtime_script for their known domain. Re-wrapping would nest and
    # break it — detect the sentinel and pass through.
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
    """Fix a ``pgrep -f <name>`` check that matches non-target processes,
    chiefly the agent's own machinery.

    ``pgrep -f`` matches the FULL command line, wrong for "is app X running":
    it self-matches the verifier's own ``bash -c '... pgrep -f X ...'``, and
    every cli_run_uno script is a ``python3 -c '<...>'`` whose source embeds
    the UNO connect boilerplate (``soffice --accept=…``,
    ``StarOffice.ComponentContext``), so ``pgrep -f soffice`` matches every
    in-flight calc/writer/impress op. (Observed: a "kill LibreOffice" verify
    reported 152 matches, 151 of them the agent's own scripts, never passing
    for 15 steps while real LibreOffice died at step 3.)

    The ``[s]office`` bracket trick doesn't help — it still matches the literal
    inside those bodies. Only matching the process NAME works: drop ``-f`` so
    pgrep matches the executable basename (``python3`` scripts have comm=python3
    and drop out). Keep ``-f`` when the pattern genuinely needs cmdline matching
    — a path, whitespace, or a script filename (whose host comm is the
    interpreter, not the script).
    """
    if not isinstance(cmd, list):
        return cmd

    def _needs_f(pat: str) -> bool:
        # Needs full-cmdline matching when not a bare process name:
        # a path, whitespace, or a script file.
        return ("/" in pat or any(c.isspace() for c in pat)
                or bool(_SCRIPT_EXT_RE.search(pat)))

    out = list(cmd)
    # argv form: ["pgrep", "-f", PAT, ...] → drop "-f" for a bare name
    for i in range(len(out) - 2):
        if (out[i] == "pgrep" and out[i + 1] == "-f"
                and isinstance(out[i + 2], str) and out[i + 2]
                and not _needs_f(out[i + 2])):
            del out[i + 1]
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
    """Run ``spec.command`` (argv) on the VM via run_bash_script and check it.

    Verdict True iff: command finished (status="success"), exit code matches
    ``spec.expected_exit_code`` (default 0), at least one ``expected_substring``
    (case-insensitive) is in stdout, and no ``forbidden_substring`` is.

    Returns ``(None, trace)`` when env has no executor (dispatcher then falls
    back to the LLM judge)."""
    # Pre-wrap an LO-lib python3 -c body with the cli_run_uno boilerplate so the
    # library functions exist in the subprocess (see _maybe_wrap_calc_verify_command).
    wrapped_cmd = _maybe_wrap_calc_verify_command(list(spec.command or []))
    # Defuse a self-matching ``pgrep -f <name>`` check.
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

    # shlex-quote each argv element, redirect stderr→stdout, and echo a
    # trailing ``__EXIT__ $?`` marker to recover the exit code from the
    # runner's combined stdout.
    #
    # Post-failure diagnostic block: on exit != 0, a tiny read-only probe
    # (pwd/echo/ls on the last positional argv token) gives the downstream LLM
    # diagnoser enough context to find root cause itself — meaningful stderr
    # ("No such file or directory") for path commands, harmless noise otherwise.
    #
    # [FIX shell_verify_tilde] LLM verify specs often use ``~``/``~/path`` in
    # argv, but element-by-element shlex.quote single-quotes each path where
    # bash does NOT expand tilde → the cmd asserts on a literal ``~/...`` that
    # doesn't exist → outcome pending forever → infinite REPLAN. Worse, a
    # "file exists?" verify then reports MISSING, the planner recreates the
    # file from scratch and OVERWRITES the real artifact (3680a5ee — fake text
    # into binary xlsx/ods). Fix: expand a leading ``~``/``~/`` to the absolute
    # home before quoting. OSWorld VMs are always /home/user, hardcoded rather
    # than a synchronous controller probe per verify call.
    # To roll back: revert to the plain ``shlex.quote(str(a))`` one-liner.
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
    # DIAG block: for path-style argv (``["test","-f","/path"]``), on non-zero
    # exit echo pwd + ls the last token to surface "no such file or directory".
    # But for argv embedding CODE in the last token (every calc_verify's
    # ``["python3","-c","<36KB UNO wrapper>"]``) it inlines that code 3 more
    # times (echo+ls+dirname) → ~144KB bash payload, which empirically triggers
    # 110s HTTP timeouts on the env /execute path. So suppress DIAG when the
    # last token is code-like (multi-line or too large to be a path).
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

    # Verbose trace logging so a 110s HTTP hang (or any failure) leaves a
    # self-describing audit trail in runtime.log: pre-call argv/script
    # head+tail+bytes+kind+start; on success exit/elapsed/stdout; on raise or
    # non-success status, exception/status + elapsed + script dumped to /tmp
    # (helps attribute HTTP timeouts to specific verifies).
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
        """Dump the failing script to /tmp/_verify_fail_<ts>.sh for post-hoc
        inspection. Returns the path, or None if the dump fails."""
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
    # Strip the trailing ``__EXIT__ <N>`` marker; treat missing as exit 0.
    exit_code: Optional[int] = 0
    stdout = raw_out
    m = re.search(r"^__EXIT__ (\d+)\s*$", raw_out, re.MULTILINE)
    if m:
        try:
            exit_code = int(m.group(1))
        except Exception:
            exit_code = 0
        stdout = raw_out[:m.start()].rstrip("\n")
        # Parse the post-failure diagnostic block (present only on exit != 0).
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
    # Forbidden first — any hit is an instant FAIL even if expected also hits.
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
