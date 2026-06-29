"""LLM-driven JSON repair helper.

When a prompt asks an LLM for structured JSON output, the model can
emit responses that are *almost* correct but fail strict parsing —
mis-nested keys (e.g. a key-value pair placed inside a JSON array),
trailing commas, unbalanced braces, half-escaped string contents,
etc.  A targeted second LLM call to "rewrite this into valid JSON
preserving every value" is more robust than ad-hoc regex fixups
because the model handles arbitrary shape mistakes.

Usage:

    from mm_agents.structagent.ledger.support.json_repair import repair_json_via_llm
    fixed = repair_json_via_llm(broken_text, call_llm=call_llm, model=model)
    if fixed is not None:
        parsed = json.loads(fixed)

This module has NO domain-specific knowledge; it just instructs the
model to reshape malformed text into valid JSON. The caller decides
how to interpret the result.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


_REPAIR_SYSTEM_PROMPT = (
    "You are a JSON repair tool. Given a piece of text that was "
    "INTENDED to be valid JSON but has structural errors "
    "(mis-nested keys, mismatched brackets, trailing commas, "
    "duplicate keys inside an array, missing/extra commas, etc.), "
    "return ONLY a valid JSON object that PRESERVES every key, "
    "value, and nested structure from the input.\n\n"
    "Rules:\n"
    "1. Output ONLY the repaired JSON object. No markdown, no prose, "
    "   no explanation, no code fences.\n"
    "2. Do NOT invent new keys or values. Do NOT drop any input data. "
    "   Do NOT duplicate a key — if the same key appears both inside "
    "   and outside an array, keep exactly ONE copy in the right place.\n"
    "3. The most common mistake: ONE OR MORE `\"<key>\": <value>` "
    "   pairs placed inside a JSON ARRAY where only ARRAY ELEMENTS "
    "   are valid. Close the array right before the FIRST mis-nested "
    "   key and lift EVERY mis-nested key out as siblings of the "
    "   array (in order).\n"
    "4. A string value with an unescaped inner double-quote. Escape "
    "   the inner quote.\n"
    "5. Preserve every character of inner string values verbatim "
    "   (including escape sequences like \\n).\n\n"
    "Concrete example of the array-misnesting fix:\n\n"
    "INPUT (broken):\n"
    "  { \"verify\": { \"command\": [\n"
    "      \"python3\", \"-c\", \"print('PASS')\",\n"
    "      \"expected_substring\": [\"PASS\"],\n"
    "      \"expected_exit_code\": 0\n"
    "    ] } }\n\n"
    "OUTPUT (repaired — both mis-nested keys lifted, array properly "
    "closed, NO duplicates):\n"
    "  { \"verify\": { \"command\": [\n"
    "      \"python3\", \"-c\", \"print('PASS')\"\n"
    "    ],\n"
    "    \"expected_substring\": [\"PASS\"],\n"
    "    \"expected_exit_code\": 0\n"
    "  } }\n"
)


def _strip_fences(text: str) -> str:
    """Strip ```...``` and ```json``` fences, plus any <think>...</think>
    prefix some thinking-mode models emit before the actual output."""
    s = text.strip()
    end = s.rfind("</think>")
    if end >= 0:
        s = s[end + len("</think>"):].lstrip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


_BAD_KEYS = ('"expected_substring"', '"expected_exit_code"',
             '"forbidden_substring"')


def _scan_for_misnesting(s: str, arr_start: int
                         ) -> Optional[tuple]:
    """Walk forward from `arr_start` (the `[` opening a `command` JSON
    array). Track bracket / string state. Return:

      None  — no problem detected (matching `]` found before any
              mis-nested key)
      (bad_pos, end_pos)  — `bad_pos` is the position of the first
              `"<bad-key>":` token found AT depth==1 inside the array,
              and `end_pos` is the position of the closing `]` (if the
              array was actually closed) OR the next-higher closer
              like `}` (when the model forgot the `]` entirely)."""
    depth = 1
    i = arr_start + 1
    in_str = False
    esc = False
    bad_pos: Optional[int] = None
    while i < len(s):
        ch = s[i]
        if esc:
            esc = False
        elif in_str:
            if ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                # Check if this is a mis-nested key BEFORE we treat it
                # as a string opener (so the key detection doesn't get
                # swallowed by string-mode).
                if bad_pos is None and depth == 1:
                    for bk in _BAD_KEYS:
                        if s[i:i + len(bk)] == bk:
                            j = i + len(bk)
                            while j < len(s) and s[j] in ' \t\n\r':
                                j += 1
                            if j < len(s) and s[j] == ':':
                                bad_pos = i
                                break
                in_str = True
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    # Array closed properly. Return if we saw a bad key
                    # somewhere inside; else signal no problem.
                    return (bad_pos, i) if bad_pos is not None else None
            elif ch == '}':
                # Array was never closed — the model used `}` to close
                # the parent (`verify` / outcome) object directly. If a
                # mis-nested key was seen, this `}` is where the
                # repair stops adding lifted keys.
                if bad_pos is not None:
                    return (bad_pos, i)
                # No bad key seen and we already hit `}` — malformed
                # but not the misnesting pattern this fast-path fixes.
                return None
        i += 1
    return None


def repair_shell_command_misnesting(broken: str) -> str:
    """Deterministic fast-path for the most common LLM JSON mistake:
    `expected_substring` / `expected_exit_code` / `forbidden_substring`
    placed INSIDE the `command` JSON array, with the array EITHER
    closed by a misplaced `]` after the keys OR not closed at all
    (the model jumps straight to the parent object's `}`).

    Walks the string with a tiny state machine that handles both
    shapes — closes the array right before the first mis-nested key,
    lifts the inner key-value pairs out as siblings of `command`, and
    (when the original `]` was misplaced) drops it. No-op when no
    misnesting is detected."""
    out_parts: List[str] = []
    cursor = 0
    needle = '"command"'
    while True:
        i = broken.find(needle, cursor)
        if i < 0:
            out_parts.append(broken[cursor:])
            break
        # Validate `"command"` is followed by `:` then `[` (skipping ws).
        j = i + len(needle)
        while j < len(broken) and broken[j] in ' \t\n\r':
            j += 1
        if j >= len(broken) or broken[j] != ':':
            out_parts.append(broken[cursor:i + 1])
            cursor = i + 1
            continue
        j += 1
        while j < len(broken) and broken[j] in ' \t\n\r':
            j += 1
        if j >= len(broken) or broken[j] != '[':
            out_parts.append(broken[cursor:i + 1])
            cursor = i + 1
            continue
        arr_start = j
        scan = _scan_for_misnesting(broken, arr_start)
        if scan is None:
            # No misnesting — advance past `"command"` and let next loop iteration
            # find the next occurrence (the scanner consumed up to the array close
            # but we don't bother re-emitting here; just step forward conservatively).
            out_parts.append(broken[cursor:arr_start + 1])
            cursor = arr_start + 1
            continue
        bad_pos, end_pos = scan
        # Walk back from bad_pos to drop the trailing comma+ws that joined
        # it to the previous array element.
        k = bad_pos - 1
        while k > arr_start and broken[k] in ' \t\n\r':
            k -= 1
        if k > arr_start and broken[k] == ',':
            head_end = k
        else:
            head_end = bad_pos
        # `end_pos` always points to a misplaced closer:
        #   `]` — model added a misplaced `]` after lifted keys (the real
        #         `}}` for verify/outcome follow it).
        #   `}` — model wrote `}` in place of the array's `]`, with the
        #         real `}}` for verify/outcome still following.
        # In both cases the char at `end_pos` is structurally wrong after
        # our lift, so drop it and let the model's remaining closers
        # match the actual nesting.
        out_parts.append(broken[cursor:arr_start + 1])
        out_parts.append(broken[arr_start + 1:head_end])
        out_parts.append("], ")
        out_parts.append(broken[bad_pos:end_pos])
        cursor = end_pos + 1
    return "".join(out_parts)


def repair_json_via_llm(broken: str,
                        *,
                        call_llm: Callable[..., str],
                        model: str,
                        max_tokens: int = 3000,
                        max_attempts: int = 2,
                        logger: Optional[logging.Logger] = None) -> Optional[str]:
    """Ask the LLM to repair `broken` into valid JSON. First applies the
    deterministic fast-path (`repair_shell_command_misnesting`) — that
    handles the most common LLM JSON shape error without burning an
    extra LLM call. Falls back to an LLM-driven repair pass for other
    failure modes. Returns the repaired JSON string (parseable by
    ``json.loads``) or None if every attempt failed."""
    log = logger or logging.getLogger(__name__)
    if not broken or not broken.strip():
        return None
    # Fast-path: deterministic regex fix for the most common mistake.
    candidate = repair_shell_command_misnesting(broken)
    if candidate != broken:
        try:
            json.loads(candidate)
        except Exception as e:
            log.info("[json_repair] regex pre-fix produced invalid JSON: %s", e)
        else:
            log.info("[json_repair] regex pre-fix succeeded (orig=%d → fixed=%d chars)",
                     len(broken), len(candidate))
            return candidate
    # Fallback: ask the LLM to repair.
    for attempt in range(1, max_attempts + 1):
        try:
            raw = call_llm(
                {"model": model,
                 "messages": [
                     {"role": "system", "content": _REPAIR_SYSTEM_PROMPT},
                     {"role": "user", "content": broken},
                 ],
                 "max_tokens": max_tokens,
                 "temperature": 0.0,
                 "top_p": 0.9},
                model,
            ) or ""
        except Exception as e:
            log.info("[json_repair] LLM call failed (attempt %d): %s", attempt, e)
            continue
        candidate = _strip_fences(raw)
        try:
            json.loads(candidate)
        except Exception as e:
            log.info("[json_repair] still invalid after attempt %d: %s", attempt, e)
            continue
        log.info("[json_repair] repaired on attempt %d (orig=%d → fixed=%d chars)",
                 attempt, len(broken), len(candidate))
        return candidate
    return None
