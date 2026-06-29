"""Abstract task-instance leakage out of the mined memory bank.

L1/L2/verifier entries mined from a single trajectory carry literal task
arguments verbatim (e.g. a subgoal "Search for Women's Nike Jerseys using store
search"). That leaks eval-task content into the agent's memory and overfits the
entry to one task. This tool rewrites the free-text fields, replacing instance
VALUES (products / queries / file names / URLs / emails / sheet values / brand
names) with natural placeholders while preserving the reusable procedure (which
control to act on, action type/order, formula shapes, app menu/dialog names).

  "Search for Women's Nike Jerseys using store search"
      -> "Search for a target product using the store search bar"
  applicable_task_classes: ["nba_store_browse_filtered_products"]
      -> ["store_browse_filtered_products"]

Placeholders are natural ("the product query"), not slot tokens ("{product}").

Scope: L1 + L2 + verifier intent recipes, across both the v3 (live) and v2
(fallback) banks. L3a/L3c rule sheets are already general. After cleaning v3,
rebuild the FAISS indexes so retrieval embeds the abstracted text:

    PYTHONPATH=. python -m mm_agents.structagent.memory.offline.build_indexes.build_multi_layer_indexes

USAGE
-----
  # scan only — counts + examples, no LLM, no writes
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.clean_leakage --dry-run [--banks v3,v2]
  # rewrite N flagged entries, print before/after (needs key)
  OPENROUTER_API_KEY=... PYTHONPATH=. python -m mm_agents.structagent.memory.offline.clean_leakage --sample 8
  # full clean — backup + rewrite all flagged + change report (needs key)
  OPENROUTER_API_KEY=... PYTHONPATH=. python -m mm_agents.structagent.memory.offline.clean_leakage --apply

The rule lives in ABSTRACTION_RULE so the mining prompts import the exact same
wording (prevents re-leakage on re-mining).
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from mm_agents.structagent._paths import REPO_ROOT

MODEL = os.environ.get("VERIFIER_POLISH_MODEL", "anthropic/claude-sonnet-4-5")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


# Single source of truth for "what to abstract" (also imported by mining prompts).
ABSTRACTION_RULE = """\
This entry is a REUSABLE skill, mined from agent trajectories. It must generalize
to ANY similar task, not the specific task instance it was mined from.

REMOVE / GENERALIZE — task-instance specifics (a different task would fill these
differently). Replace each with a natural, general placeholder phrase:
  • specific product names, search queries, titles, brand or site names tied to
    the instance  ("Women's Nike Jerseys" -> "the target product"; "NBA Store"
    -> "the store site")
  • specific file names, paths, sheet/cell values, URLs, email addresses, person
    names, dates, and numeric arguments of THIS task
  • anything a user could have typed differently for a different task

PRESERVE — what makes the skill reusable:
  • which app-native UI control / widget to act on (the search bar, the three-dot
    menu, a "Delete browsing data…" menu item, a "Save As" dialog) — real control
    labels stay
  • the action type and order (click, type, press Enter, drag)
  • structural patterns: a formula SHAPE like =SUM(<range>) but not the specific
    range; menu paths; dialog names that are part of the app
  • generic categories: "a target product", "the source file", "the search query"

STYLE: natural placeholders, NOT slot tokens. Write "type the product query", NOT
"type {product}". The text should read like a general instruction.
EXCEPTION — inside a shell command, code, or a formula, use <angle_bracket>
placeholders so the structure stays syntactically valid:
  `python3 <script_path> > <log_path> 2>&1`,  `cat <script_path>`,  =SUM(<range>)
(prose stays natural; only the command/code/formula uses <…>).
"""

_SYSTEM = f"""You rewrite fields of a computer-use skill-memory entry to remove task-instance leakage.

{ABSTRACTION_RULE}

You are given a JSON object of fields to review. Return a JSON object with EXACTLY
the same keys. For each field:
  • if it contains task-instance specifics, return the rewritten (abstracted) text
  • if it is already general, return it UNCHANGED, verbatim

IDENTIFIER keys (name ends in "_id" — e.g. action_id, skill_id, intent_id) are
slugs, not prose: return an abstracted snake_case identifier naming the GENERAL
skill/intent with NO site / brand / app-instance / task-value terms. Keep it
descriptive and distinctive.
  "accuweather_monthly_forecast_city"  ->  "check_monthly_weather_forecast"
  "apple_iphone_compare_models"        ->  "compare_product_models"
  "search_product_in_store_search_bar" ->  "search_product_in_store_search_bar" (already general)
  "gimp_fill_layer_with_color"         ->  "gimp_fill_layer_with_color" (app name OK, no instance)
App names (gimp, chrome, libreoffice…) may stay in an id; site/brand/product names may not.

Preserve list structure and length. Do not add commentary. Return ONLY the JSON object."""


# ── Leak prefilter ───────────────────────────────────────────────────────────
# leak_level(text) -> 2 strong / 1 weak / 0 none.
#   strong = literal value almost certainly task-specific (quoted value, file,
#            email, URL-with-path, big number, multi-word proper noun).
#   weak   = single capitalized non-app/UI token (e.g. "Amazon") — often a leak,
#            lower precision; reviewed in --apply, reported separately in --dry-run.
# A false positive costs one LLM call that returns the text unchanged.
_FILE_RE = re.compile(r"\b[\w\-]+\.(?:xlsx|xls|docx|doc|pptx|ppt|csv|tsv|pdf|png|jpe?g|gif|svg|txt|md|py|js|ts|json|ya?ml|html?|xml|zip|tar|gz|mp4|mp3|wav|odt|ods|odp)\b", re.I)
_URL_RE = re.compile(r"https?://\S{3,}|www\.\S{3,}")
_EMAIL_RE = re.compile(r"\b[\w.\-]+@[\w.\-]+\.\w+\b")
_QUOTED_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,})[\"'“”‘’]")
_NUM_RE = re.compile(r"[$€£]\s?\d|\b\d{4,}\b|\b\d{1,3}(?:,\d{3})+\b")  # money / big / grouped
_CAPTOK_RE = re.compile(r"[A-Z][A-Za-z'\-]{2,}")
_MULTI_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)+\b")

# App / UI / generic-noun / imperative-verb tokens that legitimately belong to a
# reusable procedure (not leaks). Brand/site names (amazon, nike, reddit, …) are
# deliberately ABSENT so they flag.
_WHITELIST = {
    # apps / formats / acronyms
    "chrome", "firefox", "libreoffice", "calc", "writer", "impress", "gimp", "vlc",
    "thunderbird", "code", "terminal", "nautilus", "gnome", "ubuntu", "linux",
    "pdf", "csv", "xlsx", "docx", "pptx", "html", "css", "json", "xml", "url",
    "ui", "ok", "id", "api", "gui", "cli", "usb", "png", "jpg", "rgb", "hex",
    "sql", "todo", "fixme", "ctrl", "shift", "alt", "esc",
    # imperative verbs (capitalized only by sentence position)
    "launch", "open", "click", "press", "type", "select", "choose", "enter",
    "set", "use", "drag", "drop", "navigate", "wait", "verify", "check", "close",
    "switch", "run", "add", "insert", "delete", "remove", "copy", "paste", "cut",
    "move", "scroll", "hover", "fill", "save", "create", "go", "name", "rename",
    "change", "modify", "edit", "apply", "confirm", "cancel", "submit", "load",
    "find", "replace", "search", "filter", "sort", "export", "import", "download",
    "upload", "right", "left", "double", "triple", "single",
    # generic UI nouns
    "file", "view", "format", "tools", "window", "help", "data", "slide", "table",
    "sheet", "layer", "return", "tab", "menu", "toolbar", "sidebar", "panel",
    "dialog", "button", "checkbox", "dropdown", "field", "bar", "box", "icon",
    "page", "row", "column", "cell", "range", "color", "colour", "text", "font",
    "size", "style", "shape", "chart", "value", "title", "header", "footer",
    "link", "item", "option", "area", "section", "toolbox", "swatch", "spectrum",
    "entry", "list", "image", "foreground", "background", "bucket", "preferences",
    "settings", "properties", "dialog", "tab", "key", "enter",
    # more verbs / control words that get Title-cased in menus & button labels
    "sign", "log", "show", "hide", "toggle", "expand", "collapse", "zoom", "pick",
    "tick", "mark", "clear", "reset", "refresh", "reload", "back", "forward",
    "send", "reply", "compose", "attach", "print", "align", "merge", "group",
    "rotate", "crop", "resize", "scale", "flatten", "render", "play", "pause",
    "stop", "mute", "browse", "upload", "picture", "never", "always", "allow",
    "block", "accept", "decline", "dismiss", "more", "less", "new", "all", "none",
    "other", "default", "custom", "advanced", "general", "options", "preview",
    "ok", "done", "next", "back", "finish", "start", "stop", "skip", "retry",
    # function words Title-cased in menus ("Open With", "Save As")
    "with", "as", "to", "from", "of", "in", "on", "for", "and", "or", "the", "a",
    "an", "your", "this", "that", "by", "at", "into", "out", "up", "down", "off",
}


def _strip_leading_cap(text: str) -> str:
    """Lowercase the first word so a sentence-initial verb ("Launch …") doesn't
    masquerade as a proper noun."""
    return re.sub(r"^(\W*)([A-Z][a-z]+)", lambda m: m.group(1) + m.group(2).lower(), text, count=1)


def _norm(tok: str) -> str:
    return tok.strip("'’‘\"“”").lower()


def _nonwhite_caps(text: str) -> List[str]:
    return [t for t in _CAPTOK_RE.findall(text) if _norm(t) and _norm(t) not in _WHITELIST]


# A quote right after these verbs is a typed VALUE, not a UI label.
_TYPING_CTX = re.compile(
    r"\b(typ\w*|enter|input|search(?:\s+for)?|nam\w*|titl\w*|call\w*|quer\w*|writ\w*|paste|fill\s+in|label(?:l?ed)?)\W*$",
    re.I)


def _quoted_leaky(text: str) -> bool:
    """A quoted span leaks only if it's a typed VALUE (typing-verb context) or a
    multi-word phrase with a non-UI proper noun. Bare UI labels ('Save',
    'Change Foreground Color') are not leaks."""
    for m in _QUOTED_RE.finditer(text):
        content = m.group(1).strip()
        if _TYPING_CTX.search(text[max(0, m.start() - 20):m.start()]):
            return True
        caps = _CAPTOK_RE.findall(content)
        if len(content.split()) >= 2 and caps and any(_norm(w) not in _WHITELIST for w in caps):
            return True
    return False


def _force_review(key: str) -> bool:
    """Fields that DEFINE the task instance — always review, since lowercase
    instance values ('green', 'best pizza') evade the regex. Identifier fields
    are forced too (the LLM leaves clean slugs unchanged)."""
    k = key.lower()
    return ("subgoal" in k) or k.endswith("task_class") or k.endswith("_id") or k == "id"


def leak_level(text: str, key: str = "") -> int:
    if not text or not isinstance(text, str):
        return 0
    if _force_review(key):
        return 2
    if _FILE_RE.search(text) or _URL_RE.search(text) or _EMAIL_RE.search(text) \
            or _NUM_RE.search(text) or _quoted_leaky(text):
        return 2
    body = _strip_leading_cap(text)
    for m in _MULTI_RE.finditer(body):                      # multi-word proper noun
        if not all(_norm(w) in _WHITELIST for w in m.group(0).split()):
            return 2
    if _nonwhite_caps(body):                                # single proper token
        return 1
    return 0


def looks_leaky(text: str) -> bool:
    return leak_level(text) >= 1


# ── In-scope field extraction / application, per bank layer ─────────────────
# A handler turns one parsed entry into a flat {key: text} dict of in-scope
# strings, and writes a {key: text} dict back into the same structure.

@dataclass
class Target:
    name: str                                   # e.g. "v3 L1"
    glob: str                                    # relative to REPO_ROOT
    extract: Callable[[dict], Dict[str, str]]    # entry -> {flat_key: text}
    apply: Callable[[dict, Dict[str, str]], None]  # entry, {flat_key: text} -> mutate


def _flatten(prefix: str, val: Any, out: Dict[str, str]) -> None:
    """Collect strings from str / list[str] / list[dict-of-str] into out[key]."""
    if isinstance(val, str):
        if val.strip():
            out[prefix] = val
    elif isinstance(val, list):
        for i, v in enumerate(val):
            _flatten(f"{prefix}[{i}]", v, out)


def _unflatten_into(prefix: str, val: Any, flat: Dict[str, str]) -> Any:
    """Rebuild val with replacements from flat (mirror of _flatten)."""
    if isinstance(val, str):
        return flat.get(prefix, val)
    if isinstance(val, list):
        return [_unflatten_into(f"{prefix}[{i}]", v, flat) for i, v in enumerate(val)]
    return val


# v2 L1  (entry = d["action"])
def _v2l1_extract(d: dict) -> Dict[str, str]:
    a = d.get("action", d); out: Dict[str, str] = {}
    _flatten("action_id", a.get("action_id"), out)
    _flatten("subgoal", a.get("subgoal"), out)
    _flatten("when_to_use", a.get("when_to_use"), out)
    for i, t in enumerate(a.get("typical_actions") or []):
        _flatten(f"typical_actions[{i}].action", t.get("action"), out)
        _flatten(f"typical_actions[{i}].notes", t.get("notes"), out)
    _flatten("gotchas", a.get("gotchas"), out)
    _flatten("applicable_task_classes", a.get("applicable_task_classes"), out)
    return out

def _v2l1_apply(d: dict, flat: Dict[str, str]) -> None:
    a = d.get("action", d)
    if "action_id" in a: a["action_id"] = flat.get("action_id", a["action_id"])
    if "subgoal" in a: a["subgoal"] = flat.get("subgoal", a["subgoal"])
    if "when_to_use" in a: a["when_to_use"] = flat.get("when_to_use", a["when_to_use"])
    for i, t in enumerate(a.get("typical_actions") or []):
        if "action" in t: t["action"] = flat.get(f"typical_actions[{i}].action", t["action"])
        if "notes" in t: t["notes"] = flat.get(f"typical_actions[{i}].notes", t["notes"])
    if "gotchas" in a: a["gotchas"] = _unflatten_into("gotchas", a["gotchas"], flat)
    if "applicable_task_classes" in a:
        a["applicable_task_classes"] = _unflatten_into("applicable_task_classes", a["applicable_task_classes"], flat)


# v3 L1  (entry top-level has cluster_subgoal_pattern + actions[])
def _v3l1_extract(d: dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    _flatten("cluster_subgoal_pattern", d.get("cluster_subgoal_pattern"), out)
    for i, a in enumerate(d.get("actions") or []):
        _flatten(f"actions[{i}].action_id", a.get("action_id"), out)
        _flatten(f"actions[{i}].approach", a.get("approach"), out)
        _flatten(f"actions[{i}].when_to_use", a.get("when_to_use"), out)
        _flatten(f"actions[{i}].typical_actions", a.get("typical_actions"), out)
        _flatten(f"actions[{i}].common_pitfalls", a.get("common_pitfalls"), out)
        _flatten(f"actions[{i}].tools_or_widgets_seen", a.get("tools_or_widgets_seen"), out)
    return out

def _v3l1_apply(d: dict, flat: Dict[str, str]) -> None:
    if "cluster_subgoal_pattern" in d:
        d["cluster_subgoal_pattern"] = flat.get("cluster_subgoal_pattern", d["cluster_subgoal_pattern"])
    for i, a in enumerate(d.get("actions") or []):
        for f in ("action_id", "approach", "when_to_use"):
            if f in a: a[f] = flat.get(f"actions[{i}].{f}", a[f])
        for f in ("typical_actions", "common_pitfalls", "tools_or_widgets_seen"):
            if f in a: a[f] = _unflatten_into(f"actions[{i}].{f}", a[f], flat)


# L2 skill  (v2 entry = d["skill"]; v3 entry = d directly or d["skill"])
def _l2_obj(d: dict) -> dict:
    return d.get("skill", d)

def _l2_extract(d: dict) -> Dict[str, str]:
    s = _l2_obj(d); out: Dict[str, str] = {}
    _flatten("skill_id", s.get("skill_id"), out)
    _flatten("task_class", s.get("task_class"), out)
    _flatten("when_to_use", s.get("when_to_use"), out)
    for i, st in enumerate(s.get("plan_template") or []):
        _flatten(f"plan[{i}].subgoal", st.get("subgoal"), out)
        _flatten(f"plan[{i}].typical_actions", st.get("typical_actions"), out)
    for i, p in enumerate(s.get("attached_pitfalls") or []):
        _flatten(f"pitfall[{i}].pitfall", p.get("pitfall"), out)
        _flatten(f"pitfall[{i}].why", p.get("why"), out)
    return out

def _l2_apply(d: dict, flat: Dict[str, str]) -> None:
    s = _l2_obj(d)
    if "skill_id" in s: s["skill_id"] = flat.get("skill_id", s["skill_id"])
    if "task_class" in s: s["task_class"] = flat.get("task_class", s["task_class"])
    if "when_to_use" in s: s["when_to_use"] = flat.get("when_to_use", s["when_to_use"])
    for i, st in enumerate(s.get("plan_template") or []):
        if "subgoal" in st: st["subgoal"] = flat.get(f"plan[{i}].subgoal", st["subgoal"])
        if "typical_actions" in st:
            st["typical_actions"] = _unflatten_into(f"plan[{i}].typical_actions", st["typical_actions"], flat)
    for i, p in enumerate(s.get("attached_pitfalls") or []):
        if "pitfall" in p: p["pitfall"] = flat.get(f"pitfall[{i}].pitfall", p["pitfall"])
        if "why" in p: p["why"] = flat.get(f"pitfall[{i}].why", p["why"])


# v3 L2 skill  (file has a skills[] list)
def _v3l2_extract(d: dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, s in enumerate(d.get("skills") or []):
        _flatten(f"skills[{i}].skill_id", s.get("skill_id"), out)
        _flatten(f"skills[{i}].task_class", s.get("task_class"), out)
        _flatten(f"skills[{i}].when_to_use", s.get("when_to_use"), out)
        for j, st in enumerate(s.get("plan_template") or []):
            _flatten(f"skills[{i}].plan[{j}].subgoal", st.get("subgoal"), out)
            _flatten(f"skills[{i}].plan[{j}].typical_actions", st.get("typical_actions"), out)
        for j, p in enumerate(s.get("attached_pitfalls") or []):
            _flatten(f"skills[{i}].pitfall[{j}].pitfall", p.get("pitfall"), out)
            _flatten(f"skills[{i}].pitfall[{j}].why", p.get("why"), out)
    return out

def _v3l2_apply(d: dict, flat: Dict[str, str]) -> None:
    for i, s in enumerate(d.get("skills") or []):
        if "skill_id" in s: s["skill_id"] = flat.get(f"skills[{i}].skill_id", s["skill_id"])
        if "task_class" in s: s["task_class"] = flat.get(f"skills[{i}].task_class", s["task_class"])
        if "when_to_use" in s: s["when_to_use"] = flat.get(f"skills[{i}].when_to_use", s["when_to_use"])
        for j, st in enumerate(s.get("plan_template") or []):
            if "subgoal" in st: st["subgoal"] = flat.get(f"skills[{i}].plan[{j}].subgoal", st["subgoal"])
            if "typical_actions" in st:
                st["typical_actions"] = _unflatten_into(f"skills[{i}].plan[{j}].typical_actions", st["typical_actions"], flat)
        for j, p in enumerate(s.get("attached_pitfalls") or []):
            if "pitfall" in p: p["pitfall"] = flat.get(f"skills[{i}].pitfall[{j}].pitfall", p["pitfall"])
            if "why" in p: p["why"] = flat.get(f"skills[{i}].pitfall[{j}].why", p["why"])


# verifier intent recipe  (entry = d["recipe"])
def _rec_extract(d: dict) -> Dict[str, str]:
    r = d.get("recipe", d); out: Dict[str, str] = {}
    _flatten("intent_id", r.get("intent_id"), out)
    _flatten("when_to_use", r.get("when_to_use"), out)
    for i, dim in enumerate(r.get("verification_dimensions") or []):
        _flatten(f"dim[{i}].description", dim.get("description"), out)
        # literal expected values inside a check are both leakage and a correctness bug
        for fld in ("expected", "expected_value", "value", "check", "how_to_check", "evidence"):
            _flatten(f"dim[{i}].{fld}", dim.get(fld), out)
    return out

def _rec_apply(d: dict, flat: Dict[str, str]) -> None:
    r = d.get("recipe", d)
    if "intent_id" in r: r["intent_id"] = flat.get("intent_id", r["intent_id"])
    if "when_to_use" in r: r["when_to_use"] = flat.get("when_to_use", r["when_to_use"])
    for i, dim in enumerate(r.get("verification_dimensions") or []):
        if "description" in dim:
            dim["description"] = flat.get(f"dim[{i}].description", dim["description"])
        for fld in ("expected", "expected_value", "value", "check", "how_to_check", "evidence"):
            if fld in dim:
                dim[fld] = _unflatten_into(f"dim[{i}].{fld}", dim[fld], flat)


BANKS: Dict[str, List[Target]] = {
    "v3": [
        Target("v3 L1", "results/unified_memory/_l1_actions_v3/*/cluster_*.json", _v3l1_extract, _v3l1_apply),
        Target("v3 L2", "results/unified_memory/_l2_skills_v3/cluster_*.json", _v3l2_extract, _v3l2_apply),
    ],
    "v2": [
        Target("v2 L1", "results/planner_experience/_l1_actions_v2/*.json", _v2l1_extract, _v2l1_apply),
        Target("v2 L1b", "results/planner_experience/_l1_actions/*.json", _v2l1_extract, _v2l1_apply),
        Target("v2 L2", "results/planner_experience/_l2_skills_v2/*.json", _l2_extract, _l2_apply),
        Target("verifier", "results/successful_ledgers/_intent_recipes_v2/*.json", _rec_extract, _rec_apply),
    ],
}


def _iter_files(t: Target) -> List[Path]:
    return [p for p in REPO_ROOT.glob(t.glob) if "/_retrieval/" not in str(p) and p.is_file()]


# ── LLM rewrite ─────────────────────────────────────────────────────────────
def _parse_obj(txt: str) -> dict:
    """Pull a JSON object out of an LLM reply (handles <think> blocks, ```json
    fences, surrounding prose, single-quoted dicts)."""
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL | re.I)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.DOTALL)
    if m:
        for parse in (json.loads, ast.literal_eval):
            try:
                return parse(m.group(1))
            except Exception:
                pass
    # balanced-brace scan from each '{'
    start = txt.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(txt)):
            if txt[i] == "{":
                depth += 1
            elif txt[i] == "}":
                depth -= 1
                if depth == 0:
                    cand = txt[start:i + 1]
                    for parse in (json.loads, ast.literal_eval):
                        try:
                            return parse(cand)
                        except Exception:
                            pass
                    break
        start = txt.find("{", start + 1)
    raise ValueError(f"no JSON object in reply: {txt[:200]!r}")


def _rewrite(flat_leaky: Dict[str, str], model: str, base_url: str, api_key: str) -> Dict[str, str]:
    import openai
    client = openai.OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    # Sanitize keys: brackets/dots in JSON keys (e.g. "actions[0].typical_actions[1]")
    # make some models drop a closing quote -> invalid JSON. Send underscore keys;
    # semantic suffixes like "_action_id" survive so the id rule still fires.
    safe2orig: Dict[str, str] = {}
    safe_flat: Dict[str, str] = {}
    for i, (k, v) in enumerate(flat_leaky.items()):
        sk = re.sub(r"[\[\].]+", "_", k).strip("_") or f"k{i}"
        if sk in safe2orig:
            sk = f"{sk}_{i}"
        safe2orig[sk] = k
        safe_flat[sk] = v
    user = json.dumps(safe_flat, ensure_ascii=False, indent=2)
    # Qwen3.x on vLLM: disable the <think> pass (faster, clean JSON out)
    extra: Dict[str, Any] = {}
    if "qwen" in model.lower():
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=8000,
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": user}],
                **extra,
            )
            obj = _parse_obj(resp.choices[0].message.content or "")
            # map back to original keys; keep original text on missing/non-str
            return {orig: (obj[sk] if isinstance(obj.get(sk), str) else flat_leaky[orig])
                    for sk, orig in safe2orig.items()}
        except Exception as e:  # noqa
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return flat_leaky


@dataclass
class Stat:
    files: int = 0
    entries_scanned: int = 0
    fields_scanned: int = 0
    review: int = 0          # task-defining fields (subgoal/task_class) sent to LLM as a safeguard
    strong: int = 0          # regex-detected literal value (quoted value, file, url, number, proper noun)
    weak: int = 0            # a single non-UI capitalized token
    fields_changed: int = 0
    examples: List[str] = field(default_factory=list)


def _records(d: Any) -> List[dict]:
    """A file is either one entry (dict) or a list of entries."""
    items = d if isinstance(d, list) else [d]
    return [r for r in items if isinstance(r, dict)]


def run(banks: List[str], mode: str, sample_n: int, model: str,
        base_url: str, api_key: str,
        backup_dir: Optional[Path], report_path: Optional[Path],
        out_version: str = "") -> None:
    targets = [t for b in banks for t in BANKS.get(b, [])]
    report_rows: List[dict] = []
    sampled = 0

    # --out-version: clean a fresh COPY (_l1_actions_v3 -> _l1_actions_v4),
    # leaving the source bank untouched. Apply-only.
    if mode == "apply" and out_version:
        def _static_dir(glob: str) -> str:
            keep = []
            for part in glob.split("/"):
                if "*" in part or "?" in part:
                    break
                keep.append(part)
            return "/".join(keep)

        copied: set = set()
        for t in targets:
            if "_v3" not in t.glob:
                continue
            base = REPO_ROOT / _static_dir(t.glob)
            if str(base) in copied:
                continue
            v4base = Path(str(base).replace("_v3", f"_{out_version}"))
            if v4base.exists():
                shutil.rmtree(v4base)
            shutil.copytree(base, v4base)
            copied.add(str(base))
            # Blank raw_response (original mining output, carries every literal).
            # Not indexed, but stripped so v4 holds no leak at rest.
            n_files = n_stripped = 0
            for fp in v4base.rglob("*.json"):
                n_files += 1
                try:
                    dd = json.loads(fp.read_text())
                except Exception:
                    continue
                if isinstance(dd, dict) and dd.get("raw_response"):
                    dd["raw_response"] = ""
                    fp.write_text(json.dumps(dd, ensure_ascii=False, indent=2))
                    n_stripped += 1
            print(f"  copied {base.name} -> {v4base.name}  ({n_files} files, "
                  f"raw_response stripped from {n_stripped})")
        targets = [Target(t.name, t.glob.replace("_v3", f"_{out_version}"), t.extract, t.apply)
                   for t in targets]
        backup_dir = None  # the v3 source IS the backup

    for t in targets:
        st = Stat()
        files = _iter_files(t)
        st.files = len(files)
        # job = (fp, parsed, [(record, leaky_dict), ...]) ; leaky_dict = level>=1 fields
        jobs: List[Tuple[Path, Any, List[Tuple[dict, Dict[str, str]]]]] = []
        for fp in files:
            try:
                d = json.loads(fp.read_text())
            except Exception:
                continue
            rec_jobs: List[Tuple[dict, Dict[str, str]]] = []
            for rec in _records(d):
                st.entries_scanned += 1
                flat = t.extract(rec)
                st.fields_scanned += len(flat)
                leaky: Dict[str, str] = {}
                for k, v in flat.items():
                    if _force_review(k):
                        st.review += 1
                        leaky[k] = v
                        continue
                    lv = leak_level(v, k)
                    if lv == 2:
                        st.strong += 1
                        if len(st.examples) < 7:
                            st.examples.append(v[:150])
                    elif lv == 1:
                        st.weak += 1
                    if lv >= 1:
                        leaky[k] = v
                if leaky:
                    rec_jobs.append((rec, leaky))
            if rec_jobs:
                jobs.append((fp, d, rec_jobs))

        if mode == "dry-run":
            _print_stat(t, st)
            continue

        if mode == "sample":
            for fp, d, rec_jobs in jobs:
                for rec, leaky in rec_jobs:
                    if sampled >= sample_n:
                        break
                    sampled += 1
                    fixed = _rewrite(leaky, model, base_url, api_key)
                    print(f"\n── {t.name}  {fp.name} ──")
                    for k in leaky:
                        if leaky[k] != fixed[k]:
                            print(f"  [{k}]\n    -  {leaky[k]}\n    +  {fixed[k]}")
                        else:
                            print(f"  [{k}]  (unchanged)")
                if sampled >= sample_n:
                    break
            if sampled >= sample_n:
                break
            continue

        # apply: per file, rewrite every leaky record, apply, write parsed once
        def _do(job):
            fp, d, rec_jobs = job
            rows = []
            for rec, leaky in rec_jobs:
                try:
                    fixed = _rewrite(leaky, model, base_url, api_key)
                except Exception as e:  # noqa — fail-soft: skip this record, keep going
                    rows.append({"target": t.name, "file": str(fp.relative_to(REPO_ROOT)),
                                 "field": "__ERROR__", "before": str(e)[:200], "after": ""})
                    continue
                changed = {k: (leaky[k], fixed[k]) for k in leaky if leaky[k] != fixed[k]}
                if changed:
                    t.apply(rec, fixed)
                    for k, (before, after) in changed.items():
                        rows.append({"target": t.name, "file": str(fp.relative_to(REPO_ROOT)),
                                     "field": k, "before": before, "after": after})
            return fp, d, rows

        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            for fp, d, rows in ex.map(_do, jobs):
                if not rows:
                    continue
                if backup_dir:
                    bdest = backup_dir / fp.relative_to(REPO_ROOT)
                    bdest.parent.mkdir(parents=True, exist_ok=True)
                    if not bdest.exists():
                        shutil.copy2(fp, bdest)
                fp.write_text(json.dumps(d, ensure_ascii=False, indent=2))
                st.fields_changed += len(rows)
                report_rows.extend(rows)
        _print_stat(t, st)

    if report_path and report_rows:
        report_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in report_rows))
        print(f"\nChange report ({len(report_rows)} field rewrites): {report_path}")
    if mode == "apply":
        print("\nNEXT: rebuild v3 indexes so retrieval embeds the abstracted text:")
        print("  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.build_indexes.build_multi_layer_indexes")


def _print_stat(t: Target, st: Stat) -> None:
    flagged = st.review + st.strong + st.weak
    line = (f"  {t.name:9s} files={st.files:4d} entries={st.entries_scanned:5d} "
            f"fields={st.fields_scanned:6d}  → LLM {flagged:5d}  "
            f"[review={st.review} strong={st.strong} weak={st.weak}]")
    if st.fields_changed:
        line += f"  changed={st.fields_changed}"
    print(line)
    for ex in st.examples:
        print(f"        STRONG · {ex}")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="scan + count + examples, no LLM, no writes")
    g.add_argument("--sample", type=int, metavar="N", help="rewrite N flagged entries, print before/after")
    g.add_argument("--apply", action="store_true", help="rewrite all flagged + backup + report")
    ap.add_argument("--banks", default="v3,v2", help="comma list: v3,v2 (default both)")
    ap.add_argument("--out-version", default="",
                    help="clean a fresh COPY into this version (e.g. v4) instead of in place; "
                         "source bank untouched. Apply-only. Use with --banks v3.")
    ap.add_argument("--local", action="store_true",
                    help="use local vLLM Qwen3.5-27B at :8002 (free) instead of OpenRouter")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--backup-dir", default=str(REPO_ROOT / "results" / "_memory_pre_abstraction_backup"))
    ap.add_argument("--report", default=str(REPO_ROOT / "results" / "_memory_abstraction_report.jsonl"))
    a = ap.parse_args()

    if a.local:
        base_url = a.base_url or "http://localhost:8002/v1"
        model = a.model or "Qwen/Qwen3.5-27B"
        api_key = a.api_key or "EMPTY"
    else:
        base_url = a.base_url or OPENROUTER_BASE
        model = a.model or MODEL
        api_key = a.api_key or os.getenv("OPENROUTER_API_KEY", "")

    banks = [b.strip() for b in a.banks.split(",") if b.strip()]
    if a.sample:
        run(banks, "sample", a.sample, model, base_url, api_key, None, None, a.out_version)
    elif a.apply:
        run(banks, "apply", 0, model, base_url, api_key, Path(a.backup_dir), Path(a.report), a.out_version)
    else:
        run(banks, "dry-run", 0, model, base_url, api_key, None, None, a.out_version)


if __name__ == "__main__":
    main()
