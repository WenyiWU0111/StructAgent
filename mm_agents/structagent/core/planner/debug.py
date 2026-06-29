"""Per-planner-turn HTML dump for offline debug.

For every planner LLM call, ``dump_planner_call()`` writes a single
self-contained HTML page that shows:

  • header — step_idx, mode (initial / force_replan / progress_check /
    subgoal_done / all_subgoals_done / actor_exhausted), attempt index,
    stuck_category if applicable, parsed decision, elapsed seconds
  • system prompt (collapsed by default)
  • each user message in order (text rendered as <pre>, images inlined
    as <img src="data:..."> with size cap)
  • raw response (formatted)
  • parsed metadata (plan / decision / refined subgoal etc.)

Layout under ``{results_dir}/planner_debug/``:

    step_0000_initial_attempt0.html
    step_0001_progress_check_attempt0.html
    step_0002_force_replan_actor_failure_attempt0.html
    step_0002_force_replan_actor_failure_attempt1.html  ← retry
    index.html                                          ← appended live

The index page is rebuilt on each dump so a long-running task always
has a usable summary even if interrupted.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _safe(s: str, n: int = 60) -> str:
    return _SAFE.sub("_", (s or ""))[:n]


# Images can be huge in base64; cap the visual size we render to keep
# the HTML responsive. CSS handles downscaling; we don't re-encode.
_IMG_CSS_MAX_WIDTH = "1100px"


def _esc(s: Any) -> str:
    return html.escape(str(s), quote=False)


def _render_one_content_item(item: Any) -> str:
    """Render one piece of a message's content list as an HTML fragment."""
    if isinstance(item, str):
        return f'<pre class="text-block">{_esc(item)}</pre>'
    if not isinstance(item, dict):
        return f'<pre class="text-block">{_esc(repr(item))}</pre>'
    t = item.get("type")
    if t == "text":
        return f'<pre class="text-block">{_esc(item.get("text") or "")}</pre>'
    if t == "image_url":
        url = (item.get("image_url") or {}).get("url") or ""
        # Truncate the url just in the title to avoid storing a 200K data
        # URL in HTML alt attribute (the src itself is the data URL).
        title = url[:80] + ("…" if len(url) > 80 else "")
        return (
            f'<div class="img-wrap">'
            f'<img src="{_esc(url)}" alt="planner image" title="{_esc(title)}" />'
            f'</div>'
        )
    # Fallback — dump JSON of the item
    return f'<pre class="text-block">{_esc(json.dumps(item, ensure_ascii=False, default=str))}</pre>'


def _render_message(role: str, content: Any, idx: int) -> str:
    role_class = {
        "system": "msg-system",
        "user": "msg-user",
        "assistant": "msg-assistant",
    }.get(role, "msg-other")
    # System messages get collapsed by default.
    open_attr = "" if role == "system" else " open"
    header_label = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
    }.get(role, role or "?")
    # Render content (may be a string or list of parts).
    if isinstance(content, list):
        body = "\n".join(_render_one_content_item(it) for it in content)
    elif content is None:
        body = '<pre class="text-block">(empty)</pre>'
    else:
        body = _render_one_content_item(content)
    return (
        f'<details class="msg {role_class}"{open_attr}>'
        f'<summary><span class="role-tag">{_esc(header_label)}</span>'
        f'<span class="msg-idx">#{idx}</span></summary>'
        f'<div class="msg-body">{body}</div>'
        f'</details>'
    )


def _render_response(response: Optional[str]) -> str:
    if not response:
        return '<pre class="text-block">(no response — LLM call failed or returned empty)</pre>'
    return f'<pre class="text-block">{_esc(response)}</pre>'


def _render_metadata_table(metadata: Dict[str, Any]) -> str:
    if not metadata:
        return ""
    rows = []
    for k, v in metadata.items():
        if isinstance(v, (dict, list)):
            v_s = json.dumps(v, ensure_ascii=False, indent=2, default=str)
            cell = f'<pre class="text-block">{_esc(v_s)}</pre>'
        else:
            cell = _esc(v if v is not None else "")
        rows.append(f'<tr><th class="meta-k">{_esc(k)}</th><td class="meta-v">{cell}</td></tr>')
    return f'<table class="metadata">{"".join(rows)}</table>'


_CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
  background: #f5f7fa;
  color: #1f2937;
  margin: 0;
  padding: 0;
}
.container { max-width: 1200px; margin: 0 auto; padding: 24px 32px; }
h1 { font-size: 1.4em; margin: 0 0 4px 0; }
.header-card {
  background: #fff; border: 1px solid #d1d5db; border-radius: 8px;
  padding: 16px 20px; margin-bottom: 20px;
}
.tag {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 0.78em; font-weight: 600; margin-right: 6px;
}
.tag-mode { background: #dbeafe; color: #1e3a8a; }
.tag-cat  { background: #fef3c7; color: #78350f; }
.tag-attempt { background: #e5e7eb; color: #374151; }
.tag-decision { background: #d1fae5; color: #064e3b; }
.tag-fail    { background: #fee2e2; color: #7f1d1d; }
.section-title {
  font-size: 1.05em; margin: 20px 0 8px 4px;
  color: #4b5563; font-weight: 600;
}
.msg {
  background: #fff; border: 1px solid #d1d5db; border-left-width: 4px;
  border-radius: 6px; margin-bottom: 8px;
}
.msg.msg-system    { border-left-color: #6366f1; }
.msg.msg-user      { border-left-color: #10b981; }
.msg.msg-assistant { border-left-color: #f59e0b; }
.msg.msg-other     { border-left-color: #6b7280; }
.msg summary {
  cursor: pointer; padding: 8px 12px; font-weight: 600;
  user-select: none; list-style: none;
}
.msg summary::-webkit-details-marker { display: none; }
.msg summary::before {
  content: "▶"; display: inline-block; margin-right: 8px;
  transition: transform 0.15s;
  color: #6b7280; font-size: 0.7em;
}
.msg[open] summary::before { transform: rotate(90deg); }
.role-tag {
  display: inline-block; padding: 1px 8px; border-radius: 3px;
  background: rgba(0,0,0,0.06); font-size: 0.82em; margin-right: 6px;
}
.msg-idx { color: #9ca3af; font-size: 0.8em; font-weight: 400; }
.msg-body { padding: 8px 14px 14px 14px; }
.text-block {
  background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 4px;
  padding: 10px 12px; margin: 0 0 8px 0;
  font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", monospace;
  font-size: 0.85em; line-height: 1.45;
  white-space: pre-wrap; word-break: break-word;
  max-height: 520px; overflow-y: auto;
}
.img-wrap { margin: 6px 0; }
.img-wrap img {
  max-width: """ + _IMG_CSS_MAX_WIDTH + """;
  height: auto; border: 1px solid #d1d5db; border-radius: 4px;
  display: block;
}
.response-card {
  background: #fff; border: 1px solid #d1d5db; border-left: 4px solid #f59e0b;
  border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;
}
.metadata {
  width: 100%; border-collapse: collapse; background: #fff;
  border: 1px solid #d1d5db; border-radius: 6px; overflow: hidden;
}
.metadata th, .metadata td {
  text-align: left; padding: 8px 12px; vertical-align: top;
  border-bottom: 1px solid #e5e7eb;
}
.metadata th.meta-k {
  width: 200px; background: #f9fafb; color: #4b5563;
  font-weight: 600; font-size: 0.85em;
}
.metadata td.meta-v { color: #1f2937; font-size: 0.88em; }
.metadata tr:last-child th, .metadata tr:last-child td { border-bottom: 0; }
a { color: #2563eb; }
.index-table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d1d5db; border-radius: 6px; overflow: hidden; }
.index-table th, .index-table td { padding: 6px 10px; border-bottom: 1px solid #e5e7eb; font-size: 0.88em; text-align: left; }
.index-table th { background: #f9fafb; color: #4b5563; font-weight: 600; }
"""


def _render_html(
    *, step_idx: int, mode: str, attempt: int,
    messages: List[Dict[str, Any]], response: Optional[str],
    metadata: Dict[str, Any],
) -> str:
    """Assemble one planner-call HTML page."""
    # Header tags
    stuck_cat = metadata.get("stuck_category")
    decision = metadata.get("parsed_decision")
    elapsed = metadata.get("elapsed_s")
    title = f"step {step_idx:04d} · {mode}"
    if stuck_cat:
        title += f" · {stuck_cat}"
    tag_html: List[str] = []
    tag_html.append(f'<span class="tag tag-mode">mode={_esc(mode)}</span>')
    if stuck_cat:
        tag_html.append(f'<span class="tag tag-cat">category={_esc(stuck_cat)}</span>')
    tag_html.append(f'<span class="tag tag-attempt">attempt {attempt}</span>')
    if decision:
        cls = "tag-decision" if decision in ("CONTINUE", "DONE", "REPLAN") else "tag-fail"
        tag_html.append(f'<span class="tag {cls}">decision={_esc(decision)}</span>')
    if elapsed is not None:
        tag_html.append(f'<span class="tag tag-attempt">{float(elapsed):.2f}s</span>')

    messages_html = "\n".join(
        _render_message(m.get("role", "?"), m.get("content"), i)
        for i, m in enumerate(messages or [])
    )
    response_html = _render_response(response)
    meta_html = _render_metadata_table({
        k: v for k, v in metadata.items()
        if k not in {"parsed_decision", "stuck_category", "elapsed_s"}
    })

    return (
        '<!doctype html>\n<html lang="en"><head>'
        '<meta charset="utf-8">'
        f'<title>{_esc(title)}</title>'
        f'<style>{_CSS}</style></head><body>'
        '<div class="container">'
        f'<div class="header-card">'
        f'<h1>{_esc(title)}</h1>'
        f'<div>{" ".join(tag_html)}</div>'
        f'<div style="margin-top:8px;"><a href="index.html">← back to index</a></div>'
        f'</div>'
        f'<div class="section-title">Response</div>'
        f'<div class="response-card">{response_html}</div>'
        f'<div class="section-title">Messages</div>'
        f'{messages_html}'
        f'{("<div class=" + chr(34) + "section-title" + chr(34) + ">Metadata</div>" + meta_html) if meta_html else ""}'
        '</div></body></html>'
    )


# ──────────────────────────────────────────────────────────────────────
# Index page — appended/refreshed on every dump_planner_call.
# ──────────────────────────────────────────────────────────────────────


def _index_record_path(out_dir: Path) -> Path:
    return out_dir / "_index_records.jsonl"


def _append_index_record(out_dir: Path, record: Dict[str, Any]) -> None:
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with (_index_record_path(out_dir)).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.info("[planner_debug] index append failed: %s", e)


def _rebuild_index(out_dir: Path) -> None:
    rec_path = _index_record_path(out_dir)
    if not rec_path.exists():
        return
    records: List[Dict[str, Any]] = []
    try:
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    except Exception as e:
        logger.info("[planner_debug] index read failed: %s", e)
        return

    # Sort by step then attempt for natural reading.
    records.sort(key=lambda r: (
        int(r.get("step_idx", 0)), int(r.get("attempt", 0)),
    ))

    rows = []
    for r in records:
        decision = r.get("parsed_decision") or ""
        cls = ("tag-decision" if decision in ("CONTINUE", "DONE", "REPLAN")
               else ("tag-fail" if decision else ""))
        _es = r.get("elapsed_s")
        elapsed_cell = "" if _es is None else f"{float(_es):.2f}s"
        rows.append(
            f'<tr>'
            f'<td>{_esc(r.get("step_idx", "?"))}</td>'
            f'<td>{_esc(r.get("mode", "?"))}</td>'
            f'<td>{_esc(r.get("stuck_category") or "")}</td>'
            f'<td>{_esc(r.get("attempt", 0))}</td>'
            f'<td><span class="tag {cls}">{_esc(decision)}</span></td>'
            f'<td>{elapsed_cell}</td>'
            f'<td><a href="{_esc(r.get("filename", ""))}">open</a></td>'
            f'</tr>'
        )
    table = (
        '<table class="index-table">'
        '<thead><tr>'
        '<th>step</th><th>mode</th><th>stuck_category</th>'
        '<th>attempt</th><th>decision</th><th>elapsed</th><th></th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
    page = (
        '<!doctype html>\n<html lang="en"><head>'
        '<meta charset="utf-8"><title>planner debug index</title>'
        f'<style>{_CSS}</style></head><body>'
        '<div class="container">'
        f'<div class="header-card"><h1>Planner debug — {len(records)} call(s)</h1>'
        '<div>One row per planner LLM call. Click "open" for full prompt + response.</div>'
        '</div>'
        f'{table}'
        '</div></body></html>'
    )
    try:
        (out_dir / "index.html").write_text(page, encoding="utf-8")
    except Exception as e:
        logger.info("[planner_debug] index write failed: %s", e)


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────


def dump_planner_call(
    *,
    results_dir: Optional[str],
    step_idx: int,
    mode: str,
    attempt: int = 0,
    messages: List[Dict[str, Any]],
    response: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one planner LLM call's HTML page + update the index.

    ``metadata`` may include any of (all optional):
      stuck_category   — recovery-rule category if mode==force_replan
      stuck_reason     — recovery-rule one-line reason
      parsed_decision  — CONTINUE / DONE / REPLAN from the response parser
      parsed_plan      — extracted <plan> text (if any)
      elapsed_s        — wall time of the LLM call in seconds
      ... anything else the caller wants to surface in the page footer

    Failures are logged at info level and never raise to the caller.
    """
    if not results_dir:
        return
    try:
        out_dir = Path(results_dir) / "planner_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_mode = _safe(mode, 40)
        cat = (metadata or {}).get("stuck_category")
        if cat:
            fname = f"step_{step_idx:04d}_{safe_mode}_{_safe(cat, 20)}_attempt{attempt}.html"
        else:
            fname = f"step_{step_idx:04d}_{safe_mode}_attempt{attempt}.html"
        page = _render_html(
            step_idx=step_idx, mode=mode, attempt=attempt,
            messages=messages or [], response=response,
            metadata=metadata or {},
        )
        (out_dir / fname).write_text(page, encoding="utf-8")
        # Append to index records + rebuild index.
        meta = metadata or {}
        _append_index_record(out_dir, {
            "step_idx": step_idx,
            "mode": mode,
            "stuck_category": meta.get("stuck_category"),
            "attempt": attempt,
            "parsed_decision": meta.get("parsed_decision"),
            "elapsed_s": meta.get("elapsed_s"),
            "filename": fname,
            "ts": time.time(),
        })
        _rebuild_index(out_dir)
    except Exception as e:
        logger.info("[planner_debug] dump_planner_call failed: %s", e)
