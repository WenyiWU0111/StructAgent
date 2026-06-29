#!/usr/bin/env python3
"""
Thread-safe results logging for OSWorld evaluations.
Appends task completion results to results.json in real-time.
"""

import html
import json
import os
import re
import time
import fcntl
from pathlib import Path
from typing import Dict, Any, List, Optional

# HTML template for trajectory report (instruction, screenshots, response, plan, memory, final reward)
TRAJECTORY_HTML_TEMPLATE = """<!DOCTYPE html>
<head>
    <meta charset="utf-8">
    <style>
        pre { white-space: pre-wrap; word-wrap: break-word; }
        .section { margin: 1em 0; padding: 10px; border-radius: 5px; }
        .instruction { background-color: #E3F2FD; }
        .memory { background-color: #F3E5F5; }
        .plan { background-color: #E8F5E9; }
        .step { border: 1px solid #ccc; margin: 1em 0; padding: 10px; }
        .response { background-color: #FFF8E1; padding: 8px; margin: 8px 0; }
        .action { background-color: #E0F7FA; padding: 8px; }
        .score { font-size: 18px; font-weight: bold; padding: 15px; border-radius: 5px; }
        .score.pass { background-color: #4CAF50; color: white; }
        .score.fail { background-color: #F44336; color: white; }
        img.screenshot { max-width: 80%; height: auto; border: 1px solid #ddd; }
    </style>
</head>
<html><body>
{body}
</body></html>
"""


def write_trajectory_html(
    result_dir: str,
    instruction: str,
    steps: list,
    final_score: float,
    memory_text: str = None,
) -> None:
    """
    Write a single trajectory report HTML to result_dir/trajectory.html.
    steps: list of dicts with keys screenshot_b64, response, action, reward, done, info, plan (optional).

    Note: there's no separate "initial plan" rendering — Step 1's
    ``plan`` field is byte-identical to the initial plan (both read
    from agent.current_plan after the first predict call), so showing
    it twice would just duplicate ~2KB of XML at the top of the page.
    """
    parts = []
    # Task instruction
    parts.append("<h2>Task instruction</h2>")
    parts.append(f"<div class='section instruction'><pre>{html.escape(instruction)}</pre></div>")
    # Retrieved memory (if any)
    if memory_text:
        parts.append("<h2>Retrieved memory</h2>")
        parts.append(f"<div class='section memory'><pre>{html.escape(memory_text)}</pre></div>")
    # Steps
    parts.append("<h2>Steps</h2>")
    for i, step in enumerate(steps):
        step_num = i + 1
        parts.append(f"<div class='step'><h3>Step {step_num}</h3>")
        if step.get("screenshot_b64"):
            parts.append(f"<img class='screenshot' src='data:image/png;base64,{step['screenshot_b64']}' alt='Step {step_num}'/>")
        if step.get("a11y_tree"):
            tree = step["a11y_tree"]
            n_lines = tree.count("\n") + 1
            parts.append(
                f"<details style='margin:4px 0;padding:4px 8px;background:#F5F5F5;border-left:3px solid #9E9E9E'>"
                f"<summary style='cursor:pointer;font-weight:bold'>Accessibility tree ({n_lines} lines)</summary>"
                f"<pre style='font-size:11px;max-height:400px;overflow:auto'>{html.escape(tree)}</pre>"
                f"</details>"
            )
        # Multi-app: PhaseBoard — the per-app phase decomposition +
        # cross-app handoff. Rendered ABOVE the ledger so the reviewer
        # sees task structure (which app phase, what handoff) first.
        if step.get("phase_board"):
            pb_block = step["phase_board"]
            pb_first = (pb_block.splitlines() or ["Phase Board"])[0]
            parts.append(
                f"<details open style='margin:4px 0;padding:4px 8px;background:#FFF3E0;border-left:3px solid #FB8C00'>"
                f"<summary style='cursor:pointer;font-weight:bold'>{html.escape(pb_first)}</summary>"
                f"<pre style='font-size:12px;line-height:1.4'>{html.escape(pb_block)}</pre>"
                f"</details>"
            )
        if step.get("ledger_block"):
            block = step["ledger_block"]
            ledger_d = step.get("ledger") or {}
            n_done = len(ledger_d.get("done") or [])
            n_total = len(ledger_d.get("required_outcomes") or [])
            n_failed = len(ledger_d.get("failed_paths") or [])
            summary = f"Progress Ledger ({n_done}/{n_total} done, {n_failed} failed path(s))"
            parts.append(
                f"<details open style='margin:4px 0;padding:4px 8px;background:#EDE7F6;border-left:3px solid #673AB7'>"
                f"<summary style='cursor:pointer;font-weight:bold'>{html.escape(summary)}</summary>"
                f"<pre style='font-size:12px'>{html.escape(block)}</pre>"
                f"</details>"
            )
        # v2: event timeline + outcome view (cache-derived). Rendered as
        # collapsible blocks below the legacy ledger block so reviewers
        # can see the structured event-stream-of-truth at any step.
        if step.get("timeline_render"):
            tl_text = step["timeline_render"]
            n_lines = tl_text.count("\n") + 1
            parts.append(
                f"<details style='margin:4px 0;padding:4px 8px;background:#E1F5FE;border-left:3px solid #0277BD'>"
                f"<summary style='cursor:pointer;font-weight:bold'>"
                f"Event Timeline ({n_lines} lines)</summary>"
                f"<pre style='font-size:12px;line-height:1.4'>{html.escape(tl_text)}</pre>"
                f"</details>"
            )
        # KeyNode reasoning — render BEFORE the outcome view so the
        # reviewer sees WHY an outcome flipped to REVERTED before
        # seeing the resulting state. Without this block the
        # detector's evidence text is buried inside the timeline_event
        # JSON below and reviewers can't easily debug a flap.
        rkns = step.get("recent_key_nodes") or []
        if rkns:
            rows = []
            for kn in rkns:
                kind = kn.get("kind", "?")
                target = kn.get("target", "?")
                evidence = kn.get("evidence", "")
                conf = kn.get("confidence", "")
                if kind == "outcome_satisfied":
                    glyph = "✓"
                    color = "#2E7D32"
                elif kind == "outcome_invalidated":
                    glyph = "✗"
                    color = "#C62828"
                elif kind == "value_committed":
                    glyph = "="
                    color = "#1565C0"
                elif kind == "navigation":
                    glyph = "→"
                    color = "#6A1B9A"
                else:
                    glyph = "•"
                    color = "#616161"
                conf_tag = f" <span style='color:#888;font-size:10px'>({html.escape(conf)})</span>" if conf and conf != "high" else ""
                rows.append(
                    f"<div style='margin:2px 0'>"
                    f"<span style='color:{color};font-weight:bold'>{glyph} {html.escape(kind)}</span> "
                    f"<code style='background:#fff;padding:1px 4px'>{html.escape(target)}</code>"
                    f"{conf_tag}"
                    f"<div style='margin-left:20px;color:#555;font-size:11px'>"
                    f"evidence: {html.escape(evidence)}</div>"
                    f"</div>"
                )
            parts.append(
                f"<details open style='margin:4px 0;padding:6px 10px;background:#FFF3E0;border-left:3px solid #E65100'>"
                f"<summary style='cursor:pointer;font-weight:bold'>"
                f"KeyNode events this step ({len(rkns)})</summary>"
                + "".join(rows)
                + f"</details>"
            )
        if step.get("outcome_view"):
            ov = step["outcome_view"]
            # Color the summary differently if any outcome is REVERTED
            has_revert = "[REVERTED]" in ov
            border = "#C62828" if has_revert else "#388E3C"
            bg = "#FFEBEE" if has_revert else "#E8F5E9"
            tag = " ⚠ REVERTED" if has_revert else ""
            parts.append(
                f"<details open style='margin:4px 0;padding:4px 8px;background:{bg};border-left:3px solid {border}'>"
                f"<summary style='cursor:pointer;font-weight:bold'>"
                f"Outcome View (timeline-derived){html.escape(tag)}</summary>"
                f"<pre style='font-size:12px;line-height:1.4'>{html.escape(ov)}</pre>"
                f"</details>"
            )
        if step.get("timeline_event"):
            try:
                te = step["timeline_event"]
                if isinstance(te, list) and te:
                    ev = te[0]
                    n_kn = sum(len(s.get("key_nodes") or []) for s in (ev.get("steps") or []))
                    if n_kn > 0:
                        ev_json = json.dumps(ev, ensure_ascii=False, indent=2, default=str)
                        parts.append(
                            f"<details style='margin:4px 0;padding:4px 8px;background:#F3E5F5;border-left:3px solid #6A1B9A'>"
                            f"<summary style='cursor:pointer;font-weight:bold'>"
                            f"Latest TimelineEvent #{ev.get('event_idx','?')} "
                            f"({n_kn} KeyNode{'s' if n_kn!=1 else ''})</summary>"
                            f"<pre style='font-size:11px;max-height:300px;overflow:auto'>"
                            f"{html.escape(ev_json)}</pre>"
                            f"</details>"
                        )
            except Exception:
                pass
        if step.get("ledger_done_reject"):
            parts.append(
                f"<div style='margin:4px 0;padding:6px 10px;background:#FFEBEE;border-left:4px solid #C62828'>"
                f"<strong>DONE rejected by ledger gate:</strong> "
                f"<code>{html.escape(step['ledger_done_reject'])}</code>"
                f"</div>"
            )
        if step.get("done_audit_verdict"):
            av = step["done_audit_verdict"]
            verdict = av.get("verdict") or "?"
            v_lower = verdict.lower()
            if v_lower == "approve_done":
                border, bg = "#2E7D32", "#E8F5E9"
            elif v_lower == "continue":
                border, bg = "#C62828", "#FFEBEE"
            else:
                border, bg = "#757575", "#FAFAFA"
            body_lines: List[str] = []
            body_lines.append(f"<strong>DONE audit verdict:</strong> "
                              f"<code>{html.escape(verdict)}</code>")
            if av.get("reason"):
                body_lines.append(f"&nbsp;&nbsp;reason: {html.escape(str(av['reason']))[:300]}")
            if av.get("next_action_hint"):
                body_lines.append(f"&nbsp;&nbsp;next_action_hint: "
                                  f"{html.escape(str(av['next_action_hint']))[:300]}")
            edits = av.get("edits") or []
            if edits:
                body_lines.append(f"&nbsp;&nbsp;edits ({len(edits)}):")
                for e in edits:
                    op = e.get('op', '?'); eid = e.get('id', '?')
                    rsn = (e.get('reason') or '')[:200]
                    body_lines.append(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;• <code>{html.escape(op)}</code> "
                        f"id={html.escape(str(eid))} — {html.escape(rsn)}"
                    )
            parts.append(
                f"<div style='margin:4px 0;padding:6px 10px;background:{bg};"
                f"border-left:4px solid {border}'>"
                + "<br/>".join(body_lines)
                + "</div>"
            )
        if step.get("planner_decision"):
            parts.append(f"<div class='section decision'><strong>Planner decision:</strong> <code>{html.escape(step['planner_decision'])}</code></div>")
        if step.get("current_subgoal"):
            parts.append(f"<div class='section subgoal' style='background:#fff3cd;padding:8px;border-left:4px solid #ffc107;margin:4px 0'><strong>Current subgoal:</strong><pre>{html.escape(step['current_subgoal'])}</pre></div>")
        if step.get("planner_response"):
            parts.append(f"<div class='section planner-response' style='background:#e8f4fd;padding:8px;border-left:4px solid #2196F3;margin:4px 0'><strong>Planner response:</strong><pre>{html.escape(step['planner_response'])}</pre></div>")
        if step.get("observations"):
            parts.append(f"<div class='section observations'><strong>Observations:</strong><pre>{html.escape(step['observations'])}</pre></div>")
        if step.get("plan"):
            parts.append(f"<div class='section plan'><strong>Plan:</strong><pre>{html.escape(step['plan'])}</pre></div>")
        if step.get("next_step_hint"):
            parts.append(f"<div class='section next_step_hint'><strong>Next step hint:</strong><pre>{html.escape(step['next_step_hint'])}</pre></div>")
        if step.get("response") not in (None, ""):
            parts.append(f"<div class='response'><strong>Response:</strong><pre>{html.escape(str(step['response']))}</pre></div>")
        if step.get("action") is not None:
            action_str = step["action"] if isinstance(step["action"], str) else json.dumps(step["action"], ensure_ascii=False)
            # Long actions (cli_run_uno wrappers, full screenshots embedded
            # in the action, etc.) bloat the page. Collapse anything over
            # ~200 chars into a <details>, with the first line + a length
            # tag as the summary so reviewers can skim without expanding.
            if len(action_str) > 200:
                first_line = action_str.lstrip().splitlines()[0] if action_str.strip() else ""
                summary_line = (first_line[:140] + "…") if len(first_line) > 140 else first_line
                parts.append(
                    f"<details class='action' style='margin:4px 0;padding:4px 8px;background:#FAFAFA;border-left:3px solid #757575'>"
                    f"<summary style='cursor:pointer;font-weight:bold'>"
                    f"Action ({len(action_str)} chars) — {html.escape(summary_line)}</summary>"
                    f"<pre style='font-size:11px;max-height:500px;overflow:auto;white-space:pre-wrap'>{html.escape(action_str)}</pre>"
                    f"</details>"
                )
            else:
                parts.append(f"<div class='action'><strong>Action:</strong><pre>{html.escape(action_str)}</pre></div>")
        cro = step.get("cli_run_output")
        if cro:
            cli_lines = [
                f"command:    {cro.get('command', '')}",
                f"mode:       {cro.get('mode', '?')}",
                f"returncode: {cro.get('returncode')}",
                f"stdout:     {(cro.get('stdout') or '(empty)')[:2000]}",
                f"stderr:     {(cro.get('stderr') or '(empty)')[:800]}",
            ]
            parts.append(
                "<div class='section cli-run-output' "
                "style='background:#fff3e0;padding:8px;border-left:4px solid #FF9800;margin:4px 0'>"
                f"<strong>cli_run output (captured from /tmp/_last_cli_run.json):</strong>"
                f"<pre>{html.escape(chr(10).join(cli_lines))}</pre></div>"
            )
        reward = step.get("reward")
        if reward is not None:
            parts.append(f"<div>Reward: {reward}</div>")
        parts.append("</div>")
    # Final score
    parts.append("<h2>Final result</h2>")
    result_text = "PASS" if final_score > 0 else "FAIL"
    css_class = "score pass" if final_score > 0 else "score fail"
    parts.append(f"<div class='{css_class}'>Score: {final_score} ({result_text})</div>")
    body = "\n".join(parts)
    path = os.path.join(result_dir, "trajectory.html")
    # Use replace instead of format: body often contains { } from JSON/tool_call, which would break .format()
    html_content = TRAJECTORY_HTML_TEMPLATE.replace("{body}", body)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Also write a sibling pretty-printed traj.pretty.json — a single
    # JSON array of step records, indented for human reading. The
    # original ``traj.jsonl`` is kept untouched (one record per line is
    # a contract relied on by many readers — `for line in f:
    # json.loads(line)`). Reviewers who want a readable file open
    # traj.pretty.json instead.
    try:
        jsonl_path = os.path.join(result_dir, "traj.jsonl")
        if os.path.exists(jsonl_path):
            records: list = []
            with open(jsonl_path, "r", encoding="utf-8") as jf:
                for line in jf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Tolerate the rare malformed line — drop it
                        # rather than failing the whole pretty dump.
                        continue
            pretty_path = os.path.join(result_dir, "traj.pretty.json")
            with open(pretty_path, "w", encoding="utf-8") as pf:
                json.dump(records, pf, ensure_ascii=False, indent=2)
                pf.write("\n")
    except Exception:
        # Pretty-dump is purely a convenience artifact; never let it
        # break the main html write.
        pass


def _model_suffix_from_args_and_path(args: Any, result_dir: str) -> str:
    """Derive a filesystem-safe model suffix for results filename."""
    model = getattr(args, "result_model", None) or getattr(args, "model", None)
    if model:
        return re.sub(r"[^\w\-.]", "_", str(model))
    # Fallback: result_dir is .../action_space/observation_type/MODEL/domain/task_id
    parts = Path(result_dir).parts
    if len(parts) >= 4:
        return re.sub(r"[^\w\-.]", "_", parts[-4])
    return "default"


def extract_domain_from_path(result_path: str) -> str:
    """
    Extract domain/application from result directory path.
    Expected structure: results/{action_space}/{observation_type}/{model}/{domain}/{task_id}/
    """
    path_parts = Path(result_path).parts
    if len(path_parts) >= 2:
        return path_parts[-2]  # Second to last part should be domain
    return "unknown"


def append_task_result(
    task_id: str,
    domain: str, 
    score: float,
    result_dir: str,
    args: Any,
    error_message: Optional[str] = None
) -> None:
    """
    Thread-safely append a task result to results.json.
    
    Args:
        task_id: UUID of the task
        domain: Application domain (chrome, vlc, etc.)
        score: Task score (0.0 or 1.0)
        result_dir: Full path to the task result directory
        args: Command line arguments object
        error_message: Error message if task failed
    """
    # Create result entry
    result_entry = {
        "application": domain,
        "task_id": task_id,
        "status": "error" if error_message else "success",
        "score": score,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if error_message:
        result_entry["err_message"] = error_message
    
    # Determine summary directory and results file path (per-model file)
    base_result_dir = Path(args.result_dir)
    summary_dir = base_result_dir / "summary"
    model_suffix = _model_suffix_from_args_and_path(args, result_dir)
    results_file = summary_dir / f"results_{model_suffix}.json"
    
    # Ensure summary directory exists
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Thread-safe JSON append with file locking
    try:
        with open(results_file, 'a+') as f:
            # Lock the file for exclusive access
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            try:
                # Move to beginning to read existing content
                f.seek(0)
                content = f.read().strip()
                
                # Parse existing JSON array or create new one
                if content:
                    try:
                        existing_results = json.loads(content)
                        if not isinstance(existing_results, list):
                            existing_results = []
                    except json.JSONDecodeError:
                        existing_results = []
                else:
                    existing_results = []
                
                # Add new result
                existing_results.append(result_entry)
                
                # Write back the complete JSON array
                f.seek(0)
                f.truncate()
                json.dump(existing_results, f, indent=2)
                f.write('\n')  # Add newline for readability
                
            finally:
                # Always unlock the file
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
        print(f"📝 Logged result: {domain}/{task_id} -> {result_entry['status']} (score: {score})")
        
    except Exception as e:
        # Don't let logging errors break the main evaluation
        print(f"⚠️  Failed to log result for {task_id}: {e}")


def log_task_completion(example: Dict, result: float, result_dir: str, args: Any) -> None:
    """
    Convenience wrapper for logging successful task completion.
    
    Args:
        example: Task configuration dictionary
        result: Task score
        result_dir: Path to task result directory  
        args: Command line arguments
    """
    task_id = example.get('id', 'unknown')
    domain = extract_domain_from_path(result_dir)
    append_task_result(task_id, domain, result, result_dir, args)


def log_task_error(example: Dict, error_msg: str, result_dir: str, args: Any) -> None:
    """
    Convenience wrapper for logging task errors.
    
    Args:
        example: Task configuration dictionary
        error_msg: Error message
        result_dir: Path to task result directory
        args: Command line arguments
    """
    task_id = example.get('id', 'unknown')
    domain = extract_domain_from_path(result_dir) 
    append_task_result(task_id, domain, 0.0, result_dir, args, error_msg)