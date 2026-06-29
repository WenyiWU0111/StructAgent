import base64
import datetime
import dataclasses
import json
import logging
import os
import time
from wrapt_timeout_decorator import *
from lib_results_logger import log_task_completion, write_trajectory_html
import mind2web_eval

logger = logging.getLogger("desktopenv.experiment")


def _dump_agent_runtime_config(agent, example_result_dir):
    """Persist StructAgent's env-resolved config next to task artifacts."""
    cfg = getattr(agent, "cfg", None)
    if cfg is None:
        return
    try:
        if dataclasses.is_dataclass(cfg):
            payload = dataclasses.asdict(cfg)
        else:
            payload = dict(getattr(cfg, "__dict__", {}) or {})
        payload.update({
            "verifier_model": getattr(agent, "verifier_model", None),
            "planner_model": getattr(agent, "model", None),
            "decomposer_model": getattr(agent, "decomposer_model", None),
            "grounding_model": getattr(agent, "grounding_model", None),
            "effective_enable_boundary_verify": getattr(
                agent, "_boundary_verify_on", None),
            "effective_enable_actor_burst": getattr(
                agent, "_actor_burst_on", None),
            "effective_actor_burst_max_turns": getattr(
                agent, "_ACTOR_BURST_MAX_TURNS", None),
            "effective_actor_burst_no_effect_limit": getattr(
                agent, "_ACTOR_BURST_NO_EFFECT_LIMIT", None),
            "effective_actor_burst_debug": getattr(
                agent, "_actor_burst_debug", None),
            "effective_enable_done_auditor": getattr(
                agent, "enable_done_auditor", None),
            "done_auditor_model": getattr(agent, "done_auditor_model", None),
            "done_auditor_max_per_task": getattr(
                agent, "done_auditor_max_per_task", None),
            "done_auditor_budget_remaining": getattr(
                agent, "_done_audit_budget_remaining", None),
            "done_audit_reject_count": getattr(
                agent, "_done_audit_reject_count", None),
            "effective_enable_stuck_diagnosis_injection": getattr(
                agent, "enable_stuck_diagnosis_injection", None),
            "use_strong_rescue": getattr(agent, "use_strong_rescue", None),
            "strong_rescue_model": getattr(agent, "strong_rescue_model", None),
            "strong_rescue_budget_remaining": getattr(
                agent, "_rescue_budget", None),
        })
        with open(
            os.path.join(example_result_dir, "structagent_config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("Failed to dump StructAgent config: %s", e)


def _dump_eval_result_files(env, example, example_result_dir):
    """Copy the VM-side files the evaluator compares into
    ``<result_dir>/eval_files/``.

    The task's ``evaluator.result`` block lists ``vm_file`` entries
    (path on the VM + dest filename) — the exact files ``env.evaluate``
    pulled and diffed against the gold. Persisting them lets offline
    debugging reproduce the comparison (e.g. for libreoffice_impress,
    ``compare_pptx_files(out, gold, enable_debug=True)`` prints the
    first mismatching field). Best-effort: never raises.
    """
    try:
        evaluator = (example or {}).get("evaluator") or {}
        result_spec = evaluator.get("result")
        if result_spec is None:
            return
        entries = result_spec if isinstance(result_spec, list) else [result_spec]
        out_dir = os.path.join(example_result_dir, "eval_files")
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") != "vm_file":
                continue
            vm_path = entry.get("path")
            dest = entry.get("dest") or (os.path.basename(vm_path) if vm_path else None)
            if not vm_path or not dest or dest in seen:
                continue
            seen.add(dest)
            try:
                raw = env.controller.get_file(vm_path)
            except Exception as e:
                logger.warning("[eval-dump] get_file(%s) failed: %s", vm_path, e)
                continue
            if raw is None:
                logger.warning("[eval-dump] %s not present on VM", vm_path)
                continue
            os.makedirs(out_dir, exist_ok=True)
            data = raw.encode("utf-8") if isinstance(raw, str) else raw
            with open(os.path.join(out_dir, dest), "wb") as f:
                f.write(data)
            logger.info("[eval-dump] saved %s -> eval_files/%s (%d bytes)",
                        vm_path, dest, len(data))
    except Exception as e:
        logger.warning("[eval-dump] failed: %s", e)


def run_single_example(agent, env, example, max_steps, instruction, args, example_result_dir, scores):
    runtime_logger = setup_logger(example, example_result_dir)
    all_actions = []  # accumulated action strings (for the Mind2Web grader)

    # NOTE: no infeasible short-circuit here by design. OSWorld hides which tasks
    # are infeasible from the agent (evaluator metadata is never passed to predict),
    # so the agent must JUDGE infeasibility itself and emit a FAIL action — that is
    # exactly what the "infeasible" evaluator scores (1.0 iff the action history ends
    # with FAIL). Peeking at example["evaluator"]["func"]=="infeasible" to auto-FAIL
    # is test-set leakage: it reads the ground-truth label and skips the agent
    # entirely, inflating the score for free. Infeasible tasks now run the normal
    # loop like any other; the agent's own infeasibility verdict drives FAIL.

    # Reset environment first to get fresh VM IP
    env.reset(task_config=example)

    # Reset agent with fresh VM IP (for snapshot reverts)
    try:
        agent.reset(runtime_logger, vm_ip=env.vm_ip)
    except Exception as e:
        # agent.reset(vm_ip=env.vm_ip)
        agent.reset(runtime_logger)

    # Set current domain for shortcut retrieval.
    # Normalize "libreoffice calc" → "libreoffice_calc" etc. because the
    # task metadata uses both forms inconsistently and the planner's routing
    # checks key on the underscore form.
    related_apps = example.get("related_apps", [])
    # Normalize each related-app label to the framework's domain id —
    # the perceiver / domain_knowledge / cli_run_uno / evidence_guide
    # all key on these. OSWorld task metadata uses "terminal" and
    # "vscode", but the framework's domains are "os" and "vs_code".
    # After normalizing, dedupe (order-preserving): e.g.
    # ['os', 'terminal'] collapses to ['os'] — correctly a SINGLE-domain
    # task, not multi-app.
    _app_to_domain = {"terminal": "os", "vscode": "vs_code"}
    _seen: set = set()
    _norm_apps: list = []
    for _a in (related_apps or []):
        if not _a or not str(_a).strip():
            continue
        _na = str(_a).strip().lower().replace(" ", "_").replace("-", "_")
        _na = _app_to_domain.get(_na, _na)
        if _na not in _seen:
            _seen.add(_na)
            _norm_apps.append(_na)
    # Full related-domain list — the agent treats a task with >1 DISTINCT
    # domain as "multi-app mode" and segments work per app. Single-app
    # tasks see a 0/1-element list and behave exactly as before.
    agent._related_apps = _norm_apps
    agent._current_domain = _norm_apps[0] if _norm_apps else None

    # Set CDP connection info for a11y-assisted element targeting
    agent._vm_ip = getattr(env, 'vm_ip', None)
    agent._chromium_port = getattr(env, 'chromium_port', 9222)
    # Per-task result directory: enables in-agent dumps of LLM prompts /
    # responses (code-actor, discovery hop, etc.) into <result_dir>/scripts/
    # for offline debugging of code-generation quality.
    agent._results_dir = example_result_dir
    _dump_agent_runtime_config(agent, example_result_dir)

    time.sleep(70) # Wait for the environment to be ready
    obs = env._get_obs() # Get the initial observation

    done = False
    step_idx = 0
    # Collect steps for trajectory.html (screenshot as base64); no separate PNGs
    steps_for_html = []
    memory_text = None
    plan_per_step = []
    env.controller.start_recording()
    if _norm_apps:
        # Multi-app tasks: tag the instruction with [multi_apps] rather
        # than mislabelling the whole task as one app (often the wrong
        # one — e.g. an os-centric "force quit" task whose first
        # related_app is libreoffice_writer). Uses the normalized,
        # deduped domain list so e.g. ['os','terminal'] (→ ['os']) is
        # correctly tagged single-domain. The agent gets the real list
        # via agent._related_apps; _current_domain is set directly above.
        _instr_tag = "multi_apps" if len(_norm_apps) > 1 else _norm_apps[0]
        instruction = f'[{_instr_tag}] {instruction}'

    while not done and step_idx < max_steps:
        # StructAgent (planner-actor) needs env + wall_step_idx for VM
        # state introspection and ledger sync. Vanilla single-model
        # agents (Qwen25VLAgent, Qwen3VLAgent, kimi, owl, uitars,
        # mano, etc.) have ``predict(instruction, obs)`` signatures and
        # crash with TypeError on the extra kwargs. Try the rich form,
        # fall back to the minimal one — same pattern as the agent.reset
        # call above so any agent class works with this runner.
        try:
            response, actions = agent.predict(
                instruction,
                obs,
                env=env,
                wall_step_idx=step_idx,
            )
        except TypeError as _e:
            if "unexpected keyword argument" in str(_e):
                response, actions = agent.predict(instruction, obs)
            else:
                raise
        # Capture memory once on the first predict (initial plan is
        # already preserved as Step 1's "plan" field — no separate copy).
        if step_idx == 0:
            memory_text = getattr(agent, "memory_steps_text", None) or None
        current_plan = getattr(agent, "current_plan", None)
        next_step_hint = getattr(agent, "next_step_hint", None)
        observations = getattr(agent, "observations", None)
        current_subgoal = getattr(agent, "current_subgoal", None)
        planner_decision = getattr(agent, "planner_decision", None)
        planner_response = getattr(agent, "planner_response", None)
        # Planner-actor mode: stop if planner decided DONE. The ledger's
        # ``can_accept_done_claim()`` gate is the primary guard now; runs
        # inside the agent before this flag is set.
        if getattr(agent, "all_done_after_step", False):
            logger.info("Planner decided DONE — terminating episode.")
            # Record this final DONE step to traj.jsonl + steps_for_html so
            # the final planner decision (and the ledger state after the
            # boundary summarizer fire that happens inside predict()) are
            # visible in trajectory.html instead of only runtime.log. We
            # do NOT execute any action for this step — the DONE decision
            # itself is what we want to preserve.
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            screenshot_b64_final = None
            if isinstance(obs, dict) and obs.get("screenshot"):
                screenshot_b64_final = base64.b64encode(obs["screenshot"]).decode("ascii")
            a11y_text_final = None
            if isinstance(obs, dict) and obs.get("accessibility_tree"):
                try:
                    from mm_agents.structagent.utils.a11y_tree.accessibility_tree_handle import (
                        linearize_accessibility_tree, trim_accessibility_tree,
                    )
                    a11y_text_final = linearize_accessibility_tree(obs["accessibility_tree"], "Ubuntu")
                    if a11y_text_final:
                        a11y_text_final = trim_accessibility_tree(a11y_text_final, 300)
                except Exception:
                    a11y_text_final = None
            ledger_dict_final = None
            ledger_block_final = None
            ledger_done_reject_final = None
            phase_board_block_final = None
            _agent_ledger = getattr(agent, "ledger", None)
            if _agent_ledger is not None:
                try:
                    ledger_dict_final = _agent_ledger.to_dict()
                    ledger_block_final = _agent_ledger.to_prompt_block()
                except Exception:
                    pass
                ledger_done_reject_final = getattr(agent, "_ledger_last_done_reason", None)
                _pb_f = getattr(_agent_ledger, "phase_board", None)
                if _pb_f is not None:
                    try:
                        phase_board_block_final = _pb_f.render_block()
                    except Exception:
                        phase_board_block_final = None
            # v2 timeline snapshot at DONE
            timeline_render_final = None
            outcome_view_final = None
            timeline_event_snap_final = None
            recent_key_nodes_final: List[dict] = []
            _tl = getattr(agent, "timeline", None)
            if _tl:
                try:
                    from mm_agents.structagent.ledger.timeline import (
                        render_timeline_for_planner,
                        render_timeline_grouped_by_strategy,
                        render_outcome_view,
                        timeline_to_dict,
                    )
                    _strat_hist = getattr(agent, "_strategy_history", None) or []
                    if _strat_hist:
                        timeline_render_final = render_timeline_grouped_by_strategy(
                            _tl, _strat_hist,
                            n_recent_spans=3,
                            bullet_cap_per_span=5,
                            curator_strategy_notes=getattr(
                                agent, "_last_curator_strategy_notes", None),
                        )
                    else:
                        timeline_render_final = render_timeline_for_planner(
                            _tl, n_recent_events=8)
                    if _agent_ledger is not None:
                        outcome_view_final = render_outcome_view(
                            _agent_ledger.outcome_cache,
                            _agent_ledger.required_outcomes,
                            _agent_ledger.failed_paths,
                        )
                    timeline_event_snap_final = timeline_to_dict(_tl[-1:])
                    last_ev = _tl[-1]
                    if last_ev.steps:
                        last_step = last_ev.steps[-1]
                        for kn in last_step.key_nodes:
                            recent_key_nodes_final.append({
                                "kind": kn.kind,
                                "target": kn.target,
                                "evidence": (kn.evidence or "")[:300],
                                "confidence": kn.confidence,
                                "step_idx": kn.detected_at_step,
                            })
                except Exception:
                    pass
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": "DONE (terminate)",
                    "response": response,
                    "reward": 0,
                    "done": True,
                    "info": {},
                }))
                f.write("\n")
            steps_for_html.append({
                "screenshot_b64": screenshot_b64_final,
                "a11y_tree": a11y_text_final,
                "response": response,
                "action": "DONE (terminate)",
                "reward": 0,
                "done": True,
                "info": {},
                "plan": current_plan,
                "next_step_hint": next_step_hint,
                "observations": observations,
                "current_subgoal": current_subgoal,
                "planner_decision": planner_decision,
                "planner_response": planner_response,
                "ledger": ledger_dict_final,
                "ledger_block": ledger_block_final,
                "ledger_done_reject": ledger_done_reject_final,
                "phase_board": phase_board_block_final,
                "timeline_render": timeline_render_final,
                "outcome_view": outcome_view_final,
                "timeline_event": timeline_event_snap_final,
                "recent_key_nodes": recent_key_nodes_final,
            })
            done = True
            break
        for action_idx, action in enumerate(actions):
            # Capture the timestamp before executing the action
            action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S%f")
            logger.info("Step %d: %s", step_idx + 1, action)
            all_actions.append(str(action))
            # Snapshot the PRE-action screenshot so we can annotate it
            # with this action (CLICK marker / TYPE-HOTKEY-SCROLL labels)
            # before storing for trajectory.html.
            pre_action_screenshot_bytes = (
                obs.get("screenshot") if isinstance(obs, dict) else None
            )
            obs, reward, done, info = env.step(action, args.sleep_after_execution)

            logger.info("Reward: %.2f", reward)
            logger.info("Done: %s", done)
            # Append to traj.jsonl only (no separate PNGs)
            with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                }))
                f.write("\n")
            # Collect for HTML: screenshot is the PRE-action observation
            # (what the agent saw to decide), annotated with the action
            # marker so reviewers can see "agent looked at THIS screen
            # and clicked HERE". The post-action screen of this step
            # becomes the pre-action screen of the NEXT iteration.
            screenshot_b64 = None
            if pre_action_screenshot_bytes:
                try:
                    from mm_agents.structagent.utils.screenshot_annotator import annotate_screenshot_with_action
                    annotated = annotate_screenshot_with_action(
                        pre_action_screenshot_bytes, [action]
                    )
                    screenshot_b64 = base64.b64encode(annotated).decode("ascii")
                except Exception:
                    screenshot_b64 = base64.b64encode(pre_action_screenshot_bytes).decode("ascii")
            elif isinstance(obs, dict) and obs.get("screenshot"):
                # Fallback (very first step has no prior obs cached) —
                # use the post-action obs without annotation.
                screenshot_b64 = base64.b64encode(obs["screenshot"]).decode("ascii")
            # Linearize a11y tree (if present) so the trajectory.html can
            # display it alongside the screenshot for debugging.
            a11y_tree_text = None
            if isinstance(obs, dict) and obs.get("accessibility_tree"):
                try:
                    from mm_agents.structagent.utils.a11y_tree.accessibility_tree_handle import (
                        linearize_accessibility_tree, trim_accessibility_tree,
                    )
                    a11y_tree_text = linearize_accessibility_tree(obs["accessibility_tree"], "Ubuntu")
                    if a11y_tree_text:
                        a11y_tree_text = trim_accessibility_tree(a11y_tree_text, 300)
                except Exception:
                    a11y_tree_text = None
            # Progress Ledger snapshot (if the agent maintains one).
            ledger_dict = None
            ledger_block = None
            ledger_done_reject = None
            phase_board_block = None
            _agent_ledger = getattr(agent, "ledger", None)
            if _agent_ledger is not None:
                try:
                    ledger_dict = _agent_ledger.to_dict()
                    ledger_block = _agent_ledger.to_prompt_block()
                except Exception:
                    ledger_dict = None
                    ledger_block = None
                ledger_done_reject = getattr(agent, "_ledger_last_done_reason", None)
                # Multi-app: render the PhaseBoard for trajectory.html.
                _pb = getattr(_agent_ledger, "phase_board", None)
                if _pb is not None:
                    try:
                        phase_board_block = _pb.render_block()
                    except Exception:
                        phase_board_block = None

            # v2: snapshot the latest TimelineEvent + the StepRecord
            # this turn produced (if v2 working memory is active). This
            # makes trajectory.html / traj.jsonl reviewers able to see
            # the structured event view per step.
            timeline_render = None
            outcome_view = None
            timeline_event_snap = None
            recent_key_nodes: List[dict] = []
            _tl = getattr(agent, "timeline", None)
            if _tl:
                try:
                    from mm_agents.structagent.ledger.timeline import (
                        render_timeline_for_planner,
                        render_timeline_grouped_by_strategy,
                        render_outcome_view,
                        timeline_to_dict,
                    )
                    _strat_hist = getattr(agent, "_strategy_history", None) or []
                    if _strat_hist:
                        timeline_render = render_timeline_grouped_by_strategy(
                            _tl, _strat_hist,
                            n_recent_spans=3,
                            bullet_cap_per_span=5,
                            curator_strategy_notes=getattr(
                                agent, "_last_curator_strategy_notes", None),
                        )
                    else:
                        timeline_render = render_timeline_for_planner(
                            _tl, n_recent_events=8)
                    if _agent_ledger is not None:
                        outcome_view = render_outcome_view(
                            _agent_ledger.outcome_cache,
                            _agent_ledger.required_outcomes,
                            _agent_ledger.failed_paths,
                        )
                    # Persist only the LATEST event (compact); the rest
                    # is already in agent.timeline if needed.
                    timeline_event_snap = timeline_to_dict(_tl[-1:])
                    # Extract THIS turn's KeyNodes (the last step of the
                    # latest event) so the HTML can render the detector's
                    # reasoning prominently — without this the
                    # evidence text is buried inside the timeline_event
                    # JSON and reviewers can't easily debug why an
                    # outcome flipped to REVERTED.
                    last_ev = _tl[-1]
                    if last_ev.steps:
                        last_step = last_ev.steps[-1]
                        for kn in last_step.key_nodes:
                            recent_key_nodes.append({
                                "kind": kn.kind,
                                "target": kn.target,
                                "evidence": (kn.evidence or "")[:300],
                                "confidence": kn.confidence,
                                "step_idx": kn.detected_at_step,
                            })
                except Exception:
                    pass

            # cli_run output snapshot — if this turn's planner consumed
            # a /tmp/_last_cli_run.json sentinel, surface it in the HTML
            # so reviewers can verify which cli_run produced what, and
            # whether the planner correctly interpreted it.
            cli_run_output = None
            _cro = getattr(agent, "_last_cli_run_output", None)
            if _cro is not None:
                try:
                    cli_run_output = dict(_cro)
                except Exception:
                    cli_run_output = None

            steps_for_html.append({
                "screenshot_b64": screenshot_b64,
                "a11y_tree": a11y_tree_text,
                "response": response,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
                "plan": current_plan,
                "next_step_hint": next_step_hint,
                "observations": observations,
                "current_subgoal": current_subgoal,
                "planner_decision": planner_decision,
                "planner_response": planner_response,
                "ledger": ledger_dict,
                "ledger_block": ledger_block,
                "ledger_done_reject": ledger_done_reject,
                "phase_board": phase_board_block,
                "timeline_render": timeline_render,
                "outcome_view": outcome_view,
                "timeline_event": timeline_event_snap,
                "recent_key_nodes": recent_key_nodes,
                "cli_run_output": cli_run_output,
            })
            if done:
                logger.info("The episode is done.")
                break
        step_idx += 1
    if mind2web_eval.text_answer_eval_mode(example):
        # Mind2Web web task: score with the answer-blind Online-Mind2Web grader
        # (the task's evaluator is a stub; env.evaluate() does not apply). The
        # grader judges from the raw intent + action history + key screenshots
        # and writes result.txt / answer.txt / judge.json itself.
        result = mind2web_eval.score_text_answer(
            example=example,
            instruction=instruction,
            agent_answer="",
            screenshots_b64=mind2web_eval.collect_step_screenshots_b64(
                example_result_dir),
            env=env,
            example_result_dir=example_result_dir,
            action_history=all_actions,
        )
    else:
        # Settle window before evaluation. Chrome SPA tasks (e.g. flight
        # search results) often hydrate via API + skeleton → real content,
        # which routinely takes longer than 20s under VM CPU pressure
        # (ffmpeg recording + LLM services). 45s avoids the "DOM still
        # skeleton when eval reads it" false-negative seen on tasks like
        # fc6d8143.
        time.sleep(45)
        result = env.evaluate()
    if float(result) < 0:
        # -1 = evaluator/env ERROR sentinel (a getter returned None, a metric
        # could not run, the VM was unreachable at eval time, ...) — NOT a real
        # 0. Do NOT write result.txt, so get_unfinished() treats the task as
        # unfinished and re-runs it next launch instead of locking in a false 0.
        # (env.reset/step/evaluate that RAISE are already handled by run.py's
        # except → also no result.txt.) Recording + html still close below so
        # the partial run is left for debugging.
        logger.warning("env.evaluate() returned %s — env/eval error; NOT "
                       "recording result, task will re-run.", result)
    else:
        result = max(0.0, min(1.0, float(result)))  # Clamp to [0, 1]
        logger.info("Result: %.2f", result)
        scores.append(result)
        with open(os.path.join(example_result_dir, "result.txt"), "w", encoding="utf-8") as f:
            f.write(f"{result}\n")
        # Dump the VM-side files the evaluator just compared (post-postconfig
        # state) into <result_dir>/eval_files/. Lets offline debugging diff
        # the agent's actual output against the gold — e.g. for impress,
        # ``compare_pptx_files(out, gold, enable_debug=True)`` prints the
        # exact first mismatching field. No-op when the task has no
        # ``vm_file`` result entries.
        _dump_eval_result_files(env, example, example_result_dir)
        # Log task completion to results.json
        log_task_completion(example, result, example_result_dir, args)

    # Write trajectory report HTML (always — keep the partial run viewable).
    write_trajectory_html(
        example_result_dir,
        instruction=instruction,
        steps=steps_for_html,
        final_score=result,
        memory_text=memory_text,
    )

    env.controller.end_recording(os.path.join(example_result_dir, "recording.mp4"))

    # Detach this task's FileHandler from per-task + shared loggers so the
    # next task's logs don't bleed into this task's runtime.log.
    teardown_logger(example, example_result_dir)


_SHARED_LOGGER_NAMES = (
    "mm_agents.debug",
    "mm_agents.diagnosis.data_logger",
    # StructAgent package root — Phase C moved the subsystems under
    # mm_agents.structagent.{ledger,actions,core,...}. Most of those
    # modules call ``logging.getLogger(__name__)``, so attaching the
    # FileHandler to the package root routes every child's records
    # (init_ledger / verifiers / timeline / search / rescue /
    # ops_lib_remote / decomposer warnings / unified memory retriever)
    # into runtime.log via the standard logging hierarchy.
    "mm_agents.structagent",
    # points_actor uses an explicit ``getLogger("mm_agents.points_actor")``
    # literal (not __name__), so a sibling entry is still needed.
    "mm_agents.points_actor",
    # CoderAgent DEBUG REPLAN / sliding-window trim messages.
    "mm_agents.os_symphony",
    # The moved planner / actor / verifier / recovery / action_compile /
    # llm_client modules all log under ``desktopenv.qwen25vl_agent_planner``.
    # Without this entry the per-step [PA-Planner] / [PA-Actor] /
    # [Ledger] / [KeyNode] lines vanish from runtime.log (the only
    # output that previously survived was [Perceiver] because the
    # perceiver module uses its own ``mm_agents.structagent.perceiver``
    # path, covered by the causal_agent root above).
    "desktopenv.qwen25vl_agent_planner",
    # Two-stage grounding + UI-Ins grounding-model client warnings.
    "desktopenv.grounding",
    # KeyNode detector standalone logger.
    "desktopenv.key_node_detector",
    # Screenshot annotator (planner-replay click overlays).
    "desktopenv.screenshot_annotator",
)


def setup_logger(example, example_result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.example.{example['id']}")
    runtime_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(os.path.join(example_result_dir, "runtime.log"))
    runtime_logger.addHandler(file_handler)
    # Also route online-debug & memory-retriever module logs into runtime.log.
    # Attach only to the PARENT packages; child loggers (e.g. .controller,
    # .apply.layer2) propagate up via Python logging hierarchy and will
    # reuse the parent's handler — avoids the double-write we'd get if we
    # attached the same handler to both parent and every child.
    # IMPORTANT: first strip any stale FileHandler left over from a PREVIOUS
    # task. Shared loggers persist across tasks in the same process; without
    # this cleanup, task N's logs get duplicated into task N-1's runtime.log
    # (observed log pollution: session_vlc_efcf0d81_* entries leaking into
    # aa4b5023's runtime.log).
    for logger_name in _SHARED_LOGGER_NAMES:
        mod_logger = logging.getLogger(logger_name)
        mod_logger.setLevel(logging.INFO)
        for h in list(mod_logger.handlers):
            if isinstance(h, logging.FileHandler) and \
               getattr(h, "baseFilename", "") != file_handler.baseFilename:
                try:
                    h.close()
                except Exception:
                    pass
                mod_logger.removeHandler(h)
        has_handler = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "") == file_handler.baseFilename
            for h in mod_logger.handlers
        )
        if not has_handler:
            mod_logger.addHandler(file_handler)
    return runtime_logger


def teardown_logger(example, example_result_dir):
    """Remove this task's FileHandlers from per-task + shared loggers.

    Called at the end of run_single_example so handlers don't leak into
    subsequent tasks (which would cause the log-pollution bug).
    """
    target_path = os.path.join(example_result_dir, "runtime.log")
    # Per-task logger
    per_task = logging.getLogger(f"desktopenv.example.{example['id']}")
    for h in list(per_task.handlers):
        if isinstance(h, logging.FileHandler) and \
           getattr(h, "baseFilename", "") == target_path:
            try:
                h.close()
            except Exception:
                pass
            per_task.removeHandler(h)
    # Shared loggers
    for logger_name in _SHARED_LOGGER_NAMES:
        mod_logger = logging.getLogger(logger_name)
        for h in list(mod_logger.handlers):
            if isinstance(h, logging.FileHandler) and \
               getattr(h, "baseFilename", "") == target_path:
                try:
                    h.close()
                except Exception:
                    pass
                mod_logger.removeHandler(h)
