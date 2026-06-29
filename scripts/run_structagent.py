from __future__ import annotations
import argparse
import datetime
import json
import logging
import os
import sys
import signal
import time
from typing import List, Dict
import math
from tqdm import tqdm
from multiprocessing import Process, Manager
from multiprocessing import current_process

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from desktop_env.api_env import APIDesktopEnv
from mm_agents.structagent import StructAgent

# Web (Mind2Web) tasks run through the same loop as OSWorld but are scored by
# the answer-blind Online-Mind2Web grader (mind2web_eval) instead of
# env.evaluate(). They are tagged so run_single_example picks the right scorer.
TEXT_ANSWER_DOMAINS = {"mind2web"}


def _is_text_answer_domain(domain: str) -> bool:
    return domain in TEXT_ANSWER_DOMAINS

# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False

# import wandb

# load the environment variables from .env file
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()

#  Logger Configs {{{ #
def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )

    # lm config
    parser.add_argument("--model", type=str, default="vllm_qwen35-vl")
    parser.add_argument("--version_num", type=str, default="v1", help="Version suffix for result folder: {model}_planner_{version_num}")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--add_thought_prefix", action="store_true", help="Add thought prefix to the response")
    # Actor: decomposer LLM (inline grounding)
    parser.add_argument("--planner_max_images", type=int, default=5, help="Max screenshots for planner context")
    parser.add_argument("--verifier_model", type=str, default=None,
                        help="Alias for the VERIFICATION model (init_ledger spec authoring + "
                             "per-step outcome checks + done-auditor). Falls back to the planner "
                             "model when omitted (single-model run = zero change). One of the 3 "
                             "canonical models: planner / verifier / grounding.")
    parser.add_argument("--decomposer_model", type=str, default=None,
                        help="Alias for the decomposer vision LLM (e.g. vllm_qwen35-vl). "
                             "Falls back to the planner model when omitted.")
    parser.add_argument("--decomposer_api_url", type=str, default=None,
                        help="Decomposer vllm server URL. Falls back to the planner URL.")
    parser.add_argument("--decomposer_system_path", type=str, default=None,
                        help="Path to an external A2 decomposer system-prompt .txt. "
                             "None/empty -> built-in DECOMPOSER_SYSTEM_TEMPLATE.")
    parser.add_argument("--disable_a11y", action="store_true",
                        help="Turn off BOTH a11y-actor routing (planner) AND "
                             "a11y-tree fetching (env). Pure vision-only run, "
                             "useful when comparing grounding models.")
    parser.add_argument("--enable_done_auditor", action="store_true",
                        help="Enable T3 Done-Auditor: after the structured "
                             "ledger gate accepts DONE, run an adversarial "
                             "fresh-context LLM call against the task text + "
                             "raw a11y + screenshots. Verdict != PASS flips "
                             "acceptance to False and stages a force_replan "
                             "with category 'done_audit_failed'. Default off. "
                             "Also honored via env ENABLE_DONE_AUDITOR=1.")
    parser.add_argument("--done_auditor_model", type=str,
                        default="vllm_qwen35-vl",
                        help="Model alias for the Done-Auditor. Default is "
                             "the same Qwen3.5-9B as the planner (cheap "
                             "auditor models rubber-stamp per plan Part I.2). "
                             "Only consulted when --enable_done_auditor is set.")
    parser.add_argument("--done_auditor_max_per_task", type=int,
                        default=3,
                        help="Hard cap on auditor LLM calls per task (default "
                             "3). Decremented on every call (PASS too) so a "
                             "thrashing auditor can't burn unbounded LLM time.")
    # NB: ``--disable_specialized_executors`` was removed. All specialized
    # executors (LO/OS/VLC/VSCode/Thunderbird/Wiki/MultiHop/MultiApp) have
    # been deleted from the planner. Tasks now route through the planner-
    # actor pure-GUI / Phase-A-terminal path uniformly.
    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")  
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")
    # aws config
    parser.add_argument(
        "--region", type=str, default="us-east-1", help="AWS region for the VM"
    )
    parser.add_argument(
        "--provider_name", type=str, default="docker", choices=["aws", "virtualbox", "vmware", "docker", "azure", "API"], help="Provider name"
    )
    parser.add_argument(
        "--client_password", type=str, default="", help="Client password"
    )
    # API provider config
    parser.add_argument(
        "--env_url", type=str, default="http://127.0.0.1", help="Base URL of the environment server (used when provider_name=API)"
    )
    parser.add_argument(
        "--env_manager_port", type=int, default=11001, help="Port of the env API manager (used when provider_name=API)"
    )
    parser.add_argument(
        "--env_port", type=int, default=None, help="Direct port to a single env API, skipping the manager (used when provider_name=API)"
    )
    parser.add_argument(
        "--screen_width", type=int, default=1920, help="Screen width"
    )
    parser.add_argument(
        "--screen_height", type=int, default=1080, help="Screen height"
    )
    args = parser.parse_args()
    return args

args = config()  # Get command line arguments first

logger = logging.getLogger()
log_level = getattr(logging, args.log_level.upper())
logger.setLevel(log_level)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(log_level)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
#  }}} Logger Configs #

logger = logging.getLogger("desktopenv.experiment")


def distribute_tasks(test_all_meta: dict) -> List[tuple]:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks


def process_signal_handler(signum, frame, env_idx):
    """Signal handler for child processes to gracefully shut down their environments."""
    logger.info(f"Process {env_idx + 1} received signal {signum}. Shutting down...")
    
    # Get the active_environments from the caller's frame
    local_vars = frame.f_locals
    active_environments = local_vars.get('active_environments', [])
    
    # Close environment in the current process context
    for env in active_environments:
        if env is not None:
            try:
                logger.info(f"Process {env_idx + 1} closing environment...")
                env.close()
                logger.info(f"Process {env_idx + 1} environment closed successfully")
            except Exception as e:
                logger.error(f"Process {env_idx + 1} error closing environment: {e}")
    
    logger.info(f"Process {env_idx + 1} shutdown complete. Exiting.")
    sys.exit(0)


def run_env_tasks(task_queue, args: argparse.Namespace, shared_scores: list):
    active_environments = []
    env = None
    try:
        REGION = args.region
        screen_size = (args.screen_width, args.screen_height)
        if args.provider_name == "API":
            env = APIDesktopEnv(
                base_url=args.env_url,
                env_port=args.env_port,
                manager_port=args.env_manager_port,
                action_space=args.action_space,
                screen_size=screen_size,
                headless=args.headless,
                # --disable_a11y turns off the a11y-tree fetch (saves a
                # per-step HTTP roundtrip when the tree is unused).
                require_a11y_tree=not args.disable_a11y,
                os_type="Ubuntu",
            )
        else:
            snapshot_name = "init_state"
            if args.provider_name == "aws":
                from desktop_env.providers.aws.manager import IMAGE_ID_MAP
                ami_id = IMAGE_ID_MAP[REGION].get(screen_size, IMAGE_ID_MAP[REGION][(1920, 1080)])
                snapshot_name = ami_id
            env = DesktopEnv(
                path_to_vm=args.path_to_vm,
                action_space=args.action_space,
                provider_name=args.provider_name,
                region=REGION,
                snapshot_name=snapshot_name,
                screen_size=screen_size,
                headless=args.headless,
                os_type="Ubuntu",
                # --disable_a11y turns off the env's a11y-tree fetch (saves
                # a per-step HTTP roundtrip when the tree is unused).
                require_a11y_tree=not args.disable_a11y,
                enable_proxy=False,  # Set True only if evaluation_examples/settings/proxy/dataimpulse.json has real credentials; else Chrome shows ERR_PROXY_AUTH_UNSUPPORTED
                client_password=args.client_password,
            )
        active_environments.append(env)
        agent_kwargs = dict(
            model=args.model,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            action_space=args.action_space,
            add_thought_prefix=args.add_thought_prefix,
            planner_max_images=getattr(args, "planner_max_images", 5),
            verifier_model=getattr(args, "verifier_model", None),   # None → planner model
            decomposer_model=getattr(args, "decomposer_model", None),
            decomposer_api_url=getattr(args, "decomposer_api_url", None),
            decomposer_system_path=getattr(args, "decomposer_system_path", None),
            # DONE auditor (off by default)
            enable_done_auditor=getattr(args, "enable_done_auditor", False),
            done_auditor_model=getattr(args, "done_auditor_model",
                                       "vllm_qwen35-vl"),
            done_auditor_max_per_task=getattr(
                args, "done_auditor_max_per_task", 3),
        )
        agent = StructAgent(**agent_kwargs)
        logger.info(f"Process {current_process().name} started.")
        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            domain, example_id = item
            try:
                config_file = os.path.join(
                    args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
                )
                if not os.path.exists(config_file):
                    continue
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)
                logger.info(f"[{current_process().name}][Domain]: {domain}")
                logger.info(f"[{current_process().name}][Example ID]: {example_id}")
                logger.info(f"[{current_process().name}][Instruction]: {example['instruction']}")
                example_result_dir = os.path.join(
                    args.result_dir,
                    args.action_space,
                    args.observation_type,
                    args.result_model,
                    domain,
                    example_id,
                )
                os.makedirs(example_result_dir, exist_ok=True)
                try:
                    if _is_text_answer_domain(domain):
                        # Mind2Web web tasks run through the SAME loop as
                        # OSWorld; only the scoring differs. Tag the example so
                        # run_single_example scores it with the answer-blind
                        # Online-Mind2Web grader instead of env.evaluate().
                        example["_text_answer_domain"] = domain
                        example["_eval_mode"] = "llm_judge"
                    lib_run_single.run_single_example(
                        agent,
                        env,
                        example,
                        args.max_steps,
                        example["instruction"],
                        args,
                        example_result_dir,
                        shared_scores,
                    )
                except Exception as e:
                    import traceback
                    logger.error(f"Exception in {current_process().name} {domain}/{example_id}: {e}")
                    logger.error(traceback.format_exc())
                    try:
                        env.controller.end_recording(
                            os.path.join(example_result_dir, "recording.mp4")
                        )
                    except Exception as rec_e:
                        logger.error(f"Failed to end recording: {rec_e}")
                    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                        f.write(
                            json.dumps(
                                {"Error": f"{domain}/{example_id} - {e}"}
                            )
                        )
                        f.write("\n")
            except Exception as e:
                logger.error(f"Task-level error in {current_process().name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Process-level error in {current_process().name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"{current_process().name} cleaning up environment...")
        try:
            if env:
                env.close()
                logger.info(f"{current_process().name} environment closed successfully")
        except Exception as e:
            logger.error(f"{current_process().name} error during environment cleanup: {e}")


def signal_handler(signum, frame):
    """Handle termination signals (SIGINT, SIGTERM) to gracefully shutdown environments."""
    global is_terminating, active_environments, processes
    
    # Avoid duplicate handling
    if is_terminating:
        return
    
    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")
    
    # Close all registered environments in the main process
    for env in active_environments:
        try:
            logger.info(f"Closing environment...")
            env.close()
            logger.info(f"Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    
    # Send termination signal to all child processes first
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Sending termination signal to process {p.name}...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error sending termination signal to process: {e}")
    
    # Allow a short time for processes to handle their own cleanup
    time.sleep(1)
    
    # Forcefully terminate any processes that didn't exit
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Forcefully terminating process {p.name}...")
                import signal as sig
                os.kill(p.pid, sig.SIGKILL)
            except Exception as e:
                logger.error(f"Error forcefully terminating process: {e}")
    
    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    logger.info("Args: %s", args)
    all_tasks = distribute_tasks(test_all_meta)
    logger.info(f"Total tasks: {len(all_tasks)}")
    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)
        num_envs = args.num_envs
        processes = []
        for i in range(num_envs):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores),
                name=f"EnvProcess-{i+1}"
            )
            p.daemon = True
            p.start()
            processes.append(p)
            logger.info(f"Started process {p.name} with PID {p.pid}")
        try:
            while True:
                alive_count = 0
                for idx, p in enumerate(processes):
                    if not p.is_alive():
                        logger.warning(f"Process {p.name} died, restarting...")
                        new_p = Process(
                            target=run_env_tasks,
                            args=(task_queue, args, shared_scores),
                            name=f"EnvProcess-Restart-{idx+1}"
                        )
                        new_p.daemon = True
                        new_p.start()
                        processes[idx] = new_p
                        logger.info(f"Restarted process {new_p.name} with PID {new_p.pid}")
                    else:
                        alive_count += 1
                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                time.sleep(5)
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt. Initiating graceful shutdown...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while waiting for processes: {e}", exc_info=True)
            for p in processes:
                if p.is_alive():
                    try:
                        logger.info(f"Terminating process {p.name} due to error...")
                        p.terminate()
                    except Exception as term_e:
                        logger.error(f"Error terminating process {p.name}: {term_e}")
            raise
        scores = list(shared_scores)
    logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0}")


def get_unfinished(
    action_space, use_model, observation_type, result_dir, total_file_json
):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    # Only scan domains relevant to the *current* run. Scanning every domain
    # under the shared result_dir caused a race when multiple runs share the
    # same VERSION / result_model: an incoming run would `os.remove` files
    # belonging to another run's in-flight task dir (which hasn't produced
    # result.txt yet), orphaning the active worker's FileHandler fd.
    domains_to_scan = list(total_file_json.keys())
    finished = {}
    for domain in domains_to_scan:
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue
        finished[domain] = []
        for example_id in os.listdir(domain_path):
            if example_id == "onboard":
                continue
            example_path = os.path.join(domain_path, example_id)
            if os.path.isdir(example_path):
                if "result.txt" not in os.listdir(example_path):
                    # empty all entries under example_id (files AND
                    # subdirs — e.g. perceiver_debug/, scripts/ — so a
                    # re-run starts from a clean slate)
                    import shutil as _sh
                    for entry in os.listdir(example_path):
                        full = os.path.join(example_path, entry)
                        try:
                            if os.path.isdir(full) and not os.path.islink(full):
                                _sh.rmtree(full)
                            else:
                                os.remove(full)
                        except FileNotFoundError:
                            # Another concurrent run may have processed the
                            # same path; ignore the race rather than crash.
                            pass
                else:
                    finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        # empty all files under example_id
                        try:
                            all_result.append(
                                float(
                                    open(
                                        os.path.join(example_path, "result.txt"), "r"
                                    ).read()
                                )
                            )
                        except:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Register signal handlers for graceful termination
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal
    
    try:
        args = config()
        base_model = args.model.rstrip("_planner") if str(args.model).endswith("_planner") else args.model
        args.result_model = f"{base_model}_planner_{args.version_num}"
        # save args to json in result_dir/action_space/observation_type/result_model/args.json
        # Includes a snapshot of search/ledger-related env vars so the
        # run's actual configuration (e.g. whether DISABLE_SEARCH was on
        # at launch) is recoverable post-hoc — the directory name is
        # user-chosen and doesn't necessarily reflect env state.
        path_to_args = os.path.join(
            args.result_dir,
            args.action_space,
            args.observation_type,
            args.result_model,
            "args.json",
        )
        os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
        args_dict = dict(vars(args))
        env_keys = [
            "DISABLE_SEARCH", "MAX_SEARCH_CALLS",
            "LEDGER_INIT_SEARCH_ROUNDS", "PLANNER_SEARCH_ROUNDS",
            "DISABLE_LEDGER", "MEMORY_VERSION",
            "GEMINI_SEARCH_MODEL",
        ]
        args_dict["env"] = {k: os.environ.get(k, "(unset)") for k in env_keys}
        # Don't store the actual key, just whether it's set, for safety.
        args_dict["env"]["GEMINI_API_KEY"] = (
            "set" if os.environ.get("GEMINI_API_KEY") else "MISSING"
        )
        with open(path_to_args, "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=4)

        with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)

        if args.domain != "all":
            if args.domain not in test_all_meta:
                available = sorted(test_all_meta.keys())
                # Common slip-up: meta path was overridden but DOMAIN
                # default ('webvoyager') still applies. Give the user
                # an actionable error instead of a KeyError.
                raise SystemExit(
                    f"[run_multienv] DOMAIN={args.domain!r} not found in "
                    f"meta {args.test_all_meta_path!r}. "
                    f"Available keys: {available}. "
                    f"Pass DOMAIN={available[0]!r} if you meant the only "
                    "key in this meta."
                    if len(available) == 1 else
                    f"[run_multienv] DOMAIN={args.domain!r} not in meta "
                    f"{args.test_all_meta_path!r}. Available: {available}"
                )
            test_all_meta = {args.domain: test_all_meta[args.domain]}

        test_file_list = get_unfinished(
            args.action_space,
            args.result_model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )
        left_info = ""
        for domain in test_file_list:
            left_info += f"{domain}: {len(test_file_list[domain])}\n"
        logger.info(f"Left tasks:\n{left_info}")

        get_result(
            args.action_space,
            args.result_model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )
        test(args, test_file_list)
    except KeyboardInterrupt:
        logger.info("Main process received KeyboardInterrupt.")
        # Signal handler will take care of cleanup
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}", exc_info=True)
        # Also trigger cleanup for unhandled exceptions
        signal_handler(signal.SIGTERM, None)
    finally:
        # Final cleanup in case any environments or processes remain
        logger.info("Main process final cleanup...")
        for env in active_environments:
            if env is not None:
                try:
                    logger.info(f"Closing environment in final cleanup...")
                    env.close()
                    logger.info(f"Environment closed successfully in final cleanup")
                except Exception as e:
                    logger.error(f"Error during final environment cleanup: {e}")
        
        # First try gentle termination
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info(f"Terminating process {p.name}...")
                    p.terminate()
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")
        
        # Wait a moment for processes to terminate
        time.sleep(1)
        
        # Then force kill if needed
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info(f"Force killing process {p.name}...")
                    os.kill(p.pid, signal.SIGKILL)
                    logger.info(f"Process {p.name} force killed")
                except Exception as e:
                    logger.error(f"Error force killing process: {e}")
