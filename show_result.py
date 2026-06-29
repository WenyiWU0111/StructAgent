import os
import json
import argparse


def get_result(action_space, use_model, observation_type, result_dir, show_detailed_scores=False, version_num=None, target_dir_override=None):
    """
    Calculate and display evaluation results from OSWorld benchmark runs.

    Args:
        action_space (str): Action space used (e.g., "pyautogui", "computer_13")
        use_model (str): Model name used for evaluation (e.g., "gpt-4o", "claude-3")
        observation_type (str): Observation type used (e.g., "screenshot", "a11y_tree")
        result_dir (str): Root directory containing results
        show_detailed_scores (bool): If True, show detailed scores per domain in format "score/total"

    Returns:
        list: List of all individual task results, or None if no results found
    """
    # --target_dir bypasses the {model}_{version} path convention so layouts
    # from other harnesses (results_baselines/*) can be aggregated directly.
    if target_dir_override:
        target_dir = target_dir_override
    else:
        target_dir = os.path.join(result_dir, action_space, observation_type,f'{use_model}_{version_num}')
    print('target_dir:', target_dir)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    # Look up task instructions from evaluation_examples/examples/{domain}/{id}.json.
    # Resolved relative to this script so the lookup works regardless of CWD.
    eval_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "evaluation_examples", "examples",
    )

    def _load_instruction(domain: str, example_id: str) -> str:
        p = os.path.join(eval_root, domain, f"{example_id}.json")
        try:
            with open(p, "r") as fh:
                return (json.load(fh).get("instruction") or "").strip()
        except Exception:
            return ""

    all_result = []
    domain_result = {}
    all_result_for_analysis = {}

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        result_path = os.path.join(example_path, "result.txt")
                        raw = open(result_path, "r").read().strip()
                        try:
                            score = float(raw)
                        except (ValueError, TypeError):
                            try:
                                score = float(eval(raw))
                            except Exception:
                                # empty/corrupt result.txt (e.g. interrupted
                                # write) — skip it; delete the file to let the
                                # runner redo this task on resume.
                                print(f"[WARN] unparsable result.txt skipped: {result_path} (content={raw!r})")
                                continue
                        if domain not in domain_result:
                            domain_result[domain] = []
                        domain_result[domain].append(score)

                        if domain not in all_result_for_analysis:
                            all_result_for_analysis[domain] = {}
                        all_result_for_analysis[domain][example_id] = {
                            "score":       score,
                            "instruction": _load_instruction(domain, example_id),
                        }

                        all_result.append(score)

    if show_detailed_scores:
        # Print detailed scores in format "score/total" for each domain
        result_order = ["chrome", "gimp", "libreoffice_calc", "libreoffice_impress",
                       "libreoffice_writer", "multi_apps", "os", "thunderbird", "vlc", "vs_code",
                       "webvoyager", "mind2web"]
        output_row = []
        for d in result_order:
            if d in domain_result:
                output_row.append(f"{round(sum(domain_result[d]),2)}/{len(domain_result[d])}")
            else:
                output_row.append("0.00/0")
        print(" ".join(output_row))
    else:
        # Print standard per-domain statistics
        for domain in domain_result:
            print("Domain:", domain, "Runned:", sum(domain_result[domain]),  '/', len(domain_result[domain]), "Success Rate:",
                  sum(domain_result[domain]) / len(domain_result[domain]) * 100, "%")

    # ── Per-site / per-split sub-breakdown for text-answer benchmarks ──
    # WebVoyager example_id  : "webvoyager_Coursera--0" → site "Coursera"
    # Mind2Web example_id    : "mind2web_domain_Info_Health_3"
    #                          "mind2web_domain_Service_Travel_7"
    #                          "mind2web_website_Entertainment_11"
    #                          → split = first segment after "mind2web_"
    def _sub_bucket_webvoyager(example_id: str) -> str:
        # "webvoyager_<site>--<n>"
        body = example_id[len("webvoyager_"):]
        return body.split("--", 1)[0] if "--" in body else "_other"

    def _sub_bucket_mind2web(example_id: str) -> str:
        # "mind2web_<split>_<topic>_<n>"
        body = example_id[len("mind2web_"):]
        # Known splits emitted by the converter:
        for prefix in ("domain_Info", "domain_Service", "website"):
            if body.startswith(prefix):
                return prefix
        return "_other"

    _sub_dispatch = {
        "webvoyager": ("site", _sub_bucket_webvoyager),
        "mind2web":   ("split", _sub_bucket_mind2web),
    }
    for dom, (label, fn) in _sub_dispatch.items():
        if dom not in all_result_for_analysis:
            continue
        buckets: dict = {}
        for example_id, rec in all_result_for_analysis[dom].items():
            key = fn(example_id) or "_other"
            buckets.setdefault(key, []).append(rec["score"])
        if not buckets:
            continue
        print(f"\n--- {dom} per-{label} ---")
        for k in sorted(buckets):
            scores = buckets[k]
            total = sum(scores)
            print(f"  {k:<24} {total:>5.2f} / {len(scores):>3}  "
                  f"({100 * total / len(scores):>5.1f}%)")

    print(">>>>>>>>>>>>>")

    # Print category-level statistics
    if all(d in domain_result for d in ["libreoffice_calc", "libreoffice_impress", "libreoffice_writer"]):
        print("Office", "Success Rate:", sum(
            domain_result["libreoffice_calc"] + domain_result["libreoffice_impress"] + domain_result[
                "libreoffice_writer"]) / len(
            domain_result["libreoffice_calc"] + domain_result["libreoffice_impress"] + domain_result[
                "libreoffice_writer"]) * 100, "%")

    if all(d in domain_result for d in ["vlc", "thunderbird", "chrome"]):
        print("Daily", "Success Rate:",
              sum(domain_result["vlc"] + domain_result["thunderbird"] + domain_result["chrome"]) / len(
                  domain_result["vlc"] + domain_result["thunderbird"] + domain_result["chrome"]) * 100, "%")

    if all(d in domain_result for d in ["gimp", "vs_code"]):
        print("Professional", "Success Rate:", sum(domain_result["gimp"] + domain_result["vs_code"]) / len(
            domain_result["gimp"] + domain_result["vs_code"]) * 100, "%")

    with open(os.path.join(target_dir, "all_result.json"), "w") as f:
        json.dump(all_result_for_analysis, f, indent=2)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Runned:", len(all_result), "Current Success Rate:",
              round(sum(all_result) / len(all_result) * 100, 2), "%",
              f"{round(sum(all_result), 2)}", "/", str(len(all_result)))
        return all_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate and display OSWorld evaluation results"
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="pyautogui",
        help="Action space used (e.g., 'pyautogui', 'computer_13')"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name used for evaluation (e.g., 'gpt-4o', 'claude-3')"
    )
    parser.add_argument(
        "--observation_type",
        type=str,
        default="screenshot",
        help="Observation type used (e.g., 'screenshot', 'a11y_tree', 'som')"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results",
        help="Root directory containing results (default: ./results)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed scores per domain in format 'score/total'"
    )
    parser.add_argument(
        "--version_num",
        type=str,
        default="1",
        help="Version number of the experiment"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="Aggregate this directory directly (expects <domain>/<example_id>/result.txt below it); bypasses --model/--version path construction"
    )

    args = parser.parse_args()

    get_result(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        args.detailed,
        args.version_num,
        target_dir_override=args.target_dir,
    )
