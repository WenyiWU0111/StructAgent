#!/usr/bin/env python3
"""Download and install StructAgent's experience-memory banks.

The banks (FAISS indexes + recipe/exemplar payloads, ~64 MB) are mined offline
from solved trajectories and are hosted on Hugging Face rather than committed to
git. They are OPTIONAL — the agent runs without them (memory retrieval silently
no-ops); they only add retrieved planning/verification hints.

Usage:
    python scripts/download_memory.py                       # default HF repo
    python scripts/download_memory.py --repo <user>/structagent-memory
    HF_TOKEN=... python scripts/download_memory.py          # if the repo is private

Installs into:  results/{unified_memory, planner_experience, verifier_memory}/
Then enable with:  MEMORY=on bash scripts/run.sh
"""
import argparse
import os
import sys
import tarfile

DEFAULT_REPO = os.environ.get("STRUCTAGENT_MEMORY_REPO", "WenyiWU0111/structagent-memory")
TARBALL = "structagent_memory.tar.gz"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="Hugging Face dataset repo id hosting the tarball.")
    ap.add_argument("--file", default=TARBALL, help="Tarball filename in the repo.")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please `pip install huggingface_hub` first.", file=sys.stderr)
        return 1

    print(f"Downloading {args.file} from hf://datasets/{args.repo} ...")
    path = hf_hub_download(repo_id=args.repo, filename=args.file,
                           repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    print(f"Extracting into {REPO_ROOT}/results/ ...")
    with tarfile.open(path) as tf:
        tf.extractall(REPO_ROOT)  # tarball contains results/{unified_memory,...}

    installed = [d for d in ("unified_memory", "planner_experience", "verifier_memory")
                 if os.path.isdir(os.path.join(REPO_ROOT, "results", d))]
    print("Installed memory banks:", ", ".join(installed) or "(none?)")
    print("Enable with:  MEMORY=on bash scripts/run.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
