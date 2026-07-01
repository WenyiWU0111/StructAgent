#!/bin/bash
# StructAgent — single entry point for OSWorld and Mind2Web.
#
# The DEFAULTS below are the full agent used in the paper. Every capability is
# a switch you can flip from the command line, e.g.:
#
#   MODEL=vllm_qwen35-vl NUM_ENVS=8 bash scripts/run.sh                 # OSWorld (default split)
#   TEST_ALL_META_PATH=evaluation_examples/test_mind2web.json bash scripts/run.sh   # Mind2Web
#   USE_FAILURE_ATTRIBUTION=0 ENABLE_DONE_AUDITOR=0 bash scripts/run.sh # ablate features
#
# See docs/CONFIG.md for the full switch reference.

set -e
cd "$(dirname "$0")/.."

# Load .env (API keys for OpenRouter / DashScope)
[[ -f .env ]] && { set -a; source .env; set +a; }
mkdir -p logs

# ── Models ───────────────────────────────────────────────────────────────────
MODEL="${MODEL:-vllm_qwen35-vl}"                  # planner / actor / grounding
VERIFIER_MODEL="${VERIFIER_MODEL:-$MODEL}"        # verifier + DONE-auditor; default = planner

# ── Environment ──────────────────────────────────────────────────────────────
PROVIDER="${PROVIDER:-docker}"                    # docker | vmware | aws | API
NUM_ENVS="${NUM_ENVS:-5}"                         # parallel environments
MAX_STEPS="${MAX_STEPS:-100}"
DOMAIN="${DOMAIN:-all}"
TEST_ALL_META_PATH="${TEST_ALL_META_PATH:-evaluation_examples/test_nogdrive.json}"

# ── Agent capabilities — DEFAULTS = the paper's full StructAgent ──────────────
export PERCEIVER_ENABLED="${PERCEIVER_ENABLED:-1}"                 # structured screen perception
export ENABLE_FEASIBILITY_VERDICT="${ENABLE_FEASIBILITY_VERDICT:-1}"  # declare infeasible after K replans
export USE_FAILURE_ATTRIBUTION="${USE_FAILURE_ATTRIBUTION:-1}"     # role-attributed failure → routed recovery
export ENABLE_STUCK_DIAGNOSIS_INJECTION="${ENABLE_STUCK_DIAGNOSIS_INJECTION:-1}"  # inject diagnosis into replan
ENABLE_DONE_AUDITOR="${ENABLE_DONE_AUDITOR:-1}"                    # evidence-gated DONE auditor
export ENABLE_DONE_AUDITOR
DONE_AUDITOR_MAX_PER_TASK="${DONE_AUDITOR_MAX_PER_TASK:-3}"
export WEB_DOM_INJECT="${WEB_DOM_INJECT:-1}"                       # inject Chrome DOM (checker-source) on web steps

# ── Experience memory — OFF by default: requires prebuilt FAISS banks ─────────
#    (see docs/CONFIG.md "Experience memory"). MEMORY=on enables both layers.
MEMORY="${MEMORY:-off}"
if [[ "$MEMORY" == "on" ]]; then
    : "${ENABLE_PLANNER_EXPERIENCE_MEMORY:=1}"
    : "${ENABLE_VERIFIER_EXPERIENCE_MEMORY:=1}"
fi
export ENABLE_PLANNER_EXPERIENCE_MEMORY="${ENABLE_PLANNER_EXPERIENCE_MEMORY:-0}"
export ENABLE_VERIFIER_EXPERIENCE_MEMORY="${ENABLE_VERIFIER_EXPERIENCE_MEMORY:-0}"
export MEMORY_TOP_K="${MEMORY_TOP_K:-10}"

# ── Ablation knobs (advanced; see docs/CONFIG.md) ────────────────────────────
export DISABLE_LEDGER="${DISABLE_LEDGER:-0}"      # 1 = run without the State ledger
DISABLE_A11Y="${DISABLE_A11Y:-0}"                 # 1 = vision-only (no a11y tree)

VERSION="${VERSION:-structagent}"

# ── Optional CLI flags ───────────────────────────────────────────────────────
FLAGS=()
[[ "${DISABLE_A11Y}" == "1" ]] && FLAGS+=(--disable_a11y)
if [[ "${ENABLE_DONE_AUDITOR}" == "1" ]]; then
    FLAGS+=(--enable_done_auditor --done_auditor_model "${VERIFIER_MODEL}" \
            --done_auditor_max_per_task "${DONE_AUDITOR_MAX_PER_TASK}")
fi
[[ -n "${DECOMPOSER_MODEL:-}" ]]   && FLAGS+=(--decomposer_model "${DECOMPOSER_MODEL}")
[[ -n "${DECOMPOSER_API_URL:-}" ]] && FLAGS+=(--decomposer_api_url "${DECOMPOSER_API_URL}")

python scripts/run_structagent.py \
    --provider_name "${PROVIDER}" --headless \
    --model "${MODEL}" --verifier_model "${VERIFIER_MODEL}" \
    --observation_type screenshot --sleep_after_execution 3 \
    --max_steps "${MAX_STEPS}" --num_envs "${NUM_ENVS}" \
    --client_password password --screen_width 1920 --screen_height 1080 \
    --version_num "${VERSION}" --domain "${DOMAIN}" \
    --test_config_base_dir evaluation_examples \
    --test_all_meta_path "${TEST_ALL_META_PATH}" \
    "${FLAGS[@]}" "$@"
