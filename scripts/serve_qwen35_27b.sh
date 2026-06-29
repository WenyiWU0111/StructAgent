#!/bin/bash
# Serve Qwen/Qwen3.5-27B with vLLM — a LOCAL endpoint (default localhost:8002)
# for use as a stronger local planner / verifier / judge, alongside the 9B on
# :8001 (see serve_qwen35_9b.sh). Wire it to the agent via a vLLM alias whose
# server-map entry points at this port (see mm_agents/structagent/utils/
# llm_client.py).
#
# Served model name is the default ``Qwen/Qwen3.5-27B`` (no
# --served-model-name), so clients address it by that id.
#
# GPUs: 27B in bf16 is ~54 GB of weights. That fits a single 80 GB card but
# leaves little room for the KV cache (tight under long prompts / thinking-on),
# so this DEFAULTS TO 2 GPUs via tensor parallelism. ``--tensor-parallel-size``
# is derived automatically from the number of GPU ids in ``GPU`` (comma-list).
#
# Examples:
#
#   # Default: detached screen "qwen35-27b", GPUs 5,6 (tensor-parallel 2), port 8002
#   bash scripts/serve_qwen35_27b.sh
#
#   # Single card (tight — only if a whole 80 GB GPU is free)
#   GPU=3 bash scripts/serve_qwen35_27b.sh
#
#   # Other GPUs / port / session
#   GPU=0,1 PORT=8012 SCREEN_NAME=qwen35-27b-2 bash scripts/serve_qwen35_27b.sh
#
#   # Run in the CURRENT shell instead (e.g. already inside a screen)
#   FOREGROUND=1 bash scripts/serve_qwen35_27b.sh

set -e

GPU="${GPU:-5,6}"          # comma-separated CUDA ids; count = tensor-parallel size
PORT="${PORT:-8012}"
MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
SCREEN_NAME="${SCREEN_NAME:-qwen35-27b}"
ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-}}"
# tensor-parallel size = number of GPU ids supplied (1 → single-card).
TP="$(echo "${GPU}" | awk -F, '{print NF}')"

if [[ -z "${FOREGROUND:-}" ]]; then
    # Refuse to double-start: a listener already on the port, or a
    # leftover session, means someone/something is using this setup.
    if ss -tln | grep -q ":${PORT} "; then
        echo "Port ${PORT} already has a listener — nothing to do." >&2
        echo "  curl http://localhost:${PORT}/v1/models   # check if it's our vllm" >&2
        exit 1
    fi
    if screen -ls | grep -q "\.${SCREEN_NAME}\s"; then
        echo "Screen session '${SCREEN_NAME}' already exists (server not up — probably crashed)." >&2
        echo "  inspect : screen -r ${SCREEN_NAME}" >&2
        echo "  kill    : screen -S ${SCREEN_NAME} -X quit   # then re-run me" >&2
        exit 1
    fi
    # vllm grabs ~90% of EACH card — starting on a busy GPU kills both jobs.
    # Guard every id in the tensor-parallel set (FORCE=1 to skip).
    for _g in ${GPU//,/ }; do
        USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${_g}")"
        if [[ "${USED_MIB}" -gt 1024 && -z "${FORCE:-}" ]]; then
            echo "GPU ${_g} already has ${USED_MIB} MiB in use. Pick other GPU=a,b or FORCE=1." >&2
            exit 1
        fi
    done
    screen -dmS "${SCREEN_NAME}" bash -c "FOREGROUND=1 GPU='${GPU}' PORT='${PORT}' MODEL='${MODEL}' ENV_PREFIX='${ENV_PREFIX}' bash '$0' $*"
    echo "Started '${SCREEN_NAME}' (GPUs ${GPU}, tensor-parallel ${TP}, port ${PORT}, model ${MODEL})."
    echo "  logs   : screen -r ${SCREEN_NAME}"
    echo "  ready? : curl -s http://localhost:${PORT}/v1/models   # 27B takes ~5-8 min to load"
    exit 0
fi

# ── Crash fixes (same as serve_qwen35_9b.sh — Qwen3.5 family + same vLLM) ──
# 1. flash_attn_2_cuda needs CXXABI_1.3.15; system libstdc++ is too old.
#    Use the conda env's libstdc++ (same trick as the points-gui server).
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# 2. paddlex self-registers as a vllm general plugin then crashes on import
#    (matplotlib NumPy 1.x vs 2.3.5) and phones home. We serve plain HF
#    checkpoints — no plugins needed.
export VLLM_PLUGINS=''
# 3. vllm's kernel_warmup probes DeepGEMM unconditionally and raises if the
#    deep_gemm package is missing (it is, in os-world).
export VLLM_USE_DEEP_GEMM=0
# 4. Qwen3.5 uses GDN (gated-delta-net) linear attention; flashinfer
#    JIT-compiles that kernel on the FIRST inference request via
#    subprocess.run(['ninja', ...]), which searches PATH. We launch vllm by
#    abspath without activating the env, so the env's bin (where ninja lives)
#    isn't on PATH -> FileNotFoundError 'ninja' -> EngineCore crash on the
#    first request (server loads fine, then dies). Put the env bin on PATH.
export PATH="${ENV_PREFIX}/bin:${PATH}"

exec env CUDA_VISIBLE_DEVICES="${GPU}" \
    "${ENV_PREFIX}/bin/vllm" serve "${MODEL}" --port "${PORT}" \
    --tensor-parallel-size "${TP}" "$@"
