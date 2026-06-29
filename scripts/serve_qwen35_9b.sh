#!/bin/bash
# Serve Qwen/Qwen3.5-9B with vLLM — the localhost:8001 endpoint used as
# planner/verifier/judge (``vllm_qwen35-vl`` in run_text_answer_planner.sh).
#
# Served model name is the default ``Qwen/Qwen3.5-9B`` (no
# --served-model-name), so existing clients keep working.
#
# Examples:
#
#   # Default: detached screen "qwen35-9b", GPU 2, port 8001
#   bash scripts/serve_qwen35_9b.sh
#
#   # Run in the CURRENT shell instead (e.g. already inside a screen)
#   FOREGROUND=1 bash scripts/serve_qwen35_9b.sh
#
#   # Other GPU/port/session, extra vllm flags pass through
#   GPU=3 PORT=8002 SCREEN_NAME=qwen35-9b-2 bash scripts/serve_qwen35_9b.sh
#   bash scripts/serve_qwen35_9b.sh --reasoning-parser qwen3

set -e

GPU="${GPU:-2}"
PORT="${PORT:-8010}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
SCREEN_NAME="${SCREEN_NAME:-qwen35-9b}"
ENV_PREFIX="${ENV_PREFIX:-${CONDA_PREFIX:-}}"

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
    # vllm grabs ~90% of the GPU — starting on a busy card kills both jobs.
    USED_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}")"
    if [[ "${USED_MIB}" -gt 1024 && -z "${FORCE:-}" ]]; then
        echo "GPU ${GPU} already has ${USED_MIB} MiB in use. Pick another GPU=N or FORCE=1." >&2
        exit 1
    fi
    screen -dmS "${SCREEN_NAME}" bash -c "FOREGROUND=1 GPU='${GPU}' PORT='${PORT}' MODEL='${MODEL}' ENV_PREFIX='${ENV_PREFIX}' bash '$0' $*"
    echo "Started '${SCREEN_NAME}' (GPU ${GPU}, port ${PORT}, model ${MODEL})."
    echo "  logs   : screen -r ${SCREEN_NAME}"
    echo "  ready? : curl -s http://localhost:${PORT}/v1/models   # takes ~3-4 min to load"
    exit 0
fi

# ── Crash fixes (found 2026-06-10, all three are needed) ──────────────
# Exports live below the screen-spawn block on purpose: the conda libs
# must reach only vllm, not screen (conda's libtinfo makes screen warn).
#
# 1. flash_attn_2_cuda needs CXXABI_1.3.15; system libstdc++ is too old.
#    Use the conda env's libstdc++ (same trick as the points-gui server).
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# 2. paddlex self-registers as a vllm general plugin, then crashes on
#    import (matplotlib built for NumPy 1.x vs installed 2.3.5) and
#    phones home ("checking connectivity to the model hosters"). We
#    serve plain HF checkpoints — no plugins needed.
export VLLM_PLUGINS=''
# 3. vllm's kernel_warmup probes DeepGEMM unconditionally and raises if
#    the deep_gemm package is missing (it is, in os-world).
export VLLM_USE_DEEP_GEMM=0
# 4. Qwen3.5-9B uses GDN (gated-delta-net) linear attention; flashinfer
#    JIT-compiles that kernel on the FIRST inference request via
#    subprocess.run(['ninja', ...]), which searches PATH. We launch vllm by
#    abspath without activating the env, so the env's bin (where ninja lives)
#    isn't on PATH -> FileNotFoundError 'ninja' -> EngineCore crash on the
#    first request (server loads fine, then dies). Put the env bin on PATH.
export PATH="${ENV_PREFIX}/bin:${PATH}"

exec env CUDA_VISIBLE_DEVICES="${GPU}" \
    "${ENV_PREFIX}/bin/vllm" serve "${MODEL}" --port "${PORT}" "$@"
