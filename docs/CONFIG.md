# Configuration & switches

StructAgent is configured through **environment variables** read by
[`scripts/run.sh`](../scripts/run.sh) (and a few CLI flags it forwards). The
defaults in `run.sh` are **the full agent used in the paper** — you only set a
variable to *change* something or to **ablate** a capability.

```bash
# full agent, OSWorld default split:
MODEL=vllm_qwen35-vl NUM_ENVS=8 bash scripts/run.sh
# turn a capability OFF for an ablation:
USE_FAILURE_ATTRIBUTION=0 bash scripts/run.sh
```

> ⚠️ Do **not** call `python scripts/run_structagent.py` directly unless you set
> these variables yourself — the agent's bare defaults (in
> `mm_agents/structagent/config.py`) have the advanced capabilities **off**.
> Always go through `scripts/run.sh`, which switches them on.

## Models

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `vllm_qwen35-vl` | planner / actor / grounding model (any alias in `model_endpoints.py`) |
| `VERIFIER_MODEL` | = `MODEL` | verifier + DONE-auditor + Mind2Web judge; set to split verification off |
| `DECOMPOSER_MODEL` / `DECOMPOSER_API_URL` | = planner | override the grounding model only |

## Environment & run

| Variable | Default | Meaning |
|---|---|---|
| `PROVIDER` | `docker` | OSWorld provider: `docker` / `vmware` / `aws` / `API` |
| `NUM_ENVS` | `5` | parallel environments |
| `MAX_STEPS` | `100` | per-task step budget |
| `DOMAIN` | `all` | restrict to one domain (e.g. `chrome`) |
| `TEST_ALL_META_PATH` | `evaluation_examples/test_nogdrive.json` | task manifest (swap to `test_mind2web.json` for Mind2Web) |

## Capabilities — **on by default** (the paper's StructAgent)

| Variable | Default | What it does |
|---|---|---|
| `PERCEIVER_ENABLED` | `1` | structured screen perception (sheets / slides / docs) instead of raw a11y |
| `USE_FAILURE_ATTRIBUTION` | `1` | role-attributed failure diagnosis → routed recovery |
| `ENABLE_STUCK_DIAGNOSIS_INJECTION` | `1` | inject the diagnosis into the replan prompt |
| `ENABLE_DONE_AUDITOR` | `1` | evidence-gated DONE auditor (adversarial re-check at completion) |
| `ENABLE_FEASIBILITY_VERDICT` | `1` | after K replans, judge whether the task is infeasible and emit FAIL |
| `WEB_DOM_INJECT` | `1` | inject Chrome's checker-source DOM on web steps |

Set any of these to `0` to ablate that component.

## Experience memory — **off by default**

| Variable | Default | What it does |
|---|---|---|
| `MEMORY` | `off` | shorthand: `on` enables both memory layers below |
| `ENABLE_PLANNER_EXPERIENCE_MEMORY` | `0` | retrieve L1/L2 plan + action recipes into planner/actor prompts |
| `ENABLE_VERIFIER_EXPERIENCE_MEMORY` | `0` | retrieve verification recipes into the ledger initializer |
| `MEMORY_TOP_K` | `10` | retrieval depth |

> **Memory requires prebuilt FAISS banks** (mined offline from a trajectory pool)
> that are **not shipped** in this repo. With `MEMORY=on` but no banks, retrieval
> degrades silently to no-op (the agent still runs, just without retrieved advice).
> See *Building memory banks* below.

## Ablations (advanced)

| Variable | Default | What it does |
|---|---|---|
| `DISABLE_LEDGER` | `0` | `1` = run without the unified State ledger |
| `DISABLE_A11Y` | `0` | `1` = vision-only (no accessibility tree) |
| `ENABLE_DOMAIN_CODING_GATE` | `0` | bar GUI-only domains from shell/file actions |
| `QWEN35_THINKING` | `0` | enable Qwen3.5 reasoning mode (slower, more tokens) |

## Building memory banks

The memory banks are built offline from successful trajectories with the tools
under [`mm_agents/structagent/memory/offline/`](../mm_agents/structagent/memory/offline/)
(load pool → cluster → polish L1/L2/L3 → build FAISS indexes). They are written
under `results/` and read at runtime via `results/planner_experience/` and
`results/unified_memory/`. If you have your own pool of solved tasks, run that
pipeline; otherwise leave memory off — it does not affect correctness, only the
quality of retrieved hints.
