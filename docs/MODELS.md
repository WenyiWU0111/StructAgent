# Models & serving

StructAgent uses up to three *roles*, each an OpenAI-compatible endpoint:

| Role | CLI flag | Default alias | Used for |
|---|---|---|---|
| Planner / Actor / Grounding | `--model`, `--decomposer_model` | `vllm_qwen35-vl` (Qwen3.5-9B) | subgoal decisions, action decomposition + grounding |
| Verifier / DONE-auditor | `--verifier_model`, `--done_auditor_model` | falls back to `--model` | milestone checks, completion audit, Mind2Web judge |

A single `--model` runs everything (single-model setup). Splitting roles (e.g. a
stronger `--verifier_model`) matches the paper's configuration.

## Aliases

Aliases map to `(served_name, base_url)` in
[`mm_agents/model_endpoints.py`](../mm_agents/model_endpoints.py). Pass them with a
`vllm_` prefix on the CLI (e.g. `--model vllm_qwen35-vl`). Highlights:

- `vllm_qwen35-vl`  → Qwen/Qwen3.5-9B  @ `localhost:8010`
- `vllm_qwen35-27b` → Qwen/Qwen3.5-27B @ `localhost:8012`
- `claude-opus`, `claude-sonnet`, `gemini`, `gpt-4o`, … → OpenRouter (needs `OPENROUTER_API_KEY`)

Override any endpoint without editing code:
```bash
export VLLM_QWEN35_VL_URL=http://gpu-box:8010/v1
```

## Local serving (vLLM)

```bash
bash scripts/serve_qwen35_9b.sh     # Qwen3.5-9B  on :8010
bash scripts/serve_qwen35_27b.sh    # Qwen3.5-27B on :8012 (TP=2)
# or both via Docker:
docker compose -f scripts/docker-compose.vllm.yml up -d
# check readiness:
curl -s http://localhost:8010/v1/models
```

The `serve_*.sh` scripts add GPU guards and a few vLLM env fixes (flash-attn libstdc++,
plugin/DeepGEMM/ninja). The minimal equivalent is just `vllm serve <model> --port <p>`.

## Hosted (OpenRouter)

Set `OPENROUTER_API_KEY` in `.env` to route any OpenRouter alias — handy for the
Mind2Web judge or for experimenting with a hosted verifier (e.g. `--verifier_model
claude-sonnet`). The planner/actor/grounding stack is built for the local Qwen3.5
models; support for other backbones (e.g. the paper's MiniMax-M3) is coming soon.
