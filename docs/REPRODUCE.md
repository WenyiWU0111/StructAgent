# Reproducing the results

End-to-end recipe. See [`ENVIRONMENT.md`](ENVIRONMENT.md) and [`MODELS.md`](MODELS.md)
for the one-time setup.

## 1. Serve the models
```bash
bash scripts/serve_qwen35_9b.sh      # planner / actor / grounding (:8010)
bash scripts/serve_qwen35_27b.sh     # verifier / Mind2Web judge   (:8012)
```

## 2. OSWorld (default split, 100-step budget)
```bash
MODEL=vllm_qwen35-vl VERIFIER_MODEL=vllm_qwen35-27b NUM_ENVS=8 \
  VERSION=repro bash scripts/run.sh
```
Shard across machines with `TEST_ALL_META_PATH=evaluation_examples/test_nogdrive_sub{1,2,3}.json`.
To reproduce the **open-model SOTA**, use `MODEL=minimax-m3` (needs `OPENROUTER_API_KEY`).
All paper capabilities are on by default; see [`CONFIG.md`](CONFIG.md) to ablate.

## 3. Mind2Web (answer-blind Online-Mind2Web judge)
```bash
MODEL=vllm_qwen35-vl VERIFIER_MODEL=vllm_qwen35-27b MAX_STEPS=50 \
  TEST_ALL_META_PATH=evaluation_examples/test_mind2web.json bash scripts/run.sh
```
Per-category splits: `test_mind2web_test_domain_{Info,Service}.json`,
`test_mind2web_test_website.json`. Re-score finished runs offline with
`python scripts/rejudge_mind2web_online.py`.

## 4. Aggregate
```bash
python show_result.py
```
Each task directory holds `result.txt` (reward), `traj.jsonl`, and a
`trajectory.html` with per-step screenshots, verifier events, and the State ledger.
