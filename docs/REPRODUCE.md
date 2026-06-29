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
python scripts/run_structagent.py \
  --provider_name docker --headless \
  --model vllm_qwen35-vl --decomposer_model vllm_qwen35-vl \
  --verifier_model vllm_qwen35-27b \
  --test_all_meta_path evaluation_examples/test_nogdrive.json \
  --max_steps 100 --num_envs 4 \
  --version_num repro
```
Shard across machines with `test_nogdrive_sub{1,2,3}.json`.

To reproduce the **open-model SOTA**, swap the backbone for MiniMax-M3
(`--model minimax-m3`, requires `OPENROUTER_API_KEY`).

## 3. Mind2Web (answer-blind Online-Mind2Web judge)
```bash
python scripts/run_structagent.py \
  --model vllm_qwen35-vl --verifier_model vllm_qwen35-27b \
  --test_all_meta_path evaluation_examples/test_mind2web.json \
  --max_steps 50
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
