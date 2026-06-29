# Agents

This package contains **StructAgent** — the agent introduced in the paper.

- [`structagent/`](structagent/) — the agent: a verifier-centered planner / actor /
  verifier loop over a unified, evidence-backed task State. See
  [`structagent/__init__.py`](structagent/__init__.py) for the module map.
- [`model_endpoints.py`](model_endpoints.py) — registry mapping model aliases to
  OpenAI-compatible endpoints (override with `VLLM_<KEY>_URL`).

```python
from mm_agents.structagent import StructAgent

agent = StructAgent(model="vllm_qwen35-vl")
actions = agent.predict(instruction, obs)
```
