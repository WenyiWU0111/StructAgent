# Mind2Web tasks (executable, OSWorld-style)

These task configs adapt the **Mind2Web** benchmark (Deng et al., 2023 —
<https://osu-nlp-group.github.io/Mind2Web/>) into the OSWorld task format so the
same runner can execute and score them.

**Grading.** Tasks are scored with the answer-blind **Online-Mind2Web** protocol
(arXiv:2504.01382): a 3-stage judge over the raw intent, LLM-extracted key
points, per-screenshot relevance, and the action history — it never reads the
agent's self-reported `finished(answer=...)`. See `web_judge_online_m2w.py`.

If you use these tasks, please cite Mind2Web and Online-Mind2Web.
