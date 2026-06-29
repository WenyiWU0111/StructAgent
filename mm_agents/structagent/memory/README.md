# `mm_agents/structagent/memory/` — Causal-Agent Multi-Layer Memory Infrastructure

This package owns the agent's **experience memory bank** — how successful
and failed trajectories get distilled into reusable knowledge, and how
that knowledge gets retrieved + injected into planner / actor / verifier
prompts at task time.

Originally named `verifier_audit/` (when its only job was auditing
verify specs). After expanding through several waves of mining +
runtime retrieval + multi-layer memory architecture, the name was
misleading. Renamed to `memory/` on 2026-06-05. See
[`docs/MIGRATION_2026_06_05.md`](docs/MIGRATION_2026_06_05.md).

## Top-level structure

```
memory/
  runtime/          # imported by the agent at task time — production critical
    retrieval/      # query the FAISS indexes (multi-layer planner/actor memory
                    #   + verifier check recipes)
    injection/      # render retrieved memory into planner / actor / replan prompts
    digester/       # state-aware relevance gate over retrieved memory
    recovery/       # failure-attribution recovery-block render

  offline/          # NOT imported at task time — mines trajectories + builds banks
    normalize_external/   # adapters for non-internal sources (AgentNet / Mind2Web)
    load_pool/            # UnifiedRecord pool loader
    cluster/              # 2-tier (tight / loose) instruction clustering
    polish/               # per-layer LLM polish (L1 / L2 / L3a / L3c)
    build_indexes/        # multi-layer FAISS index builder
    clean_leakage.py      # abstract task-instance values out of mined banks
    build_check_recipes.py  # verifier intent recipes → boundary check recipes

  common/           # shared helpers
    walker.py             # trajectory directory walker

  deferred/         # placeholder for piloted-but-not-shipped layers

  docs/
    schema.md         # per-layer JSON schema reference
    MIGRATION_*.md    # historical reorg records
```

## Runtime entry points (used by the agent loop)

| from | imports | where called |
|---|---|---|
| `structagent.core.plans.planner` | `runtime.injection.planner_prompt_injection.render_for_planner_initial` | initial-plan prompt build |
| `structagent.core.actor` | `runtime.injection.planner_prompt_injection.render_for_actor_subgoal` | actor decomposer prompt build |
| `causal_agent.ledger.core.initializer` | `runtime.injection.prompt_injection` | verifier intent recipe injection |
| `structagent.core.plans.planner` | `runtime.injection.replan_memory_injection` | force_replan prompt build |

## Memory-version env switch

The runtime `render_for_planner_initial` and `render_for_actor_subgoal`
route to either v2 or v3 retrievers based on `MEMORY_BANK_VERSION`:

```bash
# default — v2 (single-source: internal only)
ENABLE_PLANNER_EXPERIENCE_MEMORY=1 python ...

# multi-layer v3 (internal + AgentNet + Mind2Web; L2/L3a-cluster/L3a-rule/L3c)
ENABLE_PLANNER_EXPERIENCE_MEMORY=1 MEMORY_BANK_VERSION=v3 python ...
```

## v3 multi-layer architecture summary

Five retrievable surfaces (built by `offline/build_indexes/build_multi_layer_indexes.py`):

| surface | input | embed key | use at runtime |
|---|---|---|---|
| **L1** typical_actions (per-domain) | subgoal cluster | `subgoal_pattern` | per-subgoal transition: "how have others done this subgoal" |
| **L2** plan templates | tight task cluster (n≤10) | `when_to_use` | task-start: "plans that worked on similar tasks" |
| **L3a-cluster** patterns | loose task cluster | `cluster_summary` | task-start: "patterns across this kind of task" |
| **L3a-rule** specific rules | per-rule expansion of L3a-cluster | `rule + applies_when` | per-subgoal: "specific gotchas worth knowing" |
| **L3c** domain rule sheet | meta-polish over L2 by domain | `domain` (direct lookup) | task-start: "what's true in this application" |

Total v3 entries: **5100+** retrievable items across ~4725 unique trajectories.

## Outputs (off-tree, in `results/`)

| layer | path |
|---|---|
| v2 ledgers | `results/successful_ledgers/<domain>/<task_id>.json` |
| v2 L2 skills | `results/planner_experience/_l2_skills_v2/` (157) |
| v2 L1 actions | `results/planner_experience/_l1_actions_v2/` (335) |
| v2 intent recipes | `results/successful_ledgers/_intent_recipes_v2/` (178) |
| v3 L2 skills | `results/unified_memory/_l2_skills_v3/` (311 cluster, 384 skills) |
| v3 L3a cluster | `results/unified_memory/_l3a_cluster_rules_v3/` (279 cluster, 2505 rules) |
| v3 L3c | `results/unified_memory/_l3c_domain_rules_v3/` (7 domain, ~270 facts) |
| v3 L1 | `results/unified_memory/_l1_actions_v3/<domain>/` (798 cluster, 1676 actions) |
| v3 indexes | `results/unified_memory/_indexes_v3/{l1,l2,l3a_cluster,l3a_rule,l3c}/` |
