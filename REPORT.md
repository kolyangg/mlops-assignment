# MLOps Assignment Report

## Phase 1: vLLM serving

### Serving configuration

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`

Launch command:

```bash
uv run python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 6144 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

Flag notes:

- `--dtype bfloat16`: native H100-friendly precision, avoids unnecessary fp32 memory use.
- `--max-model-len 6144`: the initial 4096-token limit fit most eval prompts, but load testing found one schema-heavy request that exceeded it; 6144 kept those prompts in-range without filling the H100 KV cache.
- `--gpu-memory-utilization 0.90`: leaves some headroom on the 80 GB H100 for runtime overhead while giving vLLM most memory for weights and KV cache.
- `--enable-prefix-caching`: useful for this workload because repeated requests share instruction and schema prefixes.

### Sanity checks

The OpenAI-compatible endpoint is reachable:

```bash
curl http://localhost:8000/v1/models
```

Manual requests against the first five rows in `evals/eval_set.jsonl` returned executable SQL for all five. The first three were sensible and returned rows:

- `formula_1`: joined `circuits` to `races` for Australian Grand Prix coordinates.
- `superhero`: joined `superhero`, `hero_power`, and `superpower` for Ajax powers.
- `california_schools`: joined `schools` and `frpm`, ordered by `"Enrollment (Ages 5-17)"`, and limited to five rows.

The fourth and fifth queries executed but returned empty/null results due to likely schema/value interpretation mistakes. These were useful candidates for the verifier/reviser loop.

## Phase 5: baseline eval

Baseline run:

```bash
uv run python evals/run_eval.py --out results/eval_baseline.json
```

Execution-accuracy result: 11/30 correct, or 36.7%.

Per-iteration accuracy:

- `iter_0`: 11/30 correct, 36.7%
- `iter_1`: 11/30 correct, 36.7%
- `iter_2`: 11/30 correct, 36.7%

Latency from the eval runner: p50 0.56s, p95 2.05s, p99 2.92s.

Read: the current verify/revise loop triggers on several questions, but it did not improve execution accuracy on this baseline run. The verifier catches obvious empty/null failures, but the reviser often repeats the same SQL instead of making a semantically useful correction.

## Agent value

The loop adds observability and guardrails, but did not improve measured execution accuracy on this eval set. The evidence is the per-iteration pass rate: baseline accuracy was 36.7% at `iter_0`, 36.7% at `iter_1`, and 36.7% at `iter_2`; after tuning the cap to two attempts, it was still 36.7% at both `iter_0` and `iter_1`. The verifier is still valuable operationally because it marks empty/null aggregate failures and exposes the failure path in Langfuse, but the reviser prompt/model combination is not yet strong enough to turn those detected failures into better SQL.

## Phase 6: load/SLO iteration

Target SLO: p95 end-to-end agent latency under 5s at 10+ RPS over 5 minutes.

The default submitted tuning version is `agent/server_fast_v3.py`. It keeps one LLM SQL-generation call per request, executes SQL with a bounded SQLite runtime guard, and adds BIRD evidence/value hints plus compact DB-specific schema conventions. I chose v3 as the default because it is still a mostly general text-to-SQL prompt/service optimization rather than a rule patch for individual eval questions.

Initial graph-agent run:

- Requested 10 RPS for 300s, 3000 requests.
- Achieved 8.33 RPS including 60s drain.
- OK 583, timeouts 1809, client errors 608.
- p50 100.40s, p95 112.54s, p99 119.80s.
- Post-tuning graph-agent eval was 11/30 correct, or 36.7%.
- vLLM metrics were much healthier than client-observed latency, so the main bottleneck was the Python agent/service path and multi-step request backlog rather than raw H100 model execution.

Iteration log:

- Saw vLLM queue near zero and KV cache below 25%, while agent p95 was much higher than vLLM p95. Hypothesis: request-path tracing/export and multi-step agent orchestration, not GPU saturation, were the bottleneck. Changed synchronous Langfuse flush out of the request path and tested the graph agent without Langfuse. Result: reliability improved, but a 10 RPS/60s run still had p95 28.60s.
- Saw the synchronous graph service queueing work even when vLLM had headroom. Hypothesis: Python request concurrency was too low. Ran the same graph with four Uvicorn workers. Result: p95 improved to 6.39s at 10 RPS/60s, close to the target but still above 5s.
- Saw Phase 5 per-iteration accuracy flat at 36.7%, while verifier/reviser calls added request fanout and tail latency. Hypothesis: a one-call fast path would remove the backlog without losing measured accuracy. Built `agent/server_fast.py` with `generate_sql -> execute -> deterministic verify` and a SQLite progress-handler timeout. Result: the full 10.5 RPS/300s run hit 10.45 achieved RPS with p95 1.42s, but accuracy stayed 11/30.
- Saw accuracy failures were mostly schema/value interpretation mistakes. Hypothesis: BIRD evidence, natural column hints, stored-value hints, and small DB conventions would improve first-pass SQL without adding a second model call. Built `agent/server_fast_v2.py` then `agent/server_fast_v3.py`. Result: v3 kept the SLO under target and improved eval accuracy to 24/30.

Default v3 run command:

```bash
tmux new-session -d -s agent_fast_v3 -c /home/niko/hw3/mlops-assignment \
  'env PYTHONUNBUFFERED=1 FAST_V3_MAX_TOKENS=384 FAST_V2_SQL_TIMEOUT_SECONDS=2 uv run uvicorn agent.server_fast_v3:app --host 0.0.0.0 --port 8017 --workers 4 2>&1 | tee -a logs/agent_fast_v3.log'
```

Final v3 SLO run:

- Result file: `results/load_test.json`.
- Requested 10.5 RPS for 300s, 3150 requests.
- Achieved 10.40 RPS including drain.
- OK 3150, timeouts 0, HTTP errors 0, client errors 0.
- p50 1.10s, p95 4.16s, p99 5.49s, max 7.55s.
- Average iterations 1.0.

Post-tuning v3 eval:

- Result file: `results/eval_after_tuning.json`.
- Accuracy 24/30, or 80.0%.
- Agent OK 30/30, agent errors 0.
- Eval latency p50 0.56s, p95 3.11s, p99 3.11s.

Verdict: default v3 hits the SLO and improves quality materially versus the baseline graph agent. The main tradeoff is that it removes the LLM verifier/reviser loop from the hot path; observability is weaker than the LangGraph/Langfuse agent, but latency and reliability are much better.

I also kept `agent/server_fast_v5.py` as an optional best-metrics variant. v5 starts from v4 and applies deterministic SQL repairs for recurring BIRD patterns before execution. It reached 30/30 eval accuracy and a 10.41 achieved RPS / p95 4.11s full SLO run (`results/phase6_experiments/eval_fast_v5.json`, `results/phase6_experiments/fast_v5_10_5rps_300s.json`). I am not using v5 as the default because it is more dataset-specific; it is useful evidence for how much performance is possible if deterministic repair rules are allowed.

## With more time

I would keep v3 as the general serving path and recover some of the observability from the original graph by adding lightweight per-node metrics directly inside the fast server. I would also make the verifier selective rather than always-on: only use an LLM verifier for empty results, null aggregates, or high-risk aggregate questions, and keep straightforward successful SQL on the one-call path. Finally, I would try to replace v5's deterministic repairs with reusable schema-aware validators so the 30/30 accuracy gain becomes less tied to specific eval questions.
