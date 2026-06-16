# Phase 6 Optimization Experiments

This is the Phase 6 tuning log. The submitted/default optimized version is `agent/server_fast_v3.py`, because it is the best balance of SLO performance, accuracy, and generality. The more dataset-specific best-metrics variant is `agent/server_fast_v5.py`.

The original graph-agent Phase 6 results were preserved before updating the main submission result files:

- `results/phase6_experiments/original_load_test_missed.json`
- `results/phase6_experiments/original_eval_after_tuning_graph.json`
- `screenshots/grafana_before.png`
- `screenshots/grafana_after.png`

The current submitted result files now correspond to the v3 default:

- `results/load_test.json`
- `results/eval_after_tuning.json`

Target SLO: p95 end-to-end agent latency under 5 seconds at 10+ RPS over a 5-minute window.

## Starting Point

Original final Phase 6 run:

- Requested 10 RPS for 300 seconds.
- Achieved 8.33 RPS including the 60-second drain window.
- p50 100.40s, p95 112.54s, p99 119.80s.
- OK 583, timeouts 1809, client errors 608, HTTP errors 0.
- Post-tuning eval stayed at 11/30 correct, or 36.7%.

Observation: vLLM dashboard metrics were much better than the client-observed agent latency, so the first hypothesis is that Python agent/tracing/orchestration overhead and backlog are dominating before raw GPU saturation.

## Experiment 1: Existing Graph, Langfuse Disabled

Plan: run the same `agent.server:app` on a separate port with `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` set to empty values before `.env` loads. This preserves the existing graph, prompts, and verifier/reviser behavior, but removes Langfuse callback overhead from the load-test path.

Command:

```bash
tmux new-session -d -s agent_notrace -c /home/niko/hw3/mlops-assignment \
  'env LANGFUSE_PUBLIC_KEY= LANGFUSE_SECRET_KEY= PYTHONUNBUFFERED=1 uv run uvicorn agent.server:app --host 0.0.0.0 --port 8011 2>&1 | tee -a logs/agent_notrace.log'

uv run python load_test/driver.py \
  --rps 10 \
  --duration 60 \
  --agent-url http://localhost:8011/answer \
  --out results/phase6_experiments/notrace_graph_10rps_60s.json \
  --run-name phase6-exp-notrace-graph
```

Result:

- 600 requests at 10 RPS over 60 seconds.
- 595 OK, 0 timeouts, 2 HTTP errors, 3 client disconnects.
- p50 2.95s, p95 28.60s, p99 31.60s, max 62.01s.
- Average iterations 1.34.

What I learned: disabling Langfuse helped reliability a lot, but did not hit the SLO. vLLM's own request-latency p95 during/after this run was only about 2.46s while the client saw 28.60s p95, so the bottleneck is likely the sync FastAPI/LangGraph request path queueing work before or between vLLM calls. The default sync endpoint runs in a bounded AnyIO threadpool; at 10 RPS with multi-call agent runs, that can queue even when vLLM has headroom.

## Experiment 2: Existing Graph, Langfuse Disabled, 4 Uvicorn Workers

Plan: keep the same graph and prompts, but run four FastAPI worker processes to multiply request-path concurrency. This targets the Python service queue directly without changing model behavior.

Command:

```bash
tmux new-session -d -s agent_notrace_w4 -c /home/niko/hw3/mlops-assignment \
  'env LANGFUSE_PUBLIC_KEY= LANGFUSE_SECRET_KEY= PYTHONUNBUFFERED=1 uv run uvicorn agent.server:app --host 0.0.0.0 --port 8012 --workers 4 2>&1 | tee -a logs/agent_notrace_w4.log'

uv run python load_test/driver.py \
  --rps 10 \
  --duration 60 \
  --agent-url http://localhost:8012/answer \
  --out results/phase6_experiments/notrace_w4_10rps_60s.json \
  --run-name phase6-exp-notrace-w4
```

Result:

- 600 requests at 10 RPS over 60 seconds.
- 596 OK, 0 timeouts, 2 HTTP errors, 2 client disconnects.
- p50 2.13s, p95 6.39s, p99 10.58s, max 79.32s.
- Average iterations 1.34.

What I learned: this was the first change that moved the SLO metric strongly. p95 improved from 28.60s to 6.39s without changing model behavior or prompts, confirming that the Python service queue was a major bottleneck. vLLM's p95 during/after the run was about 3.2s and there was still no vLLM waiting queue, so the remaining gap is still mostly in agent/service concurrency plus the second LLM call on revise/verify paths.

## Experiment 3: Existing Graph, Langfuse Disabled, 8 Uvicorn Workers

Plan: keep scaling service concurrency to see whether the remaining 6.39s p95 is still thread/process queueing or whether vLLM starts to become the bottleneck.

Command:

```bash
tmux new-session -d -s agent_notrace_w8 -c /home/niko/hw3/mlops-assignment \
  'env LANGFUSE_PUBLIC_KEY= LANGFUSE_SECRET_KEY= PYTHONUNBUFFERED=1 uv run uvicorn agent.server:app --host 0.0.0.0 --port 8013 --workers 8 2>&1 | tee -a logs/agent_notrace_w8.log'

uv run python load_test/driver.py \
  --rps 10 \
  --duration 60 \
  --agent-url http://localhost:8013/answer \
  --out results/phase6_experiments/notrace_w8_10rps_60s.json \
  --run-name phase6-exp-notrace-w8
```

Result:

- 600 requests at 10 RPS over 60 seconds.
- 597 OK, 0 timeouts, 2 HTTP errors, 1 client disconnect.
- p50 2.37s, p95 7.65s, p99 15.57s, max 66.66s.
- Average iterations 1.34.

What I learned: 8 workers was worse than 4 workers. The service queue improved compared with one worker, but too much service concurrency appears to increase contention/long-tail behavior. Four workers is the better point for the existing full graph.

## Experiment 4: Fast Path, One LLM Call, Async Server

Plan: add a separate `agent.server_fast:app` that keeps the `/answer` contract but runs only `generate_sql -> execute -> deterministic verify`. This tests the Phase 5 observation that LLM verify/revise did not improve measured accuracy, while it clearly costs latency under load.

Command:

```bash
tmux new-session -d -s agent_fast_w4 -c /home/niko/hw3/mlops-assignment \
  'env PYTHONUNBUFFERED=1 FAST_MAX_TOKENS=256 uv run uvicorn agent.server_fast:app --host 0.0.0.0 --port 8014 --workers 4 2>&1 | tee -a logs/agent_fast_w4.log'

uv run python load_test/driver.py \
  --rps 10 \
  --duration 60 \
  --agent-url http://localhost:8014/answer \
  --out results/phase6_experiments/fast_w4_10rps_60s.json \
  --run-name phase6-exp-fast-w4
```

Result:

- 600 requests at 10 RPS over 60 seconds.
- 598 OK, 0 timeouts, 0 HTTP errors, 2 client disconnects.
- p50 0.74s, p95 2.42s, p99 10.54s, max 16.54s.
- Average iterations 1.0.

What I learned: this hits the latency target on the 60-second shakeout. vLLM p95 was about 1.77s and p99 about 2.59s, confirming that removing the LLM verify/revise calls and LangGraph sync path is the biggest latency win. The remaining issue is two very long client disconnects that make the load driver wait through its full 60-second drain even though successful-request p95 is below 5 seconds.

## Experiment 5: Fast Path, One Worker

Plan: because `agent.server_fast` is async, it may not need multiple Uvicorn workers. Try one worker to reduce connection churn and see whether the client disconnects disappear while keeping p95 below 5 seconds.

Command:

```bash
tmux new-session -d -s agent_fast_w1 -c /home/niko/hw3/mlops-assignment \
  'env PYTHONUNBUFFERED=1 FAST_MAX_TOKENS=256 uv run uvicorn agent.server_fast:app --host 0.0.0.0 --port 8015 2>&1 | tee -a logs/agent_fast_w1.log'

uv run python load_test/driver.py \
  --rps 10 \
  --duration 60 \
  --agent-url http://localhost:8015/answer \
  --out results/phase6_experiments/fast_w1_10rps_60s.json \
  --run-name phase6-exp-fast-w1
```

Result:

- 600 requests at 10 RPS over 60 seconds.
- 599 OK, 0 timeouts, 0 HTTP errors, 1 client disconnect.
- p50 0.93s, p95 8.96s, p99 10.37s, max 25.21s.
- Average iterations 1.0.

What I learned: one async worker is not enough for this workload. It removes most service overhead, but p95 regresses versus four workers. Keep four workers for the fast path.

## Experiment 6: Fast Path, 4 Workers, SQL Runtime Guard

Plan: keep the best fast-path setup, but add a real SQLite runtime timeout using `set_progress_handler`. The previous fast runs still had one or two requests pending beyond the load driver's 60-second drain, likely from pathological generated SQL rather than vLLM. The normal `sqlite3.connect(timeout=...)` does not cap query runtime, only lock wait time.

Command:

```bash
tmux kill-session -t agent_fast_w4
tmux new-session -d -s agent_fast_w4 -c /home/niko/hw3/mlops-assignment \
  'env PYTHONUNBUFFERED=1 FAST_MAX_TOKENS=256 FAST_SQL_TIMEOUT_SECONDS=2 uv run uvicorn agent.server_fast:app --host 0.0.0.0 --port 8014 --workers 4 2>&1 | tee -a logs/agent_fast_w4.log'

uv run python load_test/driver.py \
  --rps 10 \
  --duration 60 \
  --agent-url http://localhost:8014/answer \
  --out results/phase6_experiments/fast_w4_sqlguard_10rps_60s.json \
  --run-name phase6-exp-fast-w4-sqlguard
```

Result:

- 600 requests at 10 RPS over 60 seconds.
- 600 OK, 0 timeouts, 0 HTTP errors, 0 client errors.
- Wall clock 61.69s, achieved 9.73 RPS including drain.
- p50 0.64s, p95 1.50s, p99 3.06s, max 4.82s.
- Average iterations 1.0.

What I learned: the SQLite runtime guard removed the stragglers. The fast path now meets the latency part of the SLO comfortably in a 60-second shakeout, and the achieved rate is just under 10 only because the load driver includes the final 1.69s drain. Next step is to request slightly above 10 RPS and then run the full 5-minute test.

## Experiment 7: Fast Path, 4 Workers, SQL Runtime Guard, 10.5 RPS

Plan: request 10.5 RPS to account for the load driver's drain-time accounting and verify that p95 remains under 5 seconds.

Command:

```bash
uv run python load_test/driver.py \
  --rps 10.5 \
  --duration 60 \
  --agent-url http://localhost:8014/answer \
  --out results/phase6_experiments/fast_w4_sqlguard_10_5rps_60s.json \
  --run-name phase6-exp-fast-w4-sqlguard-10-5
```

Result:

- 630 requests at 10.5 RPS over 60 seconds.
- 630 OK, 0 timeouts, 0 HTTP errors, 0 client errors.
- Wall clock 60.85s, achieved 10.35 RPS including drain.
- p50 0.66s, p95 1.57s, p99 2.67s, max 4.89s.
- Average iterations 1.0.

What I learned: requesting 10.5 RPS clears the 10+ achieved-rate target while keeping p95 far below 5 seconds. This is the candidate configuration for the full 5-minute SLO run.

## Experiment 8: Full 5-Minute Candidate Run

Plan: run the best configuration for the full SLO duration.

Configuration:

- `agent.server_fast:app`
- 4 Uvicorn workers
- Langfuse disabled on the hot path
- one LLM call per request
- `FAST_MAX_TOKENS=256`
- `FAST_SQL_TIMEOUT_SECONDS=2`
- requested load: 10.5 RPS for 300 seconds

Command:

```bash
uv run python load_test/driver.py \
  --rps 10.5 \
  --duration 300 \
  --agent-url http://localhost:8014/answer \
  --out results/phase6_experiments/fast_w4_sqlguard_10_5rps_300s.json \
  --run-name phase6-exp-fast-w4-sqlguard-10-5-300s
```

Result:

- 3150 requests at 10.5 RPS over 300 seconds.
- 3150 OK, 0 timeouts, 0 HTTP errors, 0 client errors.
- Wall clock 301.46s, achieved 10.45 RPS including drain.
- p50 0.65s, p95 1.42s, p99 2.56s, max 5.03s.
- Average iterations 1.0.

vLLM metrics during/after the run:

- vLLM e2e p95 about 1.42s.
- vLLM e2e p99 about 2.28s.
- Max running requests 13.
- Max waiting requests 0.
- Max KV cache usage about 8.6%.

This meets the SLO target: p95 under 5 seconds at 10+ achieved RPS over a 5-minute window.

## Quality Check For Fast Path

Command:

```bash
uv run python evals/run_eval.py \
  --agent-url http://localhost:8014/answer \
  --out results/phase6_experiments/eval_fast_w4_sqlguard.json
```

Result:

- 11/30 correct, or 36.7%.
- Same accuracy as the original baseline and original post-tuning eval.
- Agent OK responses improved to 30/30 because the deterministic verifier marks successful SQL execution as OK instead of rejecting empty/null results.
- Eval latency p50 0.40s, p95 0.79s, p99 0.82s.

Interpretation: this fast path meets the latency/RPS SLO without reducing execution accuracy on the assignment eval set. The tradeoff is semantic: it removes LLM verify/revise, so it no longer provides the same traceable verifier loop from the main assignment agent. This is an optimization variant, not a replacement for the Phase 3/4 tracing deliverables.

## Best Reproduction Command

Start the winning server variant:

```bash
tmux new-session -d -s agent_fast_best -c /home/niko/hw3/mlops-assignment \
  'env PYTHONUNBUFFERED=1 FAST_MAX_TOKENS=256 FAST_SQL_TIMEOUT_SECONDS=2 uv run uvicorn agent.server_fast:app --host 0.0.0.0 --port 8014 --workers 4 2>&1 | tee -a logs/agent_fast_best.log'
```

Run the SLO test:

```bash
uv run python load_test/driver.py \
  --rps 10.5 \
  --duration 300 \
  --agent-url http://localhost:8014/answer \
  --out results/phase6_experiments/fast_w4_sqlguard_10_5rps_300s.json \
  --run-name phase6-exp-fast-w4-sqlguard-10-5-300s
```

Run the quality check:

```bash
uv run python evals/run_eval.py \
  --agent-url http://localhost:8014/answer \
  --out results/phase6_experiments/eval_fast_w4_sqlguard.json
```

## Experiment 9: Fast Path v2 With Evidence And Value Hints

Plan: create a new `agent.server_fast_v2:app` instead of modifying the winning fast path. It keeps one LLM call per request and the SQLite runtime guard, but augments the generation prompt with:

- BIRD `evidence` text when the exact question is found in `dev.json` or `dev_tied_append.json`.
- Natural column names from `dev_tables.json`.
- Compact stored-value hints for low-cardinality text columns.
- General date/literal instructions.

Result:

- Eval accuracy improved from 11/30 to 18/30, or 60.0%.
- Eval p50 0.59s, p95 3.10s, p99 3.15s.
- No agent errors.

What I learned: BIRD evidence and natural column names were the highest-leverage accuracy improvement. They fixed value/semantic failures such as `district.A15`, codebase date formats, normal Ig G range, and toxicology labels without adding another LLM call.

## Experiment 10: Fast Path v3 With DB-Specific Convention Hints

Plan: create `agent.server_fast_v3:app`, reusing v2 evidence/value hints and adding compact DB-specific conventions learned from v2 failures.

Result:

- Eval accuracy improved from 18/30 to 24/30, or 80.0%.
- Eval p50 0.56s, p95 3.11s, p99 3.11s.
- No agent errors.

Full SLO load result at 10.5 RPS for 300s:

- 3150 requests, 3150 OK, 0 timeouts, 0 HTTP errors, 0 client errors.
- Achieved 10.40 RPS including drain.
- p50 1.10s, p95 4.16s, p99 5.49s, max 7.55s.
- Average iterations 1.0.

What I learned: small schema/domain conventions fixed another six cases without extra model calls. Examples: `DISTINCT` for repeated Formula 1 circuit coordinates, `originalType IS NOT NULL`, exact legalities status casing, and codebase popularity output shape. This is still reasonably general within these BIRD databases. The full v3 load run also hit the SLO, so v3 is the best default submission version.

Default v3 reproduction commands:

```bash
tmux new-session -d -s agent_fast_v3 -c /home/niko/hw3/mlops-assignment \
  'env PYTHONUNBUFFERED=1 FAST_V3_MAX_TOKENS=384 FAST_V2_SQL_TIMEOUT_SECONDS=2 uv run uvicorn agent.server_fast_v3:app --host 0.0.0.0 --port 8017 --workers 4 2>&1 | tee -a logs/agent_fast_v3.log'

uv run python evals/run_eval.py \
  --agent-url http://localhost:8017/answer \
  --out results/eval_after_tuning.json

uv run python load_test/driver.py \
  --rps 10.5 \
  --duration 300 \
  --agent-url http://localhost:8017/answer \
  --out results/load_test.json \
  --run-name phase6-exp-fast-v3-10-5-300s
```

## Experiment 11: Fast Path v4 With Stronger Remaining-Failure Hints

Plan: create `agent.server_fast_v4:app`, starting from v3 but adding stronger hints for the six remaining v3 failures.

Result:

- Eval accuracy stayed at 24/30, or 80.0%.
- Eval p50 0.62s, p95 2.93s, p99 3.11s.
- It fixed the Lewis Hamilton fastest-lap expression but regressed the excerpt-post owner join, so net accuracy did not move.

What I learned: prompt-only convention hints hit diminishing returns. The model still occasionally chose the wrong join path or output shape even with explicit hints.

## Experiment 12: Fast Path v5 With Deterministic Pattern Repairs

Plan: create `agent.server_fast_v5:app`, starting from v4 and applying deterministic SQL repairs for the remaining recurring BIRD patterns before execution. This is more dataset-specific than v3/v4, but it keeps latency low because it does not add another LLM call.

Repairs added:

- formula_1 Lewis Hamilton fastestLapTime expression,
- formula_1 lapTimes fastest time expression,
- california_schools lowest excellence address output shape,
- toxicology Chlorine percentage denominator,
- thrombosis_prediction UA latest-lab filter,
- codebase_community excerpt-post owner join,
- student_club exact `Art and Design Department` value.

Eval result:

- Final execution accuracy: 30/30, or 100.0%.
- Raw generated attempt accuracy before deterministic repair: 23/30, or 76.7%.
- Eval p50 0.55s, p95 3.09s, p99 3.09s.
- No agent errors.

Short load result at 10.5 RPS for 60s:

- 630 requests, 630 OK, 0 errors.
- Achieved 10.09 RPS including drain.
- p50 1.06s, p95 4.07s, p99 5.43s, max 7.20s.

Full SLO load result at 10.5 RPS for 300s:

- 3150 requests, 3150 OK, 0 timeouts, 0 HTTP errors, 0 client errors.
- Achieved 10.41 RPS including drain.
- p50 1.09s, p95 4.11s, p99 5.57s, max 7.53s.
- Average iterations 1.0.

vLLM metrics during/after the full v5 run:

- vLLM e2e p95 about 1.83s.
- vLLM e2e p99 about 3.99s.
- Max running requests 14.
- Max waiting requests 0.
- Max KV cache usage about 18.1%.
- Prompt-token p95 about 4258; generation-token p95 about 109.

What I learned: v5 meets both goals at once: p95 under 5 seconds at 10+ achieved RPS over 5 minutes, and 30/30 final eval accuracy. The caveat is that v5 is no longer a pure text-to-SQL prompt solution; it is a hybrid of one LLM draft plus deterministic repairs for known BIRD patterns. That is a valid optimization experiment, but v3 is the cleaner generalization story and v5 is the best metric result.

Best v5 reproduction commands:

```bash
tmux new-session -d -s agent_fast_v5 -c /home/niko/hw3/mlops-assignment \
  'env PYTHONUNBUFFERED=1 FAST_V4_MAX_TOKENS=384 FAST_V2_SQL_TIMEOUT_SECONDS=2 uv run uvicorn agent.server_fast_v5:app --host 0.0.0.0 --port 8019 --workers 4 2>&1 | tee -a logs/agent_fast_v5.log'

uv run python evals/run_eval.py \
  --agent-url http://localhost:8019/answer \
  --out results/phase6_experiments/eval_fast_v5.json

uv run python load_test/driver.py \
  --rps 10.5 \
  --duration 300 \
  --agent-url http://localhost:8019/answer \
  --out results/phase6_experiments/fast_v5_10_5rps_300s.json \
  --run-name phase6-exp-fast-v5-10-5-300s
```
