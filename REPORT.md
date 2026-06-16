# MLOps Assignment Report

## Phase 1: vLLM serving

### Initial serving configuration

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507`

Launch command:

```bash
uv run python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

Flag notes:

- `--dtype bfloat16`: native H100-friendly precision, avoids unnecessary fp32 memory use.
- `--max-model-len 4096`: fits the expected 1.5-3K token text-to-SQL prompts while reserving room for short SQL outputs.
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

The fourth and fifth queries executed but returned empty/null results due to likely schema/value interpretation mistakes. These are good candidates for Phase 3 verifier/reviser testing.

### Screenshot TODO

Capture `screenshots/vllm_manual_query.png` showing:

- vLLM running on port `8000`.
- One manual `/v1/chat/completions` request returning SQL.

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

## Phase 6: load/SLO iteration

Target SLO: p95 end-to-end agent latency under 5s at 10+ RPS over 5 minutes.

Initial 10 RPS shakeout, 30s:

- Requested 10 RPS, 300 requests.
- Achieved 3.33 RPS including drain.
- p95 latency 12.81s.
- 257 OK, 40 HTTP 500, 3 client errors.

Iteration log:

- Saw vLLM queue near zero and KV cache below 25%, while agent p95 was much higher than vLLM p95. Hypothesis: request-path tracing/export and multi-step agent orchestration, not GPU saturation, were the bottleneck. Changed synchronous Langfuse flush out of the request path. Result: this alone did not help; p95 worsened in the next shakeout, so flush was not the main cause.
- Saw Phase 5 per-iteration accuracy flat at 36.7%, while long-tail requests often used multiple generate/revise attempts. Hypothesis: extra revise attempts were adding latency without quality gain. Changed `MAX_ITERATIONS` from 3 to 2. Result: quality stayed 11/30 after tuning, and shakeout p95 improved versus the worst run, but still missed the SLO.
- Saw repeated HTTP 500s with `NoneType.replace` and one 4096-token context error. Hypothesis: schema rendering and context limit were causing avoidable failures under random load-test DBs. Changed schema FK rendering to skip incomplete SQLite FK rows and increased vLLM `--max-model-len` to 6144. Result: HTTP failures dropped sharply in shakeout, but p95 was still above target.

Final 10 RPS, 5-minute run:

- Requested 10 RPS for 300s, 3000 requests.
- Achieved 8.33 RPS including 60s drain.
- OK 583, timeouts 1809, client errors 608.
- p50 100.40s, p95 112.54s, p99 119.80s.
- vLLM metrics during/after the run showed p95 request latency around 2.5s, max waiting around 12, and KV cache below ~31%, so the final bottleneck is the agent/service layer under backlog rather than raw model execution.

Post-tuning eval:

```bash
uv run python evals/run_eval.py --out results/eval_after_tuning.json
```

Quality after tuning stayed at 11/30 correct, or 36.7%. Per-iteration accuracy with the lower cap was 36.7% at both iter 0 and iter 1.

Verdict: SLO missed. The gap is large: achieved 8.33 RPS versus 10+ target, and p95 112.54s versus 5s target. The next fix would be agent-level backpressure/batching or simplifying the verifier path, because the GPU serving layer still had measurable headroom while the FastAPI/LangGraph request layer accumulated long-tail work.
