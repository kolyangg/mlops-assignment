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
