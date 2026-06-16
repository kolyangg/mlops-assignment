# Submission Files

This file records the clean submission package contents for the MLOps assignment. Do not submit `.env`, `data/bird/`, `.venv/`, `.git/`, `logs/`, caches, or local zip/build artifacts.

## Phase Completion Audit

- Phase 1: complete. vLLM config and rationale are in `REPORT.md`; manual SQL screenshot is `screenshots/vllm_manual_query.png`.
- Phase 2: complete. Grafana/Prometheus config is under `infra/`; dashboard screenshot is `screenshots/grafana_serving.png`.
- Phase 3: complete. The LangGraph verify/revise agent is in `agent/server.py`, `agent/graph.py`, `agent/prompts.py`, `agent/schema.py`, and `agent/execution.py`.
- Phase 4: complete. Langfuse trace screenshots are `screenshots/langfuse_trace.png` and `screenshots/langfuse_tags.png`.
- Phase 5: complete. Baseline eval output is `results/eval_baseline.json`; baseline dashboard screenshot is `screenshots/grafana_eval_run.png`.
- Phase 6: complete. Default submitted optimized server is `agent/server_fast_v3.py`; final SLO/eval outputs are `results/load_test.json` and `results/eval_after_tuning.json`; before/after screenshots are `screenshots/grafana_before.png` and `screenshots/grafana_after.png`.
- Phase 7: complete. Final writeup is `REPORT.md`. Additional Phase 6 experiment notes are in `PHASE6_OPTIMIZATION_LOG.md`.

## Key Phase 6 Numbers

- Default v3 SLO result: 10.40 achieved RPS, p95 4.16s, 3150/3150 OK.
- Default v3 eval result: 24/30 correct, 80.0% accuracy, 0 agent errors.
- Optional v5 best-metrics result: 30/30 eval accuracy, 10.41 achieved RPS, p95 4.11s. v5 is documented as more dataset-specific and is not the default submission path.

## Files Included In The Zip

### Documentation

- `README.md`
- `REPORT.md`
- `PHASE6_OPTIMIZATION_LOG.md`
- `SUBMISSION_FILES.md`

### Environment And Project Metadata

- `.gitignore`
- `.env.example`
- `pyproject.toml`
- `uv.lock`
- `docker-compose.yml`

### Source Code

- `agent/`
- `evals/`
- `load_test/`
- `scripts/`
- `infra/`

### Result Artifacts

- `results/eval_baseline.json`
- `results/eval_after_tuning.json`
- `results/load_test.json`
- `results/phase6_experiments/`

The extra Phase 6 experiment files are included so the iteration log is reproducible and the original graph-agent miss is preserved.

### Screenshots

- `screenshots/vllm_manual_query.png`
- `screenshots/grafana_serving.png`
- `screenshots/langfuse_trace.png`
- `screenshots/langfuse_tags.png`
- `screenshots/grafana_eval_run.png`
- `screenshots/grafana_before.png`
- `screenshots/grafana_after.png`

## Excluded From The Zip

- `.env` and `.env.local`
- `.git/`
- `.venv/`
- `.uv-cache/`
- `data/bird/`
- `logs/`
- `__pycache__/`
- local zip/build outputs
