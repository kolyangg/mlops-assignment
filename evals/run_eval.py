"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_ok, gold_rows, gold_error = run_sql(db_id, question["gold_sql"])

    t0 = time.monotonic()
    agent_error: str | None = None
    response: dict = {}
    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(
                agent_url,
                json={
                    "question": question["question"],
                    "db": db_id,
                    "tags": {
                        "run": "eval-baseline",
                        "phase": "phase5",
                        "db": db_id,
                    },
                },
            )
            resp.raise_for_status()
            response = resp.json()
    except Exception as e:  # noqa: BLE001
        agent_error = f"{type(e).__name__}: {e}"
    latency = time.monotonic() - t0

    attempts: list[dict] = []
    if response:
        for item in response.get("history", []):
            if item.get("node") not in {"generate_sql", "revise"}:
                continue
            sql = item.get("sql", "")
            pred_ok, pred_rows, pred_error = run_sql(db_id, sql) if sql else (False, None, "empty SQL")
            attempts.append({
                "iteration": len(attempts),
                "node": item.get("node"),
                "sql": sql,
                "exec_ok": pred_ok,
                "exec_error": pred_error,
                "correct": gold_ok and pred_ok and matches(gold_rows, pred_rows),
                "row_count": len(pred_rows or []),
            })

    final_sql = response.get("sql", "") if response else ""
    final_ok, final_rows, final_error = run_sql(db_id, final_sql) if final_sql else (False, None, "empty SQL")
    final_correct = gold_ok and final_ok and matches(gold_rows, final_rows)

    return {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": question["gold_sql"],
        "gold_exec_ok": gold_ok,
        "gold_exec_error": gold_error,
        "gold_row_count": len(gold_rows or []),
        "agent_ok": response.get("ok") if response else False,
        "agent_error": response.get("error") if response else agent_error,
        "agent_latency_seconds": latency,
        "agent_iterations": response.get("iterations", 0) if response else 0,
        "final_sql": final_sql,
        "final_exec_ok": final_ok,
        "final_exec_error": final_error,
        "final_row_count": len(final_rows or []),
        "final_correct": final_correct,
        "attempts": attempts,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "per_iteration": {},
        }

    max_iteration = max((len(r.get("attempts", [])) for r in results), default=0)
    per_iteration: dict[str, dict] = {}
    for idx in range(max_iteration):
        correct = 0
        attempted = 0
        for result in results:
            attempts = result.get("attempts", [])
            if not attempts:
                is_correct = False
            else:
                carried = attempts[min(idx, len(attempts) - 1)]
                is_correct = bool(carried.get("correct", False))
                attempted += 1
            correct += int(is_correct)
        per_iteration[f"iter_{idx}"] = {
            "correct": correct,
            "total": total,
            "attempted": attempted,
            "accuracy": correct / total,
        }

    final_correct = sum(1 for r in results if r.get("final_correct"))
    agent_ok = sum(1 for r in results if r.get("agent_ok"))
    agent_errors = sum(1 for r in results if r.get("agent_error"))
    latencies = sorted(r.get("agent_latency_seconds", 0.0) for r in results)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        k = int(round(p * (len(latencies) - 1)))
        return latencies[k]

    return {
        "total": total,
        "correct": final_correct,
        "accuracy": final_correct / total,
        "agent_ok": agent_ok,
        "agent_errors": agent_errors,
        "per_iteration": per_iteration,
        "latency_seconds": {
            "p50": pct(0.50),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": latencies[-1] if latencies else 0.0,
        },
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
