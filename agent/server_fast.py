"""Fast experimental agent server for Phase 6 tuning.

This intentionally does not replace agent.server. It preserves the same
/answer request/response shape while testing a cheaper serving path:

    generate_sql -> execute -> deterministic verify

The baseline eval showed no accuracy gain from LLM verify/revise attempts, so
this variant removes those extra full-schema LLM calls to test the latency
impact separately from the completed assignment implementation.
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

load_dotenv()

from agent import prompts  # noqa: E402
from agent.execution import ExecutionResult  # noqa: E402
from agent.schema import render_schema  # noqa: E402
from agent.schema import db_path  # noqa: E402

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
FAST_MAX_TOKENS = int(os.environ.get("FAST_MAX_TOKENS", "256"))
FAST_SQL_TIMEOUT_SECONDS = float(os.environ.get("FAST_SQL_TIMEOUT_SECONDS", "2.0"))

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key=LLM_API_KEY, timeout=120.0)
app = FastAPI()


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = Field(default_factory=dict)


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "fast"}


def _extract_sql(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = (fenced.group(1) if fenced else text).strip()
    stmt = re.search(r"\b(with|select)\b.*", sql, re.DOTALL | re.IGNORECASE)
    if stmt:
        sql = stmt.group(0).strip()
    return sql.rstrip("` \n")


async def _generate_sql(question: str, db_id: str) -> str:
    schema = await run_in_threadpool(render_schema, db_id)
    response = await client.chat.completions.create(
        model=VLLM_MODEL,
        temperature=0.0,
        max_tokens=FAST_MAX_TOKENS,
        messages=[
            {"role": "system", "content": prompts.GENERATE_SQL_SYSTEM},
            {
                "role": "user",
                "content": prompts.GENERATE_SQL_USER.format(
                    schema=schema,
                    question=question,
                ),
            },
        ],
    )
    return _extract_sql(response.choices[0].message.content)


def _execute_sql_limited(db_id: str, sql: str) -> ExecutionResult:
    """Run SQLite with a real query-runtime limit.

    sqlite3.connect(timeout=...) only limits lock waiting. A generated query can
    still spend a long time scanning or cross joining. The progress handler
    aborts those bad queries so load-test requests do not sit open forever.
    """
    deadline = time.monotonic() + FAST_SQL_TIMEOUT_SECONDS
    path = db_path(db_id)
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.0) as conn:
            def should_abort() -> int:
                return int(time.monotonic() > deadline)

            conn.set_progress_handler(should_abort, 10_000)
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return ExecutionResult(ok=True, rows=rows, columns=cols, row_count=len(rows))
    except Exception as e:  # noqa: BLE001
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {e}")


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    history: list[dict[str, Any]] = []
    try:
        sql = await _generate_sql(req.question, req.db)
    except Exception as e:  # noqa: BLE001
        return AnswerResponse(
            sql="",
            rows=None,
            iterations=0,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            history=[{"node": "generate_sql", "error": f"{type(e).__name__}: {e}"}],
        )

    history.append({"node": "generate_sql", "sql": sql})
    execution = await run_in_threadpool(_execute_sql_limited, req.db, sql)

    if not execution.ok:
        history.append({"node": "verify", "ok": False, "issue": execution.error or "SQL execution failed"})
        return AnswerResponse(
            sql=sql,
            rows=None,
            iterations=1,
            ok=False,
            error=execution.error,
            history=history,
        )

    history.append({"node": "verify", "ok": True, "issue": ""})
    return AnswerResponse(
        sql=sql,
        rows=[list(r) for r in (execution.rows or [])],
        iterations=1,
        ok=True,
        history=history,
    )
