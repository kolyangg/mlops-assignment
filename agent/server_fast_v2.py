"""Fast experimental agent server, v2.

This is a separate optimization variant and does not replace the submitted
verify/revise agent or the first fast-path experiment.

Goal: keep the SLO-friendly one-call path from server_fast.py while improving
execution accuracy by adding BIRD evidence, natural column names, and compact
stored-value hints to the SQL-generation prompt.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

load_dotenv()

from agent.execution import ExecutionResult  # noqa: E402
from agent.schema import db_path, render_schema  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BIRD_META_DIR = ROOT / "data" / "bird" / "dev_20240627"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")
FAST_V2_MAX_TOKENS = int(os.environ.get("FAST_V2_MAX_TOKENS", "384"))
FAST_V2_SQL_TIMEOUT_SECONDS = float(os.environ.get("FAST_V2_SQL_TIMEOUT_SECONDS", "2.0"))

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
    return {"status": "ok", "mode": "fast_v2"}


def _extract_sql(text: Any) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = (fenced.group(1) if fenced else text).strip()
    stmt = re.search(r"\b(with|select)\b.*", sql, re.DOTALL | re.IGNORECASE)
    if stmt:
        sql = stmt.group(0).strip()
    return sql.rstrip("` \n")


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) >= 3}


@lru_cache(maxsize=1)
def _evidence_lookup() -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for name in ("dev.json", "dev_tied_append.json"):
        path = BIRD_META_DIR / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            evidence = (row.get("evidence") or "").strip()
            if evidence:
                lookup[(row["db_id"], row["question"].strip())] = evidence
    return lookup


@lru_cache(maxsize=1)
def _table_metadata() -> dict[str, dict[str, Any]]:
    path = BIRD_META_DIR / "dev_tables.json"
    if not path.exists():
        return {}
    return {item["db_id"]: item for item in json.loads(path.read_text())}


def _render_column_hints(db_id: str, question: str, max_lines: int = 28) -> str:
    meta = _table_metadata().get(db_id)
    if not meta:
        return ""

    q_tokens = _tokens(question)
    table_names = meta["table_names_original"]
    lines: list[tuple[int, str]] = []
    for (table_idx, original), (_, natural) in zip(
        meta["column_names_original"],
        meta["column_names"],
        strict=False,
    ):
        if table_idx < 0 or original == natural:
            continue
        table = table_names[table_idx]
        haystack = f"{table} {original} {natural}"
        tokens = _tokens(haystack)
        score = len(tokens & q_tokens)
        if re.fullmatch(r"A\d+", original):
            score += 4
        if score == 0:
            continue
        lines.append((score, f"- {table}.{original}: {natural}"))

    lines.sort(key=lambda item: (-item[0], item[1]))
    return "\n".join(line for _, line in lines[:max_lines])


def _is_text_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    for _, name, ctype, *_rest in conn.execute(f'PRAGMA table_info("{table}")'):
        if name == column:
            return any(t in (ctype or "").upper() for t in ("TEXT", "CHAR", "DATE", "TIME"))
    return False


def _render_value_hints(db_id: str, question: str, max_lines: int = 18) -> str:
    """Render compact exact-value hints from low-cardinality text columns."""
    q_tokens = _tokens(question)
    low_cardinality_names = {
        "label", "element", "status", "format", "rarity", "gender", "sex",
        "admission", "colour", "type", "name", "date", "creationdate",
    }
    lines: list[tuple[int, str]] = []
    with sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for table in tables:
            columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
            for column in columns:
                col_key = column.lower()
                relevance = len(_tokens(f"{table} {column}") & q_tokens)
                if col_key not in low_cardinality_names and relevance == 0:
                    continue
                if not _is_text_column(conn, table, column):
                    continue
                try:
                    total = conn.execute(
                        f'SELECT COUNT(DISTINCT "{column}") FROM "{table}" '
                        f'WHERE "{column}" IS NOT NULL'
                    ).fetchone()[0]
                    if total is None or total > 40:
                        # High-cardinality date columns still get a date-format hint elsewhere.
                        continue
                    values = [
                        row[0]
                        for row in conn.execute(
                            f'SELECT DISTINCT "{column}" FROM "{table}" '
                            f'WHERE "{column}" IS NOT NULL '
                            f'ORDER BY "{column}" LIMIT 10'
                        )
                    ]
                except Exception:
                    continue
                values = [str(v) for v in values if str(v) != ""]
                if not values:
                    continue
                score = relevance + (2 if col_key in low_cardinality_names else 0)
                lines.append((score, f"- {table}.{column} values include: {values}"))

    lines.sort(key=lambda item: (-item[0], item[1]))
    return "\n".join(line for _, line in lines[:max_lines])


def _render_context(db_id: str, question: str) -> str:
    evidence = _evidence_lookup().get((db_id, question.strip()), "")
    column_hints = _render_column_hints(db_id, question)
    value_hints = _render_value_hints(db_id, question)
    parts = [f"Schema:\n{render_schema(db_id)}"]
    if evidence:
        parts.append(f"Question evidence:\n{evidence}")
    if column_hints:
        parts.append(f"Column meaning hints:\n{column_hints}")
    if value_hints:
        parts.append(f"Stored value hints:\n{value_hints}")
    parts.append(
        "General conventions:\n"
        "- Use exact stored literals and casing from the hints.\n"
        "- Convert dates mentioned as M/D/YYYY h:mm:ss AM/PM to stored ISO format with .0 when date samples use that suffix.\n"
        "- If evidence defines a formula, range, literal, or column mapping, follow it exactly.\n"
        "- Return the exact entity/columns requested, not explanatory columns.\n"
        "- Do not aggregate unless the question or evidence requires aggregation."
    )
    return "\n\n".join(parts)


FAST_V2_SYSTEM = """You are an expert SQLite text-to-SQL assistant.
Return exactly one read-only SQLite SELECT query.
Do not explain your reasoning. Do not use markdown fences.
Use only tables and columns present in the provided schema.
Prefer explicit joins through foreign-key-like columns.
Use the provided evidence and value hints as ground truth for formulas, ranges, literals, and date formats.
"""

FAST_V2_USER = """{context}

Question:
{question}

Write the SQLite query that answers the question. Return SQL only."""


async def _generate_sql(question: str, db_id: str) -> str:
    context = await run_in_threadpool(_render_context, db_id, question)
    response = await client.chat.completions.create(
        model=VLLM_MODEL,
        temperature=0.0,
        max_tokens=FAST_V2_MAX_TOKENS,
        messages=[
            {"role": "system", "content": FAST_V2_SYSTEM},
            {"role": "user", "content": FAST_V2_USER.format(context=context, question=question)},
        ],
    )
    return _extract_sql(response.choices[0].message.content)


def _execute_sql_limited(db_id: str, sql: str) -> ExecutionResult:
    deadline = time.monotonic() + FAST_V2_SQL_TIMEOUT_SECONDS
    try:
        with sqlite3.connect(f"file:{db_path(db_id)}?mode=ro", uri=True, timeout=1.0) as conn:
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

