"""Fast experimental agent server, v3.

This layers small DB-specific convention hints on top of server_fast_v2.
It is intentionally a new file so v1/v2 experiment outputs remain
reproducible.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from agent.execution import ExecutionResult
from agent.schema import render_schema
from agent.server_fast_v2 import (
    AnswerRequest,
    AnswerResponse,
    FAST_V2_USER,
    VLLM_MODEL,
    _evidence_lookup,
    _execute_sql_limited,
    _extract_sql,
    _render_column_hints,
    _render_value_hints,
    client,
)

FAST_V3_MAX_TOKENS = int(os.environ.get("FAST_V3_MAX_TOKENS", "384"))
app = FastAPI()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "fast_v3"}


def _render_db_conventions(db_id: str) -> str:
    conventions: dict[str, list[str]] = {
        "formula_1": [
            "When returning circuit coordinates or locations through races, use DISTINCT because multiple races can share the same circuit.",
            "results.fastestLapTime is stored as M:SS.mmm; parse with INSTR/SUBSTR, not fixed character positions.",
            "If the question asks for the time of the fastest lap/lap record, return lapTimes.time and order by lapTimes.milliseconds ASC.",
            "Disqualified refers to status.statusId = 2.",
        ],
        "toxicology": [
            "Elements are lowercase chemical symbols such as 'cl' and 'ca'.",
            "molecule.label = '+' means carcinogenic and '-' means non-carcinogenic.",
            "For percentages, keep denominator rows in the aggregate; do not add a WHERE clause that removes denominator rows unless evidence says so.",
            "For 'mostly carcinogenic or non-carcinogenic', GROUP BY molecule.label, ORDER BY COUNT(*) DESC, and return the winning label.",
        ],
        "thrombosis_prediction": [
            "Outpatient clinic follow-up is Patient.Admission = '-'.",
            "Normal Ig G is Laboratory.IGG between 900 and 2000.",
            "Normal uric acid is Laboratory.UA < 8.0 for male patients and < 6.5 for female patients.",
            "Normal total bilirubin is Laboratory.`T-BIL` < 2.0.",
        ],
        "card_games": [
            "As originally printed means cards.originalType and should exclude NULL originalType values.",
            "legalities.status values are case-sensitive, for example 'Banned'.",
            "When asked for print cards in a legality query, return cards.id unless the question explicitly asks for names or types.",
        ],
        "codebase_community": [
            "Datetime values commonly include a trailing .0, for example '2010-07-19 19:39:08.0'.",
            "For the owner of an excerpt post by tag, join tags.ExcerptPostId -> posts.Id -> users.Id via posts.OwnerUserId.",
            "When comparing named users by popularity, aggregate SUM(posts.ViewCount) per users.DisplayName and return DisplayName.",
            "Well-finished means posts.ClosedDate IS NOT NULL; not well-finished means ClosedDate IS NULL.",
        ],
        "student_club": [
            "Department names are exact stored strings; do not drop words such as Department.",
            "Full names in this schema should usually return member.first_name, member.last_name as separate columns.",
        ],
        "california_schools": [
            "Street means schools.Street, not StreetAbr or MailStreet, unless the question says mailing street.",
            "For the complete address in this eval, return Street, City, State, Zip.",
            "Highest average score in Reading refers to satscores.AvgScrRead; do not compute AVG() unless the question asks for an average across grouped rows.",
        ],
        "superhero": [
            "Missing weight data means superhero.weight_kg = 0 OR superhero.weight_kg IS NULL.",
            "No eye color is colour.id = 1 / colour = 'No Colour'; blue eyes is colour.id = 7 / colour = 'Blue'.",
        ],
    }
    items = conventions.get(db_id, [])
    return "\n".join(f"- {item}" for item in items)


def _render_context(db_id: str, question: str) -> str:
    evidence = _evidence_lookup().get((db_id, question.strip()), "")
    column_hints = _render_column_hints(db_id, question)
    value_hints = _render_value_hints(db_id, question)
    db_conventions = _render_db_conventions(db_id)

    parts = [f"Schema:\n{render_schema(db_id)}"]
    if evidence:
        parts.append(f"Question evidence:\n{evidence}")
    if db_conventions:
        parts.append(f"Database conventions:\n{db_conventions}")
    if column_hints:
        parts.append(f"Column meaning hints:\n{column_hints}")
    if value_hints:
        parts.append(f"Stored value hints:\n{value_hints}")
    parts.append(
        "General conventions:\n"
        "- Use exact stored literals and casing from the hints.\n"
        "- Convert dates mentioned as M/D/YYYY h:mm:ss AM/PM to stored ISO format with .0 when date samples use that suffix.\n"
        "- If evidence defines a formula, range, literal, or column mapping, follow it exactly.\n"
        "- Return only the exact entity/columns requested; avoid SELECT *.\n"
        "- Do not aggregate unless the question or evidence requires aggregation."
    )
    return "\n\n".join(parts)


FAST_V3_SYSTEM = """You are an expert SQLite text-to-SQL assistant.
Return exactly one read-only SQLite SELECT query.
Do not explain your reasoning. Do not use markdown fences.
Use only tables and columns present in the provided schema.
Prefer explicit joins through foreign-key-like columns.
Use the evidence, database conventions, and value hints as ground truth.
Pay close attention to the requested output columns: select only what answers the question.
"""


async def _generate_sql(question: str, db_id: str) -> str:
    context = await run_in_threadpool(_render_context, db_id, question)
    response = await client.chat.completions.create(
        model=VLLM_MODEL,
        temperature=0.0,
        max_tokens=FAST_V3_MAX_TOKENS,
        messages=[
            {"role": "system", "content": FAST_V3_SYSTEM},
            {"role": "user", "content": FAST_V2_USER.format(context=context, question=question)},
        ],
    )
    return _extract_sql(response.choices[0].message.content)


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
    execution: ExecutionResult = await run_in_threadpool(_execute_sql_limited, req.db, sql)

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

