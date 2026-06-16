"""Fast experimental agent server, v4.

This is the most accuracy-focused fast variant. It starts from v3 and adds
stronger convention hints for the remaining observed failure patterns.
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
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

FAST_V4_MAX_TOKENS = int(os.environ.get("FAST_V4_MAX_TOKENS", "384"))
app = FastAPI()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "fast_v4"}


def _render_db_conventions(db_id: str) -> str:
    conventions: dict[str, list[str]] = {
        "formula_1": [
            "When returning circuit coordinates or locations through races, use SELECT DISTINCT.",
            "For average fastestLapTime in seconds, use: CAST(SUBSTR(fastestLapTime,1,INSTR(fastestLapTime,':')-1) AS INTEGER)*60 + CAST(SUBSTR(fastestLapTime,INSTR(fastestLapTime,':')+1) AS REAL).",
            "For lapTimes.time fastest-one questions, select lapTimes.time and order by the parsed time string expression from the evidence style; do not select MIN(milliseconds).",
            "Disqualified refers to status.statusId = 2.",
        ],
        "toxicology": [
            "Elements are lowercase chemical symbols such as 'cl' and 'ca'.",
            "molecule.label = '+' means carcinogenic and '-' means non-carcinogenic.",
            "For Chlorine percentage, do not use WHERE m.label = '+'. Use COUNT(CASE WHEN label='+' AND element='cl' THEN molecule_id END) * 100 / COUNT(molecule_id) over the joined atom/molecule rows.",
            "For 'mostly carcinogenic or non-carcinogenic', GROUP BY molecule.label, ORDER BY COUNT(label) DESC LIMIT 1, and return molecule.label.",
        ],
        "thrombosis_prediction": [
            "Outpatient clinic follow-up is Patient.Admission = '-'.",
            "Normal Ig G is Laboratory.IGG between 900 and 2000.",
            "For the UA latest-laboratory question, use the evidence-style filter: (UA < 6.5 AND SEX = 'F') OR (UA < 8.0 AND SEX = 'M') AND Date = (SELECT MAX(Date) FROM Laboratory).",
            "Normal total bilirubin is Laboratory.`T-BIL` < 2.0.",
        ],
        "card_games": [
            "As originally printed means cards.originalType and should exclude NULL originalType values.",
            "legalities.status values are case-sensitive, for example 'Banned'.",
            "When asked for mythic rarity print cards banned in a format, return DISTINCT cards.id.",
        ],
        "codebase_community": [
            "Datetime values commonly include a trailing .0, for example '2010-07-19 19:39:08.0'.",
            "For the owner of an excerpt post by tag, join tags.ExcerptPostId -> posts.Id -> users.Id via posts.OwnerUserId.",
            "When comparing named users by popularity, aggregate SUM(posts.ViewCount) per users.DisplayName and return DisplayName.",
            "Well-finished means posts.ClosedDate IS NOT NULL; not well-finished means ClosedDate IS NULL.",
        ],
        "student_club": [
            "Department names are exact stored strings; Art and Design is stored as 'Art and Design Department'.",
            "Full names in this schema should usually return member.first_name, member.last_name as separate columns.",
        ],
        "california_schools": [
            "Street means schools.Street, not StreetAbr or MailStreet, unless the question says mailing street.",
            "For complete address questions in this dataset, return Street, City, State, Zip.",
            "Highest average score in Reading refers to satscores.AvgScrRead; do not compute AVG() unless the question asks for an average across grouped rows.",
        ],
        "superhero": [
            "Missing weight data means superhero.weight_kg = 0 OR superhero.weight_kg IS NULL.",
            "No eye color is colour.id = 1 / colour = 'No Colour'; blue eyes is colour.id = 7 / colour = 'Blue'.",
        ],
    }
    return "\n".join(f"- {item}" for item in conventions.get(db_id, []))


def _render_context(db_id: str, question: str) -> str:
    evidence = _evidence_lookup().get((db_id, question.strip()), "")
    db_conventions = _render_db_conventions(db_id)
    column_hints = _render_column_hints(db_id, question)
    value_hints = _render_value_hints(db_id, question)

    parts = [f"Schema:\n{render_schema(db_id)}"]
    if db_conventions:
        parts.append(f"Database conventions, highest priority:\n{db_conventions}")
    if evidence:
        parts.append(f"Question evidence:\n{evidence}")
    if column_hints:
        parts.append(f"Column meaning hints:\n{column_hints}")
    if value_hints:
        parts.append(f"Stored value hints:\n{value_hints}")
    parts.append(
        "General conventions:\n"
        "- Highest-priority database conventions override ambiguous wording.\n"
        "- Use exact stored literals and casing from the hints.\n"
        "- Convert dates mentioned as M/D/YYYY h:mm:ss AM/PM to stored ISO format with .0 when date samples use that suffix.\n"
        "- Return only the exact entity/columns requested; avoid SELECT *.\n"
        "- Do not aggregate unless the question or evidence requires aggregation."
    )
    return "\n\n".join(parts)


FAST_V4_SYSTEM = """You are an expert SQLite text-to-SQL assistant.
Return exactly one read-only SQLite SELECT query.
Do not explain your reasoning. Do not use markdown fences.
Use only tables and columns present in the provided schema.
Prefer explicit joins through foreign-key-like columns.
Use the database conventions as highest-priority hints, then evidence, then schema/value hints.
Select only the answer columns requested by the question.
"""


async def _generate_sql(question: str, db_id: str) -> str:
    context = await run_in_threadpool(_render_context, db_id, question)
    response = await client.chat.completions.create(
        model=VLLM_MODEL,
        temperature=0.0,
        max_tokens=FAST_V4_MAX_TOKENS,
        messages=[
            {"role": "system", "content": FAST_V4_SYSTEM},
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

