"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You are an expert SQLite text-to-SQL assistant.
Return exactly one read-only SQLite SELECT query.
Do not explain your reasoning. Do not use markdown fences.
Use only tables and columns present in the provided schema.
Quote identifiers with double quotes when they contain spaces, punctuation, or reserved words.
Prefer explicit joins through foreign-key-like columns instead of guessing values from unrelated tables.
"""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Schema:
{schema}

Question:
{question}

Write the SQLite query that answers the question. Return SQL only."""


VERIFY_SYSTEM = """You verify whether a SQLite query result plausibly answers a user's question.
Return exactly one JSON object with this schema:
{"ok": true|false, "issue": "short reason"}

Mark ok=false when:
- the SQL execution errored,
- the result is empty or NULL but the question appears to ask for existing rows or a count/aggregate with evidence of a likely value mismatch,
- selected columns do not answer what the question asks for,
- the SQL appears to use the wrong table, wrong join key, wrong column, wrong literal value, or wrong aggregation.

Mark ok=true when the SQL executed and the columns/rows plausibly answer the question.
Do not include markdown or prose outside the JSON object.
"""

VERIFY_USER = """Question:
{question}

Schema:
{schema}

SQL:
{sql}

Execution result:
{execution}

Does the execution result plausibly answer the question? Return JSON only."""


REVISE_SYSTEM = """You revise failed SQLite text-to-SQL attempts.
Return exactly one read-only SQLite SELECT query.
Do not explain your reasoning. Do not use markdown fences.
Use only tables and columns present in the provided schema.
Fix the concrete issue reported by the verifier.
"""

REVISE_USER = """Schema:
{schema}

Question:
{question}

Previous SQL:
{sql}

Previous execution result:
{execution}

Verifier issue:
{issue}

Write a corrected SQLite query. Return SQL only."""
