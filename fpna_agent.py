# fpna.py — FP&A Agent using CSVs from ./data (DuckDB + Gemini 1.5 only)
# - Auto-loads all CSVs in ./data (or DATA_DIR env var)
# - Registers each as a DuckDB table (table name = file name without .csv)
# - Tools: BusinessContext (schema), BusinessIntelligenceQuery (run SQL), ConversionTrendAnalysis (helper)
# - LLM: Gemini 1.5 only (flash/pro). NEVER tries gemini-pro.
# - Fallbacks: deterministic pandas/DuckDB; non-agent LLM summary; no crash on parser errors.

from __future__ import annotations

import os
import glob
from typing import Dict, Optional
from textwrap import shorten

import pandas as pd
import numpy as np

# DuckDB (optional but preferred)
try:
    import duckdb  # type: ignore
    HAVE_DUCKDB = True
except Exception:
    duckdb = None
    HAVE_DUCKDB = False

# LangChain (optional)
try:
    from langchain.agents import initialize_agent, Tool, AgentType  # type: ignore
    LANGCHAIN_OK = True
except Exception:
    LANGCHAIN_OK = False

# Gemini via LangChain (AI Studio)
GEMINI_OK = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    GEMINI_OK = True
except Exception:
    GEMINI_OK = False


# ────────────────────────────────────────────────────────────────────────────────
# Data loading / registry

DATA_DIR = os.environ.get("DATA_DIR", "data")

def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV, parsing common date columns if present."""
    df = pd.read_csv(path)
    for col in ("date", "day", "week_start", "month"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="ignore")
    return df

def load_tables(data_dir: str = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load all CSVs from data_dir -> {table_name: DataFrame}."""
    tables: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(csv_path))[0].lower()
        tables[name] = _read_csv(csv_path)
    if not tables:
        raise FileNotFoundError(f"No CSV files found in {data_dir}/")
    return tables

def register_duckdb(tables: Dict[str, pd.DataFrame]):
    """Register DataFrames as DuckDB tables."""
    if not HAVE_DUCKDB:
        raise RuntimeError("DuckDB not installed. Add `duckdb` to requirements.txt and restart.")
    con = duckdb.connect()
    for name, df in tables.items():
        con.register(name, df)
    return con

def schema_summary(tables: Dict[str, pd.DataFrame], sample_rows: int = 1) -> str:
    """Short schema (to keep LLM context lean)."""
    lines = ["**Available tables (from ./data):**"]
    for name, df in tables.items():
        cols = ", ".join(df.columns.tolist()[:12])
        lines.append(f"• {name} ({len(df)} rows) — {cols}")
        try:
            sample = df.head(sample_rows)
            for _, row in sample.iterrows():
                preview = ", ".join(f"{k}={shorten(str(v), 26)}" for k, v in row.items())
                lines.append(f"    - {preview}")
        except Exception:
            pass
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────────
# Tools

def retrieve_business_context(_: str) -> str:
    """Return business context + live schema."""
    tbls = load_tables()
    ctx = (
        "**Business Context (Demo)**\n"
        "- CSVs loaded from ./data (e.g., segment_analysis, channel_performance,\n"
        "  daily_summary, monthly_summary, transaction_data).\n"
        "- Typical asks: conversion by segment, revenue by unit, transaction trends,\n"
        "  channel mix, feature/policy impact.\n\n"
    )
    return ctx + schema_summary(tbls)

EXAMPLE_SQL = """
-- Adjust names to match your ./data schema.

-- 1) Conversion rate trends by segment
SELECT COALESCE(week_start, date, month) AS period, merchant_segment,
       SUM(leads)*1.0/NULLIF(SUM(total_visits),0) AS conversion_rate
FROM segment_analysis
GROUP BY 1,2
ORDER BY 1,2;

-- 2) Revenue by business unit
SELECT COALESCE(week_start, date, month) AS period, business_unit, SUM(revenue) AS revenue
FROM monthly_summary
GROUP BY 1,2
ORDER BY 1,2;

-- 3) Transactions over time
SELECT COALESCE(week_start, date, day) AS period, SUM(transactions) AS txns
FROM transaction_data
GROUP BY 1
ORDER BY 1;

-- 4) Channel acquisition metrics
SELECT acquisition_channel, SUM(leads) AS leads, SUM(total_visits) AS visits,
       SUM(leads)*1.0/NULLIF(SUM(total_visits),0) AS conversion_rate
FROM channel_performance
GROUP BY 1
ORDER BY 2 DESC;
""".strip()

def execute_sql(sql_or_question: str) -> str:
    """
    Run a SELECT/CTE against DuckDB tables registered from ./data.
    If input isn't SQL, return schema + examples.
    """
    q = (sql_or_question or "").strip()
    is_sql = q.lower().startswith("select") or q.lower().startswith("with ")
    if not is_sql:
        return "Provide a SQL SELECT.\n\n" + retrieve_business_context("") + "\n\nExamples:\n" + EXAMPLE_SQL
    try:
        con = register_duckdb(load_tables())
        df = con.execute(q).df()
        return "Query returned no rows." if df.empty else df.to_csv(index=False)
    except Exception as e:
        return f"SQL error: {e}\n\n" + retrieve_business_context("") + "\n\nExamples:\n" + EXAMPLE_SQL

def analyze_conversion_trends() -> str:
    """
    Deterministic path for 'conversion rate trends by merchant segment'.
    Attempts to auto-detect time/segment/visits/leads columns.
    """
    tbls = load_tables()
    candidates = ["segment_analysis", "monthly_summary", "daily_summary", "channel_performance", "transaction_data"]
    table_name = next((t for t in candidates if t in tbls), next(iter(tbls.keys())))
    df = tbls[table_name].copy()

    time_col = next((c for c in ["week_start", "date", "day", "month"] if c in df.columns), None)
    seg_col  = next((c for c in ["merchant_segment", "segment", "customer_segment"] if c in df.columns), None)
    visits   = next((c for c in ["total_visits", "sessions", "visits", "impressions"] if c in df.columns), None)
    leads    = next((c for c in ["leads", "conversions", "signups"] if c in df.columns), None)

    if time_col is None or seg_col is None or visits is None or leads is None:
        raise ValueError(
            f"Missing required columns in `{table_name}`. "
            "Need time (week_start/date/day/month), segment (merchant_segment/segment/customer_segment), "
            "visits (total_visits/sessions/visits/impressions), leads (leads/conversions/signups)."
        )

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["month"] = df[time_col].dt.to_period("M").astype(str)

    g = df.groupby(["month", seg_col]).agg(visits=(visits, "sum"), leads=(leads, "sum")).reset_index()
    g["conversion_rate"] = (g["leads"] / g["visits"]).replace([np.inf, -np.inf], 0).fillna(0.0)

    pivot = g.pivot_table(index="month", columns=seg_col, values="conversion_rate", aggfunc="mean").round(4)
    seg_avg = g.groupby(seg_col)["conversion_rate"].mean().sort_values(ascending=False).round(4)
    monthly = g.groupby("month")["conversion_rate"].mean().round(4)

    return f"""
🎯 CONVERSION RATE TRENDS BY SEGMENT (source: `{table_name}`)

**Conversion by month & segment**
{pivot.to_string()}

**Segment averages**
{seg_avg.to_string()}

**Overall monthly trend**
{monthly.to_string()}
"""


# ────────────────────────────────────────────────────────────────────────────────
# Fallback LLM (non-agent) to avoid OutputParserException crashes

def _fallback_llm_answer(question: str, api_key: Optional[str]) -> str:
    """If the agent can't parse, answer directly without tool-calling."""
    if not (GEMINI_OK and api_key):
        return retrieve_business_context("") + "\n\nTip: start your query with SELECT to run SQL directly."
    try:
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
            timeout=60,
        )
        ctx = retrieve_business_context("")
        prompt = (
            "You are a senior FP&A analyst. Using ONLY the context below, "
            "write a concise executive answer. If data is needed, propose 1–2 SQL queries "
            "against the listed tables (no tool calls, just SQL text).\n\n"
            f"Question:\n{question}\n\nContext:\n{ctx}\n\nAnswer:"
        )
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp))
    except Exception as e:
        return f"{retrieve_business_context('')}\n\n(LLM fallback failed: {e})"


# ────────────────────────────────────────────────────────────────────────────────
# Agent (tool-using) — Gemini 1.5 only

def create_agent(google_api_key: Optional[str] = None):
    """Tool-using agent if possible; otherwise ReliableAgent()."""
    if not (LANGCHAIN_OK and GEMINI_OK and google_api_key):
        return ReliableAgent()

    # Only try 1.5 models (some SDKs want 'models/...' names)
    MODEL_ATTEMPTS = [
        os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
        "gemini-1.5-pro",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
    ]

    llm = None
    for m in MODEL_ATTEMPTS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=m,
                google_api_key=google_api_key,
                temperature=0.1,
                timeout=60,  # generous so it actually finishes
            )
            _ = llm.invoke("ok")  # sanity ping
            break
        except Exception:
            llm = None
            continue

    if llm is None:
        return ReliableAgent()

    tools = [
        Tool(
            name="BusinessContext",
            func=retrieve_business_context,
            description="List tables from ./data with columns and tiny samples."
        ),
        Tool(
            name="BusinessIntelligenceQuery",
            func=execute_sql,
            description="Run a SQL SELECT over DuckDB tables registered from ./data/*.csv."
        ),
        Tool(
            name="ConversionTrendAnalysis",
            func=analyze_conversion_trends,
            description="Compute conversion rate trends by segment deterministically."
        ),
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        early_stopping_method="generate",
        max_iterations=int(os.getenv("AGENT_MAX_STEPS", "12")),
        agent_kwargs={
            "prefix": (
                "You are a senior FP&A analyst. Use BusinessContext to learn the CSV schema from ./data. "
                "For quantitative questions, call BusinessIntelligenceQuery with a concrete SQL SELECT. "
                "For 'conversion rate trends by merchant segment', call ConversionTrendAnalysis."
            ),
            "format_instructions": (
                "Thought → Action → Observation loop. When unsure which table/columns to use, call BusinessContext first."
            ),
        },
    )


# ────────────────────────────────────────────────────────────────────────────────
# Fallback agent (no LLM required)

class ReliableAgent:
    """Deterministic paths; never fails."""
    def run(self, q: str) -> str:
        ql = (q or "").lower()
        if "conversion" in ql and "segment" in ql:
            return analyze_conversion_trends()
        if ql.startswith("select") or ql.startswith("with "):
            return execute_sql(q)
        return retrieve_business_context("")


# ────────────────────────────────────────────────────────────────────────────────
# Public router — always returns an answer (string)

def run_bi(question: str, api_key: Optional[str] = None) -> str:
    """
    One-call BI entrypoint:
      - Direct SQL if question starts with SELECT/WITH
      - Deterministic conversion-trend helper for common ask
      - Otherwise try tool-using agent; on parser/agent errors, fall back to non-agent LLM
    """
    q = (question or "").strip()
    ql = q.lower()

    # 1) direct SQL path
    if ql.startswith("select") or ql.startswith("with "):
        return execute_sql(q)

    # 2) common canned analysis path
    if "conversion" in ql and "segment" in ql:
        return analyze_conversion_trends()

    # 3) agent path
    agent = create_agent(api_key)
    try:
        return agent.run(q)
    except Exception as e:
        # Catch LangChain parser issues (OutputParserException, etc.) & anything else
        err_txt = f"{type(e).__name__}: {e}"
        if "OutputParser" in err_txt or "Could not parse LLM output" in err_txt:
            return _fallback_llm_answer(q, api_key)
        return _fallback_llm_answer(q, api_key)


# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke test (works without API key for deterministic path)
    print(run_bi("conversion rate trends by merchant segment", os.environ.get("GOOGLE_API_KEY")))
