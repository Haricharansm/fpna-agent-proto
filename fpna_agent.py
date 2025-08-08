# fpna.py — FP&A Agent using CSVs from ./data (DuckDB + Gemini 1.5 only)
# - Auto-loads all CSVs in ./data (or DATA_DIR env var)
# - Registers each as a DuckDB table (table name = file name without .csv)
# - Tools: BusinessContext (schema), BusinessIntelligenceQuery (run SQL), ConversionTrendAnalysis (helper)
# - LLM: Gemini 1.5 (flash/pro) ONLY — never tries gemini-pro
# - Fallback: ReliableAgent that still answers without LLM

import os
import glob
from typing import Dict, Optional
from textwrap import shorten

import pandas as pd
import numpy as np
import duckdb

# ────────────────────────────────────────────────────────────────────────────────
# LangChain / agent (optional but preferred)
try:
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    LANGCHAIN_OK = True
except Exception:
    LANGCHAIN_OK = False

# Gemini (AI Studio) — 1.5 only
GEMINI_OK = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_OK = True
except Exception:
    GEMINI_OK = False


# ────────────────────────────────────────────────────────────────────────────────
# Data loading / registry

DATA_DIR = os.environ.get("DATA_DIR", "data")

def _read_csv(path: str) -> pd.DataFrame:
    """Read a CSV, parsing common dateish columns if present."""
    df = pd.read_csv(path)
    for col in ["date", "day", "week_start", "month"]:
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

def register_duckdb(tables: Dict[str, pd.DataFrame]) -> duckdb.DuckDBPyConnection:
    """Register each DataFrame as a DuckDB table of the same name."""
    con = duckdb.connect()
    for name, df in tables.items():
        con.register(name, df)
    return con

def schema_summary(tables: Dict[str, pd.DataFrame], sample_rows: int = 2) -> str:
    """Human-readable schema description with a couple sample rows per table."""
    lines = ["**Available tables (DuckDB registered from ./data):**"]
    for name, df in tables.items():
        cols = ", ".join(df.columns.tolist())
        lines.append(f"• {name} ({len(df)} rows) — columns: {cols}")
        try:
            sample = df.head(sample_rows)
            for _, row in sample.iterrows():
                preview = ", ".join(f"{k}={shorten(str(v), width=30)}" for k, v in row.items())
                lines.append(f"    - {preview}")
        except Exception:
            pass
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────────────────
# Tools

def retrieve_business_context(_: str) -> str:
    """Return schema + light business context so the LLM can write correct SQL."""
    tbls = load_tables()
    ctx = (
        "**Business Context (Demo)**\n"
        "- You are analyzing CSVs from ./data (e.g., segment_analysis, channel_performance,\n"
        "  daily_summary, monthly_summary, transaction_data). Common asks: conversion by segment,\n"
        "  revenue by business unit, transaction trends, channel mix, feature/policy impacts.\n\n"
    )
    return ctx + schema_summary(tbls)

EXAMPLE_SQL = """
-- Examples (adjust table/column names to ones that exist in your ./data):
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
    If the input isn't SQL, return schema + example queries.
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
    Heuristic helper: find a table with {segment + visits + leads} and compute monthly conversion.
    Tries common table/column names (works with many simple CSVs).
    """
    tbls = load_tables()
    # choose a likely table
    candidates = ["segment_analysis", "monthly_summary", "daily_summary", "channel_performance", "transaction_data"]
    table_name = next((t for t in candidates if t in tbls), next(iter(tbls.keys())))
    df = tbls[table_name].copy()

    # guess columns
    time_col = next((c for c in ["week_start", "date", "day", "month"] if c in df.columns), None)
    seg_col  = next((c for c in ["merchant_segment", "segment", "customer_segment"] if c in df.columns), None)
    visits   = next((c for c in ["total_visits", "sessions", "visits", "impressions"] if c in df.columns), None)
    leads    = next((c for c in ["leads", "conversions", "signups"] if c in df.columns), None)

    if time_col is None or seg_col is None or visits is None or leads is None:
        raise ValueError(
            f"Could not auto-detect required columns in `{table_name}`. "
            f"Need time, segment, visits, leads. "
            f"Time looked for one of: week_start,date,day,month; "
            f"segment: merchant_segment,segment,customer_segment; "
            f"visits: total_visits,sessions,visits,impressions; "
            f"leads: leads,conversions,signups."
        )

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["month"] = df[time_col].dt.to_period("M").astype(str)

    g = df.groupby(["month", seg_col]).agg(
        visits=(visits, "sum"),
        leads=(leads, "sum"),
    ).reset_index()
    g["conversion_rate"] = (g["leads"] / g["visits"]).replace([np.inf, -np.inf], 0).fillna(0.0)

    pivot = g.pivot_table(index="month", columns=seg_col, values="conversion_rate", aggfunc="mean").round(4)
    seg_avg = g.groupby(seg_col)["conversion_rate"].mean().sort_values(ascending=False).round(4)
    monthly = g.groupby("month")["conversion_rate"].mean().round(4)

    return f"""
🎯 CONVERSION RATE TRENDS BY SEGMENT (auto-detected table: `{table_name}`)

📈 Conversion by month & segment:
{pivot.to_string()}

🏆 Segment averages:
{seg_avg.to_string()}

📅 Overall monthly trend:
{monthly.to_string()}
"""


# ────────────────────────────────────────────────────────────────────────────────
# Fallback agent (no LLM required)

class ReliableAgent:
    def run(self, q: str) -> str:
        ql = (q or "").lower()
        if "conversion" in ql and "segment" in ql:
            return analyze_conversion_trends()
        if ql.startswith("select") or ql.startswith("with "):
            return execute_sql(q)
        return retrieve_business_context("")


# ────────────────────────────────────────────────────────────────────────────────
# Agent factory — Gemini 1.5 only (NEVER gemini-pro)

def create_agent(google_api_key: Optional[str] = None):
    """
    If LangChain + Gemini are available, return a tool-using agent that:
      - retrieves schema/context
      - executes SQL over ./data tables
      - runs conversion trend helper
    Otherwise, return ReliableAgent().
    """
    if not (LANGCHAIN_OK and GEMINI_OK and google_api_key):
        return ReliableAgent()

    # Try ONLY 1.5 models; some SDKs accept 'models/...' names
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
                timeout=15,
            )
            _ = llm.invoke("ok")  # quick sanity ping
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
            description="Returns the list of available tables (from ./data) with columns and small samples."
        ),
        Tool(
            name="BusinessIntelligenceQuery",
            func=execute_sql,
            description="Executes a SQL SELECT against DuckDB tables registered from ./data/*.csv."
        ),
        Tool(
            name="ConversionTrendAnalysis",
            func=analyze_conversion_trends,
            description="Auto-detects an appropriate table to compute conversion rate trends by segment."
        ),
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
        agent_kwargs={
            "prefix": (
                "You are a senior FP&A analyst. Use BusinessContext to learn the schema of CSV tables "
                "loaded from ./data. For quantitative questions, call BusinessIntelligenceQuery with a "
                "concrete SQL SELECT. For 'conversion trends by segment', you may call ConversionTrendAnalysis."
            ),
            "format_instructions": (
                "Thought → Action → Observation loop.\n"
                "When unsure which table/columns to use, call BusinessContext first."
            ),
        },
    )


# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Smoke test without LLM
    agent = create_agent(os.environ.get("GOOGLE_API_KEY"))
    print(agent.run("conversion rate trends by merchant segment"))
