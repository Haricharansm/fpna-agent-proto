# fpna_agent.py — FP&A Agent using CSVs from ./data (DuckDB + Gemini 1.5)
import os
import glob
from textwrap import shorten
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import duckdb

# LangChain / agent (optional but preferred)
try:
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    LANGCHAIN_OK = True
except Exception:
    LANGCHAIN_OK = False

# Gemini (AI Studio) preferred
GEMINI_OK = False
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_OK = True
except Exception:
    GEMINI_OK = False


# ───────────────────────────── Data loading / registry ─────────────────────────────

DATA_DIR = os.environ.get("DATA_DIR", "data")

def _read_csv(path: str) -> pd.DataFrame:
    # Parse common date-ish columns if present
    parse_candidates = ["date", "day", "week_start", "month"]
    try:
        df = pd.read_csv(path)
        for col in parse_candidates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="ignore")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed reading {path}: {e}")

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
    con = duckdb.connect()
    for name, df in tables.items():
        con.register(name, df)
    return con

def schema_summary(tables: Dict[str, pd.DataFrame], max_cols: int = 20, sample_rows: int = 2) -> str:
    lines = []
    for name, df in tables.items():
        cols = ", ".join([c for c in df.columns[:max_cols]])
        lines.append(f"• {name} ({len(df)} rows) — columns: {cols}")
        # tiny sample to help LLM infer types
        try:
            sample = df.head(sample_rows)
            lines.append("  samples:")
            for _, row in sample.iterrows():
                preview = ", ".join(f"{k}={shorten(str(v), width=30)}" for k, v in row.items())
                lines.append(f"    - {preview}")
        except Exception:
            pass
    return "**Available tables (DuckDB):**\n" + "\n".join(lines)

# ───────────────────────────── Tools ─────────────────────────────

def retrieve_business_context(_: str) -> str:
    """Returns schema + light business context so the LLM can write correct SQL."""
    tbls = load_tables()
    ctx = (
        "**Business Context (Demo)**\n"
        "- You are analyzing Q4-type business CSVs in ./data.\n"
        "- Common questions: conversion by segment, revenue by unit, transaction trends, channel mix,\n"
        "  feature/policy impact if those flags exist in a table.\n\n"
    )
    return ctx + schema_summary(tbls)

EXAMPLE_SQL = """
-- Examples (use the tables that actually exist)
-- If you have a 'segment_analysis' table with visits/leads:
SELECT COALESCE(week_start, date) AS period, merchant_segment,
       SUM(leads)*1.0/NULLIF(SUM(total_visits),0) AS conversion_rate
FROM segment_analysis
GROUP BY 1,2
ORDER BY 1,2;

-- Revenue by business_unit (try 'channel_performance' or 'monthly_summary' etc.)
SELECT COALESCE(week_start, month, date) AS period, business_unit,
       SUM(revenue) AS revenue
FROM monthly_summary
GROUP BY 1,2
ORDER BY 1,2;

-- Transactions over time (try 'transaction_data' or 'daily_summary')
SELECT COALESCE(week_start, date) AS period, SUM(transactions) AS txns
FROM transaction_data
GROUP BY 1
ORDER BY 1;
""".strip()

def execute_sql(sql_or_question: str) -> str:
    """Run a SELECT (or CTE) against the registered DuckDB tables."""
    q = (sql_or_question or "").strip()
    is_sql = q.lower().startswith("select") or q.lower().startswith("with ")
    if not is_sql:
        # Give the LLM (or user) examples based on current schema
        return "Please provide a SQL SELECT.\n\nSchema:\n" + retrieve_business_context("") + "\n\nExamples:\n" + EXAMPLE_SQL
    try:
        con = register_duckdb(load_tables())
        df = con.execute(q).df()
        return "Query returned no rows." if df.empty else df.to_csv(index=False)
    except Exception as e:
        return f"SQL error: {e}\n\nSchema:\n{retrieve_business_context('')}\n\nExamples:\n{EXAMPLE_SQL}"

def analyze_conversion_trends() -> str:
    """
    Heuristic, no-SQL helper: find a table with {segment + visits + leads} and compute monthly conversion.
    Tries common table/column names used in your repo.
    """
    tbls = load_tables()
    # prefer a table named segment_analysis if present
    candidates = ["segment_analysis", "monthly_summary", "daily_summary", "channel_performance", "transaction_data"]
    table_name = next((t for t in candidates if t in tbls), next(iter(tbls.keys())))
    df = tbls[table_name].copy()

    # figure out time column
    time_col = next((c for c in ["week_start", "date", "day", "month"] if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"No time column found in {table_name}. Expected one of: week_start, date, day, month")
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # segment column
    seg_col = next((c for c in ["merchant_segment", "segment", "customer_segment"] if c in df.columns), None)
    if seg_col is None:
        raise ValueError(f"No segment column found in {table_name}. Expected one of: merchant_segment, segment, customer_segment")

    # counts
    visits_col = next((c for c in ["total_visits", "sessions", "visits", "impressions"] if c in df.columns), None)
    leads_col  = next((c for c in ["leads", "conversions", "signups"] if c in df.columns), None)
    if not visits_col or not leads_col:
        raise ValueError(
            f"Need visits and leads columns in {table_name}. "
            f"Looked for visits in ['total_visits','sessions','visits','impressions'] and leads in ['leads','conversions','signups']."
        )

    # month for trend
    df["month"] = df[time_col].dt.to_period("M").astype(str)
    g = df.groupby(["month", seg_col]).agg(
        visits=(visits_col, "sum"),
        leads=(leads_col, "sum"),
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

# ───────────────────────────── Agent factory ─────────────────────────────

class ReliableAgent:
    """No-LLM fallback that still answers usefully."""
    def run(self, q: str) -> str:
        ql = (q or "").lower()
        if "conversion" in ql and "segment" in ql:
            return analyze_conversion_trends()
        if ql.startswith("select") or ql.startswith("with "):
            return execute_sql(q)
        # default: show schema so the user can craft SQL
        return retrieve_business_context("")

def create_agent(google_api_key: str | None = None):
    """
    If LangChain + Gemini are available, return a tool-using agent that:
      - retrieves schema/context
      - executes SQL over ./data tables
      - runs the prebuilt conversion analysis
    Otherwise, return ReliableAgent().
    """
    if not (LANGCHAIN_OK and GEMINI_OK and google_api_key):
        return ReliableAgent()

    # Prefer current Gemini names (no 'gemini-pro')
    for model in [os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"), "gemini-1.5-pro"]:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=google_api_key,
                temperature=0.1,
                timeout=15,
            )
            # sanity ping
            _ = llm.invoke("ok")
            break
        except Exception:
            llm = None
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
                "When not sure which table/columns to use, call BusinessContext first."
            ),
        },
    )

if __name__ == "__main__":
    agent = create_agent(os.environ.get("GOOGLE_API_KEY"))
    print(agent.run("conversion rate trends by merchant segment"))
