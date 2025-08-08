# fpna.py — FP&A Agent using CSVs from ./data (DuckDB + Gemini 1.5 only)
# Fast-path router included to avoid agent timeouts for common asks

import os
import glob
from typing import Dict, Optional
from textwrap import shorten

import pandas as pd
import numpy as np

# DuckDB can be optional; we guard usage
try:
    import duckdb
    HAVE_DUCKDB = True
except Exception:
    duckdb = None
    HAVE_DUCKDB = False

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
    df = pd.read_csv(path)
    for col in ["date", "day", "week_start", "month"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="ignore")
    return df

def load_tables(data_dir: str = DATA_DIR) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        name = os.path.splitext(os.path.basename(csv_path))[0].lower()
        tables[name] = _read_csv(csv_path)
    if not tables:
        raise FileNotFoundError(f"No CSV files found in {data_dir}/")
    return tables

def register_duckdb(tables: Dict[str, pd.DataFrame]):
    if not HAVE_DUCKDB:
        raise RuntimeError("DuckDB not installed. Add `duckdb` to requirements.txt and restart.")
    con = duckdb.connect()
    for name, df in tables.items():
        con.register(name, df)
    return con

def schema_summary(tables: Dict[str, pd.DataFrame], sample_rows: int = 1) -> str:
    # Kept short to avoid agent token/iteration blow-ups
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
-- Adjust table/column names to your ./data schema.

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
    """Deterministic path for 'conversion rate trends by merchant segment'."""
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

    g = df.groupby(["month", seg_col]).agg(visits=(visits,"sum"), leads=(leads,"sum")).reset_index()
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
    """Tool-using agent if possible; otherwise ReliableAgent."""
    if not (LANGCHAIN_OK and GEMINI_OK and google_api_key):
        return ReliableAgent()

    # Only try 1.5 models
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
                timeout=60,  # ↑ more time so it actually finishes
            )
            _ = llm.invoke("ok")
            break
        except Exception:
            llm = None
            continue

    if llm is None:
        return ReliableAgent()

    tools = [
        Tool(name="BusinessContext", func=retrieve_business_context,
             description="List tables from ./data with columns and tiny samples."),
        Tool(name="BusinessIntelligenceQuery", func=execute_sql,
             description="Run a SQL SELECT over DuckDB tables registered from ./data/*.csv."),
        Tool(name="ConversionTrendAnalysis", func=analyze_conversion_trends,
             description="Compute conversion rate trends by segment deterministically."),
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
        early_stopping_method="generate",
        max_iterations=int(os.getenv("AGENT_MAX_STEPS", "12")),  # ↑ more steps
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
# One-call router that always returns an answer (string)
def run_bi(question: str, api_key: Optional[str] = None) -> str:
    q = (question or "").strip()
    ql = q.lower()
    # 1) direct SQL
    if ql.startswith("select") or ql.startswith("with "):
        return execute_sql(q)
    # 2) common ask
    if "conversion" in ql and "segment" in ql:
        return analyze_conversion_trends()
    # 3) agent fallback
    agent = create_agent(api_key)
    return agent.run(q)

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(run_bi("conversion rate trends by merchant segment", os.environ.get("GOOGLE_API_KEY")))
