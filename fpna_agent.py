# fpna.py — deterministic FP&A core (no LangChain agent)
# - CSVs in ./data → DuckDB
# - Smarter deterministic helpers + optional direct Gemini 1.5 fallback (no ReAct, no parser errors)

from __future__ import annotations

import os
import glob
from typing import Dict, Optional
from textwrap import shorten

import pandas as pd
import numpy as np

# DuckDB (preferred for SQL; optional)
try:
    import duckdb  # type: ignore
    HAVE_DUCKDB = True
except Exception:
    duckdb = None
    HAVE_DUCKDB = False

# Optional direct Gemini SDK (no LangChain)
try:
    import google.generativeai as genai  # type: ignore
    GENAI_OK = True
except Exception:
    genai = None
    GENAI_OK = False

DATA_DIR = os.environ.get("DATA_DIR", "data")

# ------------------------------ Column mapping & priorities ------------------------------

PREFERRED_TABLE_ORDER = [
    "monthly_summary",
    "transaction_data",
    "segment_analysis",
    "daily_summary",
    "channel_performance",
]

TIME_ALIASES    = ["week_start", "date", "day", "month", "period"]
SEGMENT_ALIASES = ["merchant_segment", "segment", "customer_segment"]
VISITS_ALIASES  = ["total_visits", "sessions", "visits", "impressions"]
LEADS_ALIASES   = ["leads", "conversions", "converted", "signups"]

def _priority_rank(name: str) -> int:
    try:
        return PREFERRED_TABLE_ORDER.index(name)
    except Exception:
        return 99

def _pick_conversion_source(tables: Dict[str, pd.DataFrame]):
    """
    Pick the table that actually has time + segment + visits + leads.
    Tie-breaks using PREFERRED_TABLE_ORDER.
    Returns: (table_name, mapping_dict, score)
    """
    best_name, best_map, best_score = None, None, -1
    for name, df in tables.items():
        mapping = {
            "time":    next((c for c in TIME_ALIASES    if c in df.columns), None),
            "segment": next((c for c in SEGMENT_ALIASES if c in df.columns), None),
            "visits":  next((c for c in VISITS_ALIASES  if c in df.columns), None),
            "leads":   next((c for c in LEADS_ALIASES   if c in df.columns), None),
        }
        score = sum(v is not None for v in mapping.values())
        if score > best_score or (score == best_score and _priority_rank(name) < _priority_rank(best_name or "")):
            best_name, best_map, best_score = name, mapping, score
    return best_name, best_map, best_score


# ────────────────────────────────────────────────────────────────────────────────
# Data loading / registry

def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("date", "day", "week_start", "month"):
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
# Tools / helpers

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
-- Adjust names to your ./data schema.

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

# ------------------------------ Deterministic analyses ------------------------------

def analyze_conversion_trends() -> str:
    """
    Compute monthly conversion rate by merchant segment from the best available table.
    Returns CSV: month,merchant_segment,visits,leads,conversion_rate
    """
    tables = load_tables()
    table_name, mapping, score = _pick_conversion_source(tables)

    if not table_name or score < 4:
        raise ValueError(
            "No single table has the required columns.\n"
            "Need time (week_start/date/day/month), segment (merchant_segment/segment/customer_segment), "
            "visits (total_visits/sessions/visits/impressions), leads (leads/conversions/converted/signups)."
        )

    df = tables[table_name].copy()
    time_col, seg_col, visits_col, leads_col = (
        mapping["time"], mapping["segment"], mapping["visits"], mapping["leads"]
    )

    # Normalize time → month
    if time_col == "month" and df[time_col].dtype == "O":
        try:
            df["month"] = pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").astype(str)
        except Exception:
            df["month"] = df[time_col].astype(str)
    else:
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df["month"] = df[time_col].dt.to_period("M").astype(str)

    # Aggregate
    g = (
        df.groupby(["month", seg_col])
          .agg(visits=(visits_col, "sum"), leads=(leads_col, "sum"))
          .reset_index()
    )
    g["conversion_rate"] = (g["leads"] / g["visits"]).replace([np.inf, -np.inf], 0).fillna(0.0)

    # Standardize column names for the UI
    g = g.rename(columns={seg_col: "merchant_segment"})
    g = g[["month", "merchant_segment", "visits", "leads", "conversion_rate"]]
    g = g.sort_values(["month", "merchant_segment"])

    # Return CSV so Streamlit renders table + line chart automatically
    return g.to_csv(index=False)

def revenue_by_business_unit() -> str:
    """Return CSV: month,business_unit,revenue from monthly_summary if present."""
    tables = load_tables()
    if "monthly_summary" not in tables:
        raise ValueError("monthly_summary.csv not found in ./data")
    df = tables["monthly_summary"].copy()
    # Normalize month
    if df["month"].dtype == "O":
        try:
            df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").astype(str)
        except Exception:
            df["month"] = df["month"].astype(str)
    out = (
        df.groupby(["month", "business_unit"])["revenue"]
          .sum()
          .reset_index()
          .sort_values(["month", "business_unit"])
    )
    return out.to_csv(index=False)

# ────────────────────────────────────────────────────────────────────────────────
# Direct Gemini fallback (no LangChain, no agent)

def _fallback_llm_answer(question: str, api_key: Optional[str]) -> str:
    if not (GENAI_OK and api_key):
        return retrieve_business_context("") + "\n\nTip: start your query with SELECT to run SQL directly."
    try:
        genai.configure(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        prompt = (
            "You are a senior FP&A analyst. Using ONLY the context below, "
            "write a concise executive answer. If data is needed, propose 1–2 SQL queries "
            "for DuckDB over the listed tables (plain SQL text only).\n\n"
            f"Question:\n{question}\n\nContext:\n{retrieve_business_context('')}\n\nAnswer:"
        )
        resp = model.generate_content(prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"{retrieve_business_context('')}\n\n(LLM fallback failed: {e})"

# ────────────────────────────────────────────────────────────────────────────────
# Public router — always returns a string

def run_bi(question: str, api_key: Optional[str] = None) -> str:
    q = (question or "").strip()
    ql = q.lower()

    # 1) direct SQL if the user gave SQL
    if ql.startswith("select") or ql.startswith("with "):
        return execute_sql(q)

    # 2) deterministic routes
    if "conversion" in ql and "segment" in ql:
        try:
            return analyze_conversion_trends()
        except Exception as e:
            return f"{retrieve_business_context('')}\n\n(Conversion analysis failed: {e})"

    if ("revenue" in ql) and ("business unit" in ql or "business_unit" in ql or "unit" in ql):
        try:
            return revenue_by_business_unit()
        except Exception as e:
            return f"{retrieve_business_context('')}\n\n(Revenue analysis failed: {e})"

    # 3) non-agent LLM summary (never ReAct, never parser errors)
    return _fallback_llm_answer(q, api_key)

# ────────────────────────────────────────────────────────────────────────────────
# Minimal wrapper so existing app code that expects `.run()` still works

class RouterAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    def run(self, q: str) -> str:
        return run_bi(q, self.api_key)

def create_agent(google_api_key: Optional[str] = None):
    """Return a simple router agent; no LangChain, no ReAct."""
    return RouterAgent(google_api_key)

# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # smoke tests
    print(run_bi("conversion rate trends by merchant segment"))
    print(run_bi("revenue by business unit"))
    print(run_bi("SELECT 1 AS a, 2 AS b"))
