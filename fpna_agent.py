# fpna_agent.py — deterministic FP&A core + HF (Mistral) summaries, no LangChain/Gemini

from __future__ import annotations

import os
import glob
from typing import Dict, Optional, List, Tuple
from textwrap import shorten

import pandas as pd
import numpy as np

FPNA_VERSION = "1.2.0"
FPNA_DEBUG = os.getenv("FPNA_DEBUG", "0") == "1"
DATA_DIR = os.environ.get("DATA_DIR", "data")

# ────────────────────────────────────────────────────────────────────────────────
# Optional: Hugging Face Inference API (for summaries)
HAVE_HF = False
try:
    from huggingface_hub import InferenceClient
    HAVE_HF = True
except Exception:
    InferenceClient = None
    HAVE_HF = False

# ────────────────────────────────────────────────────────────────────────────────
# DuckDB (preferred for SQL; optional)
try:
    import duckdb  # type: ignore
    HAVE_DUCKDB = True
except Exception:
    duckdb = None
    HAVE_DUCKDB = False

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

def _first_in(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    return next((c for c in aliases if c in df.columns), None)

def _pick_conversion_source(tables: Dict[str, pd.DataFrame]) -> Tuple[Optional[str], Dict[str, str], int]:
    """
    Pick the table that actually has time + segment + visits + leads.
    Tie-breaks using PREFERRED_TABLE_ORDER.
    Returns: (table_name, mapping_dict, score)
    """
    best_name, best_map, best_score = None, {}, -1
    for name, df in tables.items():
        mapping = {
            "time":    _first_in(df, TIME_ALIASES),
            "segment": _first_in(df, SEGMENT_ALIASES),
            "visits":  _first_in(df, VISITS_ALIASES),
            "leads":   _first_in(df, LEADS_ALIASES),
        }
        score = sum(v is not None for v in mapping.values())
        if score > best_score or (score == best_score and _priority_rank(name) < _priority_rank(best_name or "")):
            best_name, best_map, best_score = name, mapping, score
    return best_name, best_map, best_score

# ────────────────────────────────────────────────────────────────────────────────
# Data loading / registry

def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("date", "day", "week_start"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # leave "month" as-is (string like '2024-10' is fine); normalize later
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
        raise RuntimeError("DuckDB not installed. Add `duckdb` to requirements.txt and redeploy.")
    con = duckdb.connect()
    for name, df in tables.items():
        con.register(name, df)
    return con

def schema_summary(tables: Dict[str, pd.DataFrame], sample_rows: int = 1) -> str:
    lines = [f"**Available tables (from ./data) — fpna v{FPNA_VERSION}:**"]
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

def _normalize_month_from(df: pd.DataFrame, time_col: str) -> pd.Series:
    if time_col == "month":
        try:
            return pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").astype(str)
        except Exception:
            return df[time_col].astype(str)
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df[time_col].dt.to_period("M").astype(str)

def analyze_conversion_trends() -> str:
    """
    Compute monthly conversion rate by merchant segment from the best available table.
    Returns CSV: month,merchant_segment,visits,leads,conversion_rate
    """
    tables = load_tables()
    table_name, mapping, score = _pick_conversion_source(tables)

    if FPNA_DEBUG:
        print("[fpna] conversion picker →", table_name, mapping, "score:", score)

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

    month = _normalize_month_from(df, time_col)

    g = (
        df.assign(month=month)
          .groupby(["month", seg_col])
          .agg(visits=(visits_col, "sum"), leads=(leads_col, "sum"))
          .reset_index()
    )
    g["conversion_rate"] = (g["leads"] / g["visits"]).replace([np.inf, -np.inf], 0).fillna(0.0)

    g = g.rename(columns={seg_col: "merchant_segment"})
    g = g[["month", "merchant_segment", "visits", "leads", "conversion_rate"]]
    g = g.sort_values(["month", "merchant_segment"])
    return g.to_csv(index=False)

def revenue_by_business_unit() -> str:
    """Return CSV: month,business_unit,revenue from monthly_summary if present."""
    tables = load_tables()
    if "monthly_summary" not in tables:
        raise ValueError("monthly_summary.csv not found in ./data")
    df = tables["monthly_summary"].copy()
    # Normalize month
    if "month" in df.columns:
        df["month"] = _normalize_month_from(df, "month")
    else:
        time_col = _first_in(df, TIME_ALIASES) or "date"
        df["month"] = _normalize_month_from(df, time_col)

    out = (
        df.groupby(["month", "business_unit"])["revenue"]
          .sum()
          .reset_index()
          .sort_values(["month", "business_unit"])
    )
    return out.to_csv(index=False)

# ────────────────────────────────────────────────────────────────────────────────
# HF Mistral summarizer (executive bullets on top of CSV)

def _looks_like_csv(text: str) -> bool:
    if not text or not text.strip():
        return False
    first = text.strip().splitlines()[0]
    return "," in first and len(first.split(",")) >= 2

def _hf_summarize(question: str, csv_text: str) -> Optional[str]:
    if not HAVE_HF:
        return None
    model = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
    token = os.getenv("HF_TOKEN")  # optional (anonymous works but may rate-limit)
    client = InferenceClient(model=model, token=token)

    # Keep the CSV short for free-tier latency
    lines = csv_text.splitlines()
    if len(lines) > 300:
        csv_text = "\n".join([lines[0]] + lines[1:301])

    prompt = (
        "You are a senior FP&A analyst. Use ONLY the CSV below to answer the question.\n"
        "- 5–8 concise bullets with numbers (rates, deltas, tops/bottoms).\n"
        "- If a metric is missing in the CSV, say so—do NOT invent values.\n\n"
        f"CSV:\n{csv_text}\n\nQuestion: {question}"
    )

    last_err = None
    for _ in range(2):  # light retry for HF cold starts
        try:
            out = client.text_generation(
                prompt,
                max_new_tokens=500,
                temperature=0.2,
                do_sample=False,
                repetition_penalty=1.05,
                return_full_text=False,
            )
            return out.strip()
        except Exception as e:
            last_err = e
    if FPNA_DEBUG:
        print("[fpna] HF summarizer error:", last_err)
    return None

# ────────────────────────────────────────────────────────────────────────────────
# Public router — always returns a string

def run_bi(question: str, api_key: Optional[str] = None) -> str:
    """
    Main entry: routes question to deterministic analyses or SQL.
    If the result is CSV, prepends an Executive Summary via HF (Mistral) when available.
    """
    q = (question or "").strip()
    ql = q.lower()

    # 1) Direct SQL
    if ql.startswith("select") or ql.startswith("with "):
        result = execute_sql(q)
        if _looks_like_csv(result):
            summary = _hf_summarize(q, result)
            if summary:
                return f"## Executive Summary\n\n{summary}\n\n---\n\n{result}"
        return result

    # 2) Deterministic routes
    if "conversion" in ql and "segment" in ql:
        try:
            result = analyze_conversion_trends()
            summary = _hf_summarize(q, result)
            if summary:
                return f"## Executive Summary\n\n{summary}\n\n---\n\n{result}"
            return result
        except Exception as e:
            return f"{retrieve_business_context('')}\n\n(Conversion analysis failed: {e})"

    if ("revenue" in ql) and ("business unit" in ql or "business_unit" in ql or "unit" in ql):
        try:
            result = revenue_by_business_unit()
            summary = _hf_summarize(q, result)
            if summary:
                return f"## Executive Summary\n\n{summary}\n\n---\n\n{result}"
            return result
        except Exception as e:
            return f"{retrieve_business_context('')}\n\n(Revenue analysis failed: {e})"

    # 3) Fallback: show schema + examples (no opaque LLM answers)
    return retrieve_business_context("")

# ────────────────────────────────────────────────────────────────────────────────
# Minimal wrapper so existing app code that expects `.run()` still works

class RouterAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key  # unused; kept for compatibility
    def run(self, q: str) -> str:
        return run_bi(q, self.api_key)

def create_agent(google_api_key: Optional[str] = None):
    """Return a simple router agent; no LangChain, no Gemini."""
    return RouterAgent(google_api_key)

# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("fpna version:", FPNA_VERSION)
    print(run_bi("conversion rate trends by merchant segment")[:400])
    print(run_bi("revenue by business unit")[:400])
    print(run_bi("SELECT 1 AS a, 2 AS b")[:200])
