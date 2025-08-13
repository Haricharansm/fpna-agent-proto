# fpna.py — FP&A core with executive summaries + compact schema table
from __future__ import annotations

import os
import glob
from typing import Dict, Optional, Tuple
from textwrap import shorten

import pandas as pd
import numpy as np

# DuckDB (for SQL over CSVs)
try:
    import duckdb  # type: ignore
    HAVE_DUCKDB = True
except Exception:
    duckdb = None
    HAVE_DUCKDB = False

# Optional direct Gemini SDK (not required; still supported)
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

# ------------------------------ Summary helpers ------------------------------

def _find_time_col(df: pd.DataFrame) -> Optional[str]:
    return next((c for c in TIME_ALIASES if c in df.columns), None)

def _coverage_from_col(series: pd.Series) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    s = pd.to_datetime(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None
    return (s.min(), s.max())

def _global_coverage(tables: Dict[str, pd.DataFrame]) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    windows = []
    for df in tables.values():
        tcol = _find_time_col(df)
        if tcol:
            cov = _coverage_from_col(df[tcol])
            if cov:
                windows.append(cov)
    if not windows:
        return None
    start = min(w[0] for w in windows)
    end   = max(w[1] for w in windows)
    return start, end

def _nice_num(n: float | int) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return f"{n:,}"

def schema_table_md(tables: Dict[str, pd.DataFrame]) -> str:
    """
    Compact markdown table of schema: table | rows | time field | coverage | key columns
    """
    lines = [
        "| Table | Rows | Time | Coverage | Key columns |",
        "|---|---:|---|---|---|",
    ]
    KEY_PREF = [
        "merchant_segment", "business_unit", "acquisition_channel",
        "revenue", "sessions", "conversions", "leads", "transactions"
    ]
    for name, df in tables.items():
        rows = _nice_num(len(df))
        tcol = _find_time_col(df) or "—"
        cov  = "—"
        if tcol != "—":
            cov_tuple = _coverage_from_col(df[tcol])
            if cov_tuple:
                cov = f"{str(cov_tuple[0]).split(' ')[0]} → {str(cov_tuple[1]).split(' ')[0]}"
        key_cols = [c for c in KEY_PREF if c in df.columns][:4]
        lines.append(f"| `{name}` | {rows} | {tcol} | {cov} | {', '.join(key_cols) or '—'} |")
    return "\n".join(lines)

def executive_overview_md(tables: Dict[str, pd.DataFrame]) -> str:
    """
    Build a concise executive overview (period, totals, top-line stats).
    """
    total_rows = sum(len(df) for df in tables.values())
    cov = _global_coverage(tables)
    period = f"{cov[0].date()} → {cov[1].date()}" if cov else "n/a"

    # Aggregate best-effort topline metrics
    sessions = 0
    leads = 0
    revenue = 0.0

    for df in tables.values():
        for c in ["sessions", "total_visits", "visits", "impressions"]:
            if c in df.columns:
                sessions += pd.to_numeric(df[c], errors="coerce").fillna(0).sum()
                break
        for c in ["leads", "conversions", "converted", "signups"]:
            if c in df.columns:
                leads += pd.to_numeric(df[c], errors="coerce").fillna(0).sum()
                break
        if "revenue" in df.columns:
            revenue += pd.to_numeric(df["revenue"], errors="coerce").fillna(0).sum()

    conv_rate = (leads / sessions) if sessions else None
    conv_txt  = f"{conv_rate:.2%}" if conv_rate is not None else "n/a"

    # Optional: top segment by revenue if monthly_summary present
    top_segment_txt = "n/a"
    if "monthly_summary" in tables and {"merchant_segment", "revenue"}.issubset(tables["monthly_summary"].columns):
        seg_rev = (tables["monthly_summary"]
                    .groupby("merchant_segment")["revenue"]
                    .sum()
                    .sort_values(ascending=False))
        if not seg_rev.empty:
            top_segment_txt = f"{seg_rev.index[0]} (${seg_rev.iloc[0]:,.0f})"

    return f"""## Executive Summary
- **Period:** {period}
- **Tables / Rows:** {len(tables)} / {_nice_num(total_rows)}
- **Total Sessions / Leads:** {_nice_num(sessions)} / {_nice_num(leads)}  → **Conv. {conv_txt}**
- **Total Revenue:** ${revenue:,.0f}
- **Top Segment (revenue):** {top_segment_txt}
"""

# ────────────────────────────────────────────────────────────────────────────────
# Tools / helpers exposed to the app

def retrieve_business_context(_: str) -> str:
    tbls = load_tables()
    header = (
        "## Business Intelligence Report\n\n"
        "**Business Context (Demo)**\n"
        "- CSVs loaded from `./data` (e.g., `segment_analysis`, `channel_performance`, "
        "`daily_summary`, `monthly_summary`, `transaction_data`).\n"
        "- Typical asks: conversion by segment, revenue by unit, transaction trends, "
        "channel mix, feature/policy impact.\n\n"
    )
    overview = executive_overview_md(tbls)
    schema   = schema_table_md(tbls)
    samples  = (
        "\n**Suggested queries**\n"
        "```sql\n"
        "-- Conversion rate trends by segment (adjust table/columns to your schema)\n"
        "SELECT COALESCE(week_start, date, month) AS period, merchant_segment,\n"
        "       SUM(leads)*1.0/NULLIF(SUM(total_visits),0) AS conversion_rate\n"
        "FROM segment_analysis\n"
        "GROUP BY 1,2\n"
        "ORDER BY 1,2;\n\n"
        "-- Revenue by business unit\n"
        "SELECT COALESCE(week_start, date, month) AS period, business_unit, SUM(revenue) AS revenue\n"
        "FROM monthly_summary\n"
        "GROUP BY 1,2\n"
        "ORDER BY 1,2;\n"
        "```\n"
    )
    return f"{header}{overview}\n**Available tables**\n\n{schema}\n{samples}"

def execute_sql(sql_or_question: str) -> str:
    q = (sql_or_question or "").strip()
    is_sql = q.lower().startswith("select") or q.lower().startswith("with ")
    if not is_sql:
        return "Provide a SQL SELECT.\n\n" + retrieve_business_context("")  # hints + schema
    try:
        con = register_duckdb(load_tables())
        df = con.execute(q).df()
        return "Query returned no rows." if df.empty else df.to_csv(index=False)
    except Exception as e:
        return f"SQL error: {e}\n\n" + retrieve_business_context("")

# ------------------------------ Deterministic analyses ------------------------------

def analyze_conversion_trends() -> str:
    """
    Compute monthly conversion rate by merchant segment from the best available table.
    Returns: Executive Summary + CSV: month,merchant_segment,visits,leads,conversion_rate
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

    # Summary metrics
    seg_avg = g.groupby(seg_col)["conversion_rate"].mean().sort_values(ascending=False)
    monthly = g.groupby("month")["conversion_rate"].mean()
    top_seg, top_val = (seg_avg.index[0], seg_avg.iloc[0]) if not seg_avg.empty else ("n/a", 0.0)
    bot_seg, bot_val = (seg_avg.index[-1], seg_avg.iloc[-1]) if len(seg_avg) > 1 else ("n/a", 0.0)
    trend = ""
    if len(monthly) >= 2 and monthly.iloc[0] != 0:
        trend = f"{((monthly.iloc[-1] - monthly.iloc[0]) / abs(monthly.iloc[0])):.1%}"
    period = f"{g['month'].min()} → {g['month'].max()}" if not g.empty else "n/a"

    summary = f"""## Executive Summary
- **Analysis:** Conversion rate trends by segment (source `{table_name}`)
- **Period:** {period}
- **Overall avg. conversion:** {monthly.mean():.2%}  |  **Trend (first→last):** {trend or 'n/a'}
- **Top / Bottom segments:** {top_seg} ({top_val:.2%}) / {bot_seg} ({bot_val:.2%})
- **Volume:** Visits {_nice_num(g['visits'].sum())} • Leads {_nice_num(g['leads'].sum())}
"""

    g = g.rename(columns={seg_col: "merchant_segment"})
    g = g[["month", "merchant_segment", "visits", "leads", "conversion_rate"]].sort_values(["month", "merchant_segment"])
    csv_part = g.to_csv(index=False)
    return summary + "\n---\n" + csv_part

def revenue_by_business_unit() -> str:
    """
    Return Executive Summary + CSV: month,business_unit,revenue (from monthly_summary).
    """
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

    total_rev = out["revenue"].sum()
    by_bu = out.groupby("business_unit")["revenue"].sum().sort_values(ascending=False)
    top_bu = f"{by_bu.index[0]} (${by_bu.iloc[0]:,.0f})" if not by_bu.empty else "n/a"
    period = f"{out['month'].min()} → {out['month'].max()}" if not out.empty else "n/a"

    summary = f"""## Executive Summary
- **Analysis:** Revenue by business unit
- **Period:** {period}
- **Total Revenue:** ${total_rev:,.0f}
- **Top Business Unit:** {top_bu}
"""
    return summary + "\n---\n" + out.to_csv(index=False)

# ────────────────────────────────────────────────────────────────────────────────
# Direct Gemini fallback (not required)

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

    # 3) default: show context (or LLM summary if configured)
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
    print(run_bi("conversion rate trends by merchant segment"))
    print(run_bi("revenue by business unit"))
    print(run_bi("SELECT 1 AS a, 2 AS b"))
