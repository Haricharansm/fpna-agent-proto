# fpna_agent.py — Deterministic FP&A Agent (CSV → DuckDB) with strategic insights
# - Direct SQL over ./data via DuckDB
# - Canned analyses: conversion trends, revenue by business unit, feature/policy impact,
#   segment optimization, growth trend/forecast
# - No LLM dependency (safe for Streamlit Cloud free tier)
# - Back-compat helpers:
#     execute_business_query(query)         -> SQL runner
#     analyze_monthly_conversion_rates()    -> conversion trends helper
# - Primary entry: create_agent().run(question) -> str

from __future__ import annotations

import os
import glob
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

# DuckDB for SQL-on-CSVs
try:
    import duckdb  # type: ignore
    HAVE_DUCKDB = True
except Exception:
    duckdb = None
    HAVE_DUCKDB = False

DATA_DIR = os.environ.get("DATA_DIR", "data")

# ────────────────────────────────────────────────────────────────────────────────
# Column aliases & priorities

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
REVENUE_ALIASES = ["revenue"]

FEATURE_FLAG_ALIASES = ["feature_launch_period", "feature_live", "feature_active", "feature_flag"]
POLICY_FLAG_ALIASES  = ["policy_active", "policy_flag", "policy_change", "policy_period"]


def _priority_rank(name: str) -> int:
    try:
        return PREFERRED_TABLE_ORDER.index(name)
    except Exception:
        return 99


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("date", "day", "week_start", "month"):
        if col in df.columns:
            if col == "month" and df[col].dtype == "O":
                # leave YYYY-MM strings alone if already in that form
                try:
                    pd.to_datetime(df[col], errors="ignore")
                except Exception:
                    pass
            else:
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
        raise RuntimeError("DuckDB not installed. Add `duckdb` to requirements.txt and redeploy.")
    con = duckdb.connect()
    for name, df in tables.items():
        con.register(name, df)
    return con


# ────────────────────────────────────────────────────────────────────────────────
# Summary helpers

def _find_time_col(df: pd.DataFrame) -> Optional[str]:
    return next((c for c in TIME_ALIASES if c in df.columns), None)


def _coverage_from_col(series: pd.Series):
    s = pd.to_datetime(series, errors="coerce").dropna()
    if s.empty:
        return None
    return s.min(), s.max()


def _global_coverage(tables: Dict[str, pd.DataFrame]):
    wins = []
    for df in tables.values():
        tcol = _find_time_col(df)
        if tcol:
            cov = _coverage_from_col(df[tcol])
            if cov:
                wins.append(cov)
    if not wins:
        return None
    return min(w[0] for w in wins), max(w[1] for w in wins)


def _nice_num(n: float | int) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return f"{n:,}"


def schema_table_md(tables: Dict[str, pd.DataFrame]) -> str:
    lines = [
        "| Table | Rows | Time | Coverage | Key columns |",
        "|---|---:|---|---|---|",
    ]
    KEY_PREF = [
        "merchant_segment",
        "business_unit",
        "acquisition_channel",
        "revenue",
        "sessions",
        "conversions",
        "leads",
        "transactions",
    ]
    for name, df in tables.items():
        rows = _nice_num(len(df))
        tcol = _find_time_col(df) or "—"
        cov = "—"
        if tcol != "—":
            c = _coverage_from_col(df[tcol])
            if c:
                cov = f"{str(c[0]).split(' ')[0]} → {str(c[1]).split(' ')[0]}"
        key_cols = [c for c in KEY_PREF if c in df.columns][:4]
        lines.append(f"| `{name}` | {rows} | {tcol} | {cov} | {', '.join(key_cols) or '—'} |")
    return "\n".join(lines)


def executive_overview_md(tables: Dict[str, pd.DataFrame]) -> str:
    total_rows = sum(len(df) for df in tables.values())
    cov = _global_coverage(tables)
    period = f"{cov[0].date()} → {cov[1].date()}" if cov else "n/a"

    sessions = leads = 0
    revenue = 0.0
    for df in tables.values():
        for c in VISITS_ALIASES:
            if c in df.columns:
                sessions += pd.to_numeric(df[c], errors="coerce").fillna(0).sum()
                break
        for c in LEADS_ALIASES:
            if c in df.columns:
                leads += pd.to_numeric(df[c], errors="coerce").fillna(0).sum()
                break
        if "revenue" in df.columns:
            revenue += pd.to_numeric(df["revenue"], errors="coerce").fillna(0).sum()

    conv_rate = (leads / sessions) if sessions else None
    conv_txt = f"{conv_rate:.2%}" if conv_rate is not None else "n/a"

    top_segment_txt = "n/a"
    if "monthly_summary" in tables and {"merchant_segment", "revenue"}.issubset(
        tables["monthly_summary"].columns
    ):
        seg_rev = (
            tables["monthly_summary"]
            .groupby("merchant_segment")["revenue"]
            .sum()
            .sort_values(ascending=False)
        )
        if not seg_rev.empty:
            top_segment_txt = f"{seg_rev.index[0]} (${seg_rev.iloc[0]:,.0f})"

    return f"""## Executive Summary
- **Period:** {period}
- **Tables / Rows:** {len(tables)} / {_nice_num(total_rows)}
- **Total Sessions / Leads:** {_nice_num(sessions)} / {_nice_num(leads)} → **Conv. {conv_txt}**
- **Total Revenue:** ${revenue:,.0f}
- **Top Segment (revenue):** {top_segment_txt}
"""


def retrieve_business_context(_: str) -> str:
    tbls = load_tables()
    header = (
        "**Business Context (Demo)**\n"
        "- CSVs loaded from `./data` (e.g., `segment_analysis`, `channel_performance`, "
        "`daily_summary`, `monthly_summary`, `transaction_data`).\n"
        "- Typical asks: conversion by segment, revenue by unit, transaction trends, "
        "channel mix, **feature/policy impact**, **segment optimization**, **growth trend/forecast**.\n\n"
    )
    overview = executive_overview_md(tbls)
    schema = schema_table_md(tbls)
    return f"{header}{overview}\n**Available tables**\n\n{schema}\n"


def execute_sql(sql_or_question: str) -> str:
    q = (sql_or_question or "").strip()
    if not (q.lower().startswith("select") or q.lower().startswith("with ")):
        return "Provide a SQL SELECT.\n\n" + retrieve_business_context("")
    try:
        con = register_duckdb(load_tables())
        df = con.execute(q).df()
        return "Query returned no rows." if df.empty else df.to_csv(index=False)
    except Exception as e:
        return f"SQL error: {e}\n\n" + retrieve_business_context("")


# Back-compat export for older codepaths
def execute_business_query(query: str) -> str:
    return execute_sql(query)

# ────────────────────────────────────────────────────────────────────────────────
# Deterministic analyses

def _pick_conversion_source(tables: Dict[str, pd.DataFrame]):
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


def analyze_conversion_trends() -> str:
    tables = load_tables()
    table_name, mapping, score = _pick_conversion_source(tables)
    if not table_name or score < 4:
        raise ValueError("Need time+segment+visits+leads in one table.")
    df = tables[table_name].copy()
    t, seg, vis, led = mapping["time"], mapping["segment"], mapping["visits"], mapping["leads"]

    # Normalize time → month
    if t == "month" and df[t].dtype == "O":
        try:
            df["month"] = pd.to_datetime(df[t], errors="coerce").dt.to_period("M").astype(str)
        except Exception:
            df["month"] = df[t].astype(str)
    else:
        if not pd.api.types.is_datetime64_any_dtype(df[t]):
            df[t] = pd.to_datetime(df[t], errors="coerce")
        df["month"] = df[t].dt.to_period("M").astype(str)

    g = df.groupby(["month", seg]).agg(visits=(vis, "sum"), leads=(led, "sum")).reset_index()
    g["conversion_rate"] = (g["leads"] / g["visits"]).replace([np.inf, -np.inf], 0).fillna(0.0)

    seg_avg = g.groupby(seg)["conversion_rate"].mean().sort_values(ascending=False)
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
    g = g.rename(columns={seg: "merchant_segment"})
    g = g[["month", "merchant_segment", "visits", "leads", "conversion_rate"]].sort_values(["month", "merchant_segment"])
    return summary + "\n---\n" + g.to_csv(index=False)


def analyze_monthly_conversion_rates() -> str:
    # Back-compat name
    return analyze_conversion_trends()


def revenue_by_business_unit() -> str:
    """Return revenue by business unit (monthly table preferred)."""
    tables = load_tables()

    # Preferred: monthly_summary with month + business_unit + revenue
    if "monthly_summary" in tables and {"business_unit", "revenue"}.issubset(tables["monthly_summary"].columns):
        df = tables["monthly_summary"].copy()
        if "month" in df.columns:
            if df["month"].dtype == "O":
                try:
                    df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").astype(str)
                except Exception:
                    df["month"] = df["month"].astype(str)
            else:
                df["month"] = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M").astype(str)
        else:
            t = _find_time_col(df)
            if t and not pd.api.types.is_datetime64_any_dtype(df[t]):
                df[t] = pd.to_datetime(df[t], errors="coerce")
            df["month"] = (df[t] if t else pd.to_datetime("today")).dt.to_period("M").astype(str)
        grouped = (
            df.groupby(["month", "business_unit"])["revenue"]
              .sum()
              .reset_index()
              .sort_values(["month", "business_unit"])
        )

    else:
        # Fallback: find any table with time + business_unit + revenue
        candidate = None
        for name, d in tables.items():
            if {"business_unit", "revenue"}.issubset(d.columns):
                candidate = name
                break
        if not candidate:
            raise ValueError("No table found with `business_unit` and `revenue`.")
        df = tables[candidate].copy()
        t = _find_time_col(df)
        if t is None:
            raise ValueError(f"`{candidate}` lacks a time column (one of: {', '.join(TIME_ALIASES)}).")
        if not pd.api.types.is_datetime64_any_dtype(df[t]):
            df[t] = pd.to_datetime(df[t], errors="coerce")
        df["month"] = df[t].dt.to_period("M").astype(str)
        grouped = (
            df.groupby(["month", "business_unit"])["revenue"]
              .sum()
              .reset_index()
              .sort_values(["month", "business_unit"])
        )

    # overall totals & rankings
    totals = grouped.groupby("business_unit")["revenue"].sum().sort_values(ascending=False)
    top_bu, top_rev = (totals.index[0], totals.iloc[0]) if not totals.empty else ("n/a", 0.0)
    monthly_total = grouped.groupby("month")["revenue"].sum().sort_index()
    trend = ""
    if len(monthly_total) >= 2 and monthly_total.iloc[0] != 0:
        trend = f"{((monthly_total.iloc[-1] - monthly_total.iloc[0]) / abs(monthly_total.iloc[0])):.1%}"
    period = f"{grouped['month'].min()} → {grouped['month'].max()}" if not grouped.empty else "n/a"

    summary = f"""## Executive Summary
- **Analysis:** Revenue performance across business units
- **Period:** {period}
- **Top business unit:** {top_bu} (${top_rev:,.0f})
- **Total revenue:** ${totals.sum():,.0f}  |  **Trend (first→last total):** {trend or 'n/a'}
"""
    # Optional per-month share of total
    mtot = grouped.groupby("month")["revenue"].transform("sum").replace(0, np.nan)
    grouped["share_of_month"] = (grouped["revenue"] / mtot).round(4)

    return summary + "\n---\n" + grouped.to_csv(index=False)


def _pick_flag_table(tables: Dict[str, pd.DataFrame], flag_aliases):
    best_name, best_map, best_score = None, None, -1
    for name, df in tables.items():
        mapping = {
            "time": next((c for c in TIME_ALIASES if c in df.columns), None),
            "sess": next((c for c in VISITS_ALIASES if c in df.columns), None),
            "conv": next((c for c in LEADS_ALIASES if c in df.columns), None),
            "rev": next((c for c in REVENUE_ALIASES if c in df.columns), None),
            "flag": next((c for c in flag_aliases if c in df.columns), None),
            "seg":  next((c for c in SEGMENT_ALIASES if c in df.columns), None),
        }
        score = sum(v is not None for v in mapping.values() if v != "rev") + (1 if mapping["rev"] else 0)
        if score > best_score or (score == best_score and _priority_rank(name) < _priority_rank(best_name or "")):
            best_name, best_map, best_score = name, mapping, score
    return best_name, best_map, best_score


def _before_after_summary(df: pd.DataFrame, sess: str, conv: str, rev: str | None, flag: str):
    before = df[df[flag] == 0]
    after  = df[df[flag] == 1]

    def agg(d: pd.DataFrame):
        s = pd.to_numeric(d[sess], errors="coerce").fillna(0).sum()
        c = pd.to_numeric(d[conv], errors="coerce").fillna(0).sum()
        r = pd.to_numeric(d[rev], errors="coerce").fillna(0).sum() if rev and rev in d.columns else np.nan
        return s, c, r

    s0, c0, r0 = agg(before); s1, c1, r1 = agg(after)
    cr0 = (c0/s0) if s0 else 0.0
    cr1 = (c1/s1) if s1 else 0.0
    delta_cr = cr1 - cr0
    delta_conv = c1 - c0
    rev_line = f"- **Revenue:** ${r1:,.0f} (after) vs ${r0:,.0f} (before)" if (rev and not np.isnan(r0)) else ""
    md = f"""## Executive Summary
- **Period split:** before (flag=0) vs after (flag=1)
- **Sessions:** {_nice_num(s1)} (after) vs {_nice_num(s0)} (before)
- **Conversions:** {_nice_num(c1)} vs {_nice_num(c0)}  →  **Δ {delta_conv:+,}**
- **Conversion rate:** {cr1:.2%} vs {cr0:.2%}  →  **Δ {delta_cr:+.2%}**
{rev_line}
"""
    tbl = pd.DataFrame({
        "metric": ["sessions","conversions","conversion_rate"] + (["revenue"] if (rev and not np.isnan(r0)) else []),
        "before": [s0, c0, cr0] + ([r0] if (rev and not np.isnan(r0)) else []),
        "after":  [s1, c1, cr1] + ([r1] if (rev and not np.isnan(r0)) else []),
    })
    tbl["delta_abs"] = tbl["after"] - tbl["before"]
    tbl["delta_pct"] = np.where(tbl["before"].replace(0,np.nan).notna(),
                                (tbl["after"]-tbl["before"])/tbl["before"], np.nan)
    return md, tbl


def feature_launch_impact() -> str:
    tables = load_tables()
    name, m, score = _pick_flag_table(tables, FEATURE_FLAG_ALIASES)
    if not name or score < 5:
        raise ValueError("Need time+sessions+conversions+flag (revenue optional) for feature analysis.")
    df = tables[name].copy()
    t, sess, conv, rev, flag, seg = m["time"], m["sess"], m["conv"], m["rev"], m["flag"], m["seg"]

    if not pd.api.types.is_datetime64_any_dtype(df[t]):
        df[t] = pd.to_datetime(df[t], errors="coerce")
    df = df.dropna(subset=[t, flag]).copy()
    df[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(int)

    md, tbl = _before_after_summary(df, sess, conv, rev, flag)

    extra = ""
    if seg:
        byseg = (df
                 .groupby([flag, seg])
                 .agg(sessions=(sess,"sum"), conversions=(conv,"sum"))
                 .reset_index())
        pivot = byseg.pivot(index=seg, columns=flag, values=["sessions","conversions"]).fillna(0)
        pivot.columns = [f"{a}_{'after' if b==1 else 'before'}" for a,b in pivot.columns]
        pivot["conversion_rate_before"] = pivot["conversions_before"]/pivot["sessions_before"].replace(0,np.nan)
        pivot["conversion_rate_after"]  = pivot["conversions_after"] /pivot["sessions_after"] .replace(0,np.nan)
        pivot["delta_cr"] = pivot["conversion_rate_after"] - pivot["conversion_rate_before"]
        pivot = pivot.sort_values("delta_cr", ascending=False)
        extra = "\n### By segment (Δ conversion rate)\n" + pivot.reset_index()[[seg,"conversion_rate_before","conversion_rate_after","delta_cr"]].to_csv(index=False)

    return md + "\n---\n" + tbl.to_csv(index=False) + (("\n---\n" + extra) if extra else "")


def policy_change_evaluation() -> str:
    tables = load_tables()
    name, m, score = _pick_flag_table(tables, POLICY_FLAG_ALIASES)
    if not name or score < 5:
        raise ValueError("Need time+sessions+conversions+flag (revenue optional) for policy analysis.")
    df = tables[name].copy()
    t, sess, conv, rev, flag, _seg = m["time"], m["sess"], m["conv"], m["rev"], m["flag"], m["seg"]

    if not pd.api.types.is_datetime64_any_dtype(df[t]):
        df[t] = pd.to_datetime(df[t], errors="coerce")
    df = df.dropna(subset=[t, flag]).copy()
    df[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(int)

    md, tbl = _before_after_summary(df, sess, conv, rev, flag)
    return md + "\n---\n" + tbl.to_csv(index=False)


def segment_optimization_opportunities() -> str:
    tables = load_tables()
    name = "segment_analysis" if "segment_analysis" in tables else next(iter(tables.keys()))
    df = tables[name].copy()

    seg = next((c for c in SEGMENT_ALIASES if c in df.columns), None)
    vis = next((c for c in VISITS_ALIASES if c in df.columns), None)
    led = next((c for c in LEADS_ALIASES if c in df.columns), None)
    rev = "revenue" if "revenue" in df.columns else None
    if not all([seg, vis, led]):
        raise ValueError("Need a table with segment + visits + leads (revenue optional).")

    g = df.groupby(seg).agg(visits=(vis,"sum"), leads=(led,"sum")).reset_index()
    g["cr"] = (g["leads"]/g["visits"]).replace([np.inf,-np.inf],0).fillna(0.0)
    top_cr = g["cr"].max()
    g["uplift_to_top"] = (top_cr - g["cr"]).clip(lower=0)
    g["potential_extra_conversions"] = (g["uplift_to_top"] * g["visits"]).round(0)

    md = f"""## Executive Summary
- **Analysis:** Segment comparison & optimization
- **Top conversion rate:** {top_cr:.2%}
- **Potential opportunities:** move low performers toward top CR; see table for incremental conversions.
"""
    out_cols = [seg, "visits", "leads", "cr", "uplift_to_top", "potential_extra_conversions"]
    if rev and rev in df.columns:
        g_rev = df.groupby(seg)[rev].sum().reset_index()
        g = g.merge(g_rev, on=seg, how="left")
        out_cols.append(rev)
    g = g.sort_values("uplift_to_top", ascending=False)
    return md + "\n---\n" + g[out_cols].to_csv(index=False)


def growth_trend_forecast(periods_ahead: int = 4) -> str:
    tables = load_tables()
    name = "daily_summary" if "daily_summary" in tables else ("monthly_summary" if "monthly_summary" in tables else next(iter(tables.keys())))
    df = tables[name].copy()
    t = _find_time_col(df)
    metric = "revenue" if "revenue" in df.columns else (
        next((c for c in ["conversions","leads","sessions","total_visits","visits"] if c in df.columns), None)
    )
    if not (t and metric):
        raise ValueError("Need a table with a time column and at least one metric (revenue/conversions/sessions).")
    if not pd.api.types.is_datetime64_any_dtype(df[t]):
        df[t] = pd.to_datetime(df[t], errors="coerce")

    g = df.groupby(df[t].dt.to_period("W" if name=="daily_summary" else "M"))[metric].sum().reset_index()
    g["period"] = g[t].astype(str)
    y = g[metric].values
    x = np.arange(len(y))
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        future_x = np.arange(len(y), len(y)+periods_ahead)
        forecast = intercept + slope*future_x
        f_df = pd.DataFrame({"period": [f"T+{i+1}" for i in range(periods_ahead)], metric: forecast})
        md = f"""## Executive Summary
- **Analysis:** Growth trend & simple forecast
- **Source:** `{name}`  • **Metric:** `{metric}`
- **Trend:** {'↑' if slope>0 else '↓' if slope<0 else '→'}  (slope={slope:,.2f})
"""
        hist = g[["period", metric]]
        out = pd.concat([hist, f_df], ignore_index=True)
        return md + "\n---\n" + out.to_csv(index=False)
    else:
        return "Not enough data points to compute a trend."


# ────────────────────────────────────────────────────────────────────────────────
# Agent router

def _fallback_answer(question: str) -> str:
    return retrieve_business_context("") + "\n\nTip: start your query with SELECT to run SQL directly."


def _route(question: str) -> str:
    q = (question or "").strip()
    ql = q.lower()

    # 1) Direct SQL
    if ql.startswith("select") or ql.startswith("with "):
        return execute_sql(q)

    # 2) Deterministic analyses
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

    if ("feature" in ql and ("impact" in ql or "launch" in ql)):
        try:
            return feature_launch_impact()
        except Exception as e:
            return f"{retrieve_business_context('')}\n\n(Feature impact failed: {e})"

    if ("policy" in ql and ("change" in ql or "impact" in ql or "evaluation" in ql)):
        try:
            return policy_change_evaluation()
        except Exception as e:
            return f"{retrieve_business_context('')}\n\n(Policy analysis failed: {e})"

    if ("segment" in ql and ("compare" in ql or "optimization" in ql or "opportunity" in ql)):
        try:
            return segment_optimization_opportunities()
        except Exception as e:
            return f"{retrieve_business_context('')}\n\n(Segment analysis failed: {e})"

    if ("growth" in ql and ("trend" in ql or "forecast" in ql)):
        try:
            return growth_trend_forecast()
        except Exception as e:
            return f"{retrieve_business_context('')}\n\n(Growth analysis failed: {e})"

    # 3) Default: context & tips
    return _fallback_answer(q)


class RouterAgent:
    """Minimal .run() interface used by app.py fallbacks."""
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, q: str) -> str:
        return _route(q)


def create_agent(*_args, **_kwargs) -> RouterAgent:
    """Factory used by app.py when fpna.py is unavailable."""
    return RouterAgent()


# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke tests
    print(RouterAgent().run("Revenue performance across business units"))
    print(RouterAgent().run("conversion rate trends by merchant segment"))
    print(RouterAgent().run("Impact of recent feature launches"))
    print(RouterAgent().run("Policy change performance evaluation"))
    print(RouterAgent().run("Segment comparison and optimization opportunities"))
    print(RouterAgent().run("Growth trend analysis and forecasting"))
    print(RouterAgent().run("SELECT 1 AS a, 2 AS b"))
