# app.py — FP&A BI app (DuckDB/Pandas + optional Mistral narrative)
import os
import sys
from io import StringIO

import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# Paths & env
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

os.environ.setdefault("DATA_DIR", os.path.join(current_dir, "data"))

# Load optional Mistral secrets into env for fpna.py
for key in ("MISTRAL_API_KEY", "MISTRAL_MODEL"):
    try:
        if key in st.secrets and st.secrets[key]:
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

# -------------------------------------------------------------------
# Prefer new core (fpna.run_bi); fallback to fpna_agent Router
HAVE_RUN_BI = False
CORE_OK = False
err_msg = ""

try:
    from fpna import run_bi  # preferred new API (supports embellish + mistral)
    HAVE_RUN_BI = True
    CORE_OK = True
except Exception as e1:
    try:
        # Older module; create_agent returns a Router-like object with .run()
        from fpna_agent import create_agent
        CORE_OK = True
    except Exception as e2:
        CORE_OK = False
        err_msg = f"{e1}  |  {e2}"

st.set_page_config(
    page_title="FP&A AI Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Header
st.markdown("""
<div style='text-align:center;padding:1rem 0;'>
  <h1 style='color:#1e3a8a;margin:0;'>📊 Financial Planning & Analysis</h1>
  <h2 style='color:#64748b;margin:0;font-weight:400;'>Business Intelligence Agent (DuckDB + optional Mistral)</h2>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

if not CORE_OK:
    st.error(f"❌ Cannot import FP&A core: {err_msg}")
    st.info("📁 Ensure `fpna.py` or `fpna_agent.py` sits next to `app.py` and dependencies are installed.")
    st.stop()

# -------------------------------------------------------------------
# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")

    mistral_api_key = st.text_input(
        "Mistral API key (optional for 🪄 narrative)",
        value=os.getenv("MISTRAL_API_KEY", ""),
        type="password",
        help="If provided, the app will add an executive LLM narrative using Mistral."
    )
    use_llm_narrative = st.checkbox("🪄 Add LLM narrative (Mistral)", value=bool(mistral_api_key))

    if use_llm_narrative:
        if mistral_api_key:
            st.success("Mistral narrative enabled")
        else:
            st.warning("Provide a Mistral API key to enable narrative")

    st.subheader("Data")
    st.caption(f"Reading CSVs from: `{os.environ['DATA_DIR']}`")
    st.info("📁 All `*.csv` in this folder are auto-registered as DuckDB tables.")

    st.divider()
    st.subheader("Business Context")
    st.markdown("""
**Period**: recent weeks/months (as data provides)  
**Segments**: Retail / Wholesale / New / etc.  
**Metrics**: Revenue, Conversion, Volume  
**Engine**: Deterministic DuckDB/Pandas; optional Mistral narrative
""")

# -------------------------------------------------------------------
# Query UI
st.markdown("### Business Intelligence Query")
c1, c2 = st.columns([4, 1])
with c1:
    user_question = st.text_input(
        "Enter your business question:",
        placeholder="Conversion rate trends by merchant segment",
        label_visibility="collapsed",
    )
with c2:
    st.write("")
    go = st.button("🔍 Analyze", use_container_width=True, type="primary")

with st.expander("💼 Sample Business Questions"):
    left, right = st.columns(2)
    with left:
        st.markdown("""
**Performance Analysis:**
• Conversion rate trends by merchant segment  
• Revenue performance across business units  
• Transaction volume analysis by period  
• Customer acquisition metrics by channel
""")
    with right:
        st.markdown("""
**Strategic Insights:**
• Impact of recent feature launches  
• Policy change performance evaluation  
• Segment comparison & optimization opportunities  
• Growth trend analysis and forecasting
""")

# -------------------------------------------------------------------
# Helpers

def _looks_like_csv(text: str) -> bool:
    if not text or not text.strip():
        return False
    head = text.strip().splitlines()[0]
    return ("," in head) and (len(head.split(",")) >= 2)

def _line_chart_if_time(df: pd.DataFrame):
    for time_col in ["period", "week_start", "date", "day", "month"]:
        if time_col in df.columns:
            try:
                tmp = df.copy()
                tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
                tmp = tmp.dropna(subset=[time_col])
                num_cols = tmp.select_dtypes("number").columns.tolist()
                if num_cols:
                    st.line_chart(data=tmp.set_index(time_col)[num_cols])
                return
            except Exception:
                return

def _render_result(result: str):
    """
    Supports:
      - '## Executive Summary ...\\n---\\n<CSV>\\n---\\n### 🪄 LLM Narrative ...'
      - a plain CSV
      - markdown/text only
    """
    txt = result.strip()
    parts = txt.split("\n---\n")

    if parts[0].startswith("## ") or parts[0].startswith("# "):
        # section 1: markdown summary
        st.markdown(parts[0])

        # remaining sections: CSV tables or markdown (e.g., LLM narrative)
        for sec in parts[1:]:
            if _looks_like_csv(sec):
                df = pd.read_csv(StringIO(sec))
                st.dataframe(df, use_container_width=True)
                _line_chart_if_time(df)
            else:
                st.markdown(sec)
        return

    # Plain CSV case
    if _looks_like_csv(txt):
        df = pd.read_csv(StringIO(txt))
        st.dataframe(df, use_container_width=True)
        _line_chart_if_time(df)
        return

    # Fallback: raw markdown/text
    st.markdown(txt)

# -------------------------------------------------------------------
# Router

def _run_query(question: str) -> str:
    q = (question or "").strip()

    if HAVE_RUN_BI:
        # New core path: fpna.run_bi handles deterministic + optional Mistral narrative
        api_key = mistral_api_key or None
        return run_bi(q, api_key=api_key, embellish=bool(api_key and use_llm_narrative))

    # Fallback: fpna_agent RouterAgent (no LLM embellishment)
    agent = create_agent()  # deterministic routes only
    return agent.run(q)

# -------------------------------------------------------------------
# Execute

if go:
    if not user_question.strip():
        st.warning("Please enter a business question.")
    else:
        with st.spinner("Analyzing…"):
            try:
                result = _run_query(user_question)

                st.success("Analysis Complete")
                st.markdown("---")
                st.markdown("## 📈 Business Intelligence Report")

                _render_result(result)

                st.markdown("---")
                with st.expander("📊 Analysis Details"):
                    st.markdown(f"""
**Data Sources**: CSVs in `{os.environ['DATA_DIR']}`  
**Engine**: DuckDB/Pandas (deterministic)  
**LLM Narrative**: {'Enabled (Mistral)' if (mistral_api_key and use_llm_narrative) else 'Disabled'}  
**Tip**: You can also run SQL directly (start your query with `SELECT`).
""")
            except Exception as e:
                st.error("Analysis could not be completed.")
                with st.expander("🔧 Technical Details", expanded=False):
                    st.code(str(e))

# -------------------------------------------------------------------
# Footer + light styling
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**🎯 Capabilities**  \nDeterministic analysis  \nMulti-segment insights  \nTrend identification")
with c2:
    st.markdown("**📊 Data Coverage**  \nRevenue metrics  \nConversion analytics  \nPerformance indicators")
with c3:
    st.markdown("**🪄 LLM (Optional)**  \nExecutive narrative  \nMistral API")

st.markdown("""
<style>
  .stSelectbox > div > div { background-color: #f8fafc; }
  .stTextInput > div > div > input { background-color: #f8fafc; }
  .stButton > button {
    background-color: #1e40af; color: white; border-radius: 8px; border: none;
    padding: .5rem 1rem; font-weight: 500;
  }
  .stButton > button:hover { background-color: #1d4ed8; }
</style>
""", unsafe_allow_html=True)
