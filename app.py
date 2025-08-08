# app.py — robust imports + fast-path router
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
os.environ.setdefault("GEMINI_MODEL", "gemini-1.5-flash")
os.environ.setdefault("AGENT_MAX_STEPS", "12")

# -------------------------------------------------------------------
# Try fpna.py first (has run_bi); fallback to fpna_agent.py
HAVE_RUN_BI = False
FPNA_CORE_OK = False
err_msg = ""

try:
    from fpna import run_bi  # preferred
    HAVE_RUN_BI = True
    FPNA_CORE_OK = True
except Exception as e1:
    # Fallback: older module name / API
    try:
        from fpna_agent import create_agent  # noqa: F401
        # Optional helpers if available
        try:
            from fpna_agent import execute_business_query as _exec_sql  # noqa: F401
        except Exception:
            _exec_sql = None
        try:
            from fpna_agent import analyze_monthly_conversion_rates as _conv_helper  # noqa: F401
        except Exception:
            _conv_helper = None
        FPNA_CORE_OK = True
        err_msg = ""
    except Exception as e2:
        FPNA_CORE_OK = False
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
  <h2 style='color:#64748b;margin:0;font-weight:400;'>AI-Powered Business Intelligence Agent</h2>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

if not FPNA_CORE_OK:
    st.error(f"❌ Cannot import FP&A core: {err_msg}")
    st.info("📁 Make sure either `fpna.py` or `fpna_agent.py` is next to `app.py` and dependencies are installed.")
    st.stop()

# -------------------------------------------------------------------
# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    google_api_key = st.text_input(
        "AI Model API Key",
        type="password",
        placeholder="Enter your API key (only needed for LLM agent)…",
        help="Gemini 1.5 key for model-backed analysis. Not required for canned/SQL paths."
    )
    if google_api_key and len(google_api_key) > 20:
        st.success("✅ API Key configured")
    elif google_api_key:
        st.warning("⚠️ Please check API key format")
    else:
        st.info("🔑 API key optional for most demo queries")

    st.divider()
    st.subheader("Data Configuration")
    st.caption(f"Using CSVs from: `{os.environ['DATA_DIR']}`")
    st.info("📊 All CSVs in this folder are auto-registered as DuckDB tables.")

    st.divider()
    st.subheader("Business Context")
    st.markdown("""
**Current Period**: Q4 2024  
**Segments**: Retail, Wholesale, New Merchants  
**Metrics**: Revenue, Conversion, Volume  
**Analysis Window**: 8 weeks
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
    l, r = st.columns(2)
    with l:
        st.markdown("""
**Performance Analysis:**
• Conversion rate trends by merchant segment  
• Revenue performance across business units  
• Transaction volume analysis by period  
• Customer acquisition metrics by channel
""")
    with r:
        st.markdown("""
**Strategic Insights:**
• Impact of recent feature launches  
• Policy change performance evaluation  
• Segment comparison & optimization opportunities  
• Growth trend analysis and forecasting
""")

# -------------------------------------------------------------------
# Helper: render CSV-ish responses nicely
def _try_render_tabular(result_str: str) -> bool:
    try:
        if ("," not in result_str) or ("\n" not in result_str):
            return False
        df = pd.read_csv(StringIO(result_str))
        if df.empty:
            return False
        st.dataframe(df, use_container_width=True)
        # quick chart if time-like column exists
        for time_col in ["period", "week_start", "date", "day", "month"]:
            if time_col in df.columns:
                try:
                    df_plot = df.copy()
                    df_plot[time_col] = pd.to_datetime(df_plot[time_col], errors="coerce")
                    df_plot = df_plot.dropna(subset=[time_col])
                    num_cols = df_plot.select_dtypes("number").columns.tolist()
                    if num_cols:
                        st.line_chart(data=df_plot.set_index(time_col)[num_cols])
                        break
                except Exception:
                    pass
                break
        return True
    except Exception:
        return False

# -------------------------------------------------------------------
# Router (uses fpna.run_bi if available; otherwise emulate)
def _run_query(question: str, api_key: str | None):
    q = (question or "").strip()
    ql = q.lower()

    if HAVE_RUN_BI:
        # Preferred new API
        return run_bi(q, api_key or None)

    # Fallback behavior using fpna_agent.py
    # 1) direct SQL path
    if ql.startswith("select") or ql.startswith("with "):
        if '_exec_sql' in globals() and _exec_sql:
            return _exec_sql(q)
        return "SQL engine not exposed by fpna_agent; please update to latest fpna core."

    # 2) common canned analysis path
    if ("conversion" in ql and "segment" in ql) and ('_conv_helper' in globals()) and _conv_helper:
        return _conv_helper()

    # 3) agent path requires API key
    if not api_key:
        return "API key required for agent-backed analysis. Enter it in the sidebar."
    from fpna_agent import create_agent  # local import to avoid confusion
    agent = create_agent(google_api_key=api_key)
    return agent.run(q)

# -------------------------------------------------------------------
# Execute
if go:
    if not user_question.strip():
        st.warning("Please enter a business question.")
    else:
        with st.spinner("Processing business intelligence query…"):
            try:
                result = _run_query(user_question, google_api_key or None)

                st.success("Analysis Complete")
                st.markdown("---")
                st.markdown("## 📈 Business Intelligence Report")

                # Render CSV-ish tables if possible; otherwise show as markdown/text
                if not _try_render_tabular(result):
                    st.markdown(result)

                st.markdown("---")
                with st.expander("📊 Analysis Details"):
                    st.markdown(f"""
**Data Sources**: CSVs in `{os.environ['DATA_DIR']}`  
**Method**: DuckDB/Pandas (deterministic) and Gemini 1.5 (if needed)  
**Agent Limits**: Steps={os.getenv('AGENT_MAX_STEPS','12')}, Model={os.getenv('GEMINI_MODEL','gemini-1.5-flash')}
""")
            except Exception as e:
                st.error("Analysis could not be completed.")
                with st.expander("🔧 Technical Details", expanded=False):
                    st.code(str(e))
                    st.write("**Error Type:**", type(e).__name__)

# -------------------------------------------------------------------
# Footer + light styling
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**🎯 Capabilities**  \nReal-time analysis  \nMulti-segment insights  \nTrend identification")
with c2:
    st.markdown("**📊 Data Coverage**  \nRevenue metrics  \nConversion analytics  \nPerformance indicators")
with c3:
    st.markdown("**🚀 AI Features**  \nNatural language queries  \nAutomated insights  \nExecutive summaries")

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
