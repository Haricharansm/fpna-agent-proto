# app.py — FP&A UI with fast-path routing to fpna.run_bi
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

# Import agent/router from fpna.py
try:
    from fpna import run_bi  # one-call router that always returns an answer (string)
    FPNA_AVAILABLE = True
except Exception as e:
    FPNA_AVAILABLE = False
    fpna_import_err = str(e)

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

if not FPNA_AVAILABLE:
    st.error(f"❌ Cannot import FP&A core (`fpna.py`): {fpna_import_err}")
    st.info("📁 Make sure `fpna.py` is next to `app.py` and dependencies are installed.")
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
    st.write("")  # spacing
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
# Helpers to render results nicely
def _try_render_tabular(result_str: str) -> bool:
    """If result looks like CSV, render a dataframe and a simple chart."""
    try:
        # quick sniff: must have a comma and a newline
        if ("," not in result_str) or ("\n" not in result_str):
            return False
        df = pd.read_csv(StringIO(result_str))
        if df.empty:
            return False

        st.dataframe(df, use_container_width=True)

        # quick line chart if there's a time-like column
        for time_col in ["period", "week_start", "date", "day", "month"]:
            if time_col in df.columns:
                try:
                    df_plot = df.copy()
                    df_plot[time_col] = pd.to_datetime(df_plot[time_col], errors="coerce")
                    df_plot = df_plot.dropna(subset=[time_col])
                    numeric_cols = df_plot.select_dtypes("number").columns.tolist()
                    if numeric_cols:
                        st.line_chart(data=df_plot.set_index(time_col)[numeric_cols])
                        break
                except Exception:
                    pass
                break
        return True
    except Exception:
        return False

# -------------------------------------------------------------------
# Execute
if go:
    if not user_question.strip():
        st.warning("Please enter a business question.")
    else:
        with st.spinner("Processing business intelligence query…"):
            try:
                # run_bi decides: deterministic path, direct SQL, or agent (if API key given)
                result = run_bi(user_question, google_api_key or None)

                st.success("Analysis Complete")
                st.markdown("---")
                st.markdown("## 📈 Business Intelligence Report")

                # If it's csv-ish, show a table (and chart); otherwise print as markdown
                rendered_tabular = _try_render_tabular(result)
                if not rendered_tabular:
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
# Footer
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**🎯 Capabilities**  \nReal-time analysis  \nMulti-segment insights  \nTrend identification")
with c2:
    st.markdown("**📊 Data Coverage**  \nRevenue metrics  \nConversion analytics  \nPerformance indicators")
with c3:
    st.markdown("**🚀 AI Features**  \nNatural language queries  \nAutomated insights  \nExecutive summaries")

# Styling
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
