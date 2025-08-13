# app.py — FP&A BI app (deterministic DuckDB/Pandas + optional Mistral summary)
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

# Read optional HF secrets and expose as env vars for fpna_agent to use
for key in ("HF_MODEL", "HF_TOKEN"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

# -------------------------------------------------------------------
# Prefer new core (fpna.run_bi); fallback to fpna_agent Router
HAVE_RUN_BI = False
CORE_OK = False
err_msg = ""

try:
    from fpna import run_bi  # preferred new API
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
    st.info("📁 Ensure `fpna.py` or `fpna_agent.py` is next to `app.py` and dependencies are installed.")
    st.stop()

# -------------------------------------------------------------------
# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")

    hf_enabled = bool(os.getenv("HF_MODEL") or os.getenv("HF_TOKEN"))
    st.caption(
        "LLM summarizer: " +
        ("✅ Hugging Face configured (Mistral)" if hf_enabled else "⚠️ Not configured (optional)")
    )

    st.subheader("Data")
    st.caption(f"Reading CSVs from: `{os.environ['DATA_DIR']}`")
    st.info("📁 All `*.csv` in this folder are auto-registered as DuckDB tables.")

    st.divider()
    st.subheader("Business Context")
    st.markdown("""
**Period**: recent weeks/months (as data provides)  
**Segments**: Retail / Wholesale / etc.  
**Metrics**: Revenue, Conversion, Volume  
**Engine**: Deterministic DuckDB/Pandas; optional Mistral summary (Hugging Face)
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
    first = text.strip().splitlines()[0]
    return "," in first and len(first.split(",")) >= 2

def _render_result(result: str):
    """
    Handle three cases:
      1) "## Executive Summary ...\n---\n<CSV>"
      2) plain CSV
      3) markdown/text
    """
    if result.strip().startswith("## Executive Summary"):
        parts = result.split("\n---\n", 1)
        st.markdown(parts[0])  # show summary
        if len(parts) > 1 and _looks_like_csv(parts[1]):
            df = pd.read_csv(StringIO(parts[1]))
            st.dataframe(df, use_container_width=True)
        return

    if _looks_like_csv(result):
        df = pd.read_csv(StringIO(result))
        st.dataframe(df, use_container_width=True)
        return

    st.markdown(result)

# -------------------------------------------------------------------
# Router

def _run_query(question: str) -> str:
    q = (question or "").strip()
    if HAVE_RUN_BI:
        # New core path: fpna.run_bi handles deterministic + optional summary
        return run_bi(q, api_key=None)

    # Fallback: fpna_agent RouterAgent
    agent = create_agent()  # no key needed for deterministic routes
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
**LLM Summarizer**: Mistral via Hugging Face Inference ({'enabled' if hf_enabled else 'disabled'})  
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
    st.markdown("**🤖 LLM (Optional)**  \nExecutive summaries  \nHugging Face Inference (Mistral)")

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
