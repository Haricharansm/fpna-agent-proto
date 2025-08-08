# app.py — FP&A UI with fast-path routing and env setup
import os, sys, streamlit as st

# Ensure module path & data dir
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
os.environ.setdefault("DATA_DIR", os.path.join(CURRENT_DIR, "data"))
os.environ.setdefault("GEMINI_MODEL", "gemini-1.5-flash")
os.environ.setdefault("AGENT_MAX_STEPS", "12")  # agent will read this if implemented

st.set_page_config(page_title="FP&A AI Agent", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Import agent + fast tools
try:
    from fpna_agent import create_agent
    # Optional: these exist in the version I shared; import if available
    try:
        from fpna_agent import execute_business_query as execute_sql  # SQL string -> CSV
    except Exception:
        execute_sql = None
    try:
        from fpna_agent import analyze_monthly_conversion_rates as analyze_conversion_trends
    except Exception:
        analyze_conversion_trends = None
    AGENT_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Cannot import FP&A agent: {e}")
    st.info("📁 Ensure fpna_agent.py is alongside app.py and dependencies are installed.")
    AGENT_AVAILABLE = False

# Header
st.markdown("""
<div style='text-align:center;padding:1rem 0;'>
  <h1 style='color:#1e3a8a;margin:0;'>📊 Financial Planning & Analysis</h1>
  <h2 style='color:#64748b;margin:0;font-weight:400;'>AI-Powered Business Intelligence Agent</h2>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

if not AGENT_AVAILABLE:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    google_api_key = st.text_input("AI Model API Key", type="password", placeholder="Enter your API key…")
    if google_api_key and len(google_api_key) > 20:
        st.success("✅ API Key configured")
    elif google_api_key:
        st.warning("⚠️ Please check API key format")
    else:
        st.info("🔑 API key required for model-backed analysis")

    st.divider()
    st.subheader("Data Configuration")
    st.caption(f"Using CSVs from: `{os.environ['DATA_DIR']}`")
    st.info("📊 CSVs in ./data are auto-registered as DuckDB tables")

    st.divider()
    st.subheader("Business Context")
    st.markdown("**Current Period**: Q4 2024  \n**Segments**: Retail, Wholesale, New  \n**Metrics**: Revenue, Conversion, Volume  \n**Window**: 8 weeks")

# Query UI
st.markdown("### Business Intelligence Query")
c1, c2 = st.columns([4, 1])
with c1:
    user_q = st.text_input(
        "Enter your business question:",
        placeholder="Conversion rate trends by merchant segment",
        label_visibility="collapsed"
    )
with c2:
    st.write("")
    go = st.button("🔍 Analyze", use_container_width=True, type="primary")

# Router for fast, reliable execution
def run_bi(question: str, api_key: str):
    q = (question or "").strip()
    ql = q.lower()

    # 1) Direct SQL path
    if ql.startswith("select") or ql.startswith("with "):
        if not execute_sql:
            return "SQL engine not exposed by fpna_agent; update to the latest agent file."
        return execute_sql(q)

    # 2) Common canned analysis path
    if ("conversion" in ql and "segment" in ql) and analyze_conversion_trends:
        return analyze_conversion_trends()

    # 3) Agent path (LLM)
    if not api_key:
        return "API key required for model-backed analysis. Enter it in the sidebar."
    agent = create_agent(google_api_key=api_key)
    # Keep the prompt clean—don’t prepend extra fluff
    return agent.run(q)

# Execute
if go:
    if not user_q.strip():
        st.warning("Please enter a business question.")
    else:
        with st.spinner("Processing business intelligence query…"):
            try:
                result = run_bi(user_q, google_api_key)
                st.success("Analysis Complete")
                st.markdown("---")
                st.markdown("## 📈 Business Intelligence Report")
                st.markdown(result if isinstance(result, str) else str(result))
                st.markdown("---")
                with st.expander("📊 Analysis Details"):
                    st.markdown("""
**Data Sources**: CSVs in ./data  
**Method**: SQL + stats over DuckDB (or LLM agent)  
**Window**: Recent 8-week demo  
**Notes**: Set GEMINI_MODEL env var to pick flash/pro
""")
            except Exception as e:
                st.error("Analysis could not be completed.")
                with st.expander("🔧 Technical Details"):
                    st.code(str(e))
