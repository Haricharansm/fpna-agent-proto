# app.py - Professional FP&A AI Agent Demo
import os
import streamlit as st
from fpna_agent import create_agent

st.set_page_config(
    page_title="FP&A AI Agent", 
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional header
st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='color: #1e3a8a; margin: 0;'>📊 Financial Planning & Analysis</h1>
        <h2 style='color: #64748b; margin: 0; font-weight: 400;'>AI-Powered Business Intelligence Agent</h2>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key input (simplified)
    google_api_key = st.text_input(
        "AI Model API Key",
        type="password",
        placeholder="Enter your API key...",
        help="API key for AI model access"
    )
    
    # Status indicator
    if google_api_key:
        if len(google_api_key) > 20:  # Basic validation
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please check API key format")
    else:
        st.info("🔑 API key required for analysis")
    
    st.divider()
    
    # Data source
    st.subheader("Data Configuration")
    data_source = st.selectbox(
        "Data Source",
        ["Demo Dataset", "Production Data"],
        index=0,
        help="Select data source for analysis"
    )
    
    if data_source == "Demo Dataset":
        st.info("📊 Using sample business data for demonstration")
    else:
        st.warning("🔒 Production data access requires additional configuration")
    
    st.divider()
    
    # Business context
    st.subheader("Business Context")
    st.markdown("""
    **Current Period**: Q4 2024  
    **Segments**: Retail, Wholesale, New Merchants  
    **Metrics**: Revenue, Conversion, Volume  
    **Analysis Period**: 8 weeks
    """)

# Main analysis interface
st.markdown("### Business Intelligence Query")

col1, col2 = st.columns([4, 1])

with col1:
    user_question = st.text_input(
        "Enter your business question:",
        placeholder="Analyze conversion rates by merchant segment over the past month",
        help="Ask questions about business performance, trends, or specific metrics",
        label_visibility="collapsed"
    )

with col2:
    st.write("")  # Spacing
    analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")

# Professional example queries
with st.expander("💼 Sample Business Questions"):
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.markdown("""
        **Performance Analysis:**
        • Conversion rate trends by merchant segment
        • Revenue performance across business units
        • Transaction volume analysis by period
        • Customer acquisition metrics by channel
        """)
    
    with col_ex2:
        st.markdown("""
        **Strategic Insights:**
        • Impact assessment of recent feature launches
        • Policy change performance evaluation
        • Segment comparison and optimization opportunities  
        • Growth trend analysis and forecasting
        """)

# Analysis execution
if analyze_button:
    if not user_question.strip():
        st.warning("Please enter a business question to analyze.")
    elif not google_api_key:
        st.error("API key required. Please configure in the sidebar.")
    else:
        with st.spinner("Processing business intelligence query..."):
            try:
                # Initialize AI agent
                agent = create_agent(google_api_key=google_api_key)
                
                # Process the query
                result = agent.run(f"Business Analysis Request: {user_question}")
                
                # Display professional results
                st.success("Analysis Complete")
                
                # Results presentation
                st.markdown("---")
                st.markdown("## 📈 Business Intelligence Report")
                
                # Format the results professionally
                st.markdown(result)
                
                # Additional insights section
                st.markdown("---")
                with st.expander("📊 Analysis Details"):
                    st.markdown("""
                    **Data Sources**: Internal business metrics, transaction data, performance indicators  
                    **Analysis Method**: AI-powered pattern recognition and statistical analysis  
                    **Time Period**: Recent 8-week performance window  
                    **Confidence Level**: High (based on comprehensive data analysis)
                    """)
                
            except Exception as e:
                st.error("Analysis could not be completed at this time.")
                
                # Professional error handling
                error_type = "configuration" if "API key" in str(e) else "processing"
                
                if error_type == "configuration":
                    st.info("Please verify your API configuration and try again.")
                else:
                    st.info("Please try rephrasing your question or contact support if the issue persists.")
                
                # Optional debug for internal use
                if st.checkbox("Show technical details", help="For troubleshooting purposes"):
                    st.code(str(e))

# Professional footer
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    **🎯 Capabilities**  
    Real-time analysis  
    Multi-segment insights  
    Trend identification
    """)

with footer_col2:
    st.markdown("""
    **📊 Data Coverage**  
    Revenue metrics  
    Conversion analytics  
    Performance indicators
    """)

with footer_col3:
    st.markdown("""
    **🚀 AI Features**  
    Natural language queries  
    Automated insights  
    Executive summaries
    """)

# Professional styling
st.markdown("""
<style>
    .stSelectbox > div > div {
        background-color: #f8fafc;
    }
    .stTextInput > div > div > input {
        background-color: #f8fafc;
    }
    .stButton > button {
        background-color: #1e40af;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
    }
    .stSuccess {
        background-color: #dcfce7;
        border: 1px solid #22c55e;
        border-radius: 8px;
    }
    .stInfo {
        background-color: #dbeafe;
        border: 1px solid #3b82f6;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
