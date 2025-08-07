# app.py (Streamlit UI)
import os
import streamlit as st
from fpna_agent import create_agent

# Set your OpenAI API key here
OPENAI_API_KEY = "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"  # Replace with your actual key
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

st.set_page_config(
    page_title="FP&A AI Agent", 
    page_icon="📊",
    layout="wide"
)

st.title("📊 FP&A AI-Agent Prototype")
st.markdown("Ask questions about your financial data and get AI-powered insights!")

# Optional GCP credentials (only needed if not using demo)
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Demo dataset toggle
    use_demo = st.checkbox(
        "Use demo BigQuery dataset?", 
        value=True,
        help="Enable this to use sample data without needing GCP credentials"
    )
    
    if not use_demo:
        st.subheader("GCP Credentials")
        gcp_key_json = st.text_area(
            "GCP Service Account JSON", 
            height=200,
            help="Paste your GCP service account JSON here"
        )
        
        if gcp_key_json:
            path = "/tmp/gcp_key.json"
            try:
                with open(path, "w") as f:
                    f.write(gcp_key_json)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
                st.success("✅ GCP credentials loaded")
            except Exception as e:
                st.error(f"Error saving GCP credentials: {e}")
    else:
        st.info("🔧 Using demo data - no GCP setup required")

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    description = st.text_input(
        "Ask about the weekly funnel…",
        placeholder="e.g., Show me conversion rates by merchant segment this week",
        help="Enter your question about financial performance and analytics"
    )

with col2:
    st.write("")  # Add some spacing
    run_button = st.button("🚀 Run Analysis", use_container_width=True)

# Example questions
with st.expander("💡 Example Questions"):
    st.markdown("""
    - Show me conversion rates by merchant segment this week
    - What's the trend in weekly funnel performance?
    - Compare revenue between Retail and Wholesale segments
    - How did Feature A launch impact our metrics?
    - Show me the funding rule changes impact
    """)

if run_button:
    if not description.strip():
        st.warning("⚠️ Please enter a question to analyze.")
    elif not use_demo and not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        st.error("🔑 Please provide GCP credentials or enable demo mode.")
    else:
        with st.spinner("🤖 AI Agent is analyzing your request..."):
            try:
                # Create agent with credentials available
                agent = create_agent(use_demo=use_demo)
                
                prompt = description
                if use_demo:
                    prompt = "[DEMO DATA] " + description
                
                # Run the agent
                result = agent.run(prompt)
                
                # Display results
                st.success("✅ Analysis Complete!")
                st.markdown("### 📈 Results:")
                st.markdown(result)
                
            except Exception as e:
                st.error("❌ An error occurred during analysis:")
                st.error(str(e))
                
                # Debug info in expander
                with st.expander("🔍 Debug Information"):
                    st.code(str(e))
                    st.write("**Environment Check:**")
                    st.write(f"- OpenAI API Key: {'✅ Set' if os.environ.get('OPENAI_API_KEY') else '❌ Missing'}")
                    st.write(f"- GCP Credentials: {'✅ Set' if os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') else '❌ Missing'}")
                    st.write(f"- Demo Mode: {'✅ Enabled' if use_demo else '❌ Disabled'}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🤖 Powered by OpenAI & LangChain | 📊 FP&A Analytics Agent
    </div>
    """, 
    unsafe_allow_html=True
)
