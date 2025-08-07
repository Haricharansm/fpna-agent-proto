# app.py - Google AI Streamlit App
import os
import streamlit as st
from fpna_agent import create_agent

st.set_page_config(
    page_title="FP&A AI Agent - Google AI", 
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📊 FP&A AI-Agent Prototype")
st.markdown("**Powered by Google AI (Gemini) - Free API!** 🆓")

# Sidebar for API key
with st.sidebar:
    st.header("🔑 Google AI Setup")
    
    # Instructions
    with st.expander("📋 How to get your FREE Google AI key"):
        st.markdown("""
        1. **Go to**: [aistudio.google.com](https://aistudio.google.com)
        2. **Sign in** with your Google account
        3. **Click "Get API key"** (top right)
        4. **Create API key** → "Create API key in new project"
        5. **Copy the key** (starts with `AIza...`)
        6. **Paste it below** 👇
        
        ✅ **Completely FREE** - No billing required!
        """)
    
    # API Key input
    google_api_key = st.text_input(
        "Google AI API Key",
        type="password",
        placeholder="AIza...",
        help="Get your free API key from aistudio.google.com"
    )
    
    # Status indicator
    if google_api_key:
        if google_api_key.startswith('AIza'):
            st.success("✅ API Key looks valid!")
        else:
            st.warning("⚠️ API key should start with 'AIza'")
    else:
        st.info("👆 Enter your Google AI API key to get started")
    
    st.divider()
    
    # Configuration
    st.header("⚙️ Configuration")
    use_demo = st.checkbox("Use demo data", value=True, disabled=True, help="Demo mode with realistic financial data")
    
    st.info("🎯 Using demo financial data with 8 weeks of metrics across Retail, Wholesale, and NewComers segments")

# Main interface
col1, col2 = st.columns([4, 1])

with col1:
    user_question = st.text_input(
        "Ask about your financial performance:",
        placeholder="e.g., Show me conversion rates by merchant segment this week",
        help="Ask questions about funnel metrics, trends, segments, or business performance"
    )

with col2:
    st.write("")  # Spacing
    run_button = st.button("🚀 Analyze", use_container_width=True, type="primary")

# Example questions
with st.expander("💡 Try these example questions"):
    examples = [
        "Show me conversion rates by merchant segment",
        "What's the weekly revenue trend across all segments?", 
        "Compare transaction volumes between Retail and Wholesale",
        "How did the Feature A launch on Oct 1st impact our metrics?",
        "What's the impact of the funding rule changes in November?",
        "Show me new merchant acquisition by segment",
        "Which segment has the highest average transaction value?",
        "What's our overall business performance trend?"
    ]
    
    for i, example in enumerate(examples):
        col_ex1, col_ex2 = st.columns([6, 1])
        with col_ex1:
            st.write(f"• {example}")
        with col_ex2:
            if st.button("📋", key=f"copy_{i}", help="Use this question"):
                st.rerun()

# Main analysis section
if run_button:
    if not user_question.strip():
        st.warning("⚠️ Please enter a question to analyze.")
    elif not google_api_key:
        st.error("🔑 Please enter your Google AI API key in the sidebar.")
        st.info("👈 Get your free API key from: https://aistudio.google.com")
    else:
        # Run analysis
        with st.spinner("🤖 Google AI is analyzing your request... (This may take 10-30 seconds)"):
            try:
                # Create agent with API key
                agent = create_agent(google_api_key=google_api_key)
                
                # Add demo data context to the prompt
                enhanced_prompt = f"[DEMO DATA ANALYSIS] {user_question}"
                
                # Run the agent
                result = agent.run(enhanced_prompt)
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Results section
                st.markdown("---")
                st.markdown("## 📈 Analysis Results")
                st.markdown(result)
                
                # Add a note about the data
                with st.expander("ℹ️ About this analysis"):
                    st.info("""
                    This analysis is based on demo data containing:
                    - 8 weeks of financial performance metrics
                    - 3 merchant segments: Retail, Wholesale, NewComers  
                    - Realistic business context including feature launches and policy changes
                    - Metrics: conversion rates, transaction volumes, merchant counts
                    """)
                
            except Exception as e:
                st.error("❌ An error occurred during analysis:")
                error_msg = str(e)
                
                # User-friendly error messages
                if "API key" in error_msg:
                    st.error("🔑 Invalid or missing API key")
                    st.info("Please check your Google AI API key. Get a free one at: https://aistudio.google.com")
                elif "dependencies not installed" in error_msg:
                    st.error("📦 Missing required packages")
                    st.code("pip install langchain-google-genai google-generativeai")
                elif "quota" in error_msg.lower():
                    st.error("📊 API quota exceeded")
                    st.info("Google AI has generous free limits. Try again in a few minutes.")
                else:
                    st.error(f"Unexpected error: {error_msg}")
                
                # Debug section
                with st.expander("🔍 Debug Information"):
                    st.code(error_msg)
                    st.markdown("**Troubleshooting:**")
                    st.markdown("1. Verify your API key starts with 'AIza'")
                    st.markdown("2. Ensure you have internet connection")
                    st.markdown("3. Try a simpler question")
                    st.markdown("4. Check if you've exceeded Google AI free limits")

# Footer with helpful info
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown("**🤖 Powered by:**")
    st.markdown("Google AI (Gemini)")
    
with col_f2:
    st.markdown("**💰 Cost:**")
    st.markdown("Free (No billing required)")

with col_f3:
    st.markdown("**📊 Data:**")
    st.markdown("Demo financial metrics")

# Performance tips
with st.expander("⚡ Performance Tips"):
    st.markdown("""
    **For best results:**
    - Be specific in your questions
    - Ask about trends, comparisons, or specific metrics
    - Reference segments (Retail, Wholesale, NewComers)
    - Ask about time periods or business events
    
    **Google AI Features:**
    - ✅ Free with generous limits
    - ✅ Fast response times
    - ✅ Good at data analysis
    - ✅ No billing setup required
    """)
