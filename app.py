# app.py (Streamlit UI)
import os
import streamlit as st
from fpna_agent import agent

# Load keys
ot.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')

st.title("FP&A AI-Agent Prototype")

st.sidebar.header("Demo Data Setup")
use_demo = st.sidebar.checkbox("Use demo BigQuery dataset?", value=True)

question = st.text_input("Ask about the weekly funnel…")
if st.button("Run"):
    with st.spinner("Analyzing…"):
        # Optionally switch dataset in-agent prompt if demo
        if use_demo:
            question = "[DEMO DATA] " + question
        response = agent.run(question)
        st.markdown(response)
