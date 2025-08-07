# app.py (Streamlit UI)
import os
import streamlit as st
from fpna_agent import agent

# Load OpenAI key
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')

st.title("FP&A AI-Agent Prototype")

st.sidebar.header("Demo Data Setup")
use_demo = st.sidebar.checkbox("Use demo BigQuery dataset?", value=True)

question = st.text_input("Ask about the weekly funnel…")
if st.button("Run"):
    with st.spinner("Analyzing…"):
        prompt = question
        if use_demo:
            prompt = "[DEMO DATA] " + question
        response = agent.run(prompt)
        st.markdown(response)
