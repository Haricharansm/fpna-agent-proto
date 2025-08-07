# app.py (Streamlit UI)
import os
import streamlit as st
from fpna_agent import create_agent

st.title("FP&A AI-Agent Prototype")

# Credentials input
st.sidebar.header("Credentials")
openai_key = st.sidebar.text_input(
    "OpenAI API Key", type="password"
)
gcp_key_json = st.sidebar.text_area(
    "GCP Service Account JSON", height=200
)

# Demo dataset toggle
use_demo = st.sidebar.checkbox(
    "Use demo BigQuery dataset?", value=True
)

# User question
description = st.text_input(
    "Ask about the weekly funnel…"
)

if st.button("Run"):
    if not openai_key:
        st.error("Enter your OpenAI API key.")
    elif not gcp_key_json and not use_demo:
        st.error("Enter GCP JSON or enable demo.")
    else:
        # Set credentials when needed
        os.environ['OPENAI_API_KEY'] = openai_key
        
        if gcp_key_json:
            path = "/tmp/gcp_key.json"
            with open(path, "w") as f:
                f.write(gcp_key_json)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path
        
        with st.spinner("Analyzing…"):
            try:
                # Create agent with credentials available
                agent = create_agent()
                
                prompt = description
                if use_demo:
                    prompt = "[DEMO DATA] " + description
                
                result = agent.run(prompt)
                st.markdown(result)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Please check your credentials and try again.")
