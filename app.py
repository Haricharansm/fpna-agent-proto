# app.py (Streamlit UI)
import os
import streamlit as st
from fpna_agent import agent

st.title("FP&A AI-Agent Prototype")

# -- Credentials Setup --
st.sidebar.header("Credentials")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
gcp_key_json = st.sidebar.text_area("GCP Service Account JSON", height=200)

if openai_key:
    os.environ['OPENAI_API_KEY'] = openai_key
if gcp_key_json:
    # Write the GCP JSON to a temp file and set GOOGLE_APPLICATION_CREDENTIALS
    with open("/tmp/gcp_key.json", "w") as f:
        f.write(gcp_key_json)
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/tmp/gcp_key.json"

st.sidebar.header("Demo Data Setup")
use_demo = st.sidebar.checkbox("Use demo BigQuery dataset?", value=True)

question = st.text_input("Ask about the weekly funnel…")
if st.button("Run"):
    if not openai_key:
        st.error("Please enter your OpenAI API Key in the sidebar.")
    elif not gcp_key_json and not use_demo:
        st.error("Please enter your GCP Service Account JSON or toggle demo dataset.")
    else:
        with st.spinner("Analyzing…"):
            prompt = question
            if use_demo:
                prompt = "[DEMO DATA] " + question
            response = agent.run(prompt)
            st.markdown(response)
