# fpna_agent.py
import os
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from google.cloud import bigquery

# --- Stub for document context retrieval ---
# Replace with real vector store logic as needed.
def retrieve_docs(query: str) -> str:
    return """
**Product Context (demo)**
- Feature A launched on 2024-10-01
- Changed funding rules on 2024-11-15
- Merchant segment definitions: Retail, Wholesale, NewComers
"""

# Tool: BigQuery query function
def query_bigquery(sql: str) -> str:
    client = bigquery.Client()
    df = client.query(sql).to_dataframe()
    return df.to_csv(index=False)

# Assemble tools and initialize agent
tools = [
    Tool(name="RetrieveDocs", func=retrieve_docs, description="Fetches stub product context."),
    Tool(name="BigQuery", func=query_bigquery, description="Executes BigQuery SQL and returns CSV.")
]

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.environ.get('OPENAI_API_KEY')
)
agent = initialize_agent(
    tools,
    llm,
    agent="react-with-tool-description",
    verbose=True
)
