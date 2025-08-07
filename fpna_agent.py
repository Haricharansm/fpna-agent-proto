# fpna_agent.py
import os
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from google.cloud import bigquery

# --- Stub for document context retrieval ---
# Replace with real vector store logic as needed.
def retrieve_docs(query: str) -> str:
    # Demo stub: returns placeholder context
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

# Initialize tools and agent
tools = [
    Tool(
        name="RetrieveDocs",
        func=retrieve_docs,
        description="Fetches high-level product context (demo stub)."
    ),
    Tool(
        name="BigQuery",
        func=query_bigquery,
        description="Executes SQL against BigQuery and returns CSV data."
    )
]

llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent="react-with-tool-description",
    verbose=True
)
