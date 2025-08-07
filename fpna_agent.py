# fpna_agent.py
import os
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from google.cloud import bigquery
from chromadb import Client as ChromaClient

# Load GCP credentials via env var
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.environ.get('GCP_SA_KEY_PATH', '')

# Tool: BigQuery query function
def query_bigquery(sql: str) -> str:
    client = bigquery.Client()
    df = client.query(sql).to_dataframe()
    return df.to_csv(index=False)

# Tool: Document retrieval from Chroma vector store
def retrieve_docs(query: str) -> str:
    chroma = ChromaClient()
    collection = chroma.get_collection("product_docs")
    results = collection.query(query_texts=[query], n_results=3)
    snippets = results['documents'][0]
    return "\n\n".join(snippets)

# Initialize LLM and Agent
tools = [
    Tool(name="BigQuery", func=query_bigquery, description="Execute SQL and return CSV string."),
    Tool(name="DocRetrieval", func=retrieve_docs, description="Fetch relevant product-context snippets.")
]

llm = ChatOpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent="react-with-tool-description",
    verbose=True
)
