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
    try:
        client = bigquery.Client()
        df = client.query(sql).to_dataframe()
        return df.to_csv(index=False)
    except Exception as e:
        return f"BigQuery error: {str(e)}"

def create_agent():
    """Create the agent with current environment credentials"""
    
    # Check if API key is available
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    # Assemble tools
    tools = [
        Tool(
            name="RetrieveDocs", 
            func=retrieve_docs, 
            description="Fetches stub product context."
        ),
        Tool(
            name="BigQuery", 
            func=query_bigquery, 
            description="Executes BigQuery SQL and returns CSV."
        )
    ]
    
    # Initialize LLM with current API key
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=api_key
    )
    
    # Create and return agent
    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",  # Updated agent type
        verbose=True,
        handle_parsing_errors=True  # Handle potential parsing errors
    )
    
    return agent
