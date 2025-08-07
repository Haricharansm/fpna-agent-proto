# fpna_agent.py
import os
import pandas as pd
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from datetime import datetime, timedelta
import random

try:
    from google.cloud import bigquery
    HAS_BIGQUERY = True
except ImportError:
    HAS_BIGQUERY = False

# --- Enhanced document context retrieval ---
def retrieve_docs(query: str) -> str:
    """Enhanced product context with more realistic data"""
    return """
**Product Context & Business Rules (Demo Data)**

🚀 **Recent Feature Launches:**
- Feature A (Payment Flow Optimization) launched on 2024-10-01
- Feature B (Mobile Dashboard) launched on 2024-09-15
- Feature C (Auto-reconciliation) launched on 2024-11-01

📋 **Policy Changes:**
- Funding rules updated on 2024-11-15 (reduced minimum transaction from $100 to $50)
- KYC requirements enhanced on 2024-10-20
- New merchant onboarding process implemented on 2024-11-10

🏢 **Merchant Segments:**
- **Retail**: Traditional brick-and-mortar stores (35% of volume)
- **Wholesale**: B2B distributors and suppliers (40% of volume)  
- **NewComers**: Recently onboarded merchants (<90 days, 25% of volume)

📊 **Key Metrics:**
- Conversion Rate Target: 85%
- Average Transaction Value: $250
- Monthly Active Merchants: ~2,500
- Weekly Transaction Volume: $2.1M average
"""

def generate_demo_data():
    """Generate realistic demo financial data"""
    # Generate data for last 8 weeks
    weeks = []
    base_date = datetime.now() - timedelta(weeks=8)
    
    segments = ['Retail', 'Wholesale', 'NewComers']
    
    data = []
    for i in range(8):
        week_start = base_date + timedelta(weeks=i)
        week_end = week_start + timedelta(days=6)
        
        for segment in segments:
            # Simulate realistic business metrics with trends
            base_conversion = 0.82 if segment == 'Retail' else 0.87 if segment == 'Wholesale' else 0.75
            base_volume = 800000 if segment == 'Wholesale' else 650000 if segment == 'Retail' else 450000
            
            # Add some realistic variation and trends
            trend_factor = 1 + (i * 0.02)  # 2% weekly growth
            seasonal_factor = 1 + (0.1 * (i % 4 - 2) / 2)  # Some seasonality
            
            conversion_rate = base_conversion * (0.95 + random.random() * 0.1) * trend_factor
            volume = int(base_volume * seasonal_factor * trend_factor * (0.9 + random.random() * 0.2))
            transactions = int(volume / 250)  # Avg transaction ~$250
            
            data.append({
                'week_start': week_start.strftime('%Y-%m-%d'),
                'week_end': week_end.strftime('%Y-%m-%d'),
                'segment': segment,
                'conversion_rate': round(conversion_rate, 3),
                'total_volume': volume,
                'transaction_count': transactions,
                'avg_transaction_value': round(volume / transactions, 2),
                'new_merchants': random.randint(15, 45) if segment == 'NewComers' else random.randint(3, 12)
            })
    
    return pd.DataFrame(data)

def query_bigquery_demo(sql: str) -> str:
    """Demo BigQuery function that returns realistic data"""
    try:
        # Generate demo data
        df = generate_demo_data()
        
        # Simple SQL parsing for demo purposes
        sql_lower = sql.lower()
        
        if 'where' in sql_lower and 'segment' in sql_lower:
            # Try to extract segment filter
            if 'retail' in sql_lower:
                df = df[df['segment'] == 'Retail']
            elif 'wholesale' in sql_lower:
                df = df[df['segment'] == 'Wholesale']
            elif 'newcomers' in sql_lower:
                df = df[df['segment'] == 'NewComers']
        
        if 'order by' in sql_lower and 'week' in sql_lower:
            df = df.sort_values('week_start')
        
        if 'limit' in sql_lower:
            try:
                limit = int(sql_lower.split('limit')[1].strip())
                df = df.head(limit)
            except:
                pass
        
        # Return CSV format
        result = df.to_csv(index=False)
        return f"Query executed successfully. Results:\n{result}"
        
    except Exception as e:
        return f"Demo BigQuery error: {str(e)}"

def query_bigquery_real(sql: str) -> str:
    """Real BigQuery function"""
    try:
        if not HAS_BIGQUERY:
            return "Error: google-cloud-bigquery not installed. Please install it or use demo mode."
        
        client = bigquery.Client()
        df = client.query(sql).to_dataframe()
        return f"Query executed successfully. Results:\n{df.to_csv(index=False)}"
    except Exception as e:
        return f"BigQuery error: {str(e)}"

def create_agent(use_demo=True):
    """Create the agent with current environment credentials"""
    
    # Check if API key is available
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    # Choose BigQuery function based on mode
    bigquery_func = query_bigquery_demo if use_demo else query_bigquery_real
    bigquery_desc = (
        "Executes BigQuery SQL queries on demo financial data. "
        "Available tables: weekly_funnel (with columns: week_start, week_end, segment, "
        "conversion_rate, total_volume, transaction_count, avg_transaction_value, new_merchants)"
    ) if use_demo else "Executes BigQuery SQL and returns CSV results from real data."
    
    # Assemble tools
    tools = [
        Tool(
            name="RetrieveDocs", 
            func=retrieve_docs, 
            description="Retrieves business context, feature launches, policy changes, and merchant segment definitions."
        ),
        Tool(
            name="BigQuery", 
            func=bigquery_func, 
            description=bigquery_desc
        )
    ]
    
    # Initialize LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=api_key
    )
    
    # Create agent with enhanced prompting
    agent = initialize_agent(
        tools,
        llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={
            "prefix": """You are a Financial Planning & Analysis (FP&A) AI assistant. You help analyze business performance data and provide insights.

When answering questions:
1. First use RetrieveDocs to get business context
2. Then use BigQuery to query relevant data
3. Provide clear insights with specific numbers
4. Mention any relevant business context (feature launches, policy changes, etc.)
5. Format your response in a clear, executive-ready format

Available data includes weekly funnel metrics by merchant segment (Retail, Wholesale, NewComers)."""
        }
    )
    
    return agent
