# fpna_agent.py - Google AI Version
import os
import pandas as pd
from langchain.agents import initialize_agent, Tool
from datetime import datetime, timedelta
import random

# Google AI imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GOOGLE_AI = True
except ImportError:
    try:
        from langchain.llms import GooglePalm
        from langchain.chat_models import ChatGooglePalm
        HAS_GOOGLE_AI = True
    except ImportError:
        HAS_GOOGLE_AI = False

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
    base_date = datetime.now() - timedelta(weeks=8)
    segments = ['Retail', 'Wholesale', 'NewComers']
    
    data = []
    for i in range(8):
        week_start = base_date + timedelta(weeks=i)
        week_end = week_start + timedelta(days=6)
        
        for segment in segments:
            # Base metrics by segment
            base_conversion = {
                'Retail': 0.82,
                'Wholesale': 0.87, 
                'NewComers': 0.75
            }[segment]
            
            base_volume = {
                'Retail': 650000,
                'Wholesale': 800000,
                'NewComers': 450000
            }[segment]
            
            # Add realistic trends and variation
            trend_factor = 1 + (i * 0.02)  # 2% weekly growth
            seasonal_factor = 1 + (0.1 * (i % 4 - 2) / 2)  # Seasonality
            random_factor = 0.9 + random.random() * 0.2  # ±10% variation
            
            conversion_rate = base_conversion * (0.95 + random.random() * 0.1) * trend_factor
            volume = int(base_volume * seasonal_factor * trend_factor * random_factor)
            transactions = int(volume / 250)  # Avg $250 per transaction
            
            data.append({
                'week_start': week_start.strftime('%Y-%m-%d'),
                'week_end': week_end.strftime('%Y-%m-%d'),
                'segment': segment,
                'conversion_rate': round(min(conversion_rate, 1.0), 3),  # Cap at 100%
                'total_volume': volume,
                'transaction_count': transactions,
                'avg_transaction_value': round(volume / transactions, 2),
                'new_merchants': random.randint(15, 45) if segment == 'NewComers' else random.randint(3, 12)
            })
    
    return pd.DataFrame(data)

def query_bigquery_demo(sql: str) -> str:
    """Enhanced demo BigQuery function with better SQL parsing"""
    try:
        df = generate_demo_data()
        sql_lower = sql.lower().strip()
        
        # Handle SELECT statements
        if 'select' in sql_lower:
            # Basic column selection
            if 'segment' in sql_lower and 'conversion_rate' in sql_lower:
                df_result = df[['segment', 'conversion_rate']].groupby('segment').mean().round(3)
            elif 'total_volume' in sql_lower:
                df_result = df[['segment', 'week_start', 'total_volume']]
            elif 'count' in sql_lower or 'sum' in sql_lower:
                df_result = df.groupby('segment').agg({
                    'total_volume': 'sum',
                    'transaction_count': 'sum',
                    'new_merchants': 'sum'
                }).round(0)
            else:
                df_result = df
        else:
            df_result = df
        
        # Handle WHERE clauses
        if 'where' in sql_lower:
            if 'retail' in sql_lower:
                df_result = df_result[df_result['segment'] == 'Retail'] if 'segment' in df_result.columns else df_result
            elif 'wholesale' in sql_lower:
                df_result = df_result[df_result['segment'] == 'Wholesale'] if 'segment' in df_result.columns else df_result
            elif 'newcomers' in sql_lower or 'new' in sql_lower:
                df_result = df_result[df_result['segment'] == 'NewComers'] if 'segment' in df_result.columns else df_result
        
        # Handle ORDER BY
        if 'order by' in sql_lower:
            if 'week' in sql_lower:
                df_result = df_result.sort_values('week_start') if 'week_start' in df_result.columns else df_result
            elif 'volume' in sql_lower:
                df_result = df_result.sort_values('total_volume', ascending=False) if 'total_volume' in df_result.columns else df_result
        
        # Handle LIMIT
        if 'limit' in sql_lower:
            try:
                limit = int([x.strip() for x in sql_lower.split('limit') if x.strip()][-1].split()[0])
                df_result = df_result.head(limit)
            except:
                pass
        
        # Convert to CSV
        result_csv = df_result.to_csv(index=True if isinstance(df_result.index, pd.MultiIndex) or df_result.index.name else False)
        
        return f"Query executed successfully on demo data.\n\nResults:\n{result_csv}"
        
    except Exception as e:
        return f"Demo BigQuery error: {str(e)}\n\nSample data available with columns: week_start, week_end, segment, conversion_rate, total_volume, transaction_count, avg_transaction_value, new_merchants"

def create_agent(google_api_key=None):
    """Create agent with Google AI"""
    
    # Check for API key
    api_key = google_api_key or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Google AI API key required. Get one free at: https://aistudio.google.com")
    
    if not HAS_GOOGLE_AI:
        raise ValueError("Google AI dependencies not installed. Please install: pip install langchain-google-genai google-generativeai")
    
    # Initialize Google AI model
    try:
        # Try the newer ChatGoogleGenerativeAI first
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
    except Exception as e1:
        try:
            # Fallback to older GooglePalm
            llm = ChatGooglePalm(
                google_api_key=api_key,
                temperature=0.1
            )
        except Exception as e2:
            raise ValueError(f"Failed to initialize Google AI models. Error 1: {e1}, Error 2: {e2}")
    
    # Define tools
    tools = [
        Tool(
            name="RetrieveDocs", 
            func=retrieve_docs, 
            description="Retrieves business context including feature launches, policy changes, and merchant segment definitions. Use this to understand the business background."
        ),
        Tool(
            name="BigQuery", 
            func=query_bigquery_demo, 
            description=(
                "Executes SQL queries on demo financial data. "
                "Available table: weekly_funnel with columns: "
                "week_start, week_end, segment (Retail/Wholesale/NewComers), "
                "conversion_rate, total_volume, transaction_count, avg_transaction_value, new_merchants. "
                "Use SQL syntax like: SELECT segment, AVG(conversion_rate) FROM weekly_funnel GROUP BY segment"
            )
        )
    ]
    
    # Create the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        agent_kwargs={
            "prefix": """You are a Financial Planning & Analysis (FP&A) AI assistant specialized in business performance analysis.

Your approach:
1. First use RetrieveDocs to understand business context (features, policies, segments)
2. Then use BigQuery to query relevant financial data 
3. Analyze the data and provide clear, actionable insights
4. Reference specific numbers and trends
5. Connect findings to business context (feature launches, policy changes, etc.)
6. Format your final response in a clear, executive-ready format

Available data covers 8 weeks of funnel metrics across 3 merchant segments.
Be specific with numbers and always provide business context for your findings.""",
            
            "format_instructions": """Use the following format:

Thought: I need to understand the business context and then query the relevant data
Action: RetrieveDocs
Action Input: [query about business context]
Observation: [business context information]
Thought: Now I'll query the specific data needed
Action: BigQuery  
Action Input: [SQL query]
Observation: [query results]
Thought: I now have enough information to provide insights
Final Answer: [comprehensive analysis with specific numbers and business context]"""
        }
    )
    
    return agent
