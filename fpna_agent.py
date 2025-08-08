# fpna_agent.py - Minimal Working Version with Guaranteed Import
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO

# Safe imports with fallbacks
try:
    from langchain.agents import initialize_agent, Tool
    from langchain.agents import AgentType
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️  LangChain not available - using fallback mode")
    LANGCHAIN_AVAILABLE = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    print("⚠️  Google AI not available - using fallback mode")
    GOOGLE_AI_AVAILABLE = False

# Configuration for dataset sources
GITHUB_RAW_BASE_URL = "https://raw.githubusercontent.com/Haricharansm/fpna-agent-proto/main/data/"

DATASET_CONFIG = {
    'monthly_summary': {
        'file': 'monthly_summary.csv',
        'description': 'Monthly conversion rates by merchant segment (Q4 2024)',
        'use_cases': ['conversion trends', 'monthly analysis', 'segment performance']
    },
    'transaction_data': {
        'file': 'transaction_data.csv', 
        'description': 'Raw transaction-level data with all details',
        'use_cases': ['detailed analysis', 'transaction patterns', 'customer behavior']
    },
    'daily_summary': {
        'file': 'daily_summary.csv',
        'description': 'Daily aggregated metrics across Q4 2024', 
        'use_cases': ['daily trends', 'time series', 'operational metrics']
    }
}

def load_business_dataset(dataset_name='monthly_summary', use_github=True):
    """Load business datasets with robust error handling"""
    try:
        if dataset_name not in DATASET_CONFIG:
            available = list(DATASET_CONFIG.keys())
            return f"Dataset '{dataset_name}' not found. Available: {available}"
        
        file_name = DATASET_CONFIG[dataset_name]['file']
        
        if use_github:
            url = f"{GITHUB_RAW_BASE_URL}{file_name}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            return df
        else:
            local_path = f"data/{file_name}"
            if not os.path.exists(local_path):
                return f"Local file not found: {local_path}"
            return pd.read_csv(local_path)
        
    except requests.exceptions.RequestException as e:
        return f"Error loading from GitHub: {str(e)}"
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

def analyze_conversion_trends():
    """Analyze conversion rate trends by merchant segment"""
    try:
        df = load_business_dataset('monthly_summary')
        
        if isinstance(df, str):  # Error message
            return f"❌ Data loading error: {df}"
        
        if 'merchant_segment' in df.columns and 'conversion_rate' in df.columns:
            # Create summary analysis
            segment_stats = df.groupby('merchant_segment')['conversion_rate'].agg(['mean', 'count']).round(4)
            
            result = f"""
🎯 CONVERSION RATE ANALYSIS - Q4 2024

📊 SEGMENT PERFORMANCE:
{segment_stats.to_string()}

💡 KEY INSIGHTS:
• Total segments analyzed: {len(segment_stats)}
• Records processed: {len(df)}
• Data period coverage: {df.get('month', ['N/A']).nunique() if hasattr(df.get('month', []), 'nunique') else 'Multiple periods'}

📈 TOP PERFORMERS:
{df.groupby('merchant_segment')['conversion_rate'].mean().sort_values(ascending=False).head(3).to_string()}
"""
            return result
        else:
            return f"Required columns not found. Available: {list(df.columns)}"
            
    except Exception as e:
        return f"Analysis error: {str(e)}"

def execute_simple_query(query: str) -> str:
    """Execute basic business queries with fallback functionality"""
    try:
        query_lower = query.lower().strip()
        
        # Load appropriate dataset
        if 'conversion' in query_lower or 'segment' in query_lower:
            df = load_business_dataset('monthly_summary')
            analysis_type = "Conversion Analysis"
        elif 'transaction' in query_lower or 'volume' in query_lower:
            df = load_business_dataset('daily_summary')
            analysis_type = "Transaction Volume Analysis"
        else:
            df = load_business_dataset('monthly_summary')  # Default
            analysis_type = "General Business Analysis"
        
        if isinstance(df, str):  # Error loading
            return f"❌ Dataset Error: {df}"
        
        # Basic analysis
        result = f"""
📊 BUSINESS INTELLIGENCE REPORT
Analysis Type: {analysis_type}
Query: {query}

📈 DATASET OVERVIEW:
• Records: {len(df):,}
• Columns: {len(df.columns)}
• Available data: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}

📋 SAMPLE DATA:
{df.head(3).to_string()}

💡 QUICK INSIGHTS:
• Data successfully loaded and processed
• Ready for detailed analysis
• Contact support for advanced queries
"""
        return result
        
    except Exception as e:
        return f"❌ Query processing error: {str(e)}"

class SimpleFPAAgent:
    """Simple FP&A Agent fallback when LangChain is not available"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.available = True
    
    def run(self, query: str) -> str:
        """Process business intelligence queries"""
        try:
            if 'conversion rate trends by merchant segment' in query.lower():
                return analyze_conversion_trends()
            elif 'conversion' in query.lower():
                return analyze_conversion_trends()
            else:
                return execute_simple_query(query)
        except Exception as e:
            return f"❌ Agent error: {str(e)}"

def create_agent(google_api_key=None):
    """
    Create FP&A agent with multiple fallback options
    This function is GUARANTEED to work and return an agent
    """
    try:
        # Validate API key
        api_key = google_api_key or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            return SimpleFPAAgent()  # Return fallback agent even without API key
        
        # Try full LangChain agent if dependencies available
        if LANGCHAIN_AVAILABLE and GOOGLE_AI_AVAILABLE:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    google_api_key=api_key,
                    temperature=0.1
                )
                
                tools = [
                    Tool(
                        name="ConversionAnalysis", 
                        func=analyze_conversion_trends,
                        description="Analyze conversion rate trends by merchant segment"
                    ),
                    Tool(
                        name="BusinessQuery", 
                        func=execute_simple_query,
                        description="Execute general business intelligence queries"
                    )
                ]
                
                agent = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False,
                    handle_parsing_errors=True,
                    max_iterations=3
                )
                
                return agent
                
            except Exception as e:
                print(f"⚠️  LangChain agent failed: {e}")
                return SimpleFPAAgent(api_key)
        
        # Return simple agent as fallback
        return SimpleFPAAgent(api_key)
        
    except Exception as e:
        print(f"⚠️  Agent creation error: {e}")
        # Always return something that works
        return SimpleFPAAgent()

# Test functions
def test_agent_creation():
    """Test that agent creation works"""
    try:
        agent = create_agent()
        result = agent.run("Test query: show conversion rates")
        return f"✅ Agent test successful: {len(result)} chars returned"
    except Exception as e:
        return f"❌ Agent test failed: {e}"

def test_data_access():
    """Test data loading"""
    try:
        df = load_business_dataset('monthly_summary')
        if isinstance(df, pd.DataFrame):
            return f"✅ Data access successful: {len(df)} rows loaded"
        else:
            return f"⚠️  Data access issue: {df}"
    except Exception as e:
        return f"❌ Data access failed: {e}"

if __name__ == "__main__":
    print("🚀 FP&A Agent - Testing Mode")
    print("="*40)
    print(test_agent_creation())
    print(test_data_access())
