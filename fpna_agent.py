# fpna_agent.py - Professional FP&A AI Agent with CSV Dataset Analysis
import os
import pandas as pd
import numpy as np
from langchain.agents import initialize_agent, Tool
from datetime import datetime, timedelta
import requests
from io import StringIO

# AI model imports
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
    },
    'channel_performance': {
        'file': 'channel_performance.csv',
        'description': 'Customer acquisition metrics by channel and segment',
        'use_cases': ['channel analysis', 'acquisition cost', 'marketing performance']
    },
    'segment_analysis': {
        'file': 'segment_analysis.csv',
        'description': 'Overall segment performance comparison',
        'use_cases': ['segment comparison', 'performance benchmarks', 'strategic insights']
    }
}

def load_business_dataset(dataset_name='monthly_summary', use_github=True):
    """
    Load business datasets from CSV files (local or GitHub)
    
    Args:
        dataset_name (str): Name of dataset to load
        use_github (bool): If True, load from GitHub raw URL; if False, load from local data/ folder
    
    Returns:
        pd.DataFrame: Loaded dataset or error message
    """
    try:
        if dataset_name not in DATASET_CONFIG:
            available = list(DATASET_CONFIG.keys())
            return f"Dataset '{dataset_name}' not found. Available datasets: {available}"
        
        file_name = DATASET_CONFIG[dataset_name]['file']
        
        if use_github:
            # Load from GitHub raw URL
            url = f"{GITHUB_RAW_BASE_URL}{file_name}"
            response = requests.get(url)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
        else:
            # Load from local data folder
            local_path = f"data/{file_name}"
            if not os.path.exists(local_path):
                return f"Local file not found: {local_path}. Try setting use_github=True"
            df = pd.read_csv(local_path)
        
        return df
        
    except requests.exceptions.RequestException as e:
        return f"Error loading from GitHub: {str(e)}. File may not be uploaded yet."
    except Exception as e:
        return f"Error loading dataset '{dataset_name}': {str(e)}"

def get_dataset_info():
    """Get information about available datasets"""
    info = "📊 AVAILABLE BUSINESS INTELLIGENCE DATASETS:\n\n"
    
    for name, config in DATASET_CONFIG.items():
        info += f"🔹 **{name}**\n"
        info += f"   📁 File: {config['file']}\n"  
        info += f"   📝 Description: {config['description']}\n"
        info += f"   🎯 Use cases: {', '.join(config['use_cases'])}\n\n"
    
    return info

def analyze_conversion_trends_by_segment(time_period='monthly'):
    """
    Analyze conversion rate trends by merchant segment - Perfect for your BI query
    This directly answers: 'Show monthly conversion rates for each merchant segment in Q4 2024'
    """
    try:
        if time_period == 'monthly':
            df = load_business_dataset('monthly_summary')
        elif time_period == 'daily':
            df = load_business_dataset('daily_summary') 
        else:
            df = load_business_dataset('monthly_summary')
        
        if isinstance(df, str):  # Error message
            return df
        
        if time_period == 'monthly' and 'merchant_segment' in df.columns:
            # Create pivot table for conversion trends by segment
            pivot_table = df.pivot_table(
                index='month', 
                columns='merchant_segment', 
                values='conversion_rate',
                aggfunc='mean'
            ).round(4)
            
            # Calculate segment performance metrics
            segment_stats = df.groupby('merchant_segment').agg({
                'conversion_rate': ['mean', 'std', 'min', 'max'],
                'revenue': ['sum', 'mean'],
                'sessions': 'sum'
            }).round(4)
            
            # Identify top performing segments
            avg_conversion = df.groupby('merchant_segment')['conversion_rate'].mean().sort_values(ascending=False)
            
            result = f"""
🎯 CONVERSION RATE TRENDS BY MERCHANT SEGMENT - Q4 2024

📈 MONTHLY CONVERSION RATES BY SEGMENT:
{pivot_table.to_string()}

📊 SEGMENT PERFORMANCE SUMMARY:
{segment_stats.to_string()}

🏆 TOP PERFORMING SEGMENTS (by avg conversion rate):
{avg_conversion.to_string()}

💡 KEY INSIGHTS:
• Best performing segment: {avg_conversion.index[0]} ({avg_conversion.iloc[0]:.2%})
• Lowest performing segment: {avg_conversion.index[-1]} ({avg_conversion.iloc[-1]:.2%})
• Overall average conversion rate: {df['conversion_rate'].mean():.2%}
• Total revenue across all segments: ${df['revenue'].sum():,.2f}

📅 Analysis Period: {df['month'].min()} to {df['month'].max()}
"""
            return result
            
        else:
            return f"Required columns not found in dataset. Available columns: {list(df.columns)}"
            
    except Exception as e:
        return f"Analysis error: {str(e)}"

def execute_business_query(query: str) -> str:
    """Execute sophisticated business intelligence queries on uploaded datasets"""
    try:
        query_lower = query.lower().strip()
        
        # Determine which dataset to use based on query content
        if any(keyword in query_lower for keyword in ['conversion rate trends', 'monthly conversion', 'segment', 'merchant segment']):
            df = load_business_dataset('monthly_summary')
            context = "Monthly Summary Dataset"
        elif any(keyword in query_lower for keyword in ['transaction', 'detailed', 'raw data']):
            df = load_business_dataset('transaction_data')
            context = "Transaction Data Dataset"
        elif any(keyword in query_lower for keyword in ['daily', 'day by day', 'time series']):
            df = load_business_dataset('daily_summary')
            context = "Daily Summary Dataset"
        elif any(keyword in query_lower for keyword in ['channel', 'acquisition', 'marketing']):
            df = load_business_dataset('channel_performance')
            context = "Channel Performance Dataset"
        elif any(keyword in query_lower for keyword in ['segment comparison', 'segment analysis', 'benchmark']):
            df = load_business_dataset('segment_analysis')
            context = "Segment Analysis Dataset"
        else:
            df = load_business_dataset('monthly_summary')  # Default
            context = "Monthly Summary Dataset (default)"
        
        if isinstance(df, str):  # Error loading dataset
            return f"❌ Dataset Error: {df}\n\n{get_dataset_info()}"
        
        # Handle specific BI queries
        if 'conversion rate trends by merchant segment' in query_lower:
            return analyze_conversion_trends_by_segment('monthly')
        
        # Handle SQL-like queries
        result_df = df.copy()
        
        # Apply aggregations
        if 'select' in query_lower:
            if 'avg' in query_lower and 'conversion' in query_lower:
                if 'merchant_segment' in df.columns:
                    result_df = df.groupby('merchant_segment')['conversion_rate'].mean().round(4)
                elif 'conversion_rate' in df.columns:
                    result_df = pd.DataFrame({'average_conversion_rate': [df['conversion_rate'].mean()]})
                    
            elif 'sum' in query_lower and ('revenue' in query_lower):
                if 'merchant_segment' in df.columns and 'revenue' in df.columns:
                    result_df = df.groupby('merchant_segment')['revenue'].sum().round(2)
                elif 'revenue' in df.columns:
                    result_df = pd.DataFrame({'total_revenue': [df['revenue'].sum()]})
                    
            elif 'count' in query_lower:
                if 'merchant_segment' in df.columns:
                    result_df = df.groupby('merchant_segment').size()
                else:
                    result_df = pd.DataFrame({'total_records': [len(df)]})
        
        # Apply filters
        if 'where' in query_lower:
            filter_conditions = []
            
            # Segment filters
            segment_filters = {
                'premium': 'Premium',
                'enterprise': 'Enterprise', 
                'standard': 'Standard',
                'small business': 'Small Business',
                'startup': 'Startup'
            }
            
            for keyword, segment in segment_filters.items():
                if keyword in query_lower and 'merchant_segment' in df.columns:
                    if hasattr(result_df, 'loc'):
                        result_df = result_df[result_df['merchant_segment'] == segment]
                    break
            
            # Date filters
            if '2024-10' in query_lower and 'month' in df.columns:
                if hasattr(result_df, 'loc'):
                    result_df = result_df[result_df['month'] == '2024-10']
            elif '2024-11' in query_lower and 'month' in df.columns:
                if hasattr(result_df, 'loc'):
                    result_df = result_df[result_df['month'] == '2024-11']
            elif '2024-12' in query_lower and 'month' in df.columns:
                if hasattr(result_df, 'loc'):
                    result_df = result_df[result_df['month'] == '2024-12']
        
        # Apply sorting
        if 'order by' in query_lower:
            if 'revenue' in query_lower and hasattr(result_df, 'sort_values'):
                if 'revenue' in result_df.columns:
                    result_df = result_df.sort_values('revenue', ascending=False)
            elif 'conversion' in query_lower and hasattr(result_df, 'sort_values'):
                if 'conversion_rate' in result_df.columns:
                    result_df = result_df.sort_values('conversion_rate', ascending=False)
            elif 'month' in query_lower and hasattr(result_df, 'sort_values'):
                if 'month' in result_df.columns:
                    result_df = result_df.sort_values('month')
        
        # Apply limit
        if 'limit' in query_lower:
            try:
                limit_parts = query_lower.split('limit')
                if len(limit_parts) > 1:
                    limit_value = int(limit_parts[1].strip().split()[0])
                    if hasattr(result_df, 'head'):
                        result_df = result_df.head(limit_value)
            except:
                pass
        
        # Format results
        if hasattr(result_df, 'to_string'):
            result_output = result_df.to_string()
        else:
            result_output = str(result_df)
        
        return f"""
📊 BUSINESS INTELLIGENCE QUERY RESULTS
Dataset Used: {context}
Query: {query}

📈 RESULTS:
{result_output}

📋 Dataset Info: {len(df)} records, {len(df.columns)} columns
Available Columns: {', '.join(df.columns)}
"""
        
    except Exception as e:
        return f"❌ Query processing error: {str(e)}\n\n{get_dataset_info()}"

def retrieve_business_context(query: str) -> str:
    """Retrieve comprehensive business context and dataset information"""
    dataset_info = get_dataset_info()
    
    context = f"""
🏢 **BUSINESS INTELLIGENCE CONTEXT - Q4 2024**

{dataset_info}

📈 **STRATEGIC INITIATIVES:**
- Payment Flow Optimization (Feature A) - Launched October 1, 2024
- Mobile Dashboard Enhancement (Feature B) - Deployed September 15, 2024  
- Automated Reconciliation System (Feature C) - Released November 1, 2024

⚙️ **OPERATIONAL CHANGES:**
- Transaction Threshold Adjustment: Reduced minimum from $100 to $50 (November 15, 2024)
- Enhanced KYC Protocol Implementation (October 20, 2024)
- Streamlined Merchant Onboarding Process (November 10, 2024)

🎯 **MARKET SEGMENTS:**
- **Premium**: High-value merchants with advanced features
- **Enterprise**: Large-scale business partnerships  
- **Standard**: Mid-tier merchant base
- **Small Business**: SMB segment with basic needs
- **Startup**: New and emerging businesses

📊 **PERFORMANCE BENCHMARKS:**
- Target Conversion Rate: 85%
- Average Transaction Value: $250
- Active Merchant Base: 2,500+ partners
- Weekly Processing Volume: $2.1M average

🔍 **QUICK ANALYSIS COMMANDS:**
- "Show monthly conversion rates for each merchant segment" 
- "Analyze channel performance by segment"
- "Compare revenue across business units"
- "Daily trend analysis for Q4 2024"
"""
    return context

def create_agent(google_api_key=None):
    """Initialize the professional FP&A AI agent with CSV dataset analysis capabilities"""
    
    # Validate API configuration
    api_key = google_api_key or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("AI model API key required for analysis. Set GOOGLE_API_KEY environment variable.")
    
    if not HAS_GOOGLE_AI:
        raise ValueError("Required AI dependencies not available. Install: pip install langchain-google-genai")
    
    # Initialize AI model
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
    except Exception as e1:
        try:
            llm = ChatGooglePalm(
                google_api_key=api_key,
                temperature=0.1
            )
        except Exception as e2:
            raise ValueError(f"AI model initialization failed: {e1}")
    
    # Configure business intelligence tools
    intelligence_tools = [
        Tool(
            name="BusinessContextRetrieval", 
            func=retrieve_business_context, 
            description="Retrieves comprehensive business context including available datasets, strategic initiatives, operational changes, market segments, and performance benchmarks. Always use this first to understand available data."
        ),
        Tool(
            name="BusinessIntelligenceQuery", 
            func=execute_business_query, 
            description=(
                "Executes advanced business intelligence queries on uploaded CSV datasets. "
                "Automatically selects the right dataset based on query content. "
                "Available datasets: monthly_summary (conversion trends), transaction_data (detailed analysis), "
                "daily_summary (time series), channel_performance (acquisition analysis), segment_analysis (benchmarks). "
                "Supports SQL-style queries, aggregations, filtering, and sorting. "
                "Perfect for: 'Show monthly conversion rates for each merchant segment in Q4 2024'"
            )
        ),
        Tool(
            name="ConversionTrendsAnalysis",
            func=lambda query: analyze_conversion_trends_by_segment('monthly'),
            description="Specialized analysis for conversion rate trends by merchant segment. Direct answer to BI queries about segment performance over time."
        )
    ]
    
    # Initialize the FP&A agent
    fp_agent = initialize_agent(
        tools=intelligence_tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        agent_kwargs={
            "prefix": """You are a senior Financial Planning & Analysis (FP&A) AI specialist providing executive-level business intelligence analysis using uploaded CSV datasets.

Your analytical approach:
1. ALWAYS start with BusinessContextRetrieval to understand available datasets and business context
2. Use BusinessIntelligenceQuery for general data analysis and SQL-like queries
3. Use ConversionTrendsAnalysis for specific conversion rate trend analysis by merchant segment
4. Synthesize findings into actionable business insights with specific metrics
5. Connect performance data to business context (initiatives, feature launches, operational changes)
6. Provide executive-ready analysis with clear conclusions and strategic recommendations

You analyze REAL uploaded CSV data from Q4 2024, not synthetic data.
Focus on delivering professional, data-driven insights that support strategic decision-making.
Always reference specific performance metrics from the actual datasets.""",
            
            "format_instructions": """Follow this professional analysis format:

Thought: I need to understand available datasets and business context first
Action: BusinessContextRetrieval
Action Input: business context and dataset information
Observation: [business context and available datasets]
Thought: Now I'll execute the specific analysis requested using the appropriate dataset
Action: [BusinessIntelligenceQuery OR ConversionTrendsAnalysis]
Action Input: [specific query matching the user's request]
Observation: [analysis results from real CSV data]
Thought: I have comprehensive data analysis results to provide business intelligence
Final Answer: [executive-level analysis with specific metrics from real data, insights, strategic recommendations, and connection to business context]

Always provide specific metrics, percentages, and dollar amounts from the actual CSV data."""
        }
    )
    
    return fp_agent

# Convenience functions for direct analysis
def quick_conversion_analysis():
    """Quick analysis of conversion trends - perfect for testing"""
    return analyze_conversion_trends_by_segment('monthly')

def quick_dataset_check():
    """Check if datasets are accessible"""
    results = {}
    for dataset_name in DATASET_CONFIG.keys():
        df = load_business_dataset(dataset_name)
        if isinstance(df, pd.DataFrame):
            results[dataset_name] = f"✅ Loaded successfully ({len(df)} records)"
        else:
            results[dataset_name] = f"❌ {df}"
    
    return results

if __name__ == "__main__":
    print("🚀 FP&A Agent with CSV Dataset Analysis")
    print("="*50)
    
    # Test dataset access
    print("\n📊 Testing Dataset Access:")
    check_results = quick_dataset_check()
    for dataset, status in check_results.items():
        print(f"  {dataset}: {status}")
    
    # Test conversion analysis
    print("\n🎯 Testing Conversion Rate Analysis:")
    print(quick_conversion_analysis())
