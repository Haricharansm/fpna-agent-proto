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

# fpna_agent.py - Enhanced with better query processing and error handling

def execute_business_query(query: str) -> str:
    """Execute sophisticated business intelligence queries with improved error handling"""
    try:
        query_lower = query.lower().strip()
        
        # Enhanced dataset selection with better keyword matching
        dataset_choice = 'monthly_summary'  # Default
        context = "Monthly Summary Dataset (default)"
        
        if any(keyword in query_lower for keyword in ['conversion rate trends', 'monthly conversion', 'segment conversion']):
            dataset_choice = 'monthly_summary'
            context = "Monthly Summary Dataset"
        elif any(keyword in query_lower for keyword in ['transaction volume', 'volume analysis', 'transaction count', 'volume by period']):
            dataset_choice = 'daily_summary'  # Better for volume trends
            context = "Daily Summary Dataset - Transaction Volume"
        elif any(keyword in query_lower for keyword in ['transaction', 'detailed', 'raw data']):
            dataset_choice = 'transaction_data'
            context = "Transaction Data Dataset"
        elif any(keyword in query_lower for keyword in ['daily', 'day by day', 'time series', 'by period', 'period analysis']):
            dataset_choice = 'daily_summary'
            context = "Daily Summary Dataset"
        elif any(keyword in query_lower for keyword in ['channel', 'acquisition', 'marketing']):
            dataset_choice = 'channel_performance'
            context = "Channel Performance Dataset"
        elif any(keyword in query_lower for keyword in ['segment comparison', 'segment analysis', 'benchmark']):
            dataset_choice = 'segment_analysis'
            context = "Segment Analysis Dataset"
        
        # Load the selected dataset
        df = load_business_dataset(dataset_choice)
        
        if isinstance(df, str):  # Error loading dataset
            return f"❌ Dataset Error: {df}\n\nTrying alternative approach...\n{get_dataset_info()}"
        
        # Special handling for transaction volume analysis
        if 'transaction volume' in query_lower and 'period' in query_lower:
            return analyze_transaction_volume_by_period(df, dataset_choice)
        
        # Handle conversion rate analysis
        if 'conversion rate trends by merchant segment' in query_lower:
            return analyze_conversion_trends_by_segment('monthly')
        
        # Generic data summary if specific analysis fails
        summary_result = f"""
📊 BUSINESS INTELLIGENCE QUERY RESULTS
Dataset Used: {context}
Query: {query}

📈 DATASET OVERVIEW:
• Total Records: {len(df):,}
• Date Range: {get_date_range(df)}
• Available Columns: {', '.join(df.columns)}

📋 SAMPLE DATA (First 5 rows):
{df.head().to_string()}

💡 DATA SUMMARY:
{generate_data_summary(df)}
"""
        return summary_result
        
    except Exception as e:
        error_details = f"Query processing error: {str(e)}"
        fallback_info = get_dataset_info()
        return f"❌ {error_details}\n\n📚 Available Datasets:\n{fallback_info}"

def analyze_transaction_volume_by_period(df, dataset_name):
    """Analyze transaction volume trends by time period"""
    try:
        if dataset_name == 'daily_summary' and 'date' in df.columns:
            # Daily transaction volume analysis
            if 'transaction_count' in df.columns:
                volume_col = 'transaction_count'
            elif 'volume' in df.columns:
                volume_col = 'volume'
            elif 'sessions' in df.columns:
                volume_col = 'sessions'  # Fallback
            else:
                return f"❌ No volume column found in {dataset_name}. Available: {list(df.columns)}"
            
            # Time series analysis
            df_sorted = df.sort_values('date')
            total_volume = df_sorted[volume_col].sum()
            avg_daily_volume = df_sorted[volume_col].mean()
            peak_day = df_sorted.loc[df_sorted[volume_col].idxmax()]
            
            # Weekly trends
            df_sorted['week'] = pd.to_datetime(df_sorted['date']).dt.isocalendar().week
            weekly_volume = df_sorted.groupby('week')[volume_col].sum().sort_values(ascending=False)
            
            result = f"""
🎯 TRANSACTION VOLUME ANALYSIS BY PERIOD - Q4 2024

📈 VOLUME METRICS:
• Total Transaction Volume: {total_volume:,}
• Average Daily Volume: {avg_daily_volume:,.0f}
• Peak Day: {peak_day['date']} ({peak_day[volume_col]:,} transactions)

📊 TOP 5 WEEKS BY VOLUME:
{weekly_volume.head().to_string()}

📅 DAILY VOLUME TREND:
{df_sorted[['date', volume_col]].tail(10).to_string()}

💡 KEY INSIGHTS:
• Highest volume week: Week {weekly_volume.index[0]} ({weekly_volume.iloc[0]:,} transactions)
• Volume growth trend: {calculate_growth_trend(df_sorted[volume_col])}
• Data period: {df_sorted['date'].min()} to {df_sorted['date'].max()}
"""
            return result
            
        else:
            # Fallback analysis for other datasets
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_stats = df[numeric_cols].describe().round(2)
                return f"""
📊 TRANSACTION VOLUME ANALYSIS
Dataset: {dataset_name}

📈 NUMERIC SUMMARY:
{summary_stats.to_string()}

📋 Available columns: {', '.join(df.columns)}
Records analyzed: {len(df):,}
"""
            else:
                return f"❌ No numeric columns found for volume analysis in {dataset_name}"
                
    except Exception as e:
        return f"❌ Volume analysis error: {str(e)}"

def get_date_range(df):
    """Get date range from dataframe"""
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'month' in col.lower()]
    if date_columns:
        date_col = date_columns[0]
        try:
            return f"{df[date_col].min()} to {df[date_col].max()}"
        except:
            return "Date range unavailable"
    return "No date columns found"

def generate_data_summary(df):
    """Generate summary statistics for dataframe"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary = []
            for col in numeric_cols[:5]:  # Limit to top 5 numeric columns
                summary.append(f"• {col}: {df[col].mean():.2f} avg, {df[col].sum():,.0f} total")
            return "\n".join(summary)
        else:
            return "• Non-numeric data - categorical analysis available"
    except:
        return "• Summary statistics unavailable"

def calculate_growth_trend(series):
    """Calculate simple growth trend"""
    try:
        if len(series) < 2:
            return "Insufficient data"
        first_half = series.iloc[:len(series)//2].mean()
        second_half = series.iloc[len(series)//2:].mean()
        growth = ((second_half - first_half) / first_half) * 100
        if growth > 5:
            return f"Growing (+{growth:.1f}%)"
        elif growth < -5:
            return f"Declining ({growth:.1f}%)"
        else:
            return f"Stable ({growth:.1f}%)"
    except:
        return "Trend calculation unavailable"

# Update the load_business_dataset function to handle errors better
def load_business_dataset(dataset_name='monthly_summary', use_github=True):
    """Enhanced dataset loading with better error handling"""
    try:
        if dataset_name not in DATASET_CONFIG:
            available = list(DATASET_CONFIG.keys())
            return f"Dataset '{dataset_name}' not found. Available: {available}"
        
        file_name = DATASET_CONFIG[dataset_name]['file']
        
        if use_github:
            url = f"{GITHUB_RAW_BASE_URL}{file_name}"
            print(f"🔄 Loading {dataset_name} from: {url}")  # Debug info
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            print(f"✅ Successfully loaded {len(df)} rows from {dataset_name}")
            return df
        else:
            local_path = f"data/{file_name}"
            if not os.path.exists(local_path):
                return f"Local file not found: {local_path}"
            df = pd.read_csv(local_path)
            return df
        
    except requests.exceptions.Timeout:
        return "❌ Request timeout - GitHub may be slow. Please try again."
    except requests.exceptions.ConnectionError:
        return "❌ Connection error - Please check internet connectivity."
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"❌ File not found on GitHub: {file_name}. Please verify the file exists."
        else:
            return f"❌ HTTP error {e.response.status_code}: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error loading {dataset_name}: {str(e)}"
