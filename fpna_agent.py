# fpna_agent.py - Professional FP&A AI Agent
import os
import pandas as pd
from langchain.agents import initialize_agent, Tool
from datetime import datetime, timedelta
import random

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

def retrieve_business_context(query: str) -> str:
    """Retrieve comprehensive business context and operational intelligence"""
    return """
**Business Intelligence Context - Q4 2024**

**Strategic Initiatives:**
- Payment Flow Optimization (Feature A) - Launched October 1, 2024
- Mobile Dashboard Enhancement (Feature B) - Deployed September 15, 2024  
- Automated Reconciliation System (Feature C) - Released November 1, 2024

**Operational Changes:**
- Transaction Threshold Adjustment: Reduced minimum from $100 to $50 (November 15, 2024)
- Enhanced KYC Protocol Implementation (October 20, 2024)
- Streamlined Merchant Onboarding Process (November 10, 2024)

**Market Segments:**
- **Retail Division**: Traditional commerce partners representing 35% of transaction volume
- **Wholesale Operations**: B2B distribution network contributing 40% of total volume  
- **Growth Segment**: New merchant partnerships (< 90 days) accounting for 25% of volume

**Performance Benchmarks:**
- Target Conversion Rate: 85%
- Average Transaction Value: $250
- Active Merchant Base: 2,500+ partners
- Weekly Processing Volume: $2.1M average
"""

def generate_business_metrics():
    """Generate comprehensive business performance dataset"""
    base_date = datetime.now() - timedelta(weeks=8)
    business_segments = ['Retail', 'Wholesale', 'Growth Segment']
    
    performance_data = []
    
    for week_index in range(8):
        period_start = base_date + timedelta(weeks=week_index)
        period_end = period_start + timedelta(days=6)
        
        for segment in business_segments:
            # Segment-specific performance baselines
            baseline_metrics = {
                'Retail': {'conversion': 0.82, 'volume': 650000},
                'Wholesale': {'conversion': 0.87, 'volume': 800000},
                'Growth Segment': {'conversion': 0.75, 'volume': 450000}
            }
            
            base_conversion = baseline_metrics[segment]['conversion']
            base_volume = baseline_metrics[segment]['volume']
            
            # Apply realistic business dynamics
            growth_trend = 1 + (week_index * 0.015)  # 1.5% weekly growth
            seasonal_adjustment = 1 + (0.08 * (week_index % 4 - 1.5) / 1.5)  # Seasonal variation
            market_volatility = 0.92 + random.random() * 0.16  # ±8% market variation
            
            # Calculate performance metrics
            actual_conversion = base_conversion * (0.96 + random.random() * 0.08) * growth_trend
            actual_volume = int(base_volume * seasonal_adjustment * growth_trend * market_volatility)
            transaction_count = int(actual_volume / 250)  # $250 average transaction
            
            performance_data.append({
                'period_start': period_start.strftime('%Y-%m-%d'),
                'period_end': period_end.strftime('%Y-%m-%d'),
                'business_segment': segment,
                'conversion_rate': round(min(actual_conversion, 0.99), 3),  # Cap at 99%
                'revenue_volume': actual_volume,
                'transaction_count': transaction_count,
                'average_transaction_value': round(actual_volume / transaction_count, 2),
                'new_merchant_acquisitions': random.randint(20, 50) if segment == 'Growth Segment' else random.randint(2, 8)
            })
    
    return pd.DataFrame(performance_data)

def execute_business_query(query: str) -> str:
    """Execute sophisticated business intelligence queries"""
    try:
        df = generate_business_metrics()
        query_normalized = query.lower().strip()
        
        # Advanced query processing
        if 'select' in query_normalized:
            # Handle aggregation queries
            if 'avg' in query_normalized and 'conversion' in query_normalized:
                result_df = df.groupby('business_segment')['conversion_rate'].mean().round(3)
                result_df.name = 'average_conversion_rate'
            elif 'sum' in query_normalized and ('revenue' in query_normalized or 'volume' in query_normalized):
                result_df = df.groupby('business_segment')['revenue_volume'].sum()
                result_df.name = 'total_revenue_volume'
            elif 'transaction' in query_normalized:
                result_df = df.groupby('business_segment').agg({
                    'transaction_count': 'sum',
                    'average_transaction_value': 'mean'
                }).round(2)
            else:
                result_df = df
        else:
            result_df = df
        
        # Apply business segment filters
        if 'where' in query_normalized:
            segment_filters = {
                'retail': 'Retail',
                'wholesale': 'Wholesale', 
                'growth': 'Growth Segment',
                'new': 'Growth Segment'
            }
            
            for keyword, segment in segment_filters.items():
                if keyword in query_normalized:
                    if 'business_segment' in df.columns:
                        result_df = result_df[result_df['business_segment'] == segment] if hasattr(result_df, 'loc') else result_df
                    break
        
        # Apply sorting and limiting
        if 'order by' in query_normalized:
            if 'revenue' in query_normalized or 'volume' in query_normalized:
                if hasattr(result_df, 'sort_values') and 'revenue_volume' in result_df.columns:
                    result_df = result_df.sort_values('revenue_volume', ascending=False)
            elif 'period' in query_normalized or 'date' in query_normalized:
                if hasattr(result_df, 'sort_values') and 'period_start' in result_df.columns:
                    result_df = result_df.sort_values('period_start')
        
        if 'limit' in query_normalized:
            try:
                limit_value = int([x.strip() for x in query_normalized.split('limit') if x.strip()][-1].split()[0])
                if hasattr(result_df, 'head'):
                    result_df = result_df.head(limit_value)
            except:
                pass
        
        # Format results professionally
        if hasattr(result_df, 'to_csv'):
            result_output = result_df.to_csv(index=True if isinstance(result_df.index, pd.MultiIndex) or result_df.index.name else False)
        else:
            result_output = str(result_df)
        
        return f"Business Intelligence Query Executed Successfully\n\nResults:\n{result_output}"
        
    except Exception as e:
        return f"Query processing error: {str(e)}\n\nAvailable metrics: period_start, period_end, business_segment, conversion_rate, revenue_volume, transaction_count, average_transaction_value, new_merchant_acquisitions"

def create_agent(google_api_key=None):
    """Initialize the professional FP&A AI agent"""
    
    # Validate API configuration
    api_key = google_api_key or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("AI model API key required for analysis")
    
    if not HAS_GOOGLE_AI:
        raise ValueError("Required AI dependencies not available")
    
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
            description="Retrieves comprehensive business context including strategic initiatives, operational changes, market segments, and performance benchmarks. Essential for understanding business background."
        ),
        Tool(
            name="BusinessIntelligenceQuery", 
            func=execute_business_query, 
            description=(
                "Executes advanced business intelligence queries on performance data. "
                "Available dataset: business_performance with metrics including "
                "period_start, period_end, business_segment (Retail/Wholesale/Growth Segment), "
                "conversion_rate, revenue_volume, transaction_count, average_transaction_value, new_merchant_acquisitions. "
                "Supports SQL-style queries with aggregations, filtering, and sorting."
            )
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
            "prefix": """You are a senior Financial Planning & Analysis (FP&A) AI specialist providing executive-level business intelligence.

Your analytical approach:
1. Retrieve comprehensive business context using BusinessContextRetrieval
2. Execute targeted data analysis using BusinessIntelligenceQuery
3. Synthesize findings into actionable business insights
4. Present results with specific metrics and strategic recommendations
5. Connect performance data to business context (initiatives, changes, market conditions)
6. Provide executive-ready analysis with clear conclusions

Focus on delivering professional, data-driven insights that support strategic decision-making.
Always reference specific performance metrics and business context in your analysis.""",
            
            "format_instructions": """Follow this professional analysis format:

Thought: I need to understand the business context and identify relevant performance data
Action: BusinessContextRetrieval
Action Input: [business context query]
Observation: [business context information]
Thought: Now I'll analyze the specific performance data requested
Action: BusinessIntelligenceQuery  
Action Input: [targeted data query]
Observation: [performance data results]
Thought: I have sufficient information to provide comprehensive business intelligence
Final Answer: [executive-level analysis with specific metrics, insights, and strategic recommendations]"""
        }
    )
    
    return fp_agent
