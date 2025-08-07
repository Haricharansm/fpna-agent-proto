# data_generator.py
"""
Generates synthetic weekly performance data for demo purposes.
Includes merchant segment conversion rates, revenue by business unit,
transaction volumes, acquisition channels, and flags for feature launches and policy changes.
Adjust `PROJECT_ID`, `DATASET`, and `TABLE` as needed.
"""
from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration constants
SEGMENTS = ["Retail", "Wholesale", "SMB", "Enterprise"]
BUSINESS_UNITS = ["Payments", "Capital", "Subscriptions"]
CHANNELS = ["Organic", "Paid Search", "Referral", "Email"]
FEATURE_LAUNCH_DATE = datetime(2024, 6, 15)
POLICY_CHANGE_DATE = datetime(2024, 7, 1)


def generate_weekly_performance(start_date: datetime, weeks: int = 8) -> pd.DataFrame:
    """
    Creates a DataFrame with enriched performance metrics:
    - Conversion rate (leads / visits) per segment
    - Revenue per business unit
    - Transaction volume
    - Acquisition channel breakdown
    - Flags for feature launch and policy active
    """
    records = []
    for i in range(weeks):
        week = start_date + timedelta(weeks=i)
        for segment in SEGMENTS:
            # Base volume for this segment-week
            visits = np.random.randint(2000, 10000)
            # leads & funded with variability
            leads = int(visits * np.random.uniform(0.1, 0.25))
            funded = int(leads * np.random.uniform(0.2, 0.6))
            conversion_rate = round(leads / visits, 3)
            funding_rate = round(funded / leads, 3) if leads else 0
            for unit in BUSINESS_UNITS:
                revenue = funded * np.random.uniform(1000, 5000)
                transactions = np.random.randint(100, 1000)
                channel = np.random.choice(CHANNELS)
                records.append({
                    "week_start": week.strftime("%Y-%m-%d"),
                    "merchant_segment": segment,
                    "conversion_rate": conversion_rate,
                    "funding_rate": funding_rate,
                    "business_unit": unit,
                    "revenue": round(revenue, 2),
                    "transactions": transactions,
                    "acquisition_channel": channel,
                    "feature_live": week >= FEATURE_LAUNCH_DATE,
                    "policy_active": week >= POLICY_CHANGE_DATE
                })
    return pd.DataFrame(records)


if __name__ == "__main__":
    # Configure your BigQuery details
    PROJECT_ID = "your-gcp-project"
    DATASET = "demo_fpna"
    TABLE = "weekly_performance"

    # Generate & save to CSV
    df = generate_weekly_performance(datetime.today() - timedelta(weeks=8), weeks=8)
    df.to_csv("demo_data.csv", index=False)
    print("Generated enriched demo_data.csv")

    # Optional: Load to BigQuery
    client = bigquery.Client(project=PROJECT_ID)
    table_ref = client.dataset(DATASET).table(TABLE)
    job = client.load_table_from_dataframe(df, table_ref)
    job.result()
    print(f"Loaded {job.output_rows} rows into {DATASET}.{TABLE}")
