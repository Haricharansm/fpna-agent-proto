# data_generator.py
"""
Generates synthetic weekly funnel and performance data for demo purposes.
Includes merchant segment, business unit revenue, transaction volume, and acquisition channel.
Adjust `PROJECT_ID`, `DATASET`, and `TABLE` as needed.
"""
from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SEGMENTS = ["Retail", "Wholesale", "SMB", "Enterprise"]
BUSINESS_UNITS = ["Payments", "Capital", "Subscriptions"]
CHANNELS = ["Organic", "Paid Search", "Referral", "Email"]
FEATURE_LAUNCH_DATE = datetime(2024, 6, 15)
POLICY_CHANGE_DATE = datetime(2024, 7, 1)

def generate_weekly_performance(start_date, weeks=8):
    records = []
    for i in range(weeks):
        week_start = start_date + timedelta(weeks=i)
        for segment in SEGMENTS:
            for unit in BUSINESS_UNITS:
                total_visits = np.random.randint(2000, 10000)
                leads = int(total_visits * np.random.uniform(0.1, 0.25))
                funded = int(leads * np.random.uniform(0.2, 0.6))
                revenue = funded * np.random.uniform(1000, 5000)
                transactions = np.random.randint(100, 1000)
                channel = np.random.choice(CHANNELS)
                records.append({
                    "week_start": week_start.strftime("%Y-%m-%d"),
                    "merchant_segment": segment,
                    "business_unit": unit,
                    "total_visits": total_visits,
                    "leads": leads,
                    "funded": funded,
                    "revenue": round(revenue, 2),
                    "transactions": transactions,
                    "acquisition_channel": channel,
                    "feature_live": week_start >= FEATURE_LAUNCH_DATE,
                    "policy_active": week_start >= POLICY_CHANGE_DATE
                })
    return pd.DataFrame(records)

if __name__ == "__main__":
    # Configure your BigQuery details
    PROJECT_ID = "your-gcp-project"
    DATASET = "demo_fpna"
    TABLE = "weekly_performance"

    # Generate data
    df = generate_weekly_performance(datetime.today() - timedelta(weeks=8), weeks=8)
    df.to_csv("demo_data.csv", index=False)
    print("Generated demo_data.csv with enriched performance metrics")

    # Load to BigQuery
    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = client.dataset(DATASET)
    table_ref = dataset_ref.table(TABLE)
    job = client.load_table_from_dataframe(df, table_ref)
    job.result()
    print(f"Loaded {job.output_rows} rows into {DATASET}.{TABLE}")
