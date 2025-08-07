"""
Generates synthetic weekly funnel data and loads it into BigQuery for demo purposes.
Adjust `PROJECT_ID`, `DATASET`, and `TABLE` as needed.
"""
from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_weekly_funnel(start_date, weeks=8):
    records = []
    for i in range(weeks):
        week_start = start_date + timedelta(weeks=i)
        total_visits = np.random.randint(5000, 10000)
        leads = int(total_visits * np.random.uniform(0.1, 0.2))
        funded = int(leads * np.random.uniform(0.2, 0.5))
        records.append({
            "week_start": week_start.strftime("%Y-%m-%d"),
            "total_visits": total_visits,
            "leads": leads,
            "funded": funded
        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    # Configure your BigQuery details
    PROJECT_ID = "your-gcp-project"
    DATASET = "demo_fpna"
    TABLE = "weekly_funnel"

    # Generate data
    df = generate_weekly_funnel(datetime.today() - timedelta(weeks=8), weeks=8)
    df.to_csv("demo_data.csv", index=False)
    print("Generated demo_data.csv")

    # Load to BigQuery
    client = bigquery.Client(project=PROJECT_ID)
    dataset_ref = client.dataset(DATASET)
    table_ref = dataset_ref.table(TABLE)
    job = client.load_table_from_dataframe(df, table_ref)
    job.result()
    print(f"Loaded {job.output_rows} rows into {DATASET}.{TABLE}")
