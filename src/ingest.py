import pandas as pd
import numpy as np
import os
import requests
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "german_credit_data.csv"

# URL for the raw dataset (UCI Machine Learning Repository)
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Column Names (The raw file has no headers, so we define them based on documentation)
COLUMNS = [
    "checkin_acc", "duration", "credit_history", "purpose", "amount",
    "saving_acc", "present_emp_since", "installment_rate", "personal_status",
    "other_debtors", "residing_since", "property", "age", "inst_plans",
    "housing", "num_credits", "job", "dependents", "telephone", "foreign_worker",
    "status"
]

def ingest_data():
    print("🚀 Starting Data Ingestion...")
    
    # 1. Download
    if not RAW_DATA_PATH.exists():
        print(f"   Downloading from {DATA_URL}...")
        response = requests.get(DATA_URL)
        with open(RAW_DATA_PATH, 'wb') as f:
            f.write(response.content)
    else:
        print("   File already exists. Skipping download.")

    # 2. Load and Process
    # The file is space-separated
    df = pd.read_csv(RAW_DATA_PATH, sep=' ', names=COLUMNS)
    
    # 3. Target Mapping
    # In this dataset: 1 = Good, 2 = Bad. 
    # We convert to standard Machine Learning targets: 0 = Good, 1 = Bad (Default)
    df['target'] = df['status'].map({1: 0, 2: 1})
    df = df.drop(columns=['status'])
    
    # 4. Save Clean Version
    df.to_csv(RAW_DATA_PATH, index=False)
    
    print(f"✅ Ingestion Complete. Raw Data Shape: {df.shape}")
    print(f"   Target Distribution:\n{df['target'].value_counts()}")
    print(f"   Saved to: {RAW_DATA_PATH}")

if __name__ == "__main__":
    ingest_data()