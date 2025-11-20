import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
import os

# Paths
DATA_PATH = "data/raw/german_credit_data.csv"
REPORT_DIR = "docs/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

def generate_drift_report():
    print("🕵️‍♂️  Starting Data Drift Analysis...")
    
    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    
    # 2. Simulate "Reference" (Old) vs "Current" (New) data
    # In real life, 'current' would be yesterday's live API traffic
    reference_data = df.iloc[:500] # First half
    current_data = df.iloc[500:]   # Second half
    
    # 3. Generate Visual Report (The Dashboard)
    print("   📊 Generating HTML Report...")
    report = Report(metrics=[
        DataDriftPreset(), 
        TargetDriftPreset()
    ])
    
    report.run(reference_data=reference_data, current_data=current_data)
    
    report_path = f"{REPORT_DIR}/data_drift_report.html"
    report.save_html(report_path)
    
    # 4. Run Automated Tests (Pass/Fail)
    print("   🧪 Running Drift Tests...")
    tests = TestSuite(tests=[
        DataDriftTestPreset()
    ])
    tests.run(reference_data=reference_data, current_data=current_data)
    
    # Check if tests passed
    results = tests.as_dict()
    failed_tests = results['summary']['failed_tests']
    
    print(f"✅ Report saved to: {report_path}")
    print(f"   Tests Failed: {failed_tests}")
    
    if failed_tests > 0:
        print("⚠️  WARNING: Data Drift Detected!")
    else:
        print("🟢  System Healthy. No significant drift.")

if __name__ == "__main__":
    generate_drift_report()