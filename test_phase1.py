import pandas as pd
from src.features import CreditRiskFeatures

# Load the raw data we ingested
df = pd.read_csv("data/raw/german_credit_data.csv")

# Initialize the Feature Engineer
engineer = CreditRiskFeatures()

# Transform the data
df_woe = engineer.fit_transform(df, target_col='target')

print("\n✅ Phase 1 Success!")
print("Original 'checkin_acc' (categorical):", df['checkin_acc'].iloc[0])
print("Transformed 'checkin_acc' (WoE):", df_woe['checkin_acc'].iloc[0])