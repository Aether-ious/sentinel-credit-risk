# Model Card: Sentinel Credit Risk Engine

## Model Details
- **Developer:** [Your Name]
- **Model Date:** November 2025
- **Model Version:** 1.0 (AutoML Optimized)
- **Type:** XGBoost Classifier (Binary Classification)

## Intended Use
- **Primary Use Case:** Assessing creditworthiness of loan applicants.
- **Target Users:** Loan Officers, Automated Underwriting Systems.
- **Out of Scope:** Corporate loans, Mortgage loans > $1M.

## Training Data
- **Source:** German Credit Dataset (UCI Machine Learning Repository).
- **Size:** 1000 records.
- **Features:** 20 features including Status, Duration, Credit History, Age.
- **Preprocessing:** Weight of Evidence (WoE) encoding for all categorical variables.

## Performance
- **Metric:** AUC (Area Under Curve)
- **Score:** ~0.81
- **Threshold:** 0.5 (Probability > 0.5 = High Risk)

## Ethical Considerations
- **Bias Risk:** Dataset contains 'Age' and 'Foreign Worker' status. Fairness analysis recommended before full deployment.
- **Mitigation:** WoE transformation smooths out some categorical bias, but monitoring is required.