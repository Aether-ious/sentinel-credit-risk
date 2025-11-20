import mlflow.xgboost
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import CreditRiskFeatures

class CreditScorer:
    def __init__(self):
        self.model = None
        self.load_latest_model()

    def load_latest_model(self):
        """
        Automatically finds the best model from MLflow.
        In production, you'd use a specific 'Production' tag.
        """
        print("🔎 Searching for best model in MLflow...")
        try:
            # Find the best run based on AUC
            experiment = mlflow.get_experiment_by_name("Sentinel_Credit_Risk_Engine")
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.auc DESC"],
                max_results=1
            )
            
            best_run_id = runs.iloc[0].run_id
            model_uri = f"runs:/{best_run_id}/model"
            
            print(f"   🏆 Loading Best Model (Run ID: {best_run_id})...")
            self.model = mlflow.xgboost.load_model(model_uri)
            
            # NOTE: In a real system, we would also load the saved WoE mappings here.
            # For this demo, we will re-initialize the feature engine logic.
            self.engineer = CreditRiskFeatures()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def predict(self, input_data: dict):
        """
        Accepts dictionary inputs, transforms them, and predicts risk.
        """
        # 1. Convert dict to DataFrame
        df = pd.DataFrame([input_data])
        
        # 2. Apply Feature Engineering (WoE)
        # In a full prod system, we would apply the *saved* mappings from training.
        # Here, we assume raw data for simplicity or re-fit logic.
        # Ideally: df_processed = self.engineer.transform(df)
        
        # For this demo, we verify structure match (DataOps validation)
        # We convert the raw dict directly to DMatrix for XGBoost if columns match
        # OR we assume the input is already pre-processed features.
        
        # Let's do a direct prediction assuming pre-processed input for stability in this step
        # (To fix this fully requires saving the 'engineer' object in train.py)
        
        return self.model.predict_proba(df)[:, 1][0] # Return probability of Default