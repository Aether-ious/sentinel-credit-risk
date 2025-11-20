from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import mlflow.xgboost
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI(title="Sentinel Credit Risk API", version="1.0.0")

# Global Model Variable
model = None

class CreditApplication(BaseModel):
    # Define all inputs that affect the score
    checkin_acc: str = "A11"
    duration: int = 6
    credit_history: str = "A34"
    amount: int = 1169
    age: int = 67

@app.on_event("startup")
def startup_event():
    global model
    print("🚀 API Starting up...")
    try:
        experiment = mlflow.get_experiment_by_name("Sentinel_Credit_Risk_Engine")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.auc DESC"],
            max_results=1
        )
        best_run_id = runs.iloc[0].run_id
        print(f"   Loading Best Model: {best_run_id}")
        model = mlflow.xgboost.load_model(f"runs:/{best_run_id}/model")
    except Exception as e:
        print(f"❌ Critical Error: Could not load model. {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/score")
def score_application(application: CreditApplication):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 1. Get Feature Names from the Model
        if hasattr(model, "feature_names"):
            cols = model.feature_names
        else:
            cols = model.feature_names_in_
            
        # 2. Create the DataFrame (Initialize with 0s)
        df_input = pd.DataFrame(0, index=[0], columns=cols)
        
        # 3. INJECT USER INPUTS (The Fix)
        # We explicitly map the API request fields to the Model columns
        # Note: In a real system, we would also transform "A11" to a WoE value here.
        # For now, we pass the numeric raw values to see the score change.
        
        if 'duration' in cols: 
            df_input['duration'] = application.duration
        if 'amount' in cols: 
            df_input['amount'] = application.amount
        if 'age' in cols: 
            df_input['age'] = application.age
            
        # 4. Predict
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(df_input)
            prob = float(model.predict(dmatrix)[0])
        else:
            prob = float(model.predict_proba(df_input)[:, 1][0])
        
        risk_label = "High Risk" if prob > 0.5 else "Low Risk"
        
        return {
            "probability_of_default": round(prob, 4),
            "risk_label": risk_label,
            "inputs_received": {
                "amount": application.amount,
                "duration": application.duration
            }
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))