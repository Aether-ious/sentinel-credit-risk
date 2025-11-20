import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import optuna
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

# Add project root to system path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import CreditRiskFeatures

# Config
DATA_PATH = "data/raw/german_credit_data.csv"
MLFLOW_EXPERIMENT_NAME = "Sentinel_Credit_Risk_Engine"

def load_and_process_data():
    """Loads raw data and applies WoE transformation."""
    print("⏳ Loading and processing data...")
    df = pd.read_csv(DATA_PATH)
    
    # Initialize your Feature Engine from Phase 1
    engineer = CreditRiskFeatures()
    df_processed = engineer.fit_transform(df, target_col='target')
    
    # Split
    X = df_processed.drop(columns=['target'])
    y = df_processed['target']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def eval_metrics(actual, pred_prob, pred_label):
    """Computes standard Credit Risk metrics."""
    auc = roc_auc_score(actual, pred_prob)
    acc = accuracy_score(actual, pred_label)
    return auc, acc

def objective(trial, X_train, y_train, X_test, y_test):
    """
    Optuna Objective Function:
    The AI suggests parameters, we train, and return the score.
    """
    # 1. Suggest Hyperparameters
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 1, 9),
        'eta': trial.suggest_float('eta', 0.01, 0.3),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    }

    # 2. Train Model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # We use a nested MLflow run for each trial to keep things organized
    with mlflow.start_run(nested=True):
        model = xgb.train(param, dtrain)
        
        # 3. Evaluate
        preds_prob = model.predict(dtest)
        preds_label = [1 if x > 0.5 else 0 for x in preds_prob]
        auc, acc = eval_metrics(y_test, preds_prob, preds_label)
        
        # 4. Log to MLflow
        mlflow.log_params(param)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("accuracy", acc)
        
        # Tell Optuna how good this model was
        return auc

def main(mode="manual"):
    # 1. Setup Data & Experiment
    X_train, X_test, y_train, y_test = load_and_process_data()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    print(f"🚀 Starting Training in [{mode.upper()}] mode...")

    if mode == "manual":
        # --- MANUAL TRAINING ---
        with mlflow.start_run(run_name="Manual_XGBoost"):
            # Hardcoded params (The "Old Way")
            params = {
                "max_depth": 4,
                "eta": 0.1,
                "objective": "binary:logistic",
                "eval_metric": "auc"
            }
            
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            
            model = xgb.train(params, dtrain, num_boost_round=100)
            
            # Evaluate
            preds_prob = model.predict(dtest)
            preds_label = [1 if x > 0.5 else 0 for x in preds_prob]
            auc, acc = eval_metrics(y_test, preds_prob, preds_label)
            
            print(f"   ✅ Manual Run Results: AUC={auc:.4f}, Accuracy={acc:.4f}")
            
            # Log everything
            mlflow.log_params(params)
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("accuracy", acc)
            mlflow.xgboost.log_model(model, "model")
            print("   💾 Model saved to MLflow.")

    elif mode == "auto":
        # --- AUTO ML (OPTUNA) ---
        with mlflow.start_run(run_name="Optuna_Optimization_Parent"):
            print("   🤖 Tuning Hyperparameters... (This runs multiple trials)")
            study = optuna.create_study(direction="maximize")
            
            # Lambda creates a function that only needs 'trial'
            study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=10)
            
            print("   🏆 Best Params:", study.best_params)
            print("   ⭐️ Best AUC:", study.best_value)
            
            # Log the best parameters to the parent run
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_auc", study.best_value)

if __name__ == "__main__":
    # Default to manual, but you can change this
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "manual"
    main(mode)