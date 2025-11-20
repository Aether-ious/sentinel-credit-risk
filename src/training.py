import mlflow
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def objective(trial, X, y):
    """Optuna Objective for AutoML"""
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'objective': 'binary:logistic'
    }
    
    model = xgb.XGBClassifier(**param)
    # Use cross-validation here in real code
    model.fit(X, y)
    return roc_auc_score(y, model.predict_proba(X)[:, 1])

def train_model(mode="manual"):
    mlflow.set_experiment("Credit_Risk_Sentinel")
    
    with mlflow.start_run():
        if mode == "manual":
            # Manual Configuration
            params = {"max_depth": 4, "n_estimators": 100}
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            mlflow.log_params(params)
            
        elif mode == "auto":
            # AutoML with Optuna
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20)
            best_params = study.best_params
            model = xgb.XGBClassifier(**best_params)
            model.fit(X_train, y_train)
            mlflow.log_params(best_params)

        # Log Explainability (SHAP)
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)
        mlflow.shap.log_explainer(explainer, "shap_explainer")
        
        # Log Model
        mlflow.xgboost.log_model(model, "model")