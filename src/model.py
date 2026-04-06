import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
import mlflow

def train_3_model_pipeline(X, y_pos, y_podium, y_top10, feature_cols):
    """
    Trains the 3 models using TimeSeriesSplit to prevent lookahead bias.
    """
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Common params
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'random_state': 42
    }
    
    models = {}
    metrics = {}
    
    # Helper to train and evaluate
    def train_and_eval(y_target, is_classifier=False):
        model_class = xgb.XGBClassifier if is_classifier else xgb.XGBRegressor
        model = model_class(**xgb_params)
        
        cv_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_target.iloc[train_idx], y_target.iloc[test_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            if is_classifier:
                score = accuracy_score(y_test, preds)
            else:
                score = -mean_squared_error(y_test, preds, squared=False) # RMSE
            cv_scores.append(score)
            
        # Train final on everything
        model.fit(X, y_target)
        return model, np.mean(cv_scores)

    print("Training Model A: Position Regressor...")
    pos_model, pos_score = train_and_eval(y_pos, is_classifier=False)
    models['position'] = pos_model
    metrics['pos_cv_rmse'] = -pos_score
    
    print("Training Model B: Podium Classifier...")
    pod_model, pod_score = train_and_eval(y_podium, is_classifier=True)
    models['podium'] = pod_model
    metrics['podium_cv_acc'] = pod_score
    
    print("Training Model C: Top 10 Classifier...")
    top10_model, top10_score = train_and_eval(y_top10, is_classifier=True)
    models['top10'] = top10_model
    metrics['top10_cv_acc'] = top10_score
    
    return models, metrics

