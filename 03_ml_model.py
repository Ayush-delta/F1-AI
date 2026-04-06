# ============================================================
#  F1 AI/ML Project — Phase 3: GP Winner Prediction Model
#  Refactored Pipeline:
#  - Uses pre-race features to prevent Data Leakage
#  - Implements Time-Series Cross Validation
#  - Trains 3 models (Position, Podium, Top10)
#  - Predicts 2026 Season required by the UI
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import mlflow

# Import from our new src/ package
from src.data import load_season_results
from src.features import build_features
from src.model import train_3_model_pipeline

def main():
    print("--- F1 ML Pipeline ---")
    
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    
    # 1. Load Data
    # To prevent excessive API calls during execution, we load 2023 season data.
    # We will use this to train and then simulate a 2026 dataset.
    print("Fetching historical data...")
    df_2023 = load_season_results(2023, max_races=10) # Using 10 races to make it fast
    
    if df_2023.empty:
        print("Failed to pull data. Ensure you have internet access and fastf1 is working.")
        return

    # 2. Build Features
    print("\nEngineering features (pre-race only)...")
    df_features = build_features(df_2023)
    
    # Encode 'Team'
    le = LabelEncoder()
    df_features['TeamEncoded'] = le.fit_transform(df_features['Team'].astype(str))
    
    feature_cols = ['GridPosition', 'DriverAvg3Pos', 'TeamPoints3', 'DNF_Risk', 'TeamEncoded']
    
    # Prepare labels for the 3 target concepts
    y_pos = df_features['FinishPosition']
    y_podium = (df_features['FinishPosition'] <= 3).astype(int)
    y_top10 = (df_features['FinishPosition'] <= 10).astype(int)
    
    X = df_features[feature_cols]

    # Save to CSV for inspection
    df_features.to_csv("data/processed/ml_features_all_races.csv", index=False)
    
    # 3. Train Models
    print("\nTraining Models...")
    models, metrics = train_3_model_pipeline(X, y_pos, y_podium, y_top10, feature_cols)
    
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
        
    # Save the models
    joblib.dump(models['position'], 'data/models/model_position.pkl')
    joblib.dump(models['podium'],   'data/models/model_podium.pkl')
    joblib.dump(models['top10'],    'data/models/model_top10.pkl')
    joblib.dump(le,                 'data/models/team_encoder.pkl')
    joblib.dump(feature_cols,       'data/models/feature_cols.pkl')
    print("\nModels saved to data/models/")

    # SHAP Feature Importance on Podium Model
    explainer = shap.TreeExplainer(models['podium'])
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_cols, plot_type="bar", show=False)
    plt.title("Feature Importance - SHAP")
    plt.tight_layout()
    plt.savefig("data/processed/shap_feature_importance.png")
    
    # 4. Generate 2026 Mock Predictions
    print("\nSimulating 2026 Predictions for UI...")
    # We will simulate 3 races for the 2026 season based on top drivers
    
    mock_drivers = [
        {"Driver": "VER", "Team": "Red Bull", "GridPosition": 1, "DriverAvg3Pos": 1.5, "TeamPoints3": 100, "DNF_Risk": 0.05},
        {"Driver": "NOR", "Team": "McLaren",  "GridPosition": 2, "DriverAvg3Pos": 2.5, "TeamPoints3": 85,  "DNF_Risk": 0.1},
        {"Driver": "HAM", "Team": "Ferrari",  "GridPosition": 5, "DriverAvg3Pos": 4.0, "TeamPoints3": 60,  "DNF_Risk": 0.1},
        {"Driver": "LEC", "Team": "Ferrari",  "GridPosition": 3, "DriverAvg3Pos": 3.5, "TeamPoints3": 60,  "DNF_Risk": 0.15},
        {"Driver": "PIA", "Team": "McLaren",  "GridPosition": 4, "DriverAvg3Pos": 5.0, "TeamPoints3": 85,  "DNF_Risk": 0.1},
        {"Driver": "RUS", "Team": "Mercedes", "GridPosition": 6, "DriverAvg3Pos": 6.0, "TeamPoints3": 45,  "DNF_Risk": 0.15},
        {"Driver": "ANT", "Team": "Mercedes", "GridPosition": 7, "DriverAvg3Pos": 7.0, "TeamPoints3": 45,  "DNF_Risk": 0.2},
    ]
    
    # Add filler drivers up to 20
    fillers = []
    for i in range(8, 21):
        fillers.append({"Driver": f"DRV{i}", "Team": "Unknown", "GridPosition": i, "DriverAvg3Pos": 15.0, "TeamPoints3": 5, "DNF_Risk": 0.3})
        
    base_grid = pd.DataFrame(mock_drivers + fillers)
    
    RACES = ["Bahrain", "Saudi Arabia", "Australia", "Japan", "China"]
    all_preds = []
    
    for round_num, race_name in enumerate(RACES, 1):
        grid = base_grid.copy()
        grid["Race"] = race_name
        grid["Round"] = round_num
        
        # Add a tiny bit of random noise to grid position for variety across races
        grid['GridPosition'] = grid['GridPosition'] + np.random.randint(-2, 3, size=len(grid))
        grid['GridPosition'] = grid['GridPosition'].clip(1, 20)
        
        grid['TeamEncoded'] = grid['Team'].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        
        X_pred = grid[feature_cols]
        
        # Predict
        grid['PredPos'] = models['position'].predict(X_pred)
        grid['PodiumProb'] = models['podium'].predict_proba(X_pred)[:, 1]
        grid['Top10Prob'] = models['top10'].predict_proba(X_pred)[:, 1]
        
        # Sort by PredPos to assign rank
        grid = grid.sort_values('PredPos').reset_index(drop=True)
        grid['PredRank'] = range(1, len(grid) + 1)
        
        all_preds.append(grid)
        
    pred_df = pd.concat(all_preds, ignore_index=True)
    pred_df.to_csv("data/processed/2026_season_predictions.csv", index=False)
    print("2026 Predictions saved to data/processed/2026_season_predictions.csv")
    print("\nPipeline execution complete! You can now run `streamlit run 04_ai_race_engineer.py`")

if __name__ == "__main__":
    main()
