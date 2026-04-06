import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds Machine Learning features from raw results data.
    Fixes Data Leakage by ONLY using information available BEFORE the race
    (like GridPosition and Historical Form), avoiding in-race lap times.
    """
    if df.empty:
        return df

    # We need to sort chronologically to compute rolling stats without predicting the past from the future
    df = df.sort_values(['Year', 'Round']).reset_index(drop=True)
    
    # 1. Driver's Rolling 3-Race Average Finish
    df['DriverAvg3Pos'] = df.groupby('Driver')['FinishPosition'] \
                            .transform(lambda x: x.shift().rolling(window=3, min_periods=1).mean())
    df['DriverAvg3Pos'] = df['DriverAvg3Pos'].fillna(10.0) # Fill NaN with average
    
    # 2. Team's Rolling 3-Race Average Points
    df['TeamPoints3'] = df.groupby('Team')['Points'] \
                          .transform(lambda x: x.shift().rolling(window=3, min_periods=1).sum())
    df['TeamPoints3'] = df['TeamPoints3'].fillna(0.0)

    # 3. DNF Risk (Did Not Finish - usually indicates reliability issues. Position missing means DNF)
    # We assign DNF to Position > 20 or NaN
    df['IsDNF'] = (df['FinishPosition'].isna() | (df['FinishPosition'] > 20)).astype(int)
    df['DNF_Risk'] = df.groupby('Driver')['IsDNF'] \
                       .transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
    df['DNF_Risk'] = df['DNF_Risk'].fillna(0.1)
    
    # Impute missing grid positions (e.g. pitlane starts)
    df['GridPosition'] = df['GridPosition'].replace(0, 20).fillna(20)
    
    # Fill actual FinishPosition for training targets (impute DNF as 20)
    df['FinishPosition'] = df['FinishPosition'].fillna(20)
    
    # Encode Team Names (simple Frequency encoding or Label encoding)
    # We'll use frequency encoding so the model handles new teams better 
    # but we can also use standard LabelEncoder in the final pipeline
    
    return df
