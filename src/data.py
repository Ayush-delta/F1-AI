import fastf1
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(os.path.join("data", "raw", "fastf1_cache"))

def load_season_results(year: int, max_races: int = None) -> pd.DataFrame:
    """
    Loads all race results and qualifying grid positions for a given year.
    To speed up execution, limits to `max_races` if provided.
    """
    schedule = fastf1.get_event_schedule(year)
    # Filter to only actual races (not testing)
    races = schedule[schedule['EventFormat'] != 'testing']
    
    if max_races:
        races = races.head(max_races)
        
    all_results = []
    
    print(f"Loading {len(races)} races for {year}...")
    for idx, event in races.iterrows():
        try:
            # We load the "Race" session
            session = fastf1.get_session(year, event['EventName'], 'R')
            session.load(telemetry=False, weather=False, messages=False)
            
            results = session.results.copy()
            if results.empty:
                continue
                
            # Extract relevant columns
            df = results[['Abbreviation', 'FullName', 'TeamName', 'GridPosition', 'Position', 'Points']].copy()
            df.columns = ['Driver', 'FullName', 'Team', 'GridPosition', 'FinishPosition', 'Points']
            
            # Clean types
            df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce')
            df['FinishPosition'] = pd.to_numeric(df['FinishPosition'], errors='coerce')
            df['Points'] = pd.to_numeric(df['Points'], errors='coerce')
            
            df['Race'] = event['EventName']
            df['Round'] = event['RoundNumber']
            df['Year'] = year
            
            all_results.append(df)
            print(f"  ✓ Loaded {event['EventName']}")
        except Exception as e:
            print(f"  ✗ Skipped {event['EventName']}: {e}")
            
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()
