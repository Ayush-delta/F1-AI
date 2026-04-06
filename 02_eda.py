# ============================================================
#  F1 AI/ML Project — Phase 2: EDA & Feature Analysis
#  Run after Phase 1 — uses saved CSVs from data/processed/
#  Focus: tyre degradation, constructor form, feature correlations
# ============================================================

# ── Cell 1: Imports ──────────────────────────────────────────
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import fastf1
import warnings
warnings.filterwarnings("ignore")

fastf1.Cache.enable_cache("data/raw/fastf1_cache")

print("Imports ready!")


# ── Cell 2: Load saved Monaco data ───────────────────────────
laps    = pd.read_csv("data/processed/monaco_2023_all_laps.csv")
results = pd.read_csv("data/processed/monaco_2023_results.csv")

# LapTime was saved as a string — convert back to seconds
laps["LapTimeSec"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()

print(f"Laps loaded: {len(laps)} rows")
print(f"Results loaded: {len(results)} rows")
print(f"\nDrivers in dataset: {sorted(laps['Driver'].unique())}")


# ── Cell 3: Tyre degradation — lap time vs tyre age ──────────
# Only clean laps (no pit in/out laps, no safety car outliers)
clean = laps[
    laps["LapTimeSec"].between(72, 100) &
    laps["PitInTime"].isna() &
    laps["PitOutTime"].isna()
].copy()

# Focus on the two main dry compounds used at Monaco 2023
deg_data = clean[clean["Compound"].isin(["MEDIUM", "HARD"])].copy()

# Average lap time per tyre age per compound
deg_avg = (
    deg_data.groupby(["Compound", "TyreLife"])["LapTimeSec"]
    .mean()
    .reset_index()
)

fig1 = px.line(
    deg_avg,
    x="TyreLife",
    y="LapTimeSec",
    color="Compound",
    color_discrete_map={"MEDIUM": "#FFC906", "HARD": "#ABABAB"},
    markers=True,
    title="2023 Monaco GP — Tyre Degradation (avg lap time vs tyre age)",
    labels={"TyreLife": "Tyre Age (laps)", "LapTimeSec": "Avg Lap Time (sec)"},
)
fig1.update_layout(hovermode="x unified")
fig1.show()
fig1.write_html("data/processed/monaco_tyre_degradation.html")
print("Tyre degradation chart saved.")


# ── Cell 4: Load full 2023 season results via FastF1 ─────────
# We'll pull results from 5 key races for constructor comparison
# (pulling all 23 would take too long — 5 is enough for the pattern)

races_to_load = ["Bahrain", "Monaco", "Silverstone", "Monza", "Abu Dhabi"]
all_results = []

print("\nLoading 2023 season results (5 races)...")
for race_name in races_to_load:
    try:
        s = fastf1.get_session(2023, race_name, "R")
        s.load(telemetry=False, weather=False, messages=False)
        r = s.results[["FullName", "TeamName", "Position", "Points"]].copy()
        r["Race"] = race_name
        all_results.append(r)
        print(f"  Loaded: {race_name}")
    except Exception as e:
        print(f"  Skipped {race_name}: {e}")

season_df = pd.concat(all_results, ignore_index=True)
season_df["Position"] = pd.to_numeric(season_df["Position"], errors="coerce")
season_df["Points"]   = pd.to_numeric(season_df["Points"],   errors="coerce")

season_df.to_csv("data/processed/season_2023_5races.csv", index=False)
print(f"\nSeason data saved: {len(season_df)} rows")


# ── Cell 5: Constructor points comparison ─────────────────────
constructor_pts = (
    season_df.groupby(["Race", "TeamName"])["Points"]
    .sum()
    .reset_index()
)

# Order races chronologically
race_order = races_to_load
constructor_pts["Race"] = pd.Categorical(
    constructor_pts["Race"], categories=race_order, ordered=True
)
constructor_pts = constructor_pts.sort_values("Race")

# Only show top 5 constructors by total points
top5_teams = (
    constructor_pts.groupby("TeamName")["Points"]
    .sum()
    .nlargest(5)
    .index.tolist()
)
plot_data = constructor_pts[constructor_pts["TeamName"].isin(top5_teams)]

fig2 = px.line(
    plot_data,
    x="Race",
    y="Points",
    color="TeamName",
    markers=True,
    title="2023 F1 Season — Constructor Points (Top 5, selected races)",
    labels={"Points": "Points per Race", "TeamName": "Constructor"},
)
fig2.update_layout(hovermode="x unified")
fig2.show()
fig2.write_html("data/processed/constructor_comparison.html")
print("Constructor comparison chart saved.")


# ── Cell 6: Feature engineering for ML ───────────────────────
# Build the feature table that Phase 3 (ML model) will train on

print("\nBuilding ML feature table...")

# Use Monaco laps — merge lap data with finishing position
driver_finish = results[["Abbreviation", "Position", "GridPosition", "TeamName"]].copy()
driver_finish.columns = ["Driver", "FinishPosition", "GridPosition", "Team"]
driver_finish["FinishPosition"] = pd.to_numeric(driver_finish["FinishPosition"], errors="coerce")
driver_finish["GridPosition"]   = pd.to_numeric(driver_finish["GridPosition"],   errors="coerce")

# Per-driver stats from lap data
driver_lap_stats = (
    clean.groupby("Driver")["LapTimeSec"]
    .agg(
        AvgLapTime="mean",
        BestLapTime="min",
        LapTimeStd="std",       # consistency — lower = more consistent
        TotalLaps="count",
    )
    .reset_index()
)

# Pit stop count per driver
pit_counts = (
    laps[laps["PitInTime"].notna()]
    .groupby("Driver")
    .size()
    .reset_index(name="PitStops")
)

# Merge everything into one feature table
features = (
    driver_finish
    .merge(driver_lap_stats, on="Driver", how="left")
    .merge(pit_counts,       on="Driver", how="left")
)
features["PitStops"] = features["PitStops"].fillna(0).astype(int)

# Target variable: did they finish in the points? (top 10 = 1, else 0)
features["InPoints"] = (features["FinishPosition"] <= 10).astype(int)

# Grid vs finish delta (positive = gained places, negative = lost)
features["PositionDelta"] = features["GridPosition"] - features["FinishPosition"]

print(features[[
    "Driver", "Team", "GridPosition", "FinishPosition",
    "PositionDelta", "AvgLapTime", "BestLapTime",
    "LapTimeStd", "PitStops", "InPoints"
]].to_string(index=False))

features.to_csv("data/processed/ml_features_monaco_2023.csv", index=False)
print("\nML feature table saved to data/processed/ml_features_monaco_2023.csv")


# ── Cell 7: Correlation heatmap ───────────────────────────────
import plotly.figure_factory as ff
import numpy as np

numeric_cols = ["GridPosition", "FinishPosition", "AvgLapTime",
                "BestLapTime", "LapTimeStd", "PitStops", "PositionDelta"]

corr = features[numeric_cols].corr().round(2)

fig3 = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="2023 Monaco GP — Feature Correlation Heatmap",
    aspect="auto",
)
fig3.show()
fig3.write_html("data/processed/feature_correlation_heatmap.html")
print("Correlation heatmap saved.")


# ── Cell 8: Key EDA findings ──────────────────────────────────
print("\n" + "="*55)
print("KEY FINDINGS — write these in your README")
print("="*55)

grid_finish_corr = features["GridPosition"].corr(features["FinishPosition"])
print(f"\n1. Grid vs Finish correlation: {grid_finish_corr:.2f}")
print("   (1.0 = perfect, means qualifying position decides everything at Monaco)")

avg_pit_stops = features["PitStops"].mean()
print(f"\n2. Average pit stops per driver: {avg_pit_stops:.1f}")

best_consistency = features.nsmallest(3, "LapTimeStd")[["Driver", "LapTimeStd"]]
print(f"\n3. Most consistent drivers (lowest lap time std dev):")
print(best_consistency.to_string(index=False))

print("\n" + "="*55)
print("Phase 2 complete!")
print("Files saved in data/processed/:")
print("  - monaco_tyre_degradation.html")
print("  - constructor_comparison.html")
print("  - feature_correlation_heatmap.html")
print("  - ml_features_monaco_2023.csv  <-- Phase 3 input")
print("="*55)
