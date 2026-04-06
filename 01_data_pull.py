# ============================================================
#  F1 AI/ML Project — Phase 1: Data Pull
#  Run this as a Jupyter notebook (.ipynb) or plain script
#  Focus race: 2023 Monaco Grand Prix
# ============================================================

# ── Cell 1: Imports ──────────────────────────────────────────
import fastf1
import fastf1.plotting
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# Cache speeds up repeat loads — FastF1 saves session data locally
fastf1.Cache.enable_cache("data/raw/fastf1_cache")

print("All packages loaded successfully!")
print(f"FastF1 version: {fastf1.__version__}")
print(f"Pandas version:  {pd.__version__}")


# ── Cell 2: Load the 2023 Monaco GP ──────────────────────────
# Arguments: year, circuit name, session type
# Session types: 'FP1' 'FP2' 'FP3' 'Q' (qualifying) 'R' (race)
session = fastf1.get_session(2023, "Monaco", "R")
session.load(telemetry=False, weather=False, messages=False)

print(f"\nSession loaded: {session.event['EventName']} {session.event.year}")
print(f"Date: {session.date}")
print(f"Circuit: {session.event['Location']}, {session.event['Country']}")


# ── Cell 3: Explore the race results ────────────────────────
results = session.results   # pandas DataFrame

print("\n--- Race Results ---")
print(results[["Position", "FullName", "TeamName", "Time", "Points"]].head(10))


# ── Cell 4: Lap times for all drivers ────────────────────────
laps = session.laps   # every lap, every driver

print(f"\nTotal laps in dataset: {len(laps)}")
print(f"Columns: {list(laps.columns)}")
print("\nSample rows:")
print(laps[["Driver", "LapNumber", "LapTime", "Compound", "TyreLife", "PitInTime", "PitOutTime"]].head(10))


# ── Cell 5: Filter to top 3 drivers ──────────────────────────
top3 = results.iloc[:3]["Abbreviation"].tolist()
print(f"\nTop 3 drivers: {top3}")

laps_top3 = laps[laps["Driver"].isin(top3)].copy()

# Convert LapTime (timedelta) to seconds for plotting
laps_top3["LapTimeSec"] = laps_top3["LapTime"].dt.total_seconds()

# Drop outliers (safety car laps, in/out laps) — anything > 105s at Monaco
clean_laps = laps_top3[laps_top3["LapTimeSec"].between(70, 105)].copy()

print(f"Clean laps for top 3: {len(clean_laps)}")


# ── Cell 6: Plot lap time comparison ─────────────────────────
fig = px.line(
    clean_laps,
    x="LapNumber",
    y="LapTimeSec",
    color="Driver",
    title="2023 Monaco GP — Lap Time Comparison (Top 3)",
    labels={"LapTimeSec": "Lap Time (seconds)", "LapNumber": "Lap"},
    markers=True,
)
fig.update_layout(hovermode="x unified")
fig.show()

# Save to file so you can embed it in your README
fig.write_html("data/processed/monaco_lap_times.html")
print("Chart saved to data/processed/monaco_lap_times.html")


# ── Cell 7: Tyre strategy overview ───────────────────────────
# Which compound did each driver use on each lap?
tyre_data = laps[["Driver", "LapNumber", "Compound", "TyreLife"]].copy()

# Colour map for compounds
compound_colors = {
    "SOFT": "#E8002D",
    "MEDIUM": "#FFF200",
    "HARD": "#EBEBEB",
    "INTERMEDIATE": "#39B54A",
    "WET": "#0067FF",
}

# Plot stint bars — one row per driver
fig2 = px.scatter(
    tyre_data,
    x="LapNumber",
    y="Driver",
    color="Compound",
    color_discrete_map=compound_colors,
    title="2023 Monaco GP — Tyre Strategy",
    labels={"LapNumber": "Lap", "Driver": "Driver"},
    size_max=6,
)
fig2.update_traces(marker=dict(size=8, symbol="square"))
fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
fig2.show()

fig2.write_html("data/processed/monaco_tyre_strategy.html")
print("Tyre strategy chart saved.")


# ── Cell 8: Save clean data for Phase 2 & 3 ──────────────────
laps_top3.to_csv("data/processed/monaco_2023_laps_top3.csv", index=False)
results.to_csv("data/processed/monaco_2023_results.csv", index=False)
laps.to_csv("data/processed/monaco_2023_all_laps.csv", index=False)

print("\nData saved:")
print("  data/processed/monaco_2023_laps_top3.csv")
print("  data/processed/monaco_2023_results.csv")
print("  data/processed/monaco_2023_all_laps.csv")
print("\nPhase 1 complete! Open the HTML files in your browser to see the charts.")
print("Commit everything to GitHub now.")
