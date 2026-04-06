import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from groq import Groq
from dotenv import load_dotenv
import fastf1
import json
import ast

load_dotenv()

st.set_page_config(
    page_title="F1 AI Race Engineer",
    page_icon="🏎️",
    layout="wide",
)

# Load Groq client 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Load data 
@st.cache_data
def load_data():
    data = {}
    files = {
        "laps":        "data/processed/monaco_2023_all_laps.csv",
        "results":     "data/processed/monaco_2023_results.csv",
        "features":    "data/processed/ml_features_all_races.csv",
        "predictions": "data/processed/2026_season_predictions.csv",
    }
    for key, path in files.items():
        if os.path.exists(path):
            data[key] = pd.read_csv(path)
    return data

@st.cache_resource
def load_models():
    models = {}
    paths = {
        "position": "data/models/model_position.pkl",
        "podium":   "data/models/model_podium.pkl",
        "top10":    "data/models/model_top10.pkl",
        "encoder":  "data/models/team_encoder.pkl",
        "features": "data/models/feature_cols.pkl",
    }
    for key, path in paths.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
    return models

data   = load_data()
models = load_models()

# Race context for AI 
def build_race_context():
    ctx = ["=== 2026 Season Predictions Data ===\n"]
    if "predictions" in data:
        preds = data["predictions"]
        winners = preds[preds["PredRank"]==1][["Race","Driver","Team","PodiumProb"]].head(5)
        ctx.append(f"2026 PREDICTED WINNERS (first 5 races):\n{winners.to_string(index=False)}\n")
    return "\n".join(ctx)

RACE_CONTEXT = build_race_context()

fastf1.Cache.enable_cache("data/raw/fastf1_cache")

def get_race_results(year: int, circuit: str) -> str:
    try:
        session = fastf1.get_session(year, circuit, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        res = session.results
        if res.empty: return f"No results found for {year} {circuit}."
        res["Position"] = pd.to_numeric(res["Position"], errors="coerce")
        top10 = res.nsmallest(10, "Position")[["Position","FullName","TeamName","Points"]]
        return f"Race Results ({year} {circuit}):\n" + top10.to_string(index=False)
    except Exception as e:
        return f"Error fetching results: {e}"

def get_race_weather(year: int, circuit: str) -> str:
    try:
        session = fastf1.get_session(year, circuit, 'R')
        session.load(telemetry=False, weather=True, messages=False)
        weather = session.weather_data
        if weather.empty: return f"No weather data found for {year} {circuit}."
        avg_air = weather['AirTemp'].mean()
        avg_track = weather['TrackTemp'].mean()
        rain = "Yes" if weather['Rainfall'].any() else "No"
        return f"Weather Summary ({year} {circuit}):\nAverage Air Temp: {avg_air:.1f}C\nAverage Track Temp: {avg_track:.1f}C\nRainfall: {rain}"
    except Exception as e:
        return f"Error fetching weather: {e}"

def get_fastest_laps(year: int, circuit: str) -> str:
    try:
        session = fastf1.get_session(year, circuit, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps.empty: return f"No lap data found for {year} {circuit}."
        laps["LapTimeSec"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()
        clean = laps[laps["LapTimeSec"] > 0]
        best = clean.groupby("Driver")["LapTimeSec"].min().sort_values().head(5).reset_index()
        best.columns = ["Driver","BestLap_Seconds"]
        best["BestLap_Seconds"] = best["BestLap_Seconds"].round(3)
        return f"Fastest Laps Top 5 ({year} {circuit}):\n" + best.to_string(index=False)
    except Exception as e:
        return f"Error fetching laps: {e}"

AVAILABLE_TOOLS = {
    "get_race_results": get_race_results,
    "get_race_weather": get_race_weather,
    "get_fastest_laps": get_fastest_laps,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_race_results",
            "description": "Get the top 10 finishers for a specific Formula 1 race.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "The year of the race (e.g. 2023)"},
                    "circuit": {"type": "string", "description": "The name of the circuit or grand prix (e.g. 'Monza', 'Monaco')"}
                },
                "required": ["year", "circuit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_race_weather",
            "description": "Get the weather summary (air temp, track temp, rainfall) for a specific Formula 1 race.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "The year of the race"},
                    "circuit": {"type": "string", "description": "The name of the circuit or grand prix"}
                },
                "required": ["year", "circuit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_fastest_laps",
            "description": "Get the top 5 fastest lap times for a specific Formula 1 race.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "The year of the race"},
                    "circuit": {"type": "string", "description": "The name of the circuit or grand prix"}
                },
                "required": ["year", "circuit"]
            }
        }
    }
]

SYSTEM_PROMPT = f"""You are an expert F1 race engineer with deep knowledge of Formula 1 
strategy, tyre management, and race tactics. You have access to tools that can fetch live
historical data from the FastF1 database for races between 2020 and 2026.

Check if the user is asking about an historical race or 2026 predictions.
If they ask about an established historical race, ALWAYS use your tools (`get_race_results`, `get_race_weather`, `get_fastest_laps`) to fetch the exact data before answering.
If a tool returns "No data found" or an error, politely inform the user.

You also have static predictions for the 2026 season:
{RACE_CONTEXT}

Guidelines:
- If a user asks about a past or recently completed race, CALL YOUR TOOLS to verify the data.
- Do not hallucinate race results.
- Be concise, data-driven and use F1 terminology.
"""

def get_ai_response(user_message, chat_history):
    if not client:
        return "Groq API key not found in .env file."
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in chat_history:
            messages.append({
                "role": "assistant" if msg["role"] == "assistant" else "user",
                "content": msg["content"]
            })
            
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            max_tokens=1024,
        )
        
        response_message = response.choices[0].message
        
        if not response_message.tool_calls:
            return response_message.content
            
        messages.append(response_message)
        
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = AVAILABLE_TOOLS[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            with st.spinner(f"Fetching {function_name} for {function_args.get('year')}..."):
                function_response = function_to_call(
                    year=function_args.get("year"),
                    circuit=function_args.get("circuit")
                )
                
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(function_response),
            })
            
        second_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
        )
        return second_response.choices[0].message.content
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Engineer offline: {str(e)}"

# Header 
st.title("🏎️ F1 AI Race Engineer")
st.caption("2026 Season Predictor · Monaco 2023 Analysis · Groq Llama 3.3 · XGBoost")
st.divider()

# Tabs 
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 2026 Season Predictions",
    "📊 Race Analysis",
    "🤖 AI Race Engineer",
    "ℹ️ About",
])

# TAB 1 — 2026 Season Predictions
with tab1:
    st.header("2026 F1 Season — Full Predictions")
    st.caption("Trained on FastF1 2021–2025 + 2026 Rounds 1–3 | XGBoost 3-model pipeline")

    if "predictions" not in data:
        st.error("predictions CSV not found in data/processed/")
    else:
        preds = data["predictions"].copy()

        # ── KPI row ──
        col1, col2, col3, col4 = st.columns(4)
        predicted_champ = preds[preds["PredRank"]==1]["Driver"].value_counts().index[0]
        predicted_champ_team = preds[preds["Driver"]==predicted_champ]["Team"].iloc[0]
        total_races = preds["Race"].nunique()
        top_driver_wins = preds[preds["PredRank"]==1]["Driver"].value_counts().iloc[0]

        col1.metric("Predicted Champion", predicted_champ, predicted_champ_team)
        col2.metric("Races Remaining", total_races)
        col3.metric("Predicted Wins", f"{top_driver_wins}/{total_races}", predicted_champ)
        col4.metric("Model", "XGBoost + SHAP", "3-model pipeline")

        st.divider()

        # Race selector 
        races = preds["Race"].unique().tolist()
        selected_race = st.selectbox("Select a race to inspect:", races)

        race_df = preds[preds["Race"] == selected_race].copy()

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader(f"Top 10 — {selected_race}")
            display = race_df.head(10)[["PredRank","Driver","Team","PodiumProb","Top10Prob"]].copy()
            display["PodiumProb"] = display["PodiumProb"].map(lambda x: f"{x:.1%}")
            display["Top10Prob"]  = display["Top10Prob"].map(lambda x: f"{x:.1%}")
            display.columns = ["Pos","Driver","Team","Podium %","Top 10 %"]
            st.dataframe(display, hide_index=True, use_container_width=True)

        with col_right:
            st.subheader(f"Podium Probability — {selected_race}")
            top5 = race_df.head(7).copy()
            fig = go.Figure(go.Bar(
                x=top5["PodiumProb"],
                y=top5["Driver"],
                orientation="h",
                marker_color=[
                    "#E8002D" if p >= 0.5 else
                    "#FFC906" if p >= 0.2 else "#ABABAB"
                    for p in top5["PodiumProb"]
                ],
                text=[f"{p:.0%}" for p in top5["PodiumProb"]],
                textposition="outside",
            ))
            fig.update_layout(
                xaxis=dict(tickformat=".0%", range=[0, 1.15]),
                yaxis=dict(autorange="reversed"),
                height=300,
                margin=dict(l=0, r=40, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Heatmap 
        st.subheader("Podium Probability Heatmap — Top 7 Drivers vs All Races")
        top7 = ["ANT","RUS","LEC","HAM","NOR","PIA","VER"]
        hmap = preds[preds["Driver"].isin(top7)].pivot_table(
            index="Driver", columns="Race",
            values="PodiumProb", aggfunc="first"
        ).round(2)
        race_order = preds.drop_duplicates("Race").sort_values("Round")["Race"].tolist()
        hmap = hmap[[c for c in race_order if c in hmap.columns]]

        fig2 = px.imshow(
            hmap, text_auto=".0%",
            color_continuous_scale="RdYlGn", zmin=0, zmax=1,
            aspect="auto", height=320,
        )
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # Championship 
        st.subheader("Predicted Final Championship Standings")
        POINTS_MAP = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
        ACTUAL_2026 = {
            "ANT":68,"RUS":55,"LEC":42,"HAM":35,"NOR":30,
            "PIA":24,"VER":22,"GAS":7,"HAD":2,"TSU":2,
        }
        champ = []
        for driver in preds["Driver"].unique():
            actual  = ACTUAL_2026.get(driver, 0)
            pred_pts= sum(POINTS_MAP.get(int(round(r)), 0)
                         for r in preds[preds["Driver"]==driver]["PredRank"])
            team    = preds[preds["Driver"]==driver]["Team"].iloc[0]
            champ.append({"Driver":driver,"Team":team,
                          "Actual":actual,"Predicted":pred_pts,
                          "Total":actual+pred_pts})

        champ_df = pd.DataFrame(champ).sort_values("Total", ascending=False)
        champ_df["Pos"] = range(1, len(champ_df)+1)

        fig3 = px.bar(
            champ_df.head(10), x="Driver", y="Total", color="Team",
            text="Total", height=380,
            title="Predicted 2026 Final Championship Points",
            labels={"Total":"Predicted Points"},
        )
        fig3.update_layout(margin=dict(t=40, b=0))
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Top 10 Table")
        disp = champ_df.head(10)[["Pos","Driver","Team","Actual","Predicted","Total"]].copy()
        disp.columns = ["Pos","Driver","Team","Actual Pts","Pred Pts","Total"]
        st.dataframe(disp, hide_index=True, use_container_width=True)

# TAB 2 — Race Analysis (Monaco 2023)
with tab2:
    st.header("Monaco 2023 — Race Analysis")

    if "results" not in data:
        st.error("Monaco 2023 results not found.")
    else:
        results = data["results"].copy()
        results["Position"] = pd.to_numeric(results["Position"], errors="coerce")

        col1, col2, col3 = st.columns(3)
        col1.metric("Race", "Monaco 2023")
        col2.metric("Winner", "VER")
        col3.metric("Total Laps", len(data.get("laps", pd.DataFrame())))

        st.subheader("Race Results")
        top10 = results.nsmallest(10, "Position")[
            ["Position","FullName","TeamName","GridPosition","Points"]
        ]
        st.dataframe(top10, hide_index=True, use_container_width=True)

        if "laps" in data:
            laps = data["laps"].copy()
            laps["LapTimeSec"] = pd.to_timedelta(laps["LapTime"]).dt.total_seconds()

            st.subheader("Lap Time Comparison — Top 3")
            top3 = results.nsmallest(3, "Position")["Abbreviation"].tolist()
            clean = laps[
                laps["LapTimeSec"].between(72, 105) &
                laps["Driver"].isin(top3)
            ].copy()
            fig4 = px.line(clean, x="LapNumber", y="LapTimeSec", color="Driver",
                           markers=True, height=350,
                           labels={"LapTimeSec":"Lap Time (s)","LapNumber":"Lap"})
            st.plotly_chart(fig4, use_container_width=True)

            st.subheader("Tyre Strategy")
            tyre = laps[["Driver","LapNumber","Compound"]].copy()
            compound_colors = {
                "SOFT":"#E8002D","MEDIUM":"#FFF200",
                "HARD":"#EBEBEB","INTERMEDIATE":"#39B54A","WET":"#0067FF",
            }
            fig5 = px.scatter(tyre, x="LapNumber", y="Driver", color="Compound",
                              color_discrete_map=compound_colors, height=420,
                              title="Tyre Strategy — All Drivers")
            fig5.update_traces(marker=dict(size=6, symbol="square"))
            st.plotly_chart(fig5, use_container_width=True)

        if "features" in data:
            st.subheader("Driver Performance Features")
            feats = data["features"]
            if "Race" in feats.columns:
                monaco_f = feats[feats["Race"].str.contains("Monaco", na=False)]
                if not monaco_f.empty:
                    cols = ["Driver","GridPosition","FinishPosition",
                            "AvgLapTime","LapTimeStd","PitStops","PositionDelta"]
                    cols = [c for c in cols if c in monaco_f.columns]
                    st.dataframe(monaco_f[cols].round(2), hide_index=True,
                                 use_container_width=True)

        # SHAP chart
        shap_path = "data/processed/shap_feature_importance.png"
        if not os.path.exists(shap_path):
            shap_path = "data/models/shap_importance.png"
        if os.path.exists(shap_path):
            st.subheader("SHAP Feature Importance — Podium Prediction Model")
            st.image(shap_path, use_column_width=True)

# TAB 3 — AI Race Engineer Chat
with tab3:
    st.header("🤖 AI Race Engineer")
    st.caption("Powered by Groq Llama 3.3 70B · Grounded in real F1 data")

    with st.sidebar:
        st.header("📊 Data Status")
        for label, key in [("Monaco 2023 laps","laps"),("Race results","results"),
                            ("2026 predictions","predictions")]:
            if key in data:
                st.success(f"{label} loaded")
            else:
                st.warning(f"{label} missing")

        if models:
            st.success(f"ML models loaded ({len(models)})")
        else:
            st.warning("ML models not found")

        st.divider()
        st.subheader("💡 Try asking:")
        questions = [
            "Who will win the 2026 championship?",
            "Should Verstappen have pitted earlier at Monaco?",
            "Why is Antonelli predicted to win every race?",
            "What does the tyre strategy tell us about Monaco?",
            "Compare Leclerc and Hamilton's 2026 predictions",
            "Which circuit is best for Red Bull in 2026?",
            "Was the undercut possible at Monaco 2023?",
            "Which driver is most consistent?",
        ]
        for q in questions:
            if st.button(q, use_container_width=True, key=f"btn_{q}"):
                st.session_state.quick_question = q

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Race engineer online. I have Monaco 2023 race data and full 2026 season predictions. Ask me anything about strategy, predictions or driver performance."
        }]

    if "quick_question" in st.session_state:
        q = st.session_state.pop("quick_question")
        st.session_state.messages.append({"role":"user","content":q})
        with st.spinner("Analysing..."):
            resp = get_ai_response(q, st.session_state.messages)
        st.session_state.messages.append({"role":"assistant","content":resp})

    for msg in st.session_state.messages:
        avatar = "🏎️" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask your race engineer..."):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🏎️"):
            with st.spinner("Checking data..."):
                resp = get_ai_response(prompt, st.session_state.messages)
            st.markdown(resp)
        st.session_state.messages.append({"role":"assistant","content":resp})

# TAB 4 — About
with tab4:
    st.header("About This Project")
    st.markdown("""
### F1 AI/ML Project — End-to-End Pipeline

Built as a portfolio project targeting data science and ML engineering internships.

---

### Tech Stack

| Layer | Technology |
|---|---|
| Data pipeline | FastF1, Pandas, OpenF1 API |
| Visualisation | Plotly, Seaborn |
| ML models | XGBoost, scikit-learn |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| AI chatbot | Groq Llama 3.3 70B |
| Frontend | Streamlit |

---

### Model Architecture

**3-model XGBoost pipeline trained on FastF1 2021–2025 + 2026 Rounds 1–3:**
- **Model A** — Position Regressor (predicts P1–P20 finishing position)
- **Model B** — Podium Classifier (predicts top 3 probability)
- **Model C** — Top 10 Classifier (predicts points finish probability)

**Key features:**
- Rolling 3-race form per driver
- Circuit-specific historical performance
- Constructor momentum
- Lap time consistency (std dev)
- DNF reliability index
- Recency weighting (2026 data = 3× weight)

---

### Key Findings

- Grid position is the strongest predictor of finishing position (SHAP value ~2.3)
- Constructor strength is second most important feature
- Recent form (Last3AvgPos) outperforms raw lap time in predictive power
- Circuit-specific models would further improve accuracy

---

### Data Sources
- [FastF1](https://docs.fastf1.dev/) — lap telemetry, timing (2018–2026)
- [OpenF1 API](https://openf1.org/) — live race data, team radio
- [Kaggle: jtrotman/formula-1-race-data](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data) — historical 1950–2024
    """)

st.divider()
st.caption("Built with FastF1 · XGBoost · SHAP · MLflow · Groq Llama 3.3 · Streamlit | F1 AI/ML Project")