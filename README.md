# F1 AI/ML Project

An end-to-end data science and AI project built on Formula 1 race data.
Covers data engineering, machine learning, and an AI race engineer agent.

Focus race: **2023 Monaco Grand Prix**

---

## Project structure

```
f1_project/
├── data/
│   ├── raw/              # FastF1 cache, Kaggle CSVs (gitignored)
│   └── processed/        # cleaned CSVs, exported charts
├── notebooks/
│   ├── 01_data_pull.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_ml_model.ipynb
│   └── 04_ai_race_engineer.ipynb
├── src/
│   ├── features.py       # feature engineering helpers
│   ├── model.py          # training + evaluation
│   └── engineer.py       # AI race engineer logic
├── setup.sh
├── requirements.txt
└── README.md
```

---

## Phases

| Phase | What | Tools |
|-------|------|-------|
| 1 | Data pull & exploration | FastF1, pandas, Plotly |
| 2 | EDA & visual storytelling | Plotly, Seaborn |
| 3 | GP winner prediction model | XGBoost, scikit-learn, SHAP, MLflow |
| 4 | AI race engineer chatbot | Claude API, MCP, Streamlit |
| 5 | Packaging & deployment | Docker, GitHub Actions |

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/f1-ai-ml.git
cd f1-ai-ml

# 2. Create environment
bash setup.sh
source venv/bin/activate

# 3. Run Phase 1
jupyter notebook notebooks/01_data_pull.ipynb
```

---

## Data sources

- [FastF1](https://docs.fastf1.dev/) — lap telemetry, tyre data, weather (2018–present)
- [Kaggle: jtrotman/formula-1-race-data](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data) — historical results 1950–2023
- [OpenF1 API](https://openf1.org/) — live race data, team radio

---

## Key findings (Monaco 2023)

> _Fill this in after Phase 2 EDA — this is what you talk about in interviews_

- ...
- ...

---

## Requirements

```
fastf1
pandas
plotly
scikit-learn
xgboost
shap
mlflow
streamlit
anthropic
python-dotenv
```

---

## Author

Your Name — [LinkedIn](#) · [GitHub](#)
