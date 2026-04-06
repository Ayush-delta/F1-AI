# Contributor Guide & Setup Instructions

Welcome to the F1 AI/ML Project! To run this project locally, experiment with the XGBoost pipeline, or chat with the AI Race Engineer, follow the instructions below.

## 1. Environment Setup

It's highly recommended to use a virtual environment so the FastF1 and ML dependencies do not conflict with your global python packages.

```bash
# Clone the repository
git clone https://github.com/Ayush-delta/f1-ai-ml.git
cd f1-ai-ml

# Create and activate a Virtual Environment (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

## 2. API Keys & Telemetry Caching

### FastF1 Cache
FastF1 downloads massive amounts of telemetry data from the official F1 timing servers. To ensure it runs fast and doesn't spam their servers, this project uses a local cache folder. Ensure the folder exists:
```bash
mkdir -p data/raw/fastf1_cache
```
*Note: This folder is automatically ignored by Git to prevent pushing heavy telemetry SQL files.*

### Groq API Setup
To use the Agentic AI Race Engineer (Tab 3), you need a free API key from Groq to run the LLaMA 3.3 70B model.
1. Sign up at [console.groq.com](https://console.groq.com/).
2. Generate an API Key.
3. Create a `.env` file in the root of the project.
4. Add your key to the file:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

## 3. Running the Pipeline

Before starting the Streamlit application, it is best to re-compile the Machine Learning models locally. This generates the 2026 predictions and stores them in the `data/processed/` folder.

```bash
# Train models and generate predictions
python 03_ml_model.py
```

## 4. Launching the App

Once the models are saved, start the Streamlit UI to access the Predictions, the Historical Race Analysis visualization hub, and the AI chatbot:

```bash
streamlit run 04_ai_race_engineer.py
```

## Expanding the Agent
If you wish to contribute to the AI's toolset, look inside `04_ai_race_engineer.py` for the `AVAILABLE_TOOLS` dictionary. You can easily add new Python functions (e.g. `get_driver_standings()`) mapped to specific FastF1 subroutines to make the AI Race Engineer even smarter!
