#!/bin/bash
# F1 AI/ML Project — Environment Setup
# Run this once to create your environment

echo "Creating virtual environment..."
python -m venv venv

echo "Activating..."
# source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows — uncomment this line instead

echo "Installing packages..."
pip install fastf1 pandas plotly scikit-learn xgboost shap mlflow streamlit groq python-dotenv

echo "Creating project folders..."
mkdir -p data/raw data/processed data/models src assets

echo ""
echo "Done! Next steps:"
echo "  1. Run: source venv/bin/activate (or venv\Scripts\activate on Windows)"
echo "  2. Run: python 03_ml_model.py to generate models"
echo "  3. Run: streamlit run 04_ai_race_engineer.py to view the UI"
