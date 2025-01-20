import streamlit as st
import pandas as pd
import joblib
from conversation import get_investment_strategy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

current_dir = Path(__file__).parent
XG_path = current_dir/'XGboost_model.joblib'
LR_path = current_dir/'logistic_regression_model.joblib'
try:
    xgboost_model = joblib.load(XG_path)
    regression_model = joblib.load(LR_path)
except Exception as e:
    st.error(f"Error loading models: {e}")

def prepare_data(data):
    selected_features = ["VIX", "GTITL2YR", "GTITL10YR", "GTITL30YR", "EONIA", "XAU BGNL"]
    window_size = 7

    # Apply moving averages
    for feature in selected_features:
        data[f'{feature}_MA{window_size}'] = data[feature].rolling(window=window_size).mean()

    # Handle missing values that result from rolling
    data = data.dropna().reset_index(drop=True)

    # Select features for the model
    feature_columns = [f"{feature}" for feature in selected_features] + \
                      [f"{feature}_MA{window_size}" for feature in selected_features]

    # Make sure all necessary columns are included
    data_features = data[feature_columns]

    # Scale the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_features)

    return data_scaled

def parse_strategy(response):
    try:
        # Assuming response is a string that needs to be loaded as JSON
        strategy = json.loads(response)
        return strategy
    except json.JSONDecodeError:
        print("Error decoding the JSON response")
        return None

def display_strategy(strategy):
    if strategy:
        st.markdown("### Recommended Investment Strategy")
        st.markdown(f"**Investment Type:** {strategy['investment_type']}")
        st.markdown(f"**Confidence Level:** {strategy['confidence_level']*100}%")
        st.markdown("**Strategy Description:**")
        st.write(strategy['strategy_description'])
    else:
        st.error("Failed to fetch the investment strategy.")
# Streamlit interface
st.title('Market Crash Prediction and Investment Strategy')

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        data = None

    if data is not None:
        # Model selection
        model_choice = st.radio(
            "Choose a model for prediction:",
            ('XGBoost(If you favor precision)', 'Logistic Regression(If you favor recall)')
        )

        if st.button('Predict and Suggest Strategy'):
            if model_choice == 'XGBoost(If you favor precision)':
                prediction = xgboost_model.predict(prepare_data(data))
                model_used = 'XGBoost'
            else:
                prediction = regression_model.predict(prepare_data(data))
                model_used = 'Logistic Regression'


            if prediction is not None:
                try:
                    # Fetch investment strategies from Groq API
                    strategy = get_investment_strategy(data)  # Adjust data format if necessary
                    st.write('Predictions:', prediction)
                    st.write('Recommended Strategy:', strategy)
                except Exception as e:
                    st.error(f"Error fetching investment strategy: {e}")
