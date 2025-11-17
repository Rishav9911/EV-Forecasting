import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from groq import Groq
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

import os
import re

# Page configuration
st.set_page_config(
    page_title="EV Battery Health Assistant",
    page_icon="🔋",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color:black;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
        color:black;
    }
    </style>
""", unsafe_allow_html=True)

# Feature list
FEATURE_LIST = [
    "Charging Duration (min)",
    "Optimal Charging Duration Class",
    "SOC (%)",
    "Battery Temp (°C)"
]

@st.cache_resource
def load_and_train_model():
    """Load data and train the model (cached for efficiency)"""
    try:
        # Load the filtered dataset
        df = pd.read_csv("ev_battery_charging_data.csv")
        
        # Data preprocessing
        df = df.drop_duplicates()
        df = df.dropna()
        columns_to_keep = [
            "Degradation Rate (%)",
            "Charging Duration (min)",
            "Optimal Charging Duration Class",
            "SOC (%)",
            "Battery Temp (°C)"
        ]
        df = df[columns_to_keep]
        
        target_col = 'Degradation Rate (%)'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        return model, scaler, X.columns.tolist()
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def extract_features_from_text(text):
    """Extract numerical features from user message"""
    features = {}
    text_lower = text.lower()
    
    patterns = {
        "Charging Duration (min)": r"(?:charging\s*duration|duration).*?(\d+\.?\d*)",
        "Optimal Charging Duration Class": r"(?:optimal|class).*?(\d+\.?\d*)",
        "SOC (%)": r"(?:soc|state\s*of\s*charge).*?(\d+\.?\d*)",
        "Battery Temp (°C)": r"(?:temp|temperature).*?(\d+\.?\d*)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            features[key] = float(match.group(1))
    
    return features if len(features) == len(FEATURE_LIST) else None

def predict_degradation(model, scaler, features_dict):
    """Make prediction using the trained model"""
    values = np.array([features_dict[feat] for feat in FEATURE_LIST]).reshape(1, -1)
    values_scaled = scaler.transform(values)
    prediction = model.predict(values_scaled)[0]
    return round(prediction, 4)

@st.cache_resource
def load_station_data(csv_file="ev_sessions_processed.csv"):
    """Load raw session data"""
    df = pd.read_csv(csv_file, parse_dates=['connectionTime', 'disconnectTime', 'doneChargingTime'])
    # Ensure kWhDelivered is numeric
    df['kWhDelivered'] = pd.to_numeric(df['kWhDelivered'], errors='coerce')
    df = df.dropna(subset=['kWhDelivered', 'stationID'])
    return df

def aggregate_station_load(df, station_id, freq='15T'):
    """
    Aggregate historical load per station into time series.
    freq='15T' means 15-minute bins.
    """
    df_station = df[df['stationID'] == station_id].copy()
    if df_station.empty:
        return None
    # Create charging start timestamp series
    df_station['timestamp'] = pd.to_datetime(df_station['connectionTime'])
    ts = df_station.set_index('timestamp').resample(freq)['kWhDelivered'].sum()
    ts = ts.asfreq(freq, fill_value=0)
    return ts

def forecast_station_load(ts, steps=1):
    """
    Forecast next 'steps' time intervals using ARIMA
    ts: pd.Series of historical kWhDelivered aggregated by interval
    """
    try:
        # Fit ARIMA (simple ARIMA(1,0,0) baseline)
        model = ARIMA(ts, order=(1,0,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast.values.tolist()
    except Exception as e:
        return [f"Forecasting error: {e}"]


def get_groq_response(user_input, api_key, extracted_features=None, prediction=None):
    """Get response from Groq API"""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        if extracted_features and prediction is not None:
            prompt = f"""
            A user asked about EV battery degradation.
            
            Input features:
            {extracted_features}
            
            Model Prediction:
            Degradation Rate (%) = {prediction}
            
            Explain the prediction in simple, friendly language. Provide insights about:
            - What this degradation rate means
            - Whether it's good or concerning
            - Tips to improve battery health if needed
            """
        else:
            prompt = user_input
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert EV battery health assistant. Provide helpful, accurate, and friendly advice about battery degradation and charging practices."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4 if extracted_features else 0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error: {str(e)}"
    

def aggregate_daily_load(df, station_id):
    """Aggregate daily total kWh delivered for a station."""
    df_station = df[df['stationID'].astype(str) == str(station_id)].copy()
    if df_station.empty:
        return None

    df_station['timestamp'] = pd.to_datetime(df_station['connectionTime'])
    
    # Daily SUM of kWh delivered
    ts_daily = (
        df_station.set_index('timestamp')
        .resample('D')['kWhDelivered']
        .sum()
        .asfreq('D', fill_value=0)
    )
    
    return ts_daily

def forecast_next_day(ts_daily):
    """Forecast next-day total load using ARIMA."""
    try:
        model = ARIMA(ts_daily, order=(2,1,2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return float(forecast.values[0])
    except Exception as e:
        return f"Forecast error: {e}"


# Main App
def main():
    st.markdown("<h1 class='main-header'>🔋 EV Battery Health Assistant</h1>", unsafe_allow_html=True)

    with st.sidebar:   
        st.markdown("---")
        st.header("📊 How to Use")

        st.markdown("""
        **🔋 For Battery Prediction:**  
        Provide all 4 features:
        - Charging Duration (min)
        - Optimal Charging Duration Class
        - SOC (%)
        - Battery Temp (°C)

        **Example Query:**  
        *"My charging duration is 45 minutes, optimal class is 2, SOC is 80%, and battery temp is 35°C."*

        **Or ask general battery questions:**  
        - *"How can I improve battery health?"*  
        - *"What causes battery degradation?"*
        """)

        st.markdown("---")

        st.markdown("""
        **🔮 Load Forecasting (NEW):**  
        This feature predicts the **energy load for any charging station** for the next 15 minutes using:
        - ARIMA Time-Series Forecasting  
        - Historical energy data (kWh delivered)
        
        **What you can ask:**  
        - *"Forecast for Station 2-39-123-23"*  
        - *"What is forecast for station 2_39_124_22 ?"*  
        - *" will we use renewable for tomorrow for station 2-39-124-22"*

        **What you get:**  
        - Predicted energy demand (kWh)  
        - Automatic ARIMA model selection  
        - Uses past sessions of the station  
        - Works even with inconsistent or sparse data  

        """)

        st.markdown("---")
        st.header("📈 Model Info")

        if st.button("Show Model Details"):
            st.session_state.show_model_info = True

    
    # Sidebar for API key and info
    # with st.sidebar:
        
    #     st.markdown("---")
    #     st.header("📊 How to Use")
    #     st.markdown("""
    #     **For Prediction:**
    #     Provide all 4 features:
    #     - Charging Duration (min)
    #     - Optimal Charging Duration Class
    #     - SOC (%)
    #     - Battery Temp (°C)
        
    #     **Example:**
    #     "My charging duration is 45 minutes, optimal class is 2, SOC is 80%, and battery temp is 35°C"
        
    #     **Or ask general questions:**
    #     - "How can I improve battery health?"
    #     - "What causes battery degradation?"
    #     """)
        
    #     st.markdown("---")
    #     st.header("📈 Model Info")
    #     if st.button("Show Model Details"):
    #         st.session_state.show_model_info = True
    
    # Load model
    model, scaler, feature_columns = load_and_train_model()
    
    if model is None:
        st.error("⚠️ Failed to load model. Please ensure 'filtered_ev_battery_data.csv' is in the same directory.")
        st.stop()
    
    # Display model info if requested
    if st.session_state.get('show_model_info', False):
        with st.sidebar:
            st.success("Model: Random Forest Regressor")
            st.info(f"Features: {len(feature_columns)}")
            st.session_state.show_model_info = False
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"<div class='chat-message user-message'>👤 <b>You:</b> {message['content']}</div>", 
                       unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message bot-message'>🤖 <b>Assistant:</b> {message['content']}</div>", 
                       unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Ask about battery health or provide charging data...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("🤔 Thinking..."):

            # --- 1) STATION FORECAST DETECTION (first priority!) ---
            # This regex matches "forecast station 42", "station id 42 forecast",
            # "load forecast for charger 42", "predict station 42", etc.
            # station_pattern = r"(?:station|charger)[^\d]*(\d+)"
            # Matches both 42 AND CA-313 AND 2-39-123-23
            # --- 1) STATION FORECAST DETECTION ---
            station_pattern = r"(?:station|charger)[^\w\-]*([\w\-]+)"
            forecast_keywords = ["forecast", "predict", "load", "next", "tomorrow", "24", "day"]

            station_match = re.search(station_pattern, user_input.lower())
            user_lower = user_input.lower()

            # Detect if user is asking for 24-hour forecast
            is_daily_request = any(x in user_lower for x in [
                "next day", "tomorrow", "24 hour", "24hr", "day ahead", "daily", "24h"
            ])

            if station_match and any(word in user_lower for word in forecast_keywords):

                station_id = station_match.group(1)
                df_sessions = load_station_data()

                df_station = df_sessions[df_sessions["stationID"].astype(str) == str(station_id)]

                if df_station.empty:
                    response = f"❌ No historical data found for station {station_id}."

                # ----------- DAILY FORECAST 24 HOURS -----------
                elif is_daily_request:
                    ts_daily = aggregate_daily_load(df_sessions, station_id)

                    if ts_daily is None:
                        response = f"❌ Not enough data for daily forecast for station {station_id}."
                    else:
                        forecast_val = forecast_next_day(ts_daily)

                        if isinstance(forecast_val, str):
                            response = forecast_val  # error message
                        else:
                            avg_load = ts_daily.mean()

                            if forecast_val < avg_load:
                                recommendation = "🌱 **Recommended:** Use *renewable energy* tomorrow."
                            else:
                                recommendation = "⚡ **Recommended:** Use *grid power* tomorrow due to high load."

                            response = (
                                f"📆 **Next-Day Load Forecast for Station {station_id}**\n\n"
                                f"🔮 Predicted total load (next 24 hours): **{forecast_val:.2f} kWh**\n"
                                f"📊 Average historical load: **{avg_load:.2f} kWh**\n\n"
                                f"{recommendation}"
                            )

                # ----------- 15-MIN FORECAST (existing) -----------
                else:
                    ts = aggregate_station_load(df_sessions, station_id, freq="15T")
                    forecast_vals = forecast_station_load(ts, steps=1)

                    if isinstance(forecast_vals[0], str):
                        response = forecast_vals[0]
                    else:
                        response = (
                            f"📈 **Forecast for Station {station_id}**\n\n"
                            f"Predicted load for next 15 minutes: **{forecast_vals[0]:.2f} kWh**"
                        )

            # --- 2) BATTERY HEALTH PREDICTION ---
            else:
                extracted = extract_features_from_text(user_input)

                if extracted:
                    try:
                        prediction = predict_degradation(model, scaler, extracted)
                        response = get_groq_response(
                            user_input, None, extracted, prediction
                        )
                    except ValueError:
                        st.error("⚠️ Missing features. Provide all 4 required values.")
                        st.stop()

                # --- 3) GENERAL CHAT FALLBACK ---
                else:
                    response = get_groq_response(user_input, None)

            # Store AI response
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    
    # if user_input:
    #             # Check if the user asked for station load forecasting
    #     station_pattern = r"(?:station|charger)\s*id\s*(\d+).*forecast"
    #     station_match = re.search(station_pattern, user_input.lower())

    #     if station_match:
    #         station_id = int(station_match.group(1))
    #         df_sessions = load_station_data()
    #         ts = aggregate_station_load(df_sessions, station_id, freq='15T')
    #         if ts is None:
    #             response = f"No historical data found for station {station_id}."
    #         else:
    #             forecast_vals = forecast_station_load(ts, steps=1)
    #             if isinstance(forecast_vals[0], str):
    #                 response = forecast_vals[0]  # error
    #             else:
    #                 response = f"Forecasted load at station {station_id} in next 15 minutes: {forecast_vals[0]:.2f} kWh"
    #     else:
    #         # Existing battery health chat handler
    #         extracted = extract_features_from_text(user_input)
    #         if extracted:
    #             try:
    #                 prediction = predict_degradation(model, scaler, extracted)
    #                 response = get_groq_response(user_input, None, extracted, prediction)
    #             except ValueError as e:
    #                 st.error("⚠️ **Incomplete Data!** Please provide all 4 features for a prediction.")
    #                 st.stop()
    #         else:
    #             response = get_groq_response(user_input, None)

        
    #     # Add user message to history
    #     st.session_state.messages.append({"role": "user", "content": user_input})
        
    #     # Try to extract features for prediction
    #     extracted = extract_features_from_text(user_input)
        
    #     # with st.spinner("🤔 Thinking..."):
    #     #     if extracted:
    #     #         # Make prediction
    #     #         prediction = predict_degradation(model, scaler, extracted)
    #     #         response = get_groq_response(user_input, None, extracted, prediction)
    #     #     else:
    #     #         # General Q&A
    #     #         response = get_groq_response(user_input, None)

    #     with st.spinner("🤔 Thinking..."):
    #         if extracted:
    #             try:
    #                 # Make prediction
    #                 prediction = predict_degradation(model, scaler, extracted)
    #                 response = get_groq_response(user_input, None, extracted, prediction)
    #             except ValueError as e:
    #                 st.error("⚠️ **Incomplete Data!** Please provide all 4 features for a prediction:")
    #                 st.info("""
    #                 Required features:
    #                 - Charging Duration (min)
    #                 - Optimal Charging Duration Class  
    #                 - SOC (%)
    #                 - Battery Temp (°C)
                    
    #                 **Example:** "My charging duration is 45 minutes, optimal class is 2, SOC is 80%, and battery temp is 35°C"
    #                 """)
    #                 st.stop()
    #         else:
    #             # General Q&A
    #             response = get_groq_response(user_input, None)
        
    #     # Add bot response to history
    #     st.session_state.messages.append({"role": "assistant", "content": response})
        
    #     # Rerun to display new messages
    #     st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()