import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler

# Page configuration
st.set_page_config(
    page_title="MaintAI – Predictive Maintenance",
    page_icon="🛠️",
    layout="wide"
)

# Inject custom CSS for classy light frontend
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background-image: url("https://i.imgur.com/qG4pZcC.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0);
    }

    .main-title {
        text-align: center;
        font-size: 50px;
        color: #0A2647;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #2C3E50;
        margin-bottom: 40px;
    }

    .footer {
        text-align: center;
        font-size: 14px;
        color: #888;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# App heading
st.markdown("<div class='main-title'>🛠️ MaintAI – Predictive Maintenance Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload CMAPSS sensor data to analyze engine health and predict Remaining Useful Life (RUL)</div>", unsafe_allow_html=True)

# Load trained model
try:
    model = joblib.load("xgb_rul_model.pkl")
except:
    st.error("⚠️ Model file `xgb_rul_model.pkl` not found in working directory.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CMAPSS .txt or .csv file", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        try:
            df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
        except:
            df = pd.read_csv(uploaded_file, sep=",", header=None)

        df.dropna(axis=1, inplace=True)

        # Assign standard CMAPSS column names
        expected_cols = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"] + \
                        [f"sensor_measurement_{i}" for i in range(1, 22)]

        if df.shape[1] != len(expected_cols):
            st.error(f"❌ Uploaded file has {df.shape[1]} columns, but {len(expected_cols)} expected.")
            st.stop()

        df.columns = expected_cols

        # Drop low-variance columns
        low_var_cols = ['sensor_measurement_5', 'sensor_measurement_10',
                        'sensor_measurement_16', 'sensor_measurement_18', 'sensor_measurement_19']
        df.drop(columns=[col for col in low_var_cols if col in df.columns], inplace=True)

        # Features for model prediction
        features = ['operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + \
                   [col for col in df.columns if "sensor_measurement" in col]

        # Normalize features
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        st.success("✅ File uploaded and preprocessed successfully!")

        # Plot sensor trends for first engine
        st.subheader("📊 Sensor Trend – First Engine")
        unit_1 = df[df["unit_number"] == df["unit_number"].iloc[0]]
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in features[:3]:
            ax.plot(unit_1["time_in_cycles"], unit_1[col], label=col)
        ax.set_xlabel("Time in Cycles")
        ax.set_ylabel("Sensor Value")
        ax.set_title("Top 3 Sensor Trends – First Engine")
        ax.legend()
        st.pyplot(fig)

        # Predict RUL
        st.subheader("🧠 Predicted Remaining Useful Life (RUL)")
        df["Predicted_RUL"] = model.predict(df[features])
        st.dataframe(df[["unit_number", "time_in_cycles", "Predicted_RUL"]].tail(10))

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")
        st.stop()

# Footer
st.markdown("<div class='footer'>© 2025 MaintAI Project | Group 18</div>", unsafe_allow_html=True)
