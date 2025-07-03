import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler

# Set page config
st.set_page_config(page_title="MaintAI Dashboard", layout="wide")

# Load trained model
model = joblib.load("xgb_rul_model.pkl")

# App title & description
st.title("🛠️ MaintAI – Predictive Maintenance Dashboard")
st.markdown("Upload CMAPSS sensor data to visualize trends and estimate Remaining Useful Life (RUL) using a trained XGBoost model.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload CMAPSS .txt or .csv file", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        # Try reading as whitespace-delimited, fallback to CSV
        try:
            df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
        except:
            df = pd.read_csv(uploaded_file, sep=",", header=None)

        df.dropna(axis=1, inplace=True)  # Remove empty columns

        # Define expected column names (26 columns)
        expected_cols = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"] + \
                        [f"sensor_measurement_{i}" for i in range(1, 22)]

        if df.shape[1] != len(expected_cols):
            st.error(f"❌ Uploaded file has {df.shape[1]} columns, but {len(expected_cols)} expected.")
            st.stop()

        df.columns = expected_cols

        # Drop low variance columns
        low_var_cols = ['sensor_measurement_5', 'sensor_measurement_10',
                        'sensor_measurement_16', 'sensor_measurement_18', 'sensor_measurement_19']
        df.drop(columns=[col for col in low_var_cols if col in df.columns], inplace=True)

        # Select features
        features = ['operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + \
                   [col for col in df.columns if "sensor_measurement" in col]

        # Normalize
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        st.success("✅ File uploaded and preprocessed successfully!")

        # Plot first engine sensor trends
        st.subheader("📊 Sensor Trend – First Engine")
        unit_1 = df[df["unit_number"] == df["unit_number"].iloc[0]]
        fig, ax = plt.subplots(figsize=(12, 5))
        for col in features[:3]:
            ax.plot(unit_1["time_in_cycles"], unit_1[col], label=col)
        ax.set_xlabel("Time in Cycles")
        ax.set_ylabel("Sensor Value")
        ax.set_title("Top 3 Sensors Over Time – First Engine")
        ax.legend()
        st.pyplot(fig)

        # Predict RUL
        st.subheader("🧠 Predicted Remaining Useful Life")
        df["Predicted_RUL"] = model.predict(df[features])
        st.dataframe(df[["unit_number", "time_in_cycles", "Predicted_RUL"]].tail(10))

    except Exception as e:
        st.error(f"⚠️ Something went wrong:\n\n`{e}`")
        st.stop()
