import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model = joblib.load("xgb_rul_model.pkl")

st.title("MaintAI – Predictive Maintenance Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload CMAPSS Test File (e.g., test_FD001.txt)", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        # NASA CMAPSS files are space-separated
        df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)

        # Drop empty columns (sometimes extra spaces create NaNs)
        df.dropna(axis=1, inplace=True)

        # Define standard column names (for 26 columns)
        expected_cols = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"] + \
                        [f"sensor_measurement_{i}" for i in range(1, 22)]

        # Validate column count
        if df.shape[1] != len(expected_cols):
            st.error(f"❌ Uploaded file has {df.shape[1]} columns, but expected {len(expected_cols)}.")
            st.stop()

        # Assign proper column names
        df.columns = expected_cols

        # Drop low variance sensor columns (only if they exist)
        low_var_cols = ['sensor_measurement_5', 'sensor_measurement_10',
                        'sensor_measurement_16', 'sensor_measurement_18', 'sensor_measurement_19']
        low_var_cols = [col for col in low_var_cols if col in df.columns]
        df.drop(columns=low_var_cols, inplace=True)

        # Select features
        features = ['operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + \
                   [col for col in df.columns if "sensor_measurement" in col]

        # Normalize feature data
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        st.success("✅ File uploaded and preprocessed successfully!")

        # Visualization: Plot sensor trends
        st.subheader("📊 Sensor Trend (First Engine Only)")
        unit_1 = df[df['unit_number'] == df['unit_number'].iloc[0]]

        fig, ax = plt.subplots(figsize=(12, 5))
        for col in features[:3]:  # Show first 3 sensors for simplicity
            ax.plot(unit_1["time_in_cycles"], unit_1[col], label=col)

        ax.set_xlabel("Time in Cycles")
        ax.set_ylabel("Sensor Readings")
        ax.set_title("Sensor Trends for Engine 1")
        ax.legend()
        st.pyplot(fig)

        # Predict RUL
        st.subheader("📈 Remaining Useful Life (RUL) Prediction")
        df["Predicted_RUL"] = model.predict(df[features])
        st.write(df[["unit_number", "time_in_cycles", "Predicted_RUL"]].tail(10))

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
        st.stop()
