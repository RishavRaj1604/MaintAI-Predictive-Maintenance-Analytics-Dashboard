import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load model
model = joblib.load("xgb_rul_model.pkl")

st.title("MaintAI – Predictive Maintenance Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload Sensor Data CSV", type=["csv", "txt"])

if uploaded_file is not None:
    # Load uploaded data
    df = pd.read_csv(uploaded_file, sep=",", header=None)
    df.dropna(axis=1, inplace=True)

    # Assign column names (same as training)
    df.columns = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"] + \
                 [f"sensor_measurement_{i}" for i in range(1, 22)]

    # Drop same low variance columns (just like during training)
    low_var_cols = ['sensor_measurement_5', 'sensor_measurement_10', 'sensor_measurement_16',
                    'sensor_measurement_18', 'sensor_measurement_19']
    
    # Only drop if those columns actually exist
    low_var_cols = [col for col in low_var_cols if col in df.columns]
    df.drop(columns=low_var_cols, inplace=True)

    # Normalize features (same scaler used during training)
    features = ['operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + \
               [col for col in df.columns if "sensor_measurement" in col]
    
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    st.success("✅ Data uploaded and preprocessed successfully.")
    st.subheader("📈 Sensor Trend Visualization (First Engine)")

    # Show trend of a sample engine
    unit_1 = df[df['unit_number'] == df['unit_number'].iloc[0]]
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in features[:3]:  # Plot just first 3 for now
        ax.plot(unit_1["time_in_cycles"], unit_1[col], label=col)
    ax.set_xlabel("Cycles")
    ax.set_ylabel("Sensor Values")
    ax.set_title("Sensor Trend Over Time")
    ax.legend()
    st.pyplot(fig)

    # Predict RUL
    st.subheader("🧠 Predicted RUL per Record")
    df["Predicted_RUL"] = model.predict(df[features])
    st.write(df[["unit_number", "time_in_cycles", "Predicted_RUL"]].tail(10))
