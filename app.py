import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load model
model = joblib.load("xgb_rul_model.pkl")

st.title("MaintAI – Predictive Maintenance Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload Sensor Data CSV", type=["csv", "txt"])

if uploaded_file is not None:
    # Load uploaded data
    df = pd.read_csv(uploaded_file, sep=" ", header=None)
    df.dropna(axis=1, inplace=True)
    
    # Assign column names (same as training)
    columns = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2",
               "operational_setting_3"] + [f"sensor_measurement_{i}" for i in range(1, 22)]
    df.columns = columns[:df.shape[1]]
    
    # Drop same low variance columns (use list from earlier step)
    low_var_cols = ['sensor_measurement_1', 'sensor_measurement_5', 'sensor_measurement_10', 'sensor_measurement_16', 'sensor_measurement_18', 'sensor_measurement_19']
    df.drop(columns=low_var_cols, inplace=True)
    
    # Normalize features (same scaler used during training)
    features = ['operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + \
               [col for col in df.columns if "sensor_measurement" in col]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    st.success("Data uploaded and preprocessed successfully.")
    st.subheader("Sensor Trend Visualization (First Engine)")

    unit_1 = df[df['unit_number'] == df['unit_number'].iloc[0]]
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in features[:3]:  # Just first 3 for now
        ax.plot(unit_1['time_in_cycles'], unit_1[col], label=col)
    plt.xlabel("Cycles")
    plt.ylabel("Sensor Values")
    plt.title("Sensor Trend Over Time")
    plt.legend()
    st.pyplot(fig)
    st.subheader("Predicted RUL per Record")
    df["Predicted_RUL"] = model.predict(df[features])
    st.write(df[["unit_number", "time_in_cycles", "Predicted_RUL"]].tail(10))
