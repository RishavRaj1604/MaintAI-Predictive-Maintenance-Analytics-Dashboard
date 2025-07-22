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

# Premium frontend styling (🔥 Tagda UI)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        background-attachment: fixed;
        padding: 20px;
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    .main-title {
        text-align: center;
        font-size: 3.8rem;
        font-family: 'Orbitron', sans-serif;
        color: #00fff7;
        text-shadow: 0 0 10px #00fff7, 0 0 20px #00fff7;
        margin-top: 30px;
    }

    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #ffffffbb;
        margin-bottom: 40px;
    }

    .section {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 30px;
    }

    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #ccc;
        margin-top: 50px;
        padding-bottom: 20px;
    }

    .stFileUploader > div {
        border: 2px dashed #00fff7;
        padding: 15px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.05);
    }

    .stButton>button {
        background-color: #00fff7;
        color: #000;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.4em;
        font-size: 1rem;
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #00c8ff;
        transform: scale(1.05);
    }

    .stDataFrame {
        background-color: white !important;
        border-radius: 10px;
        overflow: hidden;
    }

    .css-1v0mbdj.eknhn3m4 {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# App heading
st.markdown("<div class='main-title'>🛠️ MaintAI – Predictive Maintenance Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload CMAPSS sensor data, visualize engine trends & predict Remaining Useful Life (RUL)</div>", unsafe_allow_html=True)

# Load trained model
try:
    model = joblib.load("xgb_rul_model.pkl")
except:
    st.error("⚠️ Model file `xgb_rul_model.pkl` not found in working directory.")
    st.stop()

# File uploader section
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📂 Upload your CMAPSS .txt or .csv file", type=["txt", "csv"])
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        try:
            df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None)
        except:
            df = pd.read_csv(uploaded_file, sep=",", header=None)

        df.dropna(axis=1, inplace=True)

        expected_cols = ["unit_number", "time_in_cycles", "operational_setting_1", "operational_setting_2", "operational_setting_3"] + \
                        [f"sensor_measurement_{i}" for i in range(1, 22)]

        if df.shape[1] != len(expected_cols):
            st.error(f"❌ Uploaded file has {df.shape[1]} columns, but {len(expected_cols)} expected.")
            st.stop()

        df.columns = expected_cols

        low_var_cols = ['sensor_measurement_5', 'sensor_measurement_10',
                        'sensor_measurement_16', 'sensor_measurement_18', 'sensor_measurement_19']
        df.drop(columns=[col for col in low_var_cols if col in df.columns], inplace=True)

        features = ['operational_setting_1', 'operational_setting_2', 'operational_setting_3'] + \
                   [col for col in df.columns if "sensor_measurement" in col]

        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        st.success("✅ File uploaded and preprocessed successfully!")

        # Sensor Trend Plot
        st.markdown("<div class='section'>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)

        # Predict RUL
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("🧠 Predicted Remaining Useful Life (RUL)")
        df["Predicted_RUL"] = model.predict(df[features])
        st.dataframe(df[["unit_number", "time_in_cycles", "Predicted_RUL"]].tail(10))
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")
        st.stop()

# Footer
st.markdown("<div class='footer'>© 2025 MaintAI Project | Group 18 </div>", unsafe_allow_html=True)
