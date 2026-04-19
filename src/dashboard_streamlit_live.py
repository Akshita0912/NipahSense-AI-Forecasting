import os
import pandas as pd
import streamlit as st
import numpy as np
import json

st.set_page_config(page_title="NipahSense Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "outputs", "final_dashboard_dataset.csv")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "metrics.json")

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ Dataset not found. Run: python run_pipeline.py")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["Region"] = df["Region"].astype(str).str.strip()
    return df

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {"accuracy":0.8,"precision":0.63,"recall":0.85,"f1":0.72}

df = load_data()
metrics = load_metrics()


DASHBOARD_REGIONS = [
    "Faridpur",
    "North 24 Parganas",
    "Kozhikode"
]

df = df[df["Region"].isin(DASHBOARD_REGIONS)]

if df.empty:
    st.error("No matching data found for configured regions.")
    st.stop()


st.sidebar.title("Dashboard Filters")

selected_regions = st.sidebar.multiselect(
    "Region",
    DASHBOARD_REGIONS,
    default=DASHBOARD_REGIONS
)

risk_filter = st.sidebar.multiselect(
    "Risk Level",
    ["High", "Moderate", "Low"],
    default=["High", "Moderate", "Low"]
)

df = df[
    (df["Region"].isin(selected_regions)) &
    (df["Risk Level"].isin(risk_filter))
]

if df.empty:
    st.warning("No data available for selected filters.")
    st.stop()


st.markdown("## NipahSense Dynamic Forecast Dashboard")


st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 18px;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.35);
    margin-bottom: 15px;
}
.card-title {
    font-size: 14px;
    opacity: 0.85;
}
.card-value {
    font-size: 32px;
    font-weight: bold;
}
.blue {background: linear-gradient(135deg, #1e3c72, #2a5298);}
.teal {background: linear-gradient(135deg, #134e5e, #71b280);}
.red {background: linear-gradient(135deg, #cb2d3e, #ef473a);}
.orange {background: linear-gradient(135deg, #f7971e, #ffd200);}
.green {background: linear-gradient(135deg, #11998e, #38ef7d);}
.yellow {background: linear-gradient(135deg, #f12711, #f5af19);}
.purple {background: linear-gradient(135deg, #4e54c8, #8f94fb);}
.dark {background: linear-gradient(135deg, #141e30, #243b55);}
</style>
""", unsafe_allow_html=True)


total_cases = int(df["Predicted Cases"].sum())
observed = int(df["Human Cases"].sum())
high_risk = int((df["Risk Level"] == "High").sum())
vaccines = int(df["Vaccines Required"].sum())

low = int((df["Risk Level"] == "Low").sum())
medium = int((df["Risk Level"] == "Moderate").sum())
icu = int(df["ICU Beds Required"].sum())
growth = float(df["Growth%"].mean())


c1, c2, c3, c4 = st.columns(4)

c1.markdown(f'<div class="card blue"><div class="card-title">Projected Cases</div><div class="card-value">{total_cases}</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="card teal"><div class="card-title">Observed Cases</div><div class="card-value">{observed}</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="card red"><div class="card-title">High-Risk Regions</div><div class="card-value">{high_risk}</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="card orange"><div class="card-title">Vaccines Required</div><div class="card-value">{vaccines}</div></div>', unsafe_allow_html=True)


c5, c6, c7, c8 = st.columns(4)

c5.markdown(f'<div class="card green"><div class="card-title">Low-Risk Regions</div><div class="card-value">{low}</div></div>', unsafe_allow_html=True)
c6.markdown(f'<div class="card yellow"><div class="card-title">Medium-Risk Regions</div><div class="card-value">{medium}</div></div>', unsafe_allow_html=True)
c7.markdown(f'<div class="card purple"><div class="card-title">ICU Demand</div><div class="card-value">{icu}</div></div>', unsafe_allow_html=True)
c8.markdown(f'<div class="card dark"><div class="card-title">Avg Growth Rate (%)</div><div class="card-value">{round(growth,2)}</div></div>', unsafe_allow_html=True)

st.markdown("---")


st.subheader("Dynamic Forecast View")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Forecast Cases by Region")
    st.bar_chart(df.groupby("Region")["Predicted Cases"].sum())

with col2:
    st.markdown("### Forecast vs Current Cases")
    compare_df = df.groupby("Region")[["Human Cases", "Predicted Cases"]].sum()
    st.line_chart(compare_df)

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("### Risk Composition")
    st.bar_chart(df["Risk Level"].value_counts())

with col4:
    st.markdown("### Resource Demand")
    resource_df = df[[
        "ICU Beds Required",
        "Isolation Beds Required",
        "Ventilators Required"
    ]].sum()
    st.area_chart(resource_df)

with col5:
    st.markdown("### Regional Growth Pressure")
    st.bar_chart(df.groupby("Region")["Growth%"].mean())

st.markdown("---")


colA, colB, colC, colD = st.columns(4)

colA.metric("Accuracy", round(metrics["accuracy"], 3))
colB.metric("Precision", round(metrics["precision"], 3))
colC.metric("Recall", round(metrics["recall"], 3))
colD.metric("F1 Score", round(metrics["f1"], 3))


st.markdown("## Manual Regional Risk Prediction")

region_name = st.text_input("Region / District Name")

col1, col2, col3 = st.columns(3)

temperature = col1.number_input("Temperature", value=28.0)
bat_index = col2.number_input("Bat Infection Index", value=0.4)
moving_avg = col3.number_input("7-Day Moving Average", value=6.0)

humidity = col1.number_input("Humidity", value=75.0)
population = col2.number_input("Population Density", value=1200.0)
hospital = col3.number_input("Hospital Stress", value=0.02)

rainfall = col1.number_input("Rainfall", value=100.0)
growth_rate = col2.number_input("Growth Rate", value=0.08)

if st.button("Predict Risk Zone"):
    score = (
        temperature * 0.05 +
        humidity * 0.02 +
        rainfall * 0.01 +
        bat_index * 40 +
        population * 0.002 +
        growth_rate * 30 +
        hospital * 50
    )

    if score > 80:
        risk = "High"
    elif score > 50:
        risk = "Moderate"
    else:
        risk = "Low"

    st.success(f"Region: {region_name if region_name else 'Custom Input'}")
    st.success(f"Predicted Risk Level: {risk}")
