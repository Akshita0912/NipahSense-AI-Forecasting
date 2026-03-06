import os
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Nipah Virus  Live Dashboard", layout="wide")
st.title("🦠 Nipah Virus — Live Forecasting & Intervention Dashboard")

DATA_PATH = "outputs/final_dashboard_dataset.csv"

st.sidebar.header("Live Settings")
refresh_sec = st.sidebar.slider("Auto-refresh interval (seconds)", 5, 120, 15, 5)

# This triggers a rerun automatically
st_autorefresh(interval=refresh_sec * 1000, key="nipah_autorefresh")

if not os.path.exists(DATA_PATH):
    st.error(f"Missing file: {DATA_PATH}. Run: python src/run_pipeline.py")
    st.stop()

@st.cache_data(ttl=5)  # small cache to reduce flicker, still near-live
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "district"]).copy()
    return df

df = load_data(DATA_PATH)

st.sidebar.header("Filters")

districts = sorted(df["district"].astype(str).unique().tolist())
district = st.sidebar.selectbox("District", districts)

min_date = df["date"].min()
max_date = df["date"].max()

date_range = st.sidebar.date_input(
    "Date Range",
    (min_date.date(), max_date.date()),
)

start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1])

f = df[(df["district"] == district) & (df["date"] >= start) & (df["date"] <= end)].copy()
f = f.sort_values("date")

if f.empty:
    st.warning("No data for the selected district/date range.")
    st.stop()

def latest_and_delta(series: pd.Series):
    series = series.reset_index(drop=True)
    latest = series.iloc[-1]
    if len(series) >= 2:
        prev = series.iloc[-2]
        delta = latest - prev
    else:
        delta = 0
    return latest, delta

latest_row = f.iloc[-1]

cases_latest, cases_delta = latest_and_delta(f["confirmed_cases"])
prob_latest, prob_delta = latest_and_delta(f["outbreak_probability"])
forecast_latest, forecast_delta = latest_and_delta(f["forecasted_cases"])
beds_gap_latest, beds_gap_delta = latest_and_delta(f["beds_gap"])
icu_gap_latest, icu_gap_delta = latest_and_delta(f["icu_gap"])

risk_latest = str(latest_row.get("risk_zone", "NA"))
outbreak_pred_latest = int(latest_row.get("outbreak_pred", 0))

# Top KPI cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Confirmed Cases (Latest)", int(cases_latest), int(cases_delta))
c2.metric("Outbreak Probability", round(float(prob_latest), 3), round(float(prob_delta), 3))
c3.metric("Forecast (Latest Day)", round(float(forecast_latest), 2), round(float(forecast_delta), 2))
c4.metric("Beds Gap", int(beds_gap_latest), int(beds_gap_delta))
c5.metric("ICU Gap", int(icu_gap_latest), int(icu_gap_delta))

# Status banner
status_col1, status_col2 = st.columns([2, 3])
with status_col1:
    if risk_latest == "High":
        st.error(f"🚨 Risk Zone: {risk_latest}  |  Outbreak Pred: {outbreak_pred_latest}")
    elif risk_latest == "Moderate":
        st.warning(f"⚠️ Risk Zone: {risk_latest}  |  Outbreak Pred: {outbreak_pred_latest}")
    else:
        st.success(f"✅ Risk Zone: {risk_latest}  |  Outbreak Pred: {outbreak_pred_latest}")

with status_col2:
    st.info(f"Last updated date in view: **{latest_row['date'].date()}** | Auto-refresh every **{refresh_sec}s**")

st.subheader("✅ Recommended Public Health Intervention")
st.write(str(latest_row.get("recommended_intervention", "NA")))

left, right = st.columns(2)

with left:
    st.subheader("📈 Cases vs Forecast (Time Series)")
    chart_df = f[["date", "confirmed_cases", "forecasted_cases"]].set_index("date")
    st.line_chart(chart_df)

with right:
    st.subheader("📊 Outbreak Probability (Time Series)")
    prob_df = f[["date", "outbreak_probability"]].set_index("date")
    st.line_chart(prob_df)

left2, right2 = st.columns(2)

with left2:
    st.subheader("🏥 Infrastructure Gaps (Beds / ICU)")
    gap_df = f[["date", "beds_gap", "icu_gap"]].set_index("date")
    st.line_chart(gap_df)

with right2:
    st.subheader("🌦️ Environmental Signals")
    env_cols = [c for c in ["temperature", "humidity", "rainfall", "bat_infection_index"] if c in f.columns]
    env_df = f[["date"] + env_cols].set_index("date")
    st.line_chart(env_df)

st.subheader("📋 Live Data Table (Filtered)")
show_cols = [
    "date", "district", "confirmed_cases", "forecasted_cases",
    "outbreak_probability", "risk_zone",
    "beds_gap", "icu_gap",
    "temperature", "humidity", "rainfall", "bat_infection_index"
]
show_cols = [c for c in show_cols if c in f.columns]
st.dataframe(f[show_cols].sort_values("date", ascending=False), use_container_width=True)

st.caption(
    "Live behavior: this dashboard auto-refreshes and re-reads outputs/final_dashboard_dataset.csv. "
    "To make it truly live, re-run the pipeline on a schedule (every few minutes) or feed new data continuously."
)
