from __future__ import annotations
import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


@dataclass
class Config:
    base_dir: str = os.path.dirname(os.path.abspath(__file__))

    data_dir: str = os.path.join(base_dir, "data")
    outputs_dir: str = os.path.join(base_dir, "outputs")
    models_dir: str = os.path.join(base_dir, "models")
    reports_dir: str = os.path.join(base_dir, "reports")

    rf_n_estimators: int = 250
    rf_max_depth: int = 8
    rf_min_samples_split: int = 4
    rf_random_state: int = 42
    decision_threshold: float = 0.40

    arima_order: Tuple[int, int, int] = (2, 1, 2)
    forecast_horizon_days: int = 14

    feature_cols: Tuple[str, ...] = (
        "temperature", "humidity", "rainfall",
        "bat_infection_index", "population_density",
        "growth_rate", "moving_avg_7", "hospital_stress"
    )


def ensure_dirs(cfg: Config) -> None:
    for d in [cfg.outputs_dir, cfg.models_dir, cfg.reports_dir]:
        os.makedirs(d, exist_ok=True)


def load_inputs(cfg: Config) -> Dict[str, pd.DataFrame]:
    paths = {
        "cases": os.path.join(cfg.data_dir, "human_cases.csv"),
        "climate": os.path.join(cfg.data_dir, "climate_data.csv"),
        "population": os.path.join(cfg.data_dir, "population.csv"),
        "infra": os.path.join(cfg.data_dir, "infrastructure.csv"),
        "bat": os.path.join(cfg.data_dir, "bat_surveillance.csv"),
    }
    return {k: pd.read_csv(v) for k, v in paths.items()}


def clean_and_standardize(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    df_cases = dfs["cases"].copy()
    df_climate = dfs["climate"].copy()
    df_pop = dfs["population"].copy()
    df_infra = dfs["infra"].copy()
    df_bat = dfs["bat"].copy()

    for df in [df_cases, df_climate, df_bat]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df_cases.drop_duplicates(inplace=True)
    df_climate.drop_duplicates(inplace=True)
    df_bat.drop_duplicates(inplace=True)

    df_climate = df_climate.sort_values(["district", "date"]).ffill()
    df_bat = df_bat.sort_values(["district", "date"]).ffill()

    for c in ["confirmed_cases", "deaths", "recoveries"]:
        if c in df_cases.columns:
            df_cases[c] = pd.to_numeric(df_cases[c], errors="coerce").fillna(0)

    for c in ["temperature", "humidity", "rainfall"]:
        if c in df_climate.columns:
            df_climate[c] = pd.to_numeric(df_climate[c], errors="coerce").fillna(0)

    for c in ["population", "population_density"]:
        if c in df_pop.columns:
            df_pop[c] = pd.to_numeric(df_pop[c], errors="coerce").fillna(0)

    for c in ["hospital_beds", "icu_beds", "isolation_units"]:
        if c in df_infra.columns:
            df_infra[c] = pd.to_numeric(df_infra[c], errors="coerce").fillna(0)

    if "bat_infection_index" in df_bat.columns:
        df_bat["bat_infection_index"] = pd.to_numeric(df_bat["bat_infection_index"], errors="coerce").fillna(0)

    df_cases = df_cases.dropna(subset=["date", "district"])
    df_climate = df_climate.dropna(subset=["date", "district"])
    df_bat = df_bat.dropna(subset=["date", "district"])
    df_pop = df_pop.dropna(subset=["district"])
    df_infra = df_infra.dropna(subset=["district"])

    return {
        "cases": df_cases,
        "climate": df_climate,
        "population": df_pop,
        "infra": df_infra,
        "bat": df_bat
    }


def iqr_filter_cases(df_cases: pd.DataFrame) -> pd.DataFrame:
    q1 = df_cases["confirmed_cases"].quantile(0.25)
    q3 = df_cases["confirmed_cases"].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return df_cases[
        (df_cases["confirmed_cases"] >= lower) &
        (df_cases["confirmed_cases"] <= upper)
    ].copy()


def merge_temporal(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = dfs["cases"].merge(dfs["climate"], on=["date", "district"], how="left")
    df = df.merge(dfs["population"], on="district", how="left")
    df = df.merge(dfs["infra"], on="district", how="left")
    df = df.merge(dfs["bat"], on=["date", "district"], how="left")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    if "country" not in df.columns:
        df["country"] = "India"
    if "state" not in df.columns:
        df["state"] = "Unknown"

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["district", "date"]).copy()

    df["growth_rate"] = df.groupby("district")["confirmed_cases"].pct_change()
    df["growth_rate"] = df["growth_rate"].replace([np.inf, -np.inf], np.nan).fillna(0)

    df["moving_avg_7"] = (
        df.groupby("district")["confirmed_cases"]
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    df["hospital_stress"] = np.where(
        df["hospital_beds"].astype(float) > 0,
        df["confirmed_cases"].astype(float) / df["hospital_beds"].astype(float),
        0
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(-1e6, 1e6)

    return df


def create_outbreak_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    district_q = df.groupby("district")["confirmed_cases"].quantile(0.70).reset_index()
    district_q.columns = ["district", "district_case_threshold"]

    df = df.merge(district_q, on="district", how="left")

    df["outbreak_flag"] = np.where(
        (
            (df["confirmed_cases"] >= df["district_case_threshold"]) &
            (
                (df["growth_rate"] > 0.08) |
                (df["bat_infection_index"] > 0.45) |
                (df["hospital_stress"] > 0.01)
            )
        ),
        1,
        0
    )

    return df


def train_outbreak_model(df: pd.DataFrame, cfg: Config):
    X = df[list(cfg.feature_cols)].copy()
    y = df["outbreak_flag"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=cfg.rf_random_state,
        stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_split=cfg.rf_min_samples_split,
        random_state=cfg.rf_random_state,
        class_weight="balanced_subsample"
    )

    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > cfg.decision_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== OUTBREAK DETECTION RESULTS =====")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Outbreak", "Outbreak"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix - Outbreak Detection")
    plt.tight_layout()
    plt.show()

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "decision_threshold": cfg.decision_threshold,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
    }

    return clf, metrics


def predict_outbreak(df: pd.DataFrame, clf: RandomForestClassifier, cfg: Config) -> pd.DataFrame:
    X_all = df[list(cfg.feature_cols)].copy()
    df["outbreak_probability"] = clf.predict_proba(X_all)[:, 1]
    df["outbreak_pred"] = (df["outbreak_probability"] > cfg.decision_threshold).astype(int)
    return df


def risk_zone_from_prob(prob: float) -> str:
    if prob < 0.40:
        return "Low"
    elif prob < 0.70:
        return "Moderate"
    return "High"


def intervention_from_zone(zone: str) -> str:
    if zone == "High":
        return "Activate rapid response, restrict mobility, scale ICU & isolation, intensive contact tracing."
    elif zone == "Moderate":
        return "Increase testing, strengthen surveillance, prepare isolation wards."
    return "Routine surveillance, public awareness, maintain hygiene."


def build_decision_layer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_zone"] = df["outbreak_probability"].apply(risk_zone_from_prob)
    df["recommended_intervention"] = df["risk_zone"].apply(intervention_from_zone)

    # real-like operational demand estimates
    df["Predicted Cases Proxy"] = np.maximum(
        df["confirmed_cases"] * (1 + df["growth_rate"].clip(lower=0)),
        df["moving_avg_7"]
    )

    df["ICU Beds Required"] = np.ceil(df["Predicted Cases Proxy"] * 0.08).astype(int)
    df["Isolation Beds Required"] = np.ceil(df["Predicted Cases Proxy"] * 0.20).astype(int)
    df["Ventilators Required"] = np.ceil(df["Predicted Cases Proxy"] * 0.03).astype(int)
    df["Vaccines Required"] = np.ceil((df["population"] * 0.01) * df["outbreak_probability"]).astype(int)
    df["PPE Kits Required"] = np.ceil(df["Predicted Cases Proxy"] * 12).astype(int)
    df["Frontline Staff Required"] = np.ceil(df["Predicted Cases Proxy"] / 4).astype(int)

    df["beds_gap"] = (df["Isolation Beds Required"] - df["hospital_beds"]).clip(lower=0)
    df["icu_gap"] = (df["ICU Beds Required"] - df["icu_beds"]).clip(lower=0)

    df["Risk Score"] = (df["outbreak_probability"] * 100).round(2)

    return df

def arima_forecast_per_district(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    forecast_rows = []

    for district, g in df.sort_values("date").groupby("district"):
        ts = g.set_index("date")["confirmed_cases"].astype(float)

        if len(ts) < 12:
            continue

        try:
            model = ARIMA(ts, order=cfg.arima_order)
            fit = model.fit()
            fc = fit.forecast(steps=cfg.forecast_horizon_days)

            # keep first forecast day as main dashboard predicted value
            first_day = fc.index[0]
            first_val = float(max(fc.iloc[0], 0.0))

            forecast_rows.append({
                "district": district,
                "Forecast Date": pd.to_datetime(first_day),
                "Predicted Cases": round(first_val, 2)
            })

        except Exception:
            continue

    return pd.DataFrame(forecast_rows)


def main():
    cfg = Config()
    ensure_dirs(cfg)

    dfs = load_inputs(cfg)
    dfs = clean_and_standardize(dfs)
    dfs["cases"] = iqr_filter_cases(dfs["cases"])

    df_master = merge_temporal(dfs)
    df_master = feature_engineering(df_master)
    df_master = create_outbreak_label(df_master)

    clf, metrics = train_outbreak_model(df_master, cfg)

    df_master = predict_outbreak(df_master, clf, cfg)
    df_master = build_decision_layer(df_master)

    forecast_df = arima_forecast_per_district(df_master, cfg)

    joblib.dump(clf, os.path.join(cfg.models_dir, "outbreak_rf.pkl"))

    metadata = {
        "feature_cols": list(cfg.feature_cols),
        "decision_threshold": cfg.decision_threshold,
        "arima_order": cfg.arima_order,
        "forecast_horizon_days": cfg.forecast_horizon_days
    }

    with open(os.path.join(cfg.models_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(cfg.reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


    latest_rows = df_master.sort_values("date").groupby("district").tail(1).copy()
    final_dashboard = latest_rows.merge(forecast_df, on="district", how="left")

    final_dashboard["Predicted Cases"] = final_dashboard["Predicted Cases"].fillna(final_dashboard["Predicted Cases Proxy"])

    final_dashboard["Region"] = final_dashboard["district"]
    final_dashboard["Disease"] = "Nipah Virus"
    final_dashboard["Human Cases"] = final_dashboard["confirmed_cases"].astype(int)
    final_dashboard["Growth%"] = (final_dashboard["growth_rate"] * 100).round(2)
    final_dashboard["Risk Level"] = final_dashboard["risk_zone"]
    final_dashboard["Recommended Action"] = final_dashboard["recommended_intervention"]

    dashboard_cols = [
        "Region",
        "Disease",
        "date",
        "Human Cases",
        "Predicted Cases",
        "Growth%",
        "Risk Level",
        "Risk Score",
        "ICU Beds Required",
        "Isolation Beds Required",
        "Ventilators Required",
        "Vaccines Required",
        "PPE Kits Required",
        "Frontline Staff Required",
        "Recommended Action",
        "outbreak_probability",
        "temperature",
        "humidity",
        "rainfall",
        "bat_infection_index",
        "population_density",
        "moving_avg_7",
        "hospital_stress"
    ]

    outbreak_predictions = df_master.copy()
    outbreak_predictions.to_csv(os.path.join(cfg.outputs_dir, "outbreak_predictions.csv"), index=False)

    forecast_df.to_csv(os.path.join(cfg.outputs_dir, "forecast_results.csv"), index=False)
    final_dashboard[dashboard_cols].to_csv(os.path.join(cfg.outputs_dir, "final_dashboard_dataset.csv"), index=False)

    print("\n===== PIPELINE COMPLETED =====")
    print("Saved:")
    print(" - outputs/outbreak_predictions.csv")
    print(" - outputs/forecast_results.csv")
    print(" - outputs/final_dashboard_dataset.csv")
    print(" - models/outbreak_rf.pkl")
    print(" - models/metadata.json")
    print(" - reports/metrics.json")


if __name__ == "__main__":
    main()
