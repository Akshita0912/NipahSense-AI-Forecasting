
from __future__ import annotations
import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

@dataclass
class Config:
    data_dir: str = "data"
    outputs_dir: str = "outputs"
    models_dir: str = "models"
    reports_dir: str = "reports"

    # Outbreak threshold definition
    outbreak_k_sigma: float = 2.0  # mean + k*std

    # RandomForest to avoid suspicious 100% (controls complexity)
    rf_n_estimators: int = 150
    rf_max_depth: int = 6
    rf_min_samples_split: int = 6
    rf_random_state: int = 42
    decision_threshold: float = 0.55  # probability threshold for classification

    # ARIMA order
    arima_order: Tuple[int, int, int] = (2, 1, 2)
    forecast_horizon_days: int = 14

    # Feature set
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
    dfs = {k: pd.read_csv(v) for k, v in paths.items()}
    return dfs


def clean_and_standardize(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    df_cases = dfs["cases"].copy()
    df_climate = dfs["climate"].copy()
    df_pop = dfs["population"].copy()
    df_infra = dfs["infra"].copy()
    df_bat = dfs["bat"].copy()

    # Dates to datetime
    for df in [df_cases, df_climate, df_bat]:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop duplicates
    df_cases = df_cases.drop_duplicates()
    df_climate = df_climate.drop_duplicates()
    df_bat = df_bat.drop_duplicates()

    # Forward fill climate/bat missing
    df_climate = df_climate.sort_values(["district", "date"]).ffill()
    df_bat = df_bat.sort_values(["district", "date"]).ffill()

    # Ensure numeric types
    num_cols_cases = ["confirmed_cases", "deaths", "recoveries"]
    for c in num_cols_cases:
        if c in df_cases.columns:
            df_cases[c] = pd.to_numeric(df_cases[c], errors="coerce").fillna(0).astype(int)

    for c in ["temperature", "humidity", "rainfall"]:
        if c in df_climate.columns:
            df_climate[c] = pd.to_numeric(df_climate[c], errors="coerce")

    for c in ["population", "population_density"]:
        if c in df_pop.columns:
            df_pop[c] = pd.to_numeric(df_pop[c], errors="coerce")

    for c in ["hospital_beds", "icu_beds", "isolation_units"]:
        if c in df_infra.columns:
            df_infra[c] = pd.to_numeric(df_infra[c], errors="coerce")

    if "bat_infection_index" in df_bat.columns:
        df_bat["bat_infection_index"] = pd.to_numeric(df_bat["bat_infection_index"], errors="coerce")

    # Basic missing handling
    df_cases = df_cases.dropna(subset=["date", "district"])
    df_climate = df_climate.dropna(subset=["date", "district"])
    df_bat = df_bat.dropna(subset=["date", "district"])
    df_pop = df_pop.dropna(subset=["district"])
    df_infra = df_infra.dropna(subset=["district"])

    return {"cases": df_cases, "climate": df_climate, "population": df_pop, "infra": df_infra, "bat": df_bat}


def iqr_filter_cases(df_cases: pd.DataFrame) -> pd.DataFrame:
    if df_cases.empty:
        return df_cases
    q1 = df_cases["confirmed_cases"].quantile(0.25)
    q3 = df_cases["confirmed_cases"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df_cases[(df_cases["confirmed_cases"] >= lower) & (df_cases["confirmed_cases"] <= upper)].copy()


def merge_temporal(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df_cases = dfs["cases"]
    df_climate = dfs["climate"]
    df_pop = dfs["population"]
    df_infra = dfs["infra"]
    df_bat = dfs["bat"]

    # Merge on date + district (temporal alignment)
    df = df_cases.merge(df_climate, on=["date", "district"], how="left")
    df = df.merge(df_pop, on="district", how="left")
    df = df.merge(df_infra, on="district", how="left")
    df = df.merge(df_bat, on=["date", "district"], how="left")

    # Fill remaining missing
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["district", "date"]).copy()

    # Growth rate (safe)
    df["growth_rate"] = df.groupby("district")["confirmed_cases"].pct_change()
    df["growth_rate"] = df["growth_rate"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Moving average 7
    df["moving_avg_7"] = (
        df.groupby("district")["confirmed_cases"]
          .rolling(7).mean()
          .reset_index(level=0, drop=True)
          .fillna(0)
    )

    # Hospital stress (safe division)
    df["hospital_stress"] = np.where(
        df["hospital_beds"].astype(float) > 0,
        df["confirmed_cases"].astype(float) / df["hospital_beds"].astype(float),
        0.0
    )

    # Numeric safety clip only numeric cols
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(-1e6, 1e6)

    return df


def create_outbreak_label(df: pd.DataFrame, k_sigma: float) -> pd.DataFrame:
    # district-wise threshold: mean + k*std (more realistic)
    stats = df.groupby("district")["confirmed_cases"].agg(["mean", "std"]).reset_index()
    stats["outbreak_threshold"] = stats["mean"] + k_sigma * stats["std"]
    df = df.merge(stats[["district", "outbreak_threshold"]], on="district", how="left")
    df["outbreak_flag"] = (df["confirmed_cases"] > df["outbreak_threshold"]).astype(int)
    return df


def train_outbreak_model(df: pd.DataFrame, cfg: Config) -> Tuple[RandomForestClassifier, Dict]:
    X = df[list(cfg.feature_cols)].copy()
    y = df["outbreak_flag"].copy()

    # handle edge case: if only one class
    strat = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=cfg.rf_random_state, stratify=strat
    )

    clf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        max_depth=cfg.rf_max_depth,
        min_samples_split=cfg.rf_min_samples_split,
        random_state=cfg.rf_random_state
    )
    clf.fit(X_train, y_train)

    # probability thresholding (more realistic)
    probs = clf.predict_proba(X_test)[:, 1]
    y_pred = (probs > cfg.decision_threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
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


def risk_zone_from_prob(p: float) -> str:
    if p < 0.40:
        return "Low"
    if p < 0.70:
        return "Moderate"
    return "High"


def intervention_from_zone(zone: str) -> str:
    if zone == "High":
        return (
            "Activate rapid response; enforce isolation/quarantine; restrict gatherings; "
            "scale ICU/isolation beds; intensive contact tracing; targeted vaccination if available; "
            "risk communication to public."
        )
    if zone == "Moderate":
        return (
            "Increase testing & surveillance; prepare isolation wards; PPE stock check; "
            "targeted advisories; strengthen contact tracing; pre-position resources."
        )
    return "Routine surveillance; public awareness; maintain hygiene advisories; monitor bat spillover signals."


def build_decision_layer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_zone"] = df["outbreak_probability"].apply(risk_zone_from_prob)
    df["recommended_intervention"] = df["risk_zone"].apply(intervention_from_zone)

    # simple infrastructure gap signals
    df["required_beds_est"] = np.ceil(df["confirmed_cases"] * 0.2).astype(int)  # example planning proxy
    df["required_icu_est"] = np.ceil(df["confirmed_cases"] * 0.05).astype(int)

    df["beds_gap"] = (df["required_beds_est"] - df["hospital_beds"].astype(int)).clip(lower=0)
    df["icu_gap"] = (df["required_icu_est"] - df["icu_beds"].astype(int)).clip(lower=0)

    return df


def arima_forecast_per_district(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Forecast per district; returns long format
    out_rows = []
    for district, g in df.sort_values("date").groupby("district"):
        ts = g.set_index("date")["confirmed_cases"].astype(float)

        # require enough points
        if len(ts) < 12:
            continue

        try:
            model = ARIMA(ts, order=cfg.arima_order)
            fit = model.fit()
            fc = fit.forecast(steps=cfg.forecast_horizon_days)

            for d, val in fc.items():
                out_rows.append({
                    "date": pd.to_datetime(d),
                    "district": district,
                    "forecasted_cases": float(max(val, 0.0))
                })
        except Exception:
            # skip problematic series safely
            continue

    return pd.DataFrame(out_rows)


def main() -> None:
    cfg = Config()
    ensure_dirs(cfg)

    # 1) Load
    dfs = load_inputs(cfg)

    # 2) Clean/standardize
    dfs = clean_and_standardize(dfs)

    # 3) IQR outlier removal on cases
    dfs["cases"] = iqr_filter_cases(dfs["cases"])

    # 4) Temporal alignment merge
    df_master = merge_temporal(dfs)

    # 5) Feature engineering
    df_master = feature_engineering(df_master)

    # 6) Outbreak label
    df_master = create_outbreak_label(df_master, cfg.outbreak_k_sigma)

    # 7) Train outbreak detection model + metrics
    clf, metrics = train_outbreak_model(df_master, cfg)

    # 8) Predict probability for all rows
    df_master = predict_outbreak(df_master, clf, cfg)

    # 9) Decision layer: risk zone + interventions + infra gaps
    df_master = build_decision_layer(df_master)

    # 10) Forecast per district (ARIMA)
    forecast_df = arima_forecast_per_district(df_master, cfg)

    # 11) Save artifacts
    joblib.dump(clf, os.path.join(cfg.models_dir, "outbreak_rf.pkl"))

    meta = {
        "feature_cols": list(cfg.feature_cols),
        "rf_params": {
            "n_estimators": cfg.rf_n_estimators,
            "max_depth": cfg.rf_max_depth,
            "min_samples_split": cfg.rf_min_samples_split,
            "random_state": cfg.rf_random_state,
        },
        "decision_threshold": cfg.decision_threshold,
        "outbreak_threshold_def": f"district_mean + {cfg.outbreak_k_sigma}*district_std",
        "arima_order": cfg.arima_order,
        "forecast_horizon_days": cfg.forecast_horizon_days,
    }
    with open(os.path.join(cfg.models_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(cfg.reports_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 12) Save outputs for dashboard
    outbreak_predictions = df_master[[
        "date", "country", "state", "district",
        "confirmed_cases", "deaths", "recoveries",
        "outbreak_threshold", "outbreak_flag",
        "outbreak_probability", "outbreak_pred",
        "risk_zone", "recommended_intervention",
        "hospital_beds", "icu_beds", "isolation_units",
        "beds_gap", "icu_gap",
        "temperature", "humidity", "rainfall",
        "population", "population_density",
        "bat_infection_index",
        "growth_rate", "moving_avg_7", "hospital_stress"
    ]].copy()

    outbreak_predictions.to_csv(os.path.join(cfg.outputs_dir, "outbreak_predictions.csv"), index=False)

    if not forecast_df.empty:
        forecast_df.to_csv(os.path.join(cfg.outputs_dir, "forecast_results.csv"), index=False)
    else:
        # always create file
        pd.DataFrame(columns=["date", "district", "forecasted_cases"]).to_csv(
            os.path.join(cfg.outputs_dir, "forecast_results.csv"), index=False
        )

    # Final dashboard dataset = join predictions + forecast (left join on date+district)
    final_dashboard = outbreak_predictions.merge(
        forecast_df, on=["date", "district"], how="left"
    )
    final_dashboard["forecasted_cases"] = final_dashboard["forecasted_cases"].fillna(0)

    final_dashboard.to_csv(os.path.join(cfg.outputs_dir, "final_dashboard_dataset.csv"), index=False)

    # Console summary
    print("Rows:", len(final_dashboard), "| Columns:", final_dashboard.shape[1])
    print("Saved:")
    print(" - outputs/outbreak_predictions.csv")
    print(" - outputs/forecast_results.csv")
    print(" - outputs/final_dashboard_dataset.csv")
    print(" - models/outbreak_rf.pkl")
    print(" - reports/metrics.json")


if __name__ == "__main__":
    main()
