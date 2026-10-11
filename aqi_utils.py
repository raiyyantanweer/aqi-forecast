"""Shared helpers for the AQI forecasting project.

Used by both train.py (training) and app.py (Streamlit demo) so the
feature definitions can never drift apart.
"""
import numpy as np
import pandas as pd

# Pollutants used as model inputs (column names as in Kaggle's city_day.csv)
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# Max share of missing days allowed for a pollutant to be used for a city.
# Real data often has pollutants that are empty for some cities; those are skipped.
MAX_MISSING = 0.40

# Forecast horizons (days ahead) the project predicts
HORIZONS = list(range(1, 8))


def feature_names(pollutants):
    """Final feature list for the chosen pollutants. All can be asked from a user in the app."""
    return list(pollutants) + [
        "AQI_today",   # AQI on the day we forecast from
        "AQI_lag1",    # AQI one day before
        "AQI_lag2",    # AQI two days before
        "AQI_roll7",   # mean AQI over the last 7 days (including today)
        "month",       # 1-12
        "is_winter",   # Nov-Feb, the high-pollution season in North India
    ]


# CPCB AQI categories: (upper bound, label)
AQI_BUCKETS = [
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (float("inf"), "Severe"),
]


def aqi_bucket(value: float) -> str:
    """Convert a numeric AQI into the official CPCB category."""
    for upper, label in AQI_BUCKETS:
        if value <= upper:
            return label
    return "Severe"


def clean_city_data(raw: pd.DataFrame, city: str) -> pd.DataFrame:
    """Filter one city, make the time index daily and fill small gaps.

    Why: lag features only make sense if every row is exactly one day apart.
    """
    df = raw[raw["City"] == city].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    cols = [c for c in POLLUTANTS if c in df.columns] + ["AQI"]
    df = df[cols].apply(pd.to_numeric, errors="coerce")

    # One row per calendar day; missing days appear as NaN
    df = df.asfreq("D")

    # Fill only short gaps (max 5 days) - long gaps stay NaN and get dropped,
    # because inventing weeks of pollution data would be misleading.
    df = df.interpolate(method="time", limit=5, limit_area="inside")
    return df


def missing_report(df: pd.DataFrame) -> pd.Series:
    """Percentage of missing days per column (after gap-filling)."""
    return (df.isna().mean() * 100).round(1)


def choose_pollutants(df: pd.DataFrame):
    """Return (usable, skipped): pollutants with <= MAX_MISSING share of missing days."""
    present = [c for c in POLLUTANTS if c in df.columns]
    usable = [c for c in present if df[c].isna().mean() <= MAX_MISSING]
    skipped = [c for c in POLLUTANTS if c not in usable]
    return usable, skipped


def make_features(df: pd.DataFrame, pollutants) -> pd.DataFrame:
    """Build the feature table (one row per day). Targets are built separately."""
    out = pd.DataFrame(index=df.index)
    for col in pollutants:
        out[col] = df[col]

    out["AQI_today"] = df["AQI"]
    out["AQI_lag1"] = df["AQI"].shift(1)
    out["AQI_lag2"] = df["AQI"].shift(2)
    out["AQI_roll7"] = df["AQI"].rolling(7, min_periods=4).mean()
    out["month"] = df.index.month
    out["is_winter"] = df.index.month.isin([11, 12, 1, 2]).astype(int)
    return out.dropna()


def make_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """AQI `horizon` days after each row's date (NaN if unknown)."""
    return df["AQI"].shift(-horizon)


def regression_metrics(y_true, y_pred) -> dict:
    """RMSE, MAE, R2 and 'category accuracy' (did we hit the right AQI band?)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_cat = [aqi_bucket(v) for v in y_true]
    pred_cat = [aqi_bucket(v) for v in y_pred]
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Category_Acc": float(np.mean(np.array(true_cat) == np.array(pred_cat))),
    }
