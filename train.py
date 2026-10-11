"""AQI forecasting for Indian cities: 1 to 7 days ahead.

Dataset : "Air Quality Data in India (2015-2020)" on Kaggle (file: city_day.csv)
Task    : for each horizon h = 1..7, predict the AQI h days from now
Split   : chronological (first 80% train, last 20% test) - no random shuffling
Choice  : for every horizon the best model is chosen on a validation slice taken
          from the END of the training data, so the test set stays untouched.

Usage
-----
    python train.py --data data/city_day.csv --city Lucknow
    python train.py --data data/city_day.csv --list-cities
"""
import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aqi_utils import (
    HORIZONS,
    choose_pollutants,
    clean_city_data,
    feature_names,
    make_features,
    make_target,
    missing_report,
    regression_metrics,
)

OUT_DIR = Path("outputs")
FIG_DIR = OUT_DIR / "figures"
KAGGLE_SLUG = "rohanrao/air-quality-data-in-india"
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def get_csv_path(path_arg):
    """Use the given CSV, otherwise download the dataset from Kaggle."""
    if path_arg:
        return Path(path_arg)
    try:
        import kagglehub

        folder = kagglehub.dataset_download(KAGGLE_SLUG)
        return Path(folder) / "city_day.csv"
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Could not download from Kaggle automatically.\n"
            f"Reason: {exc}\n"
            f"Download city_day.csv manually from kaggle.com/datasets/{KAGGLE_SLUG}\n"
            "and run:  python train.py --data path/to/city_day.csv"
        )


# --------------------------------------------------------------------------
# 2. Exploratory data analysis (figures for the report)
# --------------------------------------------------------------------------
def save_eda_plots(df, city, pollutants):
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df["AQI"], alpha=0.35, label="Daily AQI")
    ax.plot(df.index, df["AQI"].rolling(30).mean(), color="crimson", label="30-day mean")
    ax.set(title=f"AQI over time - {city}", xlabel="Date", ylabel="AQI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_aqi_over_time.png", dpi=150)
    plt.close(fig)

    monthly = df["AQI"].groupby(df.index.month).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=monthly.index, y=monthly.values, color="steelblue", ax=ax)
    ax.set(title=f"Average AQI by month - {city}", xlabel="Month", ylabel="Mean AQI")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_monthly_seasonality.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(df[pollutants + ["AQI"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title(f"Correlation between pollutants and AQI - {city}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_correlation_heatmap.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------
def build_models():
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
        )
    except ImportError:
        print("xgboost not installed - skipping XGBoost (pip install xgboost)")
    return models


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def train_one_horizon(X_tr, y_tr, X_te, y_te, h):
    """Pick the best model on a validation slice, then evaluate everything on test."""
    # validation = last 15% of training rows; a gap of h rows keeps the
    # training targets from overlapping the validation period
    v0 = int(len(X_tr) * 0.85)
    X_sub, y_sub = X_tr.iloc[: max(v0 - h, 1)], y_tr.iloc[: max(v0 - h, 1)]
    X_val, y_val = X_tr.iloc[v0:], y_tr.iloc[v0:]

    val_scores = {}
    for name, model in build_models().items():
        m = clone(model).fit(X_sub, y_sub)
        val_scores[name] = rmse(y_val, m.predict(X_val))
    chosen = min(val_scores, key=val_scores.get)

    fitted, preds, rows = {}, {}, []
    rows.append({"Horizon_days": h, "Model": "Persistence baseline", "Chosen": False,
                 **regression_metrics(y_te, X_te["AQI_today"])})
    for name, model in build_models().items():
        m = clone(model).fit(X_tr, y_tr)        # final fit on ALL training rows
        fitted[name], preds[name] = m, m.predict(X_te)
        rows.append({"Horizon_days": h, "Model": name, "Chosen": name == chosen,
                     **regression_metrics(y_te, preds[name])})
    return chosen, fitted, preds, rows


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="1-7 day AQI forecasting")
    parser.add_argument("--data", help="path to city_day.csv (optional)")
    parser.add_argument("--city", default="Lucknow", help="city name, e.g. Lucknow, Delhi")
    parser.add_argument("--list-cities", action="store_true", help="print cities and exit")
    args = parser.parse_args()

    raw = pd.read_csv(get_csv_path(args.data))
    cities = sorted(raw["City"].dropna().unique())
    if args.list_cities:
        print("Cities in this file:\n  " + "\n  ".join(cities))
        return
    if args.city not in cities:
        raise SystemExit(f"City '{args.city}' not found. Available: {', '.join(cities)}")

    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- clean + EDA ------------------------------------------------------
    df = clean_city_data(raw, args.city)
    print(f"{args.city}: {df.index.min().date()} to {df.index.max().date()}, {len(df)} days")
    print("\nMissing data per column (%), after filling gaps of up to 5 days:")
    print(missing_report(df).to_string())

    if df["AQI"].isna().mean() > 0.5:
        raise SystemExit(f"\nAQI is missing for more than half the days in {args.city}. "
                         "Try another city (see --list-cities).")
    pollutants, skipped = choose_pollutants(df)
    if skipped:
        print(f"\nSkipping pollutants with too much missing data: {', '.join(skipped)}")
    if not pollutants:
        raise SystemExit("No usable pollutant columns for this city. This happens when the file has only AQI "
                         "for most days (for example newer years added without pollutants). "
                         "Try another city, or ask for the AQI-only mode.")
    print(f"Using pollutants: {', '.join(pollutants)}")
    features = feature_names(pollutants)
    save_eda_plots(df.dropna(subset=["AQI"]), args.city, pollutants)

    # ---- features + one chronological split date for every horizon ----------
    X = make_features(df, pollutants)
    if len(X) < 200:
        raise SystemExit(f"\nOnly {len(X)} usable rows left for {args.city} - too few to train. "
                         "Check the missing-data table above or try another city.")
    n_aqi_days = int(df["AQI"].notna().sum())
    lost = 1 - len(X) / max(n_aqi_days, 1)
    if lost > 0.25:
        print(f"\nWARNING: {lost:.0%} of the days with an AQI value cannot be used, because pollutant "
              f"readings are missing on those days. Usable data ends on {X.index.max().date()} although the "
              f"file runs to {df.index.max().date()}. If you added newer years, they may only contain AQI "
              "(no pollutants) - tell me and I will add an AQI-only mode.")
    split_date = X.index[int(len(X) * 0.8)]
    print(f"\nTest period starts on {split_date.date()} (everything before it is training data)")

    # ---- train one model per horizon -------------------------------------
    models, names, metrics, base_metrics, all_rows = {}, {}, {}, {}, []
    keep_for_plots = {}
    for h in HORIZONS:
        y = make_target(df, h).reindex(X.index)
        train_mask = (X.index < split_date - pd.Timedelta(days=h)) & y.notna().values
        test_mask = (X.index >= split_date) & y.notna().values
        if train_mask.sum() < 100 or test_mask.sum() < 30:
            raise SystemExit(f"Not enough rows for the {h}-day horizon. Try another city.")
        X_tr, y_tr = X.loc[train_mask, features], y[train_mask]
        X_te, y_te = X.loc[test_mask, features], y[test_mask]

        chosen, fitted, preds, rows = train_one_horizon(X_tr, y_tr, X_te, y_te, h)
        models[h], names[h] = fitted[chosen], chosen
        metrics[h] = next(r for r in rows if r["Model"] == chosen)
        base_metrics[h] = next(r for r in rows if r["Model"] == "Persistence baseline")
        all_rows += rows
        if h == 1:
            keep_for_plots = {"fitted": fitted, "preds": preds[chosen], "y_te": y_te, "chosen": chosen}
        print(f"  day +{h}: chosen {chosen:<18} test RMSE {metrics[h]['RMSE']:.1f} "
              f"(baseline {base_metrics[h]['RMSE']:.1f})")

    # ---- result tables -----------------------------------------------------
    results = pd.DataFrame(all_rows).round(3)
    results.to_csv(OUT_DIR / f"results_{args.city}.csv", index=False)

    summary = pd.DataFrame({
        "Days_ahead": HORIZONS,
        "Chosen_model": [names[h] for h in HORIZONS],
        "RMSE": [metrics[h]["RMSE"] for h in HORIZONS],
        "Baseline_RMSE": [base_metrics[h]["RMSE"] for h in HORIZONS],
        "R2": [metrics[h]["R2"] for h in HORIZONS],
        "Category_Acc": [metrics[h]["Category_Acc"] for h in HORIZONS],
    })
    summary["Improvement_%"] = ((1 - summary["RMSE"] / summary["Baseline_RMSE"]) * 100).round(1)
    summary = summary.round(3)
    print("\nTest-set summary (lower RMSE is better; baseline = 'future AQI = today's AQI'):")
    print(summary.to_string(index=False))
    summary.to_csv(OUT_DIR / f"summary_{args.city}.csv", index=False)

    # ---- figures for the report ----------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4))
    y1 = keep_for_plots["y_te"]
    ax.plot(y1.index, y1.values, label="Actual", alpha=0.8)
    ax.plot(y1.index, keep_for_plots["preds"], label=f"Predicted ({keep_for_plots['chosen']})", alpha=0.8)
    ax.set(title=f"Next-day AQI: actual vs predicted (test period) - {args.city}",
           xlabel="Date", ylabel="AQI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    if "Random Forest" in keep_for_plots["fitted"]:
        rf = keep_for_plots["fitted"]["Random Forest"]
        imp = pd.Series(rf.feature_importances_, index=features).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.plot.barh(ax=ax, color="seagreen")
        ax.set(title="Feature importance (next-day, Random Forest)", xlabel="Importance")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "05_feature_importance.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(summary["Days_ahead"], summary["RMSE"], marker="o", label="Chosen model")
    ax.plot(summary["Days_ahead"], summary["Baseline_RMSE"], marker="s", linestyle="--",
            label="Baseline (future = today)")
    ax.set(title=f"Forecast error grows with horizon - {args.city}",
           xlabel="Days ahead", ylabel="RMSE (AQI points)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_error_by_horizon.png", dpi=150)
    plt.close(fig)

    # ---- save what the app needs --------------------------------------------
    hist = df[[c for c in pollutants + ["AQI"]]].copy()
    hist.index.name = "Date"
    hist.to_csv(OUT_DIR / f"history_{args.city}.csv")

    artifact = {
        "city": args.city,
        "features": features,
        "pollutants": pollutants,
        "horizons": list(HORIZONS),
        "models": models,
        "model_names": names,
        "metrics": metrics,
        "baseline_metrics": base_metrics,
        "defaults": X.loc[X.index < split_date, features].median().to_dict(),
        "split_date": str(split_date.date()),
    }
    # compress=3 keeps the file small enough for GitHub (limit: 25 MB per file in the browser)
    joblib.dump(artifact, OUT_DIR / f"model_{args.city}.joblib", compress=3)
    print(f"\nSaved in outputs/: model_{args.city}.joblib, history_{args.city}.csv, "
          f"results_{args.city}.csv, summary_{args.city}.csv, figures/")


if __name__ == "__main__":
    main()
