"""Streamlit app: 7-day AQI forecast with charts.

Run:  streamlit run app.py      (after running train.py at least once)
"""
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from aqi_utils import POLLUTANTS, aqi_bucket, make_features

OUT_DIR = Path("outputs")
COLORS = {
    "Good": "#009865", "Satisfactory": "#98c93c", "Moderate": "#f0c300",
    "Poor": "#f58220", "Very Poor": "#e8262b", "Severe": "#7e0023",
}
BANDS = [(0, 50, "Good"), (50, 100, "Satisfactory"), (100, 200, "Moderate"),
         (200, 300, "Poor"), (300, 400, "Very Poor"), (400, 600, "Severe")]

st.set_page_config(page_title="AQI Forecast", page_icon="🌫️", layout="wide")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_city(city):
    art = joblib.load(OUT_DIR / f"model_{city}.joblib")
    if "models" not in art:
        return art, None, None
    hist = pd.read_csv(OUT_DIR / f"history_{city}.csv", parse_dates=["Date"], index_col="Date")
    hist = hist.asfreq("D")
    feats = make_features(hist, art["pollutants"])
    return art, hist, feats


def forecast(art, row):
    """Predict AQI for every horizon from one feature row (DataFrame with 1 row)."""
    X = row[art["features"]]
    return [float(art["models"][h].predict(X)[0]) for h in art["horizons"]]


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
def make_chart(history, base_date, base_value, fc_values, actual=None):
    rows = [{"Date": d, "AQI": v, "Series": "Past AQI"} for d, v in history.items()]
    fc_dates = [base_date + pd.Timedelta(days=h) for h in range(1, len(fc_values) + 1)]
    rows.append({"Date": base_date, "AQI": base_value, "Series": "Forecast"})
    rows += [{"Date": d, "AQI": v, "Series": "Forecast"} for d, v in zip(fc_dates, fc_values)]
    if actual is not None and len(actual):
        rows.append({"Date": base_date, "AQI": base_value, "Series": "Actual (what happened)"})
        rows += [{"Date": d, "AQI": v, "Series": "Actual (what happened)"} for d, v in actual.items()]
    data = pd.DataFrame(rows)

    ymax = max(120.0, float(data["AQI"].max()) * 1.15)
    bands = pd.DataFrame([{"y0": lo, "y1": hi, "Category": name} for lo, hi, name in BANDS])

    band_layer = (
        alt.Chart(bands)
        .mark_rect(opacity=0.14, clip=True)
        .encode(
            y=alt.Y("y0:Q", scale=alt.Scale(domain=[0, ymax]), title="AQI"),
            y2="y1:Q",
            color=alt.Color("Category:N", legend=None,
                            scale=alt.Scale(domain=list(COLORS), range=list(COLORS.values()))),
        )
    )
    series_colors = {"Past AQI": "#7a7266", "Forecast": "#A8553A", "Actual (what happened)": "#1f6f8b"}
    line_layer = (
        alt.Chart(data)
        .mark_line(point=alt.OverlayMarkDef(size=45), strokeWidth=2.5)
        .encode(
            x=alt.X("Date:T", title=None, axis=alt.Axis(format="%d %b", tickCount=8, labelAngle=0)),
            y=alt.Y("AQI:Q", scale=alt.Scale(domain=[0, ymax])),
            color=alt.Color("Series:N", title=None,
                            scale=alt.Scale(domain=list(series_colors), range=list(series_colors.values()))),
            strokeDash=alt.StrokeDash("Series:N", legend=None,
                                      scale=alt.Scale(domain=list(series_colors),
                                                      range=[[1, 0], [6, 4], [1, 0]])),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("AQI:Q", format=".0f"), "Series:N"],
        )
    )
    return alt.layer(band_layer, line_layer).resolve_scale(color="independent").properties(height=380)


def forecast_table(base_date, fc_values, hist_aqi):
    rows = []
    for h, v in enumerate(fc_values, start=1):
        d = base_date + pd.Timedelta(days=h)
        row = {"Day": f"+{h}", "Date": d.strftime("%a, %d %b %Y"),
               "Forecast AQI": round(v), "Category": aqi_bucket(v)}
        if hist_aqi is not None:
            actual = hist_aqi.get(d)
            row["Actual AQI"] = None if pd.isna(actual) else round(float(actual))
            row["Actual category"] = "-" if pd.isna(actual) else aqi_bucket(float(actual))
        rows.append(row)
    return pd.DataFrame(rows)


def headline_cards(fc_values):
    cols = st.columns(3)
    for col, h, label in zip(cols, (1, 3, 7), ("Tomorrow", "In 3 days", "In 7 days")):
        v = fc_values[h - 1]
        cat = aqi_bucket(v)
        col.markdown(
            f"<div style='padding:0.9rem;border-radius:10px;background:{COLORS[cat]};"
            f"color:{'#222' if cat in ('Moderate', 'Satisfactory') else 'white'};text-align:center'>"
            f"<div style='font-size:0.9rem'>{label}</div>"
            f"<div style='font-size:2rem;font-weight:700'>{v:.0f}</div>"
            f"<div>{cat}</div></div>",
            unsafe_allow_html=True,
        )
    if aqi_bucket(max(fc_values)) in ("Poor", "Very Poor", "Severe"):
        st.warning("Poor air expected within the next 7 days. Sensitive groups should limit outdoor activity.")


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.title("🌫️ 7-Day AQI Forecast")

model_files = sorted(OUT_DIR.glob("model_*.joblib"))
if not model_files:
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

city = st.sidebar.selectbox("City", [f.stem.replace("model_", "") for f in model_files])
art, hist, feats = load_city(city)
if hist is None:
    st.error(f"The saved model for {city} is from an older version. "
             f"Run `python train.py --data data/city_day.csv --city {city}` again.")
    st.stop()

mode = st.sidebar.radio("Mode", ["Explore a date", "Manual what-if"])
st.sidebar.caption(f"Data for {city}: {hist.index.min().date()} to {hist.index.max().date()}.")

if mode == "Explore a date":
    st.write("Pick a date. The model forecasts the next 7 days from the readings of that day, "
             "and you can compare with what actually happened.")
    last_ok = feats.index[-1]
    default_idx = feats.index.get_indexer([last_ok - pd.Timedelta(days=7)], method="pad")[0]
    default = feats.index[max(default_idx, 0)]
    picked = st.slider("Forecast from this date", min_value=feats.index[0].date(),
                       max_value=feats.index[-1].date(), value=default.date(), format="DD MMM YYYY")
    idx = feats.index.get_indexer([pd.Timestamp(picked)], method="pad")[0]
    base = feats.index[max(idx, 0)]

    fc = forecast(art, feats.loc[[base]])
    headline_cards(fc)

    window = hist["AQI"].loc[base - pd.Timedelta(days=45): base].dropna()
    future_dates = [base + pd.Timedelta(days=h) for h in art["horizons"]]
    actual = hist["AQI"].reindex(future_dates).dropna()
    st.altair_chart(make_chart(window, base, float(feats.loc[base, "AQI_today"]), fc, actual))

    if base >= pd.Timestamp(art["split_date"]):
        st.success(f"{base.date()} is in the test period, so the model never saw these days while training: "
                   "a fair check of accuracy.")
    else:
        st.info(f"{base.date()} is inside the training period, so this comparison looks better than "
                f"real-world accuracy. For a fair check, pick a date after {art['split_date']}.")
    st.dataframe(forecast_table(base, fc, hist["AQI"]), hide_index=True)

else:
    st.write("Type your own readings to see how the 7-day forecast changes (what-if analysis).")
    d = art["defaults"]
    st.subheader("Today's readings")
    cols = st.columns(3)
    values = {}
    pollutants = [p for p in POLLUTANTS if p in art["features"]]
    for i, p in enumerate(pollutants):
        values[p] = cols[i % 3].number_input(p, min_value=0.0, value=float(round(d[p], 1)), step=1.0)
    st.subheader("Recent AQI")
    c1, c2, c3, c4 = st.columns(4)
    values["AQI_today"] = c1.number_input("Today", 0.0, 1000.0, float(round(d["AQI_today"])))
    values["AQI_lag1"] = c2.number_input("Yesterday", 0.0, 1000.0, float(round(d["AQI_lag1"])))
    values["AQI_lag2"] = c3.number_input("2 days ago", 0.0, 1000.0, float(round(d["AQI_lag2"])))
    values["AQI_roll7"] = c4.number_input("7-day average", 0.0, 1000.0, float(round(d["AQI_roll7"])))
    month = st.selectbox("Month", list(range(1, 13)), index=int(d["month"]) - 1)
    values["month"] = month
    values["is_winter"] = int(month in (11, 12, 1, 2))

    fc = forecast(art, pd.DataFrame([values]))
    headline_cards(fc)
    today = pd.Timestamp.today().normalize()
    st.altair_chart(make_chart(pd.Series(dtype=float), today, values["AQI_today"], fc))
    st.dataframe(forecast_table(today, fc, None), hide_index=True)

with st.expander("How accurate is this model?"):
    acc = pd.DataFrame({
        "Days ahead": art["horizons"],
        "Model error (RMSE)": [art["metrics"][h]["RMSE"] for h in art["horizons"]],
        "Baseline error (future = today)": [art["baseline_metrics"][h]["RMSE"] for h in art["horizons"]],
    }).set_index("Days ahead")
    st.line_chart(acc)
    st.caption("RMSE is the typical forecast error in AQI points (lower is better), measured on days "
               "the model never saw. Error grows the further ahead we look. The baseline simply assumes "
               "the AQI stays the same as today.")

st.caption("Forecasts are estimates and get less reliable on days with sudden weather changes. "
           f"This demo uses data from {hist.index.min().year} to {hist.index.max().year}, "
           "so it shows past dates, not a live forecast.")
