import pandas as pd
import streamlit as st

st.set_page_config(page_title="India Air Quality Forecast", layout="wide")

COLORS = {"Good": "#2e9e4f", "Satisfactory": "#8bc34a", "Moderate": "#d9a800",
          "Poor": "#f28c28", "Very Poor": "#e5432d", "Severe": "#8b1a1a"}
METHODS = {"ridge": "Trained model (Ridge)", "lag_0": "Same as today",
           "roll_7": "7-day average", "roll_30": "30-day average"}


@st.cache_data(ttl=3600)
def load():
    fc = pd.read_csv("data/processed/forecast_latest.csv",
                     parse_dates=["issue_date", "target_date"])
    hist = pd.read_csv("data/processed/openmeteo_daily.csv", parse_dates=["Date"])
    return fc, hist


fc, hist = load()

st.title("India air quality forecast")
st.caption("PM2.5-based AQI (India CPCB scale) for 10 cities, 1 to 7 days ahead. "
           "Values are modelled, not measured by ground stations.")

city = st.sidebar.selectbox("City", sorted(fc["city"].unique()),
                            index=sorted(fc["city"].unique()).index("Delhi"))
f = fc[fc["city"] == city].sort_values("h")
issue = f["issue_date"].iloc[0]
today_pm = f["pm25_today"].iloc[0]

st.subheader(f"{city}")
st.write(f"Latest complete day in the data: **{issue.date()}**, "
         f"PM2.5 **{today_pm:.0f} µg/m³**. Forecasts below are for the next days.")

cols = st.columns(4)
for col, (_, r) in zip(cols, f.iterrows()):
    color = COLORS[r["category"]]
    col.markdown(
        f"<div style='border-radius:10px;padding:14px;background:{color};color:white'>"
        f"<div style='font-size:13px'>{r['target_date'].strftime('%a %d %b')} "
        f"({int(r['h'])} day{'s' if r['h'] > 1 else ''} ahead)</div>"
        f"<div style='font-size:34px;font-weight:700'>{int(r['aqi'])}</div>"
        f"<div style='font-size:15px'>{r['category']}</div>"
        f"<div style='font-size:12px;margin-top:4px'>PM2.5 {r['pm25_forecast']:.0f} µg/m³</div>"
        f"</div>", unsafe_allow_html=True)

# chart: last 60 days of PM2.5 plus the forecast
past = (hist[hist["City"] == city].set_index("Date")["pm2_5"]
        .loc[issue - pd.Timedelta(days=59):issue])
days = pd.date_range(issue, issue + pd.Timedelta(days=7))
line = pd.Series(index=days, dtype=float)
line.loc[issue] = today_pm
for _, r in f.iterrows():
    line.loc[r["target_date"]] = r["pm25_forecast"]
line = line.interpolate(limit_area="inside")

chart = pd.concat([past.rename("Past PM2.5 (modelled)"),
                   line.rename("Forecast")], axis=1)
st.line_chart(chart)
st.caption("The forecast line is drawn straight between the four forecast days "
           "(1, 3, 5 and 7 days ahead).")

with st.expander("Details for each forecast"):
    t = f[["target_date", "h", "pm25_forecast", "aqi", "category",
           "method", "open_meteo_pm25"]].copy()
    t["target_date"] = t["target_date"].dt.date
    t["method"] = t["method"].map(METHODS)
    t.columns = ["Date", "Days ahead", "PM2.5 forecast", "AQI", "Category",
                 "Method used", "Open-Meteo's own PM2.5 forecast"]
    st.dataframe(t, hide_index=True)
    st.write("Open-Meteo's forecast is shown only for days where it has "
             "complete data, so some cells are empty.")

with st.expander("How to read this, and its limits"):
    st.markdown("""
- **Method:** for each city and horizon, a simple trained model is used only if it beat
  a naive average in **two separate test years**. Otherwise the forecast is a plain
  average. Gains over the naive baseline were modest (about 2 to 10% at 1 day and
  up to about 15% at longer horizons in northern cities).
- **The AQI shown uses PM2.5 only.** India's official AQI takes the worst of several
  pollutants, so the official number can be higher on days when another pollutant dominates.
- **Modelled data:** values come from the CAMS atmospheric model via Open-Meteo, not
  from ground stations, and may differ from the numbers in official apps.
- **PM10 was excluded** because the model data shows a sudden jump in March 2025
  in Delhi and Gurugram that I could not explain.
""")

st.caption("Data: Open-Meteo Air Quality API, based on Copernicus Atmosphere "
           "Monitoring Service (CAMS) data. Forecasts generated "
           f"{fc['made_at'].iloc[0]}.")