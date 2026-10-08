import time
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
DATA = "data/processed/openmeteo_daily.csv"
CITIES = {
    "Delhi": (28.61, 77.21), "Gurugram": (28.46, 77.03),
    "Lucknow": (26.85, 80.95), "Jaipur": (26.91, 75.79),
    "Amritsar": (31.63, 74.87), "Ludhiana": (30.90, 75.86),
    "Bengaluru": (12.97, 77.59), "Chennai": (13.08, 80.27),
    "Hyderabad": (17.39, 78.49), "Thiruvananthapuram": (8.52, 76.94),
}
VARS = "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,us_aqi"
CONC = [0, 30, 60, 90, 120, 250, 380]
IDX = [0, 50, 100, 200, 300, 400, 500]
BASE_FEATS = ["lag_0", "lag_1", "lag_2", "lag_3", "lag_7", "roll_7", "month"]
NAIVE_FOR = {1: "lag_0", 3: "roll_7", 5: "roll_7", 7: "roll_30"}
LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def category(a):
    return LABELS[int(np.searchsorted([50, 100, 200, 300, 400], a, side="left"))]


def get(params):
    for _ in range(3):
        r = requests.get(URL, params=params, timeout=60)
        if r.status_code == 200:
            x = pd.DataFrame(r.json()["hourly"])
            x["time"] = pd.to_datetime(x["time"])
            return x
        time.sleep(5)
    raise RuntimeError(f"Download failed: {r.status_code} {r.text[:100]}")


# ---- 1) add the newest complete days to the data file ----
om = pd.read_csv(DATA, parse_dates=["Date"])
yesterday = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
new_parts = []
for city, (lat, lon) in CITIES.items():
    last = om.loc[om["City"] == city, "Date"].max()
    if last >= yesterday:
        continue
    x = get({"latitude": lat, "longitude": lon, "hourly": VARS,
             "start_date": (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
             "end_date": yesterday.strftime("%Y-%m-%d"),
             "timezone": "Asia/Kolkata"}).dropna(subset=["us_aqi"])
    if x.empty:
        continue
    day = x.set_index("time").resample("D").mean().reset_index()
    day = day.rename(columns={"time": "Date"})
    day.insert(0, "City", city)
    new_parts.append(day)
    time.sleep(1)

n_new = sum(len(p) for p in new_parts)
if new_parts:
    om = pd.concat([om] + new_parts, ignore_index=True)
om["aqi_pm25"] = np.interp(om["pm2_5"], CONC, IDX).round()
om = om.drop_duplicates(["City", "Date"]).sort_values(["City", "Date"])
om.to_csv(DATA, index=False)
print("New city-days added:", n_new, "| data now ends:", om["Date"].max().date())

# ---- 2) make the forecast ----
policy = pd.read_csv("models/policy_pm25.csv")
use = {(r.city, int(r.h)): bool(r.use_ridge) for r in policy.itertuples()}
made_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
rows = []

for city, (lat, lon) in CITIES.items():
    s = om[om["City"] == city].set_index("Date").asfreq("D")
    d = s[["pm2_5"]].copy()
    d["lag_0"] = d["pm2_5"]
    for lag in [1, 2, 3, 7]:
        d[f"lag_{lag}"] = d["pm2_5"].shift(lag)
    d["roll_7"] = d["pm2_5"].rolling(7).mean()
    d["roll_30"] = d["pm2_5"].rolling(30).mean()
    d["month"] = d.index.month
    last = d.dropna(subset=BASE_FEATS + ["roll_30"]).iloc[[-1]]
    issue = last.index[0]

    # Open-Meteo's own forecast, kept only for complete days
    xf = get({"latitude": lat, "longitude": lon, "hourly": "pm2_5",
              "forecast_days": 7, "timezone": "Asia/Kolkata"})
    hh = xf.set_index("time")["pm2_5"]
    dm, cnt = hh.resample("D").mean(), hh.resample("D").count()
    time.sleep(1)

    for h in [1, 3, 5, 7]:
        dh = d.copy()
        dh["target"] = dh["pm2_5"].shift(-h)
        tr = dh.dropna(subset=["target", "roll_30"] + BASE_FEATS)
        if use[(city, h)]:
            f = float(Ridge().fit(tr[BASE_FEATS], tr["target"])
                      .predict(last[BASE_FEATS])[0])
            method = "ridge"
        else:
            f = float(last[NAIVE_FOR[h]].iloc[0])
            method = NAIVE_FOR[h]
        f = max(f, 0.0)
        a = float(np.interp(f, CONC, IDX))
        t = issue + pd.Timedelta(days=h)
        om_val = float(dm.get(t, np.nan)) if cnt.get(t, 0) == 24 else np.nan
        rows.append({"city": city, "issue_date": issue.date(),
                     "target_date": t.date(), "h": h, "method": method,
                     "pm25_today": round(float(last["lag_0"].iloc[0]), 1),
                     "pm25_forecast": round(f, 1), "aqi": round(a),
                     "category": category(a),
                     "open_meteo_pm25": round(om_val, 1), "made_at": made_at})

fc = pd.DataFrame(rows)
fc.to_csv("data/processed/forecast_latest.csv", index=False)

# ---- 3) keep a permanent record, so we can score forecasts later ----
HIST = "data/processed/forecast_history.csv"
try:
    old = pd.read_csv(HIST)
    hist = pd.concat([old, fc], ignore_index=True)
except FileNotFoundError:
    hist = fc
hist = hist.drop_duplicates(["city", "issue_date", "h"], keep="last")
hist.to_csv(HIST, index=False)

print("Forecast made from data up to:", fc["issue_date"].iloc[0])
print("Rows in forecast_history.csv:", len(hist))
print(fc[fc["city"] == "Delhi"][["target_date", "h", "method",
      "pm25_forecast", "open_meteo_pm25", "aqi", "category"]].to_string(index=False))