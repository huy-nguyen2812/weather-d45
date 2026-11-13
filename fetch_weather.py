#!/usr/bin/env python3
"""
Đ45 - Trạm thời tiết ảo cho nhà máy tại Đồng Nai (Biên Hòa) qua Open-Meteo API.

Chức năng:
  1. Lấy >= 3 năm dữ liệu giờ (hourly) từ Open-Meteo Historical Weather API.
  2. Lưu ra CSV (hourly + daily) và JSON thô.
  3. Chế độ cập nhật hằng ngày: chỉ lấy phần còn thiếu, gắn cờ ngày nắng nóng,
     lấy dự báo 7 ngày tới để cảnh báo.
  4. (Tuỳ chọn) đẩy kết quả lên Google Sheets.

Cách dùng:
  pip install requests pandas gspread google-auth
  python fetch_weather.py --years 3                 # lần đầu: lấy toàn bộ
  python fetch_weather.py --update                  # chạy hằng ngày
  python fetch_weather.py --update --sheet-id <ID> --creds service_account.json
"""
import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

# ---- Cấu hình -----------------------------------------------------------
LAT, LON = 10.9574, 106.8426          # Biên Hòa, Đồng Nai
TZ = "Asia/Bangkok"                    # UTC+7
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "shortwave_radiation",
    "precipitation",
    "wind_speed_10m",
]
ARCHIVE_LAG_DAYS = 5                   # ERA5 trễ ~5 ngày
HEAT_THRESHOLD = 35.0                  # °C, ngưỡng "nắng nóng" (Tmax >= 35)
SEVERE_THRESHOLD = 37.0                # °C, "nắng nóng gay gắt" (Tmax >= 37)
APPARENT_THRESHOLD = 40.0              # °C, "oi bức" theo nhiệt độ cảm nhận (đề xuất)

DATA_DIR = Path("data")
HOURLY_CSV = DATA_DIR / "hourly.csv"
DAILY_CSV = DATA_DIR / "daily.csv"
RAW_JSON = DATA_DIR / "raw_last_response.json"
ALERT_CSV = DATA_DIR / "forecast_alerts.csv"


def get_json(url, params, retries=4):
    """GET có thử lại khi lỗi mạng / 429."""
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=(15, 60))
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))


def fetch_hourly(start: date, end: date) -> pd.DataFrame:
    """Lấy dữ liệu giờ, chia theo năm để tránh phản hồi quá lớn."""
    frames, cur = [], start
    while cur <= end:
        chunk_end = min(date(cur.year, 12, 31), end)
        data = get_json(ARCHIVE_URL, {
            "latitude": LAT, "longitude": LON,
            "start_date": cur.isoformat(), "end_date": chunk_end.isoformat(),
            "hourly": ",".join(HOURLY_VARS), "timezone": TZ,
        })
        RAW_JSON.parent.mkdir(exist_ok=True)
        RAW_JSON.write_text(json.dumps(data)[:2_000_000], encoding="utf-8")
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df)
        print(f"  đã lấy {cur} -> {chunk_end}: {len(df)} dòng")
        cur = chunk_end + timedelta(days=1)
        time.sleep(1)
    return pd.concat(frames, ignore_index=True)


def clean_hourly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    # Nội suy khoảng thiếu ngắn (<=3 giờ); khoảng dài để NaN
    num = [c for c in df.columns if c != "time"]
    df[num] = df[num].interpolate(limit=3)
    return df


def to_daily(h: pd.DataFrame) -> pd.DataFrame:
    g = h.set_index("time").resample("D")
    d = pd.DataFrame({
        "t_max": g["temperature_2m"].max(),
        "t_min": g["temperature_2m"].min(),
        "t_mean": g["temperature_2m"].mean(),
        "apparent_max": g["apparent_temperature"].max(),
        "rh_mean": g["relative_humidity_2m"].mean(),
        "radiation_sum_wh": g["shortwave_radiation"].sum(),  # Wh/m2/ngày
        "precip_sum": g["precipitation"].sum(),
        "wind_mean": g["wind_speed_10m"].mean(),
        # Cooling degree-hours cơ sở 26°C: đại diện nhu cầu điện làm mát
        "cooling_degree_hours": g["temperature_2m"].apply(lambda s: (s - 26).clip(lower=0).sum()),
        "hours_valid": g["temperature_2m"].count(),
    }).reset_index().rename(columns={"time": "date"})
    d["heat_day"] = d["t_max"] >= HEAT_THRESHOLD
    d["severe_heat_day"] = d["t_max"] >= SEVERE_THRESHOLD
    num_cols = d.select_dtypes("number").columns
    d[num_cols] = d[num_cols].round(2)
    return d


def forecast_alerts(days=7) -> pd.DataFrame:
    """Dự báo vài ngày tới -> cảnh báo nắng nóng."""
    data = get_json(FORECAST_URL, {
        "latitude": LAT, "longitude": LON, "timezone": TZ, "forecast_days": days,
        "daily": "temperature_2m_max,apparent_temperature_max,precipitation_sum",
    })
    f = pd.DataFrame(data["daily"]).rename(columns={
        "time": "date", "temperature_2m_max": "t_max_forecast"})
    def level(t):
        if t >= SEVERE_THRESHOLD: return "GAY GẮT (>=37°C)"
        if t >= HEAT_THRESHOLD:   return "NẮNG NÓNG (>=35°C)"
        return "Bình thường"
    f["alert"] = f["t_max_forecast"].apply(level)            # tầng 1: nhiệt độ không khí
    f["alert_humid"] = f["apparent_temperature_max"].apply(  # tầng 2: nhiệt độ cảm nhận
        lambda t: "OI BỨC (cảm nhận >=40°C)" if t >= APPARENT_THRESHOLD else "Bình thường")
    return f


def push_to_sheets(sheet_id, creds, tables: dict):
    import gspread
    gc = gspread.service_account(filename=creds)
    sh = gc.open_by_key(sheet_id)
    for name, df in tables.items():
        try:
            ws = sh.worksheet(name)
            ws.clear()
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(name, rows=len(df) + 10, cols=len(df.columns) + 2)
        out = df.copy()
        for c in out.columns:
            if str(out[c].dtype).startswith("datetime"):
                out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M")
        out = out.astype(object).where(out.notna(), "")
        ws.update([out.columns.tolist()] + out.astype(str).values.tolist())
        print(f"  đã đẩy sheet '{name}': {len(out)} dòng")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--update", action="store_true", help="chỉ lấy phần còn thiếu")
    ap.add_argument("--sheet-id")
    ap.add_argument("--creds", default="service_account.json")
    a = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    end = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)

    if a.update and HOURLY_CSV.exists():
        old = pd.read_csv(HOURLY_CSV, parse_dates=["time"])
        start = old["time"].max().date() + timedelta(days=1)
        if start > end:
            print("Dữ liệu đã mới nhất.")
            hourly = old
        else:
            new = fetch_hourly(start, end)
            hourly = clean_hourly(pd.concat([old, new], ignore_index=True))
    else:
        start = end.replace(year=end.year - a.years) + timedelta(days=1)
        print(f"Lấy dữ liệu {start} -> {end}")
        hourly = clean_hourly(fetch_hourly(start, end))

    daily = to_daily(hourly)
    hourly.to_csv(HOURLY_CSV, index=False)
    daily.to_csv(DAILY_CSV, index=False)

    try:
        alerts = forecast_alerts()
        alerts.to_csv(ALERT_CSV, index=False)
        print("\nCảnh báo 7 ngày tới:\n", alerts.to_string(index=False))
    except requests.RequestException as e:
        # Lỗi mạng ở phần dự báo không được làm hỏng cả quy trình
        print(f"\n[CẢNH BÁO] Không lấy được dự báo ({type(e).__name__}). Dùng bản cũ nếu có.")
        alerts = pd.read_csv(ALERT_CSV) if ALERT_CSV.exists() else pd.DataFrame()
    print(f"\nHourly: {len(hourly)} dòng | Daily: {len(daily)} ngày | "
          f"Số ngày nắng nóng: {int(daily['heat_day'].sum())}")

    if a.sheet_id:
        tables = {"daily": daily, "hourly_last_30d": hourly.tail(24 * 30)}
        if len(alerts):
            tables["forecast_alerts"] = alerts
        push_to_sheets(a.sheet_id, a.creds, tables)


if __name__ == "__main__":
    main()
