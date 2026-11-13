#!/usr/bin/env python3
"""Phân tích dữ liệu từ data/daily.csv và data/hourly.csv -> thống kê + biểu đồ + results.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = Path("data")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

daily = pd.read_csv(D / "daily.csv", parse_dates=["date"])
hourly = pd.read_csv(D / "hourly.csv", parse_dates=["time"])
daily["month"] = daily["date"].dt.month
daily["year"] = daily["date"].dt.year

res = {}

# 1. Chất lượng dữ liệu
res["period"] = [str(daily["date"].min().date()), str(daily["date"].max().date())]
res["n_days"] = int(len(daily))
res["n_hours"] = int(len(hourly))
res["missing_pct"] = hourly.drop(columns="time").isna().mean().mul(100).round(3).to_dict()

# 2. Thống kê mô tả
res["describe"] = daily[["t_max", "t_mean", "rh_mean", "radiation_sum_wh",
                         "precip_sum"]].describe().round(2).to_dict()

# 3. Ngày nắng nóng
res["heat_days_total"] = int(daily["heat_day"].sum())
res["severe_days_total"] = int(daily["severe_heat_day"].sum())
res["heat_days_by_year"] = daily.groupby("year")["heat_day"].sum().astype(int).to_dict()
res["heat_days_by_month"] = daily.groupby("month")["heat_day"].sum().astype(int).to_dict()
res["t_max_p90"] = round(daily["t_max"].quantile(0.90), 2)
res["t_max_p95"] = round(daily["t_max"].quantile(0.95), 2)

# 4. Liên hệ nhu cầu làm mát
res["corr_tmax_cdh"] = round(daily["t_max"].corr(daily["cooling_degree_hours"]), 3)
res["corr_tmean_cdh"] = round(daily["t_mean"].corr(daily["cooling_degree_hours"]), 3)
res["corr_rh_tmax"] = round(daily["rh_mean"].corr(daily["t_max"]), 3)
res["cdh_heat_vs_normal"] = daily.groupby("heat_day")["cooling_degree_hours"].mean().round(1).to_dict()
res["cdh_by_month"] = daily.groupby("month")["cooling_degree_hours"].mean().round(1).to_dict()

# 5. Giờ nóng nhất trong ngày
hourly["hour"] = hourly["time"].dt.hour
res["hottest_hour"] = int(hourly.groupby("hour")["temperature_2m"].mean().idxmax())

Path("results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

# ---- Biểu đồ ----
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3})

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(daily["date"], daily["t_max"], lw=.7, color="#d62728")
ax.axhline(35, ls="--", color="k", lw=1, label="Ngưỡng nắng nóng 35°C")
ax.axhline(37, ls=":", color="k", lw=1, label="Gay gắt 37°C")
ax.set(title="Nhiệt độ cao nhất ngày – Biên Hòa", ylabel="°C"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "01_tmax_timeseries.png"); plt.close(fig)

pv = daily.pivot_table(index="year", columns="month", values="heat_day", aggfunc="sum")
fig, ax = plt.subplots(figsize=(10, 3.5))
im = ax.imshow(pv.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(pv.columns)), pv.columns); ax.set_yticks(range(len(pv.index)), pv.index)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        v = pv.values[i, j]
        if pd.notna(v): ax.text(j, i, int(v), ha="center", va="center", fontsize=8)
ax.set(title="Số ngày nắng nóng (Tmax ≥ 35°C) theo tháng/năm", xlabel="Tháng"); ax.grid(False)
fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT / "02_heat_days_heatmap.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(daily["t_max"], daily["cooling_degree_hours"], s=6, alpha=.4)
ax.set(xlabel="Tmax (°C)", ylabel="Cooling degree-hours (cơ sở 26°C)",
       title=f"Tmax vs nhu cầu làm mát (r={res['corr_tmax_cdh']})")
fig.tight_layout(); fig.savefig(OUT / "03_tmax_vs_cooling.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
daily.boxplot(column="t_max", by="month", ax=ax, grid=False)
ax.set(title="Phân bố Tmax theo tháng", xlabel="Tháng", ylabel="°C"); fig.suptitle("")
fig.tight_layout(); fig.savefig(OUT / "04_tmax_by_month.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
hourly.groupby("hour")["temperature_2m"].mean().plot(ax=ax, marker="o")
ax.set(title="Nhiệt độ trung bình theo giờ trong ngày", xlabel="Giờ", ylabel="°C")
fig.tight_layout(); fig.savefig(OUT / "05_diurnal.png"); plt.close(fig)

print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
