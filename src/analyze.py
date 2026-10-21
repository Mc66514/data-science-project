"""
analyze.py -- quantify how fast Wikipedia attention decayed after the AI launches.

Input :  data/processed/pageviews_daily.csv     (written by src/fetch_data.py)
Output:  outputs/results.json                   machine-readable numbers
         outputs/summary_table.md               the same numbers as Markdown tables
         data/processed/pageviews_smoothed.csv  7-day centred rolling mean

Method
------
1. Smooth with a 7-day centred rolling mean (removes the day-of-week rhythm).
2. Descriptive: locate each article's wave-1 peak in 2023 and its wave-1 trough
   in the following 18 months, and report the peak-to-trough change.
3. Headline metric: fit ln(views) = a + b * t by OLS over one COMMON window,
   2023-07-01 .. 2024-08-31. The common window keeps the three half-lives
   comparable (each article's own peak falls at a different date, and the 2025
   re-acceleration would otherwise enter some fits and not others).
   If b < 0 the decline is exponential and the half-life is ln(2) / (-b) days.
4. Uncertainty: block bootstrap (14-day blocks, 2,000 resamples) so that the
   autocorrelation of daily series is respected.
5. Attention churn: standard deviation of month-on-month log changes in the
   monthly mean, measured from July 2023 so that each article's own launch
   ramp-up does not count as churn.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

SMOOTH = 7
WAVE1_START = "2023-01-01"
WAVE1_PEAK_END = "2023-12-31"
WAVE1_TROUGH_END = "2024-12-31"
COMMON_START = "2023-07-01"
COMMON_END = "2024-08-31"
N_BOOT = 2000
BLOCK = 14
SEED = 20261021


def load_wide() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "pageviews_daily.csv", parse_dates=["date"])
    return (df.pivot_table(index="date", columns="article_key", values="views")
              .asfreq("D")
              .astype(float))


def ols_log_decay(dates: pd.DatetimeIndex, y: np.ndarray, t0: pd.Timestamp):
    t = (dates - t0).days.to_numpy(dtype=float)
    slope, intercept = np.polyfit(t, np.log(y), 1)
    resid = np.log(y) - (intercept + slope * t)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((np.log(y) - np.log(y).mean()) ** 2))
    return float(slope), float(intercept), (1 - ss_res / ss_tot if ss_tot else float("nan"))


def block_bootstrap_half_life(dates: pd.DatetimeIndex, y: np.ndarray,
                              t0: pd.Timestamp) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    t = (dates - t0).days.to_numpy(dtype=float)
    ly = np.log(y)
    n = len(t)
    n_blocks = math.ceil(n / BLOCK)
    out = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, n - BLOCK, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:n]
        b = float(np.polyfit(t[idx], ly[idx], 1)[0])
        out.append(math.log(2) / -b if b < 0 else np.nan)
    return np.asarray(out, dtype=float)


def churn(monthly: pd.Series) -> float:
    d = np.log(monthly.dropna()[COMMON_START:]).diff().dropna()
    return float(d.std()) if len(d) > 12 else float("nan")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    wide = load_wide()
    sm = wide.rolling(SMOOTH, center=True, min_periods=3).mean()

    results: dict = {
        "source": {
            "api": "Wikimedia REST API - pageviews per article",
            "project": "en.wikipedia",
            "access": "all-access",
            "agent": "user (bots excluded)",
            "license": "CC0 1.0",
            "window": f"{wide.index.min().date()} .. {wide.index.max().date()}",
        },
        "common_fit_window": [COMMON_START, COMMON_END],
        "coverage": {},
        "articles": {},
    }

    for col in wide.columns:
        s = wide[col].dropna()
        m = wide[col].resample("MS").mean()
        results["coverage"][col] = {
            "first_day": str(s.index.min().date()),
            "last_day": str(s.index.max().date()),
            "n_days": int(len(s)),
            "mean_daily_views": int(round(s.mean())),
            "monthly_churn_sd_log": round(churn(m), 3),
        }

    for col in wide.columns:
        s = sm[col].dropna()
        peak_win = s[WAVE1_START:WAVE1_PEAK_END]
        peak_day, peak_val = peak_win.idxmax(), float(peak_win.max())
        trough_win = s[peak_day:WAVE1_TROUGH_END]
        trough_day, trough_val = trough_win.idxmin(), float(trough_win.min())

        seg = s[COMMON_START:COMMON_END].dropna()
        slope, _, r2 = ols_log_decay(seg.index, seg.to_numpy(), seg.index[0])
        hl = math.log(2) / -slope if slope < 0 else float("inf")
        boots = block_bootstrap_half_life(seg.index, seg.to_numpy(), seg.index[0])
        fin = boots[np.isfinite(boots)]
        ci = ([round(float(np.percentile(fin, 2.5)), 1),
               round(float(np.percentile(fin, 97.5)), 1)] if len(fin) > 100 else None)

        monthly = wide[col].resample("MS").mean()
        trough_2024 = monthly["2024"].min()
        peak_2025 = monthly["2025"].max()

        results["articles"][col] = {
            "peak_day": str(peak_day.date()),
            "peak_views_per_day": int(round(peak_val)),
            "trough_day": str(trough_day.date()),
            "trough_views_per_day": int(round(trough_val)),
            "pct_change_peak_to_trough": round((trough_val / peak_val - 1) * 100, 1),
            "decay_slope_per_day_common_window": round(slope, 6),
            "decay_r2_common_window": round(r2, 3),
            "half_life_days": round(hl, 1) if np.isfinite(hl) else None,
            "half_life_ci95_bootstrap": ci,
            "trough_month_2024": [str(monthly["2024"].idxmin().date()), int(round(trough_2024))],
            "peak_month_2025": [str(monthly["2025"].idxmax().date()), int(round(peak_2025))],
            "rebound_2025_vs_2024": round(float(peak_2025 / trough_2024), 2),
        }

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    sm.to_csv(PROCESSED / "pageviews_smoothed.csv")

    a = results["articles"]
    t1 = ["| Article | Wave-1 peak | Wave-1 trough | Change |",
          "|---|---|---|---|"]
    t2 = ["| Article | Half-life over common window | 95% CI | R² | 2025 peak month vs 2024 trough month | Attention churn |",
          "|---|---|---|---|---|---|"]
    for k, v in a.items():
        t1.append(f"| {k} | {v['peak_views_per_day']:,}/day ({v['peak_day']}) | "
                  f"{v['trough_views_per_day']:,}/day ({v['trough_day']}) | "
                  f"{v['pct_change_peak_to_trough']:+.1f}% |")
        ci = v["half_life_ci95_bootstrap"]
        t2.append(f"| {k} | {v['half_life_days']:.0f} days | "
                  f"[{ci[0]:.0f}, {ci[1]:.0f}] | {v['decay_r2_common_window']:.2f} | "
                  f"x{v['rebound_2025_vs_2024']:.2f} | "
                  f"{results['coverage'][k]['monthly_churn_sd_log']:.2f} |")
    md = "\n".join(t1) + "\n\n" + "\n".join(t2) + "\n"
    (OUT / "summary_table.md").write_text(md, encoding="utf-8")
    print(md)
    for k, v in results["coverage"].items():
        print(f"{k:<26} {v['first_day']} -> {v['last_day']}  n={v['n_days']:>4}  "
              f"mean={v['mean_daily_views']:>6,}/day")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
