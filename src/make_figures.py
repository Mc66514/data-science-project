"""
make_figures.py -- draw the two figures used in the portfolio.

Input :  data/processed/pageviews_smoothed.csv, outputs/results.json
Output:  figures/fig1_attention_timeline.png
         figures/fig2_decay_and_half_life.png

Design choices (deliberate, not default styling)
------------------------------------------------
* Both figures use a log y-axis. The three articles differ by two orders of
  magnitude, so a linear axis would flatten the two smaller series into the
  baseline and hide them. On a log axis equal vertical distance = equal
  percentage change, which is the quantity the research question is about,
  and a constant decay rate becomes a straight line.
* Figure 1 shows the full timeline with the 7-day mean, so the launch spike,
  the trough and the June-2025 burst are all visible. Figure 2 starts after
  the launch ramp-up (June 2023) and normalises each series to its own
  July-2023 level, so the *rates* are comparable on one scale.
* No dual y-axis anywhere: every series shares one axis and one unit.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"

COLORS = {"ChatGPT": "#0072B2",
          "Artificial_intelligence": "#D55E00",
          "Large_language_model": "#009E73"}
NAMES = {"ChatGPT": "ChatGPT  (product)",
         "Artificial_intelligence": "Artificial intelligence  (topic)",
         "Large_language_model": "Large language model  (technique)"}

EVENTS = [("2022-11-30", "ChatGPT\nlaunched"),
          ("2023-03-14", "GPT-4"),
          ("2024-05-13", "GPT-4o"),
          ("2025-01-20", "DeepSeek-R1"),
          ("2025-08-07", "GPT-5")]

INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#dcdcdc"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#9a9a9a",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load():
    sm = pd.read_csv(PROCESSED / "pageviews_smoothed.csv", index_col=0, parse_dates=True)
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    return sm, res


def fmt_k(v: float) -> str:
    return f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
def fig1(sm: pd.DataFrame, res: dict) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    win0 = pd.Timestamp(res["common_fit_window"][0])
    win1 = pd.Timestamp(res["common_fit_window"][1])

    ax.axvspan(win0, win1, color="#f2f2f2", zorder=0, lw=0)
    ax.text(win0 + (win1 - win0) / 2, 520000, "common fit window\n(Figure 2)",
            ha="center", va="top", fontsize=8.5, color=MUTED, style="italic",
            linespacing=1.25)

    for col in ["Large_language_model", "Artificial_intelligence", "ChatGPT"]:
        s = sm[col].dropna()
        ax.plot(s.index, s.values, color=COLORS[col], lw=1.3,
                label=NAMES[col], zorder=3)

    for date, label in EVENTS:
        d = pd.Timestamp(date)
        ax.axvline(d, color="#cfcfcf", lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.annotate(label, xy=(d, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=MUTED,
                    linespacing=1.15)

    c = res["articles"]["ChatGPT"]
    peak_d, peak_v = pd.Timestamp(c["peak_day"]), c["peak_views_per_day"]
    tro_d, tro_v = pd.Timestamp(c["trough_day"]), c["trough_views_per_day"]
    re_d, re_v = pd.Timestamp(c["peak_month_2025"][0]), c["peak_month_2025"][1]

    ax.plot([peak_d], [peak_v], "o", ms=5.5, color=COLORS["ChatGPT"], zorder=5)
    ax.annotate(f"peak {fmt_k(peak_v)}/day\nApr 2023",
                xy=(peak_d, peak_v), xytext=(-8, 16), textcoords="offset points",
                ha="right", fontsize=8.5, color=COLORS["ChatGPT"],
                fontweight="bold", linespacing=1.2)

    ax.plot([tro_d], [tro_v], "o", ms=5.5, color=COLORS["ChatGPT"], zorder=5)
    ax.annotate(f"trough {fmt_k(tro_v)}/day, Aug 2024",
                xy=(tro_d, tro_v), xytext=(10, -30), textcoords="offset points",
                ha="left", va="top", fontsize=8.5, color=COLORS["ChatGPT"])

    ax.plot([re_d], [re_v], "o", ms=5.5, color=COLORS["ChatGPT"], zorder=5)
    ax.annotate(f"second wave {fmt_k(re_v)}/day\n(Jun 2025, {c['rebound_2025_vs_2024']:.1f}x the 2024 trough)",
                xy=(re_d, re_v), xytext=(52, 62), textcoords="offset points",
                ha="left", va="center", fontsize=8.5, color=COLORS["ChatGPT"],
                linespacing=1.2,
                arrowprops=dict(arrowstyle="-", color=COLORS["ChatGPT"], lw=0.7,
                                shrinkA=2, shrinkB=4))

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1000, 3000, 10000, 30000, 100000, 300000]))
    ax.yaxis.set_major_formatter(FixedFormatter(["1k", "3k", "10k", "30k", "100k", "300k"]))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(1600, 700000)
    ax.set_xlim(pd.Timestamp("2022-11-01"), pd.Timestamp("2026-10-25"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    clean(ax)
    ax.set_ylabel("Wikipedia pageviews per day  (log scale)")

    ax.set_title("Public attention to generative AI on English Wikipedia, Nov 2022 – Sep 2026",
                 fontsize=13.5, fontweight="bold", loc="left", pad=26)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.006, 0.085),
               ncol=3, frameon=False, fontsize=9, handlelength=2.4,
               columnspacing=2.2, handletextpad=0.7)

    fig.text(0.008, 0.014,
             "Daily pageviews per article, bots excluded, 7-day centred mean; equal vertical distance = equal percentage change.\n"
             "Source: Wikimedia REST API, en.wikipedia, all-access, agent=user (CC0 1.0). Release dates from OpenAI and DeepSeek announcements.\n"
             "The large-language-model article was created on 13 Feb 2023, so its first weeks fall below the plotted range. Figure by MC66514, CISC7204.",
             fontsize=7.2, color=MUTED, linespacing=1.6)
    fig.tight_layout(rect=(0, 0.115, 1, 0.93))
    out = FIG / "fig1_attention_timeline.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
def fig2(sm: pd.DataFrame, res: dict) -> Path:
    win0 = pd.Timestamp(res["common_fit_window"][0])
    win1 = pd.Timestamp(res["common_fit_window"][1])
    start = pd.Timestamp("2023-06-01")

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.6, 5.9), gridspec_kw={"width_ratios": [2.4, 1.0]})

    base = {}
    for col in sm.columns:
        s = sm[col].dropna()
        base[col] = s[win0:win1].iloc[:28].mean()
        idx = (s / base[col] * 100)[start:]
        axL.plot(idx.index, idx.values, color=COLORS[col], lw=1.2, zorder=3)

        a = res["articles"][col]
        seg = (s / base[col] * 100)[win0:win1].dropna()
        t = (seg.index - win0).days.to_numpy(dtype=float)
        b = a["decay_slope_per_day_common_window"]
        intercept = float(np.log(seg.to_numpy()).mean() - b * t.mean())
        axL.plot(seg.index, np.exp(intercept + b * t), color=COLORS[col], lw=2.4,
                 ls=(0, (5, 2)), alpha=0.9, zorder=2)

    axL.axvspan(win0, win1, color="#f2f2f2", zorder=0, lw=0)
    axL.axhline(100, color="#8a8a8a", lw=0.9, zorder=1)
    axL.axhline(50, color="#b5b5b5", lw=0.9, ls=(0, (2, 3)), zorder=1)
    axL.annotate("half", xy=(1.0, 50), xycoords=("axes fraction", "data"),
                 xytext=(3, 0), textcoords="offset points", fontsize=7.6,
                 color=MUTED, va="center")
    axL.text(win0 + pd.Timedelta(days=10), 330, "common fit window",
             fontsize=8.5, color=MUTED, style="italic")

    axL.set_yscale("log")
    axL.set_ylim(10, 560)
    axL.yaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 500]))
    axL.yaxis.set_major_formatter(FixedFormatter(["10", "20", "50", "100", "200", "500"]))
    axL.yaxis.set_minor_locator(FixedLocator([]))
    axL.set_xlim(start, pd.Timestamp("2026-10-25"))
    axL.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    axL.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    clean(axL)
    axL.set_ylabel("Index, July 2023 = 100  (log scale)")
    axL.set_title("A.  Every series indexed to its own July-2023 level",
                  fontsize=10.5, loc="left", pad=8)

    handles = [Line2D([], [], color=COLORS[c], lw=1.6, label=NAMES[c].split("  ")[0])
               for c in ["ChatGPT", "Artificial_intelligence", "Large_language_model"]]
    handles.append(Line2D([], [], color="#7a7a7a", lw=2.2, ls=(0, (5, 2)),
                          label="log-linear fit, Jul 2023 – Aug 2024"))
    axL.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.3,
               handlelength=2.6, labelspacing=0.4)

    # ---- panel B: half-life point estimates with bootstrap CI ---------------
    order = ["ChatGPT", "Artificial_intelligence", "Large_language_model"]
    ys = np.arange(len(order))[::-1]
    XMAX, XTXT = 6200, 8200
    for y, col in zip(ys, order):
        a = res["articles"][col]
        lo, hi = a["half_life_ci95_bootstrap"]
        hl = a["half_life_days"]
        flat = a["decay_r2_common_window"] < 0.1
        axR.plot([lo, min(hi, XMAX)], [y, y], color=COLORS[col], lw=2.4,
                 solid_capstyle="round", alpha=0.85, zorder=3)
        axR.plot([hl], [y], "o", ms=8, color=COLORS[col], zorder=4,
                 markeredgecolor="white", markeredgewidth=1.0)
        if hi > XMAX:
            axR.annotate("", xy=(XMAX, y), xytext=(XMAX / 1.09, y),
                         arrowprops=dict(arrowstyle="-|>", color=COLORS[col], lw=1.6))
            axR.text(XTXT, y, "CI to\n75,500 d", fontsize=7.2, color=COLORS[col],
                     va="center", ha="left", linespacing=1.15)
        label = ("no measurable\ndecay (R² = 0.01)" if flat else f"{hl:.0f} days")
        axR.text(max(lo, 135), y + 0.24, label, fontsize=8.5, color=COLORS[col],
                 fontweight="bold", ha="left" if flat else "center",
                 va="bottom", linespacing=1.15)

    axR.set_yticks(ys)
    axR.set_yticklabels(["ChatGPT\n(product)", "Artificial\nintelligence",
                         "Large language\nmodel"], fontsize=8.6, linespacing=1.2)
    axR.set_ylim(-0.62, 2.62)
    axR.set_xscale("log")
    axR.set_xlim(120, 22000)
    axR.set_xticks([150, 300, 600, 1500, 3000, 6000])
    axR.set_xticklabels(["150", "300", "600", "1.5k", "3k", "6k"])
    axR.xaxis.set_minor_locator(FixedLocator([]))
    axR.set_xlabel("Half-life of the decline, days  (log scale)")
    axR.grid(axis="x", color=GRID, lw=0.7)
    axR.set_axisbelow(True)
    axR.spines[["top", "right"]].set_visible(False)
    axR.tick_params(axis="y", length=0)
    axR.set_title("B.  Days for attention to halve\n     (estimate, 95% bootstrap CI)",
                  fontsize=10.5, loc="left", pad=8)

    fig.suptitle("Attention decays at different speeds: product, topic, technique",
                 x=0.008, y=0.985, ha="left", va="top", fontsize=13.5, fontweight="bold")
    fig.text(0.008, 0.015,
             "Log-linear fit ln(views) = a + b·t over 1 Jul 2023 – 31 Aug 2024, on 7-day means with bots excluded. "
             "CI from a 14-day block bootstrap (2,000 resamples).\n"
             "Source: Wikimedia REST API, en.wikipedia, all-access, agent=user (CC0 1.0).  Figure by MC66514, CISC7204.",
             fontsize=7.6, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.925))
    out = FIG / "fig2_decay_and_half_life.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    sm, res = load()
    for p in (fig1(sm, res), fig2(sm, res)):
        print("wrote", p.relative_to(ROOT), f"({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
