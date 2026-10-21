"""
fetch_data.py -- download daily Wikipedia pageviews for three AI-related articles.

Data source
-----------
Wikimedia REST API, "Pageviews per article" endpoint:
https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}

Wikimedia pageview data is released under CC0 1.0 (public domain dedication):
https://www.mediawiki.org/wiki/REST_API#Terms_and_conditions

The script writes:
  data/raw/pageviews_<article>.json       raw API payload (cached, one file per article)
  data/processed/pageviews_daily.csv      tidy long table used by analyze.py

Usage
-----
    python src/fetch_data.py               # uses cache when present
    python src/fetch_data.py --refresh     # force re-download
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

PROJECT = "en.wikipedia"
ACCESS = "all-access"
AGENT = "user"          # excludes bots/spiders -> closer to "public attention"
START = "20221101"      # one month before the ChatGPT public launch
END = "20260930"

ARTICLES = {
    "ChatGPT": "ChatGPT",                                  # product-level article
    "Artificial_intelligence": "Artificial intelligence",  # topic-level article
    "Large_language_model": "Large language model",         # technique-level article
}

API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
       "{project}/{access}/{agent}/{article}/daily/{start}/{end}")

# Wikimedia's API policy asks clients to identify themselves with a descriptive
# User-Agent. No credentials are used anywhere in this project.
HEADERS = {
    "User-Agent": "cisc7204-portfolio/1.0 (student coursework, CISC7204)",
    "Accept": "application/json",
}


def api_url(article: str, start: str, end: str) -> str:
    return API.format(
        project=PROJECT,
        access=ACCESS,
        agent=AGENT,
        article=urllib.parse.quote(article, safe=""),
        start=start,
        end=end,
    )


def download(article: str, start: str, end: str) -> dict:
    url = api_url(article, start, end)
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse(payload: dict, article_key: str, label: str) -> pd.DataFrame:
    rows = payload.get("items", [])
    if not rows:
        raise RuntimeError(f"no items returned for {article_key}")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d")
    df["views"] = df["views"].astype(int)
    df["article_key"] = article_key
    df["article_label"] = label
    return df[["date", "article_key", "article_label", "project", "views"]]


def main(refresh: bool = False) -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    frames = []
    for key, label in ARTICLES.items():
        cache = RAW / f"pageviews_{key}.json"
        if cache.exists() and not refresh:
            payload = json.loads(cache.read_text(encoding="utf-8"))
            status = "cache"
        else:
            payload = download(key, START, END)
            cache.write_text(json.dumps(payload), encoding="utf-8")
            status = "downloaded"
            time.sleep(1)  # be polite to the API
        part = parse(payload, key, label)
        print(f"{status:>10}: {key:<26} {len(part):>5} days  "
              f"{part['date'].min().date()} -> {part['date'].max().date()}")
        frames.append(part)

    tidy = pd.concat(frames).sort_values(["article_key", "date"]).reset_index(drop=True)
    out = PROCESSED / "pageviews_daily.csv"
    tidy.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(tidy)} rows, "
          f"{tidy['article_key'].nunique()} articles)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true",
                    help="re-download instead of reusing data/raw/*.json")
    sys.exit(main(ap.parse_args().refresh))
