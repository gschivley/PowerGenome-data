#!/usr/bin/env python3
"""
Build capacity reserve margins by planning year and BA region.

This script:
1. Fetches the ReEDS planning reserve margin file (prm_annual.csv)
2. Reads the hierarchy.csv to get BA-to-NERC-region mapping
3. Joins the two datasets on the nercr column
4. Saves reserve margin values (from the NERC LTRA 'nerc' column) to
   data/reserve_margins.csv with columns: ba, planning_year, reserve_margin
"""

from io import StringIO

import pandas as pd
import requests

PRM_ANNUAL_URL = "https://raw.githubusercontent.com/ReEDS-Model/ReEDS/main/inputs/reserves/prm_annual.csv"
HIERARCHY_PATH = "cache_data/hierarchy.csv"
OUTPUT_PATH = "data/reserve_margins.csv"
REQUEST_TIMEOUT_SECONDS = 30


def fetch_prm_annual(url: str) -> pd.DataFrame:
    """Fetch the ReEDS planning reserve margin file from GitHub."""
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    df = df.rename(columns={"*nercr": "nercr"})
    return df


def build_reserve_margins() -> pd.DataFrame:
    """Build the reserve margins DataFrame mapping BAs to annual PRM values."""
    prm = fetch_prm_annual(PRM_ANNUAL_URL)
    hierarchy = pd.read_csv(HIERARCHY_PATH, usecols=["ba", "nercr"])

    merged = hierarchy.merge(
        prm[["nercr", "t", "nerc"]], on="nercr", how="left"
    )
    merged = merged.rename(columns={"t": "planning_year", "nerc": "reserve_margin"})
    merged = merged[["ba", "planning_year", "reserve_margin"]].sort_values(
        ["ba", "planning_year"]
    )
    return merged


if __name__ == "__main__":
    print("Building reserve margins...")
    reserve_margins = build_reserve_margins()
    reserve_margins.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(reserve_margins)} rows to {OUTPUT_PATH}")
    print(reserve_margins.head(10).to_string(index=False))
