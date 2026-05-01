#!/usr/bin/env python3
"""
Build capacity reserve margins by planning year and BA region.

This script:
1. Fetches the ReEDS planning reserve margin file (prm_annual.csv)
2. Reads the hierarchy.csv to get BA-to-NERC-region mapping
3. Joins the two datasets on the nercr column
4. Saves reserve margin values (from the NERC LTRA 'nerc' column) to
   data/reserve_margins.csv with columns: region, planning_year, reserve_margin
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


def validate_reserve_margins(df: pd.DataFrame) -> None:
    """Validate that the reserve margins DataFrame is complete and well-formed.

    Raises:
        ValueError: If any reserve_margin values are missing or if (region,
            planning_year) pairs are not unique.
    """
    missing = df["reserve_margin"].isna().sum()
    if missing > 0:
        missing_regions = df.loc[df["reserve_margin"].isna(), "region"].unique().tolist()
        raise ValueError(
            f"{missing} missing reserve_margin value(s) found for regions: {missing_regions}. "
            "Check that all nercr values in hierarchy.csv are present in prm_annual.csv."
        )

    duplicates = df.duplicated(subset=["region", "planning_year"])
    if duplicates.any():
        dup_rows = df[duplicates][["region", "planning_year"]].head(10)
        raise ValueError(
            f"{duplicates.sum()} duplicate (region, planning_year) pair(s) found:\n{dup_rows}"
        )


def build_reserve_margins() -> pd.DataFrame:
    """Build the reserve margins DataFrame mapping regions to annual PRM values."""
    prm = fetch_prm_annual(PRM_ANNUAL_URL)
    hierarchy = pd.read_csv(HIERARCHY_PATH, usecols=["ba", "nercr"])

    merged = hierarchy.merge(
        prm[["nercr", "t", "nerc"]], on="nercr", how="left"
    )
    merged = merged.rename(columns={"ba": "region", "t": "planning_year", "nerc": "reserve_margin"})
    merged = merged[["region", "planning_year", "reserve_margin"]].sort_values(
        ["region", "planning_year"]
    )
    validate_reserve_margins(merged)
    return merged


if __name__ == "__main__":
    print("Building reserve margins...")
    reserve_margins = build_reserve_margins()
    reserve_margins.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(reserve_margins)} rows to {OUTPUT_PATH}")
    print(reserve_margins.head(10).to_string(index=False))
