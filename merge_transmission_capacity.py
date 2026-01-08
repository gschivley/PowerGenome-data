#!/usr/bin/env python3
"""Merge AC and non-AC transmission capacity files into a single CSV. Add the average of
MW_f0 and MW_r0 for AC lines to MW for non-AC lines when they share the same region pair.

Downloads transmission files from ReEDS GitHub repo and merges them.

Output columns: region_from,region_to,firm_ttc_mw,notes

Usage:
  python merge_transmission_capacity.py
"""
from pathlib import Path

import pandas as pd
import requests

# GitHub URLs for transmission files
AC_URL = "https://raw.githubusercontent.com/NREL/ReEDS-2.0/main/inputs/transmission/transmission_capacity_init_AC_ba_NARIS2024.csv"
NONAC_URL = "https://raw.githubusercontent.com/NREL/ReEDS-2.0/main/inputs/transmission/transmission_capacity_init_nonAC_ba.csv"


def download_file(url, cache_path):
    """Download file from URL and cache it locally."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Using cached file: {cache_path}")
        with open(cache_path, "r") as f:
            return f.read()

    print(f"Downloading from {url}...")
    response = requests.get(url)
    response.raise_for_status()

    # Cache the file
    with open(cache_path, "w") as f:
        f.write(response.text)
    print(f"  Cached to {cache_path}")

    return response.text


def parse_ac(csv_text):
    """Parse AC transmission CSV and return DataFrame."""
    df = pd.read_csv(pd.io.common.StringIO(csv_text))

    # Convert MW columns to float, handling missing/invalid values
    df["MW_f0"] = pd.to_numeric(df.get("MW_f0", 0), errors="coerce").fillna(0)
    df["MW_r0"] = pd.to_numeric(df.get("MW_r0", 0), errors="coerce").fillna(0)

    # Calculate average
    df["firm_ttc_mw"] = (df["MW_f0"] + df["MW_r0"]) / 2.0

    # Rename and select columns
    df = df.rename(columns={"r": "region_from", "rr": "region_to"})
    df["notes"] = ""

    return df[["region_from", "region_to", "firm_ttc_mw", "notes"]]


def parse_nonac(csv_text):
    """Parse non-AC transmission CSV and return DataFrame."""
    df = pd.read_csv(pd.io.common.StringIO(csv_text))

    # Convert MW to float
    df["firm_ttc_mw"] = pd.to_numeric(df.get("MW", 0), errors="coerce").fillna(0)

    # Get notes, fallback to Project(s) if empty
    df["notes"] = df.get("Notes", "").fillna("")
    mask = df["notes"] == ""
    df.loc[mask, "notes"] = df.get("Project(s)", "").fillna("")

    # Rename and select columns
    df = df.rename(columns={"r": "region_from", "rr": "region_to"})

    return df[["region_from", "region_to", "firm_ttc_mw", "notes"]]


def main():
    """Main execution function."""
    print("=" * 60)
    print("Merging ReEDS Transmission Capacity Data")
    print("=" * 60)

    # Download files
    ac_text = download_file(
        AC_URL, "cache_data/transmission_capacity_init_AC_ba_NARIS2024.csv"
    )
    nonac_text = download_file(
        NONAC_URL, "cache_data/transmission_capacity_init_nonAC_ba.csv"
    )

    # Parse data
    print("\nParsing AC transmission data...")
    ac_df = parse_ac(ac_text)
    print(f"  Loaded {len(ac_df):,} AC transmission rows")

    print("Parsing non-AC transmission data...")
    nonac_df = parse_nonac(nonac_text)
    print(f"  Loaded {len(nonac_df):,} non-AC transmission rows")

    # Combine dataframes
    print("\nAggregating capacities by region pair...")
    combined_df = pd.concat([ac_df, nonac_df], ignore_index=True)

    # Aggregate by region pair
    agg_df = combined_df.groupby(["region_from", "region_to"], as_index=False).agg(
        {
            "firm_ttc_mw": "sum",
            "notes": lambda x: "; ".join(filter(None, x.unique())),
        }
    )

    print(f"  Aggregated to {len(agg_df):,} unique region pairs")

    # Write output
    out_file = "data/transmission_capacity_reeds.csv"
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg_df.to_csv(out_path, index=False)

    print(f"\n✓ Wrote {len(agg_df):,} total rows to {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
