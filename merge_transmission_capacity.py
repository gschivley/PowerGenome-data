#!/usr/bin/env python3
"""Merge AC and non-AC transmission capacity data into a single CSV.

The old ReEDS (NREL/ReEDS-2.0) transmission capacity CSVs no longer exist in the
new repo (ReEDS-Model/ReEDS), so this script rebuilds the merged file from the
replacement sources:

- AC: ReEDS `inputs/transmission/itl_NARIS.csv` lists AC interties keyed by md5
  zone hashes. Each hash is mapped to a z134 BA name via
  `inputs/zones/z134/zonehash.csv`. Interties whose endpoints are both in the
  134-BA map are kept, the average of MW_forward and MW_reverse is taken per
  intertie, the region pair is canonicalized by sorting, and the firm capacity
  is averaged again across the (~36) pairs that appear in both directions.
- Non-AC: ReEDS `inputs/transmission/hvdc_existing.csv` and
  `inputs/transmission/hvdc_planned-baseline.csv` list HVDC lines by lat/lon
  endpoints. Each endpoint is mapped to the nearest z134 BA via the haversine
  distance, keeping the line direction and using the line MW as capacity.
- AC and non-AC rows are aggregated by region pair: firm_ttc_mw is summed (so
  AC + non-AC capacity for the same pair add up) and notes are joined.

Output columns: region_from,region_to,firm_ttc_mw,notes

Usage:
  python merge_transmission_capacity.py
"""
import math
from pathlib import Path

import pandas as pd
import requests

# GitHub URLs for transmission files (new ReEDS-Model/ReEDS repo)
ITL_URL = "https://raw.githubusercontent.com/ReEDS-Model/ReEDS/main/inputs/transmission/itl_NARIS.csv"
ZONEHASH_URL = "https://raw.githubusercontent.com/ReEDS-Model/ReEDS/main/inputs/zones/z134/zonehash.csv"
HVDC_EXISTING_URL = "https://raw.githubusercontent.com/ReEDS-Model/ReEDS/main/inputs/transmission/hvdc_existing.csv"
HVDC_PLANNED_URL = "https://raw.githubusercontent.com/ReEDS-Model/ReEDS/main/inputs/transmission/hvdc_planned-baseline.csv"

EARTH_RADIUS_KM = 6371.0


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


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (lat, lon) points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def load_zonehash(csv_text):
    """Return (md5->r map, zonehash DataFrame) from the zonehash.csv text."""
    zonehash = pd.read_csv(pd.io.common.StringIO(csv_text))
    md5_to_r = dict(zip(zonehash["md5"].astype(str), zonehash["r"]))
    return md5_to_r, zonehash


def nearest_ba(zonehash, lat, lon):
    """Return (r, distance_km) of the z134 BA node nearest to (lat, lon)."""
    distances = zonehash.apply(
        lambda row: haversine(lat, lon, row["node_lat"], row["node_lon"]), axis=1
    )
    idx = distances.idxmin()
    return zonehash.loc[idx, "r"], distances[idx]


def parse_ac(itl_text, md5_to_r):
    """Parse AC interties into avg forward/reverse MW per canonical region pair."""
    df = pd.read_csv(pd.io.common.StringIO(itl_text))
    df["md5_from"] = df["md5_from"].astype(str)
    df["md5_to"] = df["md5_to"].astype(str)

    # Keep only interties whose endpoints are both in the 134-BA zone hash map
    df = df[df["md5_from"].isin(md5_to_r) & df["md5_to"].isin(md5_to_r)].copy()

    # Average forward/reverse MW per intertie, then map hashes to BA names
    df["firm"] = (df["MW_forward"] + df["MW_reverse"]) / 2.0
    df["region_from"] = df["md5_from"].map(md5_to_r)
    df["region_to"] = df["md5_to"].map(md5_to_r)

    # Canonicalize each pair by sorting so direction doesn't matter
    regions = df[["region_from", "region_to"]]
    df["region_from"] = regions.min(axis=1)
    df["region_to"] = regions.max(axis=1)

    # Average firm MW across the ~36 pairs that appear in both directions
    df = df.groupby(["region_from", "region_to"], as_index=False)["firm"].mean()
    df = df.rename(columns={"firm": "firm_ttc_mw"})
    df["notes"] = ""

    return df[["region_from", "region_to", "firm_ttc_mw", "notes"]]


def parse_nonac(existing_text, planned_text, zonehash):
    """Parse HVDC lines, mapping each endpoint to the nearest z134 BA."""
    rows = []
    for csv_text in (existing_text, planned_text):
        df = pd.read_csv(pd.io.common.StringIO(csv_text))
        for _, line in df.iterrows():
            from_ba, _ = nearest_ba(zonehash, line["from_lat"], line["from_lon"])
            to_ba, _ = nearest_ba(zonehash, line["to_lat"], line["to_lon"])
            rows.append(
                {
                    "region_from": from_ba,
                    "region_to": to_ba,
                    "firm_ttc_mw": float(line["MW"]),
                    "notes": str(line["name"]),
                }
            )
    return pd.DataFrame(rows, columns=["region_from", "region_to", "firm_ttc_mw", "notes"])


def main():
    """Main execution function."""
    print("=" * 60)
    print("Merging ReEDS Transmission Capacity Data")
    print("=" * 60)

    # Download files
    itl_text = download_file(ITL_URL, "cache_data/itl_NARIS.csv")
    zonehash_text = download_file(ZONEHASH_URL, "cache_data/zonehash.csv")
    hvdc_existing_text = download_file(HVDC_EXISTING_URL, "cache_data/hvdc_existing.csv")
    hvdc_planned_text = download_file(HVDC_PLANNED_URL, "cache_data/hvdc_planned-baseline.csv")

    md5_to_r, zonehash = load_zonehash(zonehash_text)

    # Parse data
    print("\nParsing AC transmission data...")
    ac_df = parse_ac(itl_text, md5_to_r)
    print(f"  Loaded {len(ac_df):,} AC region pairs")

    print("Parsing non-AC transmission data...")
    nonac_df = parse_nonac(hvdc_existing_text, hvdc_planned_text, zonehash)
    print(f"  Loaded {len(nonac_df):,} non-AC line rows")

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
