#!/usr/bin/env python3
"""
Download EIA AEO bulk data and extract electric power sector fuel prices.
Updates data/fuel_prices.parquet with new data, removing duplicates.

Run from the repo root:
    uv run .github/skills/eia-fuel-prices/scripts/fetch_eia_fuel_prices.py --years 2026
    uv run .github/skills/eia-fuel-prices/scripts/fetch_eia_fuel_prices.py --years 2025 2026
"""

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

BULK_URL = "https://www.eia.gov/opendata/bulk/AEO{year}.zip"

SERIES_FILTER = "Energy Prices : Electric Power"

# EIA series name fuel component → PowerGenome fuel name
# AEO2025 uses "Uranium"; AEO2026 uses "Nuclear Fuel"
FUEL_MAP = {
    "Steam Coal": "coal",
    "Natural Gas": "naturalgas",
    "Distillate Fuel Oil": "distillate",
    "Nuclear Fuel": "uranium",
    "Uranium": "uranium",
}

# EIA census division name (as it appears in series names) → hierarchy.csv cendiv value
CENDIV_MAP = {
    "New England": "New_England",
    "Middle Atlantic": "Mid_Atlantic",
    "East North Central": "East_North_Central",
    "West North Central": "West_North_Central",
    "South Atlantic": "South_Atlantic",
    "East South Central": "East_South_Central",
    "West South Central": "West_South_Central",
    "Mountain": "Mountain",
    "Pacific": "Pacific",
}

# Special-case scenario name overrides (matched against the raw EIA scenario name)
SCENARIO_OVERRIDES = {
    "Alternative Electricity": "no_111d",
    "Counterfactual Baseline": "baseline",
}


def normalize_scenario(raw_name: str) -> str:
    """Convert EIA scenario name to snake_case, applying special-case overrides."""
    if raw_name in SCENARIO_OVERRIDES:
        return SCENARIO_OVERRIDES[raw_name]
    return raw_name.lower().replace(" ", "_")


def download_bulk(aeo_year: int, cache_dir: Path) -> Path:
    """Download AEO{year}.zip to cache_dir if not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"AEO{aeo_year}.zip"
    if zip_path.exists():
        print(f"  Using cached {zip_path}")
        return zip_path
    url = BULK_URL.format(year=aeo_year)
    print(f"  Downloading {url} ...")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(
                        f"\r    {downloaded // (1 << 20)} / {total // (1 << 20)} MB ({pct:.0f}%)",
                        end="",
                        flush=True,
                    )
    print(f"\n  Saved to {zip_path}")
    return zip_path


def read_bulk_file(zip_path: Path) -> pd.DataFrame:
    """Read the JSON-lines .txt file inside the AEO ZIP."""
    with zipfile.ZipFile(zip_path) as zf:
        txt_names = [n for n in zf.namelist() if n.endswith(".txt")]
        if not txt_names:
            raise ValueError(f"No .txt file found in {zip_path}")
        txt_name = txt_names[0]
        print(f"  Reading {txt_name} from ZIP ...")
        with zf.open(txt_name) as f:
            df = pd.read_json(f, lines=True)
    return df


def extract_prices_from_bulk(
    bulk_df: pd.DataFrame, aeo_year: int, hierarchy_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter the bulk DataFrame to electric power fuel price series and expand
    to one row per (year, scenario, fuel, ReEDS BA).

    Series name format expected:
        "AEO{year} : {Scenario Name} : Energy Prices : Electric Power : {Fuel} : {Census Division}"
    """
    mask = bulk_df["name"].str.contains(SERIES_FILTER, na=False)
    price_series = bulk_df[mask].copy()
    print(f"  Found {len(price_series)} '{SERIES_FILTER}' series")

    rows = []
    skipped_fuels: set = set()
    skipped_cendivs: set = set()

    for _, row in price_series.iterrows():
        name: str = row["name"]
        # Format: "Energy Prices : Electric Power : {Fuel}, {Census Division}, {Scenario}, AEO{year}"
        # Multi-year comparisons: "..., AEO{y1}, AEO{y2}" — skip those.
        colon_parts = [p.strip() for p in name.split(" : ")]
        if len(colon_parts) < 3:
            print(f"  WARNING: Cannot parse series name: {name!r}", file=sys.stderr)
            continue

        # The third colon-segment is comma-separated: fuel, cendiv, scenario, AEO-year
        sub = [s.strip() for s in colon_parts[2].split(", ")]
        if len(sub) < 4:
            # No scenario field — likely a US-total or incomplete entry; skip silently
            continue

        fuel_raw = sub[0]
        cendiv_raw = sub[1]
        scenario_raw = sub[2]

        # Multi-year comparison entries have "AEO{year}" as sub[2]; skip them
        if scenario_raw.startswith("AEO"):
            continue

        fuel = FUEL_MAP.get(fuel_raw)
        if fuel is None:
            skipped_fuels.add(fuel_raw)
            continue

        cendiv = CENDIV_MAP.get(cendiv_raw)
        if cendiv is None:
            skipped_cendivs.add(cendiv_raw)
            continue

        scenario = normalize_scenario(scenario_raw)

        # Get ReEDS BAs for this census division
        bas = hierarchy_df.loc[hierarchy_df["cendiv"] == cendiv, "ba"].tolist()
        if not bas:
            print(f"  WARNING: No BAs found for cendiv={cendiv!r}", file=sys.stderr)
            continue

        # Parse time series data: list of [year, price] pairs
        data = row.get("data") or []
        if not data:
            continue
        try:
            ts = pd.DataFrame(data, columns=["year", "price"])
        except Exception as exc:
            print(
                f"  WARNING: Failed to parse data for {name!r}: {exc}", file=sys.stderr
            )
            continue
        ts["year"] = pd.to_numeric(ts["year"], errors="coerce")
        ts["price"] = pd.to_numeric(ts["price"], errors="coerce")
        ts = ts.dropna(subset=["year", "price"])
        if ts.empty:
            continue

        # Replicate price for each BA in this census division
        for ba in bas:
            ba_ts = ts.copy()
            ba_ts["data_year"] = aeo_year
            ba_ts["scenario"] = scenario
            ba_ts["fuel"] = fuel
            ba_ts["region"] = ba
            ba_ts["dollar_year"] = aeo_year - 1
            rows.append(ba_ts)

    if skipped_fuels:
        print(f"  INFO: Unmapped EIA fuel names (skipped): {sorted(skipped_fuels)}")
    if skipped_cendivs:
        print(f"  INFO: Unmapped census divisions (skipped): {sorted(skipped_cendivs)}")

    if not rows:
        print("  WARNING: No price data extracted!", file=sys.stderr)
        return pd.DataFrame(
            columns=[
                "year",
                "price",
                "data_year",
                "scenario",
                "fuel",
                "region",
                "dollar_year",
            ]
        )

    result = pd.concat(rows, ignore_index=True)
    result["year"] = result["year"].astype(int)
    return result[
        ["year", "price", "data_year", "scenario", "fuel", "region", "dollar_year"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch EIA AEO electric-power fuel prices and update fuel_prices.parquet"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="AEO year(s) to fetch, e.g. --years 2026 or --years 2025 2026",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/fuel_prices.parquet"),
        help="Path to output parquet file (default: data/fuel_prices.parquet)",
    )
    parser.add_argument(
        "--hierarchy",
        type=Path,
        default=Path("cache_data/hierarchy.csv"),
        help="ReEDS hierarchy CSV with 'ba' and 'cendiv' columns (default: cache_data/hierarchy.csv)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache_data/eia_bulk"),
        help="Directory for cached AEO ZIP files (default: cache_data/eia_bulk)",
    )
    args = parser.parse_args()

    # Load hierarchy
    if not args.hierarchy.exists():
        print(f"ERROR: Hierarchy file not found: {args.hierarchy}", file=sys.stderr)
        sys.exit(1)
    hierarchy_df = pd.read_csv(args.hierarchy, usecols=["ba", "cendiv"])

    # Load existing fuel prices (parquet preferred; fall back to CSV for migration)
    csv_path = args.output.with_suffix(".csv")
    if args.output.exists():
        existing = pd.read_parquet(args.output)
        print(f"Loaded {args.output} with {len(existing):,} existing rows")
    elif csv_path.exists():
        print(f"Migrating {csv_path} → {args.output} ...")
        existing = pd.read_csv(csv_path)
        print(f"Loaded {len(existing):,} rows from CSV")
    else:
        existing = pd.DataFrame(
            columns=[
                "year",
                "price",
                "data_year",
                "scenario",
                "fuel",
                "region",
                "dollar_year",
            ]
        )
        print(f"No existing file at {args.output}; will create it")

    all_new = []
    for year in args.years:
        print(f"\nProcessing AEO{year} ...")
        zip_path = download_bulk(year, args.cache_dir)
        bulk_df = read_bulk_file(zip_path)
        new_df = extract_prices_from_bulk(bulk_df, year, hierarchy_df)
        print(f"  Extracted {len(new_df):,} rows for AEO{year}")
        all_new.append(new_df)

    if not all_new or all(df.empty for df in all_new):
        print("Nothing to add.")
        return

    new_data = pd.concat(all_new, ignore_index=True)

    # Combine with existing data and deduplicate
    combined = pd.concat([existing, new_data], ignore_index=True)
    dedup_keys = ["year", "data_year", "scenario", "fuel", "region"]
    before = len(combined)
    combined = combined.drop_duplicates(subset=dedup_keys, keep="last")
    after = len(combined)
    if before != after:
        print(f"\nRemoved {before - after:,} duplicate rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    print(f"\nSaved {len(combined):,} rows to {args.output}")
    new_net = after - (
        len(existing) - (before - after - (len(new_data) - (before - after)))
    )
    print(
        f"Net new rows added: {len(new_data):,} extracted, {before - after:,} were duplicates"
    )


if __name__ == "__main__":
    main()
