"""
Transform ReEDS hourly load data to PowerGenome format.

Reads hourly load data from the ReEDS demand HDF5 (on Zenodo), distributes the
state-level demand to ReEDS balancing areas using ReEDS county load profile
factors, and transforms to tidy format matching the PowerGenome data structure.
"""

import os
from io import StringIO

import pandas as pd
import requests

# URLs for ReEDS demand data and county-level mapping files (ReEDS-Model/ReEDS)
DEMAND_H5_URL = "https://zenodo.org/records/18423998/files/demand_EER2023_IRAlow.h5"
COUNTY_STATE_LPF_URL = (
    "https://raw.githubusercontent.com/ReEDS-Model/ReEDS/main/"
    "inputs/disaggregation/county_state_lpf.csv"
)
COUNTY2ZONE_URL = (
    "https://raw.githubusercontent.com/ReEDS-Model/ReEDS/main/"
    "inputs/zones/z134/county2zone.csv"
)

# US state FIPS code -> abbreviation (used to join county FIPS to state demand)
FIPS_TO_ABBREV = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}


def download_h5_file(url, cache_dir="cache"):
    """
    Download HDF5 file from GitHub, handling Git LFS.

    Args:
        url: URL to the HDF5 file
        cache_dir: Directory to cache downloaded files

    Returns:
        Path to the downloaded/cached file
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache filename from URL
    filename = url.split("/")[-1]
    cache_path = os.path.join(cache_dir, filename)

    # Check if file already exists
    if os.path.exists(cache_path):
        print(f"Found cached file: {cache_path}")
        # Verify it's a valid HDF5 file
        with open(cache_path, "rb") as f:
            if f.read(8) == b"\x89HDF\r\n\x1a\n":
                print(f"  Using cached file ({os.path.getsize(cache_path):,} bytes)")
                return cache_path
            else:
                print("  Cached file is invalid, re-downloading...")
                os.remove(cache_path)

    print(f"Downloading HDF5 file from {url}...")

    # For Git LFS files on GitHub, we need to use the media URL
    # Convert github.com/owner/repo/blob/branch/path to github.com/owner/repo/raw/branch/path
    if "github.com" in url and "/blob/" in url:
        url = url.replace("/blob/", "/raw/")

    # Make initial request
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    # Check if we got a Git LFS pointer file
    content_start = response.content[:200]
    if b"version https://git-lfs.github.com/spec/" in content_start:
        print("  Detected Git LFS pointer, following to actual file...")
        # Parse the LFS pointer to get the actual download URL
        # LFS files are served from media.githubusercontent.com
        # We can use the GitHub media URL pattern
        lfs_url = url.replace("github.com", "media.githubusercontent.com").replace(
            "/raw/", "/media/"
        )

        print(f"  Downloading from LFS: {lfs_url}")
        response = requests.get(lfs_url, stream=True, allow_redirects=True)
        response.raise_for_status()

    # Verify it's an HDF5 file (starts with HDF5 signature)
    if not response.content[:8] == b"\x89HDF\r\n\x1a\n":
        raise ValueError("Downloaded file is not a valid HDF5 file")

    print(f"  Downloaded {len(response.content):,} bytes")

    # Save to cache
    with open(cache_path, "wb") as f:
        f.write(response.content)
    print(f"  Cached to: {cache_path}")

    return cache_path


def load_county_to_ba_mapping(
    lpf_url=COUNTY_STATE_LPF_URL, county2zone_url=COUNTY2ZONE_URL
):
    """
    Download county load profile factors and the county-to-BA map.

    ReEDS disaggregates state-level load to counties with a load profile factor
    (``county_state_lpf.csv``) and maps counties to ReEDS balancing areas via
    ``county2zone.csv``.

    Returns:
        DataFrame with columns [FIPS, value, ba, state]
    """
    print("Loading county load profile factors and county-to-BA mapping...")

    lpf = pd.read_csv(StringIO(requests.get(lpf_url).text), dtype={"FIPS": str})
    # FIPS in county_state_lpf.csv is 'p' + 5-digit county FIPS
    lpf["FIPS"] = lpf["FIPS"].str[1:]

    county2zone = pd.read_csv(
        StringIO(requests.get(county2zone_url).text), dtype={"FIPS": str}
    )
    county2zone = county2zone.rename(columns={"r": "ba"})

    mapping = lpf.merge(county2zone[["FIPS", "ba"]], on="FIPS", how="left")
    unmatched = int(mapping["ba"].isna().sum())
    if unmatched:
        print(f"  WARNING: {unmatched} counties without a BA will be skipped")

    mapping["state"] = mapping["FIPS"].str[:2].map(FIPS_TO_ABBREV)
    mapping = mapping.dropna(subset=["state", "ba"])

    # Verify county factors sum to 1 within each state
    state_sums = mapping.groupby("state")["value"].sum()
    if (state_sums - 1.0).abs().max() > 1e-3:
        raise ValueError("County load profile factors do not sum to 1 per state")
    print(
        f"  Loaded {len(mapping):,} county factors across "
        f"{state_sums.shape[0]} states"
    )
    return mapping[["FIPS", "value", "ba", "state"]]


def aggregate_state_load_to_ba(df, mapping):
    """
    Distribute state-level hourly load to ReEDS balancing areas (BAs).

    Each state's load is split across its counties using the county load profile
    factor (``value``, which sums to ~1 per state) and the counties are then
    aggregated to their BA. States that appear in the mapping but not in the
    demand data (e.g. DC, absent from the HDF5) are skipped.

    Args:
        df: Wide DataFrame with state columns plus year/weather_year/datetime
        mapping: DataFrame from load_county_to_ba_mapping()

    Returns:
        Wide DataFrame with one column per BA plus year/weather_year/datetime
    """
    region_cols = [c for c in df.columns if c not in ["year", "datetime", "weather_year"]]

    # Fraction of each state's hourly load in each BA (sums to ~1 per state)
    ba_share = (
        mapping.groupby(["state", "ba"], observed=True)["value"]
        .sum()
        .rename("share")
        .reset_index()
    )

    # Long form: one row per (timestep, state)
    long = df.melt(
        id_vars=["year", "weather_year", "datetime"],
        value_vars=region_cols,
        var_name="state",
        value_name="load",
    )
    long = long.merge(ba_share, on="state", how="inner")
    long["load"] = long["load"] * long["share"]

    # Aggregate to BA and pivot back to wide (one column per BA)
    ba_long = (
        long.groupby(["year", "weather_year", "datetime", "ba"])["load"]
        .sum()
        .reset_index()
    )
    wide = ba_long.pivot_table(
        index=["year", "weather_year", "datetime"], columns="ba", values="load"
    ).reset_index()
    wide = wide.fillna(0.0)

    ba_cols = [c for c in wide.columns if c not in ["year", "weather_year", "datetime"]]
    print(f"  Distributed state load across {len(ba_cols)} BAs")
    return wide


def load_reeds_load_data(url, weather_years=None):
    """
    Load ReEDS hourly load data from HDF5 file.

    Args:
        url: URL to the HDF5 file
        weather_years: List of weather years to load (default: [2007, 2008, 2009, 2010, 2011, 2012, 2013])

    Returns:
        DataFrame with loaded data
    """
    if weather_years is None:
        weather_years = [2007, 2008, 2009, 2010, 2011, 2012, 2013]

    # Download or get cached file
    h5_path = download_h5_file(url)

    try:
        print("Loading HDF5 data...")

        # Inspect the HDF5 file structure and read per-model-year groups
        import h5py

        with h5py.File(h5_path, "r") as f:
            year_groups = sorted(f.keys())
            print(f"  HDF5 groups (model years): {year_groups}")

            frames = []
            for yr in year_groups:
                g = f[yr]
                state_names = [
                    c.decode("utf-8") if isinstance(c, bytes) else str(c)
                    for c in g["columns"][:]
                ]
                dt = [
                    d.decode("utf-8") if isinstance(d, bytes) else str(d)
                    for d in g["datetime"][:]
                ]
                print(f"  {yr}: {len(state_names)} states, {len(dt)} timesteps")

                df_yr = pd.DataFrame({c: g[c][:] for c in state_names})
                df_yr["year"] = int(yr)
                df_yr["datetime"] = pd.to_datetime(dt)
                df_yr["weather_year"] = df_yr["datetime"].dt.year
                frames.append(df_yr)

            df = pd.concat(frames, ignore_index=True)

        print(f"  Regions (columns): {[c for c in df.columns if c not in ['year', 'datetime', 'weather_year']]}")

        # Filter to desired weather years
        print(f"  Filtering to weather years: {weather_years}")
        df = df[df["weather_year"].isin(weather_years)].copy()

        # Filter to years 2020-2050
        print(f"  Filtering to years 2020-2050")
        df = df[(df["year"] >= 2020) & (df["year"] <= 2050)].copy()
        print(f"  Filtered data shape: {df.shape}")

        # Distribute state-level demand to ReEDS BAs via county load profile
        # factors (county_state_lpf.csv) and the county-to-BA map (county2zone.csv)
        mapping = load_county_to_ba_mapping()
        df = aggregate_state_load_to_ba(df, mapping)

        return df

    except Exception as e:
        print(f"Error loading HDF5 file: {e}")
        raise


def transform_to_tidy_format(df, scenario="IRA_low"):
    """
    Transform wide format load data to tidy format.

    Args:
        df: DataFrame with columns [year, datetime, weather_year, region1, region2, ...]
        scenario: Scenario name to add as a column

    Returns:
        Tidy DataFrame with columns [time_index, weather_year, region, load_mw, year, scenario]
    """
    print("Transforming to tidy format...")

    # Get region columns (all columns except year, datetime, weather_year)
    region_cols = [
        col for col in df.columns if col not in ["year", "datetime", "weather_year"]
    ]

    # Melt to tidy format
    tidy = df.melt(
        id_vars=["year", "datetime", "weather_year"],
        value_vars=region_cols,
        var_name="region",
        value_name="load_mw",
    )

    # Sort by year, region, weather_year, datetime
    tidy = tidy.sort_values(["year", "region", "weather_year", "datetime"]).reset_index(
        drop=True
    )

    # Create time_index (1-8760 within each year/region/weather_year group)
    tidy["time_index"] = tidy.groupby(["year", "region", "weather_year"]).cumcount() + 1

    # Add scenario column
    tidy["scenario"] = scenario

    # Select and order columns to match test data format
    tidy = tidy[
        [
            "time_index",
            # "datetime",
            "weather_year",
            "region",
            "load_mw",
            "year",
            "scenario",
        ]
    ]

    print(f"  Tidy data shape: {tidy.shape}")
    print(f"  Unique years: {sorted(tidy['year'].unique())}")
    print(f"  Unique weather years: {sorted(tidy['weather_year'].unique())}")
    print(f"  Unique regions: {sorted(tidy['region'].unique())}")
    print(
        f"  Time index range: {tidy['time_index'].min()} - {tidy['time_index'].max()}"
    )

    return tidy


def main():
    """Main function to transform ReEDS load data."""
    print("=" * 60)
    print("ReEDS Load Data Transformation")
    print("=" * 60)

    # Configuration
    url = DEMAND_H5_URL
    weather_years = [2007, 2008, 2009, 2010, 2011, 2012, 2013]
    scenario = "IRA_low"
    output_file = "reeds_load_transformed.parquet"

    # Load data
    df = load_reeds_load_data(url, weather_years=weather_years)

    # Transform to tidy format
    tidy_df = transform_to_tidy_format(df, scenario=scenario)

    # Export to parquet
    print(f"\nExporting to {output_file}...")
    tidy_df.to_parquet(output_file, index=False)

    # Summary
    print("\n" + "=" * 60)
    print(f"✓ Exported {len(tidy_df):,} rows to {output_file}")
    print("=" * 60)

    # Show sample
    print("\nSample data:")
    print(tidy_df.head(10))


if __name__ == "__main__":
    main()
