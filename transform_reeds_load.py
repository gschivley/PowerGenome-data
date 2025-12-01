"""
Transform ReEDS hourly load data to PowerGenome format.

Reads hourly load data from ReEDS HDF5 file and transforms to tidy format
matching the PowerGenome test data structure.
"""

import os

import pandas as pd
import requests


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

        # First, inspect the HDF5 file structure
        import h5py

        with h5py.File(h5_path, "r") as f:
            print(f"  HDF5 keys: {list(f.keys())}")

            # Read the arrays directly
            data = f["data"][:]
            columns = f["columns"][:]
            index_0 = f["index_0"][:]  # year
            index_1 = f["index_1"][:]  # datetime

            print(f"  Data shape: {data.shape}")
            print(f"  Columns shape: {columns.shape}")
            print(f"  Index_0 (year) shape: {index_0.shape}")
            print(f"  Index_1 (datetime) shape: {index_1.shape}")

        # Decode bytes to strings if needed
        if columns.dtype.kind in ["S", "O"]:
            columns = [
                c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in columns
            ]

        # Decode datetime if stored as bytes
        if index_1.dtype.kind in ["S", "O"]:
            index_1 = [
                d.decode("utf-8") if isinstance(d, bytes) else str(d) for d in index_1
            ]

        print(f"  Years: {sorted(set(index_0))}")
        print(f"  Regions (columns): {columns}")

        # Convert to DataFrame with proper index
        df = pd.DataFrame(data, columns=columns)
        df["year"] = index_0

        # Add datetime - convert to pandas datetime
        df["datetime"] = pd.to_datetime(index_1)

        # Extract weather year from datetime
        df["weather_year"] = df["datetime"].dt.year

        # Filter to desired weather years
        print(f"  Filtering to weather years: {weather_years}")
        df = df[df["weather_year"].isin(weather_years)].copy()

        # Filter to years 2020-2050
        print(f"  Filtering to years 2020-2050")
        df = df[(df["year"] >= 2020) & (df["year"] <= 2050)].copy()
        print(f"  Filtered data shape: {df.shape}")

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
    url = "https://github.com/NREL/ReEDS-2.0/blob/main/inputs/load/EER_IRAlow_load_hourly.h5"
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
