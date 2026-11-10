"""
Transform ReEDS hourly load data to PowerGenome format.

Reads hourly load data from ReEDS HDF5 file and transforms to tidy format
matching the PowerGenome test data structure.
"""

import argparse
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests


SOURCE_URL = (
    "https://github.com/NatLabRockies/ReEDS-2.0/blob/v2025.7.0/"
    "inputs/load/EER_IRAlow_load_hourly.h5"
)
OUTPUT_FILE = Path("data/reeds_load_transformed.parquet")
HOURS_PER_YEAR = 8760
TIME_METADATA = {
    b"timezone": b"UTC",
    b"source_timezone": b"UTC-06:00",
    b"calendar": b"8760 hours; retain Feb 29, omit Dec 31 in leap years",
    b"hour_label": b"interval start; time_index 1 = Jan 1 00:00 UTC",
    b"year_boundary": b"cyclic +6-hour roll within each model/weather year and region",
    b"source_url": SOURCE_URL.encode(),
}


def utc_calendar(weather_year, time_index):
    """Map ReEDS' 1-based 8760-hour calendar to timezone-aware UTC starts."""
    return pd.to_datetime(weather_year.astype(str), format="%Y", utc=True) + pd.to_timedelta(
        time_index - 1, unit="h"
    )


def shift_annual_values(values):
    """Preserve each weather year's energy while placing CST values on UTC hours."""
    if len(values) != HOURS_PER_YEAR:
        raise ValueError(f"Expected 8760 hours per annual group, got {len(values)}")
    return np.roll(values, 6, axis=0)


def add_time_metadata(table):
    return table.replace_schema_metadata({**(table.schema.metadata or {}), **TIME_METADATA})


def migrate_existing_load(path=OUTPUT_FILE, batch_size=1_048_576):
    """Atomically correct the legacy sorted parquet without loading it all into RAM.

    Only legacy files without datetime_utc are accepted, preventing a second shift.
    Complete annual groups are carried across row-group boundaries before rolling.
    """
    path = Path(path)
    source = pq.ParquetFile(path)
    if "datetime_utc" in source.schema_arrow.names:
        raise ValueError("datetime_utc already exists; refusing to shift load twice")
    keys = ["year", "region", "weather_year", "scenario"]
    pending = None
    writer = None
    previous_key = None
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as f:
        temporary = Path(f.name)
    try:
        def write_groups(frame):
            nonlocal writer, previous_key
            for key, group in frame.groupby(keys, sort=False, observed=True):
                if previous_key is not None and key <= previous_key:
                    raise ValueError("Legacy load must be sorted by year, region, weather_year, scenario")
                previous_key = key
                if not np.array_equal(group["time_index"].to_numpy(), np.arange(1, 8761)):
                    raise ValueError(f"Invalid hourly grid for {key}")
                frame.loc[group.index, "load_mw"] = shift_annual_values(group["load_mw"].to_numpy())
            frame.insert(1, "datetime_utc", utc_calendar(frame["weather_year"], frame["time_index"]))
            table = add_time_metadata(pa.Table.from_pandas(frame, preserve_index=False))
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd", compression_level=3)
            writer.write_table(table)

        for batch in source.iter_batches(batch_size=batch_size):
            frame = batch.to_pandas()
            if pending is not None:
                frame = pd.concat([pending, frame], ignore_index=True)
            last_key = frame[keys].iloc[-1]
            last_group = frame[keys].eq(last_key).all(axis=1)
            pending = frame.loc[last_group].copy()
            complete = frame.loc[~last_group].copy()
            if not complete.empty:
                write_groups(complete)
        if pending is not None:
            write_groups(pending)
        if writer is None:
            raise ValueError("Cannot migrate an empty load file")
        writer.close()
        writer = None
        source.close()
        os.replace(temporary, path)
    finally:
        source.close()
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


# Representative ReEDS BAs, east to west; offsets deliberately exclude DST.
DIAGNOSTIC_REGIONS = {
    "p131": ("Massachusetts", -5),
    "p81": ("Illinois", -6),
    "p33": ("Colorado", -7),
    "p10": ("California", -8),
}


def diagnose_load_timing(path=OUTPUT_FILE, model_year=2025, weather_year=2012):
    """Warn about suspicious seasonal shapes in a small sample of generated load.

    Uses the available model/weather pair nearest 2025/2012, all its scenarios,
    and four representative BAs. This heuristic cannot establish absolute UTC
    alignment; the source-calendar and exact-shift tests remain authoritative.
    Returns the seasonal summary without modifying the file.
    """
    import duckdb

    def flag(message):
        warnings.warn(f"Load timing diagnostic: {message}", UserWarning, stacklevel=2)

    with pq.ParquetFile(path) as source:
        schema = source.schema_arrow
    if ("datetime_utc" not in schema.names
            or not pa.types.is_timestamp(schema.field("datetime_utc").type)
            or schema.field("datetime_utc").type.tz != "UTC"):
        flag("datetime_utc must be a timezone-aware UTC timestamp; cannot assess timing.")
        return pd.DataFrame()

    with duckdb.connect() as con:
        con.execute("SET TimeZone='UTC'")
        sample = con.execute(
            "SELECT DISTINCT year, weather_year FROM read_parquet(?) "
            "WHERE year IS NOT NULL AND weather_year IS NOT NULL "
            "ORDER BY abs(year - ?), abs(weather_year - ?), year, weather_year LIMIT 1",
            [str(path), model_year, weather_year],
        ).fetchone()
        if sample is None:
            flag("no model/weather years available.")
            return pd.DataFrame()
        model_year, weather_year = sample
        scenarios = con.execute(
            "SELECT DISTINCT scenario FROM read_parquet(?) WHERE year=? AND weather_year=?",
            [str(path), model_year, weather_year],
        ).fetchall()
        frame = con.execute(
            "SELECT region, scenario, time_index, datetime_utc, load_mw "
            "FROM read_parquet(?) WHERE year=? AND weather_year=? "
            "AND region IN (SELECT unnest(?)) ORDER BY time_index",
            [str(path), model_year, weather_year, list(DIAGNOSTIC_REGIONS)],
        ).df()

    print(f"Load timing diagnostic: model year {model_year}, weather year {weather_year}")
    summaries = []
    for (scenario,) in scenarios:
        for region, (state, offset) in DIAGNOSTIC_REGIONS.items():
            label = f"{region} ({state}), scenario {scenario}, {model_year}/{weather_year}"
            scenario_rows = frame.scenario.isna() if pd.isna(scenario) else frame.scenario.eq(scenario)
            group = frame.loc[(frame.region == region) & scenario_rows].copy()
            if group.empty:
                flag(f"{label}: missing representative BA.")
                continue
            timestamps = pd.DatetimeIndex(group.datetime_utc)
            expected = pd.date_range(f"{weather_year}-01-01", periods=8760, freq="h", tz="UTC")
            if (not np.array_equal(group.time_index.to_numpy(), np.arange(1, 8761))
                    or not timestamps.as_unit("ns").equals(expected)):
                flag(f"{label}: incomplete, duplicate or misaligned annual UTC hourly grid.")
                continue
            if not np.isfinite(group.load_mw).all() or (group.load_mw < 0).any():
                flag(f"{label}: nonfinite or negative load values; cannot assess shape.")
                continue
            # Fixed standard-time dates avoid DST's repeated/missing local hours.
            local = group.datetime_utc + pd.Timedelta(hours=offset)
            group["local_hour"] = local.dt.hour
            for season, months in [("winter", [12, 1, 2]), ("summer", [6, 7, 8])]:
                selected = group.loc[local.dt.month.isin(months) & (local.dt.dayofweek < 5)]
                hourly = selected.groupby("local_hour").load_mw.agg(["mean", "count"])
                if len(hourly) != 24 or hourly["count"].min() < 10:
                    flag(f"{label}, {season}: insufficient weekday coverage for all 24 hours.")
                    continue
                shape = hourly["mean"]
                peak, minimum = int(shape.idxmax()), int(shape.idxmin())
                issues = []
                if shape.max() - shape.min() <= 0.01 * abs(shape.mean()):
                    issues.append("near-flat average daily shape (range <=1% of mean)")
                else:
                    if not 12 <= peak <= 22:
                        issues.append(f"peak at {peak:02d}:00 standard time, outside 12:00-22:00")
                    if not 0 <= minimum <= 8:
                        issues.append(f"minimum at {minimum:02d}:00 standard time, outside 00:00-08:00")
                for issue in issues:
                    flag(f"{label}, {season}: {issue}; inspect source timezone and load shape.")
                summaries.append({
                    "region": region, "state": state, "year": model_year,
                    "weather_year": weather_year, "scenario": scenario, "season": season,
                    "peak_utc_hour": (peak - offset) % 24,
                    "minimum_utc_hour": (minimum - offset) % 24,
                    "peak_standard_hour": peak, "minimum_standard_hour": minimum,
                    "status": "warning" if issues else "ok",
                })
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        print(summary.to_string(index=False))
    return summary


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
        Tidy DataFrame with columns [time_index, datetime_utc, weather_year, region,
        load_mw, year, scenario]. datetime_utc labels interval starts on the
        cyclic 8760-hour UTC calendar; time_index remains 1-based.
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

    # Validate source labels before replacing them with the cyclic UTC calendar.
    for key, group in tidy.groupby(["year", "region", "weather_year"], sort=False):
        timestamps = pd.DatetimeIndex(group["datetime"])
        if timestamps.tz is None or any(
            timestamp.utcoffset().total_seconds() != -21600 for timestamp in timestamps
        ):
            raise ValueError(f"Expected timezone-aware UTC-06:00 source timestamps for {key}")
        expected = pd.date_range(f"{key[2]}-01-01", periods=8760, freq="h", tz="Etc/GMT+6")
        if not np.array_equal(timestamps.asi8, expected.asi8):
            raise ValueError(f"Invalid source hourly calendar for {key}")
        tidy.loc[group.index, "load_mw"] = shift_annual_values(group["load_mw"].to_numpy())

    tidy["datetime_utc"] = utc_calendar(tidy["weather_year"], tidy["time_index"])
    # Add scenario column
    tidy["scenario"] = scenario

    # Select and order columns to match test data format
    tidy = tidy[
        [
            "time_index",
            "datetime_utc",
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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migrate-existing", action="store_true", help="Correct legacy output without downloading source HDF5")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--diagnose-only", action="store_true", help="Check existing output without regenerating or shifting it")
    args = parser.parse_args()
    if args.diagnose_only and args.migrate_existing:
        parser.error("--diagnose-only cannot be combined with --migrate-existing")
    if args.diagnose_only:
        diagnose_load_timing(args.output)
        return
    if args.migrate_existing:
        migrate_existing_load(args.output)
        diagnose_load_timing(args.output)
        return

    url = SOURCE_URL
    weather_years = [2007, 2008, 2009, 2010, 2011, 2012, 2013]
    scenario = "IRA_low"
    output_file = args.output

    # Load data
    df = load_reeds_load_data(url, weather_years=weather_years)

    # Transform to tidy format
    tidy_df = transform_to_tidy_format(df, scenario=scenario)

    # Export to parquet
    print(f"\nExporting to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        add_time_metadata(pa.Table.from_pandas(tidy_df, preserve_index=False)),
        output_file,
        compression="zstd",
        compression_level=3,
    )

    diagnose_load_timing(output_file)

    # Summary
    print("\n" + "=" * 60)
    print(f"✓ Exported {len(tidy_df):,} rows to {output_file}")
    print("=" * 60)

    # Show sample
    print("\nSample data:")
    print(tidy_df.head(10))


if __name__ == "__main__":
    main()
