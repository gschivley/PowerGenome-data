"""
Generate hourly hydroelectric generation profiles by region from 2007-13.

This script:
1. Merges plant-region mappings with generator data to identify hydro plants
2. Queries PUDL monthly EIA data from S3 to calculate monthly capacity factors
3. Interpolates to hourly values and applies 1-week moving average smoothing
4. Outputs tidy parquet files with site_id, weather_year, time_index, value
"""

import logging
from datetime import timedelta
from pathlib import Path

import duckdb
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# PUDL data is published to a public S3 bucket. DuckDB needs these settings to
# load the httpfs extension and reach the bucket. The bucket currently lives in
# us-west-2; if this ever changes, DuckDB will surface a 301 with the correct
# region to use here.
PUDL_S3_SETTINGS = [
    "INSTALL httpfs",
    "LOAD httpfs",
    "SET s3_region='us-west-2'",
    "SET s3_use_ssl=true",
    "SET s3_url_style='path'",
]

# Constants
PUDL_PARQUET_URL = "s3://pudl.catalyst.coop/nightly/out_eia__monthly_generators.parquet"
START_YEAR = 2007
END_YEAR = 2013
HOURS_PER_WEEK = 168

# Hydro generator types to include
HYDRO_TYPES = {
    "run_of_river": "Run of River Hydroelectric",
    "conventional": "Conventional Hydroelectric",
}


def hours_in_month(report_date) -> float:
    """Number of real calendar hours in the month of ``report_date``.

    December must return 744/736 hours like any other month. The original
    inline lambda built December's "next month" as Jan 1 of the *same* year,
    producing a negative divisor that flipped the capacity factor's sign and
    zeroed out every December after the 0-1 clip.
    """
    ts = pd.Timestamp(report_date)
    month_start = pd.Timestamp(ts.year, ts.month, 1)
    next_month_start = month_start + pd.DateOffset(months=1)
    return (next_month_start - month_start).total_seconds() / 3600


def load_plant_region_data() -> pd.DataFrame:
    """Load plant-to-region mapping."""
    logger.info("Loading plant-region mappings...")
    return pd.read_csv("data/plant_region_map.csv")


def load_generator_data() -> pd.DataFrame:
    """Load transformed REEDS generators and filter for hydroelectric plants."""
    logger.info("Loading generator data...")
    generators = pd.read_csv("data/reeds_generators_transformed.csv")

    # Filter for hydroelectric plants only
    hydro_mask = generators["technology"].isin(HYDRO_TYPES.values())
    generators = generators[hydro_mask].copy()

    # Classify by hydro type
    generators["hydro_type"] = generators["technology"].map(
        {v: k for k, v in HYDRO_TYPES.items()}
    )

    logger.info(f"Found {len(generators)} hydroelectric generators")
    return generators[["plant_id", "hydro_type"]].drop_duplicates()


def merge_plant_region_hydro_data() -> pd.DataFrame:
    """Merge plant-region and generator data."""
    logger.info("Merging plant-region and generator data...")

    plant_region = load_plant_region_data()
    generators = load_generator_data()

    # Merge on plant_id
    merged = plant_region.merge(generators, on="plant_id", how="inner")

    logger.info(f"Merged data has {len(merged)} plant-region-hydro_type combinations")
    return merged[["plant_id", "region", "hydro_type"]].drop_duplicates()


def query_pudl_monthly_capacity_factors(
    parquet_url: str = PUDL_PARQUET_URL,
    s3_settings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Query PUDL EIA monthly generators data to calculate regional monthly capacity factors.

    Args:
        parquet_url: Location of the PUDL monthly generators parquet. Defaults
            to the public S3 nightly path; tests pass a local file.
        s3_settings: DuckDB statements to run before the query (httpfs setup
            for S3). ``None`` uses ``PUDL_S3_SETTINGS``; an empty list skips
            them for local files.

    Returns:
        DataFrame with columns: region, hydro_type, report_date, capacity_factor
    """
    if s3_settings is None:
        s3_settings = PUDL_S3_SETTINGS
    logger.info(f"Querying PUDL parquet from {parquet_url} for {START_YEAR}-{END_YEAR}...")

    # Get plant-region mappings
    plant_hydro_map = merge_plant_region_hydro_data()

    # Use DuckDB to query parquet with filtering
    conn = duckdb.connect()
    for setting in s3_settings:
        conn.execute(setting)

    # Read PUDL data and join with region mapping in DuckDB.
    #
    # Hydro profiles should reflect only hydroelectric generation, so we
    # restrict to generators with prime_mover_code = 'HY'. Some hydro plants
    # share a plant_id with co-located pumped-storage ('PS', which reports
    # negative net generation) or thermal ('GT'/'IC'/'ST') generators. Since
    # net_generation_mwh is summed across the whole plant, including those
    # generators corrupts the hydro capacity factor and (for pumped storage)
    # can even drive it to zero via the clip below.
    pudl_data = conn.execute(
        f"""
        SELECT 
            pg.plant_id_eia,
            pg.report_date,
            pg.net_generation_mwh,
            pg.capacity_mw
        FROM 
            read_parquet('{parquet_url}') pg
        WHERE 
            EXTRACT(YEAR FROM pg.report_date) >= {START_YEAR}
            AND EXTRACT(YEAR FROM pg.report_date) <= {END_YEAR}
            AND pg.capacity_mw > 0
            AND pg.prime_mover_code = 'HY'
    """
    ).df()

    logger.info(f"Retrieved {len(pudl_data)} records from PUDL")

    # Convert plant_id to match our mapping
    pudl_data["plant_id"] = pudl_data["plant_id_eia"].astype(int)

    # Join with plant-region mapping
    merged = pudl_data.merge(plant_hydro_map, on="plant_id", how="inner")

    logger.info(f"Matched {len(merged)} records to regions")

    # Calculate monthly capacity factors by region and hydro type
    monthly_cf = (
        merged.groupby(["region", "hydro_type", "report_date"])
        .agg({"net_generation_mwh": "sum", "capacity_mw": "sum"})
        .reset_index()
    )

    # Capacity factor = monthly generation / (capacity * hours in month),
    # using true calendar-month hours (see hours_in_month for the December
    # regression this guards against).
    monthly_cf["hours_in_month"] = monthly_cf["report_date"].apply(hours_in_month)

    monthly_cf["capacity_factor"] = (
        monthly_cf["net_generation_mwh"]
        / (monthly_cf["capacity_mw"] * monthly_cf["hours_in_month"])
    ).clip(
        0, 1
    )  # Ensure 0-1 range

    monthly_cf = monthly_cf[["region", "hydro_type", "report_date", "capacity_factor"]]

    logger.info(f"Calculated {len(monthly_cf)} monthly capacity factors")
    return monthly_cf


def interpolate_monthly_to_hourly(
    monthly_cf: pd.DataFrame, region: str, hydro_type: str
) -> pd.DataFrame:
    """
    Convert monthly capacity factors to hourly values.

    Repeats each monthly value for all hours in that month, then applies
    1-week moving average smoothing. Removes Dec 31 from leap years to ensure
    8760 hours per year.
    """
    region_data = (
        monthly_cf[
            (monthly_cf["region"] == region) & (monthly_cf["hydro_type"] == hydro_type)
        ]
        .sort_values("report_date")
        .copy()
    )

    if len(region_data) == 0:
        return None

    # Create date range covering all months
    date_range = pd.date_range(
        start=region_data["report_date"].min(),
        end=region_data["report_date"].max(),
        freq="MS",  # Month start
    )

    hourly_list = []

    for i, month_start in enumerate(date_range):
        # Get next month start for day count
        if i < len(date_range) - 1:
            month_end = date_range[i + 1]
        else:
            month_end = month_start + pd.DateOffset(months=1)

        # Days in this month
        days_in_month = (month_end - month_start).days

        # For December in leap years, use 30 days instead of 31 to ensure 8760 hours/year
        is_leap_year = month_start.year % 4 == 0 and (
            month_start.year % 100 != 0 or month_start.year % 400 == 0
        )
        if month_start.month == 12 and is_leap_year:
            days_in_month = 30

        hours_in_month = days_in_month * 24

        # Get capacity factor for this month
        month_cf = region_data[region_data["report_date"] == month_start]
        if len(month_cf) == 0:
            # Dropping the month silently would shift every later hour onto
            # the wrong time_index, so make the gap visible in the logs.
            logger.warning(
                f"{region}/{hydro_type}: no capacity factor for "
                f"{month_start:%Y-%m}; skipping month"
            )
            continue

        cf_value = month_cf["capacity_factor"].values[0]

        # Create hourly values for this month
        for hour in range(hours_in_month):
            hourly_list.append(
                {
                    "report_date": month_start + timedelta(hours=hour),
                    "capacity_factor": cf_value,
                }
            )

    hourly_df = pd.DataFrame(hourly_list)

    # Apply 1-week (168-hour) moving average
    hourly_df["capacity_factor"] = (
        hourly_df["capacity_factor"]
        .rolling(window=HOURS_PER_WEEK, center=True, min_periods=1)
        .mean()
    )

    # Forward fill first 84 hours (half week), backward fill last 84 hours
    hourly_df["capacity_factor"] = hourly_df["capacity_factor"].ffill().bfill()

    return hourly_df


def extract_weather_years_and_time_indices(
    hourly_data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Extract weather_year and time_index (1-8760) from hourly timestamps."""
    weather_year = hourly_data["report_date"].dt.year

    # Calculate day of year and hour
    day_of_year = hourly_data["report_date"].dt.dayofyear
    hour_of_day = hourly_data["report_date"].dt.hour + 1  # 1-24

    # Time index: (day - 1) * 24 + hour
    time_index = (day_of_year - 1) * 24 + hour_of_day

    return weather_year, time_index


def build_hydro_profiles():
    """Main function to build and save hydroelectric generation profiles."""

    try:
        # Query monthly capacity factors
        monthly_cf = query_pudl_monthly_capacity_factors()

        if len(monthly_cf) == 0:
            logger.warning(
                "No PUDL data matched. Ensure S3 access and parquet file exists."
            )
            return

        # Get unique regions and hydro types
        regions = monthly_cf["region"].unique()
        hydro_types = ["run_of_river", "conventional"]

        # Ensure output directory exists
        output_dir = Path("existing_resource_groups")
        output_dir.mkdir(exist_ok=True)

        # Build profiles for each hydro type
        for hydro_type in hydro_types:
            logger.info(f"\nBuilding {hydro_type} hydroelectric profiles...")

            all_hourly_data = []

            for region in tqdm(regions, desc=f"Processing {hydro_type} regions"):
                hourly_data = interpolate_monthly_to_hourly(
                    monthly_cf, region, hydro_type
                )

                if hourly_data is None or len(hourly_data) == 0:
                    continue

                # Extract weather_year and time_index
                weather_year, time_index = extract_weather_years_and_time_indices(
                    hourly_data
                )

                hourly_data["site_id"] = region
                hourly_data["weather_year"] = weather_year
                hourly_data["time_index"] = time_index
                hourly_data["value"] = hourly_data["capacity_factor"]

                # Select final columns
                hourly_data = hourly_data[
                    ["site_id", "weather_year", "time_index", "value"]
                ]

                all_hourly_data.append(hourly_data)

            if not all_hourly_data:
                logger.warning(f"No data for {hydro_type}. Skipping.")
                continue

            # Combine all regions
            output_df = pd.concat(all_hourly_data, ignore_index=True)

            # Validate: no nulls, time_index 1-8760, groups by site_id/weather_year
            null_count = output_df.isnull().sum().sum()
            if null_count > 0:
                logger.warning(f"{hydro_type}: Found {null_count} null values")

            # Save as parquet
            output_file = output_dir / f"hydro_{hydro_type}_2007_2013.parquet"
            output_df.to_parquet(output_file)

            logger.info(f"\nSaved {hydro_type} profiles to {output_file}")
            logger.info(f"  Rows: {len(output_df)}")
            logger.info(f"  Regions: {output_df['site_id'].nunique()}")
            logger.info(f"  Years: {sorted(output_df['weather_year'].unique())}")
            logger.info(f"  Nulls: {null_count}")
            logger.info(
                f"  Value range: {output_df['value'].min():.3f} - {output_df['value'].max():.3f}"
            )

    except Exception as e:
        logger.error(f"Error building hydro profiles: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    build_hydro_profiles()
