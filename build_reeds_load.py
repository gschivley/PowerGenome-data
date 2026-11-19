#!/usr/bin/env python3
"""Build PowerGenome hourly load for z134 and z90 from shared ReEDS inputs."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from locked_sources import SourceLock, add_source_options

LOAD_SOURCE = "demand_EER2025_IRAlow.h5"
STATE_SOURCE = "zones/county_state.csv"
POPULATION_SOURCE = "disaggregation/county_population.csv"
SCENARIO = "EER2025_IRAlow"
MODEL_YEARS = (2021, 2025, 2030, 2035, 2040, 2045, 2050)
WEATHER_YEARS = tuple(range(2007, 2014))
HOURS_PER_YEAR = 8760
SOURCE_TIMEZONE = "Etc/GMT+6"  # Fixed UTC-06:00; no daylight saving time.
UTC_SHIFT_HOURS = 6
ZONE_SETS = {"z134": 134, "z90": 90}
LOAD_SCHEMA = pa.schema([
    ("time_index", pa.int64()), ("weather_year", pa.int32()), ("region", pa.string()),
    ("load_mw", pa.float64()), ("year", pa.int32()), ("scenario", pa.string()),
    ("zone_set", pa.string()), ("datetime_utc", pa.timestamp("ns", tz="UTC")),
])


def population_weights(county_zones, county_states, population):
    """Aggregate the mapped counties' share of state population to each model zone."""
    frames = [frame.copy() for frame in (county_zones, county_states, population)]
    for frame in frames:
        frame["FIPS"] = frame["FIPS"].astype(str).str.removeprefix("p").str.zfill(5)
    zones, states, pop = frames
    counties = zones.merge(states[["FIPS", "state"]], on="FIPS", how="left", validate="1:1")
    counties = counties.merge(pop, on="FIPS", how="left", validate="1:1")
    if (
        counties[["r", "state", "value"]].isna().any().any()
        or not np.isfinite(counties["value"]).all()
        or (counties["value"] < 0).any()
    ):
        raise ValueError("Missing or invalid county population, zone or state")
    totals = counties.groupby(["state", "r"])["value"].sum().unstack(fill_value=0)
    weights = totals.div(totals.sum(axis=1), axis=0)
    if not np.allclose(weights.sum(axis=1), 1) or (weights.sum(axis=0) <= 0).any():
        raise ValueError("Invalid state-to-zone weights")
    return weights


def hourly_tables(source, weights, zone_set, years=MODEL_YEARS, weather_years=WEATHER_YEARS):
    hour_slots = np.tile(np.arange(HOURS_PER_YEAR), len(weights.columns))
    with h5py.File(source, "r") as h5:
        for year in years:
            group = h5[str(year)]
            states = sorted(name for name in group if name not in {"datetime", "columns"})
            mismatch = set(states) ^ set(weights.index)
            if mismatch:
                raise ValueError(f"State coverage mismatch: {sorted(mismatch)}")
            dates = pd.DatetimeIndex(pd.to_datetime(group["datetime"][:].astype("U")))
            if dates.tz is None or dates.tz.utcoffset(None) != pd.Timedelta(hours=-UTC_SHIFT_HOURS):
                raise ValueError(f"Expected timezone-aware UTC-06:00 source timestamps for {year}")
            dates = dates.tz_convert(SOURCE_TIMEZONE).as_unit("ns")
            for weather_year in weather_years:
                mask = dates.year == weather_year
                # Both clocks use 8760 interval starts from Jan 1, omitting Dec 31 in leap years.
                calendar = pd.date_range(f"{weather_year}-01-01", periods=HOURS_PER_YEAR, freq="h").as_unit("ns")
                if not dates[mask].equals(calendar.tz_localize(SOURCE_TIMEZONE)):
                    raise ValueError(f"Invalid hourly calendar for {year}/{weather_year}")
                raw = np.column_stack([group[state][:][mask] for state in states])
                # Align values to the UTC calendar, wrapping only within this annual block.
                raw_utc = np.roll(raw, UTC_SHIFT_HOURS, axis=0)
                loads = raw_utc @ weights.loc[states].to_numpy()
                if not np.isfinite(loads).all() or (loads < 0).any():
                    raise ValueError(f"Invalid loads for {year}/{weather_year}")
                np.testing.assert_allclose(loads.sum(axis=1), raw_utc.sum(axis=1), rtol=1e-12)
                frame = pd.DataFrame(
                    {
                        "time_index": hour_slots + 1,
                        "weather_year": weather_year,
                        "region": np.repeat(weights.columns.to_numpy(), HOURS_PER_YEAR),
                        "load_mw": loads.T.ravel(),
                        "year": year,
                        "scenario": SCENARIO,
                        "zone_set": zone_set,
                        "datetime_utc": calendar.tz_localize("UTC").take(hour_slots),
                    }
                )
                yield pa.Table.from_pandas(frame, schema=LOAD_SCHEMA, preserve_index=False)


def build_load(source, weights_by_zone_set, output, **hourly_options):
    with NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with pq.ParquetWriter(temporary, LOAD_SCHEMA, compression="zstd", compression_level=3) as writer:
            for zone_set, weights in weights_by_zone_set.items():
                for table in hourly_tables(source, weights, zone_set, **hourly_options):
                    writer.write_table(table, row_group_size=table.num_rows)
        temporary.chmod(output.stat().st_mode & 0o777 if output.exists() else 0o644)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/reeds_load_transformed.parquet"))
    add_source_options(parser, Path("cache_data/reeds-load"))
    args = parser.parse_args()

    sources = SourceLock(args.cache_dir, args.sources_lock)

    def read(source_id):
        return pd.read_csv(sources.fetch(source_id), dtype={"FIPS": str})

    states = read(STATE_SOURCE)
    population = read(POPULATION_SOURCE)
    weights = {}
    zone_sources = []
    for zone_set, count in ZONE_SETS.items():
        zone_source = f"zones/{zone_set}/county2zone.csv"
        zone_sources.append(zone_source)
        weights[zone_set] = population_weights(read(zone_source), states, population)
        if len(weights[zone_set].columns) != count:
            raise ValueError(f"Expected {count} zones in {zone_set}; got {len(weights[zone_set].columns)}")
    source = sources.fetch(LOAD_SOURCE)
    pins = (LOAD_SOURCE, STATE_SOURCE, POPULATION_SOURCE, *zone_sources)
    with sources.output(args.output, pins) as output:
        build_load(source, weights, output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
