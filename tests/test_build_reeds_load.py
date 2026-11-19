import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).parents[1] / "build_reeds_load.py"
SPEC = importlib.util.spec_from_file_location("build_reeds_load", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_load = MODULE.build_load
population_weights = MODULE.population_weights


def test_builds_both_zone_sets_from_same_source(tmp_path):
    weights = population_weights(
        pd.DataFrame({"FIPS": ["01001", "01003", "02001"], "r": ["east", "west", "west"]}),
        pd.DataFrame({"FIPS": ["01001", "01003", "02001"], "state": ["AA", "AA", "BB"]}),
        pd.DataFrame({"FIPS": ["p01001", "p01003", "p02001"], "value": [1, 3, 2]}),
    )
    weights_by_zone_set = {
        "z134": pd.DataFrame({"p1": [1.0, 0.0], "p2": [0.0, 1.0]}, index=["AA", "BB"]),
        "z90": weights,
    }
    years = (2021, 2025)
    weather_years = (2007, 2008)
    hour = np.arange(8760)
    dates = [
        pd.date_range(f"{weather_year}-01-01", periods=8760, freq="h", tz="Etc/GMT+6")
        for weather_year in weather_years
    ]
    source_values = {}
    source = tmp_path / "load.h5"
    with h5py.File(source, "w") as f:
        for year in years:
            group = f.create_group(str(year))
            group["datetime"] = dates[0].append(dates[1:]).map(str).to_numpy(dtype="S30")
            for weather_year in weather_years:
                # Distinct, nonconstant state profiles in every model/weather-year block.
                base = 20000 * (year - years[0]) + 10000 * (weather_year - weather_years[0])
                source_values[year, weather_year] = (base + 4 * hour, base + 2 * (8760 - hour))
            for state_idx, state in enumerate(("AA", "BB")):
                group[state] = np.concatenate([
                    source_values[year, weather_year][state_idx] for weather_year in weather_years
                ])

    output = tmp_path / "combined.parquet"
    options = {"years": years, "weather_years": weather_years}
    build_load(source, weights_by_zone_set, output, **options)
    original = pq.read_table(output)
    build_load(source, weights_by_zone_set, output, **options)
    assert pq.read_table(output).equals(original)

    with pq.ParquetFile(output) as parquet:
        assert parquet.schema_arrow.field("datetime_utc").type == pa.timestamp("ns", tz="UTC")
        partitions = [
            (zone_set, year, weather_year)
            for zone_set in weights_by_zone_set for year in years for weather_year in weather_years
        ]
        assert parquet.num_row_groups == len(partitions)
        for i, (zone_set, year, weather_year) in enumerate(partitions):
            metadata = parquet.metadata.row_group(i)
            assert metadata.num_rows == 2 * 8760
            assert all(metadata.column(c).compression == "ZSTD" for c in range(metadata.num_columns))
            rows = parquet.read_row_group(i).to_pandas()
            assert set(rows.zone_set) == {zone_set}
            assert set(rows.year) == {year}
            assert set(rows.weather_year) == {weather_year}
            assert set(rows.scenario) == {"EER2025_IRAlow"}
            aa, bb = source_values[year, weather_year]
            expected = {"p1": aa, "p2": bb} if zone_set == "z134" else {
                "east": aa / 4, "west": aa * 3 / 4 + bb,
            }
            assert set(rows.region) == set(expected)
            last_day = {2007: "2007-12-31", 2008: "2008-12-30"}[weather_year]
            utc_calendar = pd.date_range(f"{weather_year}-01-01", f"{last_day} 23:00", freq="h", tz="UTC")
            for region, values in expected.items():
                profile = rows.loc[rows.region == region]
                np.testing.assert_array_equal(profile.time_index, hour + 1)
                pd.testing.assert_index_equal(pd.DatetimeIndex(profile.datetime_utc), utc_calendar, check_names=False)
                # Jan 1 00:00–05:00 comes from this block's tail, not its neighbor.
                np.testing.assert_array_equal(profile.load_mw, np.concatenate([values[-6:], values[:-6]]))
                assert profile.load_mw.sum() == values.sum()


@pytest.mark.parametrize("problem,match", [
    ("duplicate", "Invalid hourly calendar"),
    ("naive", "Expected timezone-aware UTC-06:00"),
    ("utc", "Expected timezone-aware UTC-06:00"),
])
def test_invalid_source_clock_does_not_replace_output(tmp_path, problem, match):
    dates = pd.date_range("2008-01-01", periods=8760, freq="h", tz="Etc/GMT+6")
    source = tmp_path / "load.h5"
    with h5py.File(source, "w") as f:
        group = f.create_group("2021")
        group["datetime"] = dates.map(str).to_numpy(dtype="S30")
        group["AA"] = np.arange(8760)
    output = tmp_path / "load.parquet"
    weights = {"z134": pd.DataFrame({"p1": [1.0]}, index=["AA"])}
    options = {"years": (2021,), "weather_years": (2008,)}
    build_load(source, weights, output, **options)
    original = pq.read_table(output)
    with h5py.File(source, "r+") as f:
        timestamps = f["2021/datetime"]
        if problem == "duplicate":
            timestamps[1] = timestamps[0]
        else:
            # Converting to UTC preserves the instants, but is the wrong source clock.
            wrong_clock = dates.tz_localize(None) if problem == "naive" else dates.tz_convert("UTC")
            timestamps[:] = wrong_clock.map(str).to_numpy(dtype="S30")
    with pytest.raises(ValueError, match=match):
        build_load(source, weights, output, **options)
    assert pq.read_table(output).equals(original)
