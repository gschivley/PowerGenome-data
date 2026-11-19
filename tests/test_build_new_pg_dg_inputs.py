"""Offline regression tests using the ReEDS county HDF5 structure."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[1] / "build_new_pg_dg_inputs.py"
SPEC = importlib.util.spec_from_file_location("build_new_pg_dg_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def county_inputs(tmp_path):
    inputs = {
        "generation_profiles_path": tmp_path / "cf_distpv_county.h5",
        "county_capacity_path": tmp_path / "capacity.csv",
        "county_population_path": tmp_path / "population.csv",
        "mapping_path": tmp_path / "county2zone.csv",
    }
    inputs["county_capacity_path"].write_text("r,2036\np01001,1\np01003,3\np02001,2\n")
    inputs["county_population_path"].write_text("FIPS,value\np01001,7\n01003,1\n2001,5\n")
    inputs["mapping_path"].write_text("FIPS,r\n01001,p1\n1003,p1\n02001,p2\n")
    return inputs


def write_county_h5(path, labels):
    values = np.random.default_rng(42).random((len(labels), 3), dtype=np.float32)
    with h5py.File(path, "w") as source:
        source["index_0"] = np.asarray(labels, dtype="S40")
        source["columns"] = np.asarray(["p01001", "p01003", "p02001"], dtype="S6")
        source["data"] = values
    return values


@pytest.mark.parametrize("weight_by, weights", [("capacity", (1, 3)), ("population", (7, 1))])
def test_county_profiles_roll_within_year_and_preserve_index(
    tmp_path, county_inputs, weight_by, weights
):
    years = (2007, 2008, 2012, 2016)
    dates = [
        pd.date_range(f"{year}-01-01", periods=8760, freq="h", tz="Etc/GMT+6")
        for year in years
    ]
    values = write_county_h5(
        county_inputs["generation_profiles_path"], dates[0].append(dates[1:]).map(str)
    )
    result = MODULE.build_distributed_profiles(**county_inputs, weight_by=weight_by)
    parquet = tmp_path / "profiles.parquet"
    result.to_parquet(parquet, index=False)
    actual = pd.read_parquet(parquet)
    assert list(actual.columns) == ["time_index", "region", "weather_year", "value", "datetime_utc"]
    assert actual.dtypes.astype(str).tolist() == ["int64", "category", "int32", "float32", "datetime64[ns, UTC]"]
    assert actual.region.drop_duplicates().tolist() == ["p1", "p2"]
    expected_regions = {
        "p1": np.average(values[:, :2], weights=weights, axis=1).astype("float32"),
        "p2": values[:, 2],
    }
    for region, unshifted in expected_regions.items():
        regional = actual[actual.region == region]
        assert regional.weather_year.unique().tolist() == list(years[:3])
        np.testing.assert_array_equal(regional.time_index, np.arange(3 * 8760))
        assert regional.datetime_utc.is_unique
        for i, year in enumerate(years[:3]):
            block = regional[regional.weather_year == year]
            source_year = unshifted[i * 8760:(i + 1) * 8760]
            np.testing.assert_array_equal(block.value.iloc[:6], source_year[-6:])
            np.testing.assert_array_equal(block.value.iloc[6:], source_year[:-6])
            assert block.value.to_numpy().sum(dtype=np.float64) == source_year.sum(dtype=np.float64)
            expected_end = f"{year}-12-{'31' if year == 2007 else '30'} 23:00"
            expected_labels = pd.date_range(f"{year}-01-01", expected_end, freq="h", tz="UTC")
            pd.testing.assert_index_equal(pd.DatetimeIndex(block.datetime_utc), expected_labels.rename("datetime_utc"))


@pytest.mark.parametrize("offset", ["", "+00:00", "-05:00", "-06:30"])
def test_rejects_any_nonstandard_source_offset(county_inputs, offset):
    dates = pd.date_range("2008-01-01", periods=8760, freq="h").map(str).to_list()
    dates = [label + "-06:00" for label in dates]
    dates[100] = dates[100][:-6] + offset
    write_county_h5(county_inputs["generation_profiles_path"], dates)
    with pytest.raises(ValueError, match="fixed UTC−06:00"):
        MODULE.build_distributed_profiles(**county_inputs)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "out of order", "feb29 omitted"])
def test_rejects_wrong_source_calendar(county_inputs, problem):
    dates = pd.date_range("2008-01-01", periods=8760, freq="h", tz="Etc/GMT+6")
    if problem == "missing":
        dates = dates.delete(100)
    elif problem == "duplicate":
        dates = dates.delete(100).insert(100, dates[99])
    elif problem == "out of order":
        dates = dates.delete(100).insert(101, dates[100])
    else:
        dates = pd.date_range("2008-01-01", "2008-12-31 23:00", freq="h", tz="Etc/GMT+6")
        dates = dates[~((dates.month == 2) & (dates.day == 29))]
    write_county_h5(county_inputs["generation_profiles_path"], dates.map(str))
    with pytest.raises(ValueError, match="8,760-hour ReEDS calendar|chronological order"):
        MODULE.build_distributed_profiles(**county_inputs)
