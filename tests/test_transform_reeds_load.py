"""UTC calendar, annual shift, and lossless Parquet migration regressions."""

import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from transform_reeds_load import (
    DIAGNOSTIC_REGIONS,
    diagnose_load_timing,
    migrate_existing_load,
    transform_to_tidy_format,
)


def source_frame(weather_year=2012, offset="Etc/GMT+6"):
    return pd.DataFrame({
        "year": 2025,
        "weather_year": weather_year,
        "datetime": pd.date_range(f"{weather_year}-01-01", periods=8760, freq="h", tz=offset),
        "p1": np.arange(8760, dtype=float),
        "p2": np.arange(8760, dtype=float) + 10000,
    })


@pytest.mark.parametrize("weather_year,last_day", [(2011, "2011-12-31"), (2012, "2012-12-30")])
def test_utc_shift_and_calendar(weather_year, last_day):
    result = transform_to_tidy_format(source_frame(weather_year))
    for region, group in result.groupby("region"):
        original = source_frame(weather_year)[region].to_numpy()
        np.testing.assert_array_equal(group.load_mw, np.roll(original, 6))
        assert group.load_mw.sum() == original.sum()
        assert group.datetime_utc.iloc[0] == pd.Timestamp(f"{weather_year}-01-01", tz="UTC")
        assert group.datetime_utc.iloc[-1] == pd.Timestamp(f"{last_day} 23:00", tz="UTC")
        assert str(group.datetime_utc.dt.tz) == "UTC"
        assert group.time_index.tolist() == list(range(1, 8761))


@pytest.mark.parametrize("offset", [None, "UTC", "America/Chicago"])
def test_rejects_unknown_or_wrong_source_timezone(offset):
    with pytest.raises(ValueError, match="UTC-06:00"):
        transform_to_tidy_format(source_frame(offset=offset))


def test_rejects_incomplete_or_duplicate_hours():
    frame = source_frame()
    with pytest.raises(ValueError, match="hourly calendar"):
        transform_to_tidy_format(frame.iloc[:-1])
    frame.loc[1, "datetime"] = frame.loc[0, "datetime"]
    with pytest.raises(ValueError, match="hourly calendar"):
        transform_to_tidy_format(frame)


def test_migration_preserves_groups_and_roundtrips(tmp_path):
    expected = transform_to_tidy_format(pd.concat([
        source_frame(2011), source_frame(2012), source_frame(2012).assign(year=2030),
    ]))
    legacy = expected.drop(columns="datetime_utc").copy()
    for _, group in legacy.groupby(["year", "region", "weather_year"]):
        legacy.loc[group.index, "load_mw"] = np.roll(group.load_mw.to_numpy(), -6)
    path = tmp_path / "load.parquet"
    legacy.to_parquet(path, index=False, row_group_size=5000)
    migrate_existing_load(path, batch_size=5000)
    actual = pd.read_parquet(path)
    pd.testing.assert_frame_equal(actual, expected)
    f = pq.ParquetFile(path)
    assert f.schema_arrow.field("datetime_utc").type.tz == "UTC"
    assert f.schema_arrow.metadata[b"source_timezone"] == b"UTC-06:00"
    assert all(f.metadata.row_group(g).column(c).compression == "ZSTD"
               for g in range(f.num_row_groups) for c in range(len(f.schema_arrow)))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="twice"):
        migrate_existing_load(path)
    assert path.read_bytes() == before


def test_invalid_migration_does_not_replace_original(tmp_path):
    legacy = transform_to_tidy_format(source_frame()).drop(columns="datetime_utc").iloc[:-1]
    path = tmp_path / "load.parquet"
    legacy.to_parquet(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="hourly grid"):
        migrate_existing_load(path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def diagnostic_frame(shift=0):
    timestamps = pd.date_range("2012-01-01", periods=8760, freq="h", tz="UTC")
    frames = []
    for region, (_, offset) in DIAGNOSTIC_REGIONS.items():
        local_hour = (timestamps.hour + offset) % 24
        values = 100 + 30 * np.cos(2 * np.pi * (local_hour - 16) / 24)
        frames.append(pd.DataFrame({
            "time_index": np.arange(1, 8761), "datetime_utc": timestamps,
            "weather_year": 2012, "year": 2025, "scenario": "test",
            "region": region, "load_mw": np.roll(values, shift),
        }))
    return pd.concat(frames, ignore_index=True)


def test_diagnostic_accepts_plausible_shapes_and_preserves_file(tmp_path):
    path = tmp_path / "load.parquet"
    diagnostic_frame().assign(year=2030, weather_year=2012).to_parquet(path, index=False)
    before = path.read_bytes()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = diagnose_load_timing(path)
    assert not caught
    assert len(summary) == 8
    assert summary.status.eq("ok").all()
    assert summary.peak_standard_hour.eq(16).all()
    assert summary.minimum_standard_hour.eq(4).all()
    assert summary.year.eq(2030).all()
    assert path.read_bytes() == before


def test_diagnostic_keeps_scenarios_separate(tmp_path):
    frame = pd.concat([
        diagnostic_frame().assign(scenario="aligned"),
        diagnostic_frame(-6).assign(scenario="shifted"),
    ])
    path = tmp_path / "load.parquet"
    frame.to_parquet(path, index=False)
    with pytest.warns(UserWarning, match="scenario shifted"):
        summary = diagnose_load_timing(path)
    assert summary.loc[summary.scenario == "aligned", "status"].eq("ok").all()
    assert summary.loc[summary.scenario == "shifted", "status"].eq("warning").all()


@pytest.mark.parametrize("shift", [-6, 6])
def test_diagnostic_warns_for_shifted_shapes(tmp_path, shift):
    path = tmp_path / "load.parquet"
    diagnostic_frame(shift).to_parquet(path, index=False)
    with pytest.warns(UserWarning, match="outside"):
        summary = diagnose_load_timing(path)
    assert summary.status.eq("warning").all()


@pytest.mark.parametrize("problem,match", [
    ("missing", "missing representative BA"),
    ("grid", "annual UTC hourly grid"),
    ("flat", "near-flat"),
    ("negative", "nonfinite or negative"),
    ("null", "nonfinite or negative"),
    ("naive", "timezone-aware UTC"),
])
def test_diagnostic_warns_for_bad_data(tmp_path, problem, match):
    frame = diagnostic_frame()
    if problem == "missing":
        frame = frame.loc[frame.region != "p10"]
    elif problem == "grid":
        frame = frame.drop(index=0)
    elif problem == "flat":
        frame["load_mw"] = 100.0
    elif problem == "negative":
        frame.loc[0, "load_mw"] = -1
    elif problem == "null":
        frame.loc[0, "load_mw"] = np.nan
    elif problem == "naive":
        frame["datetime_utc"] = frame.datetime_utc.dt.tz_localize(None)
    path = tmp_path / "load.parquet"
    frame.to_parquet(path, index=False)
    with pytest.warns(UserWarning, match=match):
        diagnose_load_timing(path)
