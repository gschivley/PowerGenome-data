"""Offline distributed-PV calendar, timezone, and Parquet regression tests."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCRIPT = Path(__file__).parents[1] / "build_new_pg_dg_inputs.py"
SPEC = importlib.util.spec_from_file_location("build_new_pg_dg_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestDistributedProfiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "profiles.h5"

    def write_profiles(self, years=(2007, 2008, 2013, 2014), tz="Etc/GMT+6"):
        indices = [
            pd.date_range(f"{year}-01-01", periods=8760, freq="h", tz=tz)
            for year in years
        ]
        index = indices[0].append(indices[1:])
        values = np.random.default_rng(42).random((len(index), 2)).astype("float32")
        self.write_h5(index, values)
        return index, values

    def write_h5(self, index, values):
        with h5py.File(self.path, "w") as f:
            f["columns"] = np.array(["p1", "p130"], dtype="S4")
            f["index_0"] = np.array([x.isoformat() for x in index], dtype="S30")
            f["data"] = values

    def test_each_weather_year_rolls_independently_and_keeps_legacy_index(self):
        index, values = self.write_profiles()
        result = MODULE.build_distributed_profiles(self.path)
        self.assertEqual(
            list(result.columns),
            ["time_index", "region", "weather_year", "value", "datetime_utc"],
        )
        self.assertEqual(str(result.datetime_utc.dtype), "datetime64[ns, UTC]")
        self.assertEqual(str(result.weather_year.dtype), "int32")
        self.assertEqual(str(result.value.dtype), "float32")
        self.assertEqual(str(result.region.dtype), "category")
        self.assertEqual(set(result.weather_year), {2007, 2008, 2013})
        for column, region in enumerate(["p1", "p130"]):
            regional = result[result.region == region]
            np.testing.assert_array_equal(regional.time_index, np.arange(3 * 8760))
            for year in (2007, 2008, 2013):
                with self.subTest(region=region, year=year):
                    actual = regional[regional.weather_year == year]
                    source = values[index.year == year, column]
                    np.testing.assert_array_equal(actual.value, np.roll(source, 6))
                    self.assertAlmostEqual(actual.value.sum(), source.sum(), places=2)
                    expected_index = pd.date_range(
                        f"{year}-01-01", periods=8760, freq="h", tz="UTC"
                    )
                    pd.testing.assert_index_equal(
                        pd.DatetimeIndex(actual.datetime_utc),
                        expected_index.rename("datetime_utc"),
                    )
                    self.assertEqual(actual.value.iloc[0], source[-6])
                    self.assertEqual(actual.value.iloc[6], source[0])
        self.assertFalse(result.duplicated(["region", "datetime_utc"]).any())

    def test_leap_year_keeps_feb29_and_omits_dec31(self):
        self.write_profiles(years=(2012,))
        result = MODULE.build_distributed_profiles(self.path)
        times = result.loc[result.region == "p1", "datetime_utc"]
        self.assertEqual(len(times), 8760)
        self.assertEqual(((times.dt.month == 2) & (times.dt.day == 29)).sum(), 24)
        self.assertEqual(((times.dt.month == 12) & (times.dt.day == 31)).sum(), 0)
        self.assertEqual(times.iloc[-1], pd.Timestamp("2012-12-30 23:00", tz="UTC"))

    def test_zero_and_positive_offsets(self):
        for tz, shift in [("UTC", 0), ("Etc/GMT-2", -2)]:
            with self.subTest(tz=tz):
                _, values = self.write_profiles(years=(2010,), tz=tz)
                result = MODULE.build_distributed_profiles(self.path)
                np.testing.assert_array_equal(
                    result.loc[result.region == "p1", "value"],
                    np.roll(values[:, 0], shift),
                )
                self.assertEqual(
                    result.datetime_utc.iloc[0], pd.Timestamp("2010-01-01", tz="UTC")
                )

    def test_rejects_naive_or_fractional_offset(self):
        for tz, message in [(None, "timezone-aware"), ("Asia/Kolkata", "whole-hour")]:
            with self.subTest(tz=tz):
                self.write_profiles(years=(2010,), tz=tz)
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.build_distributed_profiles(self.path)

    def test_rejects_variable_offsets(self):
        index, values = self.write_profiles(years=(2010,))
        labels = [timestamp.isoformat() for timestamp in index]
        labels[-1] = labels[-1].replace("-06:00", "-05:00")
        with h5py.File(self.path, "a") as f:
            del f["index_0"]
            f["index_0"] = np.array(labels, dtype="S30")
        with self.assertRaisesRegex(ValueError, "fixed, whole-hour"):
            MODULE.build_distributed_profiles(self.path)

    def test_rejects_incomplete_duplicate_and_wrong_leap_calendars(self):
        index, values = self.write_profiles(years=(2012,))
        wrong_leap = pd.date_range("2012-01-01", "2012-12-31 23:00", freq="h", tz=index.tz)
        wrong_leap = wrong_leap[~((wrong_leap.month == 2) & (wrong_leap.day == 29))]
        cases = [
            (index[:-1], values[:-1]),
            (index[:1].append(index[:-1]), values),
            (wrong_leap, values),
            (index[::-1], values),
        ]
        for bad_index, bad_values in cases:
            with self.subTest(first=bad_index[0], last=bad_index[-1]):
                self.write_h5(bad_index, bad_values)
                with self.assertRaisesRegex(ValueError, "8760 consecutive hours"):
                    MODULE.build_distributed_profiles(self.path)

    def test_rejects_empty_selection(self):
        self.write_profiles(years=(2014,))
        with self.assertRaisesRegex(ValueError, "No distributed profiles"):
            MODULE.build_distributed_profiles(self.path)

    def test_downloaded_file_removed_on_success_or_read_failure(self):
        for valid in (True, False):
            with self.subTest(valid=valid):
                if valid:
                    self.write_profiles(years=(2010,))
                else:
                    self.path.write_bytes(b"not an HDF5 file")
                with mock.patch.object(MODULE, "_download_to_temp", return_value=str(self.path)):
                    if valid:
                        MODULE.build_distributed_profiles("https://example.test/profiles.h5")
                    else:
                        with self.assertRaises(OSError):
                            MODULE.build_distributed_profiles("https://example.test/profiles.h5")
                self.assertFalse(self.path.exists())

    def test_main_writes_timezone_aware_zstd3_parquet(self):
        self.write_profiles(years=(2010,))
        profiles = MODULE.build_distributed_profiles(self.path)
        capacity = pd.DataFrame({"region": ["p1"], "capacity_mw": [1.0], "year": [2020]})
        original_write = pd.DataFrame.to_parquet
        writes = []

        def write(frame, path, **kwargs):
            writes.append(kwargs)
            original_write(frame, Path(self.tmp.name) / path, **kwargs)

        with (
            mock.patch.object(MODULE, "build_distributed_capacity", return_value=capacity),
            mock.patch.object(MODULE, "build_distributed_profiles", return_value=profiles) as build,
            mock.patch.object(pd.DataFrame, "to_parquet", new=write),
            mock.patch("sys.argv", [str(SCRIPT), "--profiles-path", str(self.path)]),
        ):
            MODULE.main()
        build.assert_called_once_with(generation_profiles_path=str(self.path))
        self.assertEqual(len(writes), 2)
        for kwargs in writes:
            self.assertEqual(kwargs["compression"], "zstd")
            self.assertEqual(kwargs["compression_level"], 3)
        output = Path(self.tmp.name) / "distributed_profiles.parquet"
        pd.testing.assert_frame_equal(pd.read_parquet(output), profiles)
        parquet = pq.ParquetFile(output)
        self.assertEqual(parquet.schema_arrow.field("datetime_utc").type.tz, "UTC")
        for group in range(parquet.metadata.num_row_groups):
            for column in range(parquet.metadata.num_columns):
                self.assertEqual(parquet.metadata.row_group(group).column(column).compression, "ZSTD")


if __name__ == "__main__":
    unittest.main()
