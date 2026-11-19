"""Offline regression tests for build_hydro_profiles.py.

Covers the two bugs behind the "missing December" report on the
existing-renewables Zenodo release (2026.08.31):

1. A broken hours-in-month divisor made every December capacity factor
   negative, so the 0-1 clip zeroed all hydro output for December.
2. Summing plant-level PUDL net generation without restricting to
   prime_mover_code = 'HY' pulled in pumped storage (negative generation)
   and co-located thermal units, corrupting regional profiles.

The PUDL query test runs the same DuckDB SQL against a synthetic local
parquet, so no network access is required.
"""

import importlib.util
import json
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_hydro_profiles.py"
SPEC = importlib.util.spec_from_file_location("build_hydro_profiles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

PUDL_COLUMNS = [
    "plant_id_eia",
    "report_date",
    "prime_mover_code",
    "net_generation_mwh",
    "capacity_mw",
]


def test_hydro_uses_only_z134_plant_regions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "plant_region_map.csv").write_text(
        "plant_id,region,zone_set\n1,WA_W,z90\n1,p1,z134\n"
    )
    (data / "reeds_generators_transformed.csv").write_text(
        "plant_id,technology\n1,Conventional Hydroelectric\n"
    )

    assert MODULE.merge_plant_region_hydro_data().to_dict("records") == [
        {"plant_id": 1, "region": "p1", "hydro_type": "conventional"}
    ]


def test_build_z90_aggregates_plants_and_writes_matching_resource_groups(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "plant_region_map.csv").write_text(
        "plant_id,region,zone_set\n"
        "1,p1,z134\n2,p2,z134\n3,p2,z134\n"
        "1,WA_W,z90\n2,WA_W,z90\n3,WA_W,z90\n"
    )
    (data / "reeds_generators_transformed.csv").write_text(
        "plant_id,technology,capacity_mw\n"
        "1,Conventional Hydroelectric,15\n1,Conventional Hydroelectric,85\n"
        "2,Conventional Hydroelectric,50\n3,Run of River Hydroelectric,200\n"
        "3,Hydroelectric Pumped Storage,800\n"
    )
    source = tmp_path / "monthly.parquet"
    pd.DataFrame(
        {
            "plant_id_eia": [1, 2, 3],
            "report_date": pd.to_datetime(["2007-01-01"] * 3),
            "prime_mover_code": ["HY"] * 3,
            "net_generation_mwh": [7440, 29760, 14880],
            "capacity_mw": [25, 75, 25],
        }
    ).to_parquet(source)
    query = MODULE.query_pudl_monthly_capacity_factors
    monkeypatch.setattr(
        MODULE,
        "query_pudl_monthly_capacity_factors",
        lambda *, zone_set: query(str(source), [], zone_set=zone_set),
    )

    MODULE.build_hydro_profiles()
    legacy = tmp_path / "existing_resource_groups"
    original = {path.name: path.read_bytes() for path in legacy.iterdir()}
    z134 = pd.read_parquet(legacy / "hydro_conventional_2007_2013.parquet")
    assert set(z134["site_id"]) == {"p1", "p2"}

    MODULE.build_hydro_profiles("z90")
    assert {path.name: path.read_bytes() for path in legacy.iterdir()} == original
    z90 = tmp_path / "existing_resource_groups_z90"
    cases = [
        ("regional_hydro_True.json", False, 150, 0.5),
        ("regional_small_hydro_True.json", True, 200, 0.8),
    ]
    for filename, small, capacity, cf in cases:
        descriptor = json.loads((z90 / filename).read_text())
        assert descriptor["small"] is small
        profiles = pd.read_parquet(z90 / descriptor["profiles"])
        assert set(profiles["site_id"]) == {"WA_W"}
        assert profiles["value"].sub(cf).abs().max() < 1e-12
        metadata = pd.read_csv(z90 / descriptor["metadata"])
        assert metadata.to_dict("records") == [
            {"region": "WA_W", "id": "WA_W", "capacity_mw": capacity}
        ]


class TestHoursInMonth(unittest.TestCase):
    """December (and every month) must get a positive calendar-hour divisor."""

    def test_true_calendar_hours(self):
        cases = {
            "2012-12-01": 744,  # the regression: old code produced -8040
            "2013-12-01": 744,
            "2007-11-01": 720,
            "2012-01-01": 744,
            "2012-02-01": 696,  # leap February
            "2013-02-01": 672,
            "2013-07-01": 744,
        }
        for date_str, expected in cases.items():
            with self.subTest(date_str):
                self.assertEqual(MODULE.hours_in_month(pd.Timestamp(date_str)), expected)

    def test_december_cf_is_not_zeroed(self):
        pudl = pd.DataFrame(
            {
                "plant_id_eia": [1, 1],
                "report_date": pd.to_datetime(["2012-11-01", "2012-12-01"]),
                "prime_mover_code": ["HY", "HY"],
                "net_generation_mwh": [36000.0, 40000.0],
                "capacity_mw": [100.0, 100.0],
            }
        )
        region_map = pd.DataFrame(
            {"plant_id": [1], "region": ["p3"], "hydro_type": ["conventional"]}
        )
        with mock.patch.object(
            MODULE, "merge_plant_region_hydro_data", return_value=region_map
        ):
            cf = MODULE.query_pudl_monthly_capacity_factors(
                parquet_url=self._write(pudl), s3_settings=[]
            )
        dec = cf[cf["report_date"] == "2012-12-01"]["capacity_factor"].iloc[0]
        self.assertAlmostEqual(dec, 40000.0 / (100 * 744))
        self.assertGreater(dec, 0.0)
        nov = cf[cf["report_date"] == "2012-11-01"]["capacity_factor"].iloc[0]
        self.assertAlmostEqual(nov, 36000.0 / (100 * 720))

    @staticmethod
    def _write(pudl: pd.DataFrame) -> str:
        import tempfile

        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "monthly.parquet"
        pudl.to_parquet(path)
        return str(path)


class TestPrimeMoverFilter(unittest.TestCase):
    """Only HY generators may contribute to hydro capacity factors."""

    def test_excludes_pumped_storage_and_thermal(self):
        pudl = pd.DataFrame(
            {
                "plant_id_eia": [1, 1, 1, 2],
                "report_date": pd.to_datetime(
                    ["2012-06-01", "2012-06-01", "2012-06-01", "2012-06-01"]
                ),
                # Same plant as the HY unit: pumped storage reports negative
                # net generation (it is a load); thermal inflates output.
                "prime_mover_code": ["HY", "PS", "ST", "HY"],
                "net_generation_mwh": [30000.0, -20000.0, 500000.0, 60000.0],
                "capacity_mw": [100.0, 100.0, 1900.0, 100.0],
            }
        )
        region_map = pd.DataFrame(
            {
                "plant_id": [1, 2],
                "region": ["p3", "p3"],
                "hydro_type": ["conventional", "conventional"],
            }
        )
        with mock.patch.object(
            MODULE, "merge_plant_region_hydro_data", return_value=region_map
        ):
            cf = MODULE.query_pudl_monthly_capacity_factors(
                parquet_url=TestHoursInMonth._write(pudl), s3_settings=[]
            )
        expected = (30000.0 + 60000.0) / ((100.0 + 100.0) * 720)
        self.assertAlmostEqual(cf["capacity_factor"].iloc[0], expected)
        self.assertEqual(len(cf), 1)


class TestInterpolateMonthlyToHourly(unittest.TestCase):
    def _monthly(self, rows):
        return pd.DataFrame(
            {
                "region": "p3",
                "hydro_type": "conventional",
                "report_date": pd.to_datetime([d for d, _ in rows]),
                "capacity_factor": [v for _, v in rows],
            }
        )

    def test_december_is_not_zero_after_smoothing(self):
        months = pd.date_range("2007-01-01", "2007-12-01", freq="MS")
        rows = [(m, 0.1) for m in months]
        rows[-1] = ("2007-12-01", 0.4)
        hourly = MODULE.interpolate_monthly_to_hourly(
            self._monthly(rows), "p3", "conventional"
        )
        dec = hourly[hourly["report_date"].dt.month == 12]["capacity_factor"]
        self.assertGreater(dec.min(), 0.0)
        # Raw December is flat 0.4; the 168h centered window drags the very
        # start of December toward November's 0.1 (floor ~0.25), the rest of
        # the month stays near 0.4.
        self.assertGreater(dec.min(), 0.2)
        self.assertGreater(dec.mean(), 0.35)
        self.assertLess(dec.mean(), 0.4 + 1e-9)

    def test_leap_year_keeps_8760_hours_dropping_dec31(self):
        months = pd.date_range("2012-01-01", "2012-12-01", freq="MS")
        hourly = MODULE.interpolate_monthly_to_hourly(
            self._monthly([(m, 0.5) for m in months]), "p3", "conventional"
        )
        self.assertEqual(len(hourly), 8760)
        self.assertEqual(hourly["report_date"].min(), pd.Timestamp("2012-01-01"))
        self.assertEqual(hourly["report_date"].max(), pd.Timestamp("2012-12-30 23:00"))
        _, time_index = MODULE.extract_weather_years_and_time_indices(hourly)
        self.assertEqual(sorted(time_index.unique().tolist()), list(range(1, 8761)))

    def test_missing_month_warns(self):
        rows = [
            ("2007-01-01", 0.2),
            ("2007-02-01", 0.2),
            ("2007-04-01", 0.2),
        ]
        with self.assertLogs(MODULE.logger, level="WARNING") as ctx:
            MODULE.interpolate_monthly_to_hourly(
                self._monthly(rows), "p3", "conventional"
            )
        self.assertTrue(any("2007-03" in msg for msg in ctx.output))


if __name__ == "__main__":
    unittest.main()
