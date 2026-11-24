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
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "build_hydro_profiles.py"
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
                self.assertEqual(
                    MODULE.hours_in_month(pd.Timestamp(date_str)), expected
                )

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

    def test_unreported_generation_is_not_zero_output(self):
        # Plant 2 has capacity but no reported generation (NULL). Counting it
        # as zero output would halve the capacity factor.
        pudl = pd.DataFrame(
            {
                "plant_id_eia": [1, 2],
                "report_date": pd.to_datetime(["2012-06-01", "2012-06-01"]),
                "prime_mover_code": ["HY", "HY"],
                "net_generation_mwh": [36000.0, None],
                "capacity_mw": [100.0, 100.0],
            }
        )
        region_map = pd.DataFrame(
            {
                "plant_id": [1, 2],
                "region": ["p3", "p3"],
                "hydro_type": ["conventional"] * 2,
            }
        )
        with mock.patch.object(
            MODULE, "merge_plant_region_hydro_data", return_value=region_map
        ):
            cf = MODULE.query_pudl_monthly_capacity_factors(
                parquet_url=TestHoursInMonth._write(pudl), s3_settings=[]
            )
        self.assertAlmostEqual(cf["capacity_factor"].iloc[0], 36000.0 / (100 * 720))
        self.assertEqual(cf["capacity_mw"].iloc[0], 100.0)


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


class TestCompleteWeatherYearMonths(unittest.TestCase):
    """Every site gets all 84 weather-year months, even without 2007-13 data."""

    WEATHER = pd.date_range("2007-01-01", "2013-12-01", freq="MS")
    LATER = pd.date_range("2014-01-01", "2025-12-01", freq="MS")

    @staticmethod
    def _cf(region, dates, values, capacity=100.0, hydro_type="run_of_river"):
        return pd.DataFrame(
            {
                "region": region,
                "hydro_type": hydro_type,
                "report_date": dates,
                "capacity_factor": values,
                "capacity_mw": capacity,
            }
        )

    def _sites(self, *regions):
        return pd.DataFrame({"region": list(regions), "hydro_type": "run_of_river"})

    def test_reported_gaps_filled_with_calendar_month_mean(self):
        values = [0.1 * d.month / 12 + 0.01 * (d.year - 2007) for d in self.WEATHER]
        cf = self._cf("p1", self.WEATHER, values)
        cf = cf.loc[cf["report_date"] != "2010-03-01"]
        out, methods = MODULE.complete_weather_year_months(cf, self._sites("p1"))
        self.assertEqual(len(out), 84)
        self.assertEqual(methods["method"].tolist(), ["reported"])
        march = cf.loc[cf["report_date"].dt.month == 3, "capacity_factor"].mean()
        filled = out.loc[out["report_date"] == "2010-03-01", "capacity_factor"].iloc[0]
        self.assertAlmostEqual(filled, march)

    def test_late_reporting_site_uses_later_years(self):
        # Like p108: one zero month per weather year, real data from 2016 on.
        jan = self.WEATHER[self.WEATHER.month == 1][3:]
        later = self.LATER[self.LATER.year >= 2016]
        cf = pd.concat(
            [
                self._cf("p108", jan, 0.0),
                self._cf("p108", later, [0.3 + 0.02 * d.month for d in later]),
            ]
        )
        out, methods = MODULE.complete_weather_year_months(cf, self._sites("p108"))
        self.assertEqual(methods["method"].tolist(), ["later_years"])
        self.assertEqual(len(out), 84)
        july = out.loc[out["report_date"].dt.month == 7, "capacity_factor"]
        self.assertTrue((july - 0.44).abs().lt(1e-9).all())

    def test_site_without_data_uses_capacity_weighted_state_donor(self):
        cf = pd.concat(
            [
                self._cf("p1", self.WEATHER, 0.2, capacity=100.0),
                self._cf("p2", self.WEATHER, 0.5, capacity=300.0),
                self._cf("p9", self.WEATHER, 0.9, capacity=100.0),
            ]
        )
        state = {"p1": "WA", "p2": "WA", "p3": "WA", "p9": "TX"}
        out, methods = MODULE.complete_weather_year_months(cf, self._sites("p3"), state)
        self.assertEqual(methods["method"].tolist(), ["donor_state"])
        self.assertTrue((out["capacity_factor"] - 0.425).abs().lt(1e-9).all())

    def test_site_without_data_or_state_donor_uses_national(self):
        cf = self._cf("p9", self.WEATHER, 0.9)
        out, methods = MODULE.complete_weather_year_months(
            cf, self._sites("p3"), {"p3": "WA", "p9": "TX"}
        )
        self.assertEqual(methods["method"].tolist(), ["donor_national"])
        self.assertTrue((out["capacity_factor"] - 0.9).abs().lt(1e-9).all())


class TestCheckProfileCoverage(unittest.TestCase):
    @staticmethod
    def _profile(site, years, value=0.5, hours=8760):
        return pd.concat(
            pd.DataFrame(
                {
                    "site_id": site,
                    "weather_year": y,
                    "time_index": range(1, hours + 1),
                    "value": value,
                }
            )
            for y in years
        )

    def test_complete_profile_passes(self):
        MODULE.check_profile_coverage(self._profile("p1", range(2007, 2014)), ["p1"])

    def test_missing_years_short_years_zero_and_missing_sites_fail(self):
        cases = {
            "missing years": (self._profile("p32", [2012, 2013]), "p32/2007"),
            "short years": (
                self._profile("p108", range(2007, 2014), hours=744),
                "p108/2007: 744",
            ),
            "all zero": (
                self._profile("p1", range(2007, 2014), value=0.0),
                "all-zero",
            ),
            "missing site": (self._profile("p1", range(2007, 2014)), "p2: no profile"),
        }
        for name, (profile, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                MODULE.check_profile_coverage(
                    profile,
                    ["p1", "p2"]
                    if name == "missing site"
                    else profile["site_id"].unique(),
                )


class TestBuildHydroMetadata(unittest.TestCase):
    """Conventional and run-of-river groups need separate metadata.

    A shared file made the conventional group request run-of-river-only sites
    (e.g. p133), which PowerGenome reports as missing profile coverage.
    """

    def test_capacity_split_by_hydro_type(self):
        generators = pd.DataFrame(
            {
                "plant_id": [1, 1, 2, 3, 4, 5],
                "technology": [
                    "Conventional Hydroelectric",
                    "Conventional Hydroelectric",
                    "Run of River Hydroelectric",
                    "Run of River Hydroelectric",
                    "Hydroelectric Pumped Storage",
                    "Conventional Hydroelectric",
                ],
                "capacity_mw": [100.0, 50.0, 20.0, 5.0, 900.0, 0.0],
            }
        )
        plant_region = pd.DataFrame(
            {"plant_id": [1, 2, 3, 4, 5], "region": ["p10", "p10", "p9", "p10", "p2"]}
        )
        meta = MODULE.build_hydro_metadata(generators, plant_region)

        conv = meta["conventional"]
        self.assertEqual(conv["region"].tolist(), ["p10"])
        self.assertEqual(conv["capacity_mw"].tolist(), [150.0])
        self.assertEqual(conv["id"].tolist(), conv["region"].tolist())

        ror = meta["run_of_river"]
        self.assertEqual(ror["region"].tolist(), ["p9", "p10"])
        self.assertEqual(ror["capacity_mw"].tolist(), [5.0, 20.0])

    def test_unmapped_plant_raises(self):
        generators = pd.DataFrame(
            {
                "plant_id": [1],
                "technology": ["Run of River Hydroelectric"],
                "capacity_mw": [10.0],
            }
        )
        plant_region = pd.DataFrame({"plant_id": [2], "region": ["p1"]})
        with self.assertRaisesRegex(ValueError, "plant_region_map"):
            MODULE.build_hydro_metadata(generators, plant_region)

    def test_committed_metadata_matches_inputs(self):
        root = Path(__file__).parents[1]
        generators = pd.read_csv(root / "data" / "reeds_generators_transformed.csv")
        plant_region = pd.read_csv(root / "data" / "plant_region_map.csv")
        meta = MODULE.build_hydro_metadata(generators, plant_region)
        for hydro_type, expected in meta.items():
            name = MODULE.METADATA_FILENAME.format(hydro_type=hydro_type)
            for folder in [
                "existing_resource_groups",
                "WECC/resource_groups/existing_resource_groups",
                "EI_PJM/resource_groups/existing_resource_groups",
            ]:
                with self.subTest(folder=folder, hydro_type=hydro_type):
                    committed = pd.read_csv(root / folder / name)
                    pd.testing.assert_frame_equal(committed, expected)


if __name__ == "__main__":
    unittest.main()
