"""Offline tests for build_atb_data.py using synthetic fixtures.

The fixtures mirror the structural quirks of the real ATB sources (the
techdetail2 fallback, the 2024+ construction finance columns, the battery
workbook layout) so the builder can be exercised without network access.
"""

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "build_atb_data.py"
SPEC = importlib.util.spec_from_file_location("build_atb_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

TIDY_COLUMNS = [
    "atb_year",
    "core_metric_parameter",
    "core_metric_case",
    "crpyears",
    "technology",
    "techdetail",
    "techdetail2",
    "scenario",
    "core_metric_variable",
    "value",
]

BATTERY_BASIS_YEARS = (2023, 2024, 2025)
ENERGY_COSTS = {  # $/kWh by scenario and basis year
    "Advanced": (100.0, 90.0, 80.0),
    "Moderate": (120.0, 110.0, 100.0),
    "Conservative": (140.0, 130.0, 120.0),
}
POWER_COSTS = {  # $/kW by scenario and basis year
    "Advanced": (200.0, 190.0, 180.0),
    "Moderate": (220.0, 210.0, 200.0),
    "Conservative": (240.0, 230.0, 220.0),
}
EXPAND_ENERGY_COSTS = {  # decoy sheet with different values
    scenario: tuple(value + 1000.0 for value in values)
    for scenario, values in ENERGY_COSTS.items()
}


def tidy_frame(records, atb_year):
    """Build a synthetic tidy table; *records* override the defaults."""
    rows = []
    for record in records:
        row = {
            "atb_year": atb_year,
            "core_metric_parameter": "CAPEX",
            "core_metric_case": "Market",
            "crpyears": 20,
            "technology": "NaturalGas_FE",
            "techdetail": "CC",
            "techdetail2": "",
            "scenario": "Moderate",
            "core_metric_variable": 2025,
            "value": 1000.0,
        }
        row.update(record)
        rows.append(row)
    return pd.DataFrame(rows, columns=TIDY_COLUMNS)


def cost_frame(rows):
    """Build a synthetic committed-schema cost table."""
    records = []
    for row in rows:
        record = {
            "data_year": 2023,
            "basis_year": 2025,
            "cap_recovery_years": 20,
            "cost_case": "Moderate",
            "financial_case": "Market",
            "technology": "NaturalGas",
            "tech_detail": "CC",
            "parameter": "capex_mw",
            "parameter_value": 1_000_000.0,
            "units": np.nan,
            "dollar_year": 2021,
        }
        record.update(row)
        records.append(record)
    return pd.DataFrame(records, columns=MODULE.COST_COLUMNS)


def battery_grid(marker_row=12, include_power=True, include_years=True, scenarios=None):
    """Build a raw battery worksheet grid with markers at *marker_row*.

    The markers sit at a nonstandard row so the tests prove the parser
    searches for labels instead of assuming fixed positions.
    """
    scenarios = scenarios or list(MODULE.BATTERY_SCENARIOS)
    blocks = [(MODULE.BATTERY_ENERGY_MARKER, ENERGY_COSTS)]
    if include_power:
        blocks.append((MODULE.BATTERY_POWER_MARKER, POWER_COSTS))
    grid = pd.DataFrame(index=range(marker_row + 2 + 6 * len(blocks)), columns=range(9), dtype=object)
    for block, (marker, costs) in enumerate(blocks):
        row = marker_row + block * 6
        grid.iat[row, 4] = marker
        if include_years:
            for column, basis_year in enumerate(BATTERY_BASIS_YEARS):
                grid.iat[row + 1, 5 + column] = basis_year
        for offset, (label, canonical) in enumerate(zip(scenarios, MODULE.BATTERY_SCENARIOS)):
            grid.iat[row + 2 + offset, 4] = label
            for column, value in enumerate(costs[canonical]):
                grid.iat[row + 2 + offset, 5 + column] = value
    return grid


def expand_battery_grid(marker_row=12):
    """A decoy battery grid whose values differ from the R&D sheet."""
    grid = battery_grid(marker_row)
    for offset in range(len(MODULE.BATTERY_SCENARIOS)):
        for column in range(3):
            grid.iat[marker_row + 2 + offset, 5 + column] = (
                float(grid.iat[marker_row + 2 + offset, 5 + column]) + 1000.0
            )
    return grid


def write_workbook(path, grid, extra_sheets=("Commercial Battery Storage",)):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        grid.to_excel(writer, sheet_name="Utility-Scale Battery - R&D", header=False, index=False)
        expand_battery_grid().to_excel(
            writer, sheet_name="Utility-Scale Battery - Expand", header=False, index=False
        )
        for sheet in extra_sheets:
            pd.DataFrame().to_excel(writer, sheet_name=sheet, header=False, index=False)
    return path


class FixtureTestCase(unittest.TestCase):
    """Shared synthetic sources for all test classes."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        # 2023: techdetail is empty in the tidy data and filled from techdetail2.
        cls.parquet_2023 = cls.root / "tidy_2023.parquet"
        tidy_frame(
            [
                {"core_metric_parameter": "CAPEX", "value": 1000.0, "techdetail": "", "techdetail2": "CC"},
                {"core_metric_parameter": "CAPEX", "value": 1000.0, "techdetail": "", "techdetail2": "CC"},
                {"core_metric_parameter": "Fixed O&M", "value": 10.0, "techdetail": "", "techdetail2": "CC"},
                {"core_metric_parameter": "WACC Real", "value": 0.05, "techdetail": "", "techdetail2": "CC"},
                {"core_metric_parameter": "Heat Rate", "value": 7.0, "techdetail": "", "techdetail2": "CC"},
                {"technology": "Utility-Scale Battery Storage", "techdetail": "*", "techdetail2": ""},
                {"technology": "AEO", "techdetail": "*", "techdetail2": ""},
            ],
            atb_year=2023,
        ).to_parquet(cls.parquet_2023, index=False)
        # 2024: construction finance columns and prefixed technology details.
        cls.parquet_2024 = cls.root / "tidy_2024.parquet"
        gas = [
            {"core_metric_parameter": parameter, "value": value, "techdetail": "NG 2-on-1 Combined Cycle"}
            for parameter, value in (
                ("CAPEX", 1210.0),
                ("OCC", 1100.0),
                ("GCC", 100.0),
                ("CFC", 10.0),
                ("Fixed O&M", 20.0),
                ("Variable O&M", 30.0),
                ("Heat Rate", 7.0),
                ("WACC Real", 0.05),
            )
        ]
        coal = [
            {"core_metric_parameter": parameter, "value": value, "technology": "Coal_FE", "techdetail": "Coal-new"}
            for parameter, value in (
                ("CAPEX", 2000.0),
                ("OCC", 1800.0),
                ("GCC", 200.0),
                ("CFC", 5.0),
                ("Fixed O&M", 25.0),
                ("Heat Rate", 10.0),
                ("WACC Real", 0.06),
            )
        ]
        tidy_frame(gas + coal, atb_year=2024).to_parquet(cls.parquet_2024, index=False)
        # 2025: financial cases split into R&D / R&D + TC / Exp / Exp + TC.
        cls.parquet_2025 = cls.root / "tidy_2025.parquet"
        cases = []
        for parameter, value in (
            ("CAPEX", 1500.0),
            ("OCC", 1300.0),
            ("GCC", 150.0),
            ("CFC", 8.0),
            ("Fixed O&M", 22.0),
            ("Variable O&M", 32.0),
            ("Heat Rate", 6.5),
            ("WACC Real", 0.04),
        ):
            for case in ("R&D", "R&D + TC", "Exp", "Exp + TC"):
                cases.append(
                    {
                        "core_metric_parameter": parameter,
                        "core_metric_case": case,
                        "value": value,
                        "techdetail": "NG 2-on-1 Combined Cycle",
                    }
                )
        tidy_frame(cases, atb_year=2025).to_parquet(cls.parquet_2025, index=False)
        cls.workbook = write_workbook(cls.root / "workbook.xlsx", battery_grid())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def patch_sources(self, paths):
        def fake_source_paths(year, cache_dir):
            return paths[year]

        return mock.patch.object(MODULE, "source_paths", side_effect=fake_source_paths)

    def write_committed(self, costs, heat_rates=None):
        costs.to_parquet(self.tmp / MODULE.COST_OUTPUT_NAME, index=False)
        if heat_rates is not None:
            heat_rates.to_csv(self.tmp / MODULE.HEAT_RATE_OUTPUT_NAME, index=False)
        return self.tmp


class LoadTidyTests(FixtureTestCase):
    def test_source_year_must_match_requested_year(self):
        with self.assertRaisesRegex(ValueError, "unexpected source years"):
            MODULE.load_tidy(self.parquet_2024, 2025)

    def test_techdetail2_fills_empty_details(self):
        tidy = MODULE.load_tidy(self.parquet_2023, 2023)
        details = tidy.loc[tidy["technology"].eq("NaturalGas_FE"), "techdetail"]
        self.assertTrue((details == "CC").all())

    def test_missing_required_columns_raise(self):
        frame = tidy_frame([{}], atb_year=2023).drop(columns=["value"])
        path = self.tmp / "broken.parquet"
        frame.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            MODULE.load_tidy(path, 2023)


class NormalizeCostsTests(FixtureTestCase):
    def test_2023_normalization_keeps_historical_naming(self):
        settings = MODULE.YEAR_SETTINGS[2023]
        tidy = MODULE.load_tidy(self.parquet_2023, 2023)
        tidy = tidy.loc[tidy["core_metric_parameter"].isin(settings["core_metrics"])]
        costs = MODULE.normalize_costs(tidy, 2023, settings)
        self.assertEqual(list(costs.columns), MODULE.COST_COLUMNS)
        self.assertEqual(set(costs["technology"]), {"NaturalGas"})
        self.assertEqual(set(costs["tech_detail"]), {"CC"})
        self.assertEqual(set(costs["financial_case"]), {"Market"})
        self.assertEqual(set(costs["dollar_year"].unique()), {2021})
        self.assertEqual(set(costs["data_year"].unique()), {2023})
        self.assertTrue(costs["units"].isna().all())
        values = dict(zip(costs["parameter"], costs["parameter_value"]))
        # The duplicate CAPEX row is deduplicated and $/kW becomes $/MW.
        self.assertEqual(values["capex_mw"], 1_000_000.0)
        self.assertEqual(values["fixed_o_m_mw"], 10_000.0)
        self.assertEqual(values["wacc_real"], 0.05)
        self.assertEqual(values["heat_rate"], 7.0)
        self.assertEqual(len(costs), 4)


class ConstructionFinanceTests(FixtureTestCase):
    def test_2024_build_year_recomputes_capex_without_gcc(self):
        costs, heat_rates = MODULE.build_year(self.parquet_2024, self.workbook, 2024)
        gas = costs.loc[costs["technology"].eq("NaturalGas")]
        coal = costs.loc[costs["technology"].eq("Coal")]
        # Prefixed technology details are stripped, matching the 2024 build.
        self.assertEqual(set(gas["tech_detail"]), {"2-on-1 Combined Cycle"})
        self.assertEqual(set(coal["tech_detail"]), {"new"})
        values = dict(zip(gas["parameter"], gas["parameter_value"]))
        cff = 1210.0 / (1100.0 + 100.0)
        self.assertAlmostEqual(values["CFF"], cff)
        self.assertEqual(values["ATB_CAPEX"], 1210.0)
        self.assertAlmostEqual(values["capex_mw"], (1210.0 - 100.0 * cff) * 1000.0)
        self.assertEqual(values["OCC"], 1100.0)
        self.assertEqual(values["GCC"], 100.0)
        self.assertEqual(values["CFC"], 10.0)
        self.assertEqual(values["fixed_o_m_mw"], 20_000.0)
        self.assertEqual(values["variable_o_m_mwh"], 30.0)
        # Heat rates move to their own table from 2024 on.
        self.assertNotIn("heat_rate", set(costs["parameter"]))
        self.assertEqual(set(heat_rates["technology"]), {"NaturalGas", "Coal"})
        self.assertEqual(
            set(zip(heat_rates["technology"], heat_rates["heat_rate"])),
            {("NaturalGas", 7.0), ("Coal", 10.0)},
        )


class BatteryExtractionTests(FixtureTestCase):
    def test_extracts_power_and_energy_costs_with_derived_o_m(self):
        battery = MODULE.extract_battery_costs(self.workbook, 2023, MODULE.YEAR_SETTINGS[2023])
        self.assertEqual(list(battery.columns), MODULE.COST_COLUMNS)
        self.assertEqual(len(battery), 45)  # 3 scenarios x 3 years x 5 parameters
        self.assertEqual(set(battery["technology"]), {"Battery"})
        self.assertEqual(set(battery["tech_detail"]), {"*"})
        self.assertEqual(set(battery["financial_case"]), {"Market"})
        self.assertEqual(set(battery["cap_recovery_years"]), {15})
        self.assertEqual(set(battery["dollar_year"]), {2021})
        self.assertEqual(set(battery["cost_case"]), set(MODULE.BATTERY_SCENARIOS))
        self.assertEqual(set(battery["basis_year"]), set(BATTERY_BASIS_YEARS))
        row = battery.loc[
            (battery["cost_case"] == "Moderate") & (battery["basis_year"] == 2024)
        ]
        values = dict(zip(row["parameter"], row["parameter_value"]))
        # $/kWh and $/kW become $/MWh and $/MW; fixed O&M is 2.5% of capex.
        self.assertEqual(values["capex_mwh"], 110_000.0)
        self.assertEqual(values["capex_mw"], 210_000.0)
        self.assertEqual(values["fixed_o_m_mwh"], 2_750.0)
        self.assertEqual(values["fixed_o_m_mw"], 5_250.0)
        self.assertEqual(values["variable_o_m_mwh"], 0.0)

    def test_prefers_rd_sheet_over_expand_sheet(self):
        # The Expand sheet carries the same layout but different values; the
        # extraction must read the R&D sheet.
        battery = MODULE.extract_battery_costs(self.workbook, 2025, MODULE.YEAR_SETTINGS[2025])
        self.assertEqual(set(battery["technology"]), {"Utility-Scale Battery Storage"})
        self.assertEqual(set(battery["tech_detail"]), {"Lithium Ion"})
        self.assertEqual(set(battery["financial_case"]), {"R&D"})
        self.assertEqual(set(battery["dollar_year"]), {2023})
        moderate = battery.loc[(battery["cost_case"] == "Moderate") & (battery["basis_year"] == 2024)]
        values = dict(zip(moderate["parameter"], moderate["parameter_value"]))
        self.assertEqual(values["capex_mwh"], 110_000.0)

    def test_missing_power_marker_raises(self):
        path = write_workbook(self.tmp / "no_power.xlsx", battery_grid(include_power=False))
        with self.assertRaisesRegex(ValueError, "Battery Power Capital Cost"):
            MODULE.extract_battery_costs(path, 2023, MODULE.YEAR_SETTINGS[2023])

    def test_missing_projection_years_raise(self):
        path = write_workbook(self.tmp / "no_years.xlsx", battery_grid(include_years=False))
        with self.assertRaisesRegex(ValueError, "projection years"):
            MODULE.extract_battery_costs(path, 2023, MODULE.YEAR_SETTINGS[2023])

    def test_unexpected_scenario_label_raises(self):
        path = write_workbook(
            self.tmp / "bad_scenario.xlsx",
            battery_grid(scenarios=["Advanced", "Moderate", "Slow"]),
        )
        with self.assertRaisesRegex(ValueError, "scenarios changed"):
            MODULE.extract_battery_costs(path, 2023, MODULE.YEAR_SETTINGS[2023])

    def test_missing_battery_sheet_raises(self):
        path = self.tmp / "no_battery.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame().to_excel(writer, sheet_name="LandbasedWind", header=False, index=False)
        with self.assertRaisesRegex(ValueError, "battery sheet"):
            MODULE.extract_battery_costs(path, 2023, MODULE.YEAR_SETTINGS[2023])


class BuildYearTests(FixtureTestCase):
    def test_2023_keeps_heat_rate_rows_in_costs(self):
        costs, heat_rates = MODULE.build_year(self.parquet_2023, self.workbook, 2023)
        self.assertIn("heat_rate", set(costs["parameter"]))
        self.assertEqual(len(heat_rates), 1)
        self.assertEqual(list(heat_rates.columns), MODULE.HEAT_RATE_COLUMNS)
        battery = costs.loc[costs["technology"].eq("Battery")]
        self.assertEqual(len(battery), 45)

    def test_2025_keeps_only_rd_financial_case(self):
        costs, _ = MODULE.build_year(self.parquet_2025, self.workbook, 2025)
        self.assertEqual(set(costs["financial_case"]), {"R&D"})
        self.assertEqual(set(costs["dollar_year"]), {2023})


class ValidateOutputsTests(FixtureTestCase):
    def setUp(self):
        super().setUp()
        self.costs = cost_frame(
            [
                {"parameter": "capex_mw"},
                {"parameter": "capex_mw", "technology": "Battery", "tech_detail": "*"},
                {"parameter": "capex_mwh", "technology": "Battery", "tech_detail": "*"},
                {"parameter": "fixed_o_m_mw", "technology": "Battery", "tech_detail": "*"},
                {"parameter": "fixed_o_m_mwh", "technology": "Battery", "tech_detail": "*"},
                {"parameter": "variable_o_m_mwh", "technology": "Battery", "tech_detail": "*"},
            ]
        )
        self.heat_rates = pd.DataFrame(
            {
                "technology": ["NaturalGas"],
                "tech_detail": ["CC"],
                "cost_case": ["Moderate"],
                "basis_year": [2025],
                "heat_rate": [7.0],
                "data_year": [2023],
            }
        )

    def test_valid_outputs_pass(self):
        MODULE.validate_outputs(self.costs, self.heat_rates, {2023})

    def test_duplicate_cost_records_raise(self):
        costs = pd.concat([self.costs, self.costs.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.validate_outputs(costs, self.heat_rates, {2023})

    def test_duplicate_heat_rate_records_raise(self):
        heat_rates = pd.concat([self.heat_rates, self.heat_rates.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.validate_outputs(self.costs, heat_rates, {2023})

    def test_missing_battery_parameters_raise(self):
        costs = self.costs.loc[
            ~(
                (self.costs["technology"] == "Battery")
                & (self.costs["parameter"] == "fixed_o_m_mwh")
            )
        ].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "battery output misses"):
            MODULE.validate_outputs(costs, self.heat_rates, {2023})

    def test_nan_values_raise(self):
        costs = self.costs.copy()
        costs.loc[0, "parameter_value"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing or non-finite"):
            MODULE.validate_outputs(costs, self.heat_rates, {2023})

    def test_empty_tech_detail_raises(self):
        costs = self.costs.copy()
        costs.loc[0, "tech_detail"] = ""
        with self.assertRaisesRegex(ValueError, "technology details"):
            MODULE.validate_outputs(costs, self.heat_rates, {2023})

    def test_unexpected_years_raise(self):
        with self.assertRaisesRegex(ValueError, "years"):
            MODULE.validate_outputs(self.costs, self.heat_rates, {2024})

    def test_wrong_schema_raises(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            MODULE.validate_outputs(self.costs[list(reversed(MODULE.COST_COLUMNS))], self.heat_rates, {2023})


class BuildFallbackTests(FixtureTestCase):
    def rebuilt_2023(self):
        costs, heat_rates = MODULE.build_year(self.parquet_2023, self.workbook, 2023)
        return costs, heat_rates

    def test_coverage_gap_reuses_committed_rows_for_both_artifacts(self):
        rebuilt_costs, rebuilt_heat_rates = self.rebuilt_2023()
        # The committed data carries the same keys with different values plus
        # a pumped storage record the official source can no longer provide.
        committed_costs = rebuilt_costs.copy()
        committed_costs["parameter_value"] = committed_costs["parameter_value"] * 2
        committed_costs = pd.concat(
            [
                committed_costs,
                cost_frame(
                    [
                        {
                            "technology": "Pumped Storage Hydropower",
                            "tech_detail": "*",
                            "parameter": "capex_mw",
                            "parameter_value": 5_000_000.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        committed_heat_rates = rebuilt_heat_rates.copy()
        committed_heat_rates["heat_rate"] = committed_heat_rates["heat_rate"] * 2
        output_dir = self.write_committed(committed_costs, committed_heat_rates)

        with self.patch_sources({2023: (self.parquet_2023, self.workbook)}), self.assertLogs(
            level="WARNING"
        ) as logs:
            costs, heat_rates = MODULE.build(years=(2023,), output_dir=output_dir)

        self.assertIn("reusing committed", " ".join(logs.output))
        # Both artifacts fall back together so the year stays one vintage.
        natural_gas = costs.loc[
            (costs["technology"] == "NaturalGas") & (costs["parameter"] == "capex_mw")
        ]
        self.assertEqual(natural_gas["parameter_value"].iloc[0], 2_000_000.0)
        self.assertEqual(heat_rates["heat_rate"].iloc[0], 14.0)
        self.assertIn(
            "Pumped Storage Hydropower", set(costs["technology"])
        )

    def test_complete_coverage_keeps_rebuilt_rows(self):
        rebuilt_costs, rebuilt_heat_rates = self.rebuilt_2023()
        committed_costs = rebuilt_costs.copy()
        committed_costs["parameter_value"] = committed_costs["parameter_value"] * 2
        output_dir = self.write_committed(committed_costs, rebuilt_heat_rates.copy())

        with self.patch_sources({2023: (self.parquet_2023, self.workbook)}):
            costs, heat_rates = MODULE.build(years=(2023,), output_dir=output_dir)

        natural_gas = costs.loc[
            (costs["technology"] == "NaturalGas") & (costs["parameter"] == "capex_mw")
        ]
        self.assertEqual(natural_gas["parameter_value"].iloc[0], 1_000_000.0)
        self.assertEqual(heat_rates["heat_rate"].iloc[0], 7.0)

    def test_source_failure_reuses_committed_rows(self):
        rebuilt_costs, rebuilt_heat_rates = self.rebuilt_2023()
        output_dir = self.write_committed(rebuilt_costs, rebuilt_heat_rates)

        with mock.patch.object(
            MODULE, "source_paths", side_effect=FileNotFoundError("source gone")
        ), self.assertLogs(level="WARNING") as logs:
            MODULE.build(years=(2023,), output_dir=output_dir)

        self.assertIn("source gone", " ".join(logs.output))

    def test_source_failure_without_committed_rows_raises(self):
        with mock.patch.object(MODULE, "source_paths", side_effect=FileNotFoundError("gone")):
            with self.assertRaises(RuntimeError):
                MODULE.build(years=(2024,), output_dir=self.tmp)

    def test_required_year_never_falls_back(self):
        output_dir = self.write_committed(
            cost_frame([{"data_year": 2025, "dollar_year": 2023}])
        )
        with mock.patch.object(MODULE, "source_paths", side_effect=FileNotFoundError("gone")):
            with self.assertRaisesRegex(RuntimeError, "required ATB 2025"):
                MODULE.build(years=(2025,), output_dir=output_dir)


class BuildOutputTests(FixtureTestCase):
    def test_build_writes_atomic_deterministic_outputs(self):
        sources = {
            year: (getattr(self, f"parquet_{year}"), self.workbook)
            for year in (2023, 2024, 2025)
        }
        with self.patch_sources(sources):
            first_costs, first_heat_rates = MODULE.build(years=(2023, 2024, 2025), output_dir=self.tmp)
        cost_path = self.tmp / MODULE.COST_OUTPUT_NAME
        heat_rate_path = self.tmp / MODULE.HEAT_RATE_OUTPUT_NAME
        self.assertTrue(cost_path.exists())
        self.assertTrue(heat_rate_path.exists())
        self.assertEqual(sorted(path.name for path in self.tmp.glob("*.tmp*")), [])

        written_costs = pd.read_parquet(cost_path)
        written_heat_rates = pd.read_csv(heat_rate_path)
        pd.testing.assert_frame_equal(written_costs, first_costs, check_exact=True)
        self.assertEqual(
            written_costs["data_year"].dtype, np.dtype("int64")
        )
        self.assertEqual(set(written_costs["data_year"]), {2023, 2024, 2025})
        self.assertEqual(set(written_heat_rates["data_year"]), {2023, 2024, 2025})

        # A second build from the same sources reproduces identical outputs.
        second = self.tmp.parent / (self.tmp.name + "-second")
        second.mkdir()
        with self.patch_sources(sources):
            MODULE.build(years=(2023, 2024, 2025), output_dir=second)
        pd.testing.assert_frame_equal(
            pd.read_parquet(second / MODULE.COST_OUTPUT_NAME), first_costs, check_exact=True
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(second / MODULE.HEAT_RATE_OUTPUT_NAME),
            first_heat_rates.astype({"basis_year": int, "data_year": int}),
            check_exact=True,
        )
        shutil.rmtree(second, ignore_errors=True)

    def test_subset_build_preserves_unrequested_years(self):
        sources = {2025: (self.parquet_2025, self.workbook)}
        historical = [
            MODULE.build_year(self.parquet_2023, self.workbook, 2023),
            MODULE.build_year(self.parquet_2024, self.workbook, 2024),
        ]
        existing_costs = pd.concat(
            [costs for costs, _ in historical], ignore_index=True
        )
        existing_heat_rates = pd.concat(
            [heat_rates for _, heat_rates in historical], ignore_index=True
        )
        existing_costs.to_parquet(self.tmp / MODULE.COST_OUTPUT_NAME, index=False)
        existing_heat_rates.to_csv(self.tmp / MODULE.HEAT_RATE_OUTPUT_NAME, index=False)

        with self.patch_sources(sources):
            costs, heat_rates = MODULE.build(years=(2025,), output_dir=self.tmp)

        self.assertEqual(set(costs["data_year"]), {2023, 2024, 2025})
        self.assertEqual(set(heat_rates["data_year"]), {2023, 2024, 2025})
        preserved = costs.loc[costs["data_year"].isin({2023, 2024})]
        expected = existing_costs.astype(
                {
                    "data_year": int,
                    "basis_year": int,
                    "cap_recovery_years": int,
                    "dollar_year": int,
                }
            )
        pd.testing.assert_frame_equal(
            preserved.sort_values(MODULE.COST_KEY_COLUMNS).reset_index(drop=True),
            expected.sort_values(MODULE.COST_KEY_COLUMNS).reset_index(drop=True),
        )

    def test_failed_write_removes_temporary_files(self):
        sources = {
            year: (getattr(self, f"parquet_{year}"), self.workbook)
            for year in (2023, 2024, 2025)
        }
        with self.patch_sources(sources), mock.patch.object(
            pd.DataFrame, "to_csv", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                MODULE.build(years=(2023, 2024, 2025), output_dir=self.tmp)

        self.assertEqual(sorted(path.name for path in self.tmp.glob("*.tmp*")), [])

    def test_failed_second_replace_restores_existing_output_pair(self):
        sources = {
            year: (getattr(self, f"parquet_{year}"), self.workbook)
            for year in (2023, 2024, 2025)
        }
        cost_path = self.tmp / MODULE.COST_OUTPUT_NAME
        heat_rate_path = self.tmp / MODULE.HEAT_RATE_OUTPUT_NAME
        old_costs, old_heat_rates = MODULE.build_year(
            self.parquet_2023, self.workbook, 2023
        )
        old_costs.to_parquet(cost_path, index=False)
        old_heat_rates.to_csv(heat_rate_path, index=False)
        old_cost_bytes = cost_path.read_bytes()
        old_heat_rate_bytes = heat_rate_path.read_bytes()
        original_replace = Path.replace

        def fail_heat_rate_replace(path, target):
            if path.name.endswith(".tmp.csv"):
                raise OSError("replace failed")
            return original_replace(path, target)

        with self.patch_sources(sources), mock.patch.object(
            Path, "replace", autospec=True, side_effect=fail_heat_rate_replace
        ):
            with self.assertRaisesRegex(OSError, "replace failed"):
                MODULE.build(years=(2023, 2024, 2025), output_dir=self.tmp)

        self.assertEqual(cost_path.read_bytes(), old_cost_bytes)
        self.assertEqual(heat_rate_path.read_bytes(), old_heat_rate_bytes)
        self.assertEqual(sorted(path.name for path in self.tmp.glob("*.tmp*")), [])


if __name__ == "__main__":
    unittest.main()
