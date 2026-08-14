import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "build_operational_constraints_reeds.py"
SPEC = importlib.util.spec_from_file_location("build_operational_constraints_reeds", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

# Synthetic PCM payload. Every tech_map value must be present here or the
# tech_map copy step raises. CSP rows are included to verify they are dropped.
PCM_BASE_FIELDS = {
    "battery_li": {
        "min_stable_level_percentage": None,
        "max_ramp_up_percentage": None,
        "min_up_time": None,
        "min_down_time": None,
        "start_cost_per_MW": None,
    },
    "battery": {
        "min_stable_level_percentage": None,
        "max_ramp_up_percentage": None,
        "min_up_time": None,
        "min_down_time": None,
        "start_cost_per_MW": None,
    },
    "pumped-hydro": {
        "min_stable_level_percentage": None,
        "max_ramp_up_percentage": None,
        "min_up_time": None,
        "min_down_time": None,
        "start_cost_per_MW": None,
    },
    "gas-cc": {
        "min_stable_level_percentage": 0.5,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
    "gas-cc_h_2x1": {
        "min_stable_level_percentage": 0.5,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
    "gas-cc_h_1x1": {
        "min_stable_level_percentage": 0.5,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
    "gas-cc-ccs": {
        "min_stable_level_percentage": 0.5,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
    "gas-cc-ccs_max": {
        "min_stable_level_percentage": 0.5,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
    "gas-cc_h_2x1-ccs_mod": {
        "min_stable_level_percentage": 0.5,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
    "gas-cc_h_2x1-ccs_max": {
        "min_stable_level_percentage": 0.5,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
    "gas-cc_h_1x1-ccs_mod": {
        "min_stable_level_percentage": 0.5,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
    "gas-cc_h_1x1-ccs_max": {
        "min_stable_level_percentage": 0.5,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
    "gas-ct": {
        "min_stable_level_percentage": 0.6,
        "max_ramp_up_percentage": 0.08,
        "min_up_time": 0.0,
        "min_down_time": 0.0,
        "start_cost_per_MW": 69.0,
    },
    "gas-ct_aero": {
        "min_stable_level_percentage": 0.6,
        "max_ramp_up_percentage": 0.08,
        "min_up_time": 0.0,
        "min_down_time": 0.0,
        "start_cost_per_MW": 69.0,
    },
    "ng-fuel-cell": {
        "min_stable_level_percentage": 0.5,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
    "coal-new": {
        "min_stable_level_percentage": 0.4,
        "max_ramp_up_percentage": 0.02,
        "min_up_time": 24.0,
        "min_down_time": 12.0,
        "start_cost_per_MW": 129.0,
    },
    "coal-igcc": {
        "min_stable_level_percentage": 0.4,
        "max_ramp_up_percentage": 0.02,
        "min_up_time": 24.0,
        "min_down_time": 12.0,
        "start_cost_per_MW": 129.0,
    },
    "nuclear": {
        "min_stable_level_percentage": 1.0,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
    "hydd": {
        "min_stable_level_percentage": None,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": None,
        "min_down_time": None,
        "start_cost_per_MW": None,
    },
    "lfill-gas": {
        "min_stable_level_percentage": 1.0,
        "max_ramp_up_percentage": 0.14,
        "min_up_time": 7.0,
        "min_down_time": 7.0,
        "start_cost_per_MW": 5.3,
    },
    "csp-ns": {
        "min_stable_level_percentage": 0.5,
        "max_ramp_up_percentage": 0.05,
        "min_up_time": 6.0,
        "min_down_time": 8.0,
        "start_cost_per_MW": 79.0,
    },
}


def make_pcm_df() -> pd.DataFrame:
    return pd.DataFrame(
        {tech: fields for tech, fields in PCM_BASE_FIELDS.items()}
    ).T.reset_index().rename(columns={"index": "tech"})


class OperationalConstraintsTests(unittest.TestCase):
    def test_csp_rows_are_dropped(self):
        df = MODULE.build_operational_constraints_reeds(make_pcm_df())
        self.assertFalse(df["Resource"].astype(str).str.startswith("csp").any())

    def test_all_pcm_techs_and_tech_map_names_present(self):
        df = MODULE.build_operational_constraints_reeds(make_pcm_df())
        resources = set(df["Resource"])
        # Every non-CSP PCM tech is a row.
        non_csp = {t for t in PCM_BASE_FIELDS if not t.startswith("csp")}
        self.assertTrue(non_csp.issubset(resources))
        # Every tech_map key (new-build ATB name) is a row.
        self.assertTrue(set(MODULE.TECH_MAP).issubset(resources))

    def test_battery_efficiency_and_duration(self):
        df = MODULE.build_operational_constraints_reeds(make_pcm_df())
        row = df[df["Resource"] == "battery_li"].iloc[0]
        self.assertAlmostEqual(row["Eff_Up"], np.sqrt(0.85), places=9)
        self.assertAlmostEqual(row["Eff_Down"], np.sqrt(0.85), places=9)
        self.assertEqual(row["Max_Duration"], 8)
        self.assertEqual(row["Min_Duration"], 0)

    def test_pumped_hydro_efficiency_and_duration(self):
        df = MODULE.build_operational_constraints_reeds(make_pcm_df())
        row = df[df["Resource"] == "pumped-hydro"].iloc[0]
        self.assertAlmostEqual(row["Eff_Up"], np.sqrt(0.8), places=9)
        self.assertAlmostEqual(row["Eff_Down"], np.sqrt(0.8), places=9)
        self.assertEqual(row["Max_Duration"], 13)
        self.assertEqual(row["Min_Duration"], 11)

    def test_min_power_of_1_reduced_to_07(self):
        df = MODULE.build_operational_constraints_reeds(make_pcm_df())
        row = df[df["Resource"] == "nuclear"].iloc[0]
        self.assertEqual(row["Min_Power"], 0.7)

    def test_tech_map_renames_row(self):
        df = MODULE.build_operational_constraints_reeds(make_pcm_df())
        new_row = df[df["Resource"] == "naturalgas_2on1_combined_cycle_hframe"].iloc[0]
        self.assertEqual(new_row["Min_Power"], 0.5)
        self.assertEqual(new_row["Start_Cost_per_MW"], 79)

    def test_thermal_rows_have_ramp_and_times(self):
        df = MODULE.build_operational_constraints_reeds(make_pcm_df())
        for resource in ("gas-cc", "gas-ct", "nuclear"):
            row = df[df["Resource"] == resource].iloc[0]
            self.assertEqual(row["Ramp_Up_Percentage"], row["Ramp_Dn_Percentage"])
            self.assertFalse(pd.isna(row["Ramp_Up_Percentage"]))

    def test_curated_rows_appended(self):
        df = MODULE.build_operational_constraints_reeds(make_pcm_df())
        curated_names = {row["Resource"] for row in MODULE.CURATED_ROWS}
        self.assertTrue(curated_names.issubset(set(df["Resource"])))

    def test_output_columns(self):
        df = MODULE.build_operational_constraints_reeds(make_pcm_df())
        self.assertEqual(list(df.columns), MODULE.OUTPUT_COLUMNS)

    def test_duplicate_tech_map_value_raises(self):
        # A tech_map value that is not a PCM row must raise ValueError.
        with self.assertRaises(ValueError):
            MODULE.build_operational_constraints(
                make_pcm_df().assign(tech=lambda d: d["tech"].replace({"gas-cc": "ghost-tech"}))
            )


if __name__ == "__main__":
    unittest.main()
