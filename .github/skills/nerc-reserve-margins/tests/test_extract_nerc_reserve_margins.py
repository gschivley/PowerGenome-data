import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_nerc_reserve_margins.py"
SPEC = importlib.util.spec_from_file_location("nerc_reserve_margins", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExtractorTests(unittest.TestCase):
    def test_parse_page_converts_percentages_and_year_ranges(self):
        text = """2025 Long-Term Reliability Assessment 154

WECC-Northwest
WECC-Northwest is a winter-peaking assessment area.
Demand, Resources, and Reserve Margins
Quantity 2026-2027 2027-2028 2028-2029
Reference Margin Level (%) 17.8% 17.4% 16.1%
"""
        result = MODULE.parse_page(text, 154)
        self.assertEqual(result.region.tolist(), ["WECC-Northwest"] * 3)
        self.assertEqual(result.planning_year.tolist(), [2026, 2027, 2028])
        for actual, expected in zip(result.value.tolist(), [0.178, 0.174, 0.161]):
            self.assertAlmostEqual(actual, expected)
            self.assertEqual(format(actual, ".3f"), format(expected, ".3f"))

    def test_extend_to_target_uses_each_region_last_value(self):
        source = MODULE.pd.DataFrame(
            {
                "region": ["A", "A", "B"],
                "planning_year": [2026, 2027, 2026],
                "value": [0.1, 0.2, 0.3],
            }
        )
        result = MODULE.extend_to_target(source, 2029)
        self.assertEqual(
            result[result.region == "A"].value.tolist(), [0.1, 0.2, 0.2, 0.2]
        )
        self.assertEqual(
            result[result.region == "B"].value.tolist(), [0.3, 0.3, 0.3, 0.3]
        )

    def test_missing_reference_values_fail(self):
        text = """2025 Long-Term Reliability Assessment 42

MISO
MISO is an assessment area.
Demand, Resources, and Reserve Margins
Quantity 2026 2027
Reference Margin Level (%)
"""
        with self.assertRaises(MODULE.ExtractionError):
            MODULE.parse_page(text, 42)


if __name__ == "__main__":
    unittest.main()
