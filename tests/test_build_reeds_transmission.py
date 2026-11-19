import csv
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.parametrize("zone_set", ["z134", "z90"])
def test_combines_ac_and_dc_for_each_zone_set(tmp_path, monkeypatch, zone_set):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from build_reeds_transmission import ac_rows, combine_capacity, write_csv

    hashes = pd.Series({"A": "a", "B": "b"})
    ac = ac_rows(
        pd.DataFrame({"interface": ["A~~B"]}), hashes,
        pd.DataFrame({"md5_from": ["a"], "md5_to": ["b"],
                      "MW_forward": [90], "MW_reverse": [50]}),
    )
    dc = pd.DataFrame({"r": ["B"], "rr": ["A"], "MW": [30], "notes": ["HVDC"]})
    b2b = pd.DataFrame({"r": ["A"], "rr": ["B"], "MW": [10]})
    result = combine_capacity(ac, dc, b2b, hashes.index, zone_set)
    path = tmp_path / "transmission.csv"
    write_csv(path, result)
    with path.open(newline="") as f:
        [new] = csv.DictReader(f)
    assert (new["region_from"], new["region_to"], new["zone_set"]) == ("A", "B", zone_set)
    assert float(new["firm_ttc_mw"]) == 110
    assert new["notes"] == "HVDC; B2B converter"


def test_expansion_cost_preserves_distinct_reverse_routes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from build_reeds_transmission import expansion_costs

    routes = pd.DataFrame({"start": ["a", "b", "a"], "end": ["b", "a", "offshore"],
                           "polarity": ["ac"] * 3, "voltage": [345] * 3,
                           "cost_MUSD": [1.23, 3.45, 5], "length_miles": [4, 8, 16]})
    result = expansion_costs(routes, pd.Series({"A": "a", "B": "b"}), 2024, "z134")
    assert list(zip(result.region_from, result.region_to)) == [("A", "B"), ("B", "A")]
    assert result.capital_cost_usd.tolist() == [1_230_000, 3_450_000]
    assert result.length_miles.tolist() == [4, 8]
    assert result.dollar_year.tolist() == [2024, 2024]
