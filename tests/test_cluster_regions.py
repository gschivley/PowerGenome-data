import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "cluster_regions.py"
SPEC = importlib.util.spec_from_file_location("cluster_regions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
load_and_validate_data = MODULE.load_and_validate_data


def test_transmission_resolution_is_opt_in(tmp_path):
    hierarchy_path = tmp_path / "hierarchy.csv"
    hierarchy_path.write_text("r,st\nA,AA\nB,AA\n")
    tx_path = tmp_path / "transmission.csv"
    tx_path.write_text(
        "region_from,region_to,firm_ttc_mw,zone_set\n"
        "p1,p2,1,z134\nA,B,2,z90\n"
    )
    _, legacy = load_and_validate_data(hierarchy_path, tx_path)
    assert legacy["region_from"].tolist() == ["p1"]
    hierarchy, tx = load_and_validate_data(hierarchy_path, tx_path, "z90")
    assert hierarchy["ba"].tolist() == ["A", "B"]
    assert tx["region_from"].tolist() == ["A"]
    with pytest.raises(SystemExit, match="No transmission rows found"):
        load_and_validate_data(hierarchy_path, tx_path, "unknown")

    tx_path.write_text("region_from,region_to,firm_ttc_mw\np1,p2,1\n")
    _, legacy = load_and_validate_data(hierarchy_path, tx_path)
    assert legacy["region_from"].tolist() == ["p1"]
    with pytest.raises(SystemExit, match="without zone_set"):
        load_and_validate_data(hierarchy_path, tx_path, "z90")
