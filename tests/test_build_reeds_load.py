import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).parents[1] / "build_reeds_load.py"
SPEC = importlib.util.spec_from_file_location("build_reeds_load", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
build_load = MODULE.build_load
population_weights = MODULE.population_weights


def test_builds_both_zone_sets_from_same_source(tmp_path):
    weights = population_weights(
        pd.DataFrame({"FIPS": ["01001", "01003", "02001"], "r": ["east", "west", "west"]}),
        pd.DataFrame({"FIPS": ["01001", "01003", "02001"], "state": ["AA", "AA", "BB"]}),
        pd.DataFrame({"FIPS": ["p01001", "p01003", "p02001"], "value": [1, 3, 2]}),
    )
    source = tmp_path / "load.h5"
    with h5py.File(source, "w") as f:
        group = f.create_group("2021")
        dates = pd.date_range("2008-01-01", periods=8760, freq="h", tz="Etc/GMT+6")
        group["datetime"] = dates.map(str).to_numpy(dtype="S30")
        group["AA"] = np.full(8760, 100)
        group["BB"] = np.full(8760, 20)
    output = tmp_path / "combined.parquet"
    options = {"years": (2021,), "weather_years": (2008,)}
    weights_by_zone_set = {"z134": weights.rename(columns={"east": "p1", "west": "p2"}),
                           "z90": weights}
    build_load(source, weights_by_zone_set, output, **options)
    original = pq.read_table(output)
    build_load(source, weights_by_zone_set, output, **options)
    assert pq.read_table(output).equals(original)
    rows = original.to_pandas()
    assert rows.groupby("zone_set").size().to_dict() == {"z134": 17520, "z90": 17520}
    assert set(rows.scenario) == {"EER2025_IRAlow"}
    assert rows.query("zone_set == 'z134'").groupby("region").load_mw.mean().to_dict() == {
        "p1": 25.0, "p2": 95.0,
    }
    assert rows.query("zone_set == 'z90'").groupby("region").load_mw.mean().to_dict() == {
        "east": 25.0, "west": 95.0,
    }
    assert pq.ParquetFile(output).metadata.row_group(1).column(3).compression == "ZSTD"
    with h5py.File(source, "r+") as f:
        f["2021/datetime"][1] = f["2021/datetime"][0]
    with pytest.raises(ValueError, match="Invalid hourly calendar"):
        build_load(source, weights_by_zone_set, output, **options)
    assert pq.read_table(output).equals(original)
