import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "update_data_manifest.py"
SPEC = importlib.util.spec_from_file_location("update_data_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

SEEDED_FILE = "cpi_data.csv"


class ManifestTests(unittest.TestCase):
    def _make_data_dir(self, files):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        data_dir = Path(tmp.name) / "data"
        data_dir.mkdir()
        for name, content in files.items():
            (data_dir / name).write_text(content)
        return data_dir

    def _run(self, data_dir, date_str):
        manifest_path = data_dir / "manifest.json"
        MODULE.update_manifest(data_dir, manifest_path, date_str)
        return json.loads(manifest_path.read_text())

    def test_initial_run_seeds_source_and_sets_version(self):
        data_dir = self._make_data_dir({SEEDED_FILE: "a", "unknown.csv": "b"})
        manifest = self._run(data_dir, "2026-08-11")
        self.assertEqual(manifest["manifest_version"], 1)
        self.assertEqual(manifest["data_version"], "2026.08.11")

        seeded = manifest["files"][SEEDED_FILE]
        self.assertEqual(seeded["version"], "2026-08-11")
        self.assertEqual(seeded["md5"], MODULE.md5_of(data_dir / SEEDED_FILE))
        self.assertNotEqual(seeded["source"], "Unknown - document me")
        self.assertEqual(seeded["history"], [])

        unknown = manifest["files"]["unknown.csv"]
        self.assertEqual(unknown["source"], "Unknown - document me")

    def test_change_bumps_version_appends_history_and_advances_data_version(self):
        data_dir = self._make_data_dir({SEEDED_FILE: "a"})
        first = self._run(data_dir, "2026-08-11")
        (data_dir / SEEDED_FILE).write_text("aa")
        second = self._run(data_dir, "2026-09-03")

        self.assertEqual(second["data_version"], "2026.09.03")
        entry = second["files"][SEEDED_FILE]
        self.assertEqual(entry["version"], "2026-09-03")
        self.assertEqual(entry["last_updated"], "2026-09-03")
        self.assertEqual(len(entry["history"]), 1)
        self.assertEqual(entry["history"][0]["version"], "2026-08-11")
        self.assertEqual(entry["history"][0]["md5"], first["files"][SEEDED_FILE]["md5"])

    def test_source_edits_are_preserved_on_change(self):
        data_dir = self._make_data_dir({SEEDED_FILE: "a"})
        manifest = self._run(data_dir, "2026-08-11")
        manifest["files"][SEEDED_FILE]["source"] = "My custom upstream"
        manifest["files"][SEEDED_FILE]["source_url"] = "https://example.com/source"
        (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        (data_dir / SEEDED_FILE).write_text("different")
        result = self._run(data_dir, "2026-10-01")
        entry = result["files"][SEEDED_FILE]
        self.assertEqual(entry["source"], "My custom upstream")
        self.assertEqual(entry["source_url"], "https://example.com/source")

    def test_no_change_preserves_data_version_and_versions(self):
        data_dir = self._make_data_dir({SEEDED_FILE: "a"})
        self._run(data_dir, "2026-08-11")
        second = self._run(data_dir, "2026-12-31")

        self.assertEqual(second["data_version"], "2026.08.11")
        self.assertEqual(second["files"][SEEDED_FILE]["version"], "2026-08-11")
        self.assertEqual(second["files"][SEEDED_FILE]["history"], [])

    def test_manifest_excludes_itself_and_output_is_deterministic(self):
        data_dir = self._make_data_dir({SEEDED_FILE: "a"})
        self._run(data_dir, "2026-08-11")
        manifest = self._run(data_dir, "2026-08-11")

        # manifest.json must not appear as its own entry
        self.assertNotIn("manifest.json", manifest["files"])

        # deterministic output: ignoring the timestamp, two no-change runs are identical
        def stable_body(m):
            body = {k: v for k, v in m.items() if k != "generated_at_utc"}
            return json.dumps(body, indent=2, sort_keys=True)

        first = stable_body(json.loads((data_dir / "manifest.json").read_text()))
        self.assertEqual(first, stable_body(manifest))

    def test_md5_change_regardless_of_history_metadata(self):
        data_dir = self._make_data_dir({"stable.csv": "x"})
        first = self._run(data_dir, "2026-08-11")
        (data_dir / "stable.csv").write_text("xx")
        second = self._run(data_dir, "2026-08-11")
        self.assertEqual(second["files"]["stable.csv"]["version"], "2026-08-11")
        self.assertEqual(len(second["files"]["stable.csv"]["history"]), 1)
        self.assertEqual(
            second["files"]["stable.csv"]["history"][0]["md5"],
            first["files"]["stable.csv"]["md5"],
        )


if __name__ == "__main__":
    unittest.main()
