import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "update_data_manifest.py"
SPEC = importlib.util.spec_from_file_location("update_data_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

# A filename with a single seeded source.
SINGLE_SOURCE_FILE = "fuel_prices.parquet"
# A filename with multiple seeded sources.
MULTI_SOURCE_FILE = "plant_region_map.csv"


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

    def test_initial_run_seeds_sources_and_sets_version(self):
        data_dir = self._make_data_dir({SINGLE_SOURCE_FILE: "a", "unknown.csv": "b"})
        manifest = self._run(data_dir, "2026-08-11")
        self.assertEqual(manifest["manifest_version"], 1)
        self.assertEqual(manifest["data_version"], "2026.08.11")

        seeded = manifest["files"][SINGLE_SOURCE_FILE]
        self.assertEqual(seeded["version"], "2026-08-11")
        self.assertEqual(seeded["md5"], MODULE.md5_of(data_dir / SINGLE_SOURCE_FILE))
        self.assertEqual(seeded["history"], [])
        self.assertEqual(len(seeded["sources"]), 1)
        self.assertEqual(seeded["sources"][0]["source"], MODULE.SOURCES[SINGLE_SOURCE_FILE][0]["source"])
        self.assertIn("source_url", seeded["sources"][0])

        unknown = manifest["files"]["unknown.csv"]
        self.assertEqual(unknown["sources"], [{"source": "Unknown - document me"}])

    def test_multi_source_file_seeds_all_sources(self):
        data_dir = self._make_data_dir({MULTI_SOURCE_FILE: "a"})
        manifest = self._run(data_dir, "2026-08-11")
        sources = manifest["files"][MULTI_SOURCE_FILE]["sources"]
        expected = MODULE.SOURCES[MULTI_SOURCE_FILE]
        self.assertEqual(len(sources), len(expected))
        for i, src in enumerate(sources):
            self.assertEqual(src["source"], expected[i]["source"])
            self.assertEqual(src.get("source_url"), expected[i].get("source_url"))

    def test_change_bumps_version_appends_history_and_advances_data_version(self):
        data_dir = self._make_data_dir({SINGLE_SOURCE_FILE: "a"})
        first = self._run(data_dir, "2026-08-11")
        (data_dir / SINGLE_SOURCE_FILE).write_text("aa")
        second = self._run(data_dir, "2026-09-03")

        self.assertEqual(second["data_version"], "2026.09.03")
        entry = second["files"][SINGLE_SOURCE_FILE]
        self.assertEqual(entry["version"], "2026-09-03")
        self.assertEqual(entry["last_updated"], "2026-09-03")
        self.assertEqual(len(entry["history"]), 1)
        self.assertEqual(entry["history"][0]["version"], "2026-08-11")
        self.assertEqual(entry["history"][0]["md5"], first["files"][SINGLE_SOURCE_FILE]["md5"])

    def test_source_edits_are_preserved_on_change(self):
        data_dir = self._make_data_dir({SINGLE_SOURCE_FILE: "a"})
        manifest = self._run(data_dir, "2026-08-11")
        manifest["files"][SINGLE_SOURCE_FILE]["sources"].append(
            {"source": "A hand-added second source", "source_url": "https://example.com/x"}
        )
        manifest["files"][SINGLE_SOURCE_FILE]["sources"][0]["source"] = "Custom edited text"
        (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        (data_dir / SINGLE_SOURCE_FILE).write_text("different")
        result = self._run(data_dir, "2026-10-01")
        sources = result["files"][SINGLE_SOURCE_FILE]["sources"]
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["source"], "Custom edited text")
        self.assertEqual(sources[1]["source"], "A hand-added second source")
        self.assertEqual(sources[1]["source_url"], "https://example.com/x")

    def test_no_change_preserves_data_version_and_versions(self):
        data_dir = self._make_data_dir({SINGLE_SOURCE_FILE: "a"})
        self._run(data_dir, "2026-08-11")
        second = self._run(data_dir, "2026-12-31")

        self.assertEqual(second["data_version"], "2026.08.11")
        self.assertEqual(second["files"][SINGLE_SOURCE_FILE]["version"], "2026-08-11")
        self.assertEqual(second["files"][SINGLE_SOURCE_FILE]["history"], [])

    def test_manifest_excludes_itself_and_output_is_deterministic(self):
        data_dir = self._make_data_dir({SINGLE_SOURCE_FILE: "a"})
        self._run(data_dir, "2026-08-11")
        manifest = self._run(data_dir, "2026-08-11")

        self.assertNotIn("manifest.json", manifest["files"])

        def stable_body(m):
            body = {k: v for k, v in m.items() if k != "generated_at_utc"}
            return json.dumps(body, indent=2, sort_keys=True)

        first = stable_body(json.loads((data_dir / "manifest.json").read_text()))
        self.assertEqual(first, stable_body(manifest))

    def test_old_single_field_format_is_migrated(self):
        data_dir = self._make_data_dir({SINGLE_SOURCE_FILE: "a"})
        manifest = self._run(data_dir, "2026-08-11")
        # Rewrite the manifest in the old (single source/source_url) format.
        old = {"manifest_version": 1, "data_version": "2026.08.11", "files": {}}
        for name, ent in manifest["files"].items():
            s0 = ent["sources"][0]
            old["files"][name] = {
                "source": s0["source"],
                "source_url": s0.get("source_url", ""),
                "last_updated": ent["last_updated"],
                "version": ent["version"],
                "md5": ent["md5"],
                "history": ent["history"],
            }
        (data_dir / "manifest.json").write_text(json.dumps(old, indent=2))

        result = self._run(data_dir, "2026-08-11")
        entry = result["files"][SINGLE_SOURCE_FILE]
        self.assertNotIn("source", entry)
        self.assertNotIn("source_url", entry)
        self.assertEqual(len(entry["sources"]), 1)
        self.assertEqual(entry["sources"][0]["source"], manifest["files"][SINGLE_SOURCE_FILE]["sources"][0]["source"])
        # data_version preserved (no content changed)
        self.assertEqual(result["data_version"], "2026.08.11")

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

    def test_empty_or_null_source_normalized_to_placeholder(self):
        norm = MODULE._normalize_sources([{"source": None}, {"source": ""}, {"source": "   "}])
        self.assertEqual(norm, [{"source": "Unknown - document me"}] * 3)

    def test_placeholder_sources_backfilled_from_seed_on_rerun(self):
        data_dir = self._make_data_dir({SINGLE_SOURCE_FILE: "a"})
        manifest = self._run(data_dir, "2026-08-11")
        # Simulate an older manifest that predates documented provenance.
        manifest["files"][SINGLE_SOURCE_FILE]["sources"] = [
            {"source": "Unknown - document me"}
        ]
        (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        result = self._run(data_dir, "2026-08-11")
        sources = result["files"][SINGLE_SOURCE_FILE]["sources"]
        self.assertEqual(sources[0]["source"], MODULE.SOURCES[SINGLE_SOURCE_FILE][0]["source"])
        self.assertIn("source_url", sources[0])

    def test_hand_edited_sources_not_overwritten_by_backfill(self):
        # A real (non-placeholder) source must be preserved on rerun.
        data_dir = self._make_data_dir({SINGLE_SOURCE_FILE: "a"})
        manifest = self._run(data_dir, "2026-08-11")
        custom = {"source": "Custom documented source", "source_url": "https://example.com/x"}
        manifest["files"][SINGLE_SOURCE_FILE]["sources"] = [custom]
        (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        result = self._run(data_dir, "2026-08-11")
        self.assertEqual(result["files"][SINGLE_SOURCE_FILE]["sources"], [custom])

    def test_date_argument_validation(self):
        self.assertEqual(MODULE._validated_iso_date("2026-08-11"), "2026-08-11")
        for bad in ("2026-13-45", "08-11-2026", "not-a-date", "2026-02-30", "", "20260811"):
            with self.assertRaises(Exception):
                MODULE._validated_iso_date(bad)


class GitTrackedTests(unittest.TestCase):
    def _make_git_collection(self, directory):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        data_dir = root / directory
        data_dir.mkdir()
        (data_dir / SINGLE_SOURCE_FILE).write_text("a")
        (data_dir / "scratch.csv").write_text("b")
        subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
        subprocess.run(
            ["git", "-C", str(root), "add", f"{directory}/{SINGLE_SOURCE_FILE}"],
            check=True,
        )
        return data_dir

    def test_new_untracked_file_is_skipped_per_collection(self):
        data_dir = self._make_git_collection("resource_profiles")
        manifest_path = data_dir / "manifest.json"
        MODULE.update_manifest(data_dir, manifest_path, "2026-08-11")
        manifest = json.loads(manifest_path.read_text())
        self.assertIn(SINGLE_SOURCE_FILE, manifest["files"])
        self.assertNotIn("scratch.csv", manifest["files"])

    def test_include_untracked_restores_scan_all(self):
        data_dir = self._make_git_collection("existing_resource_groups")
        manifest_path = data_dir / "manifest.json"
        MODULE.update_manifest(
            data_dir, manifest_path, "2026-08-11", include_untracked=True
        )
        manifest = json.loads(manifest_path.read_text())
        self.assertIn("scratch.csv", manifest["files"])

    def test_manifested_untracked_file_remains_managed(self):
        data_dir = self._make_git_collection("data")
        manifest_path = data_dir / "manifest.json"
        MODULE.update_manifest(data_dir, manifest_path, "2026-08-11")
        subprocess.run(
            ["git", "-C", str(data_dir.parent), "rm", "--cached", "data/" + SINGLE_SOURCE_FILE],
            check=True,
            capture_output=True,
        )
        (data_dir / SINGLE_SOURCE_FILE).write_text("changed")
        MODULE.update_manifest(data_dir, manifest_path, "2026-08-12")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["files"][SINGLE_SOURCE_FILE]["version"], "2026-08-12")


if __name__ == "__main__":
    unittest.main()
