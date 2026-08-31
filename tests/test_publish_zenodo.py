import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).parents[1] / "publish_zenodo.py"
SPEC = importlib.util.spec_from_file_location("publish_zenodo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

MANIFEST = {
    "data_version": "2026.08.14",
    "files": {
        "core_a.csv": {
            "version": "2026-08-11",
            "last_updated": "2026-08-11",
            "md5": "aaa",
        },
    },
    "sections": {
        "profiles": {
            "files": {
                "wind.parquet": {
                    "version": "2026-08-12",
                    "last_updated": "2026-08-12",
                    "md5": "bbb",
                }
            }
        },
        "existing_resource_groups": {"files": {}},
    },
}

LEGACY_STATE = {
    "metadata": {
        "title": "PowerGenome Input Data",
        "description": "old description",
        "creators": [{"name": "Schivley, Greg"}],
    },
    "zenodo_release": {
        "environment": "sandbox",
        "data_version": "2026.08.14",
        "deposition_id": "590994",
        "doi": "10.5072/zenodo.590994",
        "published": True,
        "files": {"core_a.csv": "aaa"},
    },
}

NEW_STATE = {
    "metadata": {
        "creators": [{"name": "Schivley, Greg"}],
        "sections": {
            "profiles": {"title": "PowerGenome Renewable Resource Profiles"},
        },
    },
    "releases": {
        "profiles": {
            "environment": "sandbox",
            "data_version": "2026.08.13",
            "deposition_id": "600001",
            "published": False,
            "files": {"profiles/wind.parquet": "bbb"},
        }
    },
}


class ManifestSectionTests(unittest.TestCase):
    def test_flat_files_are_core_and_sections_map_to_collections(self):
        sections = MODULE.manifest_sections(MANIFEST)
        self.assertEqual(set(sections), set(MODULE.COLLECTIONS))
        self.assertEqual(list(sections["core"]), ["core_a.csv"])
        self.assertEqual(list(sections["profiles"]), ["wind.parquet"])
        self.assertEqual(sections["existing_resource_groups"], {})

    def test_missing_sections_default_to_empty(self):
        sections = MODULE.manifest_sections(
            {"data_version": "2026.08.14", "files": {"a.csv": {}}}
        )
        self.assertEqual(sections["profiles"], {})
        self.assertEqual(sections["existing_resource_groups"], {})

    def test_unknown_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "data_version": "2026.08.14",
                        "files": {},
                        "sections": {"bogus": {"files": {}}},
                    }
                )
            )
            with self.assertRaises(SystemExit):
                MODULE.load_manifest(path)


class StateMigrationTests(unittest.TestCase):
    def test_legacy_zenodo_release_migrates_to_core(self):
        released = MODULE.section_release(LEGACY_STATE, "core")
        self.assertEqual(released["deposition_id"], "590994")
        self.assertEqual(released["doi"], "10.5072/zenodo.590994")
        self.assertTrue(released["published"])

    def test_legacy_state_has_no_non_core_sections(self):
        self.assertEqual(MODULE.section_release(LEGACY_STATE, "profiles"), {})

    def test_new_releases_state_wins_over_legacy(self):
        state = dict(LEGACY_STATE)
        state["releases"] = {"core": {"deposition_id": "700002", "published": False}}
        released = MODULE.section_release(state, "core")
        self.assertEqual(released["deposition_id"], "700002")

    def test_new_state_section_lookup(self):
        released = MODULE.section_release(NEW_STATE, "profiles")
        self.assertEqual(released["deposition_id"], "600001")
        self.assertEqual(MODULE.section_release(NEW_STATE, "core"), {})


class MetadataTests(unittest.TestCase):
    def test_section_metadata_merges_overrides(self):
        meta = MODULE.section_metadata(NEW_STATE, "profiles")
        self.assertEqual(meta["title"], "PowerGenome Renewable Resource Profiles")
        self.assertEqual(meta["creators"], [{"name": "Schivley, Greg"}])
        self.assertNotIn("sections", meta)

    def test_section_metadata_defaults_without_overrides(self):
        meta = MODULE.section_metadata(LEGACY_STATE, "existing_resource_groups")
        self.assertEqual(meta["creators"], [{"name": "Schivley, Greg"}])
        self.assertNotIn("title", meta)  # per-release field, not shared


class DescriptionTests(unittest.TestCase):
    def test_description_uses_section_title_and_files(self):
        files = MANIFEST["sections"]["profiles"]["files"]
        description = MODULE.build_description(
            MANIFEST, "profiles", files, ["wind.parquet"], [], [], True
        )
        self.assertIn("PowerGenome Renewable Resource Profiles", description)
        self.assertIn("wind.parquet", description)
        self.assertIn("2026.08.14", description)
        self.assertNotIn("core_a.csv", description)

    def test_description_lists_changes(self):
        files = MANIFEST["files"]
        description = MODULE.build_description(
            MANIFEST, "core", files, [], ["core_a.csv"], ["gone.csv"], False
        )
        self.assertIn("Files updated in this release", description)
        self.assertIn("Files removed in this release", description)
        self.assertIn("gone.csv", description)


class DryRunTests(unittest.TestCase):
    def test_dry_run_covers_all_sections(self):
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data" / "core_a.csv").write_text("a")
            manifest = {
                "data_version": "2026.08.14",
                "files": {"core_a.csv": {"version": "2026-08-11", "md5": "x"}},
                "sections": {
                    "profiles": {"files": {}},
                    "existing_resource_groups": {"files": {}},
                },
            }
            manifest_path = root / "data" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with mock.patch.object(MODULE.Path, "cwd", return_value=root):
                out = io.StringIO()
                with redirect_stdout(out):
                    MODULE.dry_run(
                        manifest_path,
                        MODULE.requested_sections(mock.Mock(collection=None)),
                    )
            text = out.getvalue()
        self.assertIn("[core] PowerGenome Input Data", text)
        self.assertIn("ok      core_a.csv", text)
        self.assertIn("[profiles] PowerGenome Renewable Resource Profiles", text)
        self.assertIn("deposit skipped", text)
        self.assertIn(
            "[existing_resource_groups] PowerGenome Existing Renewable Resource Groups",
            text,
        )


class CollectionFilterTests(unittest.TestCase):
    def test_requested_sections_defaults_to_all(self):
        args = mock.Mock(collection=None)
        self.assertEqual(MODULE.requested_sections(args), list(MODULE.COLLECTIONS))

    def test_requested_sections_honors_repeated_flags(self):
        args = mock.Mock(collection=["profiles", "core"])
        self.assertEqual(MODULE.requested_sections(args), ["profiles", "core"])

    def test_dry_run_filters_to_requested_collection(self):
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "resource_profiles").mkdir()
            (root / "resource_profiles" / "wind.parquet").write_text("w")
            manifest = {
                "data_version": "2026.08.14",
                "files": {"core_a.csv": {"version": "2026-08-11", "md5": "x"}},
                "sections": {
                    "profiles": {
                        "files": {"wind.parquet": {"version": "2026-08-12", "md5": "y"}}
                    },
                    "existing_resource_groups": {"files": {}},
                },
            }
            manifest_path = root / "data" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with mock.patch.object(MODULE.Path, "cwd", return_value=root):
                out = io.StringIO()
                with redirect_stdout(out):
                    MODULE.dry_run(manifest_path, ["profiles"])
            text = out.getvalue()
        self.assertNotIn("[core]", text)
        self.assertIn("[profiles] PowerGenome Renewable Resource Profiles", text)
        self.assertIn("ok      profiles/wind.parquet", text)
        self.assertNotIn("existing_resource_groups", text)

    def test_unknown_collection_rejected_by_parser(self):
        parser = MODULE.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--collection", "bogus"])


if __name__ == "__main__":
    unittest.main()
