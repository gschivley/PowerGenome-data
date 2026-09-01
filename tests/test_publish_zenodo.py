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
            "files": {"wind.parquet": "bbb"},
        }
    },
}


class ManifestSectionTests(unittest.TestCase):
    def _write(self, root, rel_dir, content):
        d = root / rel_dir
        d.mkdir(parents=True, exist_ok=True)
        p = d / "manifest.json"
        p.write_text(json.dumps(content))
        return p

    def _args(self, manifest):
        args = mock.Mock()
        args.manifest = str(manifest)
        args.collection = None
        return args

    def test_local_manifests_are_used_per_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root,
                "data",
                {"data_version": "2026.08.14", "files": {"core_a.csv": {}}},
            )
            self._write(
                root,
                "resource_profiles",
                {"data_version": "2026.08.31", "files": {"wind.parquet": {}}},
            )
            self._write(
                root,
                "existing_resource_groups",
                {"data_version": "2026.08.31", "files": {"meta.csv": {}}},
            )
            manifests = MODULE.load_manifests(
                self._args(root / "data" / "manifest.json"), root
            )
            self.assertEqual(set(manifests), set(MODULE.COLLECTIONS))
            self.assertEqual(manifests["core"]["data_version"], "2026.08.14")
            self.assertEqual(list(manifests["core"]["files"]), ["core_a.csv"])
            self.assertEqual(manifests["profiles"]["data_version"], "2026.08.31")
            self.assertEqual(list(manifests["profiles"]["files"]), ["wind.parquet"])
            self.assertEqual(
                manifests["existing_resource_groups"]["data_version"], "2026.08.31"
            )

    def test_missing_local_manifest_falls_back_to_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root,
                "data",
                MANIFEST,
            )
            manifests = MODULE.load_manifests(
                self._args(root / "data" / "manifest.json"), root
            )
            # No resource_profiles/manifest.json -> falls back to sections.
            self.assertEqual(list(manifests["profiles"]["files"]), ["wind.parquet"])
            self.assertEqual(manifests["profiles"]["data_version"], "2026.08.14")
            self.assertEqual(manifests["existing_resource_groups"]["files"], {})

    def test_sections_fallback_defaults_to_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root, "data", {"data_version": "2026.08.14", "files": {"a.csv": {}}}
            )
            manifests = MODULE.load_manifests(
                self._args(root / "data" / "manifest.json"), root
            )
            self.assertEqual(manifests["profiles"]["files"], {})
            self.assertEqual(manifests["existing_resource_groups"]["files"], {})
            self.assertEqual(manifests["profiles"]["data_version"], "2026.08.14")

    def test_per_collection_data_version_is_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(
                root,
                "data",
                {"data_version": "2026.08.14", "files": {"core_a.csv": {}}},
            )
            self._write(
                root,
                "resource_profiles",
                {"data_version": "2026.08.31", "files": {"wind.parquet": {}}},
            )
            manifests = MODULE.load_manifests(
                self._args(root / "data" / "manifest.json"), root
            )
            # Updating profiles must NOT touch the core version.
            self.assertEqual(manifests["core"]["data_version"], "2026.08.14")
            self.assertEqual(manifests["profiles"]["data_version"], "2026.08.31")

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

    def test_readme_body_sits_below_change_note(self):
        import tempfile as _tempfile

        files = MANIFEST["files"]
        with _tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "README.md").write_text(
                "# Collection docs\n\nA table:\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
            )
            readme_html = MODULE.readme_to_html(data_dir)
            description = MODULE.build_description(
                MANIFEST, "core", files, ["core_a.csv"], [], [], False, readme_html
            )
        body = "<h1>Collection docs</h1>" if MODULE.markdown else "<pre>"
        self.assertIn(body, description)
        # README body appears after the change note ("Files added...") ...
        self.assertLess(
            description.index("Files added in this release"),
            description.index("Collection docs"),
        )
        # ... and before the licensing/per-file sections.
        self.assertLess(
            description.index("Collection docs"),
            description.index("Licensing"),
        )

    def test_readme_to_html_empty_without_readme(self):
        import tempfile as _tempfile

        with _tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(MODULE.readme_to_html(Path(tmp)), "")

    def test_release_description_prepends_custom_section_prose(self):
        files = MANIFEST["sections"]["profiles"]["files"]
        base = {
            "description": "<p><strong>Site mapping.</strong> CPA_ID, Site, profile_dist.</p>"
        }
        description = MODULE.build_release_description(
            MANIFEST, "profiles", files, ["wind.parquet"], [], [], True, base
        )
        self.assertTrue(description.startswith("<p><strong>Site mapping.</strong>"))
        self.assertIn("PowerGenome Renewable Resource Profiles", description)

    def test_release_description_without_override_is_unpadded(self):
        files = MANIFEST["files"]
        description = MODULE.build_release_description(
            MANIFEST, "core", files, [], [], [], True, {}
        )
        self.assertTrue(description.startswith("<p>PowerGenome Input Data"))


class ManifestEntryTests(unittest.TestCase):
    def test_real_profiles_manifest_has_six_files(self):
        import json as _json

        real = _json.loads(
            (
                Path(__file__).parents[1] / "resource_profiles" / "manifest.json"
            ).read_text()
        )
        self.assertIn("data_version", real)
        profiles = real["files"]
        self.assertEqual(len(profiles), 6)
        self.assertIn("solar_site_mapping_20240801.parquet", profiles)
        for name, entry in profiles.items():
            self.assertIn("md5", entry)
            self.assertIn("source", entry["sources"][0])
            self.assertEqual(entry["license"], "cc-by-4.0")

    def test_real_existing_resource_groups_manifest_has_fourteen_files(self):
        import json as _json

        real = _json.loads(
            (
                Path(__file__).parents[1] / "existing_resource_groups" / "manifest.json"
            ).read_text()
        )
        self.assertIn("data_version", real)
        self.assertEqual(len(real["files"]), 14)
        self.assertNotIn("existing_osw_profiles.csv", real["files"])
        for name, entry in real["files"].items():
            self.assertIn("md5", entry)
            self.assertIn("source", entry["sources"][0])


def _mkargs(manifest_path, collection=None):
    args = mock.Mock()
    args.manifest = str(manifest_path)
    args.collection = collection
    return args


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
            }
            manifest_path = root / "data" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with mock.patch.object(MODULE.Path, "cwd", return_value=root):
                out = io.StringIO()
                with redirect_stdout(out):
                    MODULE.dry_run(_mkargs(manifest_path))
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
            }
            profiles_manifest = {
                "data_version": "2026.08.31",
                "files": {"wind.parquet": {"version": "2026-08-12", "md5": "y"}},
            }
            (root / "data" / "manifest.json").write_text(json.dumps(manifest))
            (root / "resource_profiles" / "manifest.json").write_text(
                json.dumps(profiles_manifest)
            )
            with mock.patch.object(MODULE.Path, "cwd", return_value=root):
                out = io.StringIO()
                with redirect_stdout(out):
                    MODULE.dry_run(
                        _mkargs(root / "data" / "manifest.json", ["profiles"])
                    )
            text = out.getvalue()
        self.assertNotIn("[core]", text)
        self.assertIn("[profiles] PowerGenome Renewable Resource Profiles", text)
        self.assertIn("[data version 2026.08.31]", text)
        self.assertIn("ok      wind.parquet", text)
        self.assertNotIn("existing_resource_groups", text)

    def test_unknown_collection_rejected_by_parser(self):
        parser = MODULE.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--collection", "bogus"])


if __name__ == "__main__":
    unittest.main()
