import importlib.util
import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, chdir
from copy import deepcopy
from pathlib import Path
from unittest import mock

import requests

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

    def test_legacy_core_preserves_complete_payload_in_either_environment(self):
        for environment in ("sandbox", "production"):
            with self.subTest(environment=environment):
                state = deepcopy(LEGACY_STATE)
                state["zenodo_release"].update(
                    environment=environment,
                    versions=["2026.08.14.v1"],
                    removal_notes={"retired.csv": "superseded"},
                    manifest_meta_hash="hash",
                )
                normalized = MODULE.normalize_release_state(state)
                self.assertEqual(
                    normalized["releases"]["core"]["environments"][environment],
                    state["zenodo_release"],
                )
                self.assertEqual(
                    MODULE.normalize_release_state(normalized), normalized
                )

    def test_new_releases_state_wins_over_legacy(self):
        state = dict(LEGACY_STATE)
        state["releases"] = {"core": {"deposition_id": "700002", "published": False}}
        released = MODULE.section_release(state, "core")
        self.assertEqual(released["deposition_id"], "700002")

    def test_new_state_section_lookup(self):
        released = MODULE.section_release(NEW_STATE, "profiles")
        self.assertEqual(released["deposition_id"], "600001")
        self.assertEqual(MODULE.section_release(NEW_STATE, "core"), {})

    def test_flat_records_migrate_losslessly_and_idempotently(self):
        for environment in ("sandbox", "production"):
            with self.subTest(environment=environment):
                payload = dict(
                    LEGACY_STATE["zenodo_release"],
                    environment=environment,
                    versions=["2026.08.14.v1"],
                    manifest_meta_hash="hash",
                    removal_notes={"old.csv": "superseded"},
                    future_field={"preserve": True},
                )
                state = {"releases": {"core": payload}}
                original = deepcopy(state)
                normalized = MODULE.normalize_release_state(state)
                self.assertEqual(
                    normalized["releases"]["core"],
                    {"environments": {environment: payload}},
                )
                self.assertEqual(
                    MODULE.normalize_release_state(normalized), normalized
                )
                self.assertEqual(state, original)
                self.assertEqual(
                    MODULE.section_release(state, "core", environment), payload
                )
                other = "sandbox" if environment == "production" else "production"
                self.assertEqual(MODULE.section_release(state, "core", other), {})

    def test_missing_environment_defaults_to_sandbox_without_doi_inference(self):
        payload = {"deposition_id": "123", "doi": "10.5281/zenodo.123"}
        state = {"releases": {"core": payload}}
        self.assertEqual(MODULE.section_release(state, "core", "sandbox"), payload)
        self.assertEqual(MODULE.section_release(state, "core", "production"), {})

    def test_nested_key_supplies_missing_payload_environment(self):
        payload = {"deposition_id": "production-id"}
        state = {"releases": {"core": {"environments": {"production": payload}}}}
        self.assertEqual(
            MODULE.section_release(state, "core", "production"), payload
        )
        self.assertEqual(MODULE.section_release(state, "core", "sandbox"), {})

    def test_nested_selection_and_collection_precedence(self):
        state = deepcopy(LEGACY_STATE)
        production = {"environment": "production", "deposition_id": "prod"}
        state["releases"] = {
            "core": {"environments": {"production": production}},
        }
        self.assertEqual(
            MODULE.section_release(state, "core", "production"), production
        )
        # Do not resurrect legacy sandbox state when a core container exists.
        self.assertEqual(MODULE.section_release(state, "core", "sandbox"), {})
        state["releases"]["core"] = {}
        self.assertEqual(MODULE.section_release(state, "core"), {})


class EnvironmentPersistenceTests(unittest.TestCase):
    """Exercise the real release/read/modify/save flow with no network access."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.state_path = self.root / ".zenodo.json"
        self.manifests = {}
        for section in ("core", "profiles"):
            directory = self.root / MODULE.COLLECTIONS[section]["data_dir"]
            directory.mkdir()
            data = directory / "example.csv"
            data.write_text("example\n")
            manifest = {
                "data_version": "2026.08.14",
                "files": {
                    "example.csv": {
                        "md5": MODULE.md5_for_file(data),
                        "version": "v1",
                        "license": "cc-zero",
                        "sources": [{"source": "Test fixture"}],
                    }
                },
            }
            path = directory / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.manifests[section] = path
        self.stack.enter_context(mock.patch.object(Path, "cwd", return_value=self.root))
        # run_release changes these globals; restore them after each test.
        self.stack.enter_context(mock.patch.object(MODULE, "PROJECT_ROOT", self.root))
        self.stack.enter_context(mock.patch.object(MODULE, "STATE_PATH", self.state_path))
        for name in ("load_dotenv", "log"):
            self.stack.enter_context(mock.patch.object(MODULE, name))
        self.stack.enter_context(mock.patch("builtins.print"))
        self.stack.enter_context(
            mock.patch.object(MODULE, "truthy_env", return_value=False)
        )
        self.stack.enter_context(
            mock.patch.object(MODULE, "load_token", return_value="test-token")
        )
        self.session = mock.Mock(spec=requests.Session)
        self.session.request.side_effect = AssertionError("Unexpected network operation")
        self.stack.enter_context(
            mock.patch.object(MODULE, "make_session", return_value=self.session)
        )
        self.stack.enter_context(
            mock.patch.object(MODULE, "uncommitted_release_files", return_value=[])
        )
        self.stack.enter_context(
            mock.patch.object(
                MODULE,
                "check_script_provenance",
                return_value={"tag": "v1.0.0", "scripts": {}, "files": {}},
            )
        )
        self.remote = {}
        self.next_id = 100
        for name in ("create_deposition", "create_new_version", "get_deposition"):
            self.remote[name] = self.stack.enter_context(
                mock.patch.object(
                    MODULE,
                    name,
                    side_effect=lambda *a, operation=name: self.deposition(operation, *a),
                )
            )
        for name, result in (
            ("existing_draft_files", {}),
            ("upload_file", {}),
            ("set_metadata", {}),
        ):
            self.remote[name] = self.stack.enter_context(
                mock.patch.object(
                    MODULE, name, side_effect=lambda *a, r=result, **k: deepcopy(r)
                )
            )
        self.remote["publish_deposition"] = self.stack.enter_context(
            mock.patch.object(
                MODULE,
                "publish_deposition",
                side_effect=lambda session, base, dep: {"doi": f"test:{base}/{dep}"},
            )
        )
        # Keep the actual suffix calculation; only mock its remote search.
        response = mock.Mock()
        response.json.return_value = {"hits": {"hits": []}}
        self.stack.enter_context(
            mock.patch.object(MODULE, "retry_request", return_value=response)
        )

    def deposition(self, operation, session, base_url, deposition_id=None):
        if operation != "get_deposition":
            self.next_id += 1
            deposition_id = str(self.next_id)
        return {
            "id": deposition_id,
            "conceptrecid": "concept",
            "links": {"bucket": "https://bucket.invalid", "files": "https://files.invalid"},
        }

    def payload(self, environment, published=True):
        return {
            "environment": environment,
            "data_version": "2026.08.13",
            "release_version": "2026.08.14.v2",
            "deposition_id": f"{environment}-original",
            "doi": f"test:{environment}",
            "published": published,
            "versions": ["2026.08.14.v1", "2026.08.14.v2"],
            "files": {"old.csv": {"md5": "old", "license": "cc-zero"}},
            "manifest_meta_hash": "old-hash",
            "removal_notes": {"old.csv": "retired"},
            "extra": {"must_survive": [1, 2]},
        }

    def write_state(self, state):
        self.state_path.write_text(json.dumps(state))

    def read_state(self):
        return json.loads(self.state_path.read_text())

    def run_release(self, environment, *, publish=True, save=True, section="core"):
        flags = ["--collection", section, "--sleep-seconds", "0"]
        if environment == "production":
            flags.append("--production")
        if publish:
            flags.append("--publish")
        if save:
            flags.append("--save-state")
        MODULE.run_release(MODULE.build_parser().parse_args(flags))
        self.session.request.assert_not_called()
        return self.read_state()

    def test_removal_notes_reach_description_but_not_zenodo_metadata(self):
        metadata = deepcopy(NEW_STATE["metadata"])
        metadata.setdefault("sections", {}).setdefault("core", {})["removal_notes"] = {
            "old.csv": "Replaced by example.csv."
        }
        original = self.payload("production")
        original.pop("removal_notes")
        self.write_state({"metadata": metadata, "releases": {"core": original}})
        self.run_release("production", publish=False, save=False)
        sent = self.remote["set_metadata"].call_args.args[3]
        self.assertNotIn("removal_notes", sent)
        self.assertIn("Replaced by example.csv.", sent["description"])

    def test_alternating_environments_preserve_lineage_and_independent_histories(self):
        for original_env, target_env in (
            ("production", "sandbox"),
            ("sandbox", "production"),
        ):
            with self.subTest(original=original_env):
                original = self.payload(original_env)
                untouched = {"environments": {"production": self.payload("production")}}
                metadata = deepcopy(NEW_STATE["metadata"])
                self.write_state({
                    "metadata": metadata,
                    "releases": {"core": original, "profiles": untouched},
                })
                self.remote["create_deposition"].reset_mock()
                saved = self.run_release(target_env)
                environments = saved["releases"]["core"]["environments"]
                self.assertEqual(environments[original_env], original)
                self.assertEqual(environments[target_env]["versions"], ["2026.08.14.v1"])
                self.assertEqual(saved["releases"]["profiles"], untouched)
                self.assertEqual(saved["metadata"], metadata)
                target = deepcopy(environments[target_env])
                self.remote["create_deposition"].assert_called_once_with(
                    self.session, MODULE.resolve_base(target_env == "production")
                )

                self.remote["create_new_version"].reset_mock()
                saved = self.run_release(original_env)
                environments = saved["releases"]["core"]["environments"]
                self.remote["create_new_version"].assert_called_once_with(
                    self.session,
                    MODULE.resolve_base(original_env == "production"),
                    original["deposition_id"],
                )
                self.assertEqual(environments[target_env], target)
                self.assertEqual(
                    environments[original_env]["versions"],
                    original["versions"] + ["2026.08.14.v3"],
                )
                self.assertEqual(saved["releases"]["profiles"], untouched)
                self.assertEqual(saved["metadata"], metadata)

                # Returning to the other published environment is a no-op.
                before = deepcopy(saved)
                self.remote["create_new_version"].reset_mock()
                self.assertEqual(self.run_release(target_env), before)
                self.remote["create_new_version"].assert_not_called()
                self.assertEqual(self.run_release(original_env), before)
                self.remote["create_new_version"].assert_not_called()

                # A same-day metadata change versions each saved lineage with
                # its own counter, never the other environment's history.
                path = self.manifests["core"]
                manifest = json.loads(path.read_text())
                manifest["files"]["example.csv"]["version"] += "-updated"
                path.write_text(json.dumps(manifest))
                self.remote["create_new_version"].reset_mock()
                saved = self.run_release(target_env)
                environments = saved["releases"]["core"]["environments"]
                self.remote["create_new_version"].assert_called_once_with(
                    self.session,
                    MODULE.resolve_base(target_env == "production"),
                    target["deposition_id"],
                )
                self.assertEqual(
                    environments[original_env],
                    before["releases"]["core"]["environments"][original_env],
                )
                self.assertEqual(
                    environments[target_env]["versions"],
                    ["2026.08.14.v1", "2026.08.14.v2"],
                )
                target = deepcopy(environments[target_env])
                saved = self.run_release(original_env)
                environments = saved["releases"]["core"]["environments"]
                self.assertEqual(environments[target_env], target)
                self.assertEqual(
                    environments[original_env]["versions"],
                    original["versions"] + ["2026.08.14.v3", "2026.08.14.v4"],
                )

    def test_saved_sandbox_draft_resumes_after_production_release(self):
        production = self.payload("production")
        self.write_state({
            "metadata": NEW_STATE["metadata"],
            "releases": {"core": production},
        })
        saved = self.run_release("sandbox", publish=False)
        draft = saved["releases"]["core"]["environments"]["sandbox"]
        self.assertFalse(draft["published"])
        self.assertEqual(draft["versions"], [])
        self.assertEqual(
            saved["releases"]["core"]["environments"]["production"], production
        )
        saved = self.run_release("production")
        production = deepcopy(saved["releases"]["core"]["environments"]["production"])
        self.assertEqual(saved["releases"]["core"]["environments"]["sandbox"], draft)
        self.remote["get_deposition"].reset_mock()
        saved = self.run_release("sandbox", publish=False)
        self.remote["get_deposition"].assert_called_once_with(
            self.session, MODULE.resolve_base(False), draft["deposition_id"]
        )
        self.assertEqual(saved["releases"]["core"]["environments"]["sandbox"], draft)
        self.assertEqual(
            saved["releases"]["core"]["environments"]["production"], production
        )

    def test_non_core_save_preserves_legacy_core_and_reads_do_not_write(self):
        self.write_state(LEGACY_STATE)
        before = self.state_path.read_bytes()
        normalized = MODULE.load_state()
        self.assertEqual(self.state_path.read_bytes(), before)
        self.assertEqual(
            normalized["releases"]["core"]["environments"]["sandbox"],
            LEGACY_STATE["zenodo_release"],
        )
        self.run_release("sandbox", publish=False, save=False, section="profiles")
        self.assertEqual(self.state_path.read_bytes(), before)
        saved = self.run_release("sandbox", publish=False, section="profiles")
        self.assertEqual(
            saved["releases"]["core"]["environments"]["sandbox"],
            LEGACY_STATE["zenodo_release"],
        )
        self.assertNotIn("zenodo_release", saved)
        self.assertEqual(MODULE.load_state(), saved)

    def test_invalid_state_refused_before_any_remote_mutation_or_save(self):
        invalid_records = [
            None,
            [],
            {"environment": "staging"},
            {"environment": None},
            {"environments": None},
            {"environments": []},
            {"environments": {"staging": {}}},
            {"environments": {"sandbox": None}},
            {"environments": {"production": {"environment": "sandbox"}}},
            {"environments": {"sandbox": {"environment": None}}},
            {"environments": {}, "deposition_id": "ambiguous"},
            {"environments": {"sandbox": {"environments": {}}}},
        ]
        states = [
            {"releases": {"core": self.payload("production"), "profiles": record}}
            for record in invalid_records
        ] + [
            [],
            {"releases": None},
            {"zenodo_release": None},
        ]
        for state in states:
            with self.subTest(state=state):
                self.write_state(state)
                before = self.state_path.read_bytes()
                with self.assertRaisesRegex(SystemExit, "Repair .zenodo.json"):
                    self.run_release("sandbox")
                self.assertEqual(self.state_path.read_bytes(), before)
                for remote in self.remote.values():
                    remote.assert_not_called()
                self.session.request.assert_not_called()


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


class CreatorResolutionTests(unittest.TestCase):
    def test_default_creators_from_git_user_name(self):
        with mock.patch.object(MODULE, "git_user_name", return_value="Greg Schivley"):
            self.assertEqual(MODULE.default_creators(), [{"name": "Schivley, Greg"}])

    def test_default_creators_exits_without_git_user_name(self):
        with mock.patch.object(MODULE, "git_user_name", return_value=""):
            with self.assertRaises(SystemExit):
                MODULE.default_creators()


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

    def test_updated_file_shows_last_released_version(self):
        info = {
            "version": "2026-11-24",
            "md5": "new",
            "history": [
                {"version": "2026-08-31", "md5": "released"},
                {"version": "2026-09-24", "md5": "never-released"},
            ],
        }
        self.assertEqual(
            MODULE.previous_released_version(info, "released"), "2026-08-31"
        )
        # Unknown released md5 falls back to the latest history entry.
        self.assertEqual(MODULE.previous_released_version(info, None), "2026-09-24")
        manifest = {"data_version": "2026.11.24", "files": {"a.parquet": info}}
        description = MODULE.build_description(
            manifest,
            "core",
            manifest["files"],
            [],
            ["a.parquet"],
            [],
            False,
            released_files={"a.parquet": {"md5": "released"}},
        )
        self.assertIn("2026-08-31 &rarr; 2026-11-24", description)

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

    def test_updated_file_shows_version_transition(self):
        files = {
            "wind.parquet": {
                "version": "2026-08-20",
                "last_updated": "2026-08-20",
                "md5": "new",
                "history": [{"version": "2026-08-12", "md5": "old"}],
            }
        }
        description = MODULE.build_description(
            {"data_version": "2026.08.20", "files": files},
            "profiles",
            files,
            [],
            ["wind.parquet"],
            [],
            False,
        )
        self.assertIn("2026-08-12 &rarr; 2026-08-20", description)

    def test_removed_file_retains_last_release_details_and_note(self):
        details = {
            "gone.csv": {
                "version": "2026-07-01",
                "license": "cc-by-4.0",
                "sources": [
                    {"source": "Prior source", "source_url": "https://example.com"}
                ],
                "md5": "old",
            }
        }
        description = MODULE.build_description(
            {"data_version": "2026.08.20", "files": {}},
            "core",
            {},
            [],
            [],
            ["gone.csv"],
            False,
            removed_details=details,
            removal_notes={"gone.csv": "Superseded by replacement.csv"},
        )
        self.assertIn("Removed files, last released state", description)
        self.assertIn("2026-07-01", description)
        self.assertIn("Creative Commons Attribution", description)
        self.assertIn("Prior source", description)
        self.assertIn("Superseded by replacement.csv", description)

    def test_metadata_hash_covers_sources_license_and_readme(self):
        manifest = {
            "data_version": "2026.08.20",
            "files": {"a.csv": {"md5": "same", "version": "v1", "sources": []}},
        }
        original = MODULE.manifest_meta_hash(manifest, "<p>README one</p>")
        manifest["files"]["a.csv"]["license"] = "cc-zero"
        self.assertNotEqual(
            original, MODULE.manifest_meta_hash(manifest, "<p>README one</p>")
        )
        self.assertNotEqual(
            MODULE.manifest_meta_hash(manifest, "<p>README one</p>"),
            MODULE.manifest_meta_hash(manifest, "<p>README two</p>"),
        )

    def test_released_file_state_normalizes_legacy_and_preserves_details(self):
        normalized = MODULE.normalize_released_files(
            {"old.csv": "abc", "new.csv": {"md5": "def", "version": "2026-08-01"}}
        )
        self.assertEqual(normalized["old.csv"], {"md5": "abc"})
        self.assertEqual(normalized["new.csv"]["version"], "2026-08-01")


class ScriptProvenanceTests(unittest.TestCase):
    def _init_repo(self, root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=root, check=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
        subprocess.run(
            ["git", "config", "commit.gpgsign", "false"], cwd=root, check=True
        )

    def _commit_all(self, root: Path, message: str) -> None:
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True)

    def test_referenced_scripts_extracts_bare_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "build_atb_data.py").write_text("print('x')\n")
            self._commit_all(root, "add script")
            files = {
                "costs.csv": {
                    "sources": [{"source": "rebuilt by build_atb_data.py."}]
                }
            }
            self.assertEqual(
                MODULE.referenced_scripts(files, root),
                {"costs.csv": ["build_atb_data.py"]},
            )

    def test_referenced_scripts_resolves_directory_to_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            script = (
                root
                / ".github"
                / "skills"
                / "eia-fuel-prices"
                / "scripts"
                / "fetch_eia_fuel_prices.py"
            )
            script.parent.mkdir(parents=True)
            script.write_text("print('x')\n")
            test_script = (
                root
                / ".github"
                / "skills"
                / "eia-fuel-prices"
                / "tests"
                / "test_fetch_eia_fuel_prices.py"
            )
            test_script.parent.mkdir(parents=True)
            test_script.write_text("print('test')\n")
            self._commit_all(root, "add skill")
            files = {
                "fuel_prices.parquet": {
                    "sources": [{"source": "mapped by .github/skills/eia-fuel-prices."}]
                }
            }
            self.assertEqual(
                MODULE.referenced_scripts(files, root),
                {
                    "fuel_prices.parquet": [
                        ".github/skills/eia-fuel-prices/scripts/fetch_eia_fuel_prices.py"
                    ]
                },
            )

    def test_referenced_scripts_ignores_upstream_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "transform_reeds_generators.py").write_text("print('x')\n")
            self._commit_all(root, "add script")
            files = {
                "gen.csv": {
                    "sources": [
                        {
                            "source": (
                                "ReEDS-2.0/inputs/capacity_exogenous/"
                                "ReEDS_generator_database_final_EIA-NEMS.csv joined by "
                                "transform_reeds_generators.py."
                            )
                        }
                    ]
                }
            }
            self.assertEqual(
                MODULE.referenced_scripts(files, root),
                {"gen.csv": ["transform_reeds_generators.py"]},
            )

    def test_referenced_scripts_prefers_declared_list_over_prose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "a.py").write_text("print('a')\n")
            (root / "b.py").write_text("print('b')\n")
            self._commit_all(root, "add scripts")
            files = {
                "x.csv": {
                    "scripts": ["b.py"],
                    "sources": [{"source": "legacy prose naming a.py."}],
                }
            }
            self.assertEqual(MODULE.referenced_scripts(files, root), {"x.csv": ["b.py"]})

    def test_referenced_scripts_falls_back_to_prose_when_undeclared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "a.py").write_text("print('a')\n")
            self._commit_all(root, "add script")
            for scripts in (None, [], [""]):
                info = {"sources": [{"source": "built by a.py."}]}
                if scripts is not None:
                    info["scripts"] = scripts
                self.assertEqual(
                    MODULE.referenced_scripts({"x.csv": info}, root),
                    {"x.csv": ["a.py"]},
                )

    def test_referenced_scripts_resolves_declared_basename_to_tracked_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            script = root / "sub" / "build_x.py"
            script.parent.mkdir()
            script.write_text("print('x')\n")
            self._commit_all(root, "add script")
            files = {"x.csv": {"scripts": ["build_x.py"]}}
            self.assertEqual(
                MODULE.referenced_scripts(files, root), {"x.csv": ["sub/build_x.py"]}
            )

    def test_unresolved_declared_scripts_reports_unknown_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "a.py").write_text("print('a')\n")
            self._commit_all(root, "add script")
            files = {
                "x.csv": {"scripts": ["typo.py"]},
                "y.csv": {"scripts": ["a.py"]},
            }
            self.assertEqual(
                MODULE.unresolved_declared_scripts(files, root), {"x.csv": ["typo.py"]}
            )

    def test_unresolvable_declaration_does_not_fall_back_to_prose(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "a.py").write_text("print('a')\n")
            self._commit_all(root, "add script")
            files = {
                "x.csv": {
                    "scripts": ["typo.py"],
                    "sources": [{"source": "built by a.py."}],
                }
            }
            self.assertEqual(MODULE.referenced_scripts(files, root), {})
            self.assertEqual(
                MODULE.unresolved_declared_scripts(files, root), {"x.csv": ["typo.py"]}
            )

    def test_check_script_provenance_records_missing_declarations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "a.py").write_text("print('a')\n")
            self._commit_all(root, "first")
            subprocess.run(["git", "tag", "v1.0.0"], cwd=root, check=True)
            files = {"x.csv": {"scripts": ["typo.py"]}}
            provenance = MODULE.check_script_provenance(files, root)
            self.assertEqual(provenance["missing"], {"x.csv": ["typo.py"]})
            self.assertIsNotNone(MODULE.provenance_block_reason(provenance, False))
            self.assertIsNone(MODULE.provenance_block_reason(provenance, True))

    def test_latest_reachable_tag_returns_nearest_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "a.py").write_text("print('a')\n")
            self._commit_all(root, "first")
            subprocess.run(["git", "tag", "v1.0.0"], cwd=root, check=True)
            (root / "a.py").write_text("print('b')\n")
            self._commit_all(root, "second")
            subprocess.run(["git", "tag", "v1.1.0"], cwd=root, check=True)
            self.assertEqual(MODULE.latest_reachable_tag(root), "v1.1.0")

    def test_latest_reachable_tag_none_without_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "a.py").write_text("print('a')\n")
            self._commit_all(root, "first")
            self.assertIsNone(MODULE.latest_reachable_tag(root))

    def test_script_changed_since_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "a.py").write_text("print('a')\n")
            self._commit_all(root, "first")
            subprocess.run(["git", "tag", "v1.0.0"], cwd=root, check=True)
            self.assertFalse(MODULE.script_changed_since_tag("a.py", "v1.0.0", root))
            (root / "a.py").write_text("print('b')\n")
            self._commit_all(root, "second")
            self.assertTrue(MODULE.script_changed_since_tag("a.py", "v1.0.0", root))

    def test_script_changed_since_tag_detects_staged_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            script = root / "a.py"
            script.write_text("print('a')\n")
            self._commit_all(root, "first")
            subprocess.run(["git", "tag", "v1.0.0"], cwd=root, check=True)
            script.write_text("print('b')\n")
            subprocess.run(["git", "add", "a.py"], cwd=root, check=True)
            script.write_text("print('a')\n")
            self.assertTrue(MODULE.script_changed_since_tag("a.py", "v1.0.0", root))

    def test_check_script_provenance_missing_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "a.py").write_text("print('a')\n")
            self._commit_all(root, "first")
            files = {"x.csv": {"sources": [{"source": "by a.py."}]}}
            provenance = MODULE.check_script_provenance(files, root)
            self.assertIsNone(provenance["tag"])
            self.assertFalse(provenance["scripts"]["a.py"]["changed"])

    def test_provenance_block_reason(self):
        ok = {"tag": "v1.0.0", "scripts": {"a.py": {"changed": False}}, "files": {}}
        self.assertIsNone(MODULE.provenance_block_reason(ok, False))
        drifted = {"tag": "v1.0.0", "scripts": {"a.py": {"changed": True}}, "files": {}}
        self.assertIsNotNone(MODULE.provenance_block_reason(drifted, False))
        self.assertIsNone(MODULE.provenance_block_reason(drifted, True))
        no_tag = {"tag": None, "scripts": {}, "files": {}}
        self.assertIsNotNone(MODULE.provenance_block_reason(no_tag, True))

    def test_release_section_blocks_without_tag(self):
        args = mock.Mock()
        args.deposition_id = None
        args.allow_dirty = True
        args.allow_script_drift = False
        args.publish = False
        args.sleep_seconds = 0
        args.upload_retries = 0
        args.upload_retry_delay = 0
        session = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "core_a.csv").write_text("a")
            manifest = {
                "data_version": "2026.08.14",
                "files": {
                    "core_a.csv": {
                        "version": "2026-08-11",
                        "md5": MODULE.md5_for_file(data_dir / "core_a.csv"),
                    }
                },
            }
            with self.assertRaisesRegex(SystemExit, "no git tag"):
                MODULE.release_section(
                    args,
                    session,
                    "https://zenodo.org/api",
                    "sandbox",
                    manifest,
                    "core",
                    manifest["files"],
                    data_dir,
                    root,
                    {},
                )
        session.request.assert_not_called()

    def test_hydro_builder_drift_is_checked_for_both_profiles(self):
        manifest = json.loads(
            (SCRIPT.parent / "existing_resource_groups" / "manifest.json").read_text()
        )
        files = {
            name: manifest["files"][name]
            for name in (
                "hydro_conventional_2007_2013.parquet",
                "hydro_run_of_river_2007_2013.parquet",
            )
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_repo(root)
            (root / "build_hydro_profiles.py").write_text("original\n")
            (root / "transform_reeds_generators.py").write_text("original\n")
            self._commit_all(root, "Initial scripts")
            subprocess.run(["git", "tag", "v1.0.0"], cwd=root, check=True)

            (root / "build_hydro_profiles.py").write_text("changed\n")
            provenance = MODULE.check_script_provenance(files, root)

        for filename in files:
            self.assertIn("build_hydro_profiles.py", provenance["files"][filename])
        self.assertTrue(provenance["scripts"]["build_hydro_profiles.py"]["changed"])
        self.assertIsNotNone(MODULE.provenance_block_reason(provenance, False))

    def test_description_includes_code_tag_and_scripts(self):
        files = {
            "costs.csv": {
                "version": "2026-08-11",
                "last_updated": "2026-08-11",
                "md5": "x",
                "license": "cc-zero",
                "sources": [{"source": "rebuilt by build_atb_data.py."}],
            }
        }
        provenance = {
            "tag": "v1.2.0",
            "scripts": {"build_atb_data.py": {"changed": False}},
            "files": {"costs.csv": ["build_atb_data.py"]},
        }
        description = MODULE.build_description(
            {"data_version": "2026.08.14", "files": files},
            "core",
            files,
            ["costs.csv"],
            [],
            [],
            True,
            provenance=provenance,
        )
        self.assertIn("code tag <code>v1.2.0</code>", description)
        self.assertIn("build_atb_data.py</code> @ <code>v1.2.0</code>", description)
        self.assertNotIn("changed since tag", description)
        self.assertIn("Files grouped by license:", description)
        self.assertIn("<strong>CC0 (public domain dedication):</strong>", description)
        self.assertNotIn("Last updated", description)

    def test_description_marks_changed_scripts(self):
        files = {
            "costs.csv": {
                "version": "2026-08-11",
                "last_updated": "2026-08-11",
                "md5": "x",
                "sources": [{"source": "rebuilt by build_atb_data.py."}],
            }
        }
        provenance = {
            "tag": "v1.2.0",
            "scripts": {"build_atb_data.py": {"changed": True}},
            "files": {"costs.csv": ["build_atb_data.py"]},
        }
        description = MODULE.build_description(
            {"data_version": "2026.08.14", "files": files},
            "core",
            files,
            ["costs.csv"],
            [],
            [],
            True,
            provenance=provenance,
        )
        self.assertIn("changed since tag", description)
        self.assertIn("have changed since", description)

    def test_manifest_meta_hash_includes_provenance(self):
        manifest = {"data_version": "2026.08.20", "files": {"a.csv": {"md5": "x"}}}
        base = MODULE.manifest_meta_hash(
            manifest, "", {"tag": "v1.0.0", "scripts": {}, "files": {}}
        )
        changed = MODULE.manifest_meta_hash(
            manifest, "", {"tag": "v1.1.0", "scripts": {}, "files": {}}
        )
        self.assertNotEqual(base, changed)

    def test_manifest_meta_hash_includes_declared_scripts(self):
        provenance = {"tag": "v1.0.0", "scripts": {}, "files": {}}
        before = MODULE.manifest_meta_hash(
            {"data_version": "2026.08.20", "files": {"a.csv": {"md5": "x"}}},
            "",
            provenance,
        )
        after = MODULE.manifest_meta_hash(
            {
                "data_version": "2026.08.20",
                "files": {"a.csv": {"md5": "x", "scripts": ["a.py"]}},
            },
            "",
            provenance,
        )
        self.assertNotEqual(before, after)

    def test_description_notes_unverified_script_paths(self):
        files = {
            "costs.csv": {
                "version": "2026-08-11",
                "last_updated": "2026-08-11",
                "md5": "x",
                "scripts": ["typo.py"],
            }
        }
        provenance = {
            "tag": "v1.2.0",
            "scripts": {},
            "files": {},
            "missing": {"costs.csv": ["typo.py"]},
        }
        description = MODULE.build_description(
            {"data_version": "2026.08.14", "files": files},
            "core",
            files,
            ["costs.csv"],
            [],
            [],
            True,
            provenance=provenance,
        )
        self.assertIn("not found in the repository", description)
        self.assertIn("<code>typo.py</code>", description)

    def test_parser_exposes_allow_script_drift(self):
        parser = MODULE.build_parser()
        args = parser.parse_args(["--allow-script-drift"])
        self.assertTrue(args.allow_script_drift)


class ReleaseSafetyTests(unittest.TestCase):
    def test_undocumented_files_flags_missing_license(self):
        files = {
            "a.csv": {"license": "cc-zero", "sources": [{"source": "EIA"}]},
            "b.csv": {"sources": [{"source": "EIA"}]},
        }
        self.assertEqual(MODULE.undocumented_files(files), ["b.csv"])

    def test_undocumented_files_flags_placeholder_sources(self):
        files = {
            "a.csv": {
                "license": "cc-zero",
                "sources": [{"source": "Unknown - document me"}],
            },
            "b.csv": {"license": "cc-zero", "sources": []},
            "c.csv": {"license": "cc-zero", "sources": [{"source": "EIA"}]},
        }
        self.assertEqual(MODULE.undocumented_files(files), ["a.csv", "b.csv"])

    def test_undocumented_files_ok_when_documented(self):
        files = {
            "a.csv": {"license": "cc-zero", "sources": [{"source": "EIA"}]},
        }
        self.assertEqual(MODULE.undocumented_files(files), [])

    def test_production_release_guards_are_independent(self):
        args = mock.Mock()
        args.deposition_id = None
        args.allow_dirty = True
        args.allow_undocumented = False
        args.allow_script_drift = False
        args.publish = False
        args.sleep_seconds = 0
        args.upload_retries = 0
        args.upload_retry_delay = 0
        session = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "core_a.csv").write_text("a")
            manifest = {
                "data_version": "2026.08.14",
                "files": {
                    "core_a.csv": {
                        "version": "2026-08-11",
                        "md5": MODULE.md5_for_file(data_dir / "core_a.csv"),
                    }
                },
            }
            for allow_undocumented, reason in (
                (False, "Refusing production release"),
                (True, "no git tag"),
            ):
                with self.subTest(allow_undocumented=allow_undocumented):
                    args.allow_undocumented = allow_undocumented
                    with self.assertRaisesRegex(SystemExit, reason):
                        MODULE.release_section(
                            args,
                            session,
                            "https://zenodo.org/api",
                            "production",
                            manifest,
                            "core",
                            manifest["files"],
                            data_dir,
                            root,
                            {},
                        )
        # Refusal happens before any Zenodo API call.
        session.request.assert_not_called()

    def test_parser_exposes_allow_undocumented(self):
        parser = MODULE.build_parser()
        args = parser.parse_args(["--allow-undocumented"])
        self.assertTrue(args.allow_undocumented)
        self.assertFalse(parser.parse_args([]).allow_undocumented)


class LicensingAndDescriptionTests(unittest.TestCase):
    def test_licensing_paragraph_groups_files_by_license(self):
        files = {
            "a.csv": {"license": "cc-zero"},
            "b.csv": {"license": "cc-by-4.0"},
            "c.csv": {"license": "cc-zero"},
        }
        paragraph = MODULE.licensing_paragraph(files)
        self.assertIn(
            "Creative Commons Zero (CC0, public domain dedication)", paragraph
        )
        self.assertIn("CC0 (public domain dedication)", paragraph)
        self.assertIn(
            "Creative Commons Attribution 4.0 International (CC BY 4.0)", paragraph
        )
        self.assertIn("<code>a.csv</code>", paragraph)
        self.assertIn("<code>c.csv</code>", paragraph)
        self.assertIn("<code>b.csv</code>", paragraph)

    def test_licensing_paragraph_handles_missing_license(self):
        paragraph = MODULE.licensing_paragraph({"a.csv": {}})
        self.assertIn("Not specified", paragraph)

    def test_describe_file_omits_redundant_last_updated_row(self):
        html = MODULE.describe_file(
            "a.csv", {"version": "2026-08-11", "last_updated": "2026-08-11"}
        )
        self.assertNotIn("Last updated", html)

    def test_describe_file_shows_last_updated_when_divergent(self):
        html = MODULE.describe_file(
            "a.csv", {"version": "2026-08-11", "last_updated": "2026-08-12"}
        )
        self.assertIn("Last updated", html)
        self.assertIn("2026-08-12", html)


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

    def test_real_existing_resource_groups_manifest_has_fifteen_files(self):
        import json as _json

        real = _json.loads(
            (
                Path(__file__).parents[1] / "existing_resource_groups" / "manifest.json"
            ).read_text()
        )
        self.assertIn("data_version", real)
        self.assertEqual(len(real["files"]), 15)
        self.assertNotIn("existing_hydro_reeds_ba_metadata.csv", real["files"])
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
                with mock.patch.object(
                    MODULE,
                    "check_script_provenance",
                    return_value={"tag": "v1.0.0", "scripts": {}, "files": {}},
                ):
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
        self.assertIn("ships with deposit: manifest.json", text)

    def test_dry_run_warns_when_a_collection_manifest_is_missing(self):
        """Legacy layout: files come from the core manifest's ``sections``, so
        the collection has no manifest of its own to ship."""
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "resource_profiles").mkdir()
            (root / "resource_profiles" / "wind.parquet").write_text("w")
            manifest = {
                "data_version": "2026.08.14",
                "files": {},
                "sections": {
                    "profiles": {
                        "files": {"wind.parquet": {"version": "2026-08-12", "md5": "y"}}
                    }
                },
            }
            manifest_path = root / "data" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with (
                mock.patch.object(MODULE.Path, "cwd", return_value=root),
                mock.patch.object(
                    MODULE,
                    "check_script_provenance",
                    return_value={"tag": "v1.0.0", "scripts": {}, "files": {}},
                ),
            ):
                out = io.StringIO()
                with redirect_stdout(out):
                    MODULE.dry_run(_mkargs(manifest_path, ["profiles"]))
            text = out.getvalue()
        self.assertIn("no per-file provenance", text)


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
                with mock.patch.object(
                    MODULE,
                    "check_script_provenance",
                    return_value={"tag": "v1.0.0", "scripts": {}, "files": {}},
                ):
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


class DepositionOverrideTests(unittest.TestCase):
    def test_parser_exposes_deposition_id(self):
        parser = MODULE.build_parser()
        args = parser.parse_args(["--deposition-id", "22233228"])
        self.assertEqual(args.deposition_id, "22233228")

    def test_deposition_id_overrides_saved_state(self):
        """--deposition-id must resume the given draft even when saved state
        points at a different (or published) deposition."""
        args = mock.Mock()
        args.deposition_id = "22233228"
        args.allow_dirty = True
        args.publish = False
        args.sleep_seconds = 0
        args.upload_retries = 0
        args.upload_retry_delay = 0

        session = mock.Mock()
        # get_deposition for the resumed draft
        draft = {
            "id": 22233228,
            "conceptrecid": 111,
            "links": {
                "bucket": "https://bucket.example",
                "files": "https://files.example",
            },
        }
        draft_response = self._response(200)
        draft_response.json.return_value = draft
        records_response = self._response(200)
        records_response.json.return_value = {"hits": {"hits": []}}
        files_response = self._response(200)
        files_response.json.return_value = []

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "core_a.csv").write_text("a")
            upload_response = self._response(200)
            upload_response.json.return_value = {
                "checksum": f"md5:{MODULE.md5_for_file(data_dir / 'core_a.csv')}"
            }
            metadata_response = self._response(200)
            metadata_response.json.return_value = {"metadata": {}}
            session.request.side_effect = [
                draft_response,
                records_response,
                files_response,
                upload_response,
                metadata_response,
            ]
            manifest = {
                "data_version": "2026.08.14",
                "files": {
                    "core_a.csv": {
                        "version": "2026-08-11",
                        "md5": MODULE.md5_for_file(data_dir / "core_a.csv"),
                    }
                },
            }
            manifest_path = root / "data" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            state = {
                "metadata": {"creators": [{"name": "Schivley, Greg"}]},
                "releases": {
                    "core": {
                        "environment": "production",
                        "deposition_id": "999999",
                        "published": True,
                    }
                },
            }
            with mock.patch.object(MODULE.time, "sleep"):
                with mock.patch.object(
                    MODULE,
                    "check_script_provenance",
                    return_value={"tag": "v1.0.0", "scripts": {}, "files": {}},
                ):
                    result = MODULE.release_section(
                        args,
                        session,
                        "https://zenodo.org/api",
                        "production",
                        manifest,
                        "core",
                        manifest["files"],
                        data_dir,
                        root,
                        state,
                    )
        self.assertEqual(result["summary"]["deposition_id"], "22233228")
        # The resumed draft must be fetched (not a new version created), then
        # records search, draft files, upload, and metadata update follow.
        self.assertEqual(session.request.call_count, 5)

    def _response(self, status=200, checksum="md5:abc"):
        response = mock.Mock()
        response.status_code = status
        response.raise_for_status.side_effect = (
            None if status < 400 else requests.HTTPError(f"HTTP {status}")
        )
        response.json.return_value = {"checksum": checksum}
        return response


class ManifestShippedWithDepositTests(unittest.TestCase):
    """Each deposit carries its own ``manifest.json``.

    Provenance (per-file data version, license, upstream sources) only reaches
    a user if the manifest travels with the data; PowerGenome's
    ``download_zenodo`` reads the copy inside the deposit before falling back to
    GitHub.
    """

    def _args(self):
        return MODULE.build_parser().parse_args(
            ["--deposition-id", "22233228", "--allow-dirty", "--sleep-seconds", "0"]
        )

    def setUp(self):
        provenance = mock.patch.object(
            MODULE,
            "check_script_provenance",
            return_value={"tag": "v1.0.0", "scripts": {}, "files": {}},
        )
        provenance.start()
        self.addCleanup(provenance.stop)

    def _response(self, status=200, payload=None):
        response = mock.Mock()
        response.status_code = status
        response.raise_for_status.side_effect = (
            None if status < 400 else requests.HTTPError(f"HTTP {status}")
        )
        response.json.return_value = payload if payload is not None else {}
        return response

    def _setup(self, root, manifest_name="manifest.json"):
        data_dir = root / "data"
        data_dir.mkdir()
        (data_dir / "core_a.csv").write_text("a")
        manifest = {
            "data_version": "2026.08.14",
            "files": {
                "core_a.csv": {
                    "version": "2026-08-11",
                    "md5": MODULE.md5_for_file(data_dir / "core_a.csv"),
                    "license": "cc-zero",
                    "sources": [{"source": "Test fixture"}],
                }
            },
        }
        manifest_path = data_dir / manifest_name
        manifest_path.write_text(json.dumps(manifest))
        return manifest, manifest_path, data_dir

    def _draft(self):
        return {
            "id": 22233228,
            "conceptrecid": 111,
            "links": {
                "bucket": "https://bucket.example",
                "files": "https://files.example",
            },
        }

    def _state(self):
        return {
            "metadata": {"creators": [{"name": "Schivley, Greg"}]},
            "releases": {
                "core": {
                    "environment": "production",
                    "deposition_id": "999999",
                    "published": True,
                }
            },
        }

    def _run(self, draft_files, manifest_name="manifest.json"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, manifest_path, data_dir = self._setup(root, manifest_name)
            manifest_md5 = MODULE.md5_for_file(manifest_path)
            data_md5 = MODULE.md5_for_file(data_dir / "core_a.csv")

            responses = [
                self._response(payload=self._draft()),
                self._response(payload={"hits": {"hits": []}}),
                self._response(payload=draft_files),
            ]
            for key, md5 in (("core_a.csv", data_md5), ("manifest.json", manifest_md5)):
                if key not in {item["key"] for item in draft_files}:
                    responses.append(self._response(payload={"checksum": f"md5:{md5}"}))
            responses.append(self._response(payload={"metadata": {}}))

            session = mock.Mock()
            session.request.side_effect = responses
            with mock.patch.object(MODULE.time, "sleep"):
                MODULE.release_section(
                    self._args(),
                    session,
                    "https://zenodo.org/api",
                    "production",
                    manifest,
                    "core",
                    manifest["files"],
                    data_dir,
                    root,
                    self._state(),
                    manifest_path=manifest_path,
                )
            return self._calls(session)

    def _calls(self, session):
        """(method, url) pairs in call order; retry_request passes both by name."""
        return [
            (call.kwargs.get("method"), call.kwargs.get("url"))
            for call in session.request.call_args_list
        ]

    def test_manifest_is_uploaded_alongside_the_data(self):
        calls = self._run([])
        puts = [url for method, url in calls if method == "PUT"]
        self.assertIn("https://bucket.example/core_a.csv", puts)
        self.assertIn("https://bucket.example/manifest.json", puts)

    def test_custom_manifest_is_uploaded_under_the_canonical_filename(self):
        calls = self._run([], manifest_name="custom-release.json")
        uploads = {
            url for method, url in calls
            if method == "PUT" and url.startswith("https://bucket.example/")
        }
        self.assertEqual(
            uploads,
            {"https://bucket.example/core_a.csv", "https://bucket.example/manifest.json"},
        )

    def test_manifest_in_the_draft_is_kept_and_not_re_uploaded(self):
        """Being absent from the manifest's file list must not mark it removed."""
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest_path, _ = self._setup(Path(tmp))
            md5 = MODULE.md5_for_file(manifest_path)
        calls = self._run(
            [{"key": "manifest.json", "id": "uuid-1", "checksum": f"md5:{md5}"}]
        )
        self.assertNotIn("https://bucket.example/manifest.json", [u for m, u in calls if m == "PUT"])
        self.assertEqual([url for method, url in calls if method == "DELETE"], [])

    def test_missing_manifest_path_uploads_only_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, manifest_path, data_dir = self._setup(root)
            manifest_path.unlink()
            session = mock.Mock()
            session.request.side_effect = [
                self._response(payload=self._draft()),
                self._response(payload={"hits": {"hits": []}}),
                self._response(payload=[]),
                self._response(
                    payload={
                        "checksum": f"md5:{MODULE.md5_for_file(data_dir / 'core_a.csv')}"
                    }
                ),
                self._response(payload={"metadata": {}}),
            ]
            with mock.patch.object(MODULE.time, "sleep"):
                MODULE.release_section(
                    self._args(),
                    session,
                    "https://zenodo.org/api",
                    "production",
                    manifest,
                    "core",
                    manifest["files"],
                    data_dir,
                    root,
                    self._state(),
                    manifest_path=manifest_path,
                )
        self.assertNotIn(
            "https://bucket.example/manifest.json",
            [url for _method, url in self._calls(session)],
        )

    def test_release_state_does_not_track_the_manifest(self):
        """The manifest is not a data file; it must not appear in release diffs."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, manifest_path, data_dir = self._setup(root)
            session = mock.Mock()
            session.request.side_effect = [
                self._response(payload=self._draft()),
                self._response(payload={"hits": {"hits": []}}),
                self._response(payload=[]),
                self._response(
                    payload={
                        "checksum": f"md5:{MODULE.md5_for_file(data_dir / 'core_a.csv')}"
                    }
                ),
                self._response(
                    payload={"checksum": f"md5:{MODULE.md5_for_file(manifest_path)}"}
                ),
                self._response(payload={"metadata": {}}),
            ]
            with mock.patch.object(MODULE.time, "sleep"):
                result = MODULE.release_section(
                    self._args(),
                    session,
                    "https://zenodo.org/api",
                    "production",
                    manifest,
                    "core",
                    manifest["files"],
                    data_dir,
                    root,
                    self._state(),
                    manifest_path=manifest_path,
                )
        self.assertEqual(list(result["release"]["files"]), ["core_a.csv"])

    def test_manifest_git_checks_do_not_hide_dirty_data_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp).resolve()
            root = parent / "repo"
            root.mkdir()
            manifest, manifest_path, data_dir = self._setup(root)
            outside = parent / "outside.json"
            outside.write_bytes(manifest_path.read_bytes())
            (root / ".gitignore").write_text("ignored.json\n")
            ignored = root / "ignored.json"
            ignored.write_bytes(manifest_path.read_bytes())
            with chdir(root):
                subprocess.run(["git", "init", "-q"], check=True)
                subprocess.run(["git", "add", "data"], check=True)
                subprocess.run(
                    ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                     "-c", "commit.gpgsign=false", "commit", "-qm", "Initial commit"],
                    check=True,
                )
                self.assertEqual(
                    MODULE.uncommitted_release_files(manifest["files"], data_dir, manifest_path),
                    [],
                )
                manifest_path.write_text("{}")
                self.assertEqual(
                    MODULE.uncommitted_release_files(manifest["files"], data_dir, manifest_path),
                    [" M data/manifest.json"],
                )
                (data_dir / "core_a.csv").write_text("changed")
                for custom_path in (outside, ignored):
                    with self.subTest(custom_path=custom_path):
                        self.assertCountEqual(
                            MODULE.uncommitted_release_files(
                                manifest["files"], data_dir, custom_path
                            ),
                            [f"?? {custom_path}", " M data/core_a.csv"],
                        )


class UploadRetryTests(unittest.TestCase):
    def _session(self, responses):
        """Session whose request() returns responses in order, then raises."""
        session = mock.Mock()
        session.request.side_effect = responses
        return session

    def _response(self, status=200, checksum="md5:abc"):
        response = mock.Mock()
        response.status_code = status
        response.raise_for_status.side_effect = (
            None if status < 400 else requests.HTTPError(f"HTTP {status}")
        )
        response.json.return_value = {"checksum": checksum}
        return response

    def _file(self, tmp):
        p = Path(tmp) / "data.parquet"
        p.write_bytes(b"data")
        return p

    def _ok_response(self, path):
        return self._response(checksum=f"md5:{MODULE.md5_for_file(path)}")

    def test_retries_on_connection_error_then_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._file(tmp)
            expected_md5 = MODULE.md5_for_file(path)
            session = self._session(
                [
                    requests.exceptions.ConnectionError("EOF occurred"),
                    self._ok_response(path),
                ]
            )
            with mock.patch.object(MODULE.time, "sleep") as sleep:
                result = MODULE.upload_file(session, "https://bucket.example", path)
        self.assertEqual(session.request.call_count, 2)
        self.assertEqual(result["local_md5"], expected_md5)
        sleep.assert_called_once_with(10.0)

    def test_retries_on_http_500_then_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._file(tmp)
            expected_md5 = MODULE.md5_for_file(path)
            session = self._session([self._response(500), self._ok_response(path)])
            with mock.patch.object(MODULE.time, "sleep"):
                result = MODULE.upload_file(session, "https://bucket.example", path)
        self.assertEqual(session.request.call_count, 2)
        self.assertEqual(result["local_md5"], expected_md5)

    def test_exhausts_retries_and_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = self._session(
                [requests.exceptions.ConnectionError("EOF occurred")] * 4
            )
            with mock.patch.object(MODULE.time, "sleep"):
                with self.assertRaises(requests.exceptions.ConnectionError):
                    MODULE.upload_file(
                        session, "https://bucket.example", self._file(tmp)
                    )
        self.assertEqual(session.request.call_count, 4)

    def test_retry_delay_doubles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._file(tmp)
            session = self._session(
                [
                    requests.exceptions.ConnectionError("x"),
                    requests.exceptions.ConnectionError("x"),
                    self._ok_response(path),
                ]
            )
            with mock.patch.object(MODULE.time, "sleep") as sleep:
                MODULE.upload_file(
                    session,
                    "https://bucket.example",
                    path,
                    retries=3,
                    retry_delay=2.0,
                )
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [4.0, 8.0])

    def test_zero_byte_upload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.parquet"
            p.write_bytes(b"")
            session = self._session([])
            with self.assertRaises(ValueError):
                MODULE.upload_file(session, "https://bucket.example", p)
        self.assertEqual(session.request.call_count, 0)

    def test_parser_exposes_retry_flags(self):
        parser = MODULE.build_parser()
        args = parser.parse_args(
            ["--upload-retries", "5", "--upload-retry-delay", "10"]
        )
        self.assertEqual(args.upload_retries, 5)
        self.assertEqual(args.upload_retry_delay, 10.0)


class RetryRequestTests(unittest.TestCase):
    def test_retryable_status_codes_are_retried(self):
        for status in MODULE.RETRYABLE_STATUS_CODES:
            with self.subTest(status=status):
                session = mock.Mock()
                session.request.side_effect = [
                    self._response(status),
                    self._response(200),
                ]
                with mock.patch.object(MODULE.time, "sleep"):
                    response = MODULE.retry_request(
                        session, "GET", "https://example.com/api"
                    )
                self.assertEqual(session.request.call_count, 2)
                self.assertEqual(response.status_code, 200)

    def test_non_retryable_status_is_not_retried(self):
        session = mock.Mock()
        session.request.side_effect = [self._response(400)]
        with mock.patch.object(MODULE.time, "sleep"):
            response = MODULE.retry_request(session, "GET", "https://example.com/api")
        self.assertEqual(session.request.call_count, 1)
        self.assertEqual(response.status_code, 400)

    def test_data_factory_reopens_stream_per_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bin"
            path.write_bytes(b"data")
            session = mock.Mock()
            session.request.side_effect = [
                requests.exceptions.ConnectionError("EOF"),
                self._response(200),
            ]
            with mock.patch.object(MODULE.time, "sleep"):
                MODULE.retry_request(
                    session,
                    "PUT",
                    "https://example.com/bucket/data.bin",
                    data_factory=lambda: path.open("rb"),
                )
        # Each attempt must open a fresh handle (the first was consumed).
        self.assertEqual(session.request.call_count, 2)

    def test_publish_404_checks_deposition_state(self):
        session = mock.Mock()
        not_found = self._response(404)
        submitted = self._response(200, checksum="md5:abc")
        submitted.json.return_value = {"submitted": True}
        session.request.side_effect = [not_found, submitted]
        with mock.patch.object(MODULE.time, "sleep"):
            result = MODULE.publish_deposition(
                session, "https://zenodo.org/api", "12345"
            )
        self.assertEqual(result["submitted"], True)

    def test_publish_404_when_not_submitted_raises(self):
        session = mock.Mock()
        session.request.side_effect = [self._response(404), self._response(404)]
        with mock.patch.object(MODULE.time, "sleep"):
            with self.assertRaises(SystemExit):
                MODULE.publish_deposition(session, "https://zenodo.org/api", "12345")

    def _response(self, status=200, checksum="md5:abc"):
        response = mock.Mock()
        response.status_code = status
        response.raise_for_status.side_effect = (
            None if status < 400 else requests.HTTPError(f"HTTP {status}")
        )
        response.json.return_value = {"checksum": checksum}
        return response


if __name__ == "__main__":
    unittest.main()
