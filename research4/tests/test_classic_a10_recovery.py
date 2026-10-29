import copy
import importlib.util
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools" / "validate_classic_a10_recovery.py"
SPEC_PATH = ROOT / "infrastructure" / "classic_a10_scale1_v6" / "submit.json"
BINDING_PATH = (ROOT / "infrastructure" / "classic_a10_scale1_v6"
                / "PROVIDER_BINDING.json")
ARTIFACT_ROOT = pathlib.Path(
    "C:/Users/farha/Research4/artifacts/"
    "qwen9b_simulated_continuity_20261028/three_residual_ec6fdb9c")
SPEC = importlib.util.spec_from_file_location("classic_a10_validator", MODULE_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


class ClassicA10RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        self.binding = json.loads(BINDING_PATH.read_text(encoding="utf-8"))
        self.artifact_root = ARTIFACT_ROOT if ARTIFACT_ROOT.is_dir() else None

    def assert_rejected(self, mutate):
        payload = copy.deepcopy(self.payload)
        mutate(payload)
        with self.assertRaises(ValueError):
            VALIDATOR.validate(payload)

    def resign_binding(self, binding):
        payload = copy.deepcopy(binding)
        payload.pop("binding_sha256", None)
        binding["binding_sha256"] = VALIDATOR.digest(payload)

    def test_frozen_provider_spec_passes(self):
        result = VALIDATOR.validate(self.payload)
        self.assertEqual(result["status"], "validated_local_only_not_submitted")
        self.assertEqual(result["maximum_compute_cost_usd"], 3.3)
        self.assertEqual(len(result["provider_spec_sha256"]), 64)

    def test_submit_run_uses_direct_classic_cluster(self):
        gpu = self.payload["tasks"][1]
        self.assertIn("new_cluster", gpu)
        self.assertNotIn("job_cluster_key", gpu)
        self.assertNotIn("job_clusters", self.payload)
        self.assertNotIn("compute", gpu)
        self.assertNotIn("environment_key", gpu)

    def test_unexpected_fields_are_rejected_everywhere(self):
        self.assert_rejected(lambda value: value.__setitem__("queue", {}))
        self.assert_rejected(
            lambda value: value["tasks"][0].__setitem__("policy_id", "x"))
        self.assert_rejected(
            lambda value: value["tasks"][1]["new_cluster"].__setitem__(
                "policy_id", "x"))
        self.assert_rejected(
            lambda value: value["tasks"][1]["new_cluster"].__setitem__(
                "instance_pool_id", "x"))
        self.assert_rejected(
            lambda value: value["tasks"][1]["new_cluster"].__setitem__(
                "autoscale", {"min_workers": 1, "max_workers": 1}))
        self.assert_rejected(
            lambda value: value["environments"].append({
                "environment_key": "extra", "spec": {"environment_version": "6"}
            }))

    def test_serverless_fallback_is_rejected(self):
        def mutate(payload):
            gpu = payload["tasks"][1]
            gpu.pop("new_cluster")
            gpu["compute"] = {"hardware_accelerator": "GPU_1xA10"}
            gpu["environment_key"] = "continuity_gpu"
        self.assert_rejected(mutate)

    def test_retry_spot_worker_and_multi_gpu_drift_are_rejected(self):
        self.assert_rejected(
            lambda value: value["tasks"][1].__setitem__("max_retries", 1))
        self.assert_rejected(
            lambda value: value["tasks"][1]["new_cluster"].__setitem__(
                "azure_attributes", {
                    "availability": "SPOT_WITH_FALLBACK_AZURE",
                    "first_on_demand": 1,
                }))
        self.assert_rejected(
            lambda value: value["tasks"][1]["new_cluster"].__setitem__(
                "num_workers", 1))
        self.assert_rejected(
            lambda value: value["tasks"][1]["new_cluster"].__setitem__(
                "node_type_id", "Standard_NV72ads_A10_v5"))

    def test_notebooks_parameters_and_dynamic_owner_bindings_are_exact(self):
        self.assert_rejected(
            lambda value: value["tasks"][0]["notebook_task"].__setitem__(
                "notebook_path",
                VALIDATOR.SOURCE_ROOT + "/notebooks/databricks_continuity_cleanup"))
        self.assert_rejected(
            lambda value: value["tasks"][0]["notebook_task"][
                "base_parameters"].pop("intent_path"))
        self.assert_rejected(
            lambda value: value["tasks"][1]["notebook_task"][
                "base_parameters"].__setitem__("run_id", "123"))
        self.assert_rejected(
            lambda value: value["tasks"][1]["notebook_task"][
                "base_parameters"].__setitem__(
                    "base_qualification_receipt_path", "/wrong"))
        self.assert_rejected(
            lambda value: value["tasks"][0]["notebook_task"][
                "base_parameters"].__setitem__("expected_mode", "primary"))

    def test_dag_and_all_done_release_are_exact(self):
        self.assert_rejected(
            lambda value: value["tasks"][1].__setitem__("depends_on", []))
        self.assert_rejected(
            lambda value: value["tasks"][2].__setitem__("run_if", "ALL_SUCCESS"))
        self.assert_rejected(
            lambda value: value["tasks"][3].__setitem__("run_if", "ALL_SUCCESS"))
        self.assert_rejected(
            lambda value: value["tasks"][3]["depends_on"].pop())

    def test_dependencies_cpu_environments_and_timeouts_are_exact(self):
        self.assert_rejected(
            lambda value: value["tasks"][1]["libraries"][1]["pypi"].__setitem__(
                "package", "transformers==5.18.0"))
        self.assert_rejected(
            lambda value: value["tasks"][0].__setitem__(
                "environment_key", "continuity_finalize_cpu"))
        self.assert_rejected(
            lambda value: value["environments"][1]["spec"][
                "dependencies"].append("extra==1"))
        self.assert_rejected(
            lambda value: value["tasks"][1].__setitem__(
                "timeout_seconds", 2800))

    def test_cost_tags_and_offline_environment_are_exact(self):
        self.assert_rejected(
            lambda value: value["tasks"][1]["new_cluster"][
                "custom_tags"].__setitem__("cost_scope", "unbounded"))
        self.assert_rejected(
            lambda value: value["tasks"][1]["new_cluster"][
                "spark_env_vars"].__setitem__("HF_HUB_OFFLINE", "0"))

    def test_bool_aliases_are_rejected_for_counts(self):
        self.assert_rejected(
            lambda value: value["tasks"][1]["new_cluster"].__setitem__(
                "num_workers", False))
        self.assert_rejected(
            lambda value: value.__setitem__("timeout_seconds", True))

    def test_provider_binding_replays_actual_evidence(self):
        result = VALIDATOR.validate_binding(
            self.binding, self.payload, repo_root=ROOT.parent,
            artifact_root=self.artifact_root)
        self.assertEqual(
            result["provider_binding_sha256"],
            self.binding["binding_sha256"])
        self.assertEqual(
            result["binding_status"],
            "review_candidate_not_submission_authority")
        self.assertEqual(
            result["artifact_replay_performed"],
            self.artifact_root is not None)

    def test_provider_binding_tampering_is_rejected(self):
        binding = copy.deepcopy(self.binding)
        binding["cost"]["planning_cap_usd"] = 50.0
        with self.assertRaisesRegex(ValueError, "self-hash"):
            VALIDATOR.validate_binding(
                binding, self.payload, repo_root=ROOT.parent,
                artifact_root=self.artifact_root)

    def test_self_consistent_wrong_model_dataset_and_version_are_rejected(self):
        binding = copy.deepcopy(self.binding)
        binding["model"]["tree_sha256"] = "0" * 64
        self.resign_binding(binding)
        with self.assertRaisesRegex(ValueError, "model binding"):
            VALIDATOR.validate_binding(
                binding, self.payload, repo_root=ROOT.parent,
                artifact_root=self.artifact_root)
        binding = copy.deepcopy(self.binding)
        binding["dataset"]["dataset_tree_sha256"] = "1" * 64
        self.resign_binding(binding)
        with self.assertRaisesRegex(ValueError, "dataset binding"):
            VALIDATOR.validate_binding(
                binding, self.payload, repo_root=ROOT.parent,
                artifact_root=self.artifact_root)
        binding = copy.deepcopy(self.binding)
        binding["version"] = "other"
        self.resign_binding(binding)
        with self.assertRaisesRegex(ValueError, "binding version"):
            VALIDATOR.validate_binding(
                binding, self.payload, repo_root=ROOT.parent,
                artifact_root=self.artifact_root)
        binding = copy.deepcopy(self.binding)
        binding["binding_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "self-hash"):
            VALIDATOR.validate_binding(
                binding, self.payload, repo_root=ROOT.parent,
                artifact_root=self.artifact_root)


if __name__ == "__main__":
    unittest.main()
