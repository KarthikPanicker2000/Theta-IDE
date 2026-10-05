"""Tests for creating experiments that are valid for their group's paradigm.

The previous template was fixed text: it always wrote intervals_count and
eval_episodes, which offline_rl and supervised forbid, and it never wrote a
_base.yaml for a new group, so the resulting config could not be loaded at all.
"""

import tempfile
import unittest
from pathlib import Path

import yaml

from frontend.config_model import ConfigTree, ExperimentGroup, experiment_yaml, group_base_yaml


def parse(text):
    return yaml.safe_load(text)


class TestGroupDiscovery(unittest.TestCase):
    def setUp(self):
        self.tree = ConfigTree.discover()

    def test_group_reads_paradigm_from_base(self):
        self.assertEqual(self.tree.group("cartpole").paradigm, "online_rl")
        self.assertEqual(self.tree.group("mimic").paradigm, "offline_rl")

    def test_group_reads_env_override_from_base(self):
        self.assertEqual(self.tree.group("cartpole").env, "cartpole")
        self.assertEqual(self.tree.group("mimic").env, "mimic")

    def test_group_with_base_is_flagged(self):
        self.assertTrue(self.tree.group("cartpole").has_base)

    def test_unknown_group_reports_no_base(self):
        group = self.tree.group("does_not_exist")
        self.assertFalse(group.has_base)
        self.assertIsNone(group.paradigm)

    def test_group_without_paradigm_in_base(self):
        """tests/_base.yaml declares no paradigm; that must not crash."""
        self.assertTrue(self.tree.group("tests").has_base)
        self.assertIsNone(self.tree.group("tests").paradigm)

    def test_groups_includes_directories_without_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiment" / "empty_group").mkdir(parents=True)
            self.assertIn("empty_group", ConfigTree(root).groups)


class TestGroupBaseYaml(unittest.TestCase):
    def test_binds_env_and_paradigm(self):
        data = parse(group_base_yaml("cartpole", "online_rl"))
        self.assertEqual(data["paradigm"], "online_rl")
        self.assertIn({"override /env": "cartpole"}, data["defaults"])

    def test_is_a_global_package(self):
        self.assertTrue(group_base_yaml("mimic", "offline_rl").startswith("# @package _global_"))

    def test_round_trips_through_experiment_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "newgroup"
            directory.mkdir()
            (directory / "_base.yaml").write_text(group_base_yaml("mimic", "offline_rl"))
            group = ExperimentGroup.from_dir(directory)
            self.assertEqual(group.paradigm, "offline_rl")
            self.assertEqual(group.env, "mimic")
            self.assertTrue(group.has_base)


class TestExperimentYaml(unittest.TestCase):
    def setUp(self):
        self.tree = ConfigTree.discover()
        self.online = self.tree.paradigms["online_rl"]
        self.offline = self.tree.paradigms["offline_rl"]

    def test_inherits_the_group_base(self):
        data = parse(experiment_yaml("cartpole", "demo", self.online))
        self.assertEqual(data["defaults"], ["cartpole/_base"])

    def test_sets_experiment_id(self):
        self.assertEqual(parse(experiment_yaml("cartpole", "demo", self.online))["experiment_id"], "demo")

    def test_online_keeps_rollout_fields(self):
        data = parse(experiment_yaml("cartpole", "demo", self.online))
        self.assertIn("intervals_count", data)
        self.assertIn("eval_episodes", data)

    def test_offline_omits_forbidden_fields(self):
        """The group base already pins these to their only legal values."""
        data = parse(experiment_yaml("mimic", "demo", self.offline))
        self.assertNotIn("intervals_count", data)
        self.assertNotIn("eval_episodes", data)

    def test_supervised_omits_forbidden_fields(self):
        data = parse(experiment_yaml("mimic", "demo", self.tree.paradigms["supervised"]))
        self.assertNotIn("intervals_count", data)
        self.assertNotIn("eval_episodes", data)

    def test_unknown_paradigm_keeps_both_fields(self):
        """With no paradigm to consult, stay permissive and let the backend rule."""
        data = parse(experiment_yaml("cartpole", "demo", None))
        self.assertIn("intervals_count", data)
        self.assertIn("eval_episodes", data)

    def test_methods_block_is_written(self):
        data = parse(experiment_yaml("cartpole", "demo", self.online, agent="ppo", model="dnn"))
        self.assertEqual(data["methods"]["ppo_dnn"], {"agent": "ppo", "model": "dnn"})

    def test_method_name_follows_agent_model_convention(self):
        data = parse(experiment_yaml("mimic", "demo", self.offline, agent="cql", model="blendrl"))
        self.assertIn("cql_blendrl", data["methods"])

    def test_model_defaults_to_dnn(self):
        data = parse(experiment_yaml("cartpole", "demo", self.online, agent="ppo"))
        self.assertEqual(data["methods"]["ppo_dnn"]["model"], "dnn")

    def test_no_methods_block_without_an_agent(self):
        self.assertNotIn("methods", parse(experiment_yaml("cartpole", "demo", self.online)))

    def test_output_is_valid_yaml_for_every_paradigm(self):
        for name, paradigm in self.tree.paradigms.items():
            agents = [a.name for a in self.tree.agents_for(name)]
            text = experiment_yaml("grp", "demo", paradigm, agents[0] if agents else None)
            self.assertIsInstance(parse(text), dict, f"{name} produced invalid YAML")


class TestGeneratedExperimentsMatchParadigmRules(unittest.TestCase):
    """The generated file must never contain a field its paradigm forbids."""

    def test_every_group_generates_a_compliant_experiment(self):
        tree = ConfigTree.discover()
        checked = 0
        for name, group in tree.experiment_groups.items():
            if not group.paradigm:
                continue
            paradigm = tree.paradigms.get(group.paradigm)
            if paradigm is None:
                continue
            data = parse(experiment_yaml(name, "demo", paradigm))
            for field in ("intervals_count", "eval_episodes"):
                if not paradigm.field_enabled(field):
                    self.assertNotIn(field, data, f"{name} ({group.paradigm}) wrote forbidden {field}")
            checked += 1
        self.assertGreater(checked, 5, "expected several groups to declare a paradigm")


if __name__ == "__main__":
    unittest.main()
