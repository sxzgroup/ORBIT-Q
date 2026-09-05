"""Regression checks for the archived paper-data contract."""
import unittest

from verify_updated_results import load, validate


class PaperDataTests(unittest.TestCase):
    def setUp(self):
        self.tables = load()

    def test_valid_archive(self):
        validate(self.tables)

    def test_missing_task(self):
        self.tables["task_outcomes"].pop()
        with self.assertRaises(ValueError):
            validate(self.tables)

    def test_changed_astra_outcome(self):
        self.tables["task_outcomes"][-5]["final_pass"] = "0"
        with self.assertRaises(ValueError):
            validate(self.tables)

    def test_wrong_cost(self):
        self.tables["paper_agent_axis"][-2]["cost_usd"] = "0"
        with self.assertRaises(ValueError):
            validate(self.tables)

    def test_mixed_runtime_baseline(self):
        self.tables["paper_agent_axis"][-1]["gm_slowdown"] = "0.564975"
        with self.assertRaises(ValueError):
            validate(self.tables)

    def test_oom_filled_with_reference(self):
        self.tables["astra_expert_paired"][7]["expert_sec"] = "25.05"
        with self.assertRaises(ValueError):
            validate(self.tables)

    def test_inverted_speedup(self):
        row = self.tables["astra_expert_paired"][0]
        row["expert_over_astra"] = row["astra_over_expert"]
        with self.assertRaises(ValueError):
            validate(self.tables)

    def test_wrong_replacement_value(self):
        self.tables["astra_expert_runtime"][7]["expert_reference_sec"] = "25.05"
        with self.assertRaises(ValueError):
            validate(self.tables)

    def test_estimate_labeled_measured(self):
        self.tables["astra_expert_runtime"][7]["expert_reference_basis"] = "measured"
        with self.assertRaises(ValueError):
            validate(self.tables)

    def test_unrelated_pair_changed(self):
        self.tables["astra_expert_runtime"][0]["expert_reference_sec"] = "27.22"
        with self.assertRaises(ValueError):
            validate(self.tables)

    def test_old_aggregate_retained(self):
        self.tables["astra_adjustment"]["summary"]["geometric_mean_ratio"] = 0.564975
        with self.assertRaises(ValueError):
            validate(self.tables)

    def test_estimate_counted_as_measured_pair(self):
        self.tables["astra_adjustment"]["summary"]["measured_pair_count"] = 12
        with self.assertRaises(ValueError):
            validate(self.tables)


if __name__ == "__main__":
    unittest.main()
