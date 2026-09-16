from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from precisionphage.data.genomes import GenomeIndex  # noqa: E402
from precisionphage.eval import nested_group_oof_decisions  # noqa: E402
from precisionphage.splits import combined_unseen_folds  # noqa: E402
from precisionphage.temporal import TherapyParams, simulate  # noqa: E402
from precisionphage.utils import load_config  # noqa: E402


class FrozenDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(ROOT / "data/interim_v2/interactions_modelable.csv")

    def test_documented_subset(self):
        self.assertEqual(len(self.df), 1947)
        self.assertEqual(int(self.df["label"].sum()), 1488)
        self.assertEqual(set(self.df["study"]), {"NCBI_HR"})

    def test_cross_source_detail_matches_current_baseline(self):
        detail = pd.read_csv(ROOT / "data/results_v2/cross_study_detail.csv")
        baseline = json.loads((ROOT / "data/results_v2/baseline_results.json").read_text())
        self.assertEqual(set(detail["held_out_source"]),
                         {"NCBI_HR", "NahantCollection", "StaphStudy"})
        self.assertEqual(set(detail["model"]), {"GBM", "EdgeMLP"})
        for model in ("GBM", "EdgeMLP"):
            observed = detail.loc[detail["model"] == model, "auroc"].mean()
            expected = baseline["cross_study"][model]["mean_roc_auc"]
            self.assertAlmostEqual(observed, expected, places=10)

    def test_exact_genome_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "NC_000866.4.fasta").write_text(
                ">NC_000866.4\nACGTACGT\n", encoding="utf-8")
            index = GenomeIndex([root])
            self.assertIsNotNone(index.resolve("NC_000866.4"))
            self.assertIsNotNone(index.resolve("NC_000866"))
            self.assertIsNone(index.resolve("NC_00086"))  # no substring fallback

        phage_dir = ROOT / "external/phist_run/phages"
        host_dir = ROOT / "external/phist_run/hosts"
        if (phage_dir.is_dir() and host_dir.is_dir()
                and any(phage_dir.glob("*.fasta"))
                and any(host_dir.glob("*.fasta"))):
            pidx = GenomeIndex([phage_dir])
            hidx = GenomeIndex([host_dir])
            self.assertEqual(
                pidx.coverage(self.df["phage"].unique().tolist())["resolved"],
                self.df["phage"].nunique())
            self.assertEqual(
                hidx.coverage(self.df["host"].unique().tolist())["resolved"],
                self.df["host"].nunique())


class SplitTests(unittest.TestCase):
    def test_combined_split_holds_out_both_axes(self):
        rows = []
        for p in range(10):
            for h in range(10):
                rows.append({"phage_cluster": p, "host_cluster": h,
                             "label": (p + h) % 2})
        df = pd.DataFrame(rows)
        folds = list(combined_unseen_folds(
            df, "phage_cluster", "host_cluster", n_splits=5,
            seed=42, min_pos=1, min_neg=1))
        # Folds with a single test class are intentionally skipped.
        self.assertGreaterEqual(len(folds), 1)
        for fold in folds:
            train = df.iloc[fold.train_idx]
            test = df.iloc[fold.test_idx]
            self.assertTrue(set(train["phage_cluster"]).isdisjoint(
                set(test["phage_cluster"])))
            self.assertTrue(set(train["host_cluster"]).isdisjoint(
                set(test["host_cluster"])))
            self.assertEqual(np.intersect1d(fold.train_idx, fold.test_idx).size, 0)

    def test_nested_group_thresholds_cover_every_outer_row(self):
        rng = np.random.default_rng(7)
        groups = np.repeat(np.arange(20), 6)
        y = np.tile([0, 1, 0, 1, 0, 1], 20)
        X = np.column_stack([y + rng.normal(0, 0.2, len(y)), rng.normal(size=len(y))])

        def predictor(x_train, y_train, x_test, seed):
            del x_train, y_train, seed
            return 1 / (1 + np.exp(-x_test[:, 0]))

        probabilities, decisions, thresholds = nested_group_oof_decisions(
            X, y, groups, predictor, seed=42, n_splits=5, inner_splits=4)
        self.assertFalse(np.isnan(probabilities).any())
        self.assertEqual(decisions.dtype, np.bool_)
        self.assertEqual(len(thresholds), 5)


class ConfigurationTests(unittest.TestCase):
    def test_root_is_relative_to_config_not_working_directory(self):
        cfg = load_config(ROOT / "configs/default.yaml")
        self.assertEqual(cfg["paths"]["root"], ROOT.resolve())

    def test_temporal_config_maps_to_parameters(self):
        cfg = load_config(ROOT / "configs/default.yaml")["temporal"]
        steps = int(round(cfg["horizon_hours"] / cfg["dt_hours"])) + 1
        params = TherapyParams(
            mu=cfg["mutation_rate"], cost=cfg["resistance_cost"],
            burst=cfg["burst_size"], beta=cfg["adsorption_rate"],
            t_max=cfg["horizon_hours"], n_steps=steps)
        self.assertEqual(params.t_max, 96)
        self.assertEqual(params.n_steps, 193)
        self.assertEqual(params.cost, 0.05)

    def test_resistance_dependence_endpoints_are_supported(self):
        matrix = np.array([[1.0], [1.0]])
        initial = np.array([1e6])
        cross = simulate(matrix, initial, TherapyParams(
            resistance_independence=0.0, t_max=1.0, n_steps=3))
        independent = simulate(matrix, initial, TherapyParams(
            resistance_independence=1.0, t_max=1.0, n_steps=3))
        self.assertTrue(cross["success"])
        self.assertTrue(independent["success"])
        self.assertGreater(cross["R"][0, -1], independent["R"][0, -1])


if __name__ == "__main__":
    unittest.main()
