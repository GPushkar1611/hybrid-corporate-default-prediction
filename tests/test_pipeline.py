"""Tests for src/pipeline on a small synthetic filing panel (no real data needed).

Run from the repository root:  python -m unittest discover -s tests -v
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.pipeline import (Calibrator, ModelSpec, Preprocessor, backtest_buckets, delong_test,
                          downsample, evaluate, fit_predict, logit, oot_split,
                          paired_bootstrap_delta, prior_correct, purged_split, time_cv_folds, tune)


def synthetic_panel(n_firms=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for firm in range(n_firms):
        risk = rng.normal()
        for q, date in enumerate(pd.date_range("2010-03-31", "2025-06-30", freq="QE")):
            x1 = risk + rng.normal(scale=0.5)
            x2 = rng.normal()
            p = 1 / (1 + np.exp(-(-4.5 + 1.2 * x1 + 0.3 * x2)))
            rows.append({"firm": firm, "date": date, "x1": x1, "x2": x2,
                         "y": int(rng.random() < p), "sic2": 20 + firm % 5})
    frame = pd.DataFrame(rows)
    frame.loc[rng.random(len(frame)) < 0.1, "x2"] = np.nan
    return frame


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_panel()

    def test_oot_has_gap_years(self):
        split = oot_split(self.frame)
        years = self.frame["date"].dt.year
        self.assertEqual(years[split.train].max(), 2018)
        self.assertEqual(set(years[split.validation]), {2020})
        self.assertEqual(years[split.test].min(), 2022)
        self.assertFalse(((split.train & split.validation) | (split.validation & split.test)).any())
        self.assertEqual(int(years.isin([2019, 2021])[split.train | split.validation | split.test].sum()), 0)

    def test_rejects_missing_gap(self):
        with self.assertRaises(ValueError):
            purged_split(self.frame, train_end_year=2018, validation_year=2019, test_start_year=2021)

    def test_cv_folds_leave_gap(self):
        split = oot_split(self.frame)
        years = self.frame["date"].dt.year.to_numpy()
        for fit, score in time_cv_folds(self.frame, within=split.train):
            v = years[score].min()
            self.assertEqual(years[score].max(), v)
            self.assertLessEqual(years[fit].max(), v - 2)
            self.assertFalse((fit & ~split.train).any())


class PreprocessTests(unittest.TestCase):
    def test_statistics_come_from_training_rows_only(self):
        train = pd.DataFrame({"a": np.arange(100.0), "b": [np.nan] * 10 + list(range(90))})
        test = pd.DataFrame({"a": [1e9, -1e9], "b": [np.nan, 5.0]})
        prep = Preprocessor(["a", "b"], clip=(0.01, 0.99), impute=True, indicators=True, scale=False).fit(train)
        out = prep.transform(test)
        self.assertAlmostEqual(out[0, 0], np.quantile(np.arange(100.0), 0.99), places=4)
        self.assertAlmostEqual(out[1, 0], np.quantile(np.arange(100.0), 0.01), places=4)
        self.assertEqual(out[0, 1], np.float32(np.median(np.arange(90.0))))  # train median
        self.assertEqual(list(out[:, 2]), [1.0, 0.0])  # missing indicator for b
        self.assertEqual(prep.feature_names_, ["a", "b", "b__missing"])

    def test_boosting_passthrough_keeps_nan(self):
        frame = pd.DataFrame({"a": [1.0, np.nan, np.inf]})
        out = Preprocessor.preset(["a"], "boosting").fit_transform(frame)
        self.assertTrue(np.isnan(out[1, 0]) and np.isnan(out[2, 0]))


class CalibrationTests(unittest.TestCase):
    def test_prior_correction_recovers_population_rate(self):
        rng = np.random.default_rng(1)
        y = (rng.random(200_000) < 0.01).astype(int)
        idx, beta = downsample(y, negatives_per_positive=1.0, seed=0)
        sample_rate = y[idx].mean()                       # ~0.5 in the balanced sample
        self.assertAlmostEqual(float(prior_correct(sample_rate, beta)), y.mean(), places=3)

    def test_platt_fixes_overconfident_scores(self):
        rng = np.random.default_rng(2)
        true_p = 1 / (1 + np.exp(-rng.normal(-4, 1, 50_000)))
        y = (rng.random(50_000) < true_p).astype(int)
        inflated = 1 / (1 + np.exp(-(logit(true_p) + 3)))  # like class-weighted output
        cal = Calibrator("platt").fit(inflated[:25_000], y[:25_000])
        self.assertLess(abs(cal.transform(inflated[25_000:]).mean() - y[25_000:].mean()), 0.003)


class MetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.y = (rng.random(5000) < 0.05).astype(int)
        self.p_good = np.clip(0.05 + 0.3 * self.y + rng.normal(0, 0.1, 5000), 0.001, 0.999)
        self.p_bad = np.clip(0.05 + 0.05 * self.y + rng.normal(0, 0.1, 5000), 0.001, 0.999)
        self.firm = np.repeat(np.arange(500), 10)

    def test_delong_matches_sklearn_auc(self):
        result = delong_test(self.y, self.p_good, self.p_bad)
        self.assertAlmostEqual(result["auroc_a"], roc_auc_score(self.y, self.p_good), places=10)
        self.assertAlmostEqual(result["auroc_b"], roc_auc_score(self.y, self.p_bad), places=10)
        self.assertLess(result["p_value"], 0.01)

    def test_paired_bootstrap_detects_better_model(self):
        delta = paired_bootstrap_delta(self.y, self.p_good, self.p_bad, self.firm, replicates=200)
        self.assertGreater(delta.loc["auroc", "low"], 0)
        self.assertLess(delta.loc["auroc", "share_le_0"], 0.05)

    def test_backtest_counts_every_row_once(self):
        table = backtest_buckets(self.y, self.p_good)
        self.assertEqual(table["rows"].sum(), len(self.y))
        self.assertEqual(table["positives"].sum(), self.y.sum())

    def test_evaluate_keys(self):
        out = evaluate(self.y, self.p_good)
        for key in ("auroc", "average_precision", "ap_lift", "brier", "top_decile_capture", "ece"):
            self.assertIn(key, out)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_panel(n_firms=300)
        self.split = oot_split(self.frame)

    def spec(self, imbalance):
        return ModelSpec("logit", lambda p: LogisticRegression(max_iter=500, **p), "linear",
                         imbalance=imbalance, search_space={"C": [0.01, 0.1, 1.0]})

    def test_fit_predict_weights_and_downsample(self):
        for imbalance in ("weights", "downsample"):
            result = fit_predict(self.spec(imbalance), self.frame, ["x1", "x2"], self.split)
            self.assertGreater(result.metrics["test"]["auroc"], 0.7)
            # calibrated test PDs should be near the realised rate
            self.assertLess(abs(result.test["p"].mean() - result.test["y"].mean()), 0.01)
            self.assertEqual(len(result.test), int(self.split.test.sum()))

    def test_tune_uses_fixed_budget(self):
        best, trials = tune(self.spec("weights"), self.frame, ["x1", "x2"], self.split,
                            n_trials=3, verbose=False)
        self.assertEqual(len(trials), 3)
        self.assertIn(best.get("C", 1.0), [0.01, 0.1, 1.0])


if __name__ == "__main__":
    unittest.main()
