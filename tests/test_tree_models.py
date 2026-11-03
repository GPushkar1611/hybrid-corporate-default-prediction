import unittest

import numpy as np
import pandas as pd

from tree_models import (
    DatasetSpec,
    RandomForestModel,
    build_snapshot_features,
    build_window_features,
    temporal_split,
    validate_dataset,
)


def sample_panel():
    rows = []
    for firm, offset in [("A", 0), ("B", 1), ("C", 2), ("D", 3)]:
        for year in range(2015, 2021):
            rows.append({"firm": firm, "year": year, "default": int((year + offset) % 4 == 0), "ratio": year - 2010 + offset, "sector": "x" if offset % 2 else "y"})
    return pd.DataFrame(rows)


class TreeModelsTest(unittest.TestCase):
    def setUp(self):
        self.frame = sample_panel()
        self.spec = DatasetSpec("firm", "year", "default", ["ratio", "sector"], 1, ["sector"])

    def test_snapshot_rf_probability_output(self):
        validate_dataset(self.frame, self.spec)
        dataset = build_snapshot_features(self.frame, self.spec)
        split = temporal_split(dataset, "year", train_end=2017, validation_end=2018)
        model = RandomForestModel(categorical_columns=["sector"], params={"n_estimators": 20})
        model.fit(split.train)
        probabilities = model.predict_proba(split.test)
        self.assertEqual(len(probabilities), len(split.test.y))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_windows_stay_within_firm_and_use_history(self):
        dataset = build_window_features(self.frame, self.spec, window=3)
        self.assertEqual(len(dataset.X), 4 * 4)  # four firms, 2017-2020 endpoints
        self.assertIn("ratio_lag0", dataset.X.columns)
        self.assertIn("ratio_lag2", dataset.X.columns)
        self.assertTrue((dataset.metadata["year"] >= 2017).all())

    def test_duplicate_entity_time_fails(self):
        broken = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_dataset(broken, self.spec)


if __name__ == "__main__":
    unittest.main()
