from sklearn.ensemble import RandomForestClassifier

from .base import TreeModel


class RandomForestModel(TreeModel):
    """Random Forest classifier with train-only preprocessing."""

    def _estimator(self):
        defaults = {
            "n_estimators": 500,
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        defaults.update(self.params)
        return RandomForestClassifier(**defaults)
