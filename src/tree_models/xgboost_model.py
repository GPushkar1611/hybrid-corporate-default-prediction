from .base import TreeModel


class XGBoostModel(TreeModel):
    """XGBoost binary classifier. Install ``xgboost`` before fitting."""

    def _estimator(self):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoostModel requires the optional 'xgboost' package. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        defaults = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(self.params)
        return XGBClassifier(**defaults)
