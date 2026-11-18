"""Shared experiment pipeline for the SEC bankruptcy-prediction study.

data        -- load the v2 panel; named feature groups and feature sets (incl. lag sets)
lags        -- history features for XGB-lag (past filings only)
splits      -- leak-free purged temporal splits and time-ordered CV folds
preprocess  -- train-only clipping / imputation / missing indicators / scaling
calibration -- class weights, downsampling + Bayes correction, Platt/isotonic
metrics     -- AUROC/AP/Brier/..., firm bootstrap, paired deltas, DeLong, backtesting
runner      -- ModelSpec + fit_predict (train -> calibrate -> test) + equal-budget tuning
models      -- model zoo: structural logit, logit, RF, XGBoost, XGBoost with prior
hybrid      -- H2 structural prior, H3 stacking (H1 = the `hybrid` feature sets)
plots       -- figure helpers for the notebook (``from src.pipeline import plots``)
sequence    -- LSTM (import separately: ``from src.pipeline.sequence import ...``; needs torch)
"""

from .calibration import Calibrator, balanced_weights, downsample, logit, prior_correct, sigmoid
from .data import (ENTITY, FEATURE_SETS, GROUPS, PANEL_V2, STRUCTURAL_PD_COLUMNS, TARGET, TIME,
                   describe_panel, feature_columns, load_panel, relabel)
from .hybrid import stack, structural_prior
from .lags import add_lag_features
from .metrics import (backtest_buckets, calibration_table, delong_test, evaluate, firm_bootstrap,
                      paired_bootstrap_delta)
from .models import MODELS, random_forest, structural_logit, xgboost, xgboost_offset
from .preprocess import Preprocessor
from .runner import FitResult, ModelSpec, cv_score, fit_predict, permutation_importance, results_table, tune
from .splits import (Split, firm_holdout_split, industry_split, oot_split, purged_split, rolling_splits,
                     subsample_training_firms, time_cv_folds)
