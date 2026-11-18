"""Shared schema and feature contract for the SEC EDGAR panel experiments."""

from __future__ import annotations

from .tree_models.config import DatasetSpec


ENTITY_COLUMN = "firm"
TIME_COLUMN = "date"
TARGET_COLUMN = "y"
POSITIVE_LABEL = 1
LABEL_HORIZON_MONTHS = 12

CORE_FEATURES = (
    "D", "r", "debt_age_days", "debt_partial", "E", "sigma_E",
)
STRUCTURAL_FEATURES = (
    "kmv_dd", "kmv_pd", "kmv_d_va",
    "simul_mu_r_dd", "simul_mu_r_pd", "simul_mu_r_d_va",
    "simul_mu_hist_dd", "simul_mu_hist_pd", "simul_mu_hist_d_va",
    "naive_dd", "naive_pd", "naive_d_va",
)
FEATURE_SETS = {
    "core": CORE_FEATURES,
    "core_plus_structural": CORE_FEATURES + STRUCTURAL_FEATURES,
}

TRAIN_END = "2018-12-31"
VALIDATION_START = "2020-01-01"
VALIDATION_END = "2020-12-31"
TEST_START = "2022-01-01"

NEVER_FEATURES = frozenset({"firm", "date", "y", "bankrupt_firm"})


def edgar_dataset_spec(feature_columns) -> DatasetSpec:
    features = tuple(feature_columns)
    prohibited = sorted(
        column for column in features
        if column in NEVER_FEATURES or column.endswith("_status")
    )
    if prohibited:
        raise ValueError(f"Identifiers, target, and status columns cannot be predictors: {prohibited}")
    return DatasetSpec(
        entity_column=ENTITY_COLUMN,
        time_column=TIME_COLUMN,
        target_column=TARGET_COLUMN,
        feature_columns=features,
        positive_label=POSITIVE_LABEL,
    )
