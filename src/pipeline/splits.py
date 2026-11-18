"""Leak-free temporal splits.

Labels look 12 months ahead, so a training row dated in year Y can "see" an
outcome in year Y+1. Every split therefore leaves a full gap year between
train and validation, and between validation and test:

    train: filings up to Dec 31 of year T
    gap  : year T+1 is dropped
    valid: year T+2
    gap  : year T+3 is dropped
    test : year T+4 onwards

Firms can appear in several partitions (it is an out-of-time test of the same
firm universe); ``firm_holdout`` in a later step gives the unseen-firm variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import TARGET, TIME, ENTITY

# Main out-of-time split (same dates as the earlier evaluation)
TRAIN_END_YEAR = 2018
VALIDATION_YEAR = 2020
TEST_START_YEAR = 2022


@dataclass
class Split:
    """Boolean row masks over the panel for one experiment."""

    name: str
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    info: dict = field(default_factory=dict)

    def summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = {}
        for part in ("train", "validation", "test"):
            mask = getattr(self, part)
            sub = frame.loc[mask]
            pos = sub[TARGET].eq(1)
            rows[part] = {
                "rows": int(mask.sum()),
                "firms": int(sub[ENTITY].nunique()),
                "positive_rows": int(pos.sum()),
                "positive_firms": int(sub.loc[pos, ENTITY].nunique()),
                "positive_rate": float(pos.mean()) if len(sub) else np.nan,
                "first_date": sub[TIME].min(),
                "last_date": sub[TIME].max(),
            }
        return pd.DataFrame(rows).T


def purged_split(
    frame: pd.DataFrame,
    *,
    train_end_year: int,
    validation_year: int,
    test_start_year: int,
    test_end_year: int | None = None,
    train_start_year: int | None = None,
    name: str | None = None,
) -> Split:
    """Train / gap / validation / gap / test split by filing year."""
    if validation_year < train_end_year + 2:
        raise ValueError("Need a full gap year between training and validation (12-month labels)")
    if test_start_year < validation_year + 2:
        raise ValueError("Need a full gap year between validation and test (12-month labels)")
    year = frame[TIME].dt.year.to_numpy()
    train = year <= train_end_year
    if train_start_year is not None:
        train &= year >= train_start_year
    validation = year == validation_year
    test = year >= test_start_year
    if test_end_year is not None:
        test &= year <= test_end_year
    split = Split(
        name or f"train<={train_end_year}|val={validation_year}|test>={test_start_year}",
        train, validation, test,
        info={"train_end_year": train_end_year, "validation_year": validation_year,
              "test_start_year": test_start_year, "test_end_year": test_end_year},
    )
    for part in ("train", "validation", "test"):
        mask = getattr(split, part)
        if mask.sum() == 0 or frame.loc[mask, TARGET].sum() == 0:
            raise ValueError(f"{split.name}: {part} partition is empty or has no positives")
    return split


def oot_split(frame: pd.DataFrame) -> Split:
    """The main out-of-time split: train <=2018, validate 2020, test 2022-2025."""
    return purged_split(
        frame,
        train_end_year=TRAIN_END_YEAR,
        validation_year=VALIDATION_YEAR,
        test_start_year=TEST_START_YEAR,
        name="oot",
    )


def _checked(split: Split, frame: pd.DataFrame) -> Split:
    for part in ("train", "validation", "test"):
        mask = getattr(split, part)
        if mask.sum() == 0 or frame.loc[mask, TARGET].sum() == 0:
            raise ValueError(f"{split.name}: {part} partition is empty or has no positives")
    return split


# --------------------------------------------------------------------------
# evaluation regimes
# --------------------------------------------------------------------------
def rolling_splits(frame: pd.DataFrame, test_years) -> list[Split]:
    """Year-by-year out-of-time tests: for test year T train <= T-4, validate T-2, test T."""
    return [purged_split(frame, train_end_year=t - 4, validation_year=t - 2, test_start_year=t,
                         test_end_year=t, name=f"rolling_{t}") for t in test_years]


def industry_split(frame: pd.DataFrame, held_out: str, *, column: str = "sic_division",
                   out_of_time: bool = False) -> Split:
    """Leave one industry out.

    Train/validate on all *other* industries with the main OOT dates. Test on the
    held-out industry either in the training period (pure out-of-industry) or in
    the test period (out-of-industry AND out-of-time, the hardest case).
    """
    base = oot_split(frame)
    group = frame[column].eq(held_out).to_numpy()
    test_rows = (base.test if out_of_time else base.train) & group
    kind = "ooi_oot" if out_of_time else "ooi"
    return _checked(Split(f"{kind}_{held_out}", base.train & ~group, base.validation & ~group,
                          test_rows, info={"held_out": held_out, "out_of_time": out_of_time}), frame)


def firm_holdout_split(frame: pd.DataFrame, *, fraction: float = 0.3, seed: int = 0) -> Split:
    """Unseen firms: a random ``fraction`` of firms is removed from train/validation
    and the test set contains only those firms (main OOT dates)."""
    rng = np.random.default_rng(seed)
    firms = frame[ENTITY].unique()
    held = set(rng.choice(firms, size=int(round(fraction * len(firms))), replace=False))
    is_held = frame[ENTITY].isin(held).to_numpy()
    base = oot_split(frame)
    return _checked(Split(f"unseen_firms_{seed}", base.train & ~is_held, base.validation & ~is_held,
                          base.test & is_held, info={"fraction": fraction, "seed": seed}), frame)


def subsample_training_firms(frame: pd.DataFrame, split: Split, *, fraction: float, seed: int = 0) -> Split:
    """Limited-data regime: keep the training rows of a random ``fraction`` of the
    training firms; validation and test are unchanged."""
    rng = np.random.default_rng(seed)
    train_firms = frame.loc[split.train, ENTITY].unique()
    kept = rng.choice(train_firms, size=max(1, int(round(fraction * len(train_firms)))), replace=False)
    train = split.train & frame[ENTITY].isin(kept).to_numpy()
    return _checked(Split(f"{split.name}_frac{fraction}_seed{seed}", train, split.validation, split.test,
                          info={**split.info, "fraction": fraction, "seed": seed}), frame)


def time_cv_folds(
    frame: pd.DataFrame,
    *,
    within: np.ndarray | None = None,
    validation_years: tuple[int, ...] = (2015, 2016, 2017, 2018),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds for hyper-parameter tuning, inside the training period.

    Fold for validation year V: fit on filings up to Dec 31 of V-2, score on
    year V (year V-1 is the label gap). Replaces the random 5-fold CV of
    Moscatelli et al. (2020), which would leak here because the same firm
    appears in many folds and neighbouring filings share an outcome window.
    """
    year = frame[TIME].dt.year.to_numpy()
    allowed = np.ones(len(frame), bool) if within is None else np.asarray(within, bool)
    folds = []
    for v in validation_years:
        fit_mask = allowed & (year <= v - 2)
        score_mask = allowed & (year == v)
        if frame.loc[score_mask, TARGET].sum() == 0 or frame.loc[fit_mask, TARGET].sum() == 0:
            raise ValueError(f"CV fold {v} has no positives")
        folds.append((fit_mask, score_mask))
    return folds
