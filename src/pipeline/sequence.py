"""LSTM over each firm's last ``window`` filings, run through the same recipe as the tabular models.

Differences from the earlier LSTM implementation:
  * Firms with fewer than ``window`` past filings are left-padded with zeros plus a
    padding flag instead of being dropped, so the LSTM scores exactly the same
    rows as the tree models (no separate "matched support" comparison needed).
  * Optional structural prior (hybrid H2): the structural log-odds are added to
    the network's output logit (a skip connection) and the output layer starts
    at zero, so training starts from the Merton-type PD and learns a correction.
  * Preprocessing, calibration and metrics are shared with ``runner.fit_predict``.

Usage:
    from src.pipeline.sequence import fit_predict_sequence
    result = fit_predict_sequence(frame, feature_columns("ml"), split)
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from .calibration import Calibrator
from .data import ENTITY, TARGET
from .metrics import evaluate
from .preprocess import Preprocessor
from .runner import META_COLUMNS, FitResult
from .splits import Split, time_cv_folds

LSTM_DEFAULTS = {
    "window": 8,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.25,
    "learning_rate": 5e-4,
    "weight_decay": 1e-3,
    "batch_size": 512,
    "epochs": 30,
    "patience": 5,
    "grad_clip": 1.0,
}
LSTM_SPACE = {
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "dropout": [0.1, 0.25, 0.4],
    "learning_rate": [3e-4, 1e-3],
    "weight_decay": [1e-4, 1e-3, 1e-2],
}


def build_windows(frame: pd.DataFrame, X: np.ndarray, rows: np.ndarray, window: int) -> np.ndarray:
    """(len(rows), window, d+1) array: the firm's last ``window`` filings up to each row.

    ``frame`` must be sorted by firm, date with a 0..n-1 index and ``X`` aligned to
    it. The extra last channel is 1 for padded (non-existent) history steps.
    """
    if not frame[ENTITY].is_monotonic_increasing:
        raise ValueError("frame must be sorted by firm, date")
    position = frame.groupby(ENTITY, sort=False).cumcount().to_numpy()
    d = X.shape[1]
    out = np.zeros((len(rows), window, d + 1), dtype=np.float32)
    for k in range(window):                 # k = 0 oldest ... window-1 = current filing
        back = window - 1 - k
        ok = position[rows] >= back
        out[ok, k, :d] = X[rows[ok] - back]
        out[~ok, k, d] = 1.0
    return out


class _LSTMNet(nn.Module):
    def __init__(self, n_inputs: int, hidden: int, layers: int, dropout: float, zero_head: bool):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        if zero_head:  # H2: start exactly at the structural prior
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x, offset=None):
        h, _ = self.lstm(x)
        z = self.head(self.dropout(h[:, -1])).squeeze(-1)
        return z if offset is None else z + offset


def _device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict(net, X, offset, device, batch_size=4096) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[s:s + batch_size]).to(device)
            ob = None if offset is None else torch.from_numpy(offset[s:s + batch_size]).to(device)
            out.append(torch.sigmoid(net(xb, ob)).cpu().numpy())
    return np.concatenate(out).astype("float64")


def train_lstm(X_train, y_train, X_stop, y_stop, *, params: dict, seed: int = 0,
               offset_train=None, offset_stop=None, device=None, verbose: bool = False):
    """Train one network; early stopping on average precision of (X_stop, y_stop)."""
    p = {**LSTM_DEFAULTS, **params}
    device = _device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_offset = offset_train is not None
    net = _LSTMNet(X_train.shape[2], p["hidden_size"], p["num_layers"], p["dropout"],
                   zero_head=use_offset).to(device)
    y = np.asarray(y_train, dtype=np.float32)
    # class weighting only without a prior: the prior already carries the base rate
    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1.0) if not use_offset else 1.0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(net.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    X_t = torch.from_numpy(X_train)
    y_t = torch.from_numpy(y)
    o_t = None if not use_offset else torch.from_numpy(offset_train.astype(np.float32))
    o_stop = None if not use_offset else offset_stop.astype(np.float32)
    generator = torch.Generator().manual_seed(seed)
    best_score, best_state, best_epoch, bad_epochs, history = -np.inf, None, 0, 0, []
    for epoch in range(p["epochs"]):
        net.train()
        order = torch.randperm(len(y_t), generator=generator)
        total = 0.0
        for s in range(0, len(order), p["batch_size"]):
            idx = order[s:s + p["batch_size"]]
            xb, yb = X_t[idx].to(device), y_t[idx].to(device)
            ob = None if o_t is None else o_t[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(xb, ob), yb)
            loss.backward()
            if p["grad_clip"]:
                nn.utils.clip_grad_norm_(net.parameters(), p["grad_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        score = average_precision_score(y_stop, _predict(net, X_stop, o_stop, device))
        history.append({"epoch": epoch + 1, "train_loss": total / len(y_t), "stop_ap": score})
        if verbose:
            print(f"    epoch {epoch + 1:2d}  loss {total / len(y_t):.4f}  stop AP {score:.4f}", flush=True)
        if score > best_score + 1e-5:
            best_score, best_epoch, bad_epochs = score, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= p["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, {"best_epoch": best_epoch, "best_stop_ap": best_score, "history": history}


def fit_predict_sequence(
    frame: pd.DataFrame,
    features: list[str],
    split: Split,
    *,
    params: dict | None = None,
    seeds=(0, 1, 2),
    calibration: str = "platt",
    offset: np.ndarray | None = None,
    device=None,
    verbose: bool = False,
    name: str = "lstm",
) -> FitResult:
    """Seed-averaged LSTM: train on split.train, early-stop and calibrate on
    split.validation, score split.test. Same output format as runner.fit_predict.

    Note: the validation year is used both for early stopping and for the
    calibrator (the test set is never touched). Trees are tuned by CV inside the
    training period instead; report this difference in the write-up.
    """
    start = time.time()
    p = {**LSTM_DEFAULTS, **(params or {})}
    prep = Preprocessor.preset(features, "neural").fit(frame.loc[split.train])
    X_all = prep.transform(frame)
    rows = {part: np.flatnonzero(getattr(split, part)) for part in ("train", "validation", "test")}
    X = {part: build_windows(frame, X_all, r, p["window"]) for part, r in rows.items()}
    y = {part: frame[TARGET].to_numpy()[r] for part, r in rows.items()}
    off = {part: (None if offset is None else np.asarray(offset, np.float32)[r]) for part, r in rows.items()}

    raw = {"validation": [], "test": []}
    runs = []
    device = _device(device)
    for seed in seeds:
        if verbose:
            print(f"  [{name}] seed {seed}", flush=True)
        net, info = train_lstm(X["train"], y["train"], X["validation"], y["validation"], params=p,
                               seed=seed, offset_train=off["train"], offset_stop=off["validation"],
                               device=device, verbose=verbose)
        for part in raw:
            raw[part].append(_predict(net, X[part], off[part], device))
        runs.append({"seed": seed, "best_epoch": info["best_epoch"], "best_validation_ap": info["best_stop_ap"]})

    outputs = {}
    for part in raw:
        sub = frame.loc[getattr(split, part), [c for c in META_COLUMNS if c in frame.columns]].reset_index(drop=True)
        sub["p_raw"] = np.mean(raw[part], axis=0)
        outputs[part] = sub
    calibrator = Calibrator(calibration).fit(outputs["validation"]["p_raw"], outputs["validation"][TARGET])
    metrics = {}
    for part, sub in outputs.items():
        sub["p"] = calibrator.transform(sub["p_raw"])
        metrics[part] = evaluate(sub[TARGET], sub["p"])
        metrics[f"{part}_uncalibrated"] = evaluate(sub[TARGET], sub["p_raw"])
    return FitResult(
        model_name=name, feature_set=list(features), split_name=split.name,
        params={**p, "seeds": list(seeds), "runs": runs, "offset": offset is not None},
        validation=outputs["validation"], test=outputs["test"], metrics=metrics,
        calibrator=calibrator, seconds=time.time() - start,
    )


def tune_sequence(frame: pd.DataFrame, features: list[str], split: Split, *, n_trials: int = 12,
                  seed: int = 0, validation_years=(2015, 2016, 2017, 2018), device=None,
                  verbose: bool = True):
    """Random search over LSTM_SPACE with the same time-ordered folds and budget as ``runner.tune``.

    Each fold trains one seed with early stopping on that fold's scoring year and
    reports the best average precision reached. Returns (best_params, trials).
    """
    rng = np.random.default_rng(seed)
    folds = time_cv_folds(frame, within=split.train, validation_years=validation_years)
    candidates = [{}]
    while len(candidates) < n_trials:
        trial = {k: v[rng.integers(len(v))] for k, v in LSTM_SPACE.items()}
        if trial not in candidates:
            candidates.append(trial)
    window = LSTM_DEFAULTS["window"]
    rows = []
    for i, trial in enumerate(candidates):
        t0 = time.time()
        scores = []
        for fit_mask, score_mask in folds:
            prep = Preprocessor.preset(features, "neural").fit(frame.loc[fit_mask])
            X_all = prep.transform(frame)
            fit_rows, score_rows = np.flatnonzero(fit_mask), np.flatnonzero(score_mask)
            _, info = train_lstm(build_windows(frame, X_all, fit_rows, window), frame[TARGET].to_numpy()[fit_rows],
                                 build_windows(frame, X_all, score_rows, window), frame[TARGET].to_numpy()[score_rows],
                                 params={**LSTM_DEFAULTS, **trial}, seed=seed, device=device)
            scores.append(info["best_stop_ap"])
        rows.append({"trial": i, **trial, "cv_average_precision": float(np.mean(scores)),
                     "seconds": time.time() - t0})
        if verbose:
            print(f"  [lstm] trial {i + 1}/{len(candidates)}  cv AP={np.mean(scores):.4f}  {trial}", flush=True)
    trials = pd.DataFrame(rows).sort_values("cv_average_precision", ascending=False)
    return dict(candidates[int(trials.iloc[0]["trial"])]), trials
