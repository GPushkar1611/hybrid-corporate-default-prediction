"""Compare saved test predictions on filing endpoints shared with the LSTM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_sec_filing_eval import _firm_cluster_bootstrap, _metric_dict


def run(prediction_path: Path, out_dir: Path, bootstrap_replicates: int = 500) -> dict:
    predictions = pd.read_csv(prediction_path, parse_dates=["date"])
    sequence_rows = predictions.loc[predictions["model"].eq("lstm_ensemble"), ["firm", "date", "y"]]
    if sequence_rows.empty:
        raise ValueError("Prediction file contains no LSTM test rows")
    if sequence_rows.duplicated(["firm", "date"]).any():
        raise ValueError("LSTM test predictions contain duplicate firm/date endpoints")
    matched = predictions.merge(sequence_rows, on=["firm", "date", "y"], how="inner", validate="many_to_one")
    result = {"support": {}, "models": {}}
    comparison = []
    for name, group in matched.groupby("model", sort=False):
        result["support"][name] = {
            "rows": int(len(group)),
            "firms": int(group["firm"].nunique()),
            "positive_rows": int(group["y"].sum()),
            "positive_firms": int(group.loc[group["y"].eq(1), "firm"].nunique()),
        }
        result["models"][name] = {
            "test_metrics_on_lstm_endpoint_support": _metric_dict(group["y"], group["probability"]),
            "firm_cluster_bootstrap": _firm_cluster_bootstrap(
                group["y"], group["probability"], group["firm"], replicates=bootstrap_replicates
            ),
        }
        metric = result["models"][name]["test_metrics_on_lstm_endpoint_support"]
        comparison.append({
            "model": name,
            **result["support"][name],
            "auroc": metric["auroc"],
            "average_precision": metric["average_precision"],
            "brier_score": metric["brier_score"],
            "top_decile_default_capture": metric["top_decile_default_capture"],
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(out_dir / "test_predictions_matched.csv", index=False)
    pd.DataFrame(comparison).to_csv(out_dir / "matched_support_comparison.csv", index=False)
    (out_dir / "matched_support_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["support"], indent=2), flush=True)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/sec_companyfacts_final_eval/test_predictions.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/sec_companyfacts_final_eval"))
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    run(args.predictions, args.out_dir, args.bootstrap_replicates)


if __name__ == "__main__":
    main()
