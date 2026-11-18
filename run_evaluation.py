"""Run the shared out-of-time tree and LSTM comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_lstm_smoke import run as run_lstm
from run_tree_smoke import run as run_trees


def run(panel_path: Path, out_dir: Path, *, window: int = 12, epochs: int = 10) -> pd.DataFrame:
    """Run both model families on the fixed shared out-of-time split.

    Tree and LSTM scores use the same panel, core predictors, endpoint labels,
    and calendar cutoffs. The LSTM requires a full history window, so its
    eligible endpoint set is smaller; coverage counts are recorded separately.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_result = run_trees(panel_path, out_dir / "trees")
    lstm_result = run_lstm(panel_path, out_dir / "lstm", window=window, epochs=epochs)

    rows = []
    for model_name, model_result in tree_result["feature_sets"]["core"]["models"].items():
        for partition in ("validation", "test"):
            metric = model_result[partition]
            counts = tree_result["feature_sets"]["core"]["partitions"][partition]
            rows.append({
                "feature_set": "core",
                "model": model_name,
                "partition": partition,
                "rows": counts["rows"],
                "firms": counts["firms"],
                "positive_endpoints": counts["positive_rows"],
                "positive_firms": counts["positive_firms"],
                "auroc": metric["auroc"],
                "average_precision": metric["average_precision"],
                "brier_score": metric["brier_score"],
                "top_decile_default_capture": metric["top_decile_default_capture"],
            })
    for partition in ("validation", "test"):
        metric = lstm_result["metrics"][partition]
        counts = lstm_result["partitions"][partition]
        rows.append({
            "feature_set": "core",
            "model": "lstm",
            "partition": partition,
            "rows": counts["sequences"],
            "firms": counts["firms"],
            "positive_endpoints": counts["positive_endpoints"],
            "positive_firms": counts["positive_firms"],
            "auroc": metric["auroc"],
            "average_precision": metric["average_precision"],
            "brier_score": metric["brier_score"],
            "top_decile_default_capture": metric["top_decile_default_capture"],
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    summary = {
        "evaluation_type": "fixed out-of-time pipeline comparison",
        "dataset": str(panel_path.resolve()),
        "feature_set_compared_across_families": "core",
        "tree_also_reports": "core_plus_structural (tree models only)",
        "sequence_window_months": window,
        "lstm_epochs": epochs,
        "split": tree_result["split"],
        "tree_results": "trees/metrics.json",
        "lstm_results": "lstm/metrics.json",
        "comparison_table": "comparison.csv",
        "warning": (
            "Smoke comparison only. The positive test set contains four firms, firms can recur "
            "across time partitions, and LSTM full-window eligibility differs from snapshot trees. "
            "Do not treat these metrics as reliable evidence or select a final model from them."
        ),
        "out_of_industry_status": (
            "Not run: the packaged panel has no SIC/industry column. Add a dated or historically "
            "appropriate firm-to-industry mapping, choose held-out industries before examining "
            "test outcomes, then implement the OOI split as a separate evaluation."
        ),
    }
    (out_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    readme = f"""# Shared out-of-time comparison

This run orchestrates the tree and LSTM pipeline checks using the same packaged panel and temporal cutoffs. It writes model-specific metrics plus `comparison.csv` for the common core feature set.

- Tree snapshots score each eligible firm-month. LSTM scores only months with a complete {window}-month continuous history; see `lstm/metrics.json` for endpoint coverage.
- Tree runs also include the structural feature set, which is not yet included in the LSTM comparison.
- Split: train through {tree_result['split']['train_end']}, validate in {tree_result['split']['validation_start'][:4]}, test from {tree_result['split']['test_start'][:4]}, with 2019 and 2021 purged for the 12-month label horizon.
- Firms may occur across time partitions, so this is not an unseen-company test.
- Current test data has only four positive firms. This is a pipeline smoke comparison, not a reliable performance evaluation.

## Out-of-industry evaluation

Not run: the packaged panel contains no SIC or industry field. Before an OOI experiment, join a historically appropriate firm-to-industry mapping, predeclare held-out industries, and report held-out firm and event counts. Do not infer industry from target outcomes or use current classifications without noting survivorship/reclassification limitations.

See `evaluation.json`, `comparison.csv`, and the nested model artifacts.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("artifacts/edgar_bankruptcy_v1/bankruptcy_panel.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/evaluation_oot"))
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run(args.panel, args.out_dir, window=args.window, epochs=args.epochs)
    print(f"Shared comparison written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
