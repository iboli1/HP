"""
Codabench scoring program, to try with the development set to ensure the formatting of the output is correct.

Reads:
    ground truth
    submission example

Writes:
    scores.txt         (metric: value, one per line)

The submission must contain a "pred_label" column holding the predicted
category directly (e.g. "CONSPIRACY" / "CRITICAL").

Any format problem with the submission causes a non-zero exit and an
explanation on stderr, which Codabench surfaces to the participant as
a failed submission.

Note: this targets the codalab/codalab-legacy:py3 image
(python 3.6, pandas==0.22.0, scikit-learn==0.19.1), so it avoids
Series.to_numpy() (added in pandas 0.24) and the zero_division kwarg
(added in sklearn 0.24).
"""

import os
import sys
import warnings

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

REF_DIR = "." # change folders if needed
RES_DIR = "."
OUT_DIR = "."

LABEL_COL = "category"          # ground-truth column name in test.csv
GOLD_FILE = "dev.csv"          # CHANGE NAME OF REFERENCE IF NECESSARY
PRED_FILE = "pred.csv"          # NEEDS TO HAVE THIS NAME EXACTLY AND THEN BE ZIPPED INTO A .zip

POS_LABEL = "CONSPIRACY"


def fail(message):
    """Print a clear error to stderr and stop with a non-zero exit code."""
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def compute_metrics(y_true, y_pred):
    """Computes standard performance metrics agnostic to how predictions were generated.

    Note: the installed sklearn version doesn't support the `zero_division`
    kwarg, so it's omitted here; sklearn already falls back to 0.0 (with a
    warning) for undefined precision/recall/F1, which we simply silence.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            # "MCC": matthews_corrcoef(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="binary", pos_label=POS_LABEL),
            "Recall": recall_score(y_true, y_pred, average="binary", pos_label=POS_LABEL),
            "Macro_F1": f1_score(y_true, y_pred, average="macro"),
            # "Weighted_F1": f1_score(y_true, y_pred, average="weighted"),
        }

    return metrics


def main():
    gold_path = os.path.join(REF_DIR, GOLD_FILE)
    if not os.path.exists(gold_path):
        fail(f"reference file '{GOLD_FILE}' not found in reference data.")

    gold_df = pd.read_csv(gold_path)
    if LABEL_COL not in gold_df.columns:
        fail(f"reference file is missing the '{LABEL_COL}' column.")

    gold_df["id"] = gold_df.index
    gold_df = gold_df[["id", LABEL_COL]]

    pred_path = os.path.join(RES_DIR, PRED_FILE)
    if not os.path.exists(pred_path):
        found = []
        for root, _, files in os.walk(RES_DIR):
            found.extend(os.path.join(root, f) for f in files)
        fail(
            f"expected '{PRED_FILE}' at the top level of your submission zip. "
            f"Found instead: {found}"
        )

    try:
        pred_df = pd.read_csv(pred_path)
    except Exception as e:
        fail(f"could not parse '{PRED_FILE}' as CSV: {e}")

    required_cols = {"id", "pred_label"}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        fail(f"submission is missing required column(s): {sorted(missing_cols)}")

    if pred_df["id"].duplicated().any():
        dupes = pred_df.loc[pred_df["id"].duplicated(), "id"].tolist()
        fail(f"submission contains duplicate id(s): {dupes[:10]}")

    pred_df = pred_df[["id", "pred_label"]]

    merged = pd.merge(gold_df, pred_df, on="id", how="inner")

    if len(merged) != len(gold_df):
        missing_ids = set(gold_df["id"]) - set(pred_df["id"])
        fail(
            f"submission covers {len(merged)}/{len(gold_df)} test ids. "
            f"Missing id(s) include: {sorted(list(missing_ids))[:10]}"
        )

    y_true = merged[LABEL_COL].astype(str).tolist()
    y_pred = merged["pred_label"].astype(str).tolist()

    metrics = compute_metrics(y_true, y_pred)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "scores.txt"), "w") as f:
        for name, value in metrics.items():
            f.write(f"{name}: {value}\n")

    print("Scores written:", metrics)


if __name__ == "__main__":
    main()
