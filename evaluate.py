"""
evaluate.py — Post-hoc evaluation against ground-truth labels.

Only called AFTER detection is complete.  Nothing here feeds back into
the detectors — it is purely for measuring performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def evaluate(results: pd.DataFrame, threshold: float = 0.50) -> dict:
    """
    Compute detection metrics against ground truth.

    Requires columns: ensemble_score, true_anomalous (0/1).
    """
    if "true_anomalous" not in results.columns:
        print("No ground-truth column found — skipping evaluation.")
        return {}

    y_true  = results["true_anomalous"].astype(int).to_numpy()
    y_score = results["ensemble_score"].to_numpy()
    y_pred  = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision  = tp / (tp + fp + 1e-12)
    recall     = tp / (tp + fn + 1e-12)
    f1         = 2 * precision * recall / (precision + recall + 1e-12)
    auroc      = roc_auc_score(y_true, y_score) if y_true.sum() > 0 else float("nan")
    avg_prec   = average_precision_score(y_true, y_score) if y_true.sum() > 0 else float("nan")

    metrics = dict(
        threshold=threshold,
        n_contacts=len(results),
        n_true_anomalous=int(y_true.sum()),
        n_flagged=int(y_pred.sum()),
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        auroc=round(auroc, 4),
        average_precision=round(avg_prec, 4),
    )
    return metrics


def per_type_breakdown(results: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    """
    For each true anomaly type, show what fraction were flagged (recall by type)
    and which detector scored highest on average.
    """
    if "true_type" not in results.columns:
        return pd.DataFrame()

    score_cols = [c for c in results.columns if c.startswith("score_") and c != "ensemble_score"]
    rows = []
    for atype, grp in results.groupby("true_type"):
        n      = len(grp)
        flagged = int((grp["ensemble_score"] >= threshold).sum())
        row = {"anomaly_type": atype, "n": n, "flagged": flagged,
               "recall": round(flagged / n, 3)}
        for sc in score_cols:
            row[f"mean_{sc}"] = round(grp[sc].mean(), 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("recall", ascending=False)


def top_detections(results: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return top-n contacts by ensemble score with ground truth alongside."""
    cols = (["ensemble_score", "ensemble_mean"] +
            [c for c in results.columns if c.startswith("score_")] +
            [c for c in ["true_anomalous", "true_type"] if c in results.columns])
    return (results[cols]
            .sort_values("ensemble_score", ascending=False)
            .head(n))


def print_report(metrics: dict, by_type: pd.DataFrame, top: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)
    print(f"  Contacts evaluated : {metrics['n_contacts']}")
    print(f"  True anomalous     : {metrics['n_true_anomalous']} "
          f"({metrics['n_true_anomalous']/metrics['n_contacts']*100:.1f}%)")
    print(f"  Flagged by detector: {metrics['n_flagged']}")
    print()
    print(f"  Precision          : {metrics['precision']:.3f}")
    print(f"  Recall             : {metrics['recall']:.3f}")
    print(f"  F1                 : {metrics['f1']:.3f}")
    print(f"  AUROC              : {metrics['auroc']:.3f}")
    print(f"  Average Precision  : {metrics['average_precision']:.3f}")
    print()
    print(f"  Confusion: TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} TN={metrics['tn']}")

    if not by_type.empty:
        print("\n" + "-" * 60)
        print("RECALL BY ANOMALY TYPE")
        print("-" * 60)
        print(by_type.to_string(index=False))

    print("\n" + "-" * 60)
    print(f"TOP {len(top)} CONTACTS BY ANOMALY SCORE")
    print("-" * 60)
    with pd.option_context("display.max_columns", None, "display.width", 120,
                           "display.float_format", "{:.3f}".format):
        print(top.to_string())
    print("=" * 60 + "\n")
