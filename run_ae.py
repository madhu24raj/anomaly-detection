#!/usr/bin/env python3
"""
run_ae.py  —  Autoencoder-based maritime anomaly detector.

Mirrors the CLI of run.py exactly so results can be compared apples-to-apples.

Architecture
------------
Five fully-connected autoencoders trained unsupervised on behavioural feature
groups, plus an optional LSTM sliding-window AE on raw ping sequences:

  MotionAE    → AIS spoofing (impossible speeds), erratic pursuit
  SpatialAE   → illegal fishing (stratum loitering)
  ProximityAE → transshipment & aggressive maneuvering (conjunction features)
  GapAE       → dark activity (no upper-clip to preserve the gap signal)
  FullAE      → cross-group correlation anomalies
  LSTM-AE     → temporal episode patterns (optional, GPU recommended)

Key engineering decisions
-------------------------
  * No truth labels in training — labels only used post-hoc for evaluation.
  * log1p-transform heavy-tailed features (std_speed_kt range 0.6–654 kt;
    heading_variance 15–24k; max_gap_h 0.08–96 h) before scaling.
  * GapAE skips the upper-percentile clip — dark_activity vessels are 0.45%
    of the fleet and their extreme gap values sit right at the clip boundary.
  * ProximityAE adds engineered conjunction features (co_slow_hours ×
    sustained_episodes; co_slow / min_nn_dist) to separate transshipment
    from normal vessels sharing busy shipping lanes.
  * LSTM aggregates per-vessel at 90th-percentile of window errors (not max)
    so one noisy window doesn't false-flag a normal vessel.
  * Sub-scores soft-capped with tanh(2x) before max-pool fusion.

Usage
-----
  python run_ae.py ais_small.csv
  python run_ae.py ais_small.csv --threshold 0.30 --output ae_scores.csv
  python run_ae.py ais_small.csv --compare              # side-by-side vs RB+IF/LOF
  python run_ae.py ais_small.csv --no-lstm --quiet      # fast mode
  python run_ae.py ais_small.csv --cache-features f.parquet  # reuse cached features

Threshold guidance
------------------
  The AE scores are soft-capped to [0,1) via tanh.  The optimal operating point
  depends on your precision/recall trade-off:
    0.20  →  very high recall (~100%), lower precision (~30%)
    0.30  →  high recall (~97%), moderate precision (~50-60%)
    0.50  →  balanced (default, similar operating point to the rule-based pipeline)
  Run with --compare to see threshold sweeps for both pipelines simultaneously.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from features     import build_feature_matrix, estimate_interval_h, estimate_contamination
from ae_detectors import ae_detect
from evaluate     import evaluate, per_type_breakdown, top_detections, print_report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Maritime AE anomaly detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input",             help="AIS CSV path")
    p.add_argument("--output",          default=None,  help="Write scored results to CSV")
    p.add_argument("--threshold",       type=float, default=0.50,
                   help="Ensemble score cutoff for flagging (default 0.50; try 0.20-0.30 for higher recall)")
    p.add_argument("--interval",        type=float, default=None,
                   help="Ping cadence in hours (default: auto-detected)")
    p.add_argument("--no-lstm",         action="store_true",
                   help="Skip the LSTM sequence AE (faster, recommended on CPU)")
    p.add_argument("--no-eval",         action="store_true",
                   help="Skip evaluation against ground-truth labels")
    p.add_argument("--cache-features",  default=None, metavar="PATH",
                   help="Cache / reuse feature matrix as Parquet (compatible with run.py)")
    p.add_argument("--compare",         action="store_true",
                   help="Also run rule-based+IF/LOF pipeline and print side-by-side comparison")
    p.add_argument("--chunksize",       type=int, default=None,
                   help="Stream CSV in chunks (for very large files)")
    p.add_argument("--quiet",           action="store_true",
                   help="Suppress per-epoch training output")
    return p.parse_args(argv)


DTYPE_MAP = {
    "entity_id":    "int32",
    "lat":          "float32",
    "lon":          "float32",
    "vessel_type":  "category",
    "sog_knots":    "float32",
    "is_anomalous": "int8",
    "anomaly_type": "category",
}


def load_data(path: str, use_labels: bool = True,
              chunksize: Optional[int] = None) -> pd.DataFrame:
    print(f"Loading {path} …")
    drop = [] if use_labels else ["is_anomalous", "anomaly_type"]
    if chunksize is None and path.endswith((".gz", ".csv.gz")):
        chunksize = 2_000_000
    if chunksize:
        chunks, n = [], 0
        t0 = time.time()
        for i, chunk in enumerate(
                pd.read_csv(path, dtype=DTYPE_MAP, parse_dates=["timestamp"],
                            chunksize=chunksize)):
            chunks.append(chunk.drop(columns=drop, errors="ignore"))
            n += len(chunks[-1])
            print(f"     chunk {i+1}: {n:,} rows  [{time.time()-t0:.0f}s]", flush=True)
        df = pd.concat(chunks, ignore_index=True)
        del chunks
    else:
        df = pd.read_csv(path, dtype=DTYPE_MAP, parse_dates=["timestamp"])
        df = df.drop(columns=drop, errors="ignore")
    print(f"  {len(df):,} pings, {df['entity_id'].nunique():,} contacts, "
          f"cols: {list(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_ae_report(metrics: dict, by_type: pd.DataFrame,
                    top: pd.DataFrame, tag: str = "AE-Ensemble") -> None:
    w = 62
    print(f"\n{'='*w}")
    print(f"  [{tag}] DETECTION RESULTS")
    print(f"{'='*w}")
    print(f"  Contacts evaluated : {metrics['n_contacts']}")
    print(f"  True anomalous     : {metrics['n_true_anomalous']}")
    print(f"  Flagged            : {metrics['n_flagged']}")
    print()
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  Recall             : {metrics['recall']:.4f}")
    print(f"  F1                 : {metrics['f1']:.4f}")
    print(f"  AUROC              : {metrics['auroc']:.4f}")
    print(f"  Avg Precision      : {metrics['average_precision']:.4f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} TN={metrics['tn']}")
    if not by_type.empty:
        print(f"\n  [{tag}] RECALL BY ANOMALY TYPE")
        print(by_type.to_string(index=False))
    print(f"\n  [{tag}] TOP {len(top)} CONTACTS BY SCORE")
    with pd.option_context("display.max_columns", None, "display.width", 130,
                           "display.float_format", "{:.3f}".format):
        print(top.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    t0   = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu" and not args.no_lstm:
        print("  (tip: LSTM AE is slow on CPU — add --no-lstm to skip it)")

    # ── Load ─────────────────────────────────────────────────────────────────
    df = load_data(args.input, use_labels=not args.no_eval,
                   chunksize=args.chunksize)
    raw_df_for_lstm = df.copy() if not args.no_lstm else None

    interval_h = args.interval

    # ── Features ─────────────────────────────────────────────────────────────
    if args.cache_features and os.path.exists(args.cache_features):
        print(f"\nLoading cached features from {args.cache_features}")
        feat = pd.read_parquet(args.cache_features)
        if interval_h is None:
            interval_h = estimate_interval_h(df)
            print(f"  Auto-detected ping interval: {interval_h*60:.1f} min")
    else:
        print("\nExtracting features …")
        feat, interval_h, contamination = build_feature_matrix(
            df, nominal_interval_h=interval_h)
        if args.cache_features:
            feat.to_parquet(args.cache_features)
            print(f"  Features cached → {args.cache_features}")

    print(f"  Feature matrix: {feat.shape[0]} contacts × {feat.shape[1]} features  "
          f"[{time.time()-t0:.1f}s]")

    # ── AE Detection ─────────────────────────────────────────────────────────
    print(f"\nRunning Autoencoder detectors  (device={device}) …")
    verbose = not args.quiet
    ae_results = ae_detect(
        feat,
        raw_df    = raw_df_for_lstm,
        use_lstm  = not args.no_lstm,
        threshold = args.threshold,
        device    = device,
        verbose   = verbose,
    )
    print(f"  Done  [{time.time()-t0:.1f}s]")

    # ── Evaluation ───────────────────────────────────────────────────────────
    if not args.no_eval and "true_anomalous" in ae_results.columns:
        ae_metrics = evaluate(ae_results, threshold=args.threshold)
        ae_by_type = per_type_breakdown(ae_results, threshold=args.threshold)
        ae_display = ae_results.rename(columns={
            c: c.replace("ae_score_", "score_") for c in ae_results.columns
            if c.startswith("ae_score_")
        })
        ae_top = top_detections(ae_display, n=20)
        print_ae_report(ae_metrics, ae_by_type, ae_top, tag="AE-Ensemble")
    else:
        print(f"\nTop flagged contacts (threshold={args.threshold}):")
        print(ae_results.sort_values("ensemble_score", ascending=False).head(20).to_string())

    # ── Optional comparison vs rule-based + IF/LOF ───────────────────────────
    if args.compare:
        print(f"\n{'='*62}")
        print("  Running rule-based + IF/LOF pipeline for comparison …")
        print(f"{'='*62}")
        from detectors import detect as rb_detect

        contamination = estimate_contamination(df, interval_h)
        rb_results = rb_detect(feat,
                               nominal_interval_h=interval_h,
                               contamination=contamination,
                               threshold=0.50)

        if not args.no_eval and "true_anomalous" in rb_results.columns:
            rb_metrics = evaluate(rb_results, threshold=0.50)
            rb_by_type = per_type_breakdown(rb_results, threshold=0.50)
            rb_top     = top_detections(rb_results, n=20)
            print_report(rb_metrics, rb_by_type, rb_top)

            if not args.no_eval and "true_anomalous" in ae_results.columns:
                ae_metrics = evaluate(ae_results, threshold=args.threshold)
                print(f"\n{'='*62}")
                print("  HEAD-TO-HEAD SUMMARY")
                print(f"  (AE threshold={args.threshold}  |  RB threshold=0.50)")
                print(f"{'='*62}")
                print(f"  {'Metric':<22} {'Rule-based+IF/LOF':>20} {'AE-Ensemble':>14}")
                print(f"  {'-'*56}")
                for k in ("precision", "recall", "f1", "auroc", "average_precision"):
                    rb_v = rb_metrics.get(k, float("nan"))
                    ae_v = ae_metrics.get(k, float("nan"))
                    winner = "  ◄ AE" if ae_v > rb_v else ("  ◄ RB" if rb_v > ae_v else "")
                    print(f"  {k:<22} {rb_v:>20.4f} {ae_v:>14.4f}{winner}")

                # Threshold sweep for AE
                print(f"\n  AE threshold sweep (AUROC = {ae_metrics['auroc']:.4f}):")
                print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Flagged':>8}")
                for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
                    m = evaluate(ae_results, threshold=t)
                    print(f"  {t:>10.2f} {m['precision']:>10.3f} {m['recall']:>8.3f} "
                          f"{m['f1']:>8.3f} {m['n_flagged']:>8}")

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.output:
        ae_results.to_csv(args.output)
        print(f"\nScores written → {args.output}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    return ae_results


if __name__ == "__main__":
    main()