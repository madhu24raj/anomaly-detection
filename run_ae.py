#!/usr/bin/env python3
"""
run_ae.py 

Usage
# Run on a CSV and evaluate against ground truth:
python run_ae.py ais_small.csv

# Save scored output:
python run_ae.py ais_small.csv --output ae_scores.csv

# Skip LSTM AE (faster, lower memory):
python run_ae.py ais_small.csv --no-lstm

# Tune threshold:
python run_ae.py ais_small.csv --threshold 0.45

# Cache features (same Parquet file as run.py — fully compatible):
python run_ae.py ais_small.csv --cache-features feats.parquet

# Side-by-side comparison of rule-based vs AE:
python run_ae.py ais_small.csv --compare

Options
  --output PATH        Write scored results to CSV.
  --threshold FLOAT    Ensemble score cutoff (default 0.50).
  --interval FLOAT     Ping cadence hours (default auto-detected).
  --no-lstm            Skip the LSTM sequence autoencoder.
  --no-eval            Skip evaluation against ground-truth labels.
  --cache-features P   Cache / reuse feature Parquet (shares format with run.py).
  --compare            Also run the original rule-based + IF/LOF detectors and print both reports side by side.
  --chunksize N        Stream CSV in chunks of N rows (large files).
"""

import argparse
import os
import sys
import time
from typing import Optional

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from features   import build_feature_matrix, estimate_interval_h, estimate_contamination
from ae_detectors import ae_detect
from evaluate   import evaluate, per_type_breakdown, top_detections, print_report

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Maritime AE anomaly detector.")
    p.add_argument("input",  help="AIS CSV path")
    p.add_argument("--output",          default=None)
    p.add_argument("--threshold",       type=float, default=0.50)
    p.add_argument("--interval",        type=float, default=None)
    p.add_argument("--no-lstm",         action="store_true")
    p.add_argument("--no-eval",         action="store_true")
    p.add_argument("--cache-features",  default=None, metavar="PATH")
    p.add_argument("--compare",         action="store_true",
                   help="Also run rule-based+IF/LOF pipeline and compare")
    p.add_argument("--chunksize",       type=int, default=None)
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


def load_data(path: str, use_labels: bool = True, chunksize: Optional[int] = None) -> pd.DataFrame:
    print(f"Loading {path} ")
    cols_to_drop = [] if use_labels else ["is_anomalous", "anomaly_type"]

    if chunksize is None and path.endswith((".gz", ".csv.gz")):
        chunksize = 2_000_000
        print(f"  (auto-enabling --chunksize {chunksize:,} for compressed input)")

    if chunksize:
        chunks, n_rows = [], 0
        t0 = time.time()
        for i, chunk in enumerate(
                pd.read_csv(path, dtype=DTYPE_MAP, parse_dates=["timestamp"],
                            chunksize=chunksize)):
            chunk = chunk.drop(columns=cols_to_drop, errors="ignore")
            chunks.append(chunk)
            n_rows += len(chunk)
            print(f"     chunk {i+1}: {n_rows:,} rows  [{time.time()-t0:.0f}s]")
        df = pd.concat(chunks, ignore_index=True)
        del chunks
    else:
        df = pd.read_csv(path, dtype=DTYPE_MAP, parse_dates=["timestamp"])
        df = df.drop(columns=cols_to_drop, errors="ignore")

    print(f"  {len(df):,} pings, {df['entity_id'].nunique():,} contacts, "
          f"cols: {list(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
    return df


def _ae_score_cols(results: pd.DataFrame):
    return [c for c in results.columns if c.startswith("ae_score_")]


def print_ae_report(metrics: dict, by_type: pd.DataFrame,
                    top: pd.DataFrame, tag: str = "AE") -> None:
    print(f"  [{tag}] DETECTION RESULTS")
    print(f"  Contacts evaluated : {metrics['n_contacts']}")
    print(f"  True anomalous     : {metrics['n_true_anomalous']}")
    print(f"  Flagged            : {metrics['n_flagged']}")
    print()
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  F1         : {metrics['f1']:.4f}")
    print(f"  AUROC      : {metrics['auroc']:.4f}")
    print(f"  Avg Prec   : {metrics['average_precision']:.4f}")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} TN={metrics['tn']}")

    if not by_type.empty:
        print(f"\n  [{tag}] RECALL BY ANOMALY TYPE")
        print(by_type.to_string(index=False))

    print(f"\n  [{tag}] TOP {len(top)} CONTACTS BY SCORE")
    with pd.option_context("display.max_columns", None, "display.width", 130,
                           "display.float_format", "{:.3f}".format):
        print(top.to_string())


def main(argv=None):
    args = parse_args(argv)
    t0   = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_data(args.input, use_labels=not args.no_eval,
                   chunksize=args.chunksize)

    raw_df_for_lstm = df.copy() if not args.no_lstm else None

    interval_h = args.interval

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
            print(f"  Features cached -> {args.cache_features}")

    print(f"  Feature matrix: {feat.shape[0]} contacts × {feat.shape[1]} features  "
          f"[{time.time()-t0:.1f}s]")

    print(f"\nRunning Autoencoder detectors  (device={device}) …")
    verbose = not args.quiet
    ae_results = ae_detect(
        feat,
        raw_df        = raw_df_for_lstm,
        use_lstm      = not args.no_lstm,
        threshold     = args.threshold,
        device        = device,
        verbose       = verbose,
    )
    print(f"  Done  [{time.time()-t0:.1f}s]")

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

    if args.compare:
        print("  Running original rule-based + IF/LOF pipeline for comparison …")
        from detectors import detect as rb_detect
        from features  import estimate_contamination

        contamination = estimate_contamination(df, interval_h)
        rb_results = rb_detect(feat,
                               nominal_interval_h=interval_h,
                               contamination=contamination,
                               threshold=args.threshold)

        if not args.no_eval and "true_anomalous" in rb_results.columns:
            rb_metrics = evaluate(rb_results, threshold=args.threshold)
            rb_by_type = per_type_breakdown(rb_results, threshold=args.threshold)
            rb_top     = top_detections(rb_results, n=20)
            print_report(rb_metrics, rb_by_type, rb_top)

            if not args.no_eval and "true_anomalous" in ae_results.columns:
                print("  HEAD-TO-HEAD SUMMARY")
                print(f"  {'Metric':<22} {'Rule-based+IF/LOF':>20} {'AE-Ensemble':>15}")
                print(f"  {'-'*58}")
                for k in ("precision", "recall", "f1", "auroc", "average_precision"):
                    rb_v  = rb_metrics.get(k, float("nan"))
                    ae_v  = ae_metrics.get(k, float("nan"))
                    better = "  ◄ AE" if ae_v > rb_v else ("  ◄ RB" if rb_v > ae_v else "")
                    print(f"  {k:<22} {rb_v:>20.4f} {ae_v:>15.4f}{better}")

    if args.output:
        ae_results.to_csv(args.output)
        print(f"\nScores written -> {args.output}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    return ae_results


if __name__ == "__main__":
    main()