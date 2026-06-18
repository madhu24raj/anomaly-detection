#!/usr/bin/env python3
"""
run_detection.py — Entry point for the maritime anomaly detector.

Usage
-----
# Run on the small test dataset (prints metrics against ground truth):
python run_detection.py ais_small.csv

# Save scored output to a CSV:
python run_detection.py ais_small.csv --output scores.csv

# Tune threshold and contamination:
python run_detection.py ais_small.csv --threshold 0.45 --contamination 0.05

# Suppress ground-truth evaluation (simulates real deployment):
python run_detection.py ais_small.csv --no-eval

Options
-------
  --output PATH       Write full scored results to this CSV.
  --threshold FLOAT   Ensemble score cutoff for flagging (default 0.50).
  --contamination F   Expected anomaly fraction for IF/LOF (default 0.05).
  --interval FLOAT    Ping cadence in hours (default 0.25 = 15 min).
  --no-eval           Skip evaluation against ground-truth labels.
  --cache-features P  Cache the feature matrix to PATH (parquet) and reuse it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

# Local modules
sys.path.insert(0, os.path.dirname(__file__))
from features  import build_feature_matrix
from detectors import detect
from evaluate  import evaluate, per_type_breakdown, top_detections, print_report


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Maritime contact anomaly detector.")
    p.add_argument("input", help="Path to AIS CSV (entity_id, lat, lon, timestamp [, labels])")
    p.add_argument("--output",        default=None,  help="Write scored results to CSV")
    p.add_argument("--threshold",     type=float, default=0.50)
    p.add_argument("--contamination", type=float, default=0.05,
                   help="Expected anomaly fraction (used by IF/LOF)")
    p.add_argument("--interval",      type=float, default=0.25,
                   help="Ping cadence in hours (0.25 = 15 min)")
    p.add_argument("--no-eval",       action="store_true",
                   help="Skip evaluation (don't look at ground-truth columns)")
    p.add_argument("--cache-features", default=None, metavar="PATH",
                   help="Cache/load feature matrix as parquet")
    return p.parse_args(argv)


def load_data(path: str, use_labels: bool = True) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} pings, {df['entity_id'].nunique():,} contacts, "
          f"columns: {list(df.columns)}")

    if not use_labels:
        df = df.drop(columns=["is_anomalous", "anomaly_type"], errors="ignore")

    return df


def main(argv=None):
    args = parse_args(argv)
    t0   = time.time()

    df = load_data(args.input, use_labels=not args.no_eval)

    # ── Feature extraction ────────────────────────────────────────────────
    if args.cache_features and os.path.exists(args.cache_features):
        print(f"Loading cached features from {args.cache_features}")
        feat = pd.read_parquet(args.cache_features)
    else:
        print("\nExtracting features...")
        feat = build_feature_matrix(df, nominal_interval_h=args.interval)
        if args.cache_features:
            feat.to_parquet(args.cache_features)
            print(f"  Features cached → {args.cache_features}")

    print(f"  Feature matrix: {feat.shape[0]} contacts × "
          f"{feat.shape[1]} features  [{time.time()-t0:.1f}s]")

    # ── Detection ─────────────────────────────────────────────────────────
    print("\nRunning detectors...")
    results = detect(feat,
                     nominal_interval_h=args.interval,
                     contamination=args.contamination,
                     threshold=args.threshold)
    print(f"  Done  [{time.time()-t0:.1f}s]")

    # ── Evaluation ────────────────────────────────────────────────────────
    if not args.no_eval and "true_anomalous" in results.columns:
        metrics  = evaluate(results, threshold=args.threshold)
        by_type  = per_type_breakdown(results, threshold=args.threshold)
        top      = top_detections(results, n=20)
        print_report(metrics, by_type, top)
    else:
        top = top_detections(results, n=20)
        print(f"\nTop flagged contacts (threshold={args.threshold}):")
        print(top.to_string())

    # ── Save ──────────────────────────────────────────────────────────────
    if args.output:
        results.to_csv(args.output)
        print(f"Scores written → {args.output}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    return results


if __name__ == "__main__":
    main()
