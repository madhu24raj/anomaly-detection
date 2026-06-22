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
    p.add_argument("--contamination", type=float, default=None,
                   help="Expected anomaly fraction for IF/LOF (default: auto-estimated from data)")
    p.add_argument("--interval",      type=float, default=None,
                   help="Ping cadence in hours, e.g. 0.0833 = 5 min (default: auto-detected from data)")
    p.add_argument("--no-eval",       action="store_true",
                   help="Skip evaluation (don't look at ground-truth columns)")
    p.add_argument("--cache-features", default=None, metavar="PATH",
                   help="Cache/load feature matrix as parquet")
    p.add_argument("--chunksize",     type=int, default=None,
                   help="Stream CSV in chunks of this many rows (use for 50k+ "
                        "vessel runs to avoid memory spikes, e.g. --chunksize 5000000)")
    return p.parse_args(argv)


DTYPE_MAP = {
    "entity_id":   "int32",
    "lat":         "float32",
    "lon":         "float32",
    "vessel_type": "category",
    "sog_knots":   "float32",
    "is_anomalous": "int8",
    "anomaly_type": "category",
}


def load_data(path: str, use_labels: bool = True,
              chunksize: int | None = None) -> pd.DataFrame:
    """
    Load the AIS CSV with memory-efficient dtypes.

    For large file, use chunksize to stream
    """
    print(f"Loading {path} :")

    cols_to_drop = [] if use_labels else ["is_anomalous", "anomaly_type"]

    # Auto-enable 
    if chunksize is None and path.endswith((".gz", ".csv.gz")):
        chunksize = 2_000_000
        print(f"  (auto-enabling --chunksize {chunksize:,} for compressed input)")

    if chunksize:
        print(f"  Streaming in chunks of {chunksize:,} rows")
        chunks = []
        n_rows = 0
        t_start = time.time()
        for i, chunk in enumerate(pd.read_csv(path, dtype=DTYPE_MAP,
                                              parse_dates=["timestamp"],
                                              chunksize=chunksize)):
            chunk = chunk.drop(columns=cols_to_drop, errors="ignore")
            chunks.append(chunk)
            n_rows += len(chunk)
            elapsed = time.time() - t_start
            print(f"     chunk {i+1}: {n_rows:,} rows read so far "
                  f"[{elapsed:.0f}s elapsed, {n_rows/max(elapsed,0.01):,.0f} rows/s]",
                  flush=True)
        print("  Concatenating chunks", flush=True)
        df = pd.concat(chunks, ignore_index=True)
        del chunks
    else:
        df = pd.read_csv(path, dtype=DTYPE_MAP, parse_dates=["timestamp"])
        df = df.drop(columns=cols_to_drop, errors="ignore")

    print(f"  {len(df):,} pings, {df['entity_id'].nunique():,} contacts, "
          f"columns: {list(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    return df


def main(argv=None):
    args = parse_args(argv)
    t0   = time.time()

    df = load_data(args.input, use_labels=not args.no_eval, chunksize=args.chunksize)

    interval_h    = args.interval       # may be None 
    contamination = args.contamination  # may be None 

    if args.cache_features and os.path.exists(args.cache_features):
        print(f"Loading cached features from {args.cache_features}")
        feat = pd.read_parquet(args.cache_features)

        # re-derive Parquet cache if not supplied 
        if interval_h is None or contamination is None:
            from features import estimate_interval_h, estimate_contamination
            if interval_h is None:
                interval_h = estimate_interval_h(df)
                print(f"  Auto-detected ping interval : {interval_h*60:.1f} min")
            if contamination is None:
                contamination = estimate_contamination(df, interval_h)
                print(f"  Auto-estimated contamination: {contamination:.3f}")
    else:
        print("\nExtracting features")
        feat, interval_h, contamination = build_feature_matrix(
            df, nominal_interval_h=interval_h)
        if args.cache_features:
            feat.to_parquet(args.cache_features)
            print(f"  Features cached → {args.cache_features}")

    print(f"  Feature matrix: {feat.shape[0]} contacts × "
          f"{feat.shape[1]} features  [{time.time()-t0:.1f}s]")
    print(f"  Using interval={interval_h*60:.1f} min, contamination={contamination:.3f}")

    # ── Detection ─────────────────────────────────────────────────────────
    print("\nRunning detectors")
    results = detect(feat,
                     nominal_interval_h=interval_h,
                     contamination=contamination,
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