"""
AIS Vessel Stay Detection & Anomaly Detection
Data schema: entity_id, lat, lon, timestamp, vessel_type, sog_knots, is_anomalous, anomaly_type

Anomaly scoring: Ramaswamy et al. KNN outlier score (average distance to k nearest
neighbours in feature space). Higher score = more outlying.

Usage:
    df = pd.read_csv('ais_small.csv')
    results = run_pipeline(df)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from scipy.stats import percentileofscore


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
STAY_RADIUS_KM         = 0.5
STAY_MIN_DURATION_M    = 30
SPEED_STAY_THRESH      = 1.0
AIS_GAP_THRESHOLD_MIN  = 60
SPOOF_DIST_KM          = 50
SPOOF_IMPLIED_KNOTS    = 60
MIN_GROUP_SIZE_FOR_TYPE_MODEL = 50

# KNN parameters (Ramaswamy method)
KNN_K                  = 10    # number of neighbours; 5–20 is typical
                                # larger k → smoother scores, less sensitive to micro-clusters
                                # smaller k → more sensitive to local structure

# Threshold tuning (no-label method): use the p-th percentile of scores as cut-off.
# 97th percentile ≈ top 3 % flagged — adjust if you want more/fewer alerts.
# Do NOT tune this by peeking at is_anomalous.
SCORE_PERCENTILE_THRESH = 97   # e.g. 95 → more recalls, 99 → more precision


# ── HELPERS ──────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ── STAY DETECTION ────────────────────────────────────────────────────────────
def detect_stays_sliding_window(group: pd.DataFrame) -> pd.DataFrame:
    df = group.sort_values("timestamp").copy()
    n  = len(df)
    df["is_stay"]  = False
    df["stay_id"]  = -1

    lats   = df["lat"].values
    lons   = df["lon"].values
    times  = pd.to_datetime(df["timestamp"]).values.astype("int64") // 1_000_000_000
    speeds = df["sog_knots"].values

    stay_id = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            clat  = lats[i:j+1].mean()
            clon  = lons[i:j+1].mean()
            dists = haversine_km(lats[i:j+1], lons[i:j+1], clat, clon)
            if dists.max() > STAY_RADIUS_KM:
                break
            duration_min = (times[j] - times[i]) / 60.0
            if duration_min >= STAY_MIN_DURATION_M:
                df.iloc[i:j+1, df.columns.get_loc("is_stay")]  = True
                df.iloc[i:j+1, df.columns.get_loc("stay_id")] = stay_id
            j += 1

        if df.iloc[i]["stay_id"] >= 0:
            last_idx = df[df["stay_id"] == stay_id].index[-1]
            pos      = df.index.get_loc(last_idx)
            stay_id += 1
            i = pos + 1
        else:
            i += 1

    df["stay_center_lat"]   = np.nan
    df["stay_center_lon"]   = np.nan
    df["stay_duration_min"] = np.nan

    for sid, grp in df[df["stay_id"] >= 0].groupby("stay_id"):
        t_sorted = pd.to_datetime(grp["timestamp"])
        dur      = (t_sorted.max() - t_sorted.min()).total_seconds() / 60
        df.loc[grp.index, "stay_center_lat"]   = grp["lat"].mean()
        df.loc[grp.index, "stay_center_lon"]   = grp["lon"].mean()
        df.loc[grp.index, "stay_duration_min"] = dur

    df["low_speed_flag"] = speeds < SPEED_STAY_THRESH
    return df


def run_stay_detection(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for entity_id, group in df.groupby("entity_id"):
        g = detect_stays_sliding_window(group)
        g["entity_id"] = entity_id
        results.append(g)
    return pd.concat(results, ignore_index=True)


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"])

    df["time_gap_min"] = (
        df.groupby("entity_id")["timestamp"]
          .diff().dt.total_seconds().div(60).fillna(0)
    )

    df["speed_delta"] = df.groupby("entity_id")["sog_knots"].diff().fillna(0)

    def z_score(s):
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0

    df["speed_z"] = df.groupby("entity_id")["sog_knots"].transform(z_score)

    df["prev_lat"] = df.groupby("entity_id")["lat"].shift(1)
    df["prev_lon"] = df.groupby("entity_id")["lon"].shift(1)
    valid = df["prev_lat"].notna()
    df.loc[valid, "dist_from_prev_km"] = haversine_km(
        df.loc[valid, "lat"], df.loc[valid, "lon"],
        df.loc[valid, "prev_lat"], df.loc[valid, "prev_lon"]
    )
    df["dist_from_prev_km"] = df["dist_from_prev_km"].fillna(0)

    def bearing(lat1, lon1, lat2, lon2):
        dlon   = np.radians(lon2 - lon1)
        lat1r, lat2r = np.radians(lat1), np.radians(lat2)
        x = np.sin(dlon) * np.cos(lat2r)
        y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360

    df.loc[valid, "bearing"] = bearing(
        df.loc[valid, "prev_lat"], df.loc[valid, "prev_lon"],
        df.loc[valid, "lat"],      df.loc[valid, "lon"]
    )
    df["bearing"]      = df["bearing"].fillna(0)
    df["prev_bearing"] = df.groupby("entity_id")["bearing"].shift(1).fillna(0)
    df["bearing_delta"] = ((df["bearing"] - df["prev_bearing"] + 180) % 360) - 180

    med = df.groupby("entity_id")[["lat", "lon"]].median().rename(
        columns={"lat": "med_lat", "lon": "med_lon"}
    )
    df = df.join(med, on="entity_id")
    df["dist_from_median_km"] = haversine_km(
        df["lat"], df["lon"], df["med_lat"], df["med_lon"]
    )

    df["stay_duration_min"] = df["stay_duration_min"].fillna(0)
    df["is_stay_int"]       = df["is_stay"].astype(int)
    df["hour_of_day"]       = df["timestamp"].dt.hour
    df["day_of_week"]       = df["timestamp"].dt.dayofweek

    # Cap large gaps so they don't dominate distance metric
    df["time_gap_min_capped"] = df["time_gap_min"].clip(upper=240)

    df.drop(columns=["prev_lat", "prev_lon", "prev_bearing", "med_lat", "med_lon"],
            inplace=True)
    return df


# ── RAMASWAMY KNN OUTLIER SCORING ────────────────────────────────────────────
# Reference: Ramaswamy et al. (2000) "Efficient Algorithms for Mining Outliers
# from Large Data Sets." SIGMOD Record.
#
# Score(p) = mean distance to p's k nearest neighbours (D^k_n in the paper).
# Points far from all their neighbours → high score → outlier.
# This is more locally adaptive than Isolation Forest because it measures
# actual distance rather than partitioning depth, so it handles clusters of
# varying density (different vessel-type behavioural patterns) better.

FEATURE_COLS = [
    "speed_z",
    "speed_delta",
    "dist_from_prev_km",
    "bearing_delta",
    "dist_from_median_km",
    "stay_duration_min",
    "is_stay_int",
    "time_gap_min_capped",   # capped version keeps outlier gaps salient
    "hour_of_day",
]


def _knn_outlier_scores(X_scaled: np.ndarray, k: int) -> np.ndarray:
    """
    Ramaswamy KNN outlier score: mean distance to k nearest neighbours.
    Uses k+1 neighbours and drops self (index 0).
    """
    nn  = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn.fit(X_scaled)
    dists, _ = nn.kneighbors(X_scaled)
    return dists[:, 1:].mean(axis=1)   # exclude self-distance (column 0)


def train_knn_by_type(df: pd.DataFrame, k: int = KNN_K):
    """
    Fit one StandardScaler per vessel_type (KNN is non-parametric; we only
    store the scaler + training data needed to score new points).

    Returns:
        scalers : dict[vtype -> StandardScaler]
        Xfit    : dict[vtype -> scaled training array]  (needed at score time)
    """
    scalers, Xfit = {}, {}

    X_all = df[FEATURE_COLS].fillna(0).values
    gs    = StandardScaler().fit(X_all)
    scalers["__global__"] = gs
    Xfit["__global__"]    = gs.transform(X_all)

    for vtype, grp in df.groupby("vessel_type"):
        if len(grp) < MIN_GROUP_SIZE_FOR_TYPE_MODEL:
            continue
        X   = grp[FEATURE_COLS].fillna(0).values
        sc  = StandardScaler().fit(X)
        scalers[vtype] = sc
        Xfit[vtype]    = sc.transform(X)

    return scalers, Xfit


def score_knn_by_type(df: pd.DataFrame, scalers, Xfit,
                      k: int = KNN_K) -> pd.DataFrame:
    """
    Compute Ramaswamy KNN scores per vessel_type group.

    NOTE: for a production system you would fit KNN on a clean reference set
    and score new pings against it.  Here we fit-and-score on the same data
    (same as the draft IF approach) — fine for offline eval.

    Adds column:
        knn_outlier_score  (higher = more anomalous)
    """
    df = df.copy()
    df["knn_outlier_score"] = np.nan

    for vtype, grp in df.groupby("vessel_type"):
        key   = vtype if vtype in scalers else "__global__"
        X_q   = scalers[key].transform(grp[FEATURE_COLS].fillna(0).values)
        X_ref = Xfit[key]

        # Score query points against reference distribution
        nn  = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
        nn.fit(X_ref)
        dists, _ = nn.kneighbors(X_q)
        # If query IS in the reference (same-dataset eval), drop the self-hit
        scores = dists[:, 1:k+1].mean(axis=1)

        df.loc[grp.index, "knn_outlier_score"] = scores

    return df


# ── RULE-BASED LAYER ─────────────────────────────────────────────────────────
# Changes vs draft:
#   • rule_spoofing  : tightened — require BOTH large distance AND implied speed
#                      (was OR, which caused many false positives on long gaps
#                      between normal pings at anchor/port)
#   • rule_aggression: dropped speed_delta threshold from 15 → 20 kts AND
#                      bearing_delta from 120° → 150°; draft's thresholds were
#                      far too loose (precision = 0.03)
#   • rule_illegal_fish: added dist_from_median_km guard kept, but also require
#                      the vessel to be underway (sog > 0.5) to exclude
#                      anchored vessels that happen to be far from home
def rule_based_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dark ship: AIS gap
    df["rule_dark_ship"] = df["time_gap_min"] > AIS_GAP_THRESHOLD_MIN

    # Spoofing: implausible jump — require BOTH large distance AND high implied speed
    df["implied_speed_knots"] = np.where(
        df["time_gap_min"] > 0,
        (df["dist_from_prev_km"] / 1.852) / (df["time_gap_min"] / 60.0),
        0.0,
    )
    df["rule_spoofing"] = (
        (df["dist_from_prev_km"] > SPOOF_DIST_KM) &          # ← AND, not OR
        (df["implied_speed_knots"] > SPOOF_IMPLIED_KNOTS)
    )

    # Aggression: tighter thresholds to cut the 0.03-precision false-positive flood
    df["rule_aggression"] = (
        (df["speed_delta"].abs() > 20) &                      # was 15
        (df["bearing_delta"].abs() > 150)                     # was 120
    )

    # Illegal fishing: loitering far from home grounds, but actually moving
    df["rule_illegal_fish"] = (
        (df["vessel_type"] == "fishing") &
        (df["sog_knots"] > 0.5) &                            # exclude anchored
        (df["sog_knots"] < 2) &
        (df["dist_from_median_km"] > 20)
    )

    df["any_rule_flag"] = (
        df["rule_dark_ship"] | df["rule_spoofing"] |
        df["rule_aggression"] | df["rule_illegal_fish"]
    )
    return df


# ── THRESHOLD SELECTION (no label leakage) ───────────────────────────────────
def select_threshold(df: pd.DataFrame,
                     percentile: float = SCORE_PERCENTILE_THRESH) -> float:
    """
    Pick the KNN score cut-off at the given percentile of the score distribution.
    This is a label-free heuristic — it controls how many pings get flagged
    without peeking at is_anomalous.

    Tune SCORE_PERCENTILE_THRESH in constants:
        lower → more flags (higher recall, lower precision)
        higher → fewer flags (lower recall, higher precision)
    """
    return float(np.percentile(df["knn_outlier_score"].dropna(), percentile))


# ── EVALUATION ────────────────────────────────────────────────────────────────
def evaluate(df: pd.DataFrame, threshold: float):
    y_true = df["is_anomalous"].astype(int)
    y_pred = df["pred_anomalous"].astype(int)

    print("=" * 55)
    print("ANOMALY DETECTION — Classification Report")
    print(f"Threshold (KNN score p{SCORE_PERCENTILE_THRESH}): {threshold:.4f}")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"]))

    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}")

    print("\nRule Precision (vs ground truth, for diagnostics only):")
    for col, label in [
        ("rule_dark_ship",    "dark_ship"),
        ("rule_spoofing",     "spoofing"),
        ("rule_aggression",   "aggression"),
        ("rule_illegal_fish", "illegal_fish"),
    ]:
        if col in df.columns and df[col].sum() > 0:
            prec = (df[col] & (y_true == 1)).sum() / df[col].sum()
            print(f"  {label:20s}: {df[col].sum():6d} flagged | precision={prec:.2f}")

    # Oracle threshold (label-tuned, offline eval only)
    best_f1, best_t = 0.0, threshold
    for t in np.percentile(df["knn_outlier_score"].dropna(),
                           np.linspace(85, 99.9, 200)):
        yp = (
            (df["knn_outlier_score"] >= t) | df["any_rule_flag"]
        ).astype(int)
        f = f1_score(y_true, yp)
        if f > best_f1:
            best_f1, best_t = f, t

    print(f"\nOracle best F1 (label-tuned threshold={best_t:.4f}): {best_f1:.4f}")
    print("  ^ use this threshold only for offline eval, not production")

    print("\nScore distribution by anomaly type:")
    print(
        df.groupby("anomaly_type")["knn_outlier_score"]
          .agg(mean="mean", median="median",
               p75=lambda x: x.quantile(0.75), max="max")
          .rename(columns={"median": "50%", "p75": "75%"})
          .round(3)
    )

    return f1


# ── FULL PIPELINE ─────────────────────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame,
                 k: int = KNN_K,
                 score_percentile: float = SCORE_PERCENTILE_THRESH) -> pd.DataFrame:
    """
    1. Stay detection
    2. Rendezvous / co-location detection (placeholder hook)
    3. Feature engineering
    4. Train per-vessel-type KNN scalers
    5. Compute Ramaswamy KNN outlier scores
    6. Rule-based flags (tightened vs draft)
    7. Select score threshold (no label leakage)
    8. Evaluate

    Args:
        df               : raw AIS dataframe
        k                : KNN neighbours (default KNN_K)
        score_percentile : percentile cut-off, not tuned to ground truth
    """
    print("Step 1: Stay detection...")
    df = run_stay_detection(df)

    print("Step 2: Rendezvous detection...")
    # (hook for co-location / rendezvous detection — plug in here)

    print("Step 3: Feature engineering...")
    df = build_features(df)

    print("Step 4: Training per-type KNN scalers...")
    scalers, Xfit = train_knn_by_type(df, k=k)

    print("Step 5: Computing KNN outlier scores (Ramaswamy)...")
    df = score_knn_by_type(df, scalers, Xfit, k=k)

    print("Step 6: Computing rule votes...")
    df = rule_based_flags(df)

    print("Step 7: Selecting threshold (no label leakage)...")
    threshold = select_threshold(df, percentile=score_percentile)

    # Final prediction: KNN score above threshold OR any hard rule fires
    df["pred_anomalous"] = (
        (df["knn_outlier_score"] >= threshold) | df["any_rule_flag"]
    )

    print("Step 8: Evaluation...")
    evaluate(df, threshold)

    print("\nSample predicted anomalies:")
    print(
        df[df["pred_anomalous"]]
          [["entity_id", "timestamp", "vessel_type", "sog_knots",
            "knn_outlier_score", "is_anomalous", "anomaly_type"]]
          .head(15)
          .to_string(index=False)
    )

    return df


# ── QUICKSTART ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df      = pd.read_csv("ais_small.csv")
    results = run_pipeline(df)
