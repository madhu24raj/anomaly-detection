"""
AIS Vessel Anomaly Detection  —  4-Tier Weak Supervision Pipeline  v3
======================================================================
Data schema: entity_id, lat, lon, timestamp, vessel_type,
             sog_knots, is_anomalous, anomaly_type

Architecture
────────────
Tier 1 │ Spatial/Agent Profiling
       │   KNN outlier score on per-ping spatial + speed features,
       │   computed per vessel_type so a fast cargo ship isn't flagged
       │   for being faster than a fishing boat.

Tier 2 │ Temporal/Sequence Profiling
       │   Discretize the map into a grid → vessel trajectory becomes
       │   a sequence of cell tokens.  Bigram Markov chain trained per
       │   vessel_type.  Score = negative log-likelihood (NLL) of each
       │   observed transition (high NLL = surprising = anomalous).

Tier 3 │ Domain Rules
       │   Physics-based hard rules: dark ship (AIS gap), spoofing
       │   (implied speed > type max), illegal fishing (slow + OOD
       │   location).  No tuning against labels.

Tier 4 │ Weak Supervision Meta-Learner
       │   (a) Generate pseudo-labels from Tiers 1-3:
       │       1  → rule fires  OR  (T1 score AND T2 NLL both in top 95th pctile)
       │       0  → T1 score AND T2 NLL both below 40th pctile
       │       NaN → uncertain / unlabeled
       │   (b) Train HistGradientBoostingClassifier on pseudo-labeled rows only.
       │   (c) Predict anomaly probability for ALL rows.
       │   (d) Evaluate vs ground-truth is_anomalous.

Usage
─────
    df = pd.read_csv('ais_small.csv')
    results = run_pipeline(df)
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
# Stay detection
STAY_RADIUS_KM       = 0.5
STAY_MIN_DURATION_M  = 30

# Tier 2 — grid resolution for Markov chain
MARKOV_GRID_DEG      = 0.5          # ~55 km cells (coarser = less sparsity)

# Tier 3 — domain rule thresholds
AIS_GAP_THRESH_MIN   = 60           # minutes of silence → dark ship
VTYPE_MAX_SPEED_KN   = {            # knots — physics ceiling per type
    "fishing":   20,
    "cargo":     28,
    "passenger": 35,
    "pleasure":  45,
    "tanker":    20,
}
DEFAULT_MAX_KN       = 40
OOD_DIST_KM          = 20           # km from vessel's median → OOD

# Tier 4 — pseudo-label thresholds (percentiles, no label leakage)
PSEUDO_POS_PERCENTILE  = 95         # top of BOTH T1 and T2 → label=1
PSEUDO_NEG_PERCENTILE  = 40         # bottom of BOTH T1 and T2 → label=0

# KNN
KNN_K                = 10
MIN_TYPE_ROWS        = 50           # below this, use global KNN model


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2):
    dlon  = np.radians(lon2 - lon1)
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r)*np.sin(lat2r) - np.sin(lat1r)*np.cos(lat2r)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def robust_minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return ((s - lo) / (hi - lo + 1e-9)).clip(0, 1)


# ══════════════════════════════════════════════════════════════
# STAY DETECTION  (used as a feature, not a tier)
# ══════════════════════════════════════════════════════════════
def _stays_for_group(group: pd.DataFrame) -> pd.DataFrame:
    df = group.sort_values("timestamp").copy()
    n  = len(df)
    df["is_stay"]   = False
    df["stay_id"]   = -1

    lats  = df["lat"].values
    lons  = df["lon"].values
    times = pd.to_datetime(df["timestamp"]).values.astype("int64") / 1e9

    stay_id = 0
    i = 0
    while i < n:
        last_good_j = i
        j = i + 1
        while j < n:
            clat  = lats[i:j+1].mean()
            clon  = lons[i:j+1].mean()
            dists = haversine_km(lats[i:j+1], lons[i:j+1], clat, clon)
            if dists.max() > STAY_RADIUS_KM:
                break
            if (times[j] - times[i]) / 60.0 >= STAY_MIN_DURATION_M:
                last_good_j = j
            j += 1

        if last_good_j > i:
            df.iloc[i:last_good_j+1, df.columns.get_loc("is_stay")]  = True
            df.iloc[i:last_good_j+1, df.columns.get_loc("stay_id")] = stay_id
            stay_id += 1
            i = last_good_j + 1
        else:
            i += 1

    df["stay_duration_min"] = np.nan
    for sid, grp in df[df["stay_id"] >= 0].groupby("stay_id"):
        ts  = pd.to_datetime(grp["timestamp"])
        dur = (ts.max() - ts.min()).total_seconds() / 60
        df.loc[grp.index, "stay_duration_min"] = dur
        df.loc[grp.index, "stay_center_lat"]   = grp["lat"].mean()
        df.loc[grp.index, "stay_center_lon"]   = grp["lon"].mean()

    df["stay_center_lat"]   = df.get("stay_center_lat",   np.nan)
    df["stay_center_lon"]   = df.get("stay_center_lon",   np.nan)
    df["stay_duration_min"] = df["stay_duration_min"].fillna(0)
    df["is_stay_int"]       = df["is_stay"].astype(int)
    return df


def run_stay_detection(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for eid, grp in df.groupby("entity_id"):
        g = _stays_for_group(grp)
        g["entity_id"] = eid
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ══════════════════════════════════════════════════════════════
# BASE FEATURE ENGINEERING  (shared input to all tiers)
# ══════════════════════════════════════════════════════════════
def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-ping movement features used by Tier 1, Tier 2, and the
    meta-learner.  All operations are causal (shift(1) before rolling).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    # ── time gap
    df["time_gap_min"] = (
        df.groupby("entity_id")["timestamp"]
          .diff().dt.total_seconds().div(60).fillna(0)
    )

    # ── prev position
    df["prev_lat"] = df.groupby("entity_id")["lat"].shift(1)
    df["prev_lon"] = df.groupby("entity_id")["lon"].shift(1)
    valid = df["prev_lat"].notna()

    df.loc[valid, "dist_km"] = haversine_km(
        df.loc[valid, "lat"],      df.loc[valid, "lon"],
        df.loc[valid, "prev_lat"], df.loc[valid, "prev_lon"],
    )
    df["dist_km"] = df["dist_km"].fillna(0)

    df.loc[valid, "bearing"] = bearing_deg(
        df.loc[valid, "prev_lat"], df.loc[valid, "prev_lon"],
        df.loc[valid, "lat"],      df.loc[valid, "lon"],
    )
    df["bearing"] = df["bearing"].fillna(0)
    df["prev_bearing"] = df.groupby("entity_id")["bearing"].shift(1).fillna(0)
    df["bearing_delta"] = ((df["bearing"] - df["prev_bearing"] + 180) % 360) - 180

    # implied speed from position jump
    df["implied_speed_kn"] = np.where(
        df["time_gap_min"] > 0,
        (df["dist_km"] / 1.852) / (df["time_gap_min"] / 60.0),
        df["sog_knots"],
    )

    # per-vessel rolling baseline speed (causal: shift before rolling)
    df["speed_roll_mean"] = (
        df.groupby("entity_id")["sog_knots"]
          .transform(lambda s: s.shift(1).rolling(20, min_periods=3)
                                .mean().fillna(s.median()))
    )
    df["speed_roll_std"] = (
        df.groupby("entity_id")["sog_knots"]
          .transform(lambda s: s.shift(1).rolling(20, min_periods=3)
                                .std().fillna(1.0))
    )
    df["speed_vs_baseline"] = (
        (df["sog_knots"] - df["speed_roll_mean"]) /
        (df["speed_roll_std"] + 1e-6)
    )

    # distance from vessel's median location
    med = (df.groupby("entity_id")[["lat", "lon"]]
             .median()
             .rename(columns={"lat": "med_lat", "lon": "med_lon"}))
    df = df.join(med, on="entity_id")
    df["dist_from_median_km"] = haversine_km(
        df["lat"], df["lon"], df["med_lat"], df["med_lon"]
    )

    # temporal
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"]    = (df["hour_of_day"] < 6).astype(int)

    df.drop(columns=["prev_lat", "prev_lon", "prev_bearing",
                     "med_lat", "med_lon", "bearing"], inplace=True, errors="ignore")
    return df


# ══════════════════════════════════════════════════════════════
# TIER 1 — SPATIAL / AGENT PROFILING  (KNN outlier score)
# ══════════════════════════════════════════════════════════════
# Features used: location relative to vessel-type peers + speed context.
# KNN average distance to k nearest neighbours in feature space.
# High score = far from any peer cluster = spatially anomalous.

T1_FEATURES = [
    "lat", "lon",
    "sog_knots",
    "implied_speed_kn",
    "dist_from_median_km",
    "speed_vs_baseline",
    "dist_km",
    "bearing_delta",
]


def _knn_score(X: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Average distance to k nearest neighbours (excluding self)."""
    k = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree",
                          metric="euclidean", n_jobs=-1)
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    return dists[:, 1:].mean(axis=1)   # exclude self (distance=0)


def tier1_knn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute KNN outlier score per vessel_type group.
    Falls back to global model if the group is too small.
    Scores are then robust-minmax normalised to [0,1] globally.
    """
    print("  Tier 1: KNN spatial profiling...")
    df = df.copy()
    df["t1_raw"] = np.nan

    # global model for fallback
    X_all = df[T1_FEATURES].fillna(0).values
    sc_all = RobustScaler().fit(X_all)
    global_scores = _knn_score(sc_all.transform(X_all))

    for vtype, grp in df.groupby("vessel_type"):
        X = grp[T1_FEATURES].fillna(0).values
        if len(grp) < MIN_TYPE_ROWS:
            df.loc[grp.index, "t1_raw"] = global_scores[grp.index]
            continue
        sc = RobustScaler().fit(X)
        df.loc[grp.index, "t1_raw"] = _knn_score(sc.transform(X))

    df["t1_score"] = robust_minmax(df["t1_raw"])   # 0=normal, 1=anomalous
    return df


# ══════════════════════════════════════════════════════════════
# TIER 2 — TEMPORAL / SEQUENCE PROFILING  (Markov Chain NLL)
# ══════════════════════════════════════════════════════════════
# Grid-discretise lat/lon → cell token.
# Per vessel_type: build bigram transition count matrix.
# Score each ping by –log P(cell | prev_cell).
# Unseen transitions get Laplace-smoothed penalty.

def _cell_token(lat, lon, grid_deg=MARKOV_GRID_DEG):
    """Map lat/lon to a discrete grid cell string token."""
    r = int(np.floor(lat / grid_deg))
    c = int(np.floor(lon / grid_deg))
    return f"{r}_{c}"


def tier2_markov(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-vessel-type bigram Markov chains and score each ping.

    Steps:
      1. Assign each ping a cell token.
      2. For each vessel_type, count bigram transitions across all vessels.
      3. Convert counts to log-probabilities (Laplace smoothing α=1).
      4. Score: NLL = –log P(token_t | token_{t-1}).
         First ping of each vessel gets the median NLL (no prior).
      5. Normalise to [0,1].
    """
    print("  Tier 2: Markov Chain sequence profiling...")
    df = df.copy()
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    # cell tokens — vectorized (no apply)
    df["cell"] = (
        (df["lat"] / MARKOV_GRID_DEG).astype(int).astype(str)
        + "_" +
        (df["lon"] / MARKOV_GRID_DEG).astype(int).astype(str)
    )
    df["prev_cell"] = df.groupby("entity_id")["cell"].shift(1)

    df["t2_nll"] = np.nan

    for vtype, grp in df.groupby("vessel_type"):
        trans = grp.dropna(subset=["prev_cell"]).copy()

        # fully vectorized bigram counts — no Python loops
        pair_counts = (
            trans.groupby(["prev_cell", "cell"])
                 .size().reset_index(name="cnt")
        )
        from_totals = (
            pair_counts.groupby("prev_cell")["cnt"]
                       .sum().reset_index(name="from_total")
        )
        all_cells = set(grp["cell"].unique()) | set(trans["prev_cell"].unique())
        V = len(all_cells)

        pair_counts = pair_counts.merge(from_totals, on="prev_cell")
        pair_counts["nll"] = -np.log(
            (pair_counts["cnt"] + 1) / (pair_counts["from_total"] + V + 1e-9)
        )
        trans = trans.merge(
            pair_counts[["prev_cell", "cell", "nll"]],
            on=["prev_cell", "cell"], how="left"
        )
        unseen_nll = float(np.log(V + 1e-9))
        trans["nll"] = trans["nll"].fillna(unseen_nll)
        med_nll = float(trans["nll"].median())

        df.loc[trans.index, "t2_nll"] = trans["nll"].values
        first_idx = grp.index.difference(trans.index)
        df.loc[first_idx, "t2_nll"] = med_nll


    df["t2_score"] = robust_minmax(df["t2_nll"])   # 0=normal, 1=anomalous
    df.drop(columns=["cell", "prev_cell"], inplace=True, errors="ignore")
    return df


# ══════════════════════════════════════════════════════════════
# TIER 3 — DOMAIN RULES  (strict physical rules)
# ══════════════════════════════════════════════════════════════
def tier3_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hard rules — intentionally strict to keep precision high.
    Each rule is binary; combined into rule_fires (any=1).
    """
    print("  Tier 3: Domain rules...")
    df = df.copy()

    # Rule 1: Dark ship — AIS goes silent beyond threshold
    df["rule_dark_ship"] = (df["time_gap_min"] > AIS_GAP_THRESH_MIN).astype(int)

    # Rule 2: Spoofing — implied speed exceeds physical max for vessel type
    max_spd = df["vessel_type"].map(VTYPE_MAX_SPEED_KN).fillna(DEFAULT_MAX_KN)
    df["rule_spoofing"] = (df["implied_speed_kn"] > max_spd).astype(int)

    # Rule 3: Illegal fishing — fishing vessel slow & far from usual grounds
    df["rule_illegal_fish"] = (
        (df["vessel_type"] == "fishing") &
        (df["sog_knots"] < 2.0) &
        (df["dist_from_median_km"] > OOD_DIST_KM)
    ).astype(int)

    df["rule_fires"] = (
        (df["rule_dark_ship"] | df["rule_spoofing"] | df["rule_illegal_fish"])
    ).astype(int)

    return df


# ══════════════════════════════════════════════════════════════
# TIER 4 — WEAK SUPERVISION META-LEARNER
# ══════════════════════════════════════════════════════════════
META_FEATURES = [
    # raw model scores
    "t1_score",
    "t2_score",
    "t1_raw",
    "t2_nll",
    # rule votes as soft features
    "rule_dark_ship",
    "rule_spoofing",
    "rule_illegal_fish",
    # movement features
    "sog_knots",
    "speed_vs_baseline",
    "implied_speed_kn",
    "dist_km",
    "bearing_delta",
    "dist_from_median_km",
    "time_gap_min",
    # stay features
    "stay_duration_min",
    "is_stay_int",
    # temporal
    "hour_of_day",
    "is_night",
]


def generate_pseudo_labels(df: pd.DataFrame) -> pd.Series:
    """
    Assign pseudo-labels without touching is_anomalous:

      label = 1  if  rule_fires == 1
                 OR  (t1_score >= T1_HIGH  AND  t2_score >= T2_HIGH)

      label = 0  if  rule_fires == 0
                 AND  t1_score <= T1_LOW
                 AND  t2_score <= T2_LOW

      label = NaN  (unlabeled / uncertain) otherwise

    Thresholds are percentile-based on the score distribution — no labels used.
    """
    t1_high = df["t1_score"].quantile(PSEUDO_POS_PERCENTILE / 100)
    t2_high = df["t2_score"].quantile(PSEUDO_POS_PERCENTILE / 100)
    t1_low  = df["t1_score"].quantile(PSEUDO_NEG_PERCENTILE / 100)
    t2_low  = df["t2_score"].quantile(PSEUDO_NEG_PERCENTILE / 100)

    both_high = (df["t1_score"] >= t1_high) & (df["t2_score"] >= t2_high)
    both_low  = (df["t1_score"] <= t1_low)  & (df["t2_score"] <= t2_low)

    labels = pd.Series(np.nan, index=df.index)
    labels[df["rule_fires"] == 1]                  = 1   # rule fires → positive
    labels[both_high & (df["rule_fires"] == 0)]    = 1   # both models extreme → positive
    labels[both_low  & (df["rule_fires"] == 0)]    = 0   # both models calm → negative

    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    unk = labels.isna().sum()
    print(f"    Pseudo-labels → 1: {pos:,}  |  0: {neg:,}  |  unlabeled: {unk:,}")
    return labels


def tier4_meta_learner(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Generate pseudo-labels.
    2. Train HistGradientBoostingClassifier on labeled subset only.
       HGBC natively handles NaN features — no imputation needed.
    3. Predict anomaly probability for the entire dataset.
    4. Choose threshold from the score distribution (no label leakage).
    """
    print("  Tier 4a: Generating pseudo-labels...")
    pseudo = generate_pseudo_labels(df)

    labeled_mask = pseudo.notna()
    X_train = df.loc[labeled_mask, META_FEATURES].values
    y_train = pseudo[labeled_mask].values.astype(int)

    print(f"  Tier 4b: Training meta-learner on {labeled_mask.sum():,} pseudo-labeled rows...")
    clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=40,
        l2_regularization=0.1,
        class_weight="balanced",    # handles class imbalance
        random_state=42,
    )
    clf.fit(X_train, y_train)

    print("  Tier 4c: Predicting anomaly probability for full dataset...")
    X_all = df[META_FEATURES].values
    df["anomaly_prob"] = clf.predict_proba(X_all)[:, 1]

    # threshold: flag top 5th percentile by probability (no label leakage)
    threshold = df["anomaly_prob"].quantile(0.95)
    df["pred_anomalous"] = (df["anomaly_prob"] >= threshold).astype(bool)

    return df, clf, threshold


# ══════════════════════════════════════════════════════════════
# EVALUATION  (ground truth used only here)
# ══════════════════════════════════════════════════════════════
def evaluate(df: pd.DataFrame, threshold: float):
    y_true = df["is_anomalous"].astype(int)
    y_pred = df["pred_anomalous"].astype(int)

    print("\n" + "=" * 60)
    print("EVALUATION  (ground truth: is_anomalous)")
    print(f"Threshold on anomaly_prob: {threshold:.4f}")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"]))
    f1 = f1_score(y_true, y_pred)
    print(f"F1: {f1:.4f}")

    # Rule breakdown
    print("\nTier 3 Rule Precision (diagnostic):")
    for col, name in [("rule_dark_ship",   "dark_ship"),
                      ("rule_spoofing",    "spoofing"),
                      ("rule_illegal_fish","illegal_fish")]:
        flagged = df[col].astype(bool)
        prec = (flagged & (y_true==1)).sum() / max(flagged.sum(), 1)
        print(f"  {name:20s}: {flagged.sum():6,} flagged | precision={prec:.2f}")

    # Oracle best F1 (label-tuned — shows headroom, don't use in production)
    pr, rc, thr = precision_recall_curve(y_true, df["anomaly_prob"])
    f1s = 2*pr*rc / (pr+rc+1e-9)
    best = f1s.max()
    best_t = thr[f1s.argmax()] if len(thr) else threshold
    print(f"\nOracle best F1 @ threshold={best_t:.4f}: {best:.4f}  (label-tuned, eval only)")

    print("\nAnomaly prob by ground-truth anomaly_type:")
    print(
        df.groupby("anomaly_type")["anomaly_prob"]
          .describe()[["mean","50%","75%","max"]]
          .round(3)
    )
    return f1


# ══════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════
def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tier 1 → Tier 2 → Tier 3 → Tier 4 → Evaluate
    All ground-truth columns are ignored until the final evaluate() call.
    """
    print("── Stay detection...")
    df = run_stay_detection(df)

    print("── Base feature engineering...")
    df = build_base_features(df)

    print("── Tier 1: KNN spatial outlier scoring...")
    df = tier1_knn(df)

    print("── Tier 2: Markov Chain sequence scoring...")
    df = tier2_markov(df)

    print("── Tier 3: Domain rules...")
    df = tier3_rules(df)

    print("── Tier 4: Weak supervision meta-learner...")
    df, clf, threshold = tier4_meta_learner(df)

    evaluate(df, threshold)
    return df


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = pd.read_csv("ais_small.csv")
    results = run_pipeline(df)

    print("\nSample predicted anomalies:")
    cols = ["entity_id", "timestamp", "vessel_type", "sog_knots",
            "t1_score", "t2_score", "anomaly_prob",
            "is_anomalous", "anomaly_type"]
    print(results[results["pred_anomalous"]][cols].head(15).to_string(index=False))
