"""
AIS Vessel Stay Detection & Anomaly Detection - v3 (Self-Supervised)
"""
def _corrupt(X: np.ndarray) -> np.ndarray:
    """
    Targeted corruption that mimics real AIS anomaly types instead of
    pure column-shuffle (which is too easy and doesn't align with anomaly space).
    """
    rng  = np.random.default_rng(42)
    Xf   = X.copy().astype(float)
    n    = len(Xf)

    # Column indices (must match FEATURE_COLS order)
    IDX = {name: i for i, name in enumerate(FEATURE_COLS)}

    # dark-ship: inject large time gaps
    mask = rng.random(n) < 0.33
    Xf[mask, IDX["time_gap_min"]] = rng.uniform(120, 600, mask.sum())

    # spoofing: inject implausible position jumps
    mask = rng.random(n) < 0.33
    Xf[mask, IDX["dist_from_prev_km"]]  = rng.uniform(60, 300, mask.sum())
    Xf[mask, IDX["implied_speed_kn"]]   = rng.uniform(80, 200, mask.sum())

    # speed anomaly: spike speed vs baseline
    mask = rng.random(n) < 0.33
    Xf[mask, IDX["speed_vs_baseline"]]  = rng.uniform(5, 15, mask.sum())
    Xf[mask, IDX["speed_delta"]]        = rng.uniform(15, 40, mask.sum())

    return Xf


def train_and_predict_self_supervised(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    X    = df[FEATURE_COLS].fillna(0).values.astype(float)
    Xf   = _corrupt(X)

    X_train = np.vstack([X, Xf])
    y_train = np.hstack([np.zeros(len(X)), np.ones(len(Xf))])

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    df["final_score"] = clf.predict_proba(X)[:, 1]
    return df


def select_dynamic_threshold(scores: pd.Series) -> float:
    """
    Use the 98th percentile of scores as the threshold — no hard floor.
    This lets the threshold float down to where the actual score distribution
    puts anomalies instead of clipping it above them.
    """
    return float(np.percentile(scores, 98.0))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from sklearn.preprocessing import OrdinalEncoder
import warnings
warnings.filterwarnings("ignore")

# CONSTANTS & CONFIG
# ──────────────────────────────────────────────────────────────
STAY_RADIUS_KM            = 0.5
STAY_MIN_DURATION_M       = 30
SPEED_STAY_THRESH         = 1.0

# Tightened rendezvous to avoid massive port false positives (~1.1km grid)
RENDEZVOUS_GRID_DEG       = 0.01 
RENDEZVOUS_TIME_WINDOW_M  = 10   

VTYPE_MAX_SPEED = {
    "fishing":   20,
    "cargo":     28,
    "passenger": 35,
    "pleasure":  45,
    "tanker":    20,
}
DEFAULT_MAX_SPEED = 40

# ──────────────────────────────────────────────────────────────
# MATH HELPERS
# ──────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def bearing_deg(lat1, lon1, lat2, lon2):
    dlon = np.radians(lon2 - lon1)
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r)*np.sin(lat2r) - np.sin(lat1r)*np.cos(lat2r)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

# ──────────────────────────────────────────────────────────────
# 1 & 2. STAY & RENDEZVOUS DETECTION
# ──────────────────────────────────────────────────────────────
def detect_stays_sliding_window(group: pd.DataFrame) -> pd.DataFrame:
    df = group.sort_values("timestamp").copy()
    n  = len(df)
    df["is_stay"]   = False
    df["stay_id"]   = -1

    lats  = df["lat"].values
    lons  = df["lon"].values
    times = pd.to_datetime(df["timestamp"]).values.astype("int64") / 1e9
    speeds = df["sog_knots"].values

    stay_id = 0
    i = 0
    while i < n:
        j = i + 1
        last_good_j = i
        while j < n:
            clat  = lats[i:j+1].mean()
            clon  = lons[i:j+1].mean()
            dists = haversine_km(lats[i:j+1], lons[i:j+1], clat, clon)
            if dists.max() > STAY_RADIUS_KM: break
            dur = (times[j] - times[i]) / 60.0
            if dur >= STAY_MIN_DURATION_M: last_good_j = j
            j += 1

        if last_good_j > i:
            df.iloc[i:last_good_j+1, df.columns.get_loc("is_stay")]  = True
            df.iloc[i:last_good_j+1, df.columns.get_loc("stay_id")] = stay_id
            stay_id += 1
            i = last_good_j + 1
        else:
            i += 1

    df["stay_center_lat"]   = np.nan
    df["stay_center_lon"]   = np.nan
    df["stay_duration_min"] = np.nan

    for sid, grp in df[df["stay_id"] >= 0].groupby("stay_id"):
        ts  = pd.to_datetime(grp["timestamp"])
        dur = (ts.max() - ts.min()).total_seconds() / 60
        df.loc[grp.index, "stay_center_lat"]   = grp["lat"].mean()
        df.loc[grp.index, "stay_center_lon"]   = grp["lon"].mean()
        df.loc[grp.index, "stay_duration_min"] = dur

    df["low_speed_flag"]   = speeds < SPEED_STAY_THRESH
    df["n_stays_so_far"]   = df["stay_id"].clip(lower=0).replace(-1, np.nan).ffill().fillna(0)
    return df

def run_stay_detection(df: pd.DataFrame) -> pd.DataFrame:
    results = [detect_stays_sliding_window(grp).assign(entity_id=eid) for eid, grp in df.groupby("entity_id")]
    return pd.concat(results, ignore_index=True)

def detect_rendezvous(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["_slot"] = (ts.astype("int64") // (RENDEZVOUS_TIME_WINDOW_M * 60 * 1_000_000_000)).astype(int)
    df["_glat"] = (df["lat"] / RENDEZVOUS_GRID_DEG).astype(int)
    df["_glon"] = (df["lon"] / RENDEZVOUS_GRID_DEG).astype(int)

    cell_counts = df.groupby(["_slot", "_glat", "_glon"])["entity_id"].transform("nunique")
    # Cap at 3 to ignore crowded anchorages/ports
    df["rendezvous_flag"] = ((cell_counts >= 2) & (cell_counts <= 3)).astype(int)
    df.drop(columns=["_slot", "_glat", "_glon"], inplace=True)
    return df

# ──────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    df["time_gap_min"] = df.groupby("entity_id")["timestamp"].diff().dt.total_seconds().div(60).fillna(0)
    df["speed_delta"] = df.groupby("entity_id")["sog_knots"].diff().fillna(0)

    # Rolling baselines
    df["speed_roll_mean"] = df.groupby("entity_id")["sog_knots"].transform(lambda s: s.shift(1).rolling(20, min_periods=3).mean().fillna(s.median()))
    df["speed_roll_std"] = df.groupby("entity_id")["sog_knots"].transform(lambda s: s.shift(1).rolling(20, min_periods=3).std().fillna(1))
    df["speed_vs_baseline"] = (df["sog_knots"] - df["speed_roll_mean"]) / (df["speed_roll_std"] + 1e-6)

    # Spatial
    df["prev_lat"] = df.groupby("entity_id")["lat"].shift(1)
    df["prev_lon"] = df.groupby("entity_id")["lon"].shift(1)
    valid = df["prev_lat"].notna()

    df.loc[valid, "dist_from_prev_km"] = haversine_km(
        df.loc[valid, "lat"], df.loc[valid, "lon"],
        df.loc[valid, "prev_lat"], df.loc[valid, "prev_lon"]
    )
    df["dist_from_prev_km"] = df["dist_from_prev_km"].fillna(0)

    df.loc[valid, "bearing"] = bearing_deg(
        df.loc[valid, "prev_lat"], df.loc[valid, "prev_lon"],
        df.loc[valid, "lat"], df.loc[valid, "lon"]
    )
    df["bearing"] = df["bearing"].fillna(0)
    df["prev_bearing"] = df.groupby("entity_id")["bearing"].shift(1).fillna(0)
    df["bearing_delta"] = ((df["bearing"] - df["prev_bearing"] + 180) % 360) - 180

    # Prevent div-by-zero infinite speeds
    safe_time_gap = np.maximum(df["time_gap_min"], 1.0)
    df["implied_speed_kn"] = np.where(
        df["time_gap_min"] > 0,
        (df["dist_from_prev_km"] / 1.852) / (safe_time_gap / 60.0),
        df["sog_knots"],
    )

    med = df.groupby("entity_id")[["lat","lon"]].median().rename(columns={"lat":"med_lat","lon":"med_lon"})
    df = df.join(med, on="entity_id")
    df["dist_from_median_km"] = haversine_km(df["lat"], df["lon"], df["med_lat"], df["med_lon"])

    df["stay_duration_min"] = df["stay_duration_min"].fillna(0)
    df["is_stay_int"] = df["is_stay"].astype(int)
    
    # Encode categorical vessel type for the tree
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df["vessel_type_encoded"] = oe.fit_transform(df[["vessel_type"]])

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["is_night"]    = (df["hour_of_day"] < 6).astype(int)

    return df

# ──────────────────────────────────────────────────────────────
# 4. SELF-SUPERVISED ANOMALY MODEL (The Silver Bullet)
# ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "sog_knots", "speed_delta", "speed_vs_baseline", "implied_speed_kn",
    "dist_from_prev_km", "bearing_delta", "dist_from_median_km",
    "stay_duration_min", "is_stay_int", "time_gap_min",
    "rendezvous_flag", "is_night", "vessel_type_encoded"
]

# def train_and_predict_self_supervised(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Trains a HistGradientBoosting tree to distinguish between real AIS rows
#     and synthetically corrupted rows. Real rows flagged as corrupted are anomalies.
#     """
#     df = df.copy()
#     X_real = df[FEATURE_COLS].values
    
#     # 1. Create synthetic "shadow" data by independently shuffling every column
#     # This destroys normal physical relationships (e.g. fast speed + low distance)
#     X_fake = np.zeros_like(X_real)
#     for i in range(X_real.shape[1]):
#         X_fake[:, i] = np.random.permutation(X_real[:, i])
        
#     # 2. Combine and create dummy labels (0 = Real, 1 = Fake)
#     X_train = np.vstack([X_real, X_fake])
#     y_train = np.hstack([np.zeros(len(X_real)), np.ones(len(X_fake))])
    
#     # 3. Train Classifier
#     # HistGradient handles NaNs beautifully and is incredibly fast
#     clf = HistGradientBoostingClassifier(
#         max_iter=150,
#         learning_rate=0.1,
#         max_depth=7,
#         class_weight="balanced",
#         random_state=42
#     )
#     clf.fit(X_train, y_train)
    
#     # 4. Predict on REAL data. 
#     # High probability of being fake = violates physics = Anomaly!
#     df["final_score"] = clf.predict_proba(X_real)[:, 1]
    
#     return df

# def select_dynamic_threshold(scores: pd.Series) -> float:
#     """
#     Since anomalies are rare, they will sit in the extreme right tail of probabilities.
#     We target a high percentile, but bound it to ensure physical plausibility.
#     """
#     # 99th percentile equates to roughly ~1% contamination assumption
#     return float(np.clip(np.percentile(scores, 99.0), 0.75, 0.98))

def evaluate(df: pd.DataFrame, threshold: float):
    y_true = df["is_anomalous"].astype(int)
    y_pred = (df["final_score"] >= threshold).astype(int)

    print("=" * 55)
    print("ANOMALY DETECTION — Classification Report")
    print(f"Threshold: {threshold:.4f}")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"]))
    
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, df["final_score"])
    f1_arr = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
    best_f1 = f1_arr.max()
    best_thr = thr_arr[f1_arr.argmax()] if len(thr_arr) > 0 else threshold
    
    print(f"Current Pipeline F1: {f1_score(y_true, y_pred):.4f}")
    print(f"\nOracle Best F1 (Theoretical Limit): {best_f1:.4f} at threshold {best_thr:.4f}")

# ──────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ──────────────────────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("Step 1/2: Stay & Rendezvous detection...")
    df = run_stay_detection(df)
    df = detect_rendezvous(df)

    print("Step 3: Feature engineering...")
    df = build_features(df)

    print("Step 4: Training Self-Supervised Tree...")
    df = train_and_predict_self_supervised(df)

    print("Step 5: Dynamic Threshold Selection...")
    threshold = select_dynamic_threshold(df["final_score"])
    df["pred_anomalous"] = df["final_score"] >= threshold

    print("Step 6: Evaluation...\n")
    evaluate(df, threshold)
    return df

if __name__ == "__main__":
    df = pd.read_csv("ais_small.csv")
    results = run_pipeline(df)
