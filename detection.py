"""
AIS Vessel Stay Detection & Anomaly Detection
Data schema: entity_id, lat, lon, timestamp, vessel_type, sog_knots, is_anomalous, anomaly_type

Usage:
    df = pd.read_csv('ais_small.csv')
    results = run_pipeline(df)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from scipy.spatial.distance import cdist
from datetime import timedelta

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
STAY_RADIUS_KM       = 0.5    # spatial radius to consider a vessel "staying"
STAY_MIN_DURATION_M  = 30     # minimum stay duration in minutes
SPEED_STAY_THRESH    = 1.0    # knots: below this → likely stationary


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ─────────────────────────────────────────────
# 1. STAY DETECTION
# ─────────────────────────────────────────────
def detect_stays_sliding_window(group: pd.DataFrame) -> pd.DataFrame:
    """
    HAYSTAC-style stay detection using a sliding window approach.

    Algorithm:
      - Sort pings by time for one vessel.
      - Slide a window; if all pings within the window fit inside
        STAY_RADIUS_KM and span >= STAY_MIN_DURATION_M → mark as stay.
      - Merge overlapping stay windows.
      - Also flags low-speed pings independently as a quick signal.

    Returns the group df with added columns:
        is_stay (bool), stay_id (int, -1 = not a stay),
        stay_center_lat, stay_center_lon, stay_duration_min
    """
    df = group.sort_values("timestamp").copy()
    n = len(df)
    df["is_stay"] = False
    df["stay_id"] = -1

    lats   = df["lat"].values
    lons   = df["lon"].values
    times  = pd.to_datetime(df["timestamp"]).values.astype("int64") // 1e9  # unix seconds
    speeds = df["sog_knots"].values

    stay_id = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            # centroid of window [i..j]
            clat = lats[i:j+1].mean()
            clon = lons[i:j+1].mean()
            dists = haversine_km(lats[i:j+1], lons[i:j+1], clat, clon)

            if dists.max() > STAY_RADIUS_KM:
                break  # window too spread out

            duration_min = (times[j] - times[i]) / 60.0
            if duration_min >= STAY_MIN_DURATION_M:
                # Mark everything in [i..j] as a stay
                df.iloc[i:j+1, df.columns.get_loc("is_stay")]  = True
                df.iloc[i:j+1, df.columns.get_loc("stay_id")] = stay_id
                j += 1
            else:
                j += 1

        # If we assigned a stay, jump past it; otherwise advance
        if df.iloc[i]["stay_id"] >= 0:
            last_idx = df[df["stay_id"] == stay_id].index[-1]
            pos = df.index.get_loc(last_idx)
            stay_id += 1
            i = pos + 1
        else:
            i += 1

    # Add centroid + duration per stay
    df["stay_center_lat"]  = np.nan
    df["stay_center_lon"]  = np.nan
    df["stay_duration_min"] = np.nan

    for sid, grp in df[df["stay_id"] >= 0].groupby("stay_id"):
        t_sorted = pd.to_datetime(grp["timestamp"])
        dur = (t_sorted.max() - t_sorted.min()).total_seconds() / 60
        clat = grp["lat"].mean()
        clon = grp["lon"].mean()
        df.loc[grp.index, "stay_center_lat"]   = clat
        df.loc[grp.index, "stay_center_lon"]   = clon
        df.loc[grp.index, "stay_duration_min"] = dur

    # Speed-based quick signal (secondary)
    df["low_speed_flag"] = speeds < SPEED_STAY_THRESH

    return df


def run_stay_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Apply stay detection per vessel."""
    result = (
        df.groupby("entity_id", group_keys=False)
          .apply(detect_stays_sliding_window)
    )
    # pandas 2.x may drop the groupby key from the result; restore it
    if "entity_id" not in result.columns:
        result = result.reset_index(level="entity_id")
    return result.reset_index(drop=True)


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-ping features for anomaly detection.

    Speed-based:
        sog_knots, speed_delta, speed_z (z-score within vessel)

    Spatial / movement:
        dist_from_prev_km, bearing_delta (heading change)
        dist_from_median_pos (how far from vessel's typical area)

    Stay-based (uses stay detection output):
        is_stay, stay_duration_min (0 if not a stay)

    Temporal:
        hour_of_day, day_of_week
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"])

    # ── speed features
    df["speed_delta"] = df.groupby("entity_id")["sog_knots"].diff().fillna(0)

    def z_score(s):
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0
    df["speed_z"] = df.groupby("entity_id")["sog_knots"].transform(z_score)

    # ── spatial features
    df["prev_lat"] = df.groupby("entity_id")["lat"].shift(1)
    df["prev_lon"] = df.groupby("entity_id")["lon"].shift(1)
    valid = df["prev_lat"].notna()
    df.loc[valid, "dist_from_prev_km"] = haversine_km(
        df.loc[valid, "lat"], df.loc[valid, "lon"],
        df.loc[valid, "prev_lat"], df.loc[valid, "prev_lon"]
    )
    df["dist_from_prev_km"] = df["dist_from_prev_km"].fillna(0)

    # Bearing (heading) delta
    def bearing(lat1, lon1, lat2, lon2):
        dlon = np.radians(lon2 - lon1)
        lat1r, lat2r = np.radians(lat1), np.radians(lat2)
        x = np.sin(dlon) * np.cos(lat2r)
        y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360

    df.loc[valid, "bearing"] = bearing(
        df.loc[valid, "prev_lat"], df.loc[valid, "prev_lon"],
        df.loc[valid, "lat"],      df.loc[valid, "lon"]
    )
    df["bearing"] = df["bearing"].fillna(0)
    df["prev_bearing"] = df.groupby("entity_id")["bearing"].shift(1).fillna(0)
    df["bearing_delta"] = ((df["bearing"] - df["prev_bearing"] + 180) % 360) - 180

    # Distance from vessel's median position (OOD location signal)
    med = df.groupby("entity_id")[["lat", "lon"]].median().rename(
        columns={"lat": "med_lat", "lon": "med_lon"}
    )
    df = df.join(med, on="entity_id")
    df["dist_from_median_km"] = haversine_km(
        df["lat"], df["lon"], df["med_lat"], df["med_lon"]
    )

    # ── stay features (from stay detection)
    df["stay_duration_min"] = df["stay_duration_min"].fillna(0)
    df["is_stay_int"] = df["is_stay"].astype(int)

    # ── temporal features
    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek

    df.drop(columns=["prev_lat", "prev_lon", "prev_bearing",
                     "med_lat", "med_lon"], inplace=True)
    return df


# ─────────────────────────────────────────────
# 3. ANOMALY DETECTION
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "sog_knots",
    "speed_delta",
    "speed_z",
    "dist_from_prev_km",
    "bearing_delta",
    "dist_from_median_km",
    "stay_duration_min",
    "is_stay_int",
    "hour_of_day",
]


def train_isolation_forest(df: pd.DataFrame, contamination: float = 0.05):
    """
    Unsupervised anomaly detection with Isolation Forest.
    contamination ≈ expected anomaly rate (matches 5% in small dataset).

    Returns: trained model, scaler
    """
    X = df[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    return model, scaler


def predict_anomalies(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    Predict anomalies; score < 0 = anomalous in IF convention.
    Returns df with added:
        anomaly_score  (raw IF score, lower = more anomalous)
        pred_anomalous (bool)
    """
    X = df[FEATURE_COLS].fillna(0).values
    X_scaled = scaler.transform(X)

    df = df.copy()
    df["anomaly_score"]    = model.score_samples(X_scaled)
    df["pred_anomalous"]   = model.predict(X_scaled) == -1   # -1 → anomaly
    return df


# ─────────────────────────────────────────────
# 4. RULE-BASED LAYER (transparent, on top of IF)
# ─────────────────────────────────────────────
def rule_based_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight rules that map to known anomaly types in your data:

      dark_ship    → extended stay + zero speed  (vessel goes dark)
      spoofing     → large instantaneous jump in position
      aggression   → sudden high bearing + speed change
      illegal_fish → low-speed loitering in unusual location
    """
    df = df.copy()
    df["rule_dark_ship"]    = (df["is_stay_int"] == 1) & (df["sog_knots"] < 0.2) & \
                               (df["stay_duration_min"] > 120)
    df["rule_spoofing"]     = df["dist_from_prev_km"] > 50   # >50 km jump per interval
    df["rule_aggression"]   = (df["speed_delta"] > 10) & (abs(df["bearing_delta"]) > 90)
    df["rule_illegal_fish"] = (df["sog_knots"] < 2) & \
                               (df["dist_from_median_km"] > 20) & \
                               (df["vessel_type"] == "fishing")

    df["any_rule_flag"] = (
        df["rule_dark_ship"] | df["rule_spoofing"] |
        df["rule_aggression"] | df["rule_illegal_fish"]
    )
    return df


# ─────────────────────────────────────────────
# 5. EVALUATION (uses ground truth cols)
# ─────────────────────────────────────────────
def evaluate(df: pd.DataFrame):
    """
    Compare predictions to ground truth (is_anomalous).
    Prints classification report. Returns F1.
    """
    y_true = df["is_anomalous"].astype(int)
    y_pred = df["pred_anomalous"].astype(int)

    print("=" * 50)
    print("ANOMALY DETECTION — Classification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalous"]))

    f1 = f1_score(y_true, y_pred)
    print(f"F1 Score: {f1:.4f}")

    # Rule-based comparison
    print("\nRule-Based Layer Flags:")
    for col in ["rule_dark_ship", "rule_spoofing", "rule_aggression", "rule_illegal_fish"]:
        if col in df.columns:
            prec = (df[col] & (y_true == 1)).sum() / df[col].sum() \
                   if df[col].sum() > 0 else 0
            print(f"  {col:25s}: {df[col].sum():5d} flagged | precision={prec:.2f}")

    return f1


# ─────────────────────────────────────────────
# 6. FULL PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    End-to-end:
      1. Stay detection
      2. Feature engineering
      3. Isolation Forest anomaly detection
      4. Rule-based overlay
      5. Evaluation vs ground truth

    Args:
        df: raw AIS dataframe
        contamination: expected anomaly rate (match to dataset, e.g. 0.05 for small, 0.01 for large)
    """
    print("Step 1: Stay detection...")
    df = run_stay_detection(df)

    print("Step 2: Feature engineering...")
    df = build_features(df)

    print("Step 3: Training Isolation Forest...")
    model, scaler = train_isolation_forest(df, contamination=contamination)

    print("Step 4: Predicting anomalies...")
    df = predict_anomalies(df, model, scaler)

    print("Step 5: Rule-based flags...")
    df = rule_based_flags(df)

    print("Step 6: Evaluation...\n")
    evaluate(df)

    return df


# ─────────────────────────────────────────────
# QUICKSTART
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("ais_small.csv")
    results = run_pipeline(df, contamination=0.05)

    # Show sample anomalies
    print("\nSample predicted anomalies:")
    print(
        results[results["pred_anomalous"]]
        [["entity_id", "timestamp", "vessel_type", "sog_knots",
          "anomaly_score", "is_anomalous", "anomaly_type"]]
        .head(10)
        .to_string(index=False)
    )
