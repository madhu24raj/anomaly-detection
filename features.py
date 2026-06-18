"""
features.py — Per-track behavioral feature extraction from raw position streams.

Terminology is deliberately generic (maritime tracking, not simulation-specific):
  - "contact"     = a tracked object (vessel)
  - "ping"        = a single position report
  - "gap"         = silence between consecutive pings
  - "stratum"     = a restricted / sensitive geographic area
  - "rendez_zone" = a candidate at-sea meeting location
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# ── constants ───────────────────────────────────────────────────────────────
EARTH_R_KM     = 6371.0088
KM_PER_NM      = 1.852
RAD            = np.pi / 180.0

# Physical upper bound on vessel speed (kt).  Anything implied above this is
# either an error or deliberate deception.
HARD_SPEED_CAP_KT = 40.0

# Proximity thresholds
CLOSE_RANGE_NM  = 1.0    # "within visual range"
SHADOW_RANGE_NM = 2.0    # upper bound for shadowing behaviour
PORT_NEAR_NM    = 30.0   # "near port" cutoff for filtering out legitimate slow ops

# Known sensitive / restricted strata (lon, lat, radius_km).
# Names are deliberately absent from the feature names below.
STRATA: list[tuple[float, float, float]] = [
    (-80.50, 23.60, 45.0),
    (-82.50, 15.50, 60.0),
    (-66.50, 12.50, 70.0),
    (-78.20, 17.10, 40.0),
]

# Representative ports (lon, lat) — used for distance-from-port features.
PORTS_LONLAT: list[tuple[float, float]] = [
    (-80.19, 25.77), (-82.38, 23.13), (-77.34, 25.06),
    (-76.79, 17.99), (-69.93, 18.47), (-66.10, 18.47),
    (-68.93, 12.11), (-61.52, 10.65), (-59.62, 13.10),
    (-75.51, 10.40), (-79.90,  9.36), (-86.85, 21.16),
]


# ── haversine helpers ────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial bearing (°, 0=N, clockwise) from point 1 → point 2."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ── per-ping kinematics ──────────────────────────────────────────────────────

def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add speed_kt, heading_deg, heading_delta, dt_h columns to a single-contact
    dataframe sorted by timestamp.

    ``df`` must have columns: lat, lon, timestamp (datetime64).
    Returns a copy with the extra columns.
    """
    df = df.sort_values("timestamp").copy()
    n = len(df)

    dt_h = np.full(n, np.nan)
    dist_km = np.full(n, np.nan)
    hdg = np.full(n, np.nan)

    lat = df["lat"].to_numpy()
    lon = df["lon"].to_numpy()
    t   = df["timestamp"].values.astype("datetime64[s]").astype(np.int64).astype(np.float64) / 3600.0

    if n >= 2:
        dt_h[1:]   = np.diff(t)
        dist_km[1:]= haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        hdg[1:]    = bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        speed_kt = np.where(dt_h > 0, dist_km / dt_h / KM_PER_NM, np.nan)

    hdg_delta = np.full(n, np.nan)
    if n >= 3:
        raw_delta = np.diff(hdg[1:])          # diffs of valid headings
        hdg_delta[2:] = (raw_delta + 180) % 360 - 180   # signed, ±180

    df["dt_h"]         = dt_h
    df["dist_km"]      = dist_km
    df["speed_kt"]     = speed_kt
    df["heading_deg"]  = hdg
    df["heading_delta"]= hdg_delta
    return df


# ── distance-to-reference helpers ───────────────────────────────────────────

def min_dist_to_ports_nm(lat: np.ndarray, lon: np.ndarray) -> float:
    """Minimum distance (nm) from any ping to the nearest reference port."""
    ports = np.array(PORTS_LONLAT)   # (P, 2)  lon, lat
    p_lat = np.radians(ports[:, 1])
    p_lon = np.radians(ports[:, 0])
    q_lat = np.radians(lat)
    q_lon = np.radians(lon)
    # BallTree for vectorised nearest-neighbour query
    tree = BallTree(np.column_stack([p_lat, p_lon]), metric="haversine")
    dist_rad, _ = tree.query(np.column_stack([q_lat, q_lon]), k=1)
    dist_nm = dist_rad.flatten() * EARTH_R_KM / KM_PER_NM
    return float(dist_nm.min())


def strata_proximity_features(lat: np.ndarray, lon: np.ndarray,
                               dt_h: Optional[np.ndarray] = None
                               ) -> dict:
    """
    For each ping compute the distance to the nearest sensitive stratum.
    Returns:
      min_stratum_dist_nm   — closest approach to any stratum
      frac_inside_stratum   — fraction of pings inside any stratum radius
      dwell_inside_nm_h     — (distance × time) proxy for loitering inside strata
    """
    n = len(lat)
    min_d_nm = np.full(n, np.inf)

    for s_lon, s_lat, s_r_km in STRATA:
        d_km = haversine_km(lat, lon, s_lat, s_lon)
        d_nm = d_km / KM_PER_NM
        inside = d_km <= s_r_km
        min_d_nm = np.minimum(min_d_nm, d_nm)

    frac_inside = float(np.mean(min_d_nm * KM_PER_NM <= np.array(
        [s[2] for s in STRATA]).max()))  # rough

    # More precise: recompute per stratum
    inside_any = np.zeros(n, dtype=bool)
    for s_lon, s_lat, s_r_km in STRATA:
        inside_any |= haversine_km(lat, lon, s_lat, s_lon) <= s_r_km

    frac_inside = float(inside_any.mean())
    dwell = 0.0
    if dt_h is not None and len(dt_h) == n:
        hours_inside = np.nansum(np.where(inside_any, dt_h, 0.0))
        dwell = float(hours_inside)

    return {
        "min_stratum_dist_nm": float(min_d_nm.min()),
        "frac_inside_stratum": frac_inside,
        "hours_inside_stratum": dwell,
    }


# ── gap analysis ─────────────────────────────────────────────────────────────

def gap_features(dt_h: np.ndarray, speed_kt: np.ndarray,
                 dist_km: np.ndarray, nominal_interval_h: float) -> dict:
    """
    Extract features related to transmission gaps and implied motion during gaps.

    A "gap" is any inter-ping silence that is > 2× the nominal cadence.
    """
    valid_dt = dt_h[~np.isnan(dt_h)]
    if len(valid_dt) == 0:
        return {k: 0.0 for k in [
            "max_gap_h", "gap_count", "gap_frac", "max_implied_speed_kt",
            "max_gap_displacement_nm", "p95_gap_h"]}

    threshold_h = 2.5 * nominal_interval_h
    is_gap      = valid_dt > threshold_h
    n_gaps      = int(is_gap.sum())

    # Implied speed during each gap: dist / time (already in speed_kt for pings,
    # but we want to isolate the gap pings specifically)
    gap_speeds = speed_kt[~np.isnan(speed_kt)]
    gap_speeds_gaps = speed_kt[1:][~np.isnan(dt_h[1:])]  # align
    # Max implied speed across all pings — both gap and non-gap (for spoofing)
    max_speed = float(np.nanmax(speed_kt)) if np.any(~np.isnan(speed_kt)) else 0.0

    # Max displacement in a single gap (nm)
    gap_mask   = (~np.isnan(dt_h[1:])) & (dt_h[1:] > threshold_h)
    if gap_mask.any():
        gap_d_nm = dist_km[1:][gap_mask] / KM_PER_NM
        max_gap_disp = float(gap_d_nm.max())
    else:
        max_gap_disp = 0.0

    # How many pings show a suspicious implied speed (> 60 kt).
    # A single GPS jitter can spike the *max* once; genuine position manipulation
    # produces at least 2 such pings (the jump there + the jump back).
    SUSPICIOUS_KT    = 60.0
    valid_spd_arr    = speed_kt[~np.isnan(speed_kt)]
    n_suspicious     = int((valid_spd_arr > SUSPICIOUS_KT).sum())

    return {
        "max_gap_h":                float(valid_dt.max()),
        "p95_gap_h":                float(np.percentile(valid_dt, 95)),
        "gap_count":                n_gaps,
        "gap_frac":                 float(n_gaps / max(len(valid_dt), 1)),
        "max_implied_speed_kt":     max_speed,
        "max_gap_displacement_nm":  max_gap_disp,
        "n_suspicious_speed_pings": n_suspicious,
    }


# ── movement pattern features ─────────────────────────────────────────────

def motion_features(lat: np.ndarray, lon: np.ndarray,
                    speed_kt: np.ndarray, heading_delta: np.ndarray,
                    dt_h: np.ndarray) -> dict:
    """
    Characterise how a contact moves: slow vs fast, erratic vs steady,
    loitering vs transiting.
    """
    valid_spd = speed_kt[~np.isnan(speed_kt)]

    if len(valid_spd) == 0:
        return {k: 0.0 for k in [
            "mean_speed_kt", "median_speed_kt", "p95_speed_kt", "std_speed_kt",
            "frac_slow", "frac_stopped", "straightness", "hull_area_deg2",
            "heading_variance", "heading_change_rate"]}

    slow_thresh    = 2.0   # kt
    stopped_thresh = 0.5   # kt

    # Path straightness: net displacement / total path length
    net_km = haversine_km(lat[0], lon[0], lat[-1], lon[-1])
    path_km = float(np.nansum(np.where(~np.isnan(speed_kt[1:]),
                                        haversine_km(lat[:-1], lon[:-1],
                                                     lat[1:], lon[1:]), 0)))
    straightness = float(net_km / path_km) if path_km > 0 else 1.0

    # Convex hull area (degrees²) — small = loitering, large = transiting
    try:
        from scipy.spatial import ConvexHull
        pts = np.column_stack([lon, lat])
        if len(np.unique(pts, axis=0)) >= 3:
            hull_area = float(ConvexHull(pts).volume)  # 2D → area
        else:
            hull_area = 0.0
    except Exception:
        hull_area = 0.0

    # Heading erraticism
    valid_hdg_d = heading_delta[~np.isnan(heading_delta)]
    hdg_var   = float(np.var(valid_hdg_d)) if len(valid_hdg_d) else 0.0
    hdg_rate  = float(np.mean(np.abs(valid_hdg_d))) if len(valid_hdg_d) else 0.0

    # Time-weighted slow fraction
    total_h   = float(np.nansum(dt_h))
    if total_h > 0:
        slow_h    = float(np.nansum(np.where(speed_kt[1:] < slow_thresh,    dt_h[1:], 0)))
        stopped_h = float(np.nansum(np.where(speed_kt[1:] < stopped_thresh, dt_h[1:], 0)))
        frac_slow    = slow_h    / total_h
        frac_stopped = stopped_h / total_h
    else:
        frac_slow = frac_stopped = 0.0

    return {
        "mean_speed_kt":      float(np.nanmean(valid_spd)),
        "median_speed_kt":    float(np.nanmedian(valid_spd)),
        "p95_speed_kt":       float(np.nanpercentile(valid_spd, 95)),
        "std_speed_kt":       float(np.nanstd(valid_spd)),
        "frac_slow":          frac_slow,
        "frac_stopped":       frac_stopped,
        "straightness":       straightness,
        "hull_area_deg2":     hull_area,
        "heading_variance":   hdg_var,
        "heading_change_rate": hdg_rate,
    }


# ── pairwise proximity features (expensive — batched by timestamp) ──────────

def proximity_features_all(df: pd.DataFrame, timestamps: np.ndarray,
                            interval_h: float) -> pd.DataFrame:
    """
    For each contact, at each timestamp, find the nearest other contact.
    Aggregate into per-contact features:
      - min_nn_dist_nm         : closest ever approach to any other contact
      - mean_nn_dist_nm        : time-averaged nearest-neighbour distance
      - hours_within_1nm       : cumulative time within CLOSE_RANGE_NM of any contact
      - hours_within_2nm       : cumulative time within SHADOW_RANGE_NM
      - sustained_close_count  : number of distinct proximity episodes > 1 h
      - co_slow_hours          : hours where BOTH contacts are < 1 kt simultaneously

    Works in O(C²) per timestep; fine for C ~ 1000 contacts.
    """
    contacts = df["entity_id"].unique()
    n_c = len(contacts)
    cid_idx = {c: i for i, c in enumerate(contacts)}

    # Accumulator arrays per contact
    nn_min  = np.full(n_c, np.inf)
    nn_sum  = np.zeros(n_c)
    nn_cnt  = np.zeros(n_c, dtype=int)
    h_1nm   = np.zeros(n_c)
    h_2nm   = np.zeros(n_c)
    coslow  = np.zeros(n_c)
    # For sustained proximity episodes we track whether contact was "close" last step
    was_close  = np.zeros(n_c, dtype=bool)
    close_dur  = np.zeros(n_c)        # running episode length (hours)
    ep_count   = np.zeros(n_c, dtype=int)

    EPISODE_MIN_H = 1.0   # min contiguous close-range time to count as episode

    for ts in timestamps:
        sub = df[df["timestamp"] == ts]
        if len(sub) < 2:
            continue

        lats = sub["lat"].to_numpy()
        lons = sub["lon"].to_numpy()
        ids  = sub["entity_id"].to_numpy()
        spds = sub["speed_kt"].to_numpy() if "speed_kt" in sub.columns else None

        rad_latlon = np.column_stack([np.radians(lats), np.radians(lons)])
        tree = BallTree(rad_latlon, metric="haversine")
        # k=2: nearest excluding self
        dist_rad, nn_ix = tree.query(rad_latlon, k=min(2, len(sub)))
        dist_km  = dist_rad[:, -1] * EARTH_R_KM
        dist_nm  = dist_km / KM_PER_NM
        nn_ids   = ids[nn_ix[:, -1]]

        for local_i, cid in enumerate(ids):
            gi = cid_idx[cid]
            d  = dist_nm[local_i]
            nn_min[gi] = min(nn_min[gi], d)
            nn_sum[gi] += d
            nn_cnt[gi] += 1

            is_close_now = d < CLOSE_RANGE_NM
            is_2nm_now   = d < SHADOW_RANGE_NM

            if is_close_now:
                h_1nm[gi] += interval_h
            if is_2nm_now:
                h_2nm[gi] += interval_h

            # co-slow: both this contact and its nn slow
            if spds is not None and is_close_now:
                nn_local = int(nn_ix[local_i, -1])
                if spds[local_i] < 1.0 and spds[nn_local] < 1.0:
                    coslow[gi] += interval_h

            # sustained proximity tracking
            if is_close_now:
                close_dur[gi] += interval_h
            else:
                if was_close[gi] and close_dur[gi] >= EPISODE_MIN_H:
                    ep_count[gi] += 1
                close_dur[gi] = 0.0
            was_close[gi] = is_close_now

    # close any open episodes at end
    for gi in range(n_c):
        if was_close[gi] and close_dur[gi] >= EPISODE_MIN_H:
            ep_count[gi] += 1

    mean_nn = np.where(nn_cnt > 0, nn_sum / nn_cnt, np.inf)

    result = pd.DataFrame({
        "entity_id":              contacts,
        "min_nn_dist_nm":         np.where(np.isinf(nn_min), 9999, nn_min),
        "mean_nn_dist_nm":        np.where(np.isinf(mean_nn), 9999, mean_nn),
        "hours_within_1nm":       h_1nm,
        "hours_within_2nm":       h_2nm,
        "sustained_close_episodes": ep_count,
        "co_slow_hours":          coslow,
    })
    return result


# ── top-level: build feature matrix ─────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame,
                         nominal_interval_h: float = 0.25) -> pd.DataFrame:
    """
    Given the raw AIS dataframe (entity_id, lat, lon, timestamp), returns a
    per-contact feature dataframe.  Ground-truth columns (is_anomalous,
    anomaly_type) are excluded from features but carried along if present for
    later evaluation.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    label_cols = [c for c in ["is_anomalous", "anomaly_type"] if c in df.columns]

    all_feats = []

    for cid, grp in df.groupby("entity_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        grp = compute_kinematics(grp)

        lat = grp["lat"].to_numpy()
        lon = grp["lon"].to_numpy()
        spd = grp["speed_kt"].to_numpy()
        hdg_d = grp["heading_delta"].to_numpy()
        dt_h  = grp["dt_h"].to_numpy()
        dist  = grp["dist_km"].to_numpy()

        row: dict = {"entity_id": cid}

        row.update(motion_features(lat, lon, spd, hdg_d, dt_h))
        row.update(gap_features(dt_h, spd, dist, nominal_interval_h))
        row.update(strata_proximity_features(lat, lon, dt_h))
        row["min_port_dist_nm"] = min_dist_to_ports_nm(lat, lon)

        # Carry ground truth if present (NOT used in detection)
        if label_cols:
            if "is_anomalous" in grp.columns:
                row["true_anomalous"] = int(grp["is_anomalous"].max())
            if "anomaly_type" in grp.columns:
                types = grp["anomaly_type"].dropna().unique()
                types = [t for t in types if t not in ("none", "", "0")]
                row["true_type"] = types[0] if types else "none"

        all_feats.append(row)

    feat_df = pd.DataFrame(all_feats).set_index("entity_id")

    # Pairwise proximity features — merged in separately
    timestamps = df["timestamp"].unique()
    # Merge speed back onto main df so proximity step can use it
    kin_df = []
    for cid, grp in df.groupby("entity_id"):
        g = compute_kinematics(grp.sort_values("timestamp").reset_index(drop=True))
        kin_df.append(g)
    df_kin = pd.concat(kin_df, ignore_index=True)

    print(f"  Computing pairwise proximity across {len(timestamps)} timestamps "
          f"× {df['entity_id'].nunique()} contacts...")
    prox_df = proximity_features_all(df_kin, timestamps, nominal_interval_h)
    prox_df = prox_df.set_index("entity_id")

    feat_df = feat_df.join(prox_df, how="left")
    return feat_df