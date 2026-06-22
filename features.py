# # feature extraction - per contact (vessel), not per ping
# # builds a row of behavioral features from the full position history

# import warnings
# from typing import Optional

# import numpy as np
# import pandas as pd
# from sklearn.neighbors import BallTree

# EARTH_R_KM = 6371.0088
# KM_PER_NM = 1.852

# CLOSE_RANGE_NM = 1.0
# SHADOW_RANGE_NM = 2.0

# def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
#     lat1, lat2 = np.radians(lat1), np.radians(lat2)
#     dlat = lat2 - lat1
#     dlon = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
#     a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
#     return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
#     lat1, lat2 = np.radians(lat1), np.radians(lat2)
#     dlon = np.radians(lon2 - lon1)
#     x = np.sin(dlon) * np.cos(lat2)
#     y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
#     return (np.degrees(np.arctan2(x, y)) + 360) % 360



# def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.sort_values("timestamp").copy()
#     n = len(df)

#     dt_h = np.full(n, np.nan)
#     dist_km = np.full(n, np.nan)
#     hdg = np.full(n, np.nan)

#     lat = df["lat"].to_numpy()
#     lon = df["lon"].to_numpy()
#     t   = df["timestamp"].values.astype("datetime64[s]").astype(np.int64).astype(np.float64) / 3600.0

#     if n >= 2:
#         dt_h[1:]   = np.diff(t)
#         dist_km[1:]= haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
#         hdg[1:]    = bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         speed_kt = np.where(dt_h > 0, dist_km / dt_h / KM_PER_NM, np.nan)

#     hdg_delta = np.full(n, np.nan)
#     if n >= 3:
#         raw_delta = np.diff(hdg[1:])          # diffs of valid headings
#         hdg_delta[2:] = (raw_delta + 180) % 360 - 180   # signed, ±180

#     df["dt_h"]         = dt_h
#     df["dist_km"]      = dist_km
#     df["speed_kt"]     = speed_kt
#     df["heading_deg"]  = hdg
#     df["heading_delta"]= hdg_delta
#     return df



# #major ports (lon, lat)
# PORTS_LONLAT = [
#     (-80.19, 25.77), (-82.38, 23.13), (-77.34, 25.06),
#     (-76.79, 17.99), (-69.93, 18.47), (-66.10, 18.47),
#     (-68.93, 12.11), (-61.52, 10.65), (-59.62, 13.10),
#     (-75.51, 10.40), (-79.90,  9.36), (-86.85, 21.16),]

# _PORT_TREE = None

# def _port_tree():
#     global _PORT_TREE
#     if _PORT_TREE is None:
#         ports = np.array(PORTS_LONLAT)
#         p_lat = np.radians(ports[:, 1])
#         p_lon = np.radians(ports[:, 0])
#         _PORT_TREE = BallTree(np.column_stack([p_lat, p_lon]), metric="haversine")
#     return _PORT_TREE


# def min_dist_to_ports_nm(lat: np.ndarray, lon: np.ndarray) -> float:
#     q_lat = np.radians(lat)
#     q_lon = np.radians(lon)
#     dist_rad, _ = _port_tree().query(np.column_stack([q_lat, q_lon]), k=1)
#     dist_nm = dist_rad.flatten() * EARTH_R_KM / KM_PER_NM
#     return float(dist_nm.min())

# # expand list of strata as needed
# # defined as lat, lon, radius_km
# STRATA = [
#     (-80.50, 23.60, 45.0),
#     (-82.50, 15.50, 60.0),
#     (-66.50, 12.50, 70.0),
#     (-78.20, 17.10, 40.0),
# ]

# def strata_proximity_features(lat: np.ndarray, lon: np.ndarray, dt_h: Optional[np.ndarray] = None ) -> dict:
#     n = len(lat)
#     min_d_nm = np.full(n, np.inf)

#     for s_lon, s_lat, s_r_km in STRATA:
#         d_km = haversine_km(lat, lon, s_lat, s_lon)
#         d_nm = d_km / KM_PER_NM
#         inside = d_km <= s_r_km
#         min_d_nm = np.minimum(min_d_nm, d_nm)

#     frac_inside = float(np.mean(min_d_nm * KM_PER_NM <= np.array(
#         [s[2] for s in STRATA]).max()))  # rough

#     inside_any = np.zeros(n, dtype=bool)
#     for s_lon, s_lat, s_r_km in STRATA:
#         inside_any |= haversine_km(lat, lon, s_lat, s_lon) <= s_r_km

#     frac_inside = float(inside_any.mean())
#     dwell = 0.0
#     if dt_h is not None and len(dt_h) == n:
#         hours_inside = np.nansum(np.where(inside_any, dt_h, 0.0))
#         dwell = float(hours_inside)

#     return {
#         "min_stratum_dist_nm": float(min_d_nm.min()),
#         "frac_inside_stratum": frac_inside,
#         "hours_inside_stratum": dwell,
#     }


# def gap_features(dt_h: np.ndarray, speed_kt: np.ndarray, dist_km: np.ndarray, nominal_interval_h: float) -> dict:
#     valid_dt = dt_h[~np.isnan(dt_h)]
#     if len(valid_dt) == 0:
#         return {k: 0.0 for k in [
#             "max_gap_h", "gap_count", "gap_frac", "max_implied_speed_kt",
#             "max_gap_displacement_nm", "p95_gap_h"]}

#     threshold_h = 2.5 * nominal_interval_h
#     is_gap      = valid_dt > threshold_h
#     n_gaps      = int(is_gap.sum())

#     gap_speeds = speed_kt[~np.isnan(speed_kt)]
#     gap_speeds_gaps = speed_kt[1:][~np.isnan(dt_h[1:])]  # align
#     # for spoofing
#     max_speed = float(np.nanmax(speed_kt)) if np.any(~np.isnan(speed_kt)) else 0.0

#     gap_mask   = (~np.isnan(dt_h[1:])) & (dt_h[1:] > threshold_h)
#     if gap_mask.any():
#         gap_d_nm = dist_km[1:][gap_mask] / KM_PER_NM
#         max_gap_disp = float(gap_d_nm.max())
#     else:
#         max_gap_disp = 0.0

#     SUSPICIOUS_KT = 60.0
#     valid_spd_arr = speed_kt[~np.isnan(speed_kt)]
#     n_suspicious= int((valid_spd_arr > SUSPICIOUS_KT).sum())

#     return {
#         "max_gap_h":                float(valid_dt.max()),
#         "p95_gap_h":                float(np.percentile(valid_dt, 95)),
#         "gap_count":                n_gaps,
#         "gap_frac":                 float(n_gaps / max(len(valid_dt), 1)),
#         "max_implied_speed_kt":     max_speed,
#         "max_gap_displacement_nm":  max_gap_disp,
#         "n_suspicious_speed_pings": n_suspicious,
#     }


# def motion_features(lat: np.ndarray, lon: np.ndarray, speed_kt: np.ndarray, heading_delta: np.ndarray, dt_h: np.ndarray) -> dict:
#     valid_spd = speed_kt[~np.isnan(speed_kt)]

#     if len(valid_spd) == 0:
#         return {k: 0.0 for k in [
#             "mean_speed_kt", "median_speed_kt", "p95_speed_kt", "std_speed_kt",
#             "frac_slow", "frac_stopped", "straightness", "hull_area_deg2",
#             "heading_variance", "heading_change_rate"]}

#     slow_thresh = 2.0   # kt
#     stopped_thresh = 0.5   # kt

#     # path straightness: net displacement / total path length
#     net_km = haversine_km(lat[0], lon[0], lat[-1], lon[-1])
#     path_km = float(np.nansum(np.where(~np.isnan(speed_kt[1:]),
#                                         haversine_km(lat[:-1], lon[:-1],
#                                                      lat[1:], lon[1:]), 0)))
#     straightness = float(net_km / path_km) if path_km > 0 else 1.0

#     try:
#         from scipy.spatial import ConvexHull
#         pts = np.column_stack([lon, lat])
#         if len(np.unique(pts, axis=0)) >= 3:
#             hull_area = float(ConvexHull(pts).volume)  # 2D -> area
#         else:
#             hull_area = 0.0
#     except Exception:
#         hull_area = 0.0

#     valid_hdg_d = heading_delta[~np.isnan(heading_delta)]
#     hdg_var   = float(np.var(valid_hdg_d)) if len(valid_hdg_d) else 0.0
#     hdg_rate  = float(np.mean(np.abs(valid_hdg_d))) if len(valid_hdg_d) else 0.0

#     # Time-weighted slow fraction
#     total_h   = float(np.nansum(dt_h))
#     if total_h > 0:
#         slow_h    = float(np.nansum(np.where(speed_kt[1:] < slow_thresh,    dt_h[1:], 0)))
#         stopped_h = float(np.nansum(np.where(speed_kt[1:] < stopped_thresh, dt_h[1:], 0)))
#         frac_slow = slow_h/ total_h
#         frac_stopped = stopped_h / total_h
#     else:
#         frac_slow = frac_stopped = 0.0

#     return {
#         "mean_speed_kt":      float(np.nanmean(valid_spd)),
#         "median_speed_kt":    float(np.nanmedian(valid_spd)),
#         "p95_speed_kt":       float(np.nanpercentile(valid_spd, 95)),
#         "std_speed_kt":       float(np.nanstd(valid_spd)),
#         "frac_slow":          frac_slow,
#         "frac_stopped":       frac_stopped,
#         "straightness":       straightness,
#         "hull_area_deg2":     hull_area,
#         "heading_variance":   hdg_var,
#         "heading_change_rate": hdg_rate,
#     }


# def proximity_features_all(df: pd.DataFrame, timestamps: np.ndarray,
#                             interval_h: float, max_radius_nm: float = 2.0) -> pd.DataFrame:
#     """
#     nearest-neighbor proximity features across all timestamps

#     Builds a single BallTree per timestep (not per vessel-pair), then obtains
#     each point for its nearest neighbor in one batched query.
#     Done in O(C log C) per timestep (low call count) 
#     """
#     #contacts = df["entity_id"].unique()
#     df["gi_code"], contacts = pd.factorize(df["entity_id"]) #create vectorized integer mapping for whole dataset
#     n_c = len(contacts)
#     #cid_idx = {c: i for i, c in enumerate(contacts)}

#     nn_min  = np.full(n_c, np.inf)
#     nn_sum  = np.zeros(n_c)
#     nn_cnt  = np.zeros(n_c, dtype=int)
#     h_1nm   = np.zeros(n_c)
#     h_2nm   = np.zeros(n_c)
#     coslow  = np.zeros(n_c)
#     was_close  = np.zeros(n_c, dtype=bool)
#     close_dur  = np.zeros(n_c)
#     ep_count   = np.zeros(n_c, dtype=int)

#     EPISODE_MIN_H = 1.0

#     by_ts = dict(tuple(df.groupby("timestamp", sort=False)))


#     for ts in timestamps:
#         sub = by_ts.get(ts)
#         if sub is None or len(sub) < 2:
#             continue

#         lats = sub["lat"].to_numpy()
#         lons = sub["lon"].to_numpy()
#         #ids  = sub["entity_id"].to_numpy()
#         spds = sub["speed_kt"].to_numpy() if "speed_kt" in sub.columns else None

#         rad_latlon = np.column_stack([np.radians(lats), np.radians(lons)])
#         tree = BallTree(rad_latlon, metric="haversine")
#         dist_rad, nn_ix = tree.query(rad_latlon, k=min(2, len(sub)))
#         dist_km  = dist_rad[:, -1] * EARTH_R_KM
#         dist_nm  = dist_km / KM_PER_NM
#         nn_idx_local = nn_ix[:, -1]

#         #gi_arr = np.array([cid_idx[c] for c in ids])
#         gi_arr = sub["gi_code"].to_numpy() 

#         nn_min_upd = np.minimum(nn_min[gi_arr], dist_nm)
#         nn_min[gi_arr] = nn_min_upd
#         nn_sum[gi_arr] += dist_nm
#         nn_cnt[gi_arr] += 1

#         is_close_now = dist_nm < CLOSE_RANGE_NM
#         is_2nm_now   = dist_nm < SHADOW_RANGE_NM

#         h_1nm[gi_arr[is_close_now]] += interval_h
#         h_2nm[gi_arr[is_2nm_now]]   += interval_h

#         if spds is not None:
#             both_slow = is_close_now & (spds < 1.0) & (spds[nn_idx_local] < 1.0)
#             coslow[gi_arr[both_slow]] += interval_h

#         # vectorized per-step update, still
#         # needs per-vessel running state carried across timesteps
#         close_dur[gi_arr[is_close_now]] += interval_h
#         ended = gi_arr[~is_close_now]
#         end_mask = was_close[ended] & (close_dur[ended] >= EPISODE_MIN_H)
#         ep_count[ended[end_mask]] += 1
#         close_dur[ended] = 0.0

#         was_close_new = np.zeros(n_c, dtype=bool)
#         was_close_new[gi_arr] = is_close_now

#         present_mask = np.zeros(n_c, dtype=bool)
#         present_mask[gi_arr] = True
#         was_close = np.where(present_mask, was_close_new, was_close)

#     for gi in range(n_c):
#         if was_close[gi] and close_dur[gi] >= EPISODE_MIN_H:
#             ep_count[gi] += 1

#     mean_nn = np.where(nn_cnt > 0, nn_sum / nn_cnt, np.inf)

#     result = pd.DataFrame({
#         "entity_id":              contacts,
#         "min_nn_dist_nm":         np.where(np.isinf(nn_min), 9999, nn_min),
#         "mean_nn_dist_nm":        np.where(np.isinf(mean_nn), 9999, mean_nn),
#         "hours_within_1nm":       h_1nm,
#         "hours_within_2nm":       h_2nm,
#         "sustained_close_episodes": ep_count,
#         "co_slow_hours":          coslow,
#     })
#     return result


# def estimate_interval_h(df: pd.DataFrame) -> float:
#     # bucketed to 5min slots (avoid skew from long silences)
#     gaps = (
#         df.sort_values(["entity_id", "timestamp"])
#           .groupby("entity_id")["timestamp"]
#           .diff()
#           .dt.total_seconds()
#           .dropna()
#           .div(3600.0)
#     )
#     buckets = (gaps * 12).round() / 12
#     interval = float(buckets[buckets < 2.0].mode().iloc[0])
#     return interval


# def estimate_contamination(df: pd.DataFrame, interval_h: float) -> float:
#     """    
#     current optimization : vectorized via groupby/transform rather than a per-vessel

#     TODO: Python loop (doubles the cost of the whole pipeline)
#     """
#     d = df.sort_values(["entity_id", "timestamp"]).copy()
#     g = d.groupby("entity_id", sort=False)

#     dt_h = g["timestamp"].diff().dt.total_seconds().div(3600.0)
#     big_gap = (dt_h > 10 * interval_h)

#     dlat = g["lat"].diff()
#     dlon = g["lon"].diff()

#     dist_km = haversine_km(d["lat"].shift(1), d["lon"].shift(1), d["lat"], d["lon"])
#     with np.errstate(divide="ignore", invalid="ignore"):
#         spd = np.where(dt_h > 0, dist_km / dt_h.to_numpy() / KM_PER_NM, 0.0)
#     fast = pd.Series(spd, index=d.index) > 60

#     suspicious_per_vessel = (big_gap | fast).groupby(d["entity_id"]).any()
#     rate = float(suspicious_per_vessel.mean())
#     return float(np.clip(rate, 0.005, 0.15))


# def build_feature_matrix(df: pd.DataFrame,
#                          nominal_interval_h: float = None) -> tuple:
#     # returns (feat_df, interval_h, contamination)
    
#     if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
#         df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

#     if nominal_interval_h is None:
#         nominal_interval_h = estimate_interval_h(df)
#         print(f"  Auto-detected ping interval : {nominal_interval_h*60:.1f} min")

#     contamination = estimate_contamination(df, nominal_interval_h)
#     print(f"  Auto-estimated contamination: {contamination:.3f} "
#           f"({contamination*100:.1f}% of contacts flagged as structurally suspicious)")

#     label_cols = [c for c in ["is_anomalous", "anomaly_type"] if c in df.columns]

#     all_feats = []
#     kin_chunks = []   #reuse 

#     for cid, grp in df.groupby("entity_id"):
#         grp = grp.sort_values("timestamp").reset_index(drop=True)
#         grp = compute_kinematics(grp)
#         kin_slim = grp[["entity_id", "lat", "lon", "timestamp", "speed_kt"]].copy()
#         kin_slim["lat"] = kin_slim["lat"].astype("float32")
#         kin_slim["lon"] = kin_slim["lon"].astype("float32")
#         kin_slim["speed_kt"] = kin_slim["speed_kt"].astype("float32")
#         kin_chunks.append(kin_slim)

#         lat = grp["lat"].to_numpy()
#         lon = grp["lon"].to_numpy()
#         spd = grp["speed_kt"].to_numpy()
#         hdg_d = grp["heading_delta"].to_numpy()
#         dt_h  = grp["dt_h"].to_numpy()
#         dist  = grp["dist_km"].to_numpy()

#         row: dict = {"entity_id": cid}

#         row.update(motion_features(lat, lon, spd, hdg_d, dt_h))
#         row.update(gap_features(dt_h, spd, dist, nominal_interval_h))
#         row.update(strata_proximity_features(lat, lon, dt_h))
#         row["min_port_dist_nm"] = min_dist_to_ports_nm(lat, lon)

#         # carry (purely for eval later)
#         if label_cols:
#             if "is_anomalous" in grp.columns:
#                 row["true_anomalous"] = int(grp["is_anomalous"].max())
#             if "anomaly_type" in grp.columns:
#                 types = grp["anomaly_type"].dropna().unique()
#                 types = [t for t in types if t not in ("none", "", "0")]
#                 row["true_type"] = types[0] if types else "none"

#         all_feats.append(row)

#     feat_df = pd.DataFrame(all_feats).set_index("entity_id")

#     timestamps = df["timestamp"].unique()
#     del df

#     df_kin = pd.concat(kin_chunks, ignore_index=True)
#     del kin_chunks

#     print(f"  Computing pairwise proximity across {len(timestamps)} timestamps "
#           f"× {len(feat_df)} contacts")
#     prox_df = proximity_features_all(df_kin, timestamps, nominal_interval_h)
#     prox_df = prox_df.set_index("entity_id")

#     feat_df = feat_df.join(prox_df, how="left")
#     return feat_df, nominal_interval_h, contamination

# feature extraction - per contact (vessel), not per ping
# builds a row of behavioral features from the full position history

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, KDTree

EARTH_R_KM = 6371.0088
KM_PER_NM = 1.852

CLOSE_RANGE_NM = 1.0
SHADOW_RANGE_NM = 2.0

def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
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
        raw_delta = np.diff(hdg[1:])
        hdg_delta[2:] = (raw_delta + 180) % 360 - 180

    df["dt_h"]         = dt_h
    df["dist_km"]      = dist_km
    df["speed_kt"]     = speed_kt
    df["heading_deg"]  = hdg
    df["heading_delta"]= hdg_delta
    return df


#major ports (lon, lat)
PORTS_LONLAT = [
    (-80.19, 25.77), (-82.38, 23.13), (-77.34, 25.06),
    (-76.79, 17.99), (-69.93, 18.47), (-66.10, 18.47),
    (-68.93, 12.11), (-61.52, 10.65), (-59.62, 13.10),
    (-75.51, 10.40), (-79.90,  9.36), (-86.85, 21.16),]

_PORT_TREE = None

def _port_tree():
    global _PORT_TREE
    if _PORT_TREE is None:
        ports = np.array(PORTS_LONLAT)
        p_lat = np.radians(ports[:, 1])
        p_lon = np.radians(ports[:, 0])
        _PORT_TREE = BallTree(np.column_stack([p_lat, p_lon]), metric="haversine")
    return _PORT_TREE


def min_dist_to_ports_nm(lat: np.ndarray, lon: np.ndarray) -> float:
    q_lat = np.radians(lat)
    q_lon = np.radians(lon)
    dist_rad, _ = _port_tree().query(np.column_stack([q_lat, q_lon]), k=1)
    dist_nm = dist_rad.flatten() * EARTH_R_KM / KM_PER_NM
    return float(dist_nm.min())

STRATA = [
    (-80.50, 23.60, 45.0),
    (-82.50, 15.50, 60.0),
    (-66.50, 12.50, 70.0),
    (-78.20, 17.10, 40.0),
]

def strata_proximity_features(lat: np.ndarray, lon: np.ndarray, dt_h: Optional[np.ndarray] = None) -> dict:
    n = len(lat)
    min_d_nm = np.full(n, np.inf)

    for s_lon, s_lat, s_r_km in STRATA:
        d_km = haversine_km(lat, lon, s_lat, s_lon)
        d_nm = d_km / KM_PER_NM
        inside = d_km <= s_r_km
        min_d_nm = np.minimum(min_d_nm, d_nm)

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


def gap_features(dt_h: np.ndarray, speed_kt: np.ndarray, dist_km: np.ndarray, nominal_interval_h: float) -> dict:
    valid_dt = dt_h[~np.isnan(dt_h)]
    if len(valid_dt) == 0:
        return {k: 0.0 for k in [
            "max_gap_h", "gap_count", "gap_frac", "max_implied_speed_kt",
            "max_gap_displacement_nm", "p95_gap_h"]}

    threshold_h = 2.5 * nominal_interval_h
    is_gap      = valid_dt > threshold_h
    n_gaps      = int(is_gap.sum())

    max_speed = float(np.nanmax(speed_kt)) if np.any(~np.isnan(speed_kt)) else 0.0

    gap_mask = (~np.isnan(dt_h[1:])) & (dt_h[1:] > threshold_h)
    if gap_mask.any():
        gap_d_nm = dist_km[1:][gap_mask] / KM_PER_NM
        max_gap_disp = float(gap_d_nm.max())
    else:
        max_gap_disp = 0.0

    SUSPICIOUS_KT = 60.0
    valid_spd_arr = speed_kt[~np.isnan(speed_kt)]
    n_suspicious = int((valid_spd_arr > SUSPICIOUS_KT).sum())

    return {
        "max_gap_h":                float(valid_dt.max()),
        "p95_gap_h":                float(np.percentile(valid_dt, 95)),
        "gap_count":                n_gaps,
        "gap_frac":                 float(n_gaps / max(len(valid_dt), 1)),
        "max_implied_speed_kt":     max_speed,
        "max_gap_displacement_nm":  max_gap_disp,
        "n_suspicious_speed_pings": n_suspicious,
    }


def motion_features(lat: np.ndarray, lon: np.ndarray, speed_kt: np.ndarray, heading_delta: np.ndarray, dt_h: np.ndarray) -> dict:
    valid_spd = speed_kt[~np.isnan(speed_kt)]

    if len(valid_spd) == 0:
        return {k: 0.0 for k in [
            "mean_speed_kt", "median_speed_kt", "p95_speed_kt", "std_speed_kt",
            "frac_slow", "frac_stopped", "straightness", "hull_area_deg2",
            "heading_variance", "heading_change_rate"]}

    slow_thresh    = 2.0
    stopped_thresh = 0.5

    net_km = haversine_km(lat[0], lon[0], lat[-1], lon[-1])
    path_km = float(np.nansum(np.where(~np.isnan(speed_kt[1:]),
                                        haversine_km(lat[:-1], lon[:-1],
                                                     lat[1:], lon[1:]), 0)))
    straightness = float(net_km / path_km) if path_km > 0 else 1.0

    try:
        from scipy.spatial import ConvexHull
        pts = np.column_stack([lon, lat])
        if len(np.unique(pts, axis=0)) >= 3:
            hull_area = float(ConvexHull(pts).volume)
        else:
            hull_area = 0.0
    except Exception:
        hull_area = 0.0

    valid_hdg_d = heading_delta[~np.isnan(heading_delta)]
    hdg_var  = float(np.var(valid_hdg_d))  if len(valid_hdg_d) else 0.0
    hdg_rate = float(np.mean(np.abs(valid_hdg_d))) if len(valid_hdg_d) else 0.0

    total_h = float(np.nansum(dt_h))
    if total_h > 0:
        slow_h    = float(np.nansum(np.where(speed_kt[1:] < slow_thresh,    dt_h[1:], 0)))
        stopped_h = float(np.nansum(np.where(speed_kt[1:] < stopped_thresh, dt_h[1:], 0)))
        frac_slow    = slow_h    / total_h
        frac_stopped = stopped_h / total_h
    else:
        frac_slow = frac_stopped = 0.0

    return {
        "mean_speed_kt":       float(np.nanmean(valid_spd)),
        "median_speed_kt":     float(np.nanmedian(valid_spd)),
        "p95_speed_kt":        float(np.nanpercentile(valid_spd, 95)),
        "std_speed_kt":        float(np.nanstd(valid_spd)),
        "frac_slow":           frac_slow,
        "frac_stopped":        frac_stopped,
        "straightness":        straightness,
        "hull_area_deg2":      hull_area,
        "heading_variance":    hdg_var,
        "heading_change_rate": hdg_rate,
    }


def proximity_features_all(df: pd.DataFrame, timestamps: np.ndarray,
                            interval_h: float, max_radius_nm: float = 2.0) -> pd.DataFrame:
    """
    nearest-neighbor proximity features across all timestamps

    Builds a single BallTree per timestep (not per vessel-pair), then obtains
    each point for its nearest neighbor in one batched query.
    Done in O(C log C) per timestep (low call count)
    """
    df["gi_code"], contacts = pd.factorize(df["entity_id"])
    n_c = len(contacts)

    nn_min     = np.full(n_c, np.inf)
    nn_sum     = np.zeros(n_c)
    nn_cnt     = np.zeros(n_c, dtype=int)
    h_1nm      = np.zeros(n_c)
    h_2nm      = np.zeros(n_c)
    coslow     = np.zeros(n_c)
    was_close  = np.zeros(n_c, dtype=bool)
    close_dur  = np.zeros(n_c)
    ep_count   = np.zeros(n_c, dtype=int)

    EPISODE_MIN_H = 1.0

    # pre-allocate reusable arrays instead of reallocating each iteration
    was_close_new = np.zeros(n_c, dtype=bool)
    present_mask  = np.zeros(n_c, dtype=bool)

    # lazy groupby - keeps only one 50k-row slice in memory at a time
    # instead of materializing all 8640 sub-DataFrames simultaneously
    for ts, sub in df.groupby("timestamp", sort=False):
        if len(sub) < 2:
            continue

        lats = sub["lat"].to_numpy()
        lons = sub["lon"].to_numpy()
        spds = sub["speed_kt"].to_numpy() if "speed_kt" in sub.columns else None

        # project lat/lon onto unit sphere for KDTree
        # chord length is monotonic with great-circle distance so NN order is exact
        lat_r = np.radians(lats)
        lon_r = np.radians(lons)
        xyz = np.column_stack([
            np.cos(lat_r) * np.cos(lon_r),
            np.cos(lat_r) * np.sin(lon_r),
            np.sin(lat_r),
        ])
        tree = KDTree(xyz)
        chord, nn_ix = tree.query(xyz, k=min(2, len(sub)))
        # chord -> great-circle radians -> km -> nm
        chord_clipped = np.clip(chord[:, -1] / 2.0, -1.0, 1.0)
        dist_km = 2.0 * EARTH_R_KM * np.arcsin(chord_clipped)
        dist_nm = dist_km / KM_PER_NM
        nn_idx_local = nn_ix[:, -1]

        gi_arr = sub["gi_code"].to_numpy()

        nn_min[gi_arr] = np.minimum(nn_min[gi_arr], dist_nm)
        nn_sum[gi_arr] += dist_nm
        nn_cnt[gi_arr] += 1

        is_close_now = dist_nm < CLOSE_RANGE_NM
        is_2nm_now   = dist_nm < SHADOW_RANGE_NM

        h_1nm[gi_arr[is_close_now]] += interval_h
        h_2nm[gi_arr[is_2nm_now]]   += interval_h

        if spds is not None:
            both_slow = is_close_now & (spds < 1.0) & (spds[nn_idx_local] < 1.0)
            coslow[gi_arr[both_slow]] += interval_h

        close_dur[gi_arr[is_close_now]] += interval_h
        ended    = gi_arr[~is_close_now]
        end_mask = was_close[ended] & (close_dur[ended] >= EPISODE_MIN_H)
        ep_count[ended[end_mask]] += 1
        close_dur[ended] = 0.0

        # reset pre-allocated arrays in place
        was_close_new[:] = False
        present_mask[:]  = False
        was_close_new[gi_arr] = is_close_now
        present_mask[gi_arr]  = True
        was_close = np.where(present_mask, was_close_new, was_close)

    for gi in range(n_c):
        if was_close[gi] and close_dur[gi] >= EPISODE_MIN_H:
            ep_count[gi] += 1

    mean_nn = np.where(nn_cnt > 0, nn_sum / nn_cnt, np.inf)

    result = pd.DataFrame({
        "entity_id":                contacts,
        "min_nn_dist_nm":           np.where(np.isinf(nn_min), 9999, nn_min),
        "mean_nn_dist_nm":          np.where(np.isinf(mean_nn), 9999, mean_nn),
        "hours_within_1nm":         h_1nm,
        "hours_within_2nm":         h_2nm,
        "sustained_close_episodes": ep_count,
        "co_slow_hours":            coslow,
    })
    return result


def estimate_interval_h(df: pd.DataFrame) -> float:
    gaps = (
        df.sort_values(["entity_id", "timestamp"])
          .groupby("entity_id")["timestamp"]
          .diff()
          .dt.total_seconds()
          .dropna()
          .div(3600.0)
    )
    buckets = (gaps * 12).round() / 12
    interval = float(buckets[buckets < 2.0].mode().iloc[0])
    return interval


def estimate_contamination(df: pd.DataFrame, interval_h: float) -> float:
    d = df.sort_values(["entity_id", "timestamp"]).copy()
    g = d.groupby("entity_id", sort=False)

    dt_h    = g["timestamp"].diff().dt.total_seconds().div(3600.0)
    big_gap = dt_h > 10 * interval_h

    dist_km = haversine_km(d["lat"].shift(1), d["lon"].shift(1), d["lat"], d["lon"])
    with np.errstate(divide="ignore", invalid="ignore"):
        spd = np.where(dt_h > 0, dist_km / dt_h.to_numpy() / KM_PER_NM, 0.0)
    fast = pd.Series(spd, index=d.index) > 60

    suspicious_per_vessel = (big_gap | fast).groupby(d["entity_id"]).any()
    rate = float(suspicious_per_vessel.mean())
    return float(np.clip(rate, 0.005, 0.15))


def build_feature_matrix(df: pd.DataFrame,
                         nominal_interval_h: float = None) -> tuple:
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if nominal_interval_h is None:
        nominal_interval_h = estimate_interval_h(df)
        print(f"  Auto-detected ping interval : {nominal_interval_h*60:.1f} min")

    contamination = estimate_contamination(df, nominal_interval_h)
    print(f"  Auto-estimated contamination: {contamination:.3f} "
          f"({contamination*100:.1f}% of contacts flagged as structurally suspicious)")

    label_cols = [c for c in ["is_anomalous", "anomaly_type"] if c in df.columns]

    all_feats  = []
    kin_chunks = []

    for cid, grp in df.groupby("entity_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        grp = compute_kinematics(grp)
        kin_slim = grp[["entity_id", "lat", "lon", "timestamp", "speed_kt"]].copy()
        kin_slim["lat"]      = kin_slim["lat"].astype("float32")
        kin_slim["lon"]      = kin_slim["lon"].astype("float32")
        kin_slim["speed_kt"] = kin_slim["speed_kt"].astype("float32")
        kin_chunks.append(kin_slim)

        lat   = grp["lat"].to_numpy()
        lon   = grp["lon"].to_numpy()
        spd   = grp["speed_kt"].to_numpy()
        hdg_d = grp["heading_delta"].to_numpy()
        dt_h  = grp["dt_h"].to_numpy()
        dist  = grp["dist_km"].to_numpy()

        row: dict = {"entity_id": cid}
        row.update(motion_features(lat, lon, spd, hdg_d, dt_h))
        row.update(gap_features(dt_h, spd, dist, nominal_interval_h))
        row.update(strata_proximity_features(lat, lon, dt_h))
        row["min_port_dist_nm"] = min_dist_to_ports_nm(lat, lon)

        if label_cols:
            if "is_anomalous" in grp.columns:
                row["true_anomalous"] = int(grp["is_anomalous"].max())
            if "anomaly_type" in grp.columns:
                types = grp["anomaly_type"].dropna().unique()
                types = [t for t in types if t not in ("none", "", "0")]
                row["true_type"] = types[0] if types else "none"

        all_feats.append(row)

    feat_df = pd.DataFrame(all_feats).set_index("entity_id")

    timestamps = df["timestamp"].unique()
    del df

    df_kin = pd.concat(kin_chunks, ignore_index=True)
    del kin_chunks

    print(f"  Computing pairwise proximity across {len(timestamps)} timestamps "
          f"x {len(feat_df)} contacts")
    prox_df = proximity_features_all(df_kin, timestamps, nominal_interval_h)
    prox_df = prox_df.set_index("entity_id")

    feat_df = feat_df.join(prox_df, how="left")
    return feat_df, nominal_interval_h, contamination