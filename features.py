import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from scipy.spatial import cKDTree  # High-performance C-implemented KDTree

# Keep your constants intact
EARTH_R_KM = 6371.0088
KM_PER_NM = 1.852
CLOSE_RANGE_NM = 1.0
SHADOW_RANGE_NM = 2.0

def proximity_features_all(df: pd.DataFrame, timestamps: np.ndarray,
                           interval_h: float, max_radius_nm: float = 2.0) -> pd.DataFrame:
    """
    Nearest-neighbor proximity features across all timestamps.
    
    Optimized by mapping Lat/Lon coordinates to a 3D Unit Sphere and using 
    SciPy's cKDTree, alongside a pre-mapped vectorized global index array.
    """
    # Create a shallow copy to safely append temporary processing columns
    df = df.copy()

    # OPTIMIZATION 1: Vectorized Global Index Mapping (Removes Python dict loop lookup)
    contacts = df["entity_id"].unique()
    n_c = len(contacts)
    cid_idx = {c: i for i, c in enumerate(contacts)}
    df["global_index"] = df["entity_id"].map(cid_idx)

    # OPTIMIZATION 2: Pre-calculate 3D Unit Sphere Cartesian Coordinates
    # Avoids computing radians, sines, and cosines inside the per-timestep loop
    lats_rad = np.radians(df["lat"].to_numpy())
    lons_rad = np.radians(df["lon"].to_numpy())
    df["x"] = np.cos(lats_rad) * np.cos(lons_rad)
    df["y"] = np.cos(lats_rad) * np.sin(lons_rad)
    df["z"] = np.sin(lats_rad)

    # Initialize tracking arrays using global integer index positions
    nn_min  = np.full(n_c, np.inf)
    nn_sum  = np.zeros(n_c)
    nn_cnt  = np.zeros(n_c, dtype=int)
    h_1nm   = np.zeros(n_c)
    h_2nm   = np.zeros(n_c)
    coslow  = np.zeros(n_c)
    was_close  = np.zeros(n_c, dtype=bool)
    close_dur  = np.zeros(n_c)
    ep_count   = np.zeros(n_c, dtype=int)

    EPISODE_MIN_H = 1.0

    # Groupby optimization step
    by_ts = dict(tuple(df.groupby("timestamp", sort=False)))

    for ts in timestamps:
        sub = by_ts.get(ts)
        if sub is None or len(sub) < 2:
            continue

        # Fast extraction of arrays from precalculated DataFrame slice
        xyz = sub[["x", "y", "z"]].to_numpy()
        gi_arr = sub["global_index"].to_numpy()
        spds = sub["speed_kt"].to_numpy() if "speed_kt" in sub.columns else None

        # Build Euclidean cKDTree on the unit sphere chunk
        tree = cKDTree(xyz)
        dist_euclidean, nn_ix = tree.query(xyz, k=min(2, len(sub)))
        
        # Target the immediate nearest neighbor columns
        de = dist_euclidean[:, -1]
        nn_idx_local = nn_ix[:, -1]

        # Convert straight-line Euclidean chord distance on unit sphere to central angle radians:
        # Δσ = 2 * arcsin(d_E / 2)
        dist_rad = 2 * np.arcsin(np.clip(de / 2, 0.0, 1.0))
        dist_km  = dist_rad * EARTH_R_KM
        dist_nm  = dist_km / KM_PER_NM

        # Run vector updates on global indices
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

        # Running history tracking across timesteps
        close_dur[gi_arr[is_close_now]] += interval_h
        ended = gi_arr[~is_close_now]
        end_mask = was_close[ended] & (close_dur[ended] >= EPISODE_MIN_H)
        ep_count[ended[end_mask]] += 1
        close_dur[ended] = 0.0

        was_close_new = np.zeros(n_c, dtype=bool)
        was_close_new[gi_arr] = is_close_now

        present_mask = np.zeros(n_c, dtype=bool)
        present_mask[gi_arr] = True
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