# detectors.py
# rule-based + ml anomaly scoring, per contact
# no ground truth used anywhere here

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


def _scale(X: np.ndarray) -> np.ndarray:
    return RobustScaler().fit_transform(X)


def _if_score(X: np.ndarray, contamination: float = 0.05,
              n_estimators: int = 200, seed: int = 0) -> np.ndarray:
    """Isolation Forest anomaly score, normalised to [0, 1] (higher = stranger)."""
    clf = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                          random_state=seed, n_jobs=-1)
    clf.fit(X)
    raw = -clf.score_samples(X)          # already a "strangeness" score
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo + 1e-12)


def _lof_score(X: np.ndarray, n_neighbors: int = 20,
               contamination: float = 0.05) -> np.ndarray:
    """LOF anomaly score, normalised to [0, 1]."""
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination,
                              novelty=False, n_jobs=-1)
    clf.fit(X)
    raw = -clf.negative_outlier_factor_
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo + 1e-12)


def score_teleport(feat: pd.DataFrame) -> pd.Series:
    """
    Contacts that exhibit an implied speed far beyond any plausible vessel
    physics between consecutive pings — a hard physics violation.

    Requires at least 2 suspicious-speed pings (not just a single max spike)
    to score highly, so isolated GPS jitter doesn't dominate.
    """
    HARD_CAP     = 50.0   # kt — raised; single-ping jitter can reach 40-50
    SOFT_CAP     = 35.0   # kt — above this we start scoring
    MIN_PINGS    = 2      # need at least this many suspicious pings to score > 0.5

    spd   = feat["max_implied_speed_kt"].fillna(0.0)
    n_sus = feat["n_suspicious_speed_pings"].fillna(0.0)

    # Base score from max speed
    base  = np.clip((spd - SOFT_CAP) / (HARD_CAP - SOFT_CAP + 1e-6), 0, 1)
    # Confidence gate: single-ping spikes get halved
    conf  = np.where(n_sus >= MIN_PINGS, 1.0, 0.45)
    score = base * conf

    return pd.Series(score.to_numpy(), index=feat.index, name="score_teleport")


def score_blackout(feat: pd.DataFrame, nominal_interval_h: float = 0.25) -> pd.Series:
    # long silence + displacement when it comes back
    SILENCE_THRESHOLD_H = 4.0    # gaps below this are just noise / lost signal

    gap_h  = feat["max_gap_h"].fillna(0.0)
    disp   = feat["max_gap_displacement_nm"].fillna(0.0)

    # Normalise gap length: 0 below threshold, 1 at 48 h
    gap_score  = np.clip((gap_h - SILENCE_THRESHOLD_H) / (48.0 - SILENCE_THRESHOLD_H), 0, 1)
    # Normalise displacement: 0 at 0 nm, 1 at 300 nm
    disp_score = np.clip(disp / 300.0, 0, 1)

    score = 0.6 * gap_score + 0.4 * disp_score
    return pd.Series(score.to_numpy(), index=feat.index, name="score_blackout")


def score_slow_in_stratum(feat: pd.DataFrame,
                           contamination: float = 0.05) -> pd.Series:
    # slow loitering near sensitive zones, away from port
    cols = [
        "frac_slow", "frac_stopped",
        "hours_inside_stratum", "min_stratum_dist_nm",
        "straightness", "hull_area_deg2",
        "min_port_dist_nm",
    ]
    X = feat[cols].fillna(0.0).to_numpy()

    # Flip sign for features where *low* is suspicious
    X[:, 3] = -X[:, 3]   # min_stratum_dist_nm: closer = worse
    X[:, 4] = -X[:, 4]   # straightness: lower = more loitering-like
    X[:, 5] = -X[:, 5]   # hull_area: smaller = confined area

    Xs = _scale(X)
    s_if  = _if_score(Xs, contamination=contamination)
    s_lof = _lof_score(Xs, contamination=contamination)

    score = 0.55 * s_if + 0.45 * s_lof
    return pd.Series(score, index=feat.index, name="score_slow_stratum")


def score_colocation(feat: pd.DataFrame,
                     contamination: float = 0.05) -> pd.Series:
    """
    Two contacts meeting at sea: both near-stopped, co-located, far from port.

    Features: co_slow_hours, hours_within_1nm, sustained_close_episodes,
              min_port_dist_nm (far from port makes it more suspicious),
              mean_speed_kt (slow during rendezvous).
    """
    cols = [
        "co_slow_hours", "hours_within_1nm",
        "sustained_close_episodes", "min_port_dist_nm",
        "frac_stopped",
    ]
    X = feat[cols].fillna(0.0).to_numpy()

    # min_port_dist_nm: far = suspicious for co-location
    # (flip so high distance = high anomaly contribution)
    X[:, 3] = X[:, 3]   # already: farther port → bigger value → IF sees it as outlier

    Xs = _scale(X)
    s_if  = _if_score(Xs, contamination=contamination)
    s_lof = _lof_score(Xs, n_neighbors=15, contamination=contamination)

    score = 0.6 * s_if + 0.4 * s_lof
    return pd.Series(score, index=feat.index, name="score_colocation")


def score_erratic_pursuit(feat: pd.DataFrame,
                           contamination: float = 0.05) -> pd.Series:
    # erratic movement at close range to another vessel
    cols = [
        "heading_variance", "heading_change_rate",
        "hours_within_2nm", "p95_speed_kt",
        "std_speed_kt", "min_nn_dist_nm",
    ]
    X = feat[cols].fillna(0.0).to_numpy()
    X[:, 5] = -X[:, 5]   # min_nn_dist: closer = more suspicious

    Xs = _scale(X)
    s_if  = _if_score(Xs, contamination=contamination)
    s_lof = _lof_score(Xs, contamination=contamination)

    score = 0.5 * s_if + 0.5 * s_lof
    return pd.Series(score, index=feat.index, name="score_erratic_pursuit")


def ensemble_scores(feat: pd.DataFrame,
                    nominal_interval_h: float = 0.25,
                    contamination: float = 0.05) -> pd.DataFrame:
    """run all detectors and combine into ensemble_score"""
    print("  [1/5] Score: speed teleport (rule-based)...")
    s1 = score_teleport(feat)

    print("  [2/5] Score: transmission blackout (rule-based)...")
    s2 = score_blackout(feat, nominal_interval_h)

    print("  [3/5] Score: slow operations near sensitive areas (IF + LOF)...")
    s3 = score_slow_in_stratum(feat, contamination)

    print("  [4/5] Score: at-sea co-location (IF + LOF)...")
    s4 = score_colocation(feat, contamination)

    print("  [5/5] Score: erratic close-range pursuit (IF + LOF)...")
    s5 = score_erratic_pursuit(feat, contamination)

    scores = pd.DataFrame({
        "score_teleport":       s1,
        "score_blackout":       s2,
        "score_slow_stratum":   s3,
        "score_colocation":     s4,
        "score_erratic_pursuit": s5,
    })

    # Weighted combination.
    # Teleport weight reduced: a speed violation alone is necessary but not
    # sufficient — real deployments have GPS glitches and AIS relay errors.
    # The ensemble needs at least partial corroboration from a behavioural detector.
    W = np.array([1.05, 1.2, 1.0, 1.0, 1.0])
    weighted = scores.to_numpy() * W[None, :]

    # Primary score: weighted max (loudest single alarm)
    max_score = weighted.max(axis=1)

    # Soft corroboration bonus: if the top detector is teleport AND at least one
    # behavioural detector also scores > 0.25, keep full score; otherwise dampen.
    teleport_col  = list(scores.columns).index("score_teleport")
    behav_cols    = [i for i, c in enumerate(scores.columns)
                     if c not in ("score_teleport", "score_blackout")]
    is_teleport_led = (weighted.argmax(axis=1) == teleport_col)
    has_corroboration = (scores.iloc[:, behav_cols].values.max(axis=1) > 0.25)
    dampen = is_teleport_led & ~has_corroboration
    max_score = np.where(dampen, max_score * 0.72, max_score)

    scores["ensemble_score"] = max_score
    scores["ensemble_mean"]  = weighted.mean(axis=1)

    return scores


def detect(feat: pd.DataFrame,
           nominal_interval_h: float = 0.25,
           contamination: float = 0.05,
           threshold: float = 0.50) -> pd.DataFrame:
    """
    Full detection pipeline.  Returns a result dataframe with scores + flag.

    Parameters
    ----------
    feat               Feature matrix from features.build_feature_matrix().
    nominal_interval_h Expected ping cadence in hours (default 15 min → 0.25).
    contamination      Expected fraction of anomalous contacts; used by IF/LOF.
    threshold          Ensemble score above which a contact is flagged.

    Returns
    -------
    DataFrame indexed by entity_id with all detector scores, ensemble_score,
    and a boolean `flagged` column.  Ground-truth columns (true_anomalous,
    true_type) are passed through if present.
    """
    truth_cols = [c for c in ["true_anomalous", "true_type"] if c in feat.columns]
    truth = feat[truth_cols].copy() if truth_cols else None

    feat_only = feat.drop(columns=truth_cols, errors="ignore")

    scores = ensemble_scores(feat_only, nominal_interval_h, contamination)
    scores["flagged"] = scores["ensemble_score"] >= threshold

    if truth is not None:
        scores = scores.join(truth)

    return scores