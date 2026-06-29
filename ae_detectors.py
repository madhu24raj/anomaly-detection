"""
ae_detectors.py  —  Deep Autoencoder ensemble for maritime anomaly detection.

Five complementary fully-connected autoencoders, each targeting a different
behavioural slice of the feature space, plus an optional LSTM sliding-window AE:

  1. MotionAE        – speed / heading kinematics
  2. SpatialAE       – stratum proximity + port distance + hull geometry
  3. ProximityAE     – inter-vessel co-location (with engineered conjunction features)
  4. GapAE           – transmission gap / dark-activity (NO upper-clip to preserve signal)
  5. FullAE          – all features together, wider network

Design decisions (informed by data analysis):
  - Log1p-transform all heavy-tailed columns BEFORE scaling (heading_variance 
    range 15-24k; std_speed 0.6-654; gap_h 0.08-96 — raw these dominate MSE loss).
  - frac_slow / frac_stopped have near-zero IQR so RobustScaler explodes their 
    post-scale range; they are log1p'd and then StandardScaled in their columns.
  - GapAE: do NOT clip to the 99.5th percentile — dark_activity vessels are 
    only 0.45% of the fleet and their 25-96h gaps ARE in the top 0.5%. Clipping 
    would destroy the exact signal we need to detect.
  - ProximityAE: engineered conjunction features (co_slow_hours * sustained_episodes, 
    co_slow / min_nn_dist) separate transshipment cleanly from normal vessels 
    that happen to share busy shipping lanes.
  - LSTM aggregation: 90th-percentile of window reconstruction errors (not max) 
    to avoid one noisy window flagging a normal vessel.
  - Sub-score soft-capping with tanh(2x) prevents a single sub-AE outlier from 
    overwhelming the ensemble via the max-pool fusion.

Training is 100% unsupervised — truth labels are stripped before any model sees the data.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Device / tensor helpers
# ─────────────────────────────────────────────────────────────────────────────

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _to_tensor(X: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(X.astype(np.float32)).to(device)

def _minmax_norm(s: np.ndarray) -> np.ndarray:
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo + 1e-12)

def _soft_cap(s: np.ndarray, alpha: float = 2.0) -> np.ndarray:
    """tanh soft-cap: maps [0,∞) → [0,1), compresses extreme outliers."""
    return np.tanh(alpha * s)


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation — the core of the fix
# ─────────────────────────────────────────────────────────────────────────────

def _prepare(df: pd.DataFrame, cols: List[str],
             log_cols: List[str]   = (),
             std_cols: List[str]   = (),
             no_upper_clip: bool   = False,
             clip_pct: Tuple[float,float] = (0.5, 99.5),
             ) -> np.ndarray:
    """
    Fill NaN → log1p selected columns → clip → RobustScale (or StandardScale
    for near-zero-IQR columns).

    Parameters
    ----------
    cols          Columns to use (filtered to those actually in df).
    log_cols      Apply log1p before scaling (heavy right tails).
    std_cols      Use StandardScaler instead of RobustScaler for these columns
                  (needed when IQR ≈ 0, e.g. frac_slow, frac_stopped).
    no_upper_clip When True skip the upper percentile clip (needed for GapAE 
                  where the anomaly signal IS in the extreme upper tail).
    clip_pct      Percentile bounds for clipping (ignored for upper if no_upper_clip).
    """
    avail = [c for c in cols if c in df.columns]
    X = df[avail].fillna(0.0).to_numpy(dtype=np.float64).copy()

    # 1. Log1p transform (before scaling to tame heavy tails)
    for c in log_cols:
        if c in avail:
            ci = avail.index(c)
            X[:, ci] = np.log1p(np.abs(X[:, ci]))

    # 2. Clip (optionally skip upper bound)
    lo_pct = np.percentile(X, clip_pct[0], axis=0)
    hi_pct = np.percentile(X, clip_pct[1], axis=0)
    if no_upper_clip:
        X = np.clip(X, lo_pct, None)
    else:
        X = np.clip(X, lo_pct, hi_pct)

    # 3. Scale column-by-column
    result = np.zeros_like(X, dtype=np.float32)
    for ci, c in enumerate(avail):
        col = X[:, ci].reshape(-1, 1)
        if c in std_cols:
            result[:, ci] = StandardScaler().fit_transform(col).ravel()
        else:
            result[:, ci] = RobustScaler().fit_transform(col).ravel()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

class _FCAE(nn.Module):
    def __init__(self, input_dim: int, hidden: Tuple[int,...],
                 bottleneck: int, dropout: float = 0.10):
        super().__init__()
        enc, prev = [], input_dim
        for h in hidden:
            enc += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        enc += [nn.Linear(prev, bottleneck), nn.ELU()]
        self.encoder = nn.Sequential(*enc)

        dec, prev = [], bottleneck
        for h in reversed(hidden):
            dec += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return ((x - self(x)) ** 2).mean(dim=1)


class _LSTMAE(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 48, latent: int = 20,
                 n_layers: int = 2, dropout: float = 0.10):
        super().__init__()
        self.n_layers = n_layers
        self.enc = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.lat_dn = nn.Linear(hidden, latent)
        self.lat_up = nn.Linear(latent, hidden)
        self.dec    = nn.LSTM(hidden, hidden, n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        self.out    = nn.Linear(hidden, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        _, (h_n, _) = self.enc(x)
        lat = torch.tanh(self.lat_dn(h_n[-1]))
        h0  = torch.tanh(self.lat_up(lat)).unsqueeze(0).repeat(self.n_layers, 1, 1)
        dec_in = h0[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.dec(dec_in, (h0, torch.zeros_like(h0)))
        return self.out(out)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# Training helper (shared for all FC AEs)
# ─────────────────────────────────────────────────────────────────────────────

def _train(model: nn.Module, X: np.ndarray,
           epochs: int = 100, batch: int = 512,
           lr: float = 3e-3, patience: int = 15,
           device: Optional[torch.device] = None,
           verbose: bool = True, label: str = "") -> nn.Module:
    dev   = device or _device()
    model = model.to(dev)
    Xt    = _to_tensor(X, dev)
    dl    = DataLoader(TensorDataset(Xt), batch_size=batch, shuffle=True)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.MSELoss()

    best, best_sd, pc = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            loss = crit(model(xb), xb.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        sched.step()
        ep_loss /= len(Xt)
        if ep_loss < best - 1e-7:
            best, best_sd, pc = ep_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= patience:
                break
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"      ep {ep:3d}  loss={ep_loss:.5f}  [{label}]")
    if best_sd:
        model.load_state_dict(best_sd)
    return model.eval()


def _score_fc(feat: pd.DataFrame, cols: List[str],
              hidden: Tuple[int,...], bottleneck: int,
              label: str, epochs: int,
              log_cols: List[str] = (),
              std_cols: List[str] = (),
              no_upper_clip: bool = False,
              device: torch.device = None,
              verbose: bool = True) -> np.ndarray:
    avail = [c for c in cols if c in feat.columns]
    if len(avail) < 2:
        warnings.warn(f"[{label}] only {len(avail)} cols — skipping")
        return np.zeros(len(feat), dtype=np.float32)
    X     = _prepare(feat, avail, log_cols=log_cols, std_cols=std_cols,
                     no_upper_clip=no_upper_clip)
    model = _FCAE(X.shape[1], hidden, bottleneck)
    model = _train(model, X, epochs=epochs, device=device,
                   label=label, verbose=verbose)
    err   = model.reconstruction_error(_to_tensor(X, device or _device())).cpu().numpy()
    return _soft_cap(_minmax_norm(err))


# ─────────────────────────────────────────────────────────────────────────────
# Feature column sets (with fixes baked in)
# ─────────────────────────────────────────────────────────────────────────────

MOTION_COLS = [
    "mean_speed_kt", "median_speed_kt", "p95_speed_kt", "std_speed_kt",
    "frac_slow", "frac_stopped",
    "straightness", "heading_variance", "heading_change_rate",
]
# Heavy right tails → log1p; near-zero-IQR → StandardScale
MOTION_LOG = ["std_speed_kt", "heading_variance", "heading_change_rate",
              "frac_slow", "frac_stopped"]
MOTION_STD = ["frac_slow", "frac_stopped"]

SPATIAL_COLS = [
    "frac_inside_stratum", "hours_inside_stratum", "min_stratum_dist_nm",
    "hull_area_deg2", "min_port_dist_nm",
]
SPATIAL_LOG = ["hours_inside_stratum", "hull_area_deg2", "min_port_dist_nm"]
SPATIAL_STD = ["frac_inside_stratum"]

# Proximity: base columns + engineered conjunction features added inline
PROXIMITY_BASE = [
    "co_slow_hours", "hours_within_1nm", "hours_within_2nm",
    "sustained_close_episodes", "min_nn_dist_nm", "mean_nn_dist_nm",
]
PROXIMITY_LOG = ["co_slow_hours", "hours_within_1nm", "hours_within_2nm"]

# Gap: NO upper clip — dark_activity vessels (0.45% of fleet) are in the upper tail
GAP_COLS = [
    "max_gap_h", "p95_gap_h", "gap_count", "gap_frac",
    "max_implied_speed_kt", "max_gap_displacement_nm", "n_suspicious_speed_pings",
]
GAP_LOG = ["max_gap_h", "max_gap_displacement_nm", "max_implied_speed_kt"]

ALL_COLS = list(dict.fromkeys(MOTION_COLS + SPATIAL_COLS + PROXIMITY_BASE + GAP_COLS))
ALL_LOG  = list(dict.fromkeys(MOTION_LOG + SPATIAL_LOG + PROXIMITY_LOG + GAP_LOG))
ALL_STD  = list(dict.fromkeys(MOTION_STD + SPATIAL_STD))

LSTM_COLS = ["speed_kt", "heading_deg", "heading_delta", "dt_h", "dist_km"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-group scorers
# ─────────────────────────────────────────────────────────────────────────────

def score_motion_ae(feat: pd.DataFrame, device: torch.device,
                    verbose: bool = True) -> pd.Series:
    s = _score_fc(feat, MOTION_COLS,
                  hidden=(64, 32), bottleneck=12,
                  label="MotionAE", epochs=120,
                  log_cols=MOTION_LOG, std_cols=MOTION_STD,
                  device=device, verbose=verbose)
    return pd.Series(s, index=feat.index, name="ae_score_motion")


def score_spatial_ae(feat: pd.DataFrame, device: torch.device,
                     verbose: bool = True) -> pd.Series:
    s = _score_fc(feat, SPATIAL_COLS,
                  hidden=(32, 16), bottleneck=6,
                  label="SpatialAE", epochs=100,
                  log_cols=SPATIAL_LOG, std_cols=SPATIAL_STD,
                  device=device, verbose=verbose)
    return pd.Series(s, index=feat.index, name="ae_score_spatial")


def score_proximity_ae(feat: pd.DataFrame, device: torch.device,
                       verbose: bool = True) -> pd.Series:
    """
    Proximity AE with engineered conjunction features.

    Key insight: normal vessels in busy shipping lanes have high hours_within_1nm
    but near-zero co_slow_hours.  Transshipment vessels have high co_slow_hours
    AND low min_nn_dist simultaneously.  Conjunction features capture this:
      transship_signature = co_slow_hours * sustained_close_episodes
      deep_proximity      = co_slow_hours / (min_nn_dist + 0.01)
    """
    f = feat.copy()
    f["transship_signature"] = (
        np.log1p(f["co_slow_hours"].fillna(0)) *
        np.log1p(f["sustained_close_episodes"].fillna(0)))
    f["deep_proximity"] = (
        f["co_slow_hours"].fillna(0) /
        (f["min_nn_dist_nm"].fillna(0.1) + 0.01))
    f["slow_prox_ratio"] = (
        f["co_slow_hours"].fillna(0) /
        (f["hours_within_2nm"].fillna(0) + 0.1))
    f["neg_log_min_nn"] = -np.log1p(f["min_nn_dist_nm"].fillna(0))

    cols = PROXIMITY_BASE + ["transship_signature", "deep_proximity",
                              "slow_prox_ratio", "neg_log_min_nn"]
    log_cols = PROXIMITY_LOG + ["deep_proximity", "transship_signature"]

    s = _score_fc(f, cols,
                  hidden=(64, 32), bottleneck=10,
                  label="ProximityAE", epochs=100,
                  log_cols=log_cols,
                  device=device, verbose=verbose)
    return pd.Series(s, index=feat.index, name="ae_score_proximity")


def score_gap_ae(feat: pd.DataFrame, device: torch.device,
                 verbose: bool = True) -> pd.Series:
    """
    Gap AE — no upper-percentile clip.

    dark_activity vessels have max_gap_h 25-96h; normal vessels 0.083h.
    This separation only exists in the extreme upper tail, which a 99.5th-percentile
    clip would destroy (dark vessels are 0.45% of fleet = right at the clip boundary).
    """
    s = _score_fc(feat, GAP_COLS,
                  hidden=(48, 24), bottleneck=8,
                  label="GapAE", epochs=100,
                  log_cols=GAP_LOG, no_upper_clip=True,
                  device=device, verbose=verbose)
    return pd.Series(s, index=feat.index, name="ae_score_gap")


def score_full_ae(feat: pd.DataFrame, device: torch.device,
                  verbose: bool = True) -> pd.Series:
    """Wide AE across all features — captures cross-group correlations."""
    avail = [c for c in ALL_COLS if c in feat.columns]
    X     = _prepare(feat, avail, log_cols=ALL_LOG, std_cols=ALL_STD,
                     no_upper_clip=False)
    bn    = max(12, X.shape[1] // 5)
    model = _FCAE(X.shape[1], hidden=(128, 64, 32), bottleneck=bn, dropout=0.12)
    model = _train(model, X, epochs=140, lr=2e-3,
                   device=device, label="FullAE", verbose=verbose)
    err   = model.reconstruction_error(_to_tensor(X, device)).cpu().numpy()
    return pd.Series(_soft_cap(_minmax_norm(err)), index=feat.index, name="ae_score_full")


def score_lstm_ae(raw_df: pd.DataFrame, index: pd.Index,
                  device: torch.device,
                  window: int = 48,
                  stride: int = 12,
                  verbose: bool = True) -> pd.Series:
    """
    LSTM AE using sliding windows (scalable to large fleets).

    Each vessel track is sliced into windows of `window` pings.  The LSTM AE
    trains on all windows jointly and learns normal temporal motion patterns.
    Per-vessel score = 90th-percentile of window reconstruction errors
    (not max, to avoid flagging normal vessels with one unusual acceleration event).

    window=48 @ 15-min cadence = 12h window (covers any anomaly episode).
    window=48 @ 5-min cadence  = 4h window (adjust if needed).
    """
    from features import compute_kinematics

    needed = {"speed_kt", "heading_deg", "dt_h", "dist_km"}
    if not needed.issubset(raw_df.columns):
        chunks = []
        for _, grp in raw_df.groupby("entity_id"):
            chunks.append(compute_kinematics(grp.copy()))
        raw_df = pd.concat(chunks, ignore_index=True)

    avail = [c for c in LSTM_COLS if c in raw_df.columns]
    F     = len(avail)

    # Single global scaler fitted on all pings
    all_vals = raw_df[avail].fillna(0.0).to_numpy(dtype=np.float32).copy()
    for ci, c in enumerate(avail):
        if c in ["dt_h", "dist_km", "speed_kt"]:   # right-skewed
            all_vals[:, ci] = np.log1p(np.abs(all_vals[:, ci]))
    lo = np.percentile(all_vals, 1.0, axis=0)
    hi = np.percentile(all_vals, 99.0, axis=0)
    all_vals = np.clip(all_vals, lo, hi)
    scaler   = StandardScaler().fit(all_vals)

    # Build sliding windows
    win_list: List[np.ndarray] = []
    win_eid:  List[int]        = []

    for eid, grp in raw_df.groupby("entity_id"):
        raw = grp.sort_values("timestamp")[avail].fillna(0.0).to_numpy(dtype=np.float32).copy()
        for ci, c in enumerate(avail):
            if c in ["dt_h", "dist_km", "speed_kt"]:
                raw[:, ci] = np.log1p(np.abs(raw[:, ci]))
        v = scaler.transform(np.clip(raw, lo, hi))
        T = len(v)
        if T < window:
            pad = np.zeros((window, F), dtype=np.float32)
            pad[:T] = v
            win_list.append(pad)
            win_eid.append(eid)
        else:
            for start in range(0, T - window + 1, stride):
                win_list.append(v[start:start + window].astype(np.float32))
                win_eid.append(eid)

    X_win = np.stack(win_list, axis=0)   # (N_windows, T, F)
    Xt    = torch.from_numpy(X_win).to(device)
    dl    = DataLoader(TensorDataset(Xt), batch_size=256, shuffle=True)

    model = _LSTMAE(F, hidden=48, latent=20, n_layers=2)
    model = _train(model, X_win.reshape(len(X_win), -1),   # dummy — train below
                   epochs=1, verbose=False)   # init weights only; real training below
    model = model.to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    crit  = nn.MSELoss()
    best, best_sd, pc = np.inf, None, 0

    for ep in range(40):
        model.train()
        ep_loss = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            loss = crit(model(xb), xb.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        sched.step()
        ep_loss /= len(X_win)
        if ep_loss < best - 1e-7:
            best, best_sd, pc = ep_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= 8:
                break
        if verbose and (ep % 10 == 0 or ep == 39):
            print(f"      ep {ep:3d}  loss={ep_loss:.5f}  [LSTM-AE]")
    if best_sd:
        model.load_state_dict(best_sd)
    model.eval()

    # Score all windows; aggregate per vessel at 90th percentile (not max)
    with torch.no_grad():
        errs = model.reconstruction_error(Xt).cpu().numpy()

    win_eid_arr = np.array(win_eid)
    vessel_score: Dict[int, float] = {}
    for eid in np.unique(win_eid_arr):
        e = errs[win_eid_arr == eid]
        # 90th pct: catches sustained anomalies, resistant to single-window noise
        vessel_score[eid] = float(np.percentile(e, 90))

    eids   = list(vessel_score.keys())
    scores = np.array([vessel_score[e] for e in eids], dtype=np.float32)

    return pd.Series(
        _soft_cap(_minmax_norm(scores)),
        index=pd.Index(eids, name="entity_id"),
        name="ae_score_lstm",
    ).reindex(index).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble fusion
# ─────────────────────────────────────────────────────────────────────────────

AE_WEIGHTS: Dict[str, float] = {
    "ae_score_motion":    1.10,   # AIS spoofing (impossible speeds), erratic pursuit
    "ae_score_spatial":   1.00,   # illegal fishing (loitering in restricted zones)
    "ae_score_proximity": 1.10,   # transshipment & aggressive maneuvering
    "ae_score_gap":       1.20,   # dark activity (strongest clean signal after fix)
    "ae_score_full":      1.05,   # cross-group correlation anomalies
    "ae_score_lstm":      0.95,   # temporal episode patterns (slightly downweighted on CPU)
}


def ae_ensemble_scores(feat: pd.DataFrame,
                       raw_df: Optional[pd.DataFrame] = None,
                       use_lstm: bool = True,
                       device: Optional[torch.device] = None,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Train all AE sub-detectors and fuse into a single ensemble score.

    Parameters
    ----------
    feat      Feature matrix from features.build_feature_matrix()
              (truth columns stripped automatically before training).
    raw_df    Raw AIS pings — required when use_lstm=True.
    use_lstm  Include LSTM sliding-window AE (slower, set --no-lstm to skip).
    device    Torch device (CUDA if available, else CPU).
    verbose   Print per-epoch training loss.

    Returns
    -------
    DataFrame indexed by entity_id with:
      ae_score_motion, ae_score_spatial, ae_score_proximity,
      ae_score_gap, ae_score_full, [ae_score_lstm],
      ensemble_score, ensemble_mean, [true_anomalous, true_type]
    """
    dev = device or _device()

    # Strip truth — models must NEVER see labels
    truth_cols = [c for c in ["true_anomalous", "true_type"] if c in feat.columns]
    feat_only  = feat.drop(columns=truth_cols, errors="ignore")

    print("  [1/5] AE: motion kinematics")
    s1 = score_motion_ae(feat_only, dev, verbose)

    print("  [2/5] AE: spatial / stratum proximity")
    s2 = score_spatial_ae(feat_only, dev, verbose)

    print("  [3/5] AE: inter-vessel proximity  (conjunction features)")
    s3 = score_proximity_ae(feat_only, dev, verbose)

    print("  [4/5] AE: transmission gap / dark activity  (no upper clip)")
    s4 = score_gap_ae(feat_only, dev, verbose)

    print("  [5/5] AE: full feature vector  (cross-group correlations)")
    s5 = score_full_ae(feat_only, dev, verbose)

    scores: Dict[str, pd.Series] = {
        "ae_score_motion":    s1,
        "ae_score_spatial":   s2,
        "ae_score_proximity": s3,
        "ae_score_gap":       s4,
        "ae_score_full":      s5,
    }

    if use_lstm:
        if raw_df is not None:
            print("  [6/6] AE: LSTM sliding-window temporal")
            s6 = score_lstm_ae(raw_df, feat_only.index, dev, verbose=verbose)
            scores["ae_score_lstm"] = s6
        else:
            warnings.warn("use_lstm=True but raw_df not provided — skipping LSTM AE")

    scores_df = pd.DataFrame(scores)

    # Weighted max-pool — same fusion strategy as the rule-based pipeline
    W        = np.array([AE_WEIGHTS.get(c, 1.0) for c in scores_df.columns])
    weighted = scores_df.to_numpy() * W[None, :]
    scores_df["ensemble_score"] = weighted.max(axis=1)
    scores_df["ensemble_mean"]  = weighted.mean(axis=1)

    if truth_cols:
        scores_df = scores_df.join(feat[truth_cols])

    return scores_df


def ae_detect(feat: pd.DataFrame,
              raw_df: Optional[pd.DataFrame] = None,
              use_lstm: bool = True,
              threshold: float = 0.50,
              device: Optional[torch.device] = None,
              verbose: bool = True) -> pd.DataFrame:
    """
    Full AE detection pipeline.

    Parameters
    ----------
    feat       Feature matrix (index = entity_id).  Truth columns passed through
               for evaluation but NEVER used in training.
    raw_df     Raw AIS pings — only needed when use_lstm=True.
    use_lstm   Include LSTM sequence AE (recommended for GPU; use --no-lstm on CPU).
    threshold  Ensemble score cutoff for binary flagging (default 0.50, tune down
               to 0.20 for higher recall at cost of precision).
    device     Torch device override.
    verbose    Print training progress.

    Returns
    -------
    DataFrame with ae_score_*, ensemble_score, ensemble_mean, flagged.
    """
    results = ae_ensemble_scores(feat, raw_df=raw_df, use_lstm=use_lstm,
                                 device=device, verbose=verbose)
    results["flagged"] = results["ensemble_score"] >= threshold
    return results
