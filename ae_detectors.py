"""
ae_detectors.py  —  Deep Autoencoder ensemble for maritime anomaly detection.

Architecture
------------
Five complementary unsupervised autoencoders, each targeting a different
behavioural slice of the feature space:

  1. MotionAE        – speed / heading / acceleration kinematics
                       → catches AIS spoofing & erratic pursuit
  2. SpatialAE       – stratum proximity + port distance + hull geometry
                       → catches illegal fishing (loitering in restricted zones)
  3. ProximityAE     – pairwise co-location / shadowing features
                       → catches transshipment & aggressive maneuvering
  4. GapAE           – transmission gap / dark-activity features
                       → catches vessels going dark & displacement
  5. FullAE          – all features together (wider net, deeper architecture)
                       → catches cross-group correlation anomalies

Plus an optional LSTM sequence AE that operates on raw per-vessel ping tracks.

Design decisions
----------------
* All autoencoders are trained UNSUPERVISED — truth labels never enter training.
* Per-group AEs avoid one noisy feature group drowning out another.
* Log-transform + RobustScaler tames the extreme range in gap/proximity cols.
* Separate reconstruction errors are fused via weighted max-pool (same strategy
  as the rule-based ensemble) so the strongest sub-signal always wins.
* Threshold-free AUROC is the primary performance metric; a 0.50 default
  threshold is provided for binary flagging but can be tuned post-hoc.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(X: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(X.astype(np.float32)).to(device)


def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + 1e-12)


def _prepare_features(df: pd.DataFrame, cols: List[str],
                       log_cols: Optional[List[str]] = None) -> np.ndarray:
    """
    Fill NaNs, optionally log1p-transform heavy-tailed columns, clip extremes,
    and RobustScale.  Returns float32 numpy array.
    """
    avail = [c for c in cols if c in df.columns]
    X = df[avail].fillna(0.0).to_numpy(dtype=np.float64).copy()

    if log_cols:
        for c in log_cols:
            if c in avail:
                ci = avail.index(c)
                X[:, ci] = np.log1p(np.clip(X[:, ci], 0, None))

    # Clip to [0.5th, 99.5th] percentile to reduce outlier dominance
    lo = np.percentile(X, 0.5, axis=0)
    hi = np.percentile(X, 99.5, axis=0)
    X  = np.clip(X, lo, hi)

    return RobustScaler().fit_transform(X).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

class _FCAutoencoder(nn.Module):
    """Fully-connected symmetric autoencoder with bottleneck."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...],
                 bottleneck: int, dropout: float = 0.10):
        super().__init__()
        enc: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        enc += [nn.Linear(prev, bottleneck), nn.ELU()]
        self.encoder = nn.Sequential(*enc)

        dec: List[nn.Module] = []
        prev = bottleneck
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self(x)) ** 2).mean(dim=1)


class _LSTMAutoencoder(nn.Module):
    """Seq-to-seq LSTM autoencoder (encoder collapses to latent; decoder reconstructs)."""

    def __init__(self, input_dim: int, hidden: int = 64, latent: int = 32,
                 n_layers: int = 2, dropout: float = 0.10):
        super().__init__()
        self.n_layers = n_layers
        self.hidden   = hidden
        self.enc_rnn  = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                                dropout=dropout if n_layers > 1 else 0.0)
        self.lat_proj = nn.Linear(hidden, latent)
        self.lat_exp  = nn.Linear(latent, hidden)
        self.dec_rnn  = nn.LSTM(hidden, hidden, n_layers, batch_first=True,
                                dropout=dropout if n_layers > 1 else 0.0)
        self.out_proj = nn.Linear(hidden, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.enc_rnn(x)
        lat  = torch.tanh(self.lat_proj(h_n[-1]))
        h0   = torch.tanh(self.lat_exp(lat)).unsqueeze(0).repeat(self.n_layers, 1, 1)
        dec_in = h0[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.dec_rnn(dec_in, (h0, torch.zeros_like(h0)))
        return self.out_proj(out)

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self(x)) ** 2).mean(dim=(1, 2))


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _train_fc(model: _FCAutoencoder, X: np.ndarray,
              epochs: int = 80, batch_size: int = 512,
              lr: float = 3e-3, patience: int = 12,
              device: Optional[torch.device] = None,
              verbose: bool = True, label: str = "") -> _FCAutoencoder:
    dev   = device or _device()
    model = model.to(dev)
    Xt    = _to_tensor(X, dev)
    dl    = DataLoader(TensorDataset(Xt), batch_size=batch_size,
                       shuffle=True, drop_last=False)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.MSELoss()

    best_loss, best_sd, patience_cnt = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        sched.step()
        ep_loss /= len(Xt)

        if ep_loss < best_loss - 1e-7:
            best_loss, best_sd, patience_cnt = ep_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"      ep {ep:3d}  loss={ep_loss:.6f}  [{label}]")

    if best_sd:
        model.load_state_dict(best_sd)
    return model.eval()


def _train_lstm(model: _LSTMAutoencoder, seqs: List[np.ndarray],
                epochs: int = 50, batch_size: int = 128,
                lr: float = 1e-3, patience: int = 10,
                device: Optional[torch.device] = None,
                verbose: bool = True) -> _LSTMAutoencoder:
    dev   = device or _device()
    model = model.to(dev)
    F     = seqs[0].shape[1]
    T     = max(len(s) for s in seqs)
    pad   = np.zeros((len(seqs), T, F), dtype=np.float32)
    for i, s in enumerate(seqs):
        pad[i, :len(s)] = s
    Xt    = torch.from_numpy(pad).to(dev)
    dl    = DataLoader(TensorDataset(Xt), batch_size=batch_size, shuffle=True)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.MSELoss()

    best_loss, best_sd, pc = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        sched.step()
        ep_loss /= len(seqs)
        if ep_loss < best_loss - 1e-7:
            best_loss, best_sd, pc = ep_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= patience:
                break
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"      ep {ep:3d}  loss={ep_loss:.6f}  [LSTM-AE]")
    if best_sd:
        model.load_state_dict(best_sd)
    return model.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Feature-group column lists
# ─────────────────────────────────────────────────────────────────────────────

MOTION_COLS = [
    "mean_speed_kt", "median_speed_kt", "p95_speed_kt", "std_speed_kt",
    "frac_slow", "frac_stopped",
    "straightness", "heading_variance", "heading_change_rate",
]
MOTION_LOG = []  # already bounded

SPATIAL_COLS = [
    "frac_inside_stratum", "hours_inside_stratum", "min_stratum_dist_nm",
    "hull_area_deg2", "min_port_dist_nm",
]
SPATIAL_LOG = ["hours_inside_stratum", "hull_area_deg2", "min_port_dist_nm"]

PROXIMITY_COLS = [
    "co_slow_hours", "hours_within_1nm", "hours_within_2nm",
    "sustained_close_episodes", "min_nn_dist_nm", "mean_nn_dist_nm",
]
PROXIMITY_LOG = ["co_slow_hours", "hours_within_1nm", "hours_within_2nm",
                 "sustained_close_episodes"]

GAP_COLS = [
    "max_gap_h", "p95_gap_h", "gap_count", "gap_frac",
    "max_implied_speed_kt", "max_gap_displacement_nm", "n_suspicious_speed_pings",
]
GAP_LOG = ["max_gap_h", "max_gap_displacement_nm", "max_implied_speed_kt"]

ALL_COLS  = MOTION_COLS + SPATIAL_COLS + PROXIMITY_COLS + GAP_COLS
ALL_LOG   = list(set(SPATIAL_LOG + PROXIMITY_LOG + GAP_LOG))

LSTM_COLS = ["speed_kt", "heading_deg", "heading_delta", "dt_h", "dist_km"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-group scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_group(feat: pd.DataFrame, cols: List[str],
                 log_cols: List[str],
                 hidden_dims: Tuple[int, ...], bottleneck: int,
                 label: str, epochs: int,
                 device: torch.device, verbose: bool) -> np.ndarray:
    avail = [c for c in cols if c in feat.columns]
    if len(avail) < 2:
        warnings.warn(f"[{label}] only {len(avail)} cols available — skipping")
        return np.zeros(len(feat), dtype=np.float32)

    X     = _prepare_features(feat, avail, log_cols)
    model = _FCAutoencoder(X.shape[1], hidden_dims, bottleneck)
    model = _train_fc(model, X, epochs=epochs, device=device,
                      label=label, verbose=verbose)
    err   = model.reconstruction_error(_to_tensor(X, device)).cpu().numpy()
    return _minmax_norm(err)


def score_motion_ae(feat: pd.DataFrame, device: torch.device,
                    verbose: bool = True) -> pd.Series:
    s = _score_group(feat, MOTION_COLS, MOTION_LOG,
                     hidden_dims=(64, 32), bottleneck=12,
                     label="MotionAE", epochs=100,
                     device=device, verbose=verbose)
    return pd.Series(s, index=feat.index, name="ae_score_motion")


def score_spatial_ae(feat: pd.DataFrame, device: torch.device,
                     verbose: bool = True) -> pd.Series:
    s = _score_group(feat, SPATIAL_COLS, SPATIAL_LOG,
                     hidden_dims=(32, 16), bottleneck=6,
                     label="SpatialAE", epochs=90,
                     device=device, verbose=verbose)
    return pd.Series(s, index=feat.index, name="ae_score_spatial")


def score_proximity_ae(feat: pd.DataFrame, device: torch.device,
                       verbose: bool = True) -> pd.Series:
    s = _score_group(feat, PROXIMITY_COLS, PROXIMITY_LOG,
                     hidden_dims=(48, 24), bottleneck=8,
                     label="ProximityAE", epochs=90,
                     device=device, verbose=verbose)
    return pd.Series(s, index=feat.index, name="ae_score_proximity")


def score_gap_ae(feat: pd.DataFrame, device: torch.device,
                 verbose: bool = True) -> pd.Series:
    """Gap AE — the signal is extremely clean (normals have max_gap_h ~0.08 h,
    anomalies 25–96 h), so log-transform + small net is enough."""
    s = _score_group(feat, GAP_COLS, GAP_LOG,
                     hidden_dims=(48, 24), bottleneck=8,
                     label="GapAE", epochs=90,
                     device=device, verbose=verbose)
    return pd.Series(s, index=feat.index, name="ae_score_gap")


def score_full_ae(feat: pd.DataFrame, device: torch.device,
                  verbose: bool = True) -> pd.Series:
    """Wide AE across all features — captures cross-group correlations."""
    avail = [c for c in ALL_COLS if c in feat.columns]
    X     = _prepare_features(feat, avail, ALL_LOG)
    bn    = max(10, X.shape[1] // 5)
    model = _FCAutoencoder(X.shape[1], hidden_dims=(128, 64, 32),
                           bottleneck=bn, dropout=0.12)
    model = _train_fc(model, X, epochs=120, lr=2e-3,
                      device=device, label="FullAE", verbose=verbose)
    err   = model.reconstruction_error(_to_tensor(X, device)).cpu().numpy()
    return pd.Series(_minmax_norm(err), index=feat.index, name="ae_score_full")


def score_lstm_ae(raw_df: pd.DataFrame, index: pd.Index,
                  device: torch.device,
                  window: int = 48,
                  stride: int = 24,
                  verbose: bool = True) -> pd.Series:
    """
    LSTM AE using a sliding-window approach that scales to 50k+ vessels.

    Instead of padding full tracks (O(N × T) memory), we slice each vessel's
    track into fixed-length windows of `window` pings with `stride` overlap,
    train on all windows jointly, and aggregate per-vessel by max reconstruction
    error.  At 5-min cadence, window=48 covers 4 hours — enough to capture
    episode-length anomalies (transshipment min 3 h, fishing min 6 h).

    Parameters
    ----------
    raw_df   Raw AIS dataframe (must have speed_kt, heading_deg, dt_h, dist_km).
    index    Target entity_id index to align output to feat.
    device   Torch device.
    window   Number of consecutive pings per training window.
    stride   Slide step (overlap = window - stride).
    verbose  Print training progress.
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

    # Single global scaler on all pings
    all_vals = raw_df[avail].fillna(0.0).to_numpy(dtype=np.float32)
    lo = np.percentile(all_vals, 0.5, axis=0)
    hi = np.percentile(all_vals, 99.5, axis=0)
    all_vals = np.clip(all_vals, lo, hi)
    scaler   = RobustScaler().fit(all_vals)

    # Build sliding windows and track which vessel each belongs to
    win_list: List[np.ndarray] = []   # each is (window, F)
    win_eid:  List[int]        = []   # entity_id for each window

    for eid, grp in raw_df.groupby("entity_id"):
        v = scaler.transform(
            np.clip(grp.sort_values("timestamp")[avail].fillna(0.0)
                    .to_numpy(dtype=np.float32), lo, hi))
        T = len(v)
        if T < window:
            # Short track: zero-pad to window length
            pad = np.zeros((window, F), dtype=np.float32)
            pad[:T] = v
            win_list.append(pad)
            win_eid.append(eid)
        else:
            for start in range(0, T - window + 1, stride):
                win_list.append(v[start:start + window])
                win_eid.append(eid)

    # Stack all windows: (N_windows, window, F)
    X_win = np.stack(win_list, axis=0)                # (N, T, F)
    Xt    = torch.from_numpy(X_win).to(device)
    ds    = TensorDataset(Xt)
    dl    = DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)

    model = _LSTMAutoencoder(F, hidden=48, latent=20, n_layers=2)
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    crit  = nn.MSELoss()

    best_loss, best_sd, pc = np.inf, None, 0
    for ep in range(40):
        model.train()
        ep_loss = 0.0
        for (xb,) in dl:
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            ep_loss += loss.item() * len(xb)
        sched.step()
        ep_loss /= len(X_win)
        if ep_loss < best_loss - 1e-7:
            best_loss, best_sd, pc = ep_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= 8:
                break
        if verbose and (ep % 10 == 0 or ep == 39):
            print(f"      ep {ep:3d}  loss={ep_loss:.6f}  [LSTM-AE]")
    if best_sd:
        model.load_state_dict(best_sd)
    model.eval()

    # Score all windows, then aggregate per vessel by max error
    with torch.no_grad():
        errs = model.reconstruction_error(Xt).cpu().numpy()  # (N_windows,)

    # Max error per vessel (worst window = most anomalous episode)
    win_eid_arr = np.array(win_eid)
    vessel_max: Dict[int, float] = {}
    for eid in np.unique(win_eid_arr):
        mask = win_eid_arr == eid
        vessel_max[eid] = float(errs[mask].max())

    eids_out   = list(vessel_max.keys())
    scores_out = np.array([vessel_max[e] for e in eids_out], dtype=np.float32)

    return pd.Series(
        _minmax_norm(scores_out),
        index=pd.Index(eids_out, name="entity_id"),
        name="ae_score_lstm",
    ).reindex(index).fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble fusion
# ─────────────────────────────────────────────────────────────────────────────

# Weights: reflect expected signal strength per anomaly category
AE_WEIGHTS: Dict[str, float] = {
    "ae_score_motion":    1.10,   # spoofing, erratic pursuit
    "ae_score_spatial":   1.00,   # illegal fishing
    "ae_score_proximity": 1.10,   # transshipment, shadowing
    "ae_score_gap":       1.20,   # dark activity (cleanest univariate signal)
    "ae_score_full":      1.05,   # cross-group
    "ae_score_lstm":      1.00,   # temporal
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
              (truth columns are stripped automatically before training).
    raw_df    Raw AIS DataFrame — required only when use_lstm=True.
    use_lstm  Whether to include the LSTM sequence AE.
    device    Torch device (defaults to CUDA if available, else CPU).
    verbose   Print per-epoch training loss.

    Returns
    -------
    DataFrame indexed by entity_id with:
      ae_score_motion, ae_score_spatial, ae_score_proximity,
      ae_score_gap, ae_score_full, [ae_score_lstm],
      ensemble_score, ensemble_mean, [true_anomalous, true_type]
    """
    dev = device or _device()

    # Strip truth so models never see labels
    truth_cols = [c for c in ["true_anomalous", "true_type"] if c in feat.columns]
    feat_only  = feat.drop(columns=truth_cols, errors="ignore")

    print("  [1/5] AE: motion kinematics  →  spoofing, erratic pursuit")
    s1 = score_motion_ae(feat_only, dev, verbose)

    print("  [2/5] AE: spatial / stratum proximity  →  illegal fishing")
    s2 = score_spatial_ae(feat_only, dev, verbose)

    print("  [3/5] AE: inter-vessel proximity  →  transshipment, shadowing")
    s3 = score_proximity_ae(feat_only, dev, verbose)

    print("  [4/5] AE: transmission gap / blackout  →  dark activity")
    s4 = score_gap_ae(feat_only, dev, verbose)

    print("  [5/5] AE: full feature vector  →  cross-group correlations")
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
            print("  [6/6] AE: LSTM temporal sequence  →  episode-level patterns")
            s6 = score_lstm_ae(raw_df, feat_only.index, dev, verbose=verbose)
            scores["ae_score_lstm"] = s6
        else:
            warnings.warn("use_lstm=True but raw_df not provided — skipping LSTM AE")

    scores_df = pd.DataFrame(scores)

    # Weighted max-pool (same fusion strategy as rule-based pipeline)
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
    Full AE detection pipeline.  Returns scored DataFrame with a `flagged` column.

    Parameters
    ----------
    feat       Feature matrix (index = entity_id).  Truth columns are passed
               through for evaluation but never used in training.
    raw_df     Raw AIS pings — only needed when use_lstm=True.
    use_lstm   Include LSTM sequence AE (recommended; needs raw_df).
    threshold  Ensemble score cutoff for binary flagging.
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