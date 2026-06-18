"""Anomaly episode planning and injection.

Two kinds of anomaly:

* **solo**  -- modifies a single vessel's track in place
              (illegal_fishing, ais_spoofing, dark_activity)
* **paired** -- needs two vessels generated together so they reach genuine
              close-range proximity (transshipment, aggressive_maneuvering)

Each injector mutates the position arrays for the episode window ``[s, e)`` and
returns the set of row indices to label. ``dark_activity`` additionally returns
indices to *drop* (the vessel stops transmitting during the gap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import geography as geo
from .config import Config, PAIRED_ANOMALIES


@dataclass
class SoloEpisode:
    atype: str
    start: int
    end: int


@dataclass
class PairEpisode:
    atype: str
    start: int
    end: int


@dataclass
class AnomalyPlan:
    # entity_id -> list of solo episodes
    solo: Dict[int, List[SoloEpisode]]
    # (entity_a, entity_b) -> pair episode  (a is the "anchor"/victim)
    pairs: List[Tuple[int, int, PairEpisode]]
    # every entity id that is anomalous (for quick membership checks)
    anomalous_ids: set

    def n_anomalous(self) -> int:
        return len(self.anomalous_ids)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def _episode_len_steps(rng, cfg: Config, atype: str) -> int:
    lo, hi = cfg.episode_hours[atype]
    hours = rng.uniform(lo, hi)
    steps = int(round(hours / cfg.report_interval_hours))
    return max(2, min(steps, cfg.n_steps - 1))


def _random_window(rng, cfg: Config, length: int) -> Tuple[int, int]:
    start = int(rng.integers(0, max(1, cfg.n_steps - length)))
    return start, start + length


def plan_anomalies(rng: np.random.Generator, cfg: Config) -> AnomalyPlan:
    """Decide which entities are anomalous, of which type, and when."""
    n_anom = int(round(cfg.anomaly_rate * cfg.num_entities))
    n_anom = min(n_anom, cfg.num_entities)

    weights = cfg.normalized_weights()
    # Pool of entity ids that will be anomalous, drawn without replacement.
    pool = list(rng.choice(cfg.num_entities, size=n_anom, replace=False)) if n_anom else []
    rng.shuffle(pool)
    cursor = 0

    solo: Dict[int, List[SoloEpisode]] = {}
    pairs: List[Tuple[int, int, PairEpisode]] = []
    anomalous_ids: set = set()

    for atype in weights:
        share = weights[atype]
        n_for_type = int(round(share * n_anom))
        if n_for_type <= 0:
            continue

        if atype in PAIRED_ANOMALIES:
            n_pairs = n_for_type // 2
            for _ in range(n_pairs):
                if cursor + 2 > len(pool):
                    break
                a_id, b_id = int(pool[cursor]), int(pool[cursor + 1])
                cursor += 2
                length = _episode_len_steps(rng, cfg, atype)
                s, e = _random_window(rng, cfg, length)
                pairs.append((a_id, b_id, PairEpisode(atype, s, e)))
                anomalous_ids.update((a_id, b_id))
        else:
            for _ in range(n_for_type):
                if cursor >= len(pool):
                    break
                ent = int(pool[cursor])
                cursor += 1
                episodes: List[SoloEpisode] = []
                n_ep = int(rng.integers(cfg.episodes_per_entity[0],
                                        cfg.episodes_per_entity[1] + 1))
                for _ in range(n_ep):
                    length = _episode_len_steps(rng, cfg, atype)
                    s, e = _random_window(rng, cfg, length)
                    episodes.append(SoloEpisode(atype, s, e))
                _resolve_overlaps(episodes)
                solo[ent] = episodes
                anomalous_ids.add(ent)

    return AnomalyPlan(solo=solo, pairs=pairs, anomalous_ids=anomalous_ids)


def _resolve_overlaps(episodes: List[SoloEpisode]) -> None:
    """Drop later episodes that overlap an earlier one (keep it simple)."""
    episodes.sort(key=lambda ep: ep.start)
    kept: List[SoloEpisode] = []
    last_end = -1
    for ep in episodes:
        if ep.start > last_end:
            kept.append(ep)
            last_end = ep.end
    episodes[:] = kept


# --------------------------------------------------------------------------- #
# Small geometry helpers
# --------------------------------------------------------------------------- #
def _km_offsets(rng, n, max_km, lat_ref):
    """n random (dlat_deg, dlon_deg) offsets, each within ``max_km``."""
    ang = rng.uniform(0, 2 * np.pi, size=n)
    rad = max_km * np.sqrt(rng.uniform(0, 1, size=n))
    dx_km = rad * np.cos(ang)
    dy_km = rad * np.sin(ang)
    dlat = geo.km_to_deg_lat(dy_km)
    dlon = geo.km_to_deg_lon(dx_km, lat_ref)
    return dlat, dlon


def _random_walk(rng, n, step_km, lat0, lon0):
    """Slow bounded random walk producing low speed-over-ground."""
    ang = rng.uniform(0, 2 * np.pi, size=n)
    dx = step_km * np.cos(ang)
    dy = step_km * np.sin(ang)
    lat = lat0 + np.cumsum(geo.km_to_deg_lat(dy))
    lon = lon0 + np.cumsum(geo.km_to_deg_lon(dx, lat0))
    return lat, lon


# --------------------------------------------------------------------------- #
# Solo injectors -- each returns label_idx (indices to mark anomalous)
# --------------------------------------------------------------------------- #
def inject_spoofing(rng, cfg: Config, lat, lon, ep: SoloEpisode):
    """Teleport to a false location and hold there, then jump back.

    Produces an impossible implied speed at both the start and end of the
    window -- the canonical AIS-spoofing signature.
    """
    s, e = ep.start, ep.end
    # A false position over water, far from the real one (>~300 km).
    flon, flat = lon[s], lat[s]
    for _ in range(10):
        flon, flat = geo.sample_water_point(rng, cfg.region_bbox)
        if geo.haversine_km(lat[s], lon[s], flat, flon) > 300.0:
            break
    n = e - s
    # Slow drift around the false point so it is not suspiciously frozen.
    dlat, dlon = _km_offsets(rng, n, 2.0, flat)
    lat[s:e] = flat + dlat
    lon[s:e] = flon + dlon
    return np.arange(s, e)


def inject_illegal_fishing(rng, cfg: Config, lat, lon, ep: SoloEpisode):
    """Slow loitering inside the nearest restricted zone."""
    s, e = ep.start, ep.end
    # Nearest restricted zone to where the vessel is when the episode starts.
    zlat = np.array([z[2] for z in geo.RESTRICTED_ZONES])
    zlon = np.array([z[1] for z in geo.RESTRICTED_ZONES])
    d = geo.haversine_km(lat[s], lon[s], zlat, zlon)
    zi = int(np.argmin(d))
    _, czlon, czlat, zr = geo.RESTRICTED_ZONES[zi]
    # Start somewhere inside the zone, then crawl around at ~1-3 kn.
    dlat0, dlon0 = _km_offsets(rng, 1, zr * 0.6, czlat)
    lat0 = czlat + float(dlat0[0])
    lon0 = czlon + float(dlon0[0])
    n = e - s
    step_km = 1.5 * geo.KM_PER_NM * cfg.report_interval_hours  # ~1.5 kn
    wlat, wlon = _random_walk(rng, n, step_km, lat0, lon0)
    lat[s:e] = wlat
    lon[s:e] = wlon
    return np.arange(s, e)


def inject_dark_activity(rng, cfg: Config, lat, lon, ep: SoloEpisode):
    """Vessel goes dark: drop reports during the window.

    The track itself continues underneath (the vessel keeps moving), so when it
    reappears it is displaced -- a long time gap between consecutive reports.
    We label the report just before the gap and the first report after it so
    the gap is bracketed by ground truth.
    """
    s, e = ep.start, ep.end
    drop = np.arange(s, e)
    label = []
    if s - 1 >= 0:
        label.append(s - 1)
    if e < len(lat):
        label.append(e)
    return drop, np.array(label, dtype=int)


# --------------------------------------------------------------------------- #
# Paired injectors -- modify the *partner* (b) relative to the anchor (a)
# --------------------------------------------------------------------------- #
def inject_transshipment(rng, cfg: Config, lat_a, lon_a, lat_b, lon_b, ep: PairEpisode):
    """Two vessels rendezvous at sea: <~300 m apart, both nearly stopped."""
    s, e = ep.start, ep.end
    n = e - s
    # Rendezvous point: midpoint of the two vessels at episode start, drifting
    # slowly (a vessel pair adrift while transferring catch/cargo).
    rp_lat0 = 0.5 * (lat_a[s] + lat_b[s])
    rp_lon0 = 0.5 * (lon_a[s] + lon_b[s])
    drift_km = 0.3 * geo.KM_PER_NM * cfg.report_interval_hours  # ~0.3 kn drift
    rp_lat, rp_lon = _random_walk(rng, n, drift_km, rp_lat0, rp_lon0)
    # Each vessel sits within ~150 m of the drifting rendezvous point.
    da_lat, da_lon = _km_offsets(rng, n, 0.15, rp_lat0)
    db_lat, db_lon = _km_offsets(rng, n, 0.15, rp_lat0)
    lat_a[s:e] = rp_lat + da_lat
    lon_a[s:e] = rp_lon + da_lon
    lat_b[s:e] = rp_lat + db_lat
    lon_b[s:e] = rp_lon + db_lon
    return np.arange(s, e)  # both vessels are labeled


def inject_aggressive(rng, cfg: Config, lat_a, lon_a, lat_b, lon_b, ep: PairEpisode):
    """Aggressor (b) shadows the victim (a) at close range with erratic,
    high-rate maneuvering and occasional close 'passes'."""
    s, e = ep.start, ep.end
    n = e - s
    # Shadow within ~0.2-1.0 nm, jittering every step (erratic close maneuvers).
    base_km = rng.uniform(0.4, 1.8)  # nominal standoff in km (~0.2-1 nm)
    jlat, jlon = _km_offsets(rng, n, 0.4, lat_a[s])
    follow_lat = lat_a[s:e] + geo.km_to_deg_lat(base_km) * np.sin(np.arange(n) * 0.7)
    follow_lon = lon_a[s:e] + geo.km_to_deg_lon(base_km, lat_a[s]) * np.cos(np.arange(n) * 0.9)
    lat_b[s:e] = follow_lat + jlat
    lon_b[s:e] = follow_lon + jlon
    # A few aggressive "closing" passes: nearly coincide with the victim.
    if n >= 4:
        n_close = max(1, n // 20)
        idx = rng.choice(np.arange(s, e), size=min(n_close, n), replace=False)
        lat_b[idx] = lat_a[idx] + geo.km_to_deg_lat(0.03)  # ~30 m
        lon_b[idx] = lon_a[idx]
    # Only the aggressor is labeled anomalous; the victim's track is normal.
    return np.arange(s, e)
