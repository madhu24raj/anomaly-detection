"""Vessel fleet definition and normal-trajectory generation."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import geography as geo

# Vessel type -> (population share, (min_speed_kn, max_speed_kn)).
# Shares are relative and normalized at fleet-build time.
VESSEL_TYPES = {
    "cargo":     (0.28, (12.0, 18.0)),
    "tanker":    (0.15, (10.0, 15.0)),
    "fishing":   (0.30, (4.0, 9.0)),
    "passenger": (0.07, (15.0, 22.0)),
    "tug":       (0.08, (6.0, 10.0)),
    "pleasure":  (0.12, (5.0, 12.0)),
}


def assign_vessel_types(rng: np.random.Generator, n: int) -> np.ndarray:
    names = np.array(list(VESSEL_TYPES.keys()))
    shares = np.array([VESSEL_TYPES[k][0] for k in names], dtype=float)
    shares /= shares.sum()
    return rng.choice(names, size=n, p=shares)


def cruise_speed(rng: np.random.Generator, vessel_type: str) -> float:
    lo, hi = VESSEL_TYPES[vessel_type][1]
    return float(rng.uniform(lo, hi))


def _port_lonlat(rng: np.random.Generator) -> Tuple[float, float]:
    _, lon, lat = geo.PORTS[rng.integers(len(geo.PORTS))]
    return lon, lat


def build_route(
    rng: np.random.Generator,
    vessel_type: str,
    bbox,
    center: Optional[Tuple[float, float]] = None,
    local_radius_km: float = 30.0,
    avoid_land: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a meandering, land-avoiding patrol route (waypoint polyline).

    Routes are generated as a correlated random walk between a water-borne
    origin and destination (see ``geo.correlated_walk_route``), so they curve
    organically and steer around land instead of running in straight lines
    across islands.

    If ``center`` (lon, lat) is given, the route is confined near it -- used to
    localize vessels in paired anomalies so a mid-track rendezvous needs no
    large, spoofing-like jump.
    """
    if center is not None:
        origin = geo.sample_water_point(rng, bbox, near=center, radius_km=local_radius_km)
        dest = geo.sample_water_point(rng, bbox, near=center, radius_km=local_radius_km)
        return geo.correlated_walk_route(rng, origin, dest, step_km=4.0,
                                         turn_std_deg=18.0, avoid_land=avoid_land)

    if vessel_type in ("cargo", "tanker", "passenger"):
        # Liner traffic: water lane between two different ports.
        origin = geo.sample_water_point(rng, bbox, near=_port_lonlat(rng))
        dest = geo.sample_water_point(rng, bbox, near=_port_lonlat(rng))
        return geo.correlated_walk_route(rng, origin, dest, step_km=18.0,
                                         turn_std_deg=10.0, avoid_land=avoid_land)

    if vessel_type == "fishing":
        # Home port -> offshore grounds, then a slow wandering loiter over the
        # grounds. Both legs are land-avoiding validated walks (the loiter uses
        # short, twisty steps to mimic working the grounds without crossing land).
        origin = geo.sample_water_point(rng, bbox, near=_port_lonlat(rng))
        ground = geo.sample_water_point(rng, bbox)
        lat, lon = geo.correlated_walk_route(rng, origin, ground, step_km=9.0,
                                             turn_std_deg=16.0, avoid_land=avoid_land)
        end = (float(lon[-1]), float(lat[-1]))  # where the transit actually ended
        loiter_target = geo.sample_water_point(rng, bbox, near=end, radius_km=25.0)
        llat, llon = geo.correlated_walk_route(rng, end, loiter_target, step_km=3.0,
                                               turn_std_deg=32.0, dest_pull=0.08,
                                               avoid_land=avoid_land)
        return np.concatenate([lat, llat[1:]]), np.concatenate([lon, llon[1:]])

    # tug / pleasure: short coastal hops near a single port.
    home = _port_lonlat(rng)
    origin = geo.sample_water_point(rng, bbox, near=home, radius_km=20.0)
    dest = geo.sample_water_point(rng, bbox, near=home, radius_km=20.0)
    return geo.correlated_walk_route(rng, origin, dest, step_km=4.0,
                                     turn_std_deg=20.0, avoid_land=avoid_land)


def _smooth_speed_profile(
    rng: np.random.Generator, cruise_kn: float, n_steps: int
) -> np.ndarray:
    """A smoothly varying speed (knots) around the vessel's cruise speed.

    White noise is low-pass filtered (moving average) so the vessel speeds up
    and slows down gradually instead of jittering -- simple 'vessel dynamics'.
    """
    raw = rng.standard_normal(n_steps)
    window = max(3, n_steps // 360)  # ~ a few hours of smoothing at 5-min steps
    kernel = np.ones(window) / window
    smooth = np.convolve(raw, kernel, mode="same")
    if smooth.std() > 0:
        smooth = smooth / smooth.std()
    speed = cruise_kn * (1.0 + 0.30 * smooth)
    return np.clip(speed, 0.25 * cruise_kn, 1.6 * cruise_kn)


def generate_track(
    rng: np.random.Generator,
    route_lat: np.ndarray,
    route_lon: np.ndarray,
    speed_kn: float,
    n_steps: int,
    interval_hours: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a normal (lat, lon) track of length ``n_steps``.

    Distance per step follows a smooth speed profile, with a random phase along
    the (folded) route so vessels are not synchronized.
    """
    speed = _smooth_speed_profile(rng, speed_kn, n_steps)
    step_km = speed * geo.KM_PER_NM * interval_hours
    phase = rng.uniform(0.0, max(speed_kn * geo.KM_PER_NM * interval_hours * n_steps, 1.0))
    arc_km = phase + np.cumsum(step_km)
    lat, lon = geo.positions_along_route(route_lat, route_lon, arc_km)
    return lat, lon
