"""Geography helpers: region, ports, restricted zones, distance, and routing.

All routing uses simple equirectangular / linear interpolation in (lat, lon).
At Caribbean scale this is more than accurate enough for a *simulator* whose
purpose is to produce realistic-looking, detectable behavior -- not to be a
navigational tool.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

EARTH_RADIUS_KM = 6371.0088
KM_PER_NM = 1.852  # 1 knot == 1.852 km/h

# Optional coarse land/sea mask. `global-land-mask` bundles a global grid, so we
# can ask "is this point water?" without shipping any coastline geometry. If it
# is not installed, land avoidance is silently disabled (everything is treated
# as water) and the simulator still runs.
try:
    from global_land_mask import globe as _globe
    _HAS_LAND_MASK = True
except Exception:  # pragma: no cover - optional dependency
    _globe = None
    _HAS_LAND_MASK = False


def has_land_mask() -> bool:
    return _HAS_LAND_MASK


def is_water(lat, lon):
    """Vectorized water test. True == water. All-True if mask unavailable."""
    if not _HAS_LAND_MASK:
        shape = np.shape(np.asarray(lat))
        return np.ones(shape, dtype=bool) if shape else True
    return ~_globe.is_land(lat, lon)


# --------------------------------------------------------------------------- #
# Reference geography (lon, lat). A representative set of Caribbean ports.
# --------------------------------------------------------------------------- #
PORTS: List[Tuple[str, float, float]] = [
    ("Miami",          -80.19, 25.77),
    ("Havana",         -82.38, 23.13),
    ("Nassau",         -77.34, 25.06),
    ("Kingston",       -76.79, 17.99),
    ("Santo Domingo",  -69.93, 18.47),
    ("San Juan",       -66.10, 18.47),
    ("Willemstad",     -68.93, 12.11),
    ("Port of Spain",  -61.52, 10.65),
    ("Bridgetown",     -59.62, 13.10),
    ("Cartagena",      -75.51, 10.40),
    ("Colon",          -79.90,  9.36),
    ("Cozumel",        -86.85, 21.16),
]

# Restricted / sensitive zones (name, center_lon, center_lat, radius_km).
# Used as targets for "illegal fishing" loitering. These are illustrative
# locations, not official boundaries.
RESTRICTED_ZONES: List[Tuple[str, float, float, float]] = [
    ("Cay Sal Bank MPA",        -80.50, 23.60, 45.0),
    ("Miskito Bank MPA",        -82.50, 15.50, 60.0),
    ("Venezuela EEZ edge",      -66.50, 12.50, 70.0),
    ("Pedro Bank fishery",      -78.20, 17.10, 40.0),
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts scalars or numpy arrays."""
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def km_to_deg_lat(km: float) -> float:
    return km / 110.574


def km_to_deg_lon(km: float, lat: float) -> float:
    denom = 111.320 * np.cos(np.radians(lat))
    return km / np.where(np.abs(denom) < 1e-6, 1e-6, denom)


def positions_along_route(
    waypoints_lat: np.ndarray,
    waypoints_lon: np.ndarray,
    arc_km: np.ndarray,
):
    """Interpolate positions along a polyline, given arc-length traveled.

    The vessel "ping-pongs" back and forth along the polyline (a triangle wave
    over arc length) so it patrols its route indefinitely without ever running
    off the end -- a cargo ship shuttles port A <-> port B, a fishing boat goes
    out to the grounds and back, etc.

    Parameters
    ----------
    waypoints_lat, waypoints_lon : (K,) arrays of the polyline vertices.
    arc_km : (N,) cumulative distance traveled at each timestep.

    Returns
    -------
    (lat, lon) : two (N,) arrays of interpolated positions.
    """
    seg_km = haversine_km(
        waypoints_lat[:-1], waypoints_lon[:-1],
        waypoints_lat[1:], waypoints_lon[1:],
    )
    cum = np.concatenate([[0.0], np.cumsum(seg_km)])
    total = cum[-1]
    if total <= 1e-9:
        # Degenerate route (all waypoints coincident): stay put.
        return (
            np.full(arc_km.shape, waypoints_lat[0]),
            np.full(arc_km.shape, waypoints_lon[0]),
        )

    # Triangle wave: fold arc length into [0, total] reflecting at the ends.
    period = 2.0 * total
    folded = np.mod(arc_km, period)
    eff = np.where(folded <= total, folded, period - folded)

    lat = np.interp(eff, cum, waypoints_lat)
    lon = np.interp(eff, cum, waypoints_lon)
    return lat, lon


def random_point_in_bbox(rng: np.random.Generator, bbox) -> Tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = rng.uniform(min_lon, max_lon)
    lat = rng.uniform(min_lat, max_lat)
    return lon, lat


# --------------------------------------------------------------------------- #
# Water sampling + land-aware route building
# --------------------------------------------------------------------------- #
def sample_water_point(
    rng: np.random.Generator,
    bbox,
    near: Optional[Tuple[float, float]] = None,
    radius_km: float = 25.0,
    max_tries: int = 300,
) -> Tuple[float, float]:
    """Rejection-sample a (lon, lat) that is over water.

    If ``near`` (lon, lat) is given, samples within ``radius_km`` of it;
    otherwise samples the whole bbox. Falls back to the last candidate if no
    water point is found (or if the land mask is unavailable).
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = lat = 0.0
    for _ in range(max_tries):
        if near is not None:
            nlon, nlat = near
            rlat = km_to_deg_lat(radius_km)
            rlon = float(km_to_deg_lon(radius_km, nlat))
            lon = float(np.clip(nlon + rng.uniform(-rlon, rlon), min_lon, max_lon))
            lat = float(np.clip(nlat + rng.uniform(-rlat, rlat), min_lat, max_lat))
        else:
            lon = rng.uniform(min_lon, max_lon)
            lat = rng.uniform(min_lat, max_lat)
        if not _HAS_LAND_MASK or is_water(lat, lon):
            return lon, lat
    return lon, lat


def _bearing_rad(lat, lon, tlat, tlon):
    """Bearing (radians, 0 = north, clockwise) from a point toward a target."""
    north = (tlat - lat) * 110.574
    east = (tlon - lon) * 111.320 * np.cos(np.radians(lat))
    return np.arctan2(east, north)


def _advance(lat, lon, heading, step_km):
    """Step ``step_km`` along ``heading``. Works for scalar or array heading."""
    north_km = step_km * np.cos(heading)
    east_km = step_km * np.sin(heading)
    nlat = lat + km_to_deg_lat(north_km)
    nlon = lon + km_to_deg_lon(east_km, lat)
    return nlat, nlon


# Candidate heading offsets (radians), tried in order, when the straight-ahead
# step would hit land -- the vessel turns to find open water (coast-following).
_STEER_OFFSETS = np.radians(
    [0, 25, -25, 50, -50, 80, -80, 115, -115, 150, -150, 180]
)


def _segment_water(lat, lon, tlat, tlon, sample_km: float = 6.0) -> bool:
    """True if the straight segment (lat,lon)->(tlat,tlon) stays over water.

    Uses the same linear interpolation that ``positions_along_route`` renders,
    so this exactly reflects the path a vessel will trace between waypoints --
    catching land that lies *between* two water endpoints (capes, isthmuses,
    or a whole island the straight line would cut across).
    """
    if not _HAS_LAND_MASK:
        return True
    d = haversine_km(lat, lon, tlat, tlon)
    k = max(2, int(d / sample_km) + 1)
    f = np.linspace(0.0, 1.0, k)[1:]  # skip the origin (already water)
    return bool(is_water(lat + (tlat - lat) * f, lon + (tlon - lon) * f).all())


def correlated_walk_route(
    rng: np.random.Generator,
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    step_km: float,
    turn_std_deg: float = 12.0,
    dest_pull: float = 0.18,
    max_waypoints: Optional[int] = None,
    avoid_land: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a meandering, land-avoiding route from ``origin`` toward ``dest``.

    The heading is *correlated* across steps (a gentle random turn each step)
    with a soft pull toward the destination, which produces organic curves
    rather than straight lines. At each step the candidate hop is accepted only
    if the whole *segment* stays over water, so vessels steer around islands and
    coastlines instead of cutting across them.

    The waypoint budget scales with the origin->destination distance (so the
    walk can actually round large landmasses), and the destination is only
    connected if that final segment is over water -- preventing the
    "straight line across the island" artifact when the walk falls short.

    Returns (lat[], lon[]) waypoints.
    """
    olon, olat = origin
    dlon, dlat = dest
    lons = [olon]
    lats = [olat]
    lat, lon = olat, olon
    heading = float(_bearing_rad(lat, lon, dlat, dlon))
    turn_std = np.radians(turn_std_deg)
    use_mask = avoid_land and _HAS_LAND_MASK

    if max_waypoints is None:
        # Generous budget: ~2.5x the straight-line hop count, room for detours.
        straight = haversine_km(olat, olon, dlat, dlon) / step_km
        max_waypoints = int(np.clip(straight * 2.5 + 10, 10, 600))

    # Fraction points used to test each candidate hop segment for water.
    k_seg = max(2, int(step_km / 6.0) + 1)
    seg_f = np.linspace(0.0, 1.0, k_seg)[1:]

    for _ in range(max_waypoints):
        if haversine_km(lat, lon, dlat, dlon) <= step_km:
            break
        # Correlated heading: pull toward destination + random turn.
        to_dest = float(_bearing_rad(lat, lon, dlat, dlon))
        diff = (to_dest - heading + np.pi) % (2 * np.pi) - np.pi
        heading = heading + dest_pull * diff + rng.normal(0.0, turn_std)

        if use_mask:
            cand_h = heading + _STEER_OFFSETS
            end_lat, end_lon = _advance(lat, lon, cand_h, step_km)
            # Test the full hop segment (origin->endpoint) for each candidate.
            seg_lat = lat + (end_lat[:, None] - lat) * seg_f[None, :]
            seg_lon = lon + (end_lon[:, None] - lon) * seg_f[None, :]
            water_frac = is_water(seg_lat, seg_lon).mean(axis=1)
            clear = water_frac >= 1.0
            if not clear.any():
                # Boxed in by land: adopt the least-bad heading but do NOT step
                # onto land. Stay put; next step's random turn probes elsewhere.
                heading = float(cand_h[int(np.argmax(water_frac))])
                continue
            j = int(np.argmax(clear))  # first (most-forward) fully-clear heading
            heading = float(cand_h[j])
            lat, lon = float(end_lat[j]), float(end_lon[j])
        else:
            nlat, nlon = _advance(lat, lon, heading, step_km)
            lat, lon = float(nlat), float(nlon)

        lats.append(lat)
        lons.append(lon)

    # Connect to the destination only if the final leg stays over water.
    if not use_mask or _segment_water(lat, lon, dlat, dlon):
        lats.append(dlat)
        lons.append(dlon)
    return np.asarray(lats), np.asarray(lons)
