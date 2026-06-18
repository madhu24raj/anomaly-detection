"""Orchestration: build the fleet, inject anomalies, stream output to disk.

Memory stays bounded regardless of total output size: each vessel's full track
is generated independently (paired vessels two at a time) and handed to a
streaming writer that flushes to disk incrementally.

Two output writers share one ``emit(...)`` interface:
    * CsvChunkWriter    -- one row per position report (default)
    * TripGeoJsonWriter -- one LineString feature per contiguous-anomaly-state
                           segment, for the kepler.gl Trip layer
"""

from __future__ import annotations

import json
import time as _time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import anomalies as anom
from . import geography as geo
from . import vessels as ves
from .config import Config


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
class CsvChunkWriter:
    """Buffers per-vessel arrays and flushes to CSV in chunks (header once).

    A ``.gz`` output path is gzip-compressed automatically (pandas infers it).
    """

    def __init__(self, cfg: Config, path: Optional[str] = None):
        self.cfg = cfg
        self.path = path or cfg.output_path
        self.columns = _csv_columns(cfg)
        self._buf: Dict[str, list] = {c: [] for c in self.columns}
        self._n = 0
        self._wrote_header = False
        self.total_rows = 0
        self.total_features = 0  # unused for CSV

    def emit(self, entity_id, vessel_type, lat, lon, times, sog, is_anom, atype):
        row = {
            "entity_id": np.full(len(lat), entity_id, dtype=np.int64),
            "lat": np.round(lat, self.cfg.coord_decimals),
            "lon": np.round(lon, self.cfg.coord_decimals),
            "timestamp": times,
        }
        if self.cfg.include_vessel_type:
            row["vessel_type"] = np.full(len(lat), vessel_type)
        if self.cfg.include_speed and sog is not None:
            row["sog_knots"] = np.round(sog, 2)
        if self.cfg.include_label:
            row["is_anomalous"] = is_anom.astype(np.int8)
        if self.cfg.include_anomaly_type:
            row["anomaly_type"] = atype
        for c in self.columns:
            self._buf[c].append(row[c])
        self._n += len(lat)
        if self._n >= self.cfg.flush_rows:
            self._flush()

    def _flush(self):
        if self._n == 0:
            return
        data = {c: np.concatenate(self._buf[c]) for c in self.columns}
        df = pd.DataFrame(data, columns=self.columns)
        mode = "w" if not self._wrote_header else "a"
        df.to_csv(self.path, mode=mode, header=not self._wrote_header, index=False)
        self._wrote_header = True
        self.total_rows += self._n
        self._buf = {c: [] for c in self.columns}
        self._n = 0

    def close(self):
        self._flush()


class TripGeoJsonWriter:
    """Streams a GeoJSON FeatureCollection for the kepler.gl Trip layer.

    Each vessel's track is split into contiguous segments of constant
    ``anomaly_type`` and written as a separate LineString feature, so the
    anomalous stretch of a trail can be colored distinctly. Coordinates are
    ``[lon, lat, 0, epoch_seconds]`` -- the 4th element is the timestamp kepler
    animates over. Adjacent segments share a boundary vertex so trails stay
    visually continuous.
    """

    def __init__(self, cfg: Config, path: Optional[str] = None):
        self.cfg = cfg
        self.path = path or cfg.output_path
        self._fh = open(self.path, "w")
        self._fh.write('{"type":"FeatureCollection","features":[')
        self._first = True
        self.total_rows = 0      # position fixes consumed
        self.total_features = 0

    def _write_feature(self, entity_id, vessel_type, atype_val, coords, sog_mean):
        if len(coords) < 2:
            return
        props = {
            "entity_id": int(entity_id),
            "anomaly_type": str(atype_val),
            "is_anomalous": 0 if atype_val == "none" else 1,
        }
        if self.cfg.include_vessel_type:
            props["vessel_type"] = str(vessel_type)
        if sog_mean is not None:
            props["sog_knots_mean"] = round(float(sog_mean), 2)
        feature = {
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "LineString", "coordinates": coords},
        }
        if not self._first:
            self._fh.write(",")
        self._fh.write(json.dumps(feature, separators=(",", ":")))
        self._first = False
        self.total_features += 1

    def emit(self, entity_id, vessel_type, lat, lon, times, sog, is_anom, atype):
        n = len(lat)
        self.total_rows += n
        if n < 2:
            return
        epoch = times.astype("datetime64[s]").astype(np.int64)
        lon_r = np.round(lon, self.cfg.coord_decimals)
        lat_r = np.round(lat, self.cfg.coord_decimals)
        # Boundaries where anomaly_type changes -> contiguous segments.
        change = np.flatnonzero(atype[1:] != atype[:-1]) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [n]])
        for s, e in zip(starts, ends):
            hi = min(e, n - 1)  # include next vertex for continuity
            sl = slice(s, hi + 1)
            coords = [
                [float(lon_r[i]), float(lat_r[i]), 0, int(epoch[i])]
                for i in range(sl.start, sl.stop)
            ]
            sm = float(np.mean(sog[s:e])) if sog is not None else None
            self._write_feature(entity_id, vessel_type, atype[s], coords, sm)

    def close(self):
        self._fh.write("]}")
        self._fh.close()


def _csv_columns(cfg: Config) -> List[str]:
    cols = ["entity_id", "lat", "lon", "timestamp"]
    if cfg.include_vessel_type:
        cols.append("vessel_type")
    if cfg.include_speed:
        cols.append("sog_knots")
    if cfg.include_label:
        cols.append("is_anomalous")
    if cfg.include_anomaly_type:
        cols.append("anomaly_type")
    return cols


class CompositeWriter:
    """Fans every ``emit`` out to several underlying writers (e.g. CSV + GeoJSON
    from a single generation pass, so both outputs describe identical data)."""

    def __init__(self, writers):
        self.writers = writers

    def emit(self, *args):
        for w in self.writers:
            w.emit(*args)

    def close(self):
        for w in self.writers:
            w.close()

    @property
    def total_rows(self):
        return self.writers[0].total_rows if self.writers else 0

    @property
    def total_features(self):
        return sum(getattr(w, "total_features", 0) for w in self.writers)

    @property
    def paths(self):
        return [w.path for w in self.writers]


def _derive_both_paths(output_path: str):
    """Derive (csv_path, geojson_path) from a single --output for 'both' mode."""
    p = output_path
    if p.endswith(".geojson"):
        base = p[: -len(".geojson")]
        return base + ".csv", p
    if p.endswith(".csv.gz"):
        return p, p[: -len(".csv.gz")] + ".geojson"
    if p.endswith(".csv"):
        return p, p[: -len(".csv")] + ".geojson"
    return p + ".csv", p + ".geojson"


def _make_writer(cfg: Config):
    if cfg.output_format == "csv":
        return CompositeWriter([CsvChunkWriter(cfg)])
    if cfg.output_format == "trip-geojson":
        return CompositeWriter([TripGeoJsonWriter(cfg)])
    # both
    csv_path, gj_path = _derive_both_paths(cfg.output_path)
    return CompositeWriter([CsvChunkWriter(cfg, csv_path), TripGeoJsonWriter(cfg, gj_path)])


# --------------------------------------------------------------------------- #
# Per-vessel emit
# --------------------------------------------------------------------------- #
def _sog_knots(lat, lon, dt_hours):
    if len(lat) < 2:
        return np.zeros(len(lat))
    d_km = geo.haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    sog = np.empty(len(lat))
    sog[0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        sog[1:] = np.where(dt_hours > 0, d_km / dt_hours / geo.KM_PER_NM, 0.0)
    return sog


def _emit_entity(writer, cfg, entity_id, vessel_type, lat, lon, times,
                 is_anom, atype, keep, rng, want_speed):
    if keep is not None:
        lat, lon = lat[keep], lon[keep]
        t = times[keep]
        is_anom, atype = is_anom[keep], atype[keep]
    else:
        t = times

    if cfg.position_noise_km > 0:
        lat = lat + geo.km_to_deg_lat(rng.normal(0.0, cfg.position_noise_km, size=len(lat)))
        lon = lon + geo.km_to_deg_lon(rng.normal(0.0, cfg.position_noise_km, size=len(lon)), lat)

    sog = None
    if want_speed and len(t) >= 2:
        dt_h = np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0
        sog = _sog_knots(lat, lon, dt_h)

    writer.emit(entity_id, vessel_type, lat, lon, t, sog, is_anom, atype)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def simulate(cfg: Config, verbose: bool = True) -> dict:
    """Run the simulation and write the output. Returns a summary dict."""
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_steps
    bbox = cfg.region_bbox

    if cfg.avoid_land and not geo.has_land_mask() and verbose:
        print("  [note] avoid_land requested but `global-land-mask` is not "
              "installed; proceeding without land avoidance.")
    if cfg.output_format in ("trip-geojson", "both") and verbose:
        est = cfg.num_entities * n
        if est > 3_000_000:
            print(f"  [warn] trip-geojson with ~{est:,} fixes will be a very "
                  "large/slow file; consider a smaller viz-sized run.")

    start = np.datetime64(cfg.start_time.replace("Z", ""), "s")
    interval_s = int(round(cfg.report_interval_minutes * 60))
    times = start + np.arange(n) * np.timedelta64(interval_s, "s")

    vessel_types = ves.assign_vessel_types(rng, cfg.num_entities)
    plan = anom.plan_anomalies(rng, cfg)

    writer = _make_writer(cfg)
    want_speed = cfg.include_speed or cfg.output_format in ("trip-geojson", "both")
    empty_atype = np.full(n, "none", dtype="<U24")
    done = np.zeros(cfg.num_entities, dtype=bool)

    def fresh_labels():
        return np.zeros(n, dtype=bool), empty_atype.copy()

    def route(vt, center=None):
        return ves.build_route(rng, vt, bbox, center=center, avoid_land=cfg.avoid_land)

    def track(vt, route_lat, route_lon):
        return ves.generate_track(rng, route_lat, route_lon,
                                  ves.cruise_speed(rng, vt), n, cfg.report_interval_hours)

    t0 = _time.time()

    # --- 1) paired anomalies (both vessels generated together) --------------
    for a_id, b_id, ep in plan.pairs:
        center = geo.sample_water_point(rng, bbox)
        lat_a, lon_a = track(vessel_types[a_id], *route(vessel_types[a_id], center))
        lat_b, lon_b = track(vessel_types[b_id], *route(vessel_types[b_id], center))
        ia, ta = fresh_labels()
        ib, tb = fresh_labels()
        if ep.atype == "transshipment":
            idx = anom.inject_transshipment(rng, cfg, lat_a, lon_a, lat_b, lon_b, ep)
            ia[idx] = True; ta[idx] = "transshipment"
            ib[idx] = True; tb[idx] = "transshipment"
        else:  # aggressive_maneuvering -- only the aggressor (b) is anomalous
            idx = anom.inject_aggressive(rng, cfg, lat_a, lon_a, lat_b, lon_b, ep)
            ib[idx] = True; tb[idx] = "aggressive_maneuvering"

        _emit_entity(writer, cfg, a_id, vessel_types[a_id], lat_a, lon_a,
                     times, ia, ta, None, rng, want_speed)
        _emit_entity(writer, cfg, b_id, vessel_types[b_id], lat_b, lon_b,
                     times, ib, tb, None, rng, want_speed)
        done[a_id] = done[b_id] = True

    # --- 2) solo anomalies + 3) normal vessels ------------------------------
    for entity_id in range(cfg.num_entities):
        if done[entity_id]:
            continue
        vt = vessel_types[entity_id]
        lat, lon = track(vt, *route(vt))
        is_anom, atype = fresh_labels()
        keep = None

        for ep in plan.solo.get(entity_id, []):
            if ep.atype == "ais_spoofing":
                idx = anom.inject_spoofing(rng, cfg, lat, lon, ep)
                is_anom[idx] = True; atype[idx] = "ais_spoofing"
            elif ep.atype == "illegal_fishing":
                idx = anom.inject_illegal_fishing(rng, cfg, lat, lon, ep)
                is_anom[idx] = True; atype[idx] = "illegal_fishing"
            elif ep.atype == "dark_activity":
                drop, label = anom.inject_dark_activity(rng, cfg, lat, lon, ep)
                if keep is None:
                    keep = np.ones(n, dtype=bool)
                keep[drop] = False
                if len(label):
                    is_anom[label] = True; atype[label] = "dark_activity"

        _emit_entity(writer, cfg, entity_id, vt, lat, lon,
                     times, is_anom, atype, keep, rng, want_speed)

        if verbose and entity_id % 5000 == 0 and entity_id > 0:
            print(f"  ... {entity_id:,}/{cfg.num_entities:,} entities "
                  f"({writer.total_rows:,} fixes, {_time.time() - t0:,.0f}s)")

    writer.close()
    elapsed = _time.time() - t0

    summary = {
        "output_paths": writer.paths,
        "output_format": cfg.output_format,
        "entities": cfg.num_entities,
        "anomalous_entities": plan.n_anomalous(),
        "anomaly_rate": plan.n_anomalous() / cfg.num_entities,
        "total_rows": writer.total_rows,
        "n_steps": n,
        "seconds": round(elapsed, 1),
        "pairs": len(plan.pairs),
        "land_avoidance": cfg.avoid_land and geo.has_land_mask(),
    }
    if cfg.output_format in ("trip-geojson", "both"):
        summary["features"] = writer.total_features
    if verbose:
        print(f"\nDone in {elapsed:,.1f}s -> {', '.join(writer.paths)}")
        extra = (f", {writer.total_features:,} trip features"
                 if cfg.output_format in ("trip-geojson", "both") else "")
        print(f"  {writer.total_rows:,} fixes{extra}, {plan.n_anomalous():,} "
              f"anomalous entities ({summary['anomaly_rate']*100:.2f}%)")
    return summary
