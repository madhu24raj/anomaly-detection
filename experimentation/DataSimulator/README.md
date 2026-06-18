# Maritime AIS Data Simulator

Generates synthetic [AIS](https://en.wikipedia.org/wiki/Automatic_identification_system)
vessel-tracking data for a configurable geographic region (default: the wider
Caribbean), with a tunable, low rate of **labeled anomalous behavior**. It
exists so a data science team can start building and evaluating anomaly
detectors *before* real commercial AIS data is procured.

This is a **data simulator**, not a maritime physics engine. The goal is data
with realistic, *detectable* anomaly signatures plus quasi ground-truth labels —
not navigational accuracy.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Small smoke test first (a few seconds, ~tens of MB):
python run_simulation.py --entities 500 --days 3 --interval 30 \
    --output sample.csv --vessel-type --speed
```

Then scale up. **The defaults are large**: 50,000 vessels × 30 days at a 5-minute
cadence is **~432M rows / ~20 GB** and takes ~9 minutes on a laptop. Use a
`.csv.gz` output path to compress (~4x smaller), and start smaller while you
build the pipeline.

```bash
# The full default run (heads-up: ~20 GB, ~9 min). .gz cuts that to ~5 GB.
python run_simulation.py --output ais_full.csv.gz
```

Peak memory stays ~1–1.5 GB regardless of output size — rows are streamed to
disk in chunks.

## Output schema

| column         | always? | description |
|----------------|---------|-------------|
| `entity_id`    | yes     | integer vessel id |
| `lat`, `lon`   | yes     | position (WGS84 degrees) |
| `timestamp`    | yes     | UTC report time (ISO 8601) |
| `vessel_type`  | `--vessel-type` | cargo / tanker / fishing / passenger / tug / pleasure |
| `sog_knots`    | `--speed`       | speed over ground, derived from consecutive fixes |
| `is_anomalous` | on by default (`--no-label` to drop)        | `1` if this report is inside an anomaly episode |
| `anomaly_type` | on by default (`--no-anomaly-type` to drop) | which anomaly, else `none` |

**Labels are per-row and episodic.** A vessel flagged anomalous is normal most
of the time and only labeled `1` during its anomaly episode(s) — so the labels
give you precise start/stop ground truth, not just a per-vessel flag.

## Anomaly types

A vessel is anomalous only ~`anomaly_rate` of the fleet (default 1%), and even
those are only anomalous during specific episodes. Signatures are grounded in
open-source maritime-domain-awareness / IUU-fishing literature:

| type | what it looks like in the data | detection hint |
|------|-------------------------------|----------------|
| `illegal_fishing` | slow (~1–3 kn) loitering / zig-zag **inside a restricted zone** | low SOG + inside a zone polygon |
| `transshipment` | two vessels meet at sea: co-located (<~150 m), both <~1 kn, sustained, far from port | pairwise proximity + low SOG |
| `ais_spoofing` | position teleports to a false location and back | impossible implied speed (>>30 kn) between consecutive fixes |
| `dark_activity` | vessel stops transmitting for hours/days, then reappears displaced | large time gap between consecutive reports |
| `aggressive_maneuvering` | one vessel shadows another at close range (~0.2–1 nm) with erratic, rapid maneuvers and close passes | sustained close proximity to another track + high course/speed variance |

`transshipment` and `aggressive_maneuvering` involve a **pair** of vessels and
are generated together so the proximity is real. For `aggressive_maneuvering`
only the aggressor is labeled; the victim's track is normal. For
`transshipment` both vessels are labeled.

Restricted zones and ports are defined in [`ais_sim/geography.py`](ais_sim/geography.py)
(illustrative Caribbean locations — swap in real boundaries when you have them).

## Common options

```
--entities N         number of vessels                (default 50000)
--days D             simulation duration in days       (default 30)
--interval M         report cadence in minutes         (default 5)
--anomaly-rate R     fraction of vessels anomalous     (default 0.01)
--weight TYPE=VAL    relative mix of anomaly types (repeatable)
--output PATH        .csv, .csv.gz (gzip auto-detected), or .geojson
--format FMT         csv | trip-geojson | both         (default csv)
--no-land-avoidance  skip steering around land (faster; allows over-land tracks)
--vessel-type        add vessel_type column
--speed              add sog_knots column
--no-label           drop is_anomalous column
--no-anomaly-type    drop anomaly_type column
--seed S             RNG seed for reproducibility      (default 42)
```

Re-weight the anomaly mix (relative weights, normalized internally):

```bash
python run_simulation.py --entities 5000 --anomaly-rate 0.03 \
    --weight illegal_fishing=0.5 --weight ais_spoofing=0.3 \
    --weight transshipment=0.2
```

Everything is also settable in code via the `Config` dataclass
([`ais_sim/config.py`](ais_sim/config.py)):

```python
from ais_sim import Config, simulate
simulate(Config(num_entities=10_000, duration_days=14, report_interval_minutes=10))
```

## Vessel dynamics & land avoidance

Tracks are not straight lines. Each route is a **correlated random walk** — a
gentle random turn each step plus a soft pull toward the destination — so paths
curve organically. Speed follows a **smoothed profile** (vessels accelerate and
decelerate gradually) rather than a constant velocity.

Vessels also **steer around land**. The optional [`global-land-mask`](https://pypi.org/project/global-land-mask/)
package bundles a coarse global land/sea grid, so as each route is built the
simulator checks whether the next hop's whole *segment* stays over water and
rotates the heading until it does — vessels round islands and follow coastlines
instead of driving over them. The waypoint budget scales with route distance so
the walk can fully round large landmasses (e.g. Hispaniola), and a leg is only
drawn to its destination if that segment is clear water. This drops over-land
position fixes to under ~1% (the residual is short coastline clipping where the
straight segment between waypoints cuts a cape or bay — typically a few fixes,
at most ~2 h), versus ~15% (the land fraction of the bounding box) without it.

If `global-land-mask` is not installed, the simulator still runs — land
avoidance just turns off (a notice is printed). `--no-land-avoidance` disables
it explicitly for a faster run. Land avoidance costs runtime (~15 min vs ~9 min
for the full 50k default), independent of report cadence, because steering
happens at waypoint granularity, not per report.

## Visualizing in kepler.gl

**CSV + Point layer** (simplest): load the CSV, color the point layer by
`anomaly_type`, add a time filter on `timestamp` for the playback slider. Keep
the viz dataset small (≲1M rows) — kepler renders in-browser. A good viz run:

```bash
python run_simulation.py --entities 1000 --days 7 --interval 15 \
    --anomaly-rate 0.05 --vessel-type --speed --output viz.csv
```

**Trip-GeoJSON + Trip layer** (animated fading trails): export with
`--format trip-geojson`. Each vessel's track is split into one LineString
feature per contiguous anomaly state, with coordinates `[lon, lat, 0, epoch]`,
so kepler animates moving trails and the anomalous stretch of a trail can be
colored on its own (color the Trip layer by `anomaly_type`).

```bash
python run_simulation.py --entities 1000 --days 7 --interval 15 \
    --anomaly-rate 0.05 --format trip-geojson --vessel-type --output viz.geojson
```

Trip-GeoJSON is for viz-sized runs only — it is far larger per fix than CSV.

**Both at once** (`--format both`): one generation pass writes a `.csv` (for
analysis) and a `.geojson` (for kepler) describing *identical* data — same
seed, same vessels, same anomaly placements. The two paths are derived from
`--output` (e.g. `--output viz.csv` → `viz.csv` + `viz.geojson`). This is the
recommended way to get both, since converting GeoJSON back to CSV is lossy
(coordinates are rounded and `sog` is stored per-segment, not per-fix).

```bash
python run_simulation.py --entities 1000 --days 7 --interval 15 \
    --anomaly-rate 0.05 --format both --vessel-type --speed --output viz.csv
```

## Project layout

```
run_simulation.py        CLI entry point
ais_sim/
  config.py              Config dataclass + anomaly weights / durations
  geography.py           region, ports, restricted zones, distance, routing
  vessels.py             vessel types, fleet generation, normal trajectories
  anomalies.py           episode planning + per-type injection
  simulator.py           orchestration + CSV / trip-geojson streaming writers
```

## Caveats (by design)

- Land avoidance uses a *coarse* mask and steers at waypoint granularity, so a
  small fraction (<~1%) of fixes still clip coastlines via interpolation between
  waypoints. No bathymetry, shipping-lane rules, or port geometry. Fine for
  behavioral anomaly detection.
- Movement is piecewise-linear between waypoints (equirectangular), not true
  great-circle navigation. Accurate enough at this scale.
- Normal traffic is stylized (port-to-port liners, fishing grounds, coastal
  hops); it is *plausible*, not calibrated to real traffic densities.
- Labels are quasi ground-truth for *injected* anomalies. Normal vessels may
  occasionally produce borderline-looking behavior from noise — that is
  realistic and useful for measuring false-positive rates.
