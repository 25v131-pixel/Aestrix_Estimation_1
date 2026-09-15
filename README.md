# Aestrix Online Cone Mapping — Phase 2

An online (streaming) cone-mapping pipeline for a Formula Student–style
perception stack. Fuses telemetry + perception logs, tick-by-tick, through
coordinate transform → Mahalanobis-gated association → a static-target
Kalman filter → a three-signal ghost-rejection state machine, producing a
track-level cone map with uncertainty ellipses. Includes an offline
ablation/sensitivity harness used to tune the deployed configuration.

---

## 1. Project layout

```
main.py                     Orchestration: runs the streaming pipeline, writes
                              final_tracks.csv, the map PNG, and demo frames.
config.py                   DEFAULT_CONFIG — every tunable parameter, in one place.
ablation_study.py           Offline dev-time tool: ablation grid + parameter
                              sensitivity sweep. Never runs on the Jetson.
src/
  coordinate_transform.py   Sensor-frame → global-frame transform (per-detection
                            online version + a Phase-1 batch version).
  clustering.py             Detection-to-track association (chi-squared gate
                            on Mahalanobis distance) + a Phase-1 batch version.
  static_kalman_filter.py   Static-target Kalman filter (predict/update),
                              per-track, streaming.
  ghost_rejection.py        Confidence / persistence / spatial-consistency
                            filters + candidate→confirmed→ghost lifecycle.
  plot_map.py               Renders the track map (with covariance ellipses)
                              and the incremental demo-frame sequence.
data/
   telemetry_log.csv        Vehicle pose stream (timestamp, x, y, yaw_rad, ...).
   perception_log.csv       Raw cone detections (timestamp, rel_x/y_sensor,
                              confidence, cone_type, label).
outputs/                    Created on first run: final_tracks.csv,
                              final_track_map.png, demo_frames/, ablation_report.txt
```

**Data flow (per tick):** `telemetry row → predict_track (all active tracks)
→ transform_single_detection (each detection this tick) →
associate_single_detection → init_track (new) or update_track (matched) →
record_detection → evaluate_track (lifecycle) → prune stale non-confirmed
tracks.`

The online loop in `main.py` (`run_pipeline_online`) is intentionally
**single-threaded** — each tick depends on the previous tick's Kalman state,
so there's no safe way to parallelize it. Only the *offline* ablation grid
(independent runs) is parallelized.

---

## 2. Setup & dependencies

Requires Python 3.9+.

```bash
pip install pandas numpy scipy matplotlib
```

That's the full dependency list — no ML framework or GPU library is needed
for this stage, since the pipeline is classical/geometric (transform →
gating → Kalman filter → rule-based lifecycle), not a learned model.

No installation step is required beyond the packages above; everything runs
from the project root with plain `python`.

### File paths

By default, `config.py` points at:

```python
PERCEPTION_FILE = "data/perception_log.csv"
TELEMETRY_FILE  = "data/telemetry_log.csv"
```

Either place your logs at `data/perception_log.csv` and
`data/telemetry_log.csv` relative to where you run the scripts, or pass
explicit paths (see §3 and §5) — every entry point accepts
`perception_csv` / `telemetry_csv` overrides, so you never have to edit
`config.py` just to point at a different log pair.

---

## 3. Replaying a log (running the online pipeline)

```bash
python main.py
```

This will:
1. Read `telemetry_log.csv` and `perception_log.csv` in timestamp order.
2. Run the full streaming loop tick-by-tick (predict → associate → update →
   ghost-evaluate → prune), using `DEFAULT_CONFIG` from `config.py`.
3. Write:
   - `outputs/final_tracks.csv` — every track (still-live + archived/pruned),
     with position, status, and covariance.
   - `outputs/final_track_map.png` — the 2D map: vehicle trajectory, cones
     colored by lifecycle status, covariance ellipses.
   - `outputs/demo_frames/frame_0000.png, frame_0001.png, ...` — an
     incremental snapshot every `SNAPSHOT_INTERVAL_TICKS` ticks (default 20),
     for building the live-demo video (tracks locking in, ghosts decaying
     over time).

### Replaying a different log pair or a custom config

`run_pipeline_online` is the reusable entry point (this is exactly what
`ablation_study.py` also calls into):

```python
from main import run_pipeline_online, full_tracks_dataframe

result = run_pipeline_online(
    "path/to/my_perception_log.csv",
    "path/to/my_telemetry_log.csv",
    config={"gating_confidence": 0.99, "min_avg_confidence": 0.5},  # partial override
    collect_snapshots=True,   # set False to skip demo-frame snapshots (faster)
)

tracks_df = full_tracks_dataframe(result)   # live + archived tracks combined
tracks_df.to_csv("outputs/my_run_tracks.csv", index=False)
```

`config=None` reproduces `main()`'s exact default behavior. A partial dict
only overrides the keys you name; everything else falls back to
`DEFAULT_CONFIG`.

---

## 4. Configuration reference (`config.py` → `DEFAULT_CONFIG`)

All tunable parameters live in one place so pipeline code never needs
editing to change behavior.

| Key | Default | Module it drives | What it controls |
|---|---|---|---|
| `camera_offset` | `(0.0, 0.0)` | `coordinate_transform.py` | Sensor mount offset (x, y) in the vehicle frame, meters. Change only with a confirmed mounting-offset spec. |
| `apply_sensor_correction` | `True` | `coordinate_transform.py` | Applies the 180° backward-facing-sensor correction before the SE(2) transform. `False` is an ablation-only setting to measure the correction's contribution. |
| `max_time_gap` | `2.0` s | `clustering.py`, main loop pruning | Max time since last detection for a track to remain eligible for (a) new-detection association and (b) staying alive at all (non-confirmed tracks older than this are pruned/archived). Must stay well under the ~60 s lap period to avoid stitching cones across laps. |
| `gating_confidence` | `0.95` | `clustering.py` | Chi-squared confidence level (2 DOF) for the Mahalanobis association gate. Higher = looser gate (more detections accepted per track); lower = tighter. |
| `process_noise_rate` | `1e-6` /s | `static_kalman_filter.py` | Per-second process noise added during `predict_track`. Keeps an unseen track's covariance growing (prevents overconfidence) between detections. |
| `base_sigma` | `0.3` m | `static_kalman_filter.py` | Baseline measurement std-dev at `confidence == 1.0`. Scaled by `1/confidence` per detection, so low-confidence hits pull the estimate less. |
| `min_avg_confidence` | `0.6` | `ghost_rejection.py` (`confidence_filter`) | Minimum running-mean detection confidence for a track to pass. |
| `persistence_threshold` | `5` hits | `ghost_rejection.py` (`persistence_filter`) | Minimum accumulated hit count before a track can be trusted / promoted. |
| `max_avg_mahalanobis_sq` | `9.0` | `ghost_rejection.py` (`spatial_consistency_filter`) | Max running-mean squared Mahalanobis residual; catches tracks whose hits keep landing near the gate edge (stitched noise) rather than tightly clustered. |
| `promotion_hysteresis_ticks` | `3` | `ghost_rejection.py` (`evaluate_track`) | Consecutive good ticks required before `candidate → confirmed`. |
| `demotion_hysteresis_ticks` | `3` | `ghost_rejection.py` (`evaluate_track`) | Consecutive bad ticks required before `→ ghost`. |
| `enable_confidence_filter` | `True` | `ghost_rejection.py` | Toggle the confidence check on/off (ablation). |
| `enable_persistence_filter` | `True` | `ghost_rejection.py` | Toggle the persistence check on/off (ablation). |
| `enable_spatial_consistency_filter` | `True` | `ghost_rejection.py` | Toggle the spatial-consistency check on/off (ablation). |

Filters are checked cheapest-first (confidence → persistence → spatial
consistency); a track failing an earlier check skips the rest that tick.
`label` (ground truth) is never read anywhere in this table's code path —
only inside `ablation_study.validate_results()` for scoring.

Other `config.py` settings (not part of `DEFAULT_CONFIG`, but worth
knowing): `PERCEPTION_FILE` / `TELEMETRY_FILE` (default log paths),
`OUTPUT_DIR` / `FINAL_TRACKS_CSV` / `FINAL_MAP_PNG` / `DEMO_FRAMES_DIR`
(output locations), and `SNAPSHOT_INTERVAL_TICKS` (demo-frame cadence,
default every 20 ticks).

---

## 5. Ablation study & parameter sensitivity

`ablation_study.py` is strictly an **offline, dev-time** tool. It only ever
calls *into* `main.py` (`run_pipeline_online`, `full_tracks_dataframe`) —
`main.py` has no knowledge of it, and it never runs on the Jetson. Deleting
this file changes nothing about `python main.py`. Only the final tuned
`DEFAULT_CONFIG` values that come out of this process are what get deployed.

### 5.1 Run everything (ablation grid + sensitivity sweep)

```bash
python ablation_study.py
```

This runs the full ablation grid (see table below), then a 27-point
parameter sensitivity grid search (`gating_confidence` × `persistence_threshold`
× `min_avg_confidence`), and writes a combined report to
`outputs/ablation_report.txt`. Before running, update the
`PERCEPTION_CSV` / `TELEMETRY_CSV` constants at the bottom of the file (or
call the functions directly, see 5.4) to point at your actual log files.

> **Note on parallelism:** the ablation grid runs each condition as an
> independent process via `ProcessPoolExecutor` for a near-N-core speedup,
> with output order preserved via `executor.map`. This is safe *only*
> because each condition is a fully independent pipeline run — it has
> nothing to do with, and never touches, the single-threaded online loop.
> **On Windows**, if you call `run_ablation_study` from your own script
> (rather than running `ablation_study.py` directly), guard the call with
> `if __name__ == "__main__":` to prevent recursive subprocess spawning.

### 5.2 Built-in ablation conditions

| Condition | Config override | Measures |
|---|---|---|
| Baseline | — | Reference performance with `DEFAULT_CONFIG`. |
| Filter Disabled (all ghost checks off) | all three `enable_*_filter=False` | Pipeline behavior with ghost rejection entirely off. |
| Transform Disabled (no 180° correction) | `apply_sensor_correction=False` | Contribution of the sensor-mount correction. |
| No Confidence Filter | `enable_confidence_filter=False` | Isolated contribution of the confidence check. |
| No Persistence Filter | `enable_persistence_filter=False` | Isolated contribution of the persistence check. |
| No Spatial Consistency Filter | `enable_spatial_consistency_filter=False` | Isolated contribution of the spatial-consistency check. |
| No Time Gate | `max_time_gap=1e9` | Streaming robustness without the time-gap cutoff. |
| Loose Chi-Squared Gate (99.9%) | `gating_confidence=0.999` | Sensitivity to a looser association gate. |
| Tight Chi-Squared Gate (80%) | `gating_confidence=0.80` | Sensitivity to a tighter association gate. |

Each condition starts from `DEFAULT_CONFIG` and overrides only the keys
listed — everything else stays at default, so results isolate the effect of
that one change.

### 5.3 Metrics reported (per condition)

- **Track counts:** total / confirmed / candidate / ghost, plus how many
  candidate vs. ghost tracks were pruned (archived) during the run.
- **Classification metrics** (only computed if `label` is present in the
  log): accuracy, precision, recall, F1, ghost-leak rate (fraction of
  *confirmed* tracks whose majority ground-truth label is actually
  `"ghost"`), and the raw confusion matrix. A track's predicted class comes
  from its final lifecycle status — `confirmed` = predicted real cone,
  anything else = predicted not-a-cone.
- **Map-accuracy proxy:** mean confirmed-track position std (from the KF
  covariance) — an *internal consistency* proxy, not RMSE against true cone
  positions (no ground-truth cone coordinates exist in this dataset).
- **Latency & memory:** mean/median/max per-tick latency (ms), peak memory
  (KB) via `tracemalloc`.

### 5.4 Adding your own ablation condition or running a subset

```python
from ablation_study import run_ablation_study

custom_conditions = {
    "Baseline": {},
    "Very Tight Gate": dict(gating_confidence=0.60),
    "High Persistence": dict(persistence_threshold=10),
}

run_ablation_study(
    "path/to/perception_log.csv",
    "path/to/telemetry_log.csv",
    "outputs/custom_ablation_report.txt",
    conditions=custom_conditions,
)
```

Any subset or superset of `config.py`'s `DEFAULT_CONFIG` keys can be used as
an override dict — the function merges `{**DEFAULT_CONFIG, **overrides}` per
condition, so you never need to restate unrelated keys.

### 5.5 Parameter sensitivity sweep only

```python
from ablation_study import run_parameter_sensitivity

best_params, all_results = run_parameter_sensitivity(
    "path/to/perception_log.csv",
    "path/to/telemetry_log.csv",
)
```

Sweeps `gating_confidence ∈ {0.80, 0.95, 0.999}`,
`persistence_threshold ∈ {3, 5, 8}`, and
`min_avg_confidence ∈ {0.5, 0.6, 0.7}` (27 combinations), scoring each by
`F1 × 100` plus a +10 bonus if total track count lands in a sane `[10, 150]`
range. Returns the best-scoring config dict and every combination's full
metrics, for building a "top 5 parameter combinations" table.

---

## 6. Running tests / reproducing results

There's no separate unit-test suite in this repo yet; correctness is
verified through the pipeline's own regression and validation paths:

1. **Reproduce the reference run:**
   ```bash
   python main.py
   ```
   Compare the resulting `outputs/final_tracks.csv` track counts (total /
   confirmed / candidate / ghost) against a previously saved run to confirm
   determinism — the pipeline has no randomness, so identical inputs +
   identical config always produce identical output.

2. **Reproduce the full metrics report:**
   ```bash
   python ablation_study.py
   ```
   Re-generates `outputs/ablation_report.txt` (ablation table + best
   sensitivity params + top-5 table) from scratch. Because `run_pipeline_online`
   is deterministic, this file should be byte-for-byte reproducible given the
   same logs and `config.py`.

3. **Phase 1 regression check:** `clustering.cluster_detections`,
   `static_kalman_filter.fuse_cluster` / `fuse_all_clusters`, and
   `coordinate_transform.map_cones_to_global` / `plot_map.plot_global_map`
   are the retained offline/batch (Phase 1) equivalents of the online path.
   Running both on the same log and comparing cluster assignments / fused
   positions is the intended way to sanity-check that the streaming
   rewrite (Phase 2) hasn't silently changed the underlying association or
   fusion math.

4. **Visual check:** open `outputs/final_track_map.png` — confirmed (green),
   candidate (orange), and ghost (red) tracks should sit sensibly along the
   vehicle trajectory, each with a covariance ellipse; `outputs/demo_frames/`
   gives the same picture incrementally, tick by tick, for the live-demo
   recording.

---

## 7. Notes on the offline/online boundary (deployment)

- Only the tuned key/value pairs inside `DEFAULT_CONFIG` (in `config.py`)
  are meant to cross into the Jetson deployment — not any code from
  `ablation_study.py` itself.
- `ProcessPoolExecutor` and any other dev-time tooling should never run on
  the Jetson; the online pipeline (`main.py` / `run_pipeline_online`)
  remains single-threaded by design, since each tick's Kalman predict/update
  depends sequentially on the previous tick's state.
- When updating deployed parameters, change them in `config.py`'s
  `DEFAULT_CONFIG` only — this is the single source of truth both `main.py`
  and `ablation_study.py` build from.
