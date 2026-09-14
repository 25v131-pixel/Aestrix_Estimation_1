"""
ablation_study.py
------------------
Ablation and parameter-sensitivity analysis for the online pipeline.

This file is DELETABLE without affecting main.py or anything in
src/ -- it only ever calls INTO main.py (run_pipeline,
full_tracks_dataframe), never the other way around. main.py has no
import of, or knowledge of, this file. Removing this file changes
nothing about how `python main.py` runs.

Conditions covered (per Phase 2 Deliverable #4):
  - Filter disabled       : all three ghost-rejection signals off
  - Transform disabled    : 180-degree sensor correction skipped
  - Each ghost-rejection signal disabled individually
  - A few extra structural/gating conditions (No Time Gate,
    Loose/Tight chi-squared gate) -- cheap to add, useful for the
    "streaming robustness" evaluation criterion.

'label' (ground truth) is used ONLY inside validate_results() for
scoring -- never inside run_pipeline() or fed into any
threshold the pipeline itself uses.

PERFORMANCE NOTE (added):
Each call to run_pipeline() replays the full log and is independent
of every other call -- there's no shared state between conditions or
between sensitivity-grid points. That makes this file's 9 ablation
runs + 27 sensitivity-grid runs (36 total) embarrassingly parallel,
so they're now farmed out across a ProcessPoolExecutor instead of
run one-at-a-time. Results are still collected via executor.map (not
as_completed), which preserves input order -- so the printed
"New best" progression in run_parameter_sensitivity reads identically
to the old sequential version, just faster. Nothing about main.py,
config.py, or the pipeline's own logic changed.
"""

import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from main import run_pipeline_online as run_pipeline, full_tracks_dataframe, DEFAULT_CONFIG


def validate_results(tracks_df, detection_records):
    """
    Ground-truth 'label' is used HERE ONLY. A track's predicted
    class comes from its final lifecycle status: 'confirmed' =
    predicted real cone, anything else ('candidate' or 'ghost') =
    predicted not-a-confirmed-cone.
    """
    results = {}

    detections_df = pd.DataFrame(detection_records)

    if len(detections_df) > 0 and "label" in detections_df.columns:
        majority_label = (
            detections_df.groupby("track_id")["label"]
            .agg(lambda s: s.value_counts().index[0])
            .rename("majority_label")
        )
        merged = tracks_df.merge(majority_label, on="track_id", how="left")

        valid = merged.dropna(subset=["majority_label"])
        actual_ghost = (valid["majority_label"] == "ghost")
        pred_ghost = (valid["status"] != "confirmed")

        tp = int((pred_ghost & actual_ghost).sum())
        tn = int((~pred_ghost & ~actual_ghost).sum())
        fp = int((pred_ghost & ~actual_ghost).sum())
        fn = int((~pred_ghost & actual_ghost).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(valid) if len(valid) > 0 else 0.0

        confirmed = valid[valid["status"] == "confirmed"]
        leak_rate = (
            (confirmed["majority_label"] == "ghost").mean()
            if len(confirmed) > 0 else 0.0
        )

        results.update(
            accuracy=accuracy, precision=precision, recall=recall,
            f1_score=f1, ghost_leak_rate=leak_rate,
            confusion={"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        )

    results["total_tracks"] = len(tracks_df)
    results["confirmed_tracks"] = int((tracks_df["status"] == "confirmed").sum())
    results["candidate_tracks"] = int((tracks_df["status"] == "candidate").sum())
    results["ghost_tracks"] = int((tracks_df["status"] == "ghost").sum())

    confirmed_only = tracks_df[tracks_df["status"] == "confirmed"]
    if len(confirmed_only) > 0 and "cov_xx" in confirmed_only.columns:
        # Map-accuracy proxy: no ground-truth cone coordinates exist
        # in this dataset, so internal consistency (final position
        # std from the KF covariance) stands in for it -- NOT the
        # same as RMSE against true cone positions.
        results["mean_confirmed_std_m"] = float(
            (confirmed_only["cov_xx"] ** 0.5).mean() * 0.5
            + (confirmed_only["cov_yy"] ** 0.5).mean() * 0.5
        )

    return results


ABLATION_CONDITIONS = {
    "Baseline": {},
    "Filter Disabled (all ghost checks off)": dict(
        enable_confidence_filter=False,
        enable_persistence_filter=False,
        enable_spatial_consistency_filter=False,
    ),
    "Transform Disabled (no 180deg correction)": dict(
        apply_sensor_correction=False
    ),
    "No Confidence Filter": dict(enable_confidence_filter=False),
    "No Persistence Filter": dict(enable_persistence_filter=False),
    "No Spatial Consistency Filter": dict(enable_spatial_consistency_filter=False),
    "No Time Gate": dict(max_time_gap=1e9),
    "Loose Chi-Squared Gate (99.9%)": dict(gating_confidence=0.999),
    "Tight Chi-Squared Gate (80%)": dict(gating_confidence=0.80),
}


# =============================================================
# Worker functions -- module-level (not nested) so they can be
# pickled and shipped to worker processes by ProcessPoolExecutor.
# Each one runs exactly one full pipeline pass and is fully
# self-contained: no shared/mutable state with the caller.
# =============================================================

def _run_ablation_condition(args):
    name, overrides, perception_csv, telemetry_csv = args
    config = {**DEFAULT_CONFIG, **overrides}

    result = run_pipeline(
        perception_csv, telemetry_csv,
        config=config,
        collect_snapshots=False,
    )

    tracks_df = full_tracks_dataframe(result)
    metrics = validate_results(tracks_df, result["detection_records"])
    metrics.update(result["timing"])

    return name, metrics


def _run_sensitivity_point(args):
    gc, pt, mac, perception_csv, telemetry_csv = args
    config = {
        **DEFAULT_CONFIG,
        "gating_confidence": gc,
        "persistence_threshold": pt,
        "min_avg_confidence": mac,
    }
    try:
        result = run_pipeline(
            perception_csv, telemetry_csv,
            config=config, collect_snapshots=False,
        )
        tracks_df = full_tracks_dataframe(result)
        metrics = validate_results(tracks_df, result["detection_records"])
        metrics.update(result["timing"])

        score = metrics.get("f1_score", 0.0) * 100.0
        if 10 <= metrics["total_tracks"] <= 150:
            score += 10.0

        return {"params": config, "metrics": metrics, "score": score, "error": None}
    except Exception as e:
        return {"params": config, "metrics": None, "score": -1.0, "error": str(e)}


def _worker_count():
    """
    Number of worker processes to use. Leaves this overridable via
    env var for debugging (e.g. ABLATION_WORKERS=1 to force serial
    execution), otherwise uses all available cores, capped at the
    number of jobs so we never spin up idle workers for a small run.
    """
    env_override = os.environ.get("ABLATION_WORKERS")
    if env_override:
        return max(1, int(env_override))
    return max(1, os.cpu_count() or 1)


def run_ablation_study(perception_csv, telemetry_csv, output_file, conditions=None):
    conditions = conditions or ABLATION_CONDITIONS

    print("\n" + "=" * 60)
    print("ABLATION STUDY (calls main.run_pipeline)")
    print("=" * 60)

    jobs = [
        (name, overrides, perception_csv, telemetry_csv)
        for name, overrides in conditions.items()
    ]
    n_workers = min(_worker_count(), len(jobs))

    all_results = {}
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # map() preserves input order, so results print in the
            # same order as the original sequential for-loop even
            # though the underlying work runs concurrently.
            for name, metrics in executor.map(_run_ablation_condition, jobs):
                all_results[name] = metrics
                _print_condition_summary(name, metrics)
    else:
        for job in jobs:
            name, metrics = _run_ablation_condition(job)
            all_results[name] = metrics
            _print_condition_summary(name, metrics)

    _write_report(all_results, output_file)
    return all_results


def _print_condition_summary(name, metrics):
    print(f"\n{name}:")
    print(f"  Tracks: {metrics['total_tracks']} "
          f"(confirmed={metrics['confirmed_tracks']}, "
          f"candidate={metrics['candidate_tracks']}, "
          f"ghost={metrics['ghost_tracks']}) | "
          f"pruned: candidate={metrics['pruned_candidate_count']}, "
          f"ghost={metrics['pruned_ghost_count']}")
    if "f1_score" in metrics:
        print(f"  F1={metrics['f1_score']:.3f}  "
              f"Precision={metrics['precision']:.3f}  "
              f"Recall={metrics['recall']:.3f}  "
              f"Ghost-leak={metrics['ghost_leak_rate']:.3f}")
    print(f"  Latency: mean={metrics['mean_tick_ms']:.3f}ms "
          f"max={metrics['max_tick_ms']:.3f}ms  "
          f"PeakMem={metrics['peak_memory_kb']:.1f}KB")


def run_parameter_sensitivity(perception_csv, telemetry_csv):
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY (calls main.run_pipeline)")
    print("=" * 60)

    param_grid = {
        "gating_confidence": [0.80, 0.95, 0.999],
        "persistence_threshold": [3, 5, 8],
        "min_avg_confidence": [0.5, 0.6, 0.7],
    }

    jobs = [
        (gc, pt, mac, perception_csv, telemetry_csv)
        for gc in param_grid["gating_confidence"]
        for pt in param_grid["persistence_threshold"]
        for mac in param_grid["min_avg_confidence"]
    ]
    n_workers = min(_worker_count(), len(jobs))

    best_score, best_params = -1.0, None
    all_results = []

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Same order-preservation point as above: executor.map
            # yields results in job-submission order (i.e. the same
            # nested gc/pt/mac order as the original loops), so the
            # "New best" log below fires in the same sequence it did
            # when this ran serially.
            raw_results = list(executor.map(_run_sensitivity_point, jobs))
    else:
        raw_results = [_run_sensitivity_point(job) for job in jobs]

    for job, r in zip(jobs, raw_results):
        gc, pt, mac, _, _ = job
        if r["error"] is not None:
            print(f"Failed: gc={gc}, pt={pt}, mac={mac} - {r['error']}")
            continue

        metrics, score, config = r["metrics"], r["score"], r["params"]
        all_results.append({"params": config, "metrics": metrics, "score": score})

        if score > best_score:
            best_score, best_params = score, config
            print(f"\nNew best (score={score:.2f}): "
                  f"gating_confidence={gc}, persistence={pt}, min_conf={mac}")
            print(f"  F1={metrics.get('f1_score', 0):.3f}, "
                  f"Tracks={metrics['total_tracks']}")

    return best_params, all_results


def _write_report(ablation_results, output_file, best_params=None, sensitivity_results=None):
    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("PIPELINE ABLATION & SENSITIVITY REPORT (Phase 2)\n")
        f.write("=" * 60 + "\n\n")

        f.write("ABLATION STUDY RESULTS\n")
        f.write("-" * 60 + "\n")
        for name, m in ablation_results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Total tracks: {m['total_tracks']} "
                    f"(confirmed={m['confirmed_tracks']}, "
                    f"candidate={m['candidate_tracks']}, ghost={m['ghost_tracks']})\n")
            f.write(f"  Pruned (stale, removed from active state): "
                    f"candidate={m['pruned_candidate_count']}, "
                    f"ghost={m['pruned_ghost_count']}\n")
            if "f1_score" in m:
                f.write(f"  Accuracy: {m['accuracy']:.3f}\n")
                f.write(f"  Precision: {m['precision']:.3f}\n")
                f.write(f"  Recall: {m['recall']:.3f}\n")
                f.write(f"  F1-Score: {m['f1_score']:.3f}\n")
                f.write(f"  Ghost-leak rate: {m['ghost_leak_rate']:.3f}\n")
                f.write(f"  Confusion: {m['confusion']}\n")
            if "mean_confirmed_std_m" in m:
                f.write(f"  Mean confirmed-track position std (m): "
                        f"{m['mean_confirmed_std_m']:.4f}\n")
            f.write(f"  Latency (ms/tick): mean={m['mean_tick_ms']:.3f}, "
                    f"median={m['median_tick_ms']:.3f}, max={m['max_tick_ms']:.3f}\n")
            f.write(f"  Peak memory: {m['peak_memory_kb']:.1f} KB\n")

        if best_params is not None:
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("BEST PARAMETERS FOUND (sensitivity sweep)\n")
            f.write("-" * 60 + "\n")
            for k, v in best_params.items():
                f.write(f"  {k}: {v}\n")

        if sensitivity_results:
            f.write("\n\nTOP 5 PARAMETER COMBINATIONS\n")
            f.write("-" * 60 + "\n")
            top = sorted(sensitivity_results, key=lambda r: r["score"], reverse=True)[:5]
            for i, r in enumerate(top, 1):
                f.write(f"\n{i}. Score: {r['score']:.1f}\n")
                f.write(f"   gating_confidence: {r['params']['gating_confidence']}\n")
                f.write(f"   persistence_threshold: {r['params']['persistence_threshold']}\n")
                f.write(f"   min_avg_confidence: {r['params']['min_avg_confidence']}\n")
                if "f1_score" in r["metrics"]:
                    f.write(f"   F1: {r['metrics']['f1_score']:.3f}\n")

    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    PERCEPTION_CSV = "data/perception_log.csv"
    TELEMETRY_CSV = "data/telemetry_log.csv"
    OUTPUT_FILE = "outputs/ablation_report.txt"

    os.makedirs("outputs", exist_ok=True)

    ablation_results = run_ablation_study(PERCEPTION_CSV, TELEMETRY_CSV, OUTPUT_FILE)
    best_params, sensitivity_results = run_parameter_sensitivity(PERCEPTION_CSV, TELEMETRY_CSV)
    _write_report(ablation_results, OUTPUT_FILE, best_params, sensitivity_results)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)