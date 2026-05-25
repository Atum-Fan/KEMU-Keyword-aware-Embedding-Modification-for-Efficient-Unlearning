#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


def setup_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "task_name",
        type=str,
        help="Task name used to locate eval outputs and write the merged summary.",
    )
    return parser.parse_args()


def _resolve_saves_dir() -> Path:
    """Get saves directory path"""
    local_saves = Path.cwd() / "saves"
    if local_saves.exists():
        return local_saves
    return Path("/data/home/${USER:-user}/run/openunlearning/saves")


def main():
    args = setup_args()
    task_name = args.task_name

    source_base = _resolve_saves_dir()
    source_dir = source_base / "unlearn" / task_name / "evals"

    # Support TOFU and MUSE formats
    summary_file = source_dir / "MUSE_SUMMARY.json"
    if not summary_file.exists():
        summary_file = source_dir / "TOFU_SUMMARY.json"

    param_stats_file = source_dir / "PARAM_CHANGE.json"
    training_stats_file = source_dir / "TRAINING_STATS.json"
    memory_trace_file = source_dir / "MEMORY_TRACE.json"

    dest_dir = Path.cwd() / "grid_search"
    dest_file = dest_dir / f"{task_name}.json"

    # Check source directory
    if not source_dir.exists():
        print(f"[ERROR] no dir: {source_dir}")
        sys.exit(2)

    if not summary_file.exists():
        print(f"[ERROR] no summary file: {summary_file}")
        sys.exit(3)

    # Check target directory
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"[ERROR] cannot create {dest_dir}: {exc}")
        sys.exit(4)

    try:
        # 1. Read key metrics
        with summary_file.open("r", encoding="utf-8") as fh:
            merged = json.load(fh)

        # 2. Merge parameter changes
        param_stats = None
        if param_stats_file.exists():
            with param_stats_file.open("r", encoding="utf-8") as fh:
                param_stats = json.load(fh)
            merged["parameter_change"] = param_stats.get("parameter_change", param_stats)

        # 3. Merge training stats
        training_stats = None
        if training_stats_file.exists():
            with training_stats_file.open("r", encoding="utf-8") as fh:
                training_stats = json.load(fh)
            merged["training_cost"] = training_stats

        # 4. Merge memory trace
        if memory_trace_file.exists():
            with memory_trace_file.open("r", encoding="utf-8") as fh:
                merged["memory_trace"] = json.load(fh)

        # 5. Extract resource_usage
        resource_usage = {}
        if param_stats is not None:
            param_change = param_stats.get("parameter_change", param_stats)
            resource_usage["trainable_params_b"] = param_change.get("trainable_numel_b")
            resource_usage["trainable_ratio"] = param_change.get("trainable_ratio")
            resource_usage["trainable_scope"] = param_change.get("trainable_scope")

            
            all_numel = param_change.get("all_numel", 0)
            all_changed_ratio = param_change.get("all_changed_ratio", 0)
            if all_numel > 0 and all_changed_ratio > 0:
                modified_count = int(all_numel * all_changed_ratio)
                resource_usage["modified_params_count"] = modified_count
                resource_usage["modified_params_b"] = modified_count / 1e9
                resource_usage["modified_params_ratio"] = all_changed_ratio
            else:
                resource_usage["modified_params_b"] = param_change.get("all_modified_count_b")
                resource_usage["modified_params_count"] = param_change.get("all_modified_count")
                resource_usage["modified_params_ratio"] = param_change.get("all_modified_ratio")

        if training_stats is not None:
            resource_usage["peak_gpu_memory_gb"] = training_stats.get("peak_gpu_memory_gb")
            resource_usage["gpu_hours"] = training_stats.get("gpu_hours")
            resource_usage["wall_time_hours"] = training_stats.get("wall_time_hours")
            resource_usage["training_time_hours"] = training_stats.get("wall_time_hours")
            resource_usage["num_gpus"] = training_stats.get("num_gpus")

        if resource_usage:
            merged["resource_usage"] = resource_usage

        # 6. Write merged result
        with dest_file.open("w", encoding="utf-8") as fh:
            json.dump(merged, fh, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] : {dest_file}")
        sys.exit(0)
    except Exception as exc:
        print(f"[ERROR] failed to write merged results: {exc}")
        sys.exit(5)


if __name__ == "__main__":
    main()
