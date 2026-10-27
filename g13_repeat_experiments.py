#!/usr/bin/env python
"""Repeat G13 experiments over independent seeds and aggregate results.

Example:
  python g13_repeat_experiments.py --script g13_ga_cnn_final.py --data_root "C:\\...\\iccad-official" --output_root results_repeated --mode full --seeds 11 22 33 44 55

Each child run writes its own complete results directory. The launcher then
aggregates per-benchmark/model test metrics across seeds as mean and standard
deviation. Test metrics are for reporting only, never used for model selection.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd

METRICS = ["Accuracy", "Balanced Accuracy", "Precision", "Recall/Sensitivity",
           "Specificity", "F1-score", "Parameters", "Trainable Parameters",
           "Estimated MACs (Conv/Dense only)", "Estimated FLOPs (2x MACs)", "Inference Time (ms)"]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--script", default="g13_ga_cnn_final.py")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_root", default="g13_repeated_runs")
    p.add_argument("--mode", choices=["debug", "full"], default="full")
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    p.add_argument("--population", type=int, default=None)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--search_epochs", type=int, default=None)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--max_images_per_class", type=int, default=None)
    args = p.parse_args()
    script = Path(args.script).resolve()
    if not script.exists():
        p.error(f"Script not found: {script}")
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dirs = []
    for seed in args.seeds:
        out = root / f"seed_{seed}"
        cmd = [sys.executable, str(script), "--data_root", args.data_root,
               "--output_dir", str(out), "--mode", args.mode, "--seed", str(seed),
               "--image_size", str(args.image_size), "--overwrite"]
        if args.population is not None: cmd += ["--population", str(args.population)]
        if args.generations is not None: cmd += ["--generations", str(args.generations)]
        if args.search_epochs is not None: cmd += ["--search_epochs", str(args.search_epochs)]
        if args.max_images_per_class is not None: cmd += ["--max_images_per_class", str(args.max_images_per_class)]
        print("\n=== Running seed", seed, "===\n", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: seed {seed} exited with code {result.returncode}; continuing.", flush=True)
        else:
            run_dirs.append(out)
    frames = []
    for out in run_dirs:
        csv = out / "overall" / "baseline_vs_ga.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["Seed"] = int(out.name.split("_")[-1])
            frames.append(df)
    if not frames:
        raise SystemExit("No successful run produced overall/baseline_vs_ga.csv")
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(root / "all_seed_results.csv", index=False)
    metrics = [m for m in METRICS if m in all_df.columns]
    grouped = all_df.groupby(["Benchmark", "Model"])[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else c for c in grouped.columns]
    grouped.to_csv(root / "mean_std_by_benchmark_model.csv", index=False)
    overall = all_df.groupby("Model")[metrics].agg(["mean", "std"]).reset_index()
    overall.columns = ["_".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else c for c in overall.columns]
    overall.to_csv(root / "mean_std_overall.csv", index=False)
    (root / "REPEATED_RUNS_README.txt").write_text(
        f"Requested seeds: {args.seeds}\nSuccessful result directories: {[str(x) for x in run_dirs]}\n"
        "Files: all_seed_results.csv (individual rows), mean_std_by_benchmark_model.csv, "
        "mean_std_overall.csv. Values are mean and standard deviation across available seeds. "
        "Do not use test metrics to choose architectures or thresholds. A GA score higher than "
        "baseline is an empirical outcome, not guaranteed by the program.\n", encoding="utf-8")
    print("\nRepeated-run summaries written to:", root, flush=True)

if __name__ == "__main__":
    main()
