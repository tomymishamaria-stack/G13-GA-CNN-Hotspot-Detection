
import csv
from pathlib import Path
import pandas as pd

ROOT = Path("results_debug_oct27_seed11")
OUT = ROOT / "overall"
OUT.mkdir(parents=True, exist_ok=True)

metric_map = {
    "Accuracy": "test_accuracy",
    "Balanced Accuracy": "test_balanced_accuracy",
    "Precision": "test_precision",
    "Recall/Sensitivity": "test_recall",
    "Specificity": "test_specificity",
    "F1-score": "test_f1",
    "Fixed-0.5 Accuracy": "test_fixed_0_5_accuracy",
    "Fixed-0.5 Balanced Accuracy": "test_fixed_0_5_balanced_accuracy",
    "Fixed-0.5 Precision": "test_fixed_0_5_precision",
    "Fixed-0.5 Recall/Sensitivity": "test_fixed_0_5_recall",
    "Fixed-0.5 Specificity": "test_fixed_0_5_specificity",
    "Fixed-0.5 F1-score": "test_fixed_0_5_f1",
    "TP": "test_TP",
    "TN": "test_TN",
    "FP": "test_FP",
    "FN": "test_FN",
    "Parameters": "total_params",
    "Trainable Parameters": "trainable_params",
    "Estimated MACs (Conv/Dense only)": "estimated_macs",
    "Estimated FLOPs (2x MACs)": "estimated_flops",
    "Inference Time (ms)": "inference_time_ms_mean",
}

rows = []
extra_rows = {
    "GA Candidates": [],
    "Pareto Results": [],
    "Ablation Results": [],
    "Dataset Statistics": [],
}

for benchmark in ["ICCAD1", "ICCAD2", "ICCAD3", "ICCAD4", "ICCAD5"]:
    folder = ROOT / benchmark

    for filename, model_name in [
        ("baseline_metrics.csv", "Baseline"),
        ("hand_designed_metrics.csv", "Hand-designed lightweight CNN"),
        ("final_metrics.csv", "GA"),
    ]:
        path = folder / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        with path.open(newline="", encoding="utf-8-sig") as f:
            data = list(csv.DictReader(f))

        if not data:
            raise ValueError(f"No metric rows in {path}")

        source = data[0]
        row = {"Benchmark": benchmark, "Model": model_name}

        for output_name, source_name in metric_map.items():
            value = source.get(source_name, "")
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass
            row[output_name] = value

        row["Complexity Measurement Note"] = source.get(
            "flops_measurement_note", ""
        )
        rows.append(row)

    for filename, key in [
        ("ga_candidates.csv", "GA Candidates"),
        ("pareto_front.csv", "Pareto Results"),
        ("ablation_results.csv", "Ablation Results"),
        ("dataset_statistics.csv", "Dataset Statistics"),
    ]:
        path = folder / filename
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "Benchmark", benchmark)
            extra_rows[key].append(df)

comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "baseline_vs_ga.csv", index=False)

mean_columns = [
    c for c in metric_map
    if c not in ["TP", "TN", "FP", "FN"]
]
means = comparison.groupby("Model")[mean_columns].mean().reset_index()
means.insert(0, "Benchmark", "MEAN (benchmark-level average)")
means["TP"] = ""
means["TN"] = ""
means["FP"] = ""
means["FN"] = ""
means["Complexity Measurement Note"] = (
    "Unweighted mean across five benchmark-level results. "
    "MAC/FLOP values are estimates, not hardware execution counts."
)
means = means.reindex(columns=comparison.columns)
means.to_csv(OUT / "overall_summary.csv", index=False)

with pd.ExcelWriter(OUT / "G13_results.xlsx", engine="openpyxl") as writer:
    comparison.to_excel(writer, sheet_name="Benchmark Comparison", index=False)
    means.to_excel(writer, sheet_name="Overall Summary", index=False)

    for sheet, frames in extra_rows.items():
        if frames:
            combined = pd.concat(frames, ignore_index=True, sort=False)
            combined.to_excel(writer, sheet_name=sheet, index=False)

print("\nRecovery completed successfully.")
print("Created:")
for name in [
    "baseline_vs_ga.csv",
    "overall_summary.csv",
    "G13_results.xlsx",
]:
    print(" -", OUT / name)

print("\nMean test balanced accuracy by model:")
print(
    means[["Model", "Balanced Accuracy"]]
    .to_string(index=False)
)
