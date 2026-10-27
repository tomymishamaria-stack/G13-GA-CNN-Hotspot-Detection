import csv
from pathlib import Path

root = Path("results_debug_oct27_seed11")
files = ["baseline_metrics.csv", "hand_designed_metrics.csv", "final_metrics.csv"]

for benchmark in ["ICCAD1", "ICCAD2", "ICCAD3", "ICCAD4", "ICCAD5"]:
    print("\n" + "=" * 20, benchmark, "=" * 20)
    for filename in files:
        path = root / benchmark / filename
        print("\n---", filename, "---")
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            print("COLUMNS:", reader.fieldnames)
            for row in reader:
                print(row)
