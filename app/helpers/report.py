import io
import csv

from pathlib import Path

metrics_file = Path("/app/results/metrics.csv")

def save_metrics_csv(data):
    write_header = not metrics_file.exists()
    with open(metrics_file, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["timestamp", "model", "image_name", "elapsed_time_sec", "alpha_accurate"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(data)