"""
Generate synthetic Databricks cluster metrics sample data.

Drops two files:
 - cluster_sample_data.json
 - cluster_sample_data.csv

"""

import os
import json
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Reproducible randomness
random.seed(42)
np.random.seed(42)

# OUT_DIR = "data"
# os.makedirs(OUT_DIR, exist_ok=True)

# Cluster size definitions , calculating cost in DBU's per hour (example rates)
CLUSTER_SIZES = [
    {"size_key": "2x-small", "worker_cores": 2, "worker_mem_gb": 8,  "driver_cores": 2,  "driver_mem_gb": 8,  "base_workers_min": 1, "base_workers_max": 2, "cost_per_worker_hr": 0.05},
    {"size_key": "small",    "worker_cores": 4, "worker_mem_gb": 16, "driver_cores": 4,  "driver_mem_gb": 16, "base_workers_min": 2, "base_workers_max": 4, "cost_per_worker_hr": 0.10},
    {"size_key": "medium",   "worker_cores": 8, "worker_mem_gb": 32, "driver_cores": 8,  "driver_mem_gb": 32, "base_workers_min": 4, "base_workers_max": 8, "cost_per_worker_hr": 0.20},
    {"size_key": "large",    "worker_cores": 16,"worker_mem_gb": 64, "driver_cores": 16, "driver_mem_gb": 64, "base_workers_min": 6, "base_workers_max": 12,"cost_per_worker_hr": 0.40},
    {"size_key": "2x-large", "worker_cores": 32,"worker_mem_gb":128, "driver_cores": 32, "driver_mem_gb":128, "base_workers_min": 8, "base_workers_max": 20,"cost_per_worker_hr": 0.80},
    {"size_key": "3x-large", "worker_cores": 48,"worker_mem_gb":192, "driver_cores": 48, "driver_mem_gb":192, "base_workers_min": 10,"base_workers_max": 30,"cost_per_worker_hr": 1.20},
]

# Simulation parameters
CLUSTERS_PER_SIZE = 3            # number of distinct clusters per size family
DAYS = 1                         # days of history (short for quick runs)
SAMPLES_PER_DAY = 2              # samples per day (every ~12 hours)
TOTAL_SAMPLES_PER_CLUSTER = DAYS * SAMPLES_PER_DAY

# ---------------------------------------------------------------------
# Helper: gaussian noise clipped
# ---------------------------------------------------------------------
def clipped_normal(loc, scale, low=None, high=None):
    """Return a normally-distributed sample centered at loc with sd=scale, clipped to [low, high] when provided."""
    v = random.gauss(loc, scale)
    if low is not None:
        v = max(v, low)
    if high is not None:
        v = min(v, high)
    return v

# ---------------------------------------------------------------------
# Build rows
# ---------------------------------------------------------------------
rows = []
now = datetime.utcnow()

cluster_id_seq = 1
for size in CLUSTER_SIZES:
    for cluster_instance in range(CLUSTERS_PER_SIZE):
        cluster_id = f"cluster-{size['size_key']}-{cluster_id_seq}"
        cluster_id_seq += 1

        # choose baseline number of workers within allowed range (per cluster)
        base_workers = random.randint(size["base_workers_min"], size["base_workers_max"])

        # created_at metadata (per cluster)
        created_at = now - timedelta(days=random.randint(60, 600))

        # per-cluster baseline usage (keep constant across the cluster's samples)
        base_cpu_worker = clipped_normal(0.35, 0.12, 0.05, 0.95)  # fraction 0-1
        base_cpu_driver = clipped_normal(0.25, 0.10, 0.01, 0.95)

        # Simulate a time series of samples
        for sample_idx in range(TOTAL_SAMPLES_PER_CLUSTER):
            # timestamp for this sample (monotonic, going backwards from now)
            hours_back = (TOTAL_SAMPLES_PER_CLUSTER - 1 - sample_idx) * (24.0 / SAMPLES_PER_DAY)
            ts = now - timedelta(hours=hours_back)

            # slightly vary worker count to simulate autoscaling
            workers = max(1, int(np.round(base_workers * clipped_normal(1.0, 0.12, 0.6, 1.6))))
            drivers = 1  # usually single driver in standard clusters

            # compute capacities
            worker_total_cores = workers * size["worker_cores"]
            worker_total_mem = workers * size["worker_mem_gb"]
            driver_cores = size["driver_cores"]
            driver_mem = size["driver_mem_gb"]

            # simulate workload intensity (0..1) with some clusters heating up over time
            time_factor = sample_idx / float(TOTAL_SAMPLES_PER_CLUSTER) if TOTAL_SAMPLES_PER_CLUSTER > 0 else 0.0  # 0 -> ~1
            growth_trend = 1.0 + 0.5 * time_factor * random.uniform(0.0, 1.0)

            # Simulate current usage with noise and growth
            # ------------------ FIX: clamp lower bound to avoid negative CPU fractions ------------------
            cpu_worker_frac = (base_cpu_worker * growth_trend) + random.gauss(0, 0.07)
            cpu_worker_frac = max(0.0, min(0.99, cpu_worker_frac))

            cpu_driver_frac = (base_cpu_driver * growth_trend * random.uniform(0.9, 1.1)) + random.gauss(0, 0.06)
            cpu_driver_frac = max(0.0, min(0.99, cpu_driver_frac))
            # -------------------------------------------------------------------------------------------

            # Memory usage in GB (some clusters use more memory due to caching)
            mem_worker_used = min(
                worker_total_mem * 0.98,
                max(
                    0.2,
                    (size["worker_mem_gb"] * workers) * (0.2 + base_cpu_worker * 1.2 * growth_trend) + random.gauss(0, 2.0),
                ),
            )
            mem_driver_used = min(
                driver_mem * 0.98,
                max(
                    0.1,
                    (driver_mem) * (0.15 + base_cpu_driver * 1.0 * growth_trend) + random.gauss(0, 1.0),
                ),
            )

            # job counts and failures
            running_jobs = max(0, int(np.round(random.expovariate(1 / 1.8) * (1 + growth_trend))))
            if cpu_worker_frac > 0.9 or mem_worker_used > worker_total_mem * 0.92:
                failed_jobs = random.choices([0, 1, 2], weights=[0.6, 0.3, 0.1])[0]
            else:
                failed_jobs = random.choices([0, 1], weights=[0.9, 0.1])[0]

            # estimated duration for "sample window" in minutes
            duration_min = max(5.0, 30.0 * (0.4 + cpu_worker_frac + random.gauss(0, 0.05)))

            # cost estimate per sample window (simple model)
            window_hours = 1.0 / SAMPLES_PER_DAY if SAMPLES_PER_DAY > 0 else 0.0
            cost_estimated = workers * size["cost_per_worker_hr"] * window_hours

            # prepare row
            row = {
                "cluster_id": cluster_id,
                "cluster_size": size["size_key"],
                "created_at": created_at.isoformat(),
                "timestamp": ts.isoformat(),
                "sample_index": sample_idx,
                "num_workers": int(workers),
                "num_drivers": int(drivers),
                "worker_cores_per_node": int(size["worker_cores"]),
                "worker_mem_gb_per_node": float(size["worker_mem_gb"]),
                "driver_cores": int(driver_cores),
                "driver_mem_gb": float(driver_mem),
                "worker_total_cores": int(worker_total_cores),
                "worker_total_mem_gb": float(worker_total_mem),
                "cpu_usage_percent_workers_avg": round(float(cpu_worker_frac * 100), 2),
                "cpu_usage_percent_driver": round(float(cpu_driver_frac * 100), 2),
                "memory_used_gb_workers_avg": round(float(mem_worker_used / workers) if workers > 0 else 0.0, 2),
                "memory_used_gb_driver": round(float(mem_driver_used), 2),
                "running_jobs": int(running_jobs),
                "failed_jobs": int(failed_jobs),
                "duration_min_estimate": round(duration_min, 2),
                "cost_estimated_usd": round(cost_estimated, 4),
            }
            rows.append(row)

print(f"Generated {len(rows)} sample rows across {len(CLUSTER_SIZES) * CLUSTERS_PER_SIZE} clusters.")

# ---------------------------------------------------------------------
# Save to CSV & JSON
# ---------------------------------------------------------------------
df = pd.DataFrame(rows)
csv_path = os.path.join("cluster_sample_data.csv")
json_path = os.path.join("cluster_sample_data.json")

df.to_csv(csv_path, index=False)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print("Saved CSV to:", csv_path)
print("Saved JSON to:", json_path)
