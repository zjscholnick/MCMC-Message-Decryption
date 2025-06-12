import subprocess
import time
import re
import numpy as np

def run_benchmark_script(script_name):
    cmd = [
        "python", script_name,
        "-i", "data/warpeace_input.txt",
        "-d", "secret_message.txt",
        "-e", "5000",
        "-t", "0.02",
        "-p", "10000"
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    end = time.time()
    stdout = result.stdout

    info = {
        "wall_runtime": end - start,
        "total_cpu_time": None,
        "avg_cpu": None,
        "core_util": None,
        "accept_rate": None,
        "runtime_reported": None
    }

    def get(pattern):
        match = re.search(pattern, stdout)
        return float(match.group(1)) if match else None

    info["total_cpu_time"] = get(r"Total CPU time used:\s+([\d.]+)")
    info["avg_cpu"] = get(r"Avg CPU usage per chain:\s+([\d.]+)")
    info["core_util"] = get(r"Avg core utilization.*?:\s+([\d.]+)")
    info["accept_rate"] = get(r"Overall accept rate:\s+([\d.]+)")
    info["runtime_reported"] = get(r"Runtime \(s\):\s+([\d.]+)")

    return info

def summarize_dicts(dicts, label):
    print(f"\n=== Summary for {label} ===")
    keys = dicts[0].keys()
    for key in keys:
        values = [d[key] for d in dicts if d[key] is not None]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            print(f"{key:<20}: {mean:.3f} ± {std:.3f}")

def run_n_benchmarks(script_name, n_runs=10):
    results = []
    for i in range(n_runs):
        print(f"Running {script_name} [{i+1}/{n_runs}]...")
        result = run_benchmark_script(script_name)
        results.append(result)
    return results

if __name__ == "__main__":
    pooled_results = run_n_benchmarks("run_deciphering_parallel.py", n_runs=45)
    non_pooled_results = run_n_benchmarks("run_deciphering.py", n_runs=45)

    print("\n======= Final Benchmark Summary =======")
    summarize_dicts(pooled_results, "POOLED")
    summarize_dicts(non_pooled_results, "NON-POOLED")
