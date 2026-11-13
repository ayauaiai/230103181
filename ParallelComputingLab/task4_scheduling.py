
import multiprocessing as mp
import math
import time
import statistics
import csv

from harness import emit_verify_tag


STUDENT_ID = 230103181

N = 8000
P_PHYS = 6
CHUNK_SIZES = [1, 8, 64, 512]
RUNS = 3


# --------------------------------------------------
# WORKLOAD
# --------------------------------------------------

def workload(i):
    """
    K(i) = 1 + floor((i / 40)^2)
    Each iteration performs sin, cos and sqrt.
    """
    k = 1 + (i // 40) ** 2

    x = (i % 100 + 1) * 0.01

    for _ in range(k):
        x = (
            math.sin(x) * math.cos(x)
            + math.sqrt(abs(x) + 1e-12) * 0.5
        )

    return x


# --------------------------------------------------
# WORKER
# --------------------------------------------------

def scheduling_worker(
    worker_id,
    mode,
    worker_times,
    checksums,
    counter=None,
    chunk_size=1
):
    start_time = time.perf_counter()

    checksum = 0.0

    # Strategy 1: Contiguous Block Static
    if mode == "block":
        start = worker_id * N // P_PHYS
        end = (worker_id + 1) * N // P_PHYS

        for i in range(start, end):
            checksum += workload(i)

    # Strategy 2: Cyclic Round-Robin Static
    elif mode == "cyclic":
        for i in range(worker_id, N, P_PHYS):
            checksum += workload(i)

    # Strategy 3: Shared-Memory Dynamic Queue
    elif mode == "dynamic":
        while True:
            with counter.get_lock():
                start = counter.value
                counter.value += chunk_size

            if start >= N:
                break

            end = min(start + chunk_size, N)

            for i in range(start, end):
                checksum += workload(i)

    worker_times[worker_id] = (
        time.perf_counter() - start_time
    ) * 1000

    checksums[worker_id] = checksum


# --------------------------------------------------
# SINGLE TRIAL
# --------------------------------------------------

def run_trial(mode, chunk_size=1):

    worker_times = mp.Array(
        'd', P_PHYS, lock=False
    )

    checksums = mp.Array(
        'd', P_PHYS, lock=False
    )

    counter = (
        mp.Value('i', 0)
        if mode == "dynamic"
        else None
    )

    processes = []

    for worker_id in range(P_PHYS):
        process = mp.Process(
            target=scheduling_worker,
            args=(
                worker_id,
                mode,
                worker_times,
                checksums,
                counter,
                chunk_size
            )
        )

        processes.append(process)

    wall_start = time.perf_counter()

    try:
        for process in processes:
            process.start()

        for process in processes:
            process.join()

        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError(
                f"Worker failed in {mode} scheduling"
            )

        wall_ms = (
            time.perf_counter() - wall_start
        ) * 1000

        times = list(worker_times)
        checksum = sum(checksums)

        return wall_ms, times, checksum

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()


# --------------------------------------------------
# BENCHMARK
# --------------------------------------------------

def benchmark_strategy(mode, chunk_size=1):

    samples = []

    for run in range(RUNS):
        wall_ms, worker_times, checksum = run_trial(
            mode, chunk_size
        )

        samples.append({
            "wall": wall_ms,
            "workers": worker_times,
            "checksum": checksum
        })

        print(
            f"  Run {run + 1}: "
            f"Twall={wall_ms:.3f} ms"
        )

    median_wall = statistics.median(
        item["wall"] for item in samples
    )

    # Select worker telemetry from the trial
    # whose wall time is closest to the median.
    representative = min(
        samples,
        key=lambda item: abs(
            item["wall"] - median_wall
        )
    )

    times = representative["workers"]

    average_worker = statistics.mean(times)

    imbalance = (
        max(times) / average_worker
        if average_worker > 0
        else 0
    )

    return {
        "median_wall": median_wall,
        "worker_times": times,
        "imbalance": imbalance,
        "checksum": representative["checksum"]
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 55)
    print("PHASE 4: SKEWED WORKLOADS & DYNAMIC CHUNK QUEUE")
    print("=" * 55)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Number of items: {N}")
    print(f"Physical workers: {P_PHYS}")
    print(f"Runs per strategy: {RUNS}")

    results = {}

    # Strategy 1: Block Static
    print("\n=== 1. CONTIGUOUS BLOCK STATIC ===")

    block = benchmark_strategy("block")

    results["block"] = block

    print(
        f"Median Twall: "
        f"{block['median_wall']:.3f} ms"
    )

    print(
        f"Load Imbalance Factor: "
        f"{block['imbalance']:.4f}"
    )

    print(
        "Worker times (ms):",
        [round(x, 3) for x in block["worker_times"]]
    )

    # Strategy 2: Cyclic Static
    print("\n=== 2. CYCLIC ROUND-ROBIN STATIC ===")

    cyclic = benchmark_strategy("cyclic")

    results["cyclic"] = cyclic

    print(
        f"Median Twall: "
        f"{cyclic['median_wall']:.3f} ms"
    )

    print(
        f"Load Imbalance Factor: "
        f"{cyclic['imbalance']:.4f}"
    )

    print(
        "Worker times (ms):",
        [round(x, 3) for x in cyclic["worker_times"]]
    )

    # Strategy 3: Dynamic Queue
    print("\n=== 3. DYNAMIC QUEUE CHUNK SWEEP ===")

    dynamic_results = {}

    for chunk in CHUNK_SIZES:

        print(f"\n--- Chunk size C={chunk} ---")

        result = benchmark_strategy(
            "dynamic",
            chunk_size=chunk
        )

        dynamic_results[chunk] = result

        print(
            f"Median Twall: "
            f"{result['median_wall']:.3f} ms"
        )

        print(
            f"Load Imbalance Factor: "
            f"{result['imbalance']:.4f}"
        )

        print(
            "Worker times (ms):",
            [round(x, 3) for x in result["worker_times"]]
        )

    # Select optimal chunk by minimum median wall time
    best_chunk = min(
        dynamic_results,
        key=lambda c: dynamic_results[c]["median_wall"]
    )

    best_dynamic = dynamic_results[best_chunk]

    print("\n=== COMPARISON ===")

    print(
        f"Block Static: "
        f"{block['median_wall']:.3f} ms"
    )

    print(
        f"Cyclic Static: "
        f"{cyclic['median_wall']:.3f} ms"
    )

    for chunk in CHUNK_SIZES:
        print(
            f"Dynamic C={chunk}: "
            f"{dynamic_results[chunk]['median_wall']:.3f} ms"
        )

    print(f"\nOptimal Dynamic Chunk: C*={best_chunk}")

    # Save CSV
    with open(
        "phase4_results.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Strategy",
            "Chunk",
            "Median_Twall_ms",
            "Load_Imbalance",
            "Worker_Times_ms",
            "Checksum"
        ])

        for name, result in [
            ("Block Static", block),
            ("Cyclic Static", cyclic)
        ]:
            writer.writerow([
                name,
                "N/A",
                result["median_wall"],
                result["imbalance"],
                result["worker_times"],
                result["checksum"]
            ])

        for chunk, result in dynamic_results.items():
            writer.writerow([
                "Dynamic Queue",
                chunk,
                result["median_wall"],
                result["imbalance"],
                result["worker_times"],
                result["checksum"]
            ])

    print("\nResults saved to phase4_results.csv")

    # Verification tag
    print("\n=== VERIFICATION ===")

    print(
        emit_verify_tag(
            "PHASE4",
            STUDENT_ID,
            block["median_wall"],
            best_dynamic["median_wall"]
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
