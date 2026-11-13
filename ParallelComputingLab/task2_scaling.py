
import multiprocessing as mp
import math
import time
import csv

from harness import emit_verify_tag


STUDENT_ID = 230103181

P_PHYS = 6
P_LOG = mp.cpu_count()

N_STRONG = 1_200_000
N0 = 200_000

WARMUP = 1
RUNS = 5


def fma(a, b, c):
    if hasattr(math, "fma"):
        return math.fma(a, b, c)
    return a * b + c


# Stage A: 10 sin/sqrt operations per element
def stage_a(n):
    last = 0.0

    for i in range(n):
        x = (i + 1) * 0.001

        for _ in range(5):
            x = math.sin(x)
            x = math.sqrt(abs(x) + 1e-12)

        last = x

    return last


# Stage B: 30 FMA/trigonometric operations per element
def stage_b_worker(args):
    start, end = args
    last = 0.0

    for i in range(start, end):
        x = (i % 997 + 1) * 0.001

        for _ in range(10):
            x = math.sin(x)
            y = math.cos(x + 0.01)
            x = fma(x, y, 0.000001)

        last = x

    return last


def make_ranges(n, p):
    return [
        (i * n // p, (i + 1) * n // p)
        for i in range(p)
    ]


def run_trial(pool, n, p):
    start = time.perf_counter_ns()

    # Serial Stage A
    stage_a_n = int(0.05 * n)

    a_start = time.perf_counter_ns()
    checksum_a = stage_a(stage_a_n)
    stage_a_ms = (
        time.perf_counter_ns() - a_start
    ) / 1e6

    # Parallel Stage B
    ranges = make_ranges(n, p)
    results = pool.map(stage_b_worker, ranges, chunksize=1)

    checksum = checksum_a + sum(results)

    total_ms = (
        time.perf_counter_ns() - start
    ) / 1e6

    return total_ms, stage_a_ms, checksum


def benchmark(n, p):
    total_samples = []
    stage_a_samples = []

    # Persistent pool: created outside timed trials
    with mp.Pool(processes=p) as pool:

        for _ in range(WARMUP):
            run_trial(pool, n, p)

        for _ in range(RUNS):
            total_ms, a_ms, _ = run_trial(pool, n, p)

            total_samples.append(total_ms)
            stage_a_samples.append(a_ms)

    total_samples.sort()
    stage_a_samples.sort()

    return (
        total_samples[len(total_samples) // 2],
        stage_a_samples[len(stage_a_samples) // 2]
    )


def main():
    print("=== PHASE 2: STRONG AND WEAK SCALING ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores: {P_PHYS}")
    print(f"Logical CPUs: {P_LOG}")
    print(f"Strong N: {N_STRONG:,}")
    print(f"Weak N0: {N0:,}")

    strong = {}
    stage_a_strong = {}

    print("\n--- STRONG SCALING ---")
    print("p | Total ms | Stage A ms | Speedup | Efficiency")

    for p in range(1, P_LOG + 1):
        total, a_time = benchmark(N_STRONG, p)

        strong[p] = total
        stage_a_strong[p] = a_time

        speedup = strong[1] / total if p == 1 else strong[1] / total
        efficiency = speedup / p

        print(
            f"{p:2d} | {total:10.3f} | "
            f"{a_time:10.3f} | "
            f"{speedup:7.3f} | {efficiency:10.3f}"
        )

    serial_fraction = stage_a_strong[1] / strong[1]
    amdahl_cap = 1 / serial_fraction

    print(f"\nSerial fraction: {serial_fraction:.5f}")
    print(f"Amdahl theoretical cap: {amdahl_cap:.3f}")
    print(
        f"Observed strong speedup at {P_LOG} processes: "
        f"{strong[P_LOG] and strong[1] / strong[P_LOG]:.3f}"
    )

    weak = {}
    stage_a_weak = {}
    weak_efficiency = {}
    gustafson = {}

    print("\n--- WEAK SCALING ---")
    print("p | N | Total ms | Stage A ms | Efficiency | Gustafson")

    for p in range(1, P_LOG + 1):
        n = p * N0

        total, a_time = benchmark(n, p)

        weak[p] = total
        stage_a_weak[p] = a_time

        efficiency = weak[1] / total
        alpha = a_time / total

        gustafson_speedup = p - alpha * (p - 1)

        weak_efficiency[p] = efficiency
        gustafson[p] = gustafson_speedup

        print(
            f"{p:2d} | {n:9,} | {total:10.3f} | "
            f"{a_time:10.3f} | {efficiency:10.3f} | "
            f"{gustafson_speedup:10.3f}"
        )

    # Architectural inflection: 6 to 7 processes
    s5 = strong[1] / strong[5]
    s6 = strong[1] / strong[6]
    s7 = strong[1] / strong[7]

    gain_6 = s6 - s5
    gain_7 = s7 - s6

    print("\n--- ARCHITECTURAL INFLECTION ---")
    print(f"Speedup at 5 processes: {s5:.3f}")
    print(f"Speedup at 6 processes: {s6:.3f}")
    print(f"Speedup at 7 processes: {s7:.3f}")
    print(f"Marginal gain 5 -> 6: {gain_6:.3f}")
    print(f"Marginal gain 6 -> 7: {gain_7:.3f}")
    print(f"Marginal gain change: {gain_7 - gain_6:.3f}")

    print("\n--- VERIFICATION ---")

    print(
        emit_verify_tag(
            "PHASE2",
            STUDENT_ID,
            s7 if P_LOG == 7 else strong[1] / strong[P_LOG],
            gustafson[P_LOG]
        )
    )

    # Save results
    with open("phase2_results.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "p", "Strong_ms", "Strong_StageA_ms",
            "Strong_Speedup", "Strong_Efficiency",
            "Weak_N", "Weak_ms", "Weak_StageA_ms",
            "Weak_Efficiency", "Gustafson"
        ])

        for p in range(1, P_LOG + 1):
            writer.writerow([
                p,
                strong[p],
                stage_a_strong[p],
                strong[1] / strong[p],
                (strong[1] / strong[p]) / p,
                p * N0,
                weak[p],
                stage_a_weak[p],
                weak_efficiency[p],
                gustafson[p]
            ])

    print("\nResults saved to phase2_results.csv")


if __name__ == "__main__":
    mp.freeze_support()
    main()
