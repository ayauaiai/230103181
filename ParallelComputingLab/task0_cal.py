
import multiprocessing as mp
import time

from harness import emit_verify_tag


STUDENT_ID = 230103181
N = 1_000_000


def main():
    # Check logical CPU count
    logical_cpus = mp.cpu_count()

    print("=== PHASE 0: CALIBRATION ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical CPUs: {logical_cpus}")

    # Measure timer call latency
    start = time.perf_counter_ns()

    for _ in range(N):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_ns = end - start
    average_ns = total_ns / N
    total_ms = total_ns / 1_000_000

    print(f"Number of calls: {N:,}")
    print(f"Average call latency: {average_ns:.2f} ns")
    print(f"Total measurement time: {total_ms:.4f} ms")

    # Generate verification tag
    tag = emit_verify_tag(
        "PHASE0",
        STUDENT_ID,
        average_ns,
        total_ms
    )

    print("\n" + tag)


if __name__ == "__main__":
    main()
