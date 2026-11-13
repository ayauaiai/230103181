
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import gc

from harness import run_benchmark, emit_verify_tag


STUDENT_ID = 230103181

M = 16_000_000
STRIDES = [1, 2, 4, 8, 16, 32]

P_PHYS = 6
P_LOG = mp.cpu_count()

MASK = 0xFFFFFFFFFFFFFFFF
MULTIPLIER = 6364136223846793005


def fs_worker(shm_name, slot_idx, iters):
    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]

    for i in range(iters):
        acc = (
            acc * MULTIPLIER + (i | 1)
        ) & MASK

        arr[slot_idx] = acc

    del arr
    shm.close()


def run_false_sharing(p, stride):
    shm = shared_memory.SharedMemory(create=True, size=4096)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    for i in range(512):
        arr[i] = 1

    del arr

    iters = M // p

    processes = []

    try:
        for worker_id in range(p):
            slot_idx = worker_id * stride

            process = mp.Process(
                target=fs_worker,
                args=(shm.name, slot_idx, iters)
            )

            processes.append(process)

        start = time.perf_counter_ns()

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        elapsed_ms = (
            time.perf_counter_ns() - start
        ) / 1e6

        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError("A worker process failed")

        return elapsed_ms

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

        shm.close()
        shm.unlink()


def benchmark_stride(p, stride):
    stats = run_benchmark(
        lambda: run_false_sharing(p, stride),
        warmup=2,
        runs=7
    )

    throughput = M / stats.median_ms / 1000

    return stats, throughput


def main():
    print("=== PHASE 1: FALSE SHARING ===")
    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores: {P_PHYS}")
    print(f"Logical CPUs: {P_LOG}")
    print(f"Operations: {M:,}")

    print("\n--- STRIDE SWEEP (p=6) ---")

    results = {}

    for stride in STRIDES:
        stats, throughput = benchmark_stride(P_PHYS, stride)

        results[stride] = stats.median_ms

        print(
            f"Stride={stride:2d} | "
            f"Offset={stride * 8:3d} bytes | "
            f"Median={stats.median_ms:.3f} ms | "
            f"IQR={stats.iqr_ms:.3f} ms | "
            f"Throughput={throughput:.3f} Mops/s"
        )

    print("\n--- PROCESS SCALING ---")

    process_counts = [1, 2, 4, 8, 12]
    scaling_results = {}

    for stride in [1, 16]:
        print(f"\nStride = {stride}")

        scaling_results[stride] = {}

        for p in process_counts:
            if p > P_LOG:
                continue

            stats, throughput = benchmark_stride(p, stride)

            scaling_results[stride][p] = stats.median_ms

            t1 = scaling_results[stride].get(1)

            if t1 is not None:
                speedup = t1 / stats.median_ms
                efficiency = speedup / p

                if p >= 2:
                    karp_flatt = (
                        (1 / speedup) - (1 / p)
                    ) / (1 - (1 / p))
                else:
                    karp_flatt = None

                kf_text = (
                    f"{karp_flatt:.4f}"
                    if karp_flatt is not None
                    else "N/A"
                )

                print(
                    f"p={p:2d} | "
                    f"Median={stats.median_ms:.3f} ms | "
                    f"Speedup={speedup:.3f} | "
                    f"Efficiency={efficiency:.3f} | "
                    f"Karp-Flatt={kf_text}"
                )

    print("\n--- VERIFICATION ---")

    print(
        emit_verify_tag(
            "PHASE1",
            STUDENT_ID,
            results[1],
            results[16]
        )
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
