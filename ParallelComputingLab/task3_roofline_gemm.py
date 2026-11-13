
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import math
import time
import statistics
import csv
import gc

from harness import emit_verify_tag


STUDENT_ID = 230103181

P_PHYS = 6
N = 256

MEMORY_SIZE = 64 * 1024 * 1024
COMPUTE_FLOPS_PER_WORKER = 60_000_000

BLOCK_SIZES = [8, 16, 32, 64, 96, 128]
RUNS = 3

# Shared globals for memory benchmark
_MEM_A = None
_MEM_B = None
_MEM_SHMS = []

# Shared globals for GEMM
_A = None
_B = None
_C = None
_GEMM_SHMS = []


# --------------------------------------------------
# PART 3A: MEMORY BANDWIDTH
# --------------------------------------------------

def init_memory_workers(name_a, name_b):
    global _MEM_A, _MEM_B, _MEM_SHMS

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)

    _MEM_SHMS = [shm_a, shm_b]
    _MEM_A = shm_a.buf
    _MEM_B = shm_b.buf


def memory_copy_worker(bounds):
    start, end = bounds
    _MEM_A[start:end] = _MEM_B[start:end]
    return end - start


def benchmark_bandwidth():
    print("\n=== PART 3A: DRAM BANDWIDTH ===")

    shm_a = shared_memory.SharedMemory(
        create=True, size=MEMORY_SIZE
    )
    shm_b = shared_memory.SharedMemory(
        create=True, size=MEMORY_SIZE
    )

    pool = None

    try:
        # Initialize source buffer
        shm_b.buf[:] = b"\x5A" * MEMORY_SIZE

        chunk = MEMORY_SIZE // P_PHYS

        bounds = [
            (i * chunk, (i + 1) * chunk)
            for i in range(P_PHYS)
        ]

        pool = mp.Pool(
            processes=P_PHYS,
            initializer=init_memory_workers,
            initargs=(shm_a.name, shm_b.name)
        )

        samples = []

        for run in range(RUNS):
            start = time.perf_counter_ns()

            pool.map(memory_copy_worker, bounds)

            elapsed_ms = (
                time.perf_counter_ns() - start
            ) / 1e6

            samples.append(elapsed_ms)

            print(
                f"Memory run {run + 1}: "
                f"{elapsed_ms:.3f} ms"
            )

        median_ms = statistics.median(samples)

        # Read + write traffic
        bandwidth = (
            2 * 64 * 1_000_000
        ) / (median_ms * 1_000_000)

        print(f"Median time: {median_ms:.3f} ms")
        print(f"Sustained bandwidth: {bandwidth:.3f} GB/s")

        return median_ms, bandwidth

    finally:
        if pool is not None:
            pool.close()
            pool.join()

        shm_a.close()
        shm_a.unlink()

        shm_b.close()
        shm_b.unlink()


# --------------------------------------------------
# PART 3A: COMPUTE CEILING
# --------------------------------------------------

def compute_worker(worker_id):
    # 16 independent accumulators
    a = [1.0 + i * 0.01 for i in range(16)]
    b = [1.0000001 + i * 0.001 for i in range(16)]
    c = [0.000001 * (i + 1) for i in range(16)]

    fma = getattr(math, "fma", None)

    iterations = COMPUTE_FLOPS_PER_WORKER // 32

    for _ in range(iterations):
        if fma is not None:
            for j in range(16):
                a[j] = fma(a[j], b[j], c[j])
        else:
            for j in range(16):
                a[j] = a[j] * b[j] + c[j]

    return sum(a)


def benchmark_compute():
    print("\n=== PART 3A: COMPUTE CEILING ===")

    pool = mp.Pool(processes=P_PHYS)

    try:
        samples = []

        for run in range(RUNS):
            start = time.perf_counter_ns()

            results = pool.map(
                compute_worker,
                range(P_PHYS)
            )

            elapsed_ms = (
                time.perf_counter_ns() - start
            ) / 1e6

            samples.append(elapsed_ms)

            print(
                f"Compute run {run + 1}: "
                f"{elapsed_ms:.3f} ms"
            )

        median_ms = statistics.median(samples)

        total_flops = (
            COMPUTE_FLOPS_PER_WORKER * P_PHYS
        )

        gflops = total_flops / (
            median_ms * 1_000_000
        )

        print(f"Median time: {median_ms:.3f} ms")
        print(f"Compute ceiling: {gflops:.3f} GFLOP/s")
        print(f"Checksum: {sum(results):.8f}")

        return median_ms, gflops

    finally:
        pool.close()
        pool.join()


# --------------------------------------------------
# PART 3B: GEMM INITIALIZATION
# --------------------------------------------------

def init_gemm_workers(name_a, name_b, name_c):
    global _A, _B, _C, _GEMM_SHMS

    shm_a = shared_memory.SharedMemory(name=name_a)
    shm_b = shared_memory.SharedMemory(name=name_b)
    shm_c = shared_memory.SharedMemory(name=name_c)

    _GEMM_SHMS = [shm_a, shm_b, shm_c]

    count = N * N

    _A = (ctypes.c_double * count).from_buffer(shm_a.buf)
    _B = (ctypes.c_double * count).from_buffer(shm_b.buf)
    _C = (ctypes.c_double * count).from_buffer(shm_c.buf)


def gemm_worker(args):
    variant, row_start, row_end, bs = args

    # Variant 1: Naive i-j-k
    if variant == 1:

        for i in range(row_start, row_end):
            for j in range(N):
                acc = 0.0

                for k in range(N):
                    acc += _A[i * N + k] * _B[k * N + j]

                _C[i * N + j] = acc

    # Variant 2: Reordered i-k-j
    elif variant == 2:

        for i in range(row_start, row_end):
            row = i * N

            for k in range(N):
                aik = _A[row + k]
                b_row = k * N

                for j in range(N):
                    idx = row + j
                    _C[idx] += aik * _B[b_row + j]

    # Variant 3: Cache-blocked i-k-j
    elif variant == 3:

        for ii in range(row_start, row_end, bs):
            i_end = min(ii + bs, row_end)

            for kk in range(0, N, bs):
                k_end = min(kk + bs, N)

                for jj in range(0, N, bs):
                    j_end = min(jj + bs, N)

                    for i in range(ii, i_end):
                        row = i * N

                        for k in range(kk, k_end):
                            aik = _A[row + k]
                            b_row = k * N

                            for j in range(jj, j_end):
                                idx = row + j

                                _C[idx] += (
                                    aik * _B[b_row + j]
                                )

    return row_end - row_start


# --------------------------------------------------
# PART 3B: GEMM BENCHMARK
# --------------------------------------------------

def benchmark_gemm(pool, c_shm, variant, bs=32):

    rows = [
        (
            variant,
            i * N // P_PHYS,
            (i + 1) * N // P_PHYS,
            bs
        )
        for i in range(P_PHYS)
    ]

    samples = []

    zero_buffer = b"\x00" * (N * N * 8)

    for _ in range(RUNS):

        # Reset C outside timed section
        c_shm.buf[:] = zero_buffer

        start = time.perf_counter_ns()

        pool.map(gemm_worker, rows)

        elapsed_ms = (
            time.perf_counter_ns() - start
        ) / 1e6

        samples.append(elapsed_ms)

    median_ms = statistics.median(samples)

    gflops = (
        2 * N ** 3
    ) / (median_ms * 1_000_000)

    return median_ms, gflops


def main():
    print("=" * 55)
    print("PHASE 3: EMPIRICAL ROOFLINE & CACHE-BLOCKED GEMM")
    print("=" * 55)

    print(f"Student ID: {STUDENT_ID}")
    print(f"CPU: AMD Ryzen 5 5600H")
    print(f"Physical workers: {P_PHYS}")
    print(f"Matrix size: {N} x {N}")

    # Part 3A
    bandwidth_ms, bandwidth = benchmark_bandwidth()

    compute_ms, compute_gflops = benchmark_compute()

    ridge_point = compute_gflops / bandwidth

    print("\n--- ROOFLINE CEILINGS ---")
    print(f"Bandwidth: {bandwidth:.3f} GB/s")
    print(f"Compute ceiling: {compute_gflops:.3f} GFLOP/s")
    print(f"Ridge point: {ridge_point:.3f} FLOP/byte")

    # Part 3B: Allocate shared matrices
    count = N * N
    size = count * 8

    shm_a = shared_memory.SharedMemory(
        create=True, size=size
    )
    shm_b = shared_memory.SharedMemory(
        create=True, size=size
    )
    shm_c = shared_memory.SharedMemory(
        create=True, size=size
    )

    arr_a = (ctypes.c_double * count).from_buffer(shm_a.buf)
    arr_b = (ctypes.c_double * count).from_buffer(shm_b.buf)

    for i in range(N):
        for j in range(N):
            arr_a[i * N + j] = (
                ((i * 17 + j * 13) % 101) / 101.0
            )

            arr_b[i * N + j] = (
                ((i * 7 + j * 19) % 97) / 97.0
            )

    pool = None

    try:
        pool = mp.Pool(
            processes=P_PHYS,
            initializer=init_gemm_workers,
            initargs=(shm_a.name, shm_b.name, shm_c.name)
        )

        print("\n=== PART 3B: GEMM VARIANTS ===")

        results = {}

        for variant in [1, 2]:
            median_ms, gflops = benchmark_gemm(
                pool, shm_c, variant
            )

            results[variant] = (median_ms, gflops)

            print(
                f"Variant {variant}: "
                f"Median={median_ms:.3f} ms | "
                f"GFLOP/s={gflops:.5f}"
            )

        # Save Variant 2 result for verification
        arr_c = (
            ctypes.c_double * count
        ).from_buffer(shm_c.buf)

        reference = list(arr_c)

        print("\n=== CACHE TILE SWEEP ===")

        tile_results = []

        for bs in BLOCK_SIZES:
            median_ms, gflops = benchmark_gemm(
                pool, shm_c, 3, bs
            )

            tile_results.append((bs, median_ms, gflops))

            print(
                f"BS={bs:3d} | "
                f"Median={median_ms:.3f} ms | "
                f"GFLOP/s={gflops:.5f}"
            )

        # Best tile by measured GFLOP/s
        best_bs, best_ms, best_gflops = max(
            tile_results,
            key=lambda x: x[2]
        )

        blocked_result = list(arr_c)

        max_error = max(
            abs(reference[i] - blocked_result[i])
            for i in range(count)
        )

        print("\n=== NUMERICAL VERIFICATION ===")
        print(f"Maximum absolute error: {max_error:.12e}")
        print(f"Required error: < 1e-9")
        print(f"Verification passed: {max_error < 1e-9}")

        print("\n=== OPTIMAL TILE ===")
        print(f"Best block size: {best_bs}")
        print(
            f"Tile footprint: "
            f"{3 * best_bs ** 2 * 8 / 1024:.2f} KB"
        )
        print(f"Operational intensity: {best_bs / 12:.3f} FLOP/byte")

        print("\n=== ROOFLINE MAPPING ===")

        print(
            f"Variant 1: {results[1][1]:.5f} GFLOP/s"
        )
        print(
            f"Variant 2: {results[2][1]:.5f} GFLOP/s"
        )
        print(
            f"Variant 3: {best_gflops:.5f} GFLOP/s"
        )

        # Save CSV
        with open(
            "phase3_results.csv",
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Variant",
                "BlockSize",
                "Median_ms",
                "GFLOPS"
            ])

            for variant in [1, 2]:
                writer.writerow([
                    variant,
                    "N/A",
                    results[variant][0],
                    results[variant][1]
                ])

            for bs, median, gflops in tile_results:
                writer.writerow([
                    3,
                    bs,
                    median,
                    gflops
                ])

        print("\nResults saved to phase3_results.csv")

        print("\n=== VERIFICATION TAG ===")

        print(
            emit_verify_tag(
                "PHASE3",
                STUDENT_ID,
                results[1][1],
                best_gflops
            )
        )

    finally:
        if pool is not None:
            pool.close()
            pool.join()

        del arr_a
        del arr_b

        if "arr_c" in locals():
            del arr_c

        gc.collect()

        for shm in [shm_a, shm_b, shm_c]:
            shm.close()
            shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
