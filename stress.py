#!/usr/bin/env python3

"""
COLLATZ CONJECTURE CPU/GPU STRESS TESTER
========================================

Interactive Collatz computation designed primarily as a hardware
stress test.

Features:
    - Interactive configuration
    - Uses all CPU cores by default
    - Uses multiprocessing rather than Python threads for CPU work
    - Optional NVIDIA CUDA GPU acceleration
    - CPU + GPU can run simultaneously
    - Continuous "forever" mode
    - Configurable starting number
    - Configurable end number
    - Configurable maximum steps
    - Configurable CPU worker count
    - Configurable GPU batch size
    - Live calculations/second statistics
    - Tracks highest stopping time
    - Tracks largest value reached
    - Detects uint64 overflow on GPU
    - Saves suspicious results to a text file

IMPORTANT:
    Testing numbers does NOT prove the Collatz conjecture for infinity.

    "MAX STEPS REACHED" means the program stopped because the configured
    limit was reached. It does NOT mean a counterexample was found.

CPU:
    Python multiprocessing + arbitrary precision integers.

GPU:
    NVIDIA CUDA + Numba + uint64 arithmetic.

The GPU cannot safely represent arbitrarily large integers. If a GPU
trajectory would exceed uint64, it is reported as OVERFLOW rather than
being allowed to wrap around.

Install:

    pip install numpy numba

For GPU mode, you also need a compatible NVIDIA driver/CUDA setup.
"""


import os
import sys
import time
import math
import signal
import threading
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support


# ============================================================
# OPTIONAL NUMPY / CUDA
# ============================================================

try:
    import numpy as np
except ImportError:
    np = None


GPU_AVAILABLE = False
cuda = None

try:
    from numba import cuda

    GPU_AVAILABLE = cuda.is_available()

except Exception:
    GPU_AVAILABLE = False


# ============================================================
# GLOBAL SETTINGS
# ============================================================

MAX_UINT64 = 18_446_744_073_709_551_615

# Largest n for which 3*n + 1 fits in uint64.
MAX_SAFE_ODD_UINT64 = MAX_UINT64 // 3


# ============================================================
# CPU COLLATZ WORKER
# ============================================================

def cpu_collatz_chunk(start, end, max_steps):
    """
    Calculate a complete range of Collatz sequences.

    This function runs inside a separate process.

    It intentionally does NOT use memoization because this program
    is intended as a stress test.

    Returns:

        (
            numbers_tested,
            suspicious_results,
            local_max_steps,
            local_max_peak,
            local_max_step_number
        )
    """

    numbers_tested = 0

    suspicious = []

    local_max_steps = 0
    local_max_peak = 0
    local_max_step_number = start

    for original in range(start, end + 1):

        n = original
        steps = 0
        peak = n

        while n != 1 and steps < max_steps:

            if n > peak:
                peak = n

            # Standard Collatz operation.
            if n & 1:
                n = 3 * n + 1
            else:
                n //= 2

            steps += 1

        numbers_tested += 1

        # Statistics.
        if steps > local_max_steps:
            local_max_steps = steps
            local_max_step_number = original

        if peak > local_max_peak:
            local_max_peak = peak

        # Normally this means we reached the step limit.
        if n != 1:

            suspicious.append(
                (
                    original,
                    "MAX_STEPS_REACHED",
                    steps,
                    peak
                )
            )

    return (
        numbers_tested,
        suspicious,
        local_max_steps,
        local_max_peak,
        local_max_step_number
    )


# ============================================================
# GPU KERNEL
# ============================================================

if GPU_AVAILABLE:

    @cuda.jit
    def collatz_kernel(
        start,
        max_steps,
        status,
        steps_out,
        peak_out
    ):
        """
        One CUDA thread = one starting number.

        status:

            0 = reached 1
            1 = maximum steps reached
            2 = uint64 overflow

        Because GPU arithmetic is uint64, overflow is explicitly
        detected before 3*n + 1.
        """

        idx = cuda.grid(1)

        if idx >= status.size:
            return

        n = start + idx

        steps = 0
        peak = n

        while n != 1 and steps < max_steps:

            if n > peak:
                peak = n

            # Odd number:
            #
            # n -> 3n + 1
            #
            # Check overflow before doing the multiplication.
            if n & 1:

                if n > MAX_SAFE_ODD_UINT64:
                    status[idx] = 2
                    steps_out[idx] = steps
                    peak_out[idx] = peak
                    return

                n = 3 * n + 1

            else:

                n //= 2

            steps += 1

        steps_out[idx] = steps
        peak_out[idx] = peak

        if n == 1:
            status[idx] = 0
        else:
            status[idx] = 1


# ============================================================
# GPU BATCH
# ============================================================

def gpu_collatz_batch(start, count, max_steps):

    if not GPU_AVAILABLE:
        raise RuntimeError(
            "CUDA GPU is not available."
        )

    if np is None:
        raise RuntimeError(
            "NumPy is required for GPU mode."
        )

    # Allocate host arrays.
    status = np.zeros(
        count,
        dtype=np.uint8
    )

    steps = np.zeros(
        count,
        dtype=np.uint64
    )

    peaks = np.zeros(
        count,
        dtype=np.uint64
    )

    # Send arrays to GPU.
    d_status = cuda.to_device(status)
    d_steps = cuda.to_device(steps)
    d_peaks = cuda.to_device(peaks)

    threads_per_block = 256

    blocks = (
        count + threads_per_block - 1
    ) // threads_per_block

    # Launch.
    collatz_kernel[
        blocks,
        threads_per_block
    ](
        np.uint64(start),
        np.uint64(max_steps),
        d_status,
        d_steps,
        d_peaks
    )

    cuda.synchronize()

    # Copy results back.
    status = d_status.copy_to_host()
    steps = d_steps.copy_to_host()
    peaks = d_peaks.copy_to_host()

    suspicious = []

    max_steps_seen = 0
    max_peak_seen = 0
    max_step_number = start

    for i in range(count):

        number = start + i

        current_steps = int(steps[i])
        current_peak = int(peaks[i])

        if current_steps > max_steps_seen:
            max_steps_seen = current_steps
            max_step_number = number

        if current_peak > max_peak_seen:
            max_peak_seen = current_peak

        if status[i] == 1:

            suspicious.append(
                (
                    number,
                    "MAX_STEPS_REACHED",
                    current_steps,
                    current_peak
                )
            )

        elif status[i] == 2:

            suspicious.append(
                (
                    number,
                    "UINT64_OVERFLOW",
                    current_steps,
                    current_peak
                )
            )

    return (
        count,
        suspicious,
        max_steps_seen,
        max_peak_seen,
        max_step_number
    )


# ============================================================
# STATISTICS
# ============================================================

class Statistics:

    def __init__(self):

        self.lock = threading.Lock()

        self.total_numbers = 0

        self.total_suspicious = 0

        self.max_steps = 0

        self.max_steps_number = 0

        self.max_peak = 0

        self.start_time = time.perf_counter()

        self.last_print = self.start_time

        self.last_numbers = 0

    def update(
        self,
        numbers,
        suspicious,
        max_steps,
        max_peak,
        max_steps_number
    ):

        with self.lock:

            self.total_numbers += numbers

            self.total_suspicious += len(
                suspicious
            )

            if max_steps > self.max_steps:

                self.max_steps = max_steps

                self.max_steps_number = (
                    max_steps_number
                )

            if max_peak > self.max_peak:

                self.max_peak = max_peak

    def print_status(self, force=False):

        now = time.perf_counter()

        with self.lock:

            elapsed = now - self.start_time

            if elapsed <= 0:
                return

            current_total = self.total_numbers

            rate = current_total / elapsed

            interval = now - self.last_print

            if not force and interval < 1.0:
                return

            interval_numbers = (
                current_total - self.last_numbers
            )

            interval_rate = (
                interval_numbers / interval
                if interval > 0
                else 0
            )

            self.last_print = now
            self.last_numbers = current_total

            print(
                "\n"
                "------------------------------------------------------------"
            )

            print(
                f"Numbers tested : {current_total:,}"
            )

            print(
                f"Overall rate   : {rate:,.0f} numbers/sec"
            )

            print(
                f"Current rate   : {interval_rate:,.0f} numbers/sec"
            )

            print(
                f"Max steps      : {self.max_steps:,}"
            )

            print(
                f"Number         : {self.max_steps_number:,}"
            )

            print(
                f"Largest value  : {self.max_peak:,}"
            )

            print(
                f"Suspicious     : {self.total_suspicious:,}"
            )

            print(
                f"Runtime        : {elapsed:,.1f} seconds"
            )

            print(
                "------------------------------------------------------------"
            )


# ============================================================
# SUSPICIOUS RESULT LOGGING
# ============================================================

def log_suspicious(results):

    if not results:
        return

    filename = "collatz_suspicious.txt"

    with open(
        filename,
        "a",
        encoding="utf-8"
    ) as file:

        for result in results:

            number, status, steps, peak = result

            file.write(
                f"{number} | "
                f"{status} | "
                f"steps={steps} | "
                f"peak={peak}\n"
            )

    print(
        f"\n[!] Suspicious results written to "
        f"{filename}"
    )


# ============================================================
# INPUT HELPERS
# ============================================================

def ask_int(
    prompt,
    default,
    minimum=None
):

    while True:

        raw = input(
            f"{prompt} [{default}]: "
        ).strip()

        if raw == "":
            value = default
        else:

            try:
                value = int(raw)

            except ValueError:
                print(
                    "Please enter a valid integer."
                )
                continue

        if minimum is not None and value < minimum:

            print(
                f"Value must be >= {minimum}."
            )
            continue

        return value


def ask_choice(prompt, choices, default):

    choices_text = "/".join(choices)

    while True:

        raw = input(
            f"{prompt} ({choices_text}) "
            f"[{default}]: "
        ).strip().lower()

        if raw == "":
            return default

        if raw in choices:
            return raw

        print(
            "Please choose one of:",
            ", ".join(choices)
        )


def ask_yes_no(prompt, default=True):

    default_text = "Y/n" if default else "y/N"

    while True:

        raw = input(
            f"{prompt} [{default_text}]: "
        ).strip().lower()

        if raw == "":
            return default

        if raw in ("y", "yes"):
            return True

        if raw in ("n", "no"):
            return False

        print("Please enter y or n.")


# ============================================================
# GPU WORKER THREAD
# ============================================================

def gpu_loop(
    next_range,
    next_range_lock,
    stats,
    max_steps,
    batch_size,
    stop_event
):
    """
    GPU worker.

    It receives its own ranges through the shared range allocator.

    CPU workers use the same allocator, so CPU and GPU never calculate
    the same number.
    """

    while not stop_event.is_set():

        with next_range_lock:

            start = next_range[0]

            next_range[0] += batch_size

        try:

            result = gpu_collatz_batch(
                start,
                batch_size,
                max_steps
            )

        except Exception as exc:

            print(
                f"\nGPU ERROR: {exc}"
            )

            stop_event.set()
            return

        (
            count,
            suspicious,
            max_steps_seen,
            max_peak,
            max_step_number
        ) = result

        stats.update(
            count,
            suspicious,
            max_steps_seen,
            max_peak,
            max_step_number
        )

        if suspicious:

            log_suspicious(
                suspicious
            )

            print(
                "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

            print(
                "GPU SUSPICIOUS RESULT"
            )

            print(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

            stop_event.set()

            return


# ============================================================
# CPU WORKER THREAD
# ============================================================

def cpu_loop(
    executor,
    next_range,
    next_range_lock,
    stats,
    max_steps,
    chunk_size,
    stop_event,
    worker_count
):

    futures = set()

    def submit_chunk():

        with next_range_lock:

            start = next_range[0]

            next_range[0] += chunk_size

        future = executor.submit(
            cpu_collatz_chunk,
            start,
            start + chunk_size - 1,
            max_steps
        )

        futures.add(future)

    # Keep many chunks in flight so CPU cores remain busy.
    for _ in range(worker_count * 2):

        submit_chunk()

    while not stop_event.is_set():

        completed = []

        for future in futures:

            if future.done():
                completed.append(future)

        if not completed:

            time.sleep(0.01)
            continue

        for future in completed:

            futures.remove(future)

            try:

                result = future.result()

            except Exception as exc:

                print(
                    f"\nCPU WORKER ERROR: {exc}"
                )

                stop_event.set()
                return

            (
                count,
                suspicious,
                max_steps_seen,
                max_peak,
                max_step_number
            ) = result

            stats.update(
                count,
                suspicious,
                max_steps_seen,
                max_peak,
                max_step_number
            )

            if suspicious:

                log_suspicious(
                    suspicious
                )

                print(
                    "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )

                print(
                    "CPU SUSPICIOUS RESULT"
                )

                print(
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                )

                stop_event.set()

                return

            submit_chunk()


# ============================================================
# STATUS DISPLAY THREAD
# ============================================================

def status_loop(stats, stop_event):

    while not stop_event.is_set():

        stats.print_status()

        time.sleep(1)

    stats.print_status(
        force=True
    )


# ============================================================
# MAIN STRESS TEST
# ============================================================

def run_stress_test():

    print()
    print("=" * 64)
    print("              COLLATZ STRESS TESTER")
    print("=" * 64)
    print()

    cpu_count = os.cpu_count() or 1

    print(
        f"Detected CPU logical processors: {cpu_count}"
    )

    if GPU_AVAILABLE:

        print(
            "NVIDIA CUDA GPU: DETECTED"
        )

    else:

        print(
            "NVIDIA CUDA GPU: NOT DETECTED"
        )

    print()

    # --------------------------------------------------------
    # BACKEND
    # --------------------------------------------------------

    if GPU_AVAILABLE:

        backend = ask_choice(
            "Use CPU, GPU, or BOTH?",
            ["cpu", "gpu", "both"],
            "both"
        )

    else:

        print(
            "GPU unavailable, so CPU will be used."
        )

        backend = "cpu"

    print()

    # --------------------------------------------------------
    # RANGE
    # --------------------------------------------------------

    start_number = ask_int(
        "Starting number",
        1,
        1
    )

    forever = ask_yes_no(
        "Run continuously",
        True
    )

    if forever:

        end_number = None

    else:

        end_number = ask_int(
            "Ending number",
            start_number + 1_000_000,
            start_number
        )

    print()

    # --------------------------------------------------------
    # MAX STEPS
    # --------------------------------------------------------

    max_steps = ask_int(
        "Maximum Collatz steps per number",
        1_000_000,
        1
    )

    print()

    # --------------------------------------------------------
    # CPU SETTINGS
    # --------------------------------------------------------

    if backend in ("cpu", "both"):

        print(
            f"Available CPU logical processors: "
            f"{cpu_count}"
        )

        cpu_workers = ask_int(
            "CPU worker processes",
            cpu_count,
            1
        )

        cpu_chunk_size = ask_int(
            "CPU chunk size",
            10_000,
            1
        )

    else:

        cpu_workers = 0
        cpu_chunk_size = 0

    print()

    # --------------------------------------------------------
    # GPU SETTINGS
    # --------------------------------------------------------

    if backend in ("gpu", "both"):

        # Large batch sizes improve GPU utilization but consume
        # more VRAM for result arrays.
        gpu_batch_size = ask_int(
            "GPU batch size",
            1_000_000,
            1
        )

    else:

        gpu_batch_size = 0

    print()

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("=" * 64)
    print("CONFIGURATION")
    print("=" * 64)

    print(
        f"Backend              : {backend.upper()}"
    )

    print(
        f"Starting number      : {start_number:,}"
    )

    if forever:

        print(
            "Ending number        : INFINITE"
        )

    else:

        print(
            f"Ending number        : {end_number:,}"
        )

    print(
        f"Maximum steps        : {max_steps:,}"
    )

    if cpu_workers:

        print(
            f"CPU processes        : {cpu_workers}"
        )

        print(
            f"CPU chunk size       : {cpu_chunk_size:,}"
        )

    if gpu_batch_size:

        print(
            f"GPU batch size       : {gpu_batch_size:,}"
        )

    print("=" * 64)

    print()

    print(
        "The test will heavily load the selected hardware."
    )

    print(
        "Press Ctrl+C at any time to stop."
    )

    print()

    input(
        "Press ENTER to start..."
    )

    # --------------------------------------------------------
    # SHARED RANGE
    # --------------------------------------------------------

    next_range = [
        start_number
    ]

    next_range_lock = threading.Lock()

    stop_event = threading.Event()

    stats = Statistics()

    # --------------------------------------------------------
    # STATUS THREAD
    # --------------------------------------------------------

    status_thread = threading.Thread(
        target=status_loop,
        args=(
            stats,
            stop_event
        ),
        daemon=True
    )

    status_thread.start()

    cpu_executor = None

    cpu_thread = None
    gpu_thread = None

    try:

        # ----------------------------------------------------
        # CPU
        # ----------------------------------------------------

        if backend in ("cpu", "both"):

            cpu_executor = ProcessPoolExecutor(
                max_workers=cpu_workers
            )

            cpu_thread = threading.Thread(
                target=cpu_loop,
                args=(
                    cpu_executor,
                    next_range,
                    next_range_lock,
                    stats,
                    max_steps,
                    cpu_chunk_size,
                    stop_event,
                    cpu_workers
                ),
                daemon=True
            )

            cpu_thread.start()

        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------

        if backend in ("gpu", "both"):

            gpu_thread = threading.Thread(
                target=gpu_loop,
                args=(
                    next_range,
                    next_range_lock,
                    stats,
                    max_steps,
                    gpu_batch_size,
                    stop_event
                ),
                daemon=True
            )

            gpu_thread.start()

        # ----------------------------------------------------
        # FINITE RANGE MONITOR
        # ----------------------------------------------------

        if not forever:

            while not stop_event.is_set():

                with next_range_lock:

                    current = next_range[0]

                if current > end_number:

                    stop_event.set()
                    break

                time.sleep(0.1)

        else:

            # Infinite mode.
            while not stop_event.is_set():

                time.sleep(1)

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Stopping stress test..."
        )

        stop_event.set()

    finally:

        stop_event.set()

        if cpu_executor is not None:

            # Cancel work that hasn't started.
            cpu_executor.shutdown(
                wait=False,
                cancel_futures=True
            )

        status_thread.join(
            timeout=2
        )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print()
    print()
    print("=" * 64)
    print("FINAL RESULTS")
    print("=" * 64)

    stats.print_status(
        force=True
    )

    print()
    print(
        "No mathematical conclusion about the Collatz"
    )

    print(
        "conjecture can be drawn from this finite computation."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    try:

        run_stress_test()

    except KeyboardInterrupt:

        print()
        print(
            "Stopped."
        )

    except Exception as exc:

        print()
        print(
            f"Fatal error: {exc}"
        )

        sys.exit(1)
