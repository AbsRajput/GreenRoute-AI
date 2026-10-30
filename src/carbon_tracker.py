
"""Energy and carbon tracking utilities for GreenRoute-AI."""

from pathlib import Path
from time import perf_counter

import psutil


def repeat_task(task, repeats=100):
    """Execute a task repeatedly and return the final result."""
    if not isinstance(repeats, int) or isinstance(repeats, bool):
        raise TypeError("repeats must be an integer.")

    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    result = None

    for _ in range(repeats):
        result = task()

    return result


def measure_cpu_activity(task):
    """Measure elapsed time and process CPU time for a task."""
    process = psutil.Process()
    cpu_before = process.cpu_times()

    start = perf_counter()

    try:
        result = task()
    finally:
        elapsed_seconds = perf_counter() - start
        cpu_after = process.cpu_times()

    cpu_seconds = (
        cpu_after.user
        + cpu_after.system
        - cpu_before.user
        - cpu_before.system
    )

    return {
        "result": result,
        "elapsed_seconds": elapsed_seconds,
        "process_cpu_seconds": max(0.0, cpu_seconds),
        "measurement_type": "Process CPU-time estimate",
    }


def track_emissions(task, output_dir="results/carbon"):
    """Run a task and return CodeCarbon's estimated CO2 emissions."""
    from codecarbon import EmissionsTracker

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="GreenRoute-AI",
        output_dir=str(output_path),
        output_file="emissions.csv",
        log_level="error",
        save_to_file=True,
    )

    tracker.start()

    try:
        result = task()
    finally:
        emissions_kg = tracker.stop()

    return {
        "result": result,
        "emissions_kg_co2eq": emissions_kg,
        "measurement_type": "CodeCarbon estimate",
    }


if __name__ == "__main__":
    def example_task():
        return sum(i * i for i in range(100_000))

    cpu_report = measure_cpu_activity(example_task)

    print("CPU ACTIVITY REPORT")
    print(f"Task result: {cpu_report['result']}")
    print(
        f"Elapsed seconds: "
        f"{cpu_report['elapsed_seconds']:.6f}"
    )
    print(
        f"Process CPU seconds: "
        f"{cpu_report['process_cpu_seconds']:.6f}"
    )
    print("CPU time is not a direct electricity measurement.")

    repeated_result = repeat_task(example_task, repeats=100)
    print(f"\nRepeated task final result: {repeated_result}")
