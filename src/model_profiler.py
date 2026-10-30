
"""Benchmark ML models for GreenRoute-AI."""

from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_engine import prepare_dataset
from src.carbon_tracker import track_emissions, repeat_task


LATENCY_REPEATS = 100
EMISSIONS_REPEATS = 100


def get_models():
    """Return reproducible baseline models."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=1
        ),
    }


def measure_prediction_latency(
    estimator,
    X_test,
    repeats=LATENCY_REPEATS,
):
    """Measure prediction latency separately from carbon tracking."""
    if not isinstance(repeats, int) or isinstance(repeats, bool):
        raise TypeError("repeats must be an integer.")

    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    # Warm-up before timing.
    estimator.predict(X_test)

    start = perf_counter()

    for _ in range(repeats):
        predictions = estimator.predict(X_test)

    elapsed_seconds = perf_counter() - start

    total_samples = len(X_test) * repeats
    latency_ms_per_sample = (
        elapsed_seconds / total_samples * 1000
    )

    return predictions, elapsed_seconds, latency_ms_per_sample


def benchmark_models(dataset_name="breast_cancer"):
    """Benchmark model quality, latency and estimated emissions."""
    X_train, X_test, y_train, y_test, _ = prepare_dataset(
        dataset_name
    )
    results = []

    for model_name, model in get_models().items():
        print(f"\nBenchmarking: {model_name}")

        estimator = clone(model)

        # Measure training time independently.
        train_start = perf_counter()
        estimator.fit(X_train, y_train)
        train_seconds = perf_counter() - train_start

        # Measure latency without CodeCarbon tracking overhead.
        predictions, prediction_batch_seconds, latency_ms = (
            measure_prediction_latency(
                estimator,
                X_test,
                repeats=LATENCY_REPEATS,
            )
        )

        # Track emissions for one training run.
        training_emissions = track_emissions(
            lambda: clone(model).fit(X_train, y_train),
            output_dir=f"results/carbon/{model_name}_training",
        )

        # Track the same prediction batch repeatedly.
        # This is a separate run from the latency measurement.
        prediction_emissions = track_emissions(
            lambda: repeat_task(
                lambda: estimator.predict(X_test),
                repeats=EMISSIONS_REPEATS,
            ),
            output_dir=f"results/carbon/{model_name}_prediction",
        )

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "f1_score": f1_score(
                y_test,
                predictions,
                average="weighted",
            ),
            "train_time_seconds": train_seconds,
            "prediction_batch_seconds_100_repeats":
                prediction_batch_seconds,
            "prediction_latency_ms_per_sample": latency_ms,
            "training_emissions_kg_co2eq_estimate":
                training_emissions["emissions_kg_co2eq"],
            "prediction_emissions_kg_co2eq_estimate":
                prediction_emissions["emissions_kg_co2eq"],
            "test_samples": len(X_test),
            "latency_repeats": LATENCY_REPEATS,
            "emissions_prediction_repeats": EMISSIONS_REPEATS,
            "emissions_measurement_type": "CodeCarbon estimate",
        })

    return pd.DataFrame(results).sort_values(
        by="accuracy",
        ascending=False,
    ).reset_index(drop=True)


if __name__ == "__main__":
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = benchmark_models()

    output_file = output_dir / "model_benchmark.csv"
    results.to_csv(output_file, index=False)

    print("\nGREENROUTE-AI MODEL BENCHMARK")
    print(results.to_string(index=False))
    print(f"\nResults saved to: {output_file}")

    print(
        "\nNote: CO2 emissions are CodeCarbon estimates, "
        "not direct electricity measurements."
    )
    print(
        "Prediction latency and emissions are measured in "
        "separate runs. Both prediction workloads use "
        f"{LATENCY_REPEATS} and {EMISSIONS_REPEATS} repetitions, "
        "respectively."
    )
    print(
        "Results can vary with system load, background activity, "
        "and the accuracy of the emissions estimation method."
    )
