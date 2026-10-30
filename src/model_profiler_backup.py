
"""Benchmark ML models for GreenRoute-AI."""

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


def benchmark_models(dataset_name="breast_cancer"):
    """Train and benchmark models on the same data split."""
    X_train, X_test, y_train, y_test, _ = prepare_dataset(dataset_name)
    results = []

    for model_name, model in get_models().items():
        estimator = clone(model)

        start_train = perf_counter()
        estimator.fit(X_train, y_train)
        train_seconds = perf_counter() - start_train

        start_predict = perf_counter()
        predictions = estimator.predict(X_test)
        predict_seconds = perf_counter() - start_predict

        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions, average="weighted"),
            "train_time_seconds": train_seconds,
            "prediction_time_seconds": predict_seconds,
            "prediction_latency_ms_per_sample": (
                predict_seconds / len(X_test) * 1000
            ),
            "test_samples": len(X_test),
        })

    return pd.DataFrame(results).sort_values(
        by="accuracy", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path

    results = benchmark_models()

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "model_benchmark.csv"
    results.to_csv(output_file, index=False)

    print(results.to_string(index=False))
    print(f"\nResults saved to: {output_file}")
