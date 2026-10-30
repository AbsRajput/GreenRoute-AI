
"""Evaluate GreenRoute-AI models on a user-uploaded CSV."""

from time import perf_counter

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.model_profiler import get_models, measure_prediction_latency


def evaluate_uploaded_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train and compare supported classifiers on a CSV dataset."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("The uploaded dataset is empty.")

    if target_column not in dataframe.columns:
        raise ValueError("Please select a valid target column.")

    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")

    data = dataframe.copy()
    data = data.dropna(subset=[target_column])

    if data[target_column].nunique() < 2:
        raise ValueError(
            "The target column must contain at least two classes."
        )

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if X.shape[1] < 1:
        raise ValueError("The dataset needs at least one feature column.")

    # This initial version supports numeric feature columns only.
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        raise ValueError(
            "Some feature values are missing or non-numeric. "
            "Clean these columns before evaluation."
        )

    if len(data) < 20:
        raise ValueError(
            "Please provide at least 20 usable rows for this experiment."
        )

    class_counts = y.value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Each target class needs at least two rows for a stratified split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    results = []

    for model_name, model in get_models().items():
        estimator = clone(model)

        start = perf_counter()
        estimator.fit(X_train, y_train)
        train_seconds = perf_counter() - start

        predictions, batch_seconds, latency_ms = (
            measure_prediction_latency(estimator, X_test)
        )

        results.append({
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "f1_score": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "train_time_seconds": train_seconds,
            "prediction_batch_seconds": batch_seconds,
            "prediction_latency_ms_per_sample": latency_ms,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": X.shape[1],
            "classes": y.nunique(),
            "carbon_emissions": (
                "Not measured in this dataset evaluation"
            ),
        })

    return pd.DataFrame(results).sort_values(
        by=["accuracy", "f1_score"],
        ascending=False,
    ).reset_index(drop=True)
