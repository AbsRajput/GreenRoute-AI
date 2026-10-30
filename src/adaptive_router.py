
"""Constraint-aware model selection for GreenRoute-AI."""

from pathlib import Path

import pandas as pd


DEFAULT_RESULTS = Path("results/model_benchmark.csv")


def select_model(
    results: pd.DataFrame,
    min_accuracy: float = 0.90,
    max_latency_ms: float = 1.0,
) -> dict:
    """Select the fastest model satisfying accuracy and latency constraints."""
    required_columns = {
        "model",
        "accuracy",
        "prediction_latency_ms_per_sample",
    }

    missing = required_columns - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if results.empty:
        raise ValueError("Benchmark results are empty.")

    if not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must be between 0 and 1.")

    if max_latency_ms <= 0:
        raise ValueError("max_latency_ms must be greater than zero.")

    eligible = results[
        (results["accuracy"] >= min_accuracy)
        & (
            results["prediction_latency_ms_per_sample"]
            <= max_latency_ms
        )
    ].copy()

    if eligible.empty:
        return {
            "selected_model": None,
            "status": "no_eligible_model",
            "reason": "No model meets both accuracy and latency constraints.",
            "eligible_models": [],
        }

    eligible = eligible.sort_values(
        by="prediction_latency_ms_per_sample",
        ascending=True,
    )
    best = eligible.iloc[0]

    return {
        "selected_model": str(best["model"]),
        "status": "selected",
        "reason": "Fastest model satisfying accuracy and latency constraints.",
        "accuracy": float(best["accuracy"]),
        "latency_ms_per_sample": float(
            best["prediction_latency_ms_per_sample"]
        ),
        "eligible_models": eligible["model"].astype(str).tolist(),
    }


def select_carbon_aware_model(
    results: pd.DataFrame,
    min_accuracy: float = 0.90,
    max_latency_ms: float = 1.0,
    max_emissions_kg_co2eq: float = 1e-5,
) -> dict:
    """Select the fastest eligible model within an emissions estimate limit."""
    emissions_column = "prediction_emissions_kg_co2eq_estimate"

    required_columns = {
        "model",
        "accuracy",
        "prediction_latency_ms_per_sample",
        emissions_column,
    }

    missing = required_columns - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if results.empty:
        raise ValueError("Benchmark results are empty.")

    if not 0 <= min_accuracy <= 1:
        raise ValueError("min_accuracy must be between 0 and 1.")

    if max_latency_ms <= 0:
        raise ValueError("max_latency_ms must be greater than zero.")

    if max_emissions_kg_co2eq < 0:
        raise ValueError("max_emissions_kg_co2eq cannot be negative.")

    eligible = results[
        (results["accuracy"] >= min_accuracy)
        & (results["prediction_latency_ms_per_sample"] <= max_latency_ms)
        & (results[emissions_column].notna())
        & (results[emissions_column] >= 0)
        & (results[emissions_column] <= max_emissions_kg_co2eq)
    ].copy()

    if eligible.empty:
        return {
            "selected_model": None,
            "status": "no_eligible_model",
            "reason": (
                "No model meets the accuracy, latency and emissions "
                "estimate constraints."
            ),
            "eligible_models": [],
        }

    eligible = eligible.sort_values(
        by="prediction_latency_ms_per_sample",
        ascending=True,
    )
    best = eligible.iloc[0]

    return {
        "selected_model": str(best["model"]),
        "status": "selected",
        "reason": (
            "Fastest model satisfying accuracy, latency and estimated "
            "emissions constraints."
        ),
        "accuracy": float(best["accuracy"]),
        "latency_ms_per_sample": float(
            best["prediction_latency_ms_per_sample"]
        ),
        "prediction_emissions_kg_co2eq_estimate": float(
            best[emissions_column]
        ),
        "eligible_models": eligible["model"].astype(str).tolist(),
        "emissions_measurement": "CodeCarbon estimate; potentially noisy",
    }


def route_request(
    min_accuracy: float = 0.90,
    max_latency_ms: float = 1.0,
    results_path: Path = DEFAULT_RESULTS,
) -> dict:
    """Load benchmark results and perform latency-based routing."""
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {results_path}. "
            "Run `python -m src.model_profiler` first."
        )

    results = pd.read_csv(results_path)

    return select_model(
        results,
        min_accuracy=min_accuracy,
        max_latency_ms=max_latency_ms,
    )


def route_carbon_aware_request(
    min_accuracy: float = 0.90,
    max_latency_ms: float = 1.0,
    max_emissions_kg_co2eq: float = 1e-5,
    results_path: Path = DEFAULT_RESULTS,
) -> dict:
    """Load benchmark results and perform carbon-aware routing."""
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {results_path}. "
            "Run `python -m src.model_profiler` first."
        )

    results = pd.read_csv(results_path)

    return select_carbon_aware_model(
        results,
        min_accuracy=min_accuracy,
        max_latency_ms=max_latency_ms,
        max_emissions_kg_co2eq=max_emissions_kg_co2eq,
    )


if __name__ == "__main__":
    print("LATENCY-BASED ROUTING")
    decision = route_request()
    for key, value in decision.items():
        print(f"{key}: {value}")

    print("\nCARBON-AWARE ROUTING")
    carbon_decision = route_carbon_aware_request()
    for key, value in carbon_decision.items():
        print(f"{key}: {value}")
