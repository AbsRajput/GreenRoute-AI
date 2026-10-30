
"""Automated tests for the GreenRoute-AI adaptive router."""

import pandas as pd
import pytest

from src.adaptive_router import (
    select_model,
    select_carbon_aware_model,
)


@pytest.fixture
def sample_results():
    """Create sample benchmark results for testing."""
    return pd.DataFrame([
        {
            "model": "logistic_regression",
            "accuracy": 0.982,
            "prediction_latency_ms_per_sample": 0.025,
            "prediction_emissions_kg_co2eq_estimate": 4e-9,
        },
        {
            "model": "decision_tree",
            "accuracy": 0.921,
            "prediction_latency_ms_per_sample": 0.015,
            "prediction_emissions_kg_co2eq_estimate": 5e-9,
        },
        {
            "model": "random_forest",
            "accuracy": 0.956,
            "prediction_latency_ms_per_sample": 0.123,
            "prediction_emissions_kg_co2eq_estimate": 2e-6,
        },
    ])


# =========================================================
# LATENCY-BASED ROUTER TESTS
# =========================================================

def test_selects_fastest_eligible_model(sample_results):
    result = select_model(sample_results)

    assert result["status"] == "selected"
    assert result["selected_model"] == "decision_tree"


def test_accuracy_constraint_excludes_models(sample_results):
    result = select_model(
        sample_results,
        min_accuracy=0.95,
        max_latency_ms=1.0,
    )

    assert result["status"] == "selected"
    assert result["selected_model"] == "logistic_regression"
    assert "decision_tree" not in result["eligible_models"]


def test_returns_no_eligible_model(sample_results):
    result = select_model(
        sample_results,
        min_accuracy=0.99,
        max_latency_ms=1.0,
    )

    assert result["status"] == "no_eligible_model"
    assert result["selected_model"] is None


def test_rejects_missing_columns():
    invalid_results = pd.DataFrame([
        {"model": "example", "accuracy": 0.95}
    ])

    with pytest.raises(ValueError, match="Missing required columns"):
        select_model(invalid_results)


def test_rejects_invalid_accuracy_threshold(sample_results):
    with pytest.raises(ValueError, match="min_accuracy"):
        select_model(sample_results, min_accuracy=1.5)


def test_rejects_empty_results():
    empty_results = pd.DataFrame(
        columns=[
            "model",
            "accuracy",
            "prediction_latency_ms_per_sample",
        ]
    )

    with pytest.raises(ValueError, match="empty"):
        select_model(empty_results)


# =========================================================
# CARBON-AWARE ROUTER TESTS
# =========================================================

def test_carbon_aware_router_selects_fastest_eligible_model(
    sample_results,
):
    # A strict limit excludes the high-emission Random Forest sample.
    result = select_carbon_aware_model(
        sample_results,
        max_emissions_kg_co2eq=1e-8,
    )

    assert result["status"] == "selected"
    assert result["selected_model"] == "decision_tree"
    assert "random_forest" not in result["eligible_models"]


def test_carbon_limit_excludes_high_emission_model(sample_results):
    result = select_carbon_aware_model(
        sample_results,
        max_emissions_kg_co2eq=1e-8,
    )

    assert result["status"] == "selected"
    assert result["selected_model"] == "decision_tree"
    assert "random_forest" not in result["eligible_models"]


def test_carbon_aware_router_returns_none_if_no_model_qualifies(
    sample_results,
):
    result = select_carbon_aware_model(
        sample_results,
        min_accuracy=0.99,
    )

    assert result["status"] == "no_eligible_model"
    assert result["selected_model"] is None


def test_carbon_aware_router_rejects_missing_emissions_column(
    sample_results,
):
    invalid_results = sample_results.drop(
        columns=["prediction_emissions_kg_co2eq_estimate"]
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        select_carbon_aware_model(invalid_results)


def test_carbon_aware_router_excludes_missing_emissions(sample_results):
    sample_results.loc[
        sample_results["model"] == "decision_tree",
        "prediction_emissions_kg_co2eq_estimate",
    ] = float("nan")

    result = select_carbon_aware_model(sample_results)

    assert result["status"] == "selected"
    assert result["selected_model"] == "logistic_regression"
    assert "decision_tree" not in result["eligible_models"]


def test_carbon_aware_router_rejects_negative_carbon_limit(
    sample_results,
):
    with pytest.raises(ValueError, match="cannot be negative"):
        select_carbon_aware_model(
            sample_results,
            max_emissions_kg_co2eq=-1.0,
        )
