
"""Dataset loading and validation for GreenRoute-AI."""

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
import pandas as pd


SUPPORTED_DATASETS = {
    "breast_cancer": load_breast_cancer,
    "wine": load_wine,
}


def load_dataset(name: str):
    """Load a supported scikit-learn dataset."""
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {name}. "
            f"Choose from {list(SUPPORTED_DATASETS)}"
        )

    dataset = SUPPORTED_DATASETS[name](as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()

    return X, y


def validate_dataset(X: pd.DataFrame, y: pd.Series) -> dict:
    """Return basic data-quality information."""
    return {
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows": int(X.duplicated().sum()),
        "target_classes": int(y.nunique()),
        "target_missing_values": int(y.isna().sum()),
    }


def prepare_dataset(
    name: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Load, validate and split a dataset for ML experiments."""
    X, y = load_dataset(name)
    report = validate_dataset(X, y)

    if report["missing_values"] > 0 or report["target_missing_values"] > 0:
        raise ValueError("Dataset contains missing values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, report
