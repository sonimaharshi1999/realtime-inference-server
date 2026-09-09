# Real-Time ML Inference Server - Train Demo Model on Synthetic Data
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_samples: int = 5000,
    n_features: int = 10,
    n_classes: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a reproducible synthetic classification dataset."""

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=7,
        n_redundant=2,
        n_classes=n_classes,
        random_state=random_state,
        flip_y=0.05,
    )
    return X, y


def train_demo_model(output_dir: Path | None = None) -> Path:
    """Train a GradientBoostingClassifier and persist it with joblib.

    Returns the path to the saved model file.
    """

    if output_dir is None:
        output_dir = MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic data...")
    X, y = generate_synthetic_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Training GradientBoostingClassifier (100 estimators)...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    logger.info("Validation accuracy=%.4f  f1=%.4f", acc, f1)

    model_path = output_dir / "demo_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    return model_path


def train_challenger_model(output_dir: Path | None = None) -> Path:
    """Train a lighter model to serve as the B variant in A/B tests."""

    if output_dir is None:
        output_dir = MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = generate_synthetic_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        random_state=99,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Challenger model accuracy=%.4f", acc)

    model_path = output_dir / "challenger_model.joblib"
    joblib.dump(model, model_path)
    logger.info("Challenger model saved to %s", model_path)

    return model_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_demo_model()
    train_challenger_model()
