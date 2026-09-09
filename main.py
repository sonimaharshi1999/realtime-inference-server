# Real-Time ML Inference Server - Entry Point
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import uvicorn

# Ensure the project root is on sys.path so ``src.*`` imports resolve.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import default_config, MODELS_DIR
from src.train import train_demo_model, train_challenger_model
from src.server import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def ensure_models_exist() -> None:
    """Train the default models if they don't already exist on disk."""

    demo_path = MODELS_DIR / "demo_model.joblib"
    if not demo_path.exists():
        logger.info("Demo model not found -- training from synthetic data...")
        train_demo_model()

    challenger_path = MODELS_DIR / "challenger_model.joblib"
    if not challenger_path.exists():
        logger.info("Challenger model not found -- training from synthetic data...")
        train_challenger_model()


def main() -> None:
    ensure_models_exist()

    config = default_config()
    app = create_app(config)

    host = os.getenv("HOST", config.host)
    port = int(os.getenv("PORT", str(config.port)))

    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
