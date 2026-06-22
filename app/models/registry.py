"""Model registry — save/load models with metadata. FULLY BUILT.

Keeps the model artifact and its ModelCard (metadata) together in one joblib
file so serving always has the feature list and training context available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelCard:
    """Metadata that travels with every saved model."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    data_window: dict[str, Any] = field(default_factory=dict)


def save_model(model: Any, card: ModelCard, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "card": card}, path)


def load_model(path: Path) -> tuple[Any, ModelCard]:
    bundle = joblib.load(path)
    return bundle["model"], bundle["card"]
