from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load an experiment YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
