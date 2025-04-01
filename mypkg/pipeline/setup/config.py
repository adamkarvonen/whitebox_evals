from pathlib import Path
import yaml
from typing import Dict, Any


def load_config(config_path: str = "default.yaml") -> Dict[str, Any]:
    config_file = Path("configs") / config_path
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

get_config = load_config