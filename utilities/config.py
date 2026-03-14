# utilities/config.py
import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml"):
    """
    Load the YAML configuration file (UTF-8 encoding).

    - Checks that the file exists.
    - Raises an explicit error if the YAML is empty or invalid.
    - Always returns a dict when everything is fine.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration '{config_path}' introuvable (dossier courant = {Path.cwd()})"
        )

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        # Empty file or comments only
        raise ValueError(f"Le fichier de configuration '{config_path}' est vide.")

    if not isinstance(config, dict):
        raise TypeError(f"Le fichier de configuration '{config_path}' ne décrit pas un dictionnaire YAML.")

    print(f"Configuration chargée depuis {config_path}")
    return config


def save_config(config: dict, config_path: str) -> None:
    """Save a configuration dict to YAML (UTF-8)."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
