import json
import logging
from pathlib import Path
from pprint import pprint

import requests

log = logging.getLogger(__name__)


def load_math_train(local_path: str = "math_train.json", save_copy: bool = True) -> list:
    path = Path(local_path)
    url = "https://raw.githubusercontent.com/rasbt/" "math_full_minus_math500/refs/heads/main/" "math_full_minus_math500.json"
    backup_url = "https://f001.backblazeb2.com/file/reasoning-from-scratch/" "MATH/math_full_minus_math500.json"

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException:
        log.warning("Primary URL failed, trying backup")
        r = requests.get(backup_url, timeout=30)
        r.raise_for_status()

    data = r.json()

    if save_copy:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_math_train()
    log.info("Loaded %d examples", len(data))
    pprint(data[4])
