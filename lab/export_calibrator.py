from __future__ import annotations

import json
from pathlib import Path


def validate(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["method"] == "isotonic"
    assert data["version"] == 1
    assert len(data["x_thresholds"]) == len(data["y_probabilities"])
    assert data["x_thresholds"] == sorted(data["x_thresholds"])
    assert all(0.0 <= value <= 1.0 for value in data["y_probabilities"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("calibrator_json", type=Path)
    args = parser.parse_args()
    validate(args.calibrator_json)
    print("calibrator ok")

