from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.scores_jsonl)
    raw_scores = np.array([row["raw_score"] for row in rows], dtype=float)
    correct = np.array([1 if row["correct"] else 0 for row in rows], dtype=float)

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(raw_scores, correct)

    payload = {
        "method": "isotonic",
        "version": 1,
        "x_thresholds": [float(x) for x in model.X_thresholds_],
        "y_probabilities": [float(y) for y in model.y_thresholds_],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

