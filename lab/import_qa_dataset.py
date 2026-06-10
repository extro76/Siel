from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args()

    with args.input_csv.open("r", encoding="utf-8", newline="") as src, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        for idx, row in enumerate(reader, start=1):
            dst.write(
                json.dumps(
                    {
                        "id": row.get("id") or f"qa_{idx:06d}",
                        "question": row["question"],
                        "answer": row["answer"],
                        "source": row.get("source"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

