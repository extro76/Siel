from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("golden_jsonl", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.golden_jsonl.read_text(encoding="utf-8").splitlines() if line]
    print(
        json.dumps(
            {
                "items": len(rows),
                "status": "not_run",
                "reason": "wire this script to the local /query API once the server is running",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

