from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "testdata" / "golden" / "generated_query_sample.jsonl"


def main() -> None:
    rows = [
        {
            "id": "known_001",
            "kind": "known",
            "question": "Che cos'e SIEL?",
            "expected_item_id": "item_siel_definition",
        },
        {
            "id": "ood_001",
            "kind": "out_of_domain",
            "question": "Che tempo fara tra cento anni?",
            "expected_status": "unknown",
        },
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

