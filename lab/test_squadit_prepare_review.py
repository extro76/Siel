from __future__ import annotations

import gzip
import json
import sqlite3
import sys
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from squadit_prepare_review import (
    canonical_text,
    conversational_key,
    create_schema,
    create_rule_clusters,
    insert_candidates,
    iter_squad_candidates,
)

WORKSPACE_TMP = Path(__file__).resolve().parents[1] / "target" / "squadit-review-tests"


def write_squad(path: Path, qas: list[dict[str, object]]) -> None:
    payload = {
        "version": "1.1",
        "data": [
            {
                "title": "Fixture",
                "paragraphs": [
                    {
                        "context": "Vasco Rossi e Everest fixture context.",
                        "qas": qas,
                    }
                ],
            }
        ],
    }
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


class SquadItReviewTests(unittest.TestCase):
    @contextmanager
    def tempdir(self):
        WORKSPACE_TMP.mkdir(parents=True, exist_ok=True)
        path = WORKSPACE_TMP / f"case-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        yield str(path)

    def test_canonical_text_preserves_meaningful_text(self) -> None:
        self.assertEqual(canonical_text("  Ciao   MONDO  "), "ciao mondo")
        self.assertEqual(canonical_text("e\u0300"), "\u00e8")

    def test_conversational_key_collapses_light_prefixes(self) -> None:
        self.assertEqual(
            conversational_key("Mi sai dire chi e Vasco Rossi?"),
            conversational_key("Chi e Vasco Rossi?"),
        )

    def test_iterator_preserves_utf8_and_skips_repeated_answer_in_same_qa(self) -> None:
        with self.tempdir() as tmp:
            path = Path(tmp) / "squad.json.gz"
            write_squad(
                path,
                [
                    {
                        "id": "q1",
                        "question": "Quanto e alto l'Everest?",
                        "answers": [
                            {"text": "8848 metri", "answer_start": 1},
                            {"text": "8848 metri", "answer_start": 1},
                        ],
                    }
                ],
            )
            rows = list(iter_squad_candidates(path, "train"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].answer_original, "8848 metri")

    def test_rule_clusters_keep_duplicates_and_conflicts(self) -> None:
        with self.tempdir() as tmp:
            path = Path(tmp) / "squad.json.gz"
            write_squad(
                path,
                [
                    {
                        "id": "q1",
                        "question": "Mi sai dire chi e Vasco Rossi?",
                        "answers": [{"text": "Un cantante italiano.", "answer_start": 0}],
                    },
                    {
                        "id": "q2",
                        "question": "Chi e Vasco Rossi?",
                        "answers": [{"text": "Un cantante italiano.", "answer_start": 0}],
                    },
                    {
                        "id": "q3",
                        "question": "Chi e Vasco Rossi?",
                        "answers": [{"text": "Un cantautore italiano.", "answer_start": 0}],
                    },
                ],
            )
            conn = sqlite3.connect(":memory:")
            create_schema(conn)
            conn.row_factory = sqlite3.Row
            insert_candidates(conn, iter_squad_candidates(path, "train"))
            create_rule_clusters(conn)
            candidates = conn.execute("SELECT COUNT(*) FROM candidate_pair").fetchone()[0]
            conflicts = conn.execute(
                "SELECT COUNT(*) FROM duplicate_cluster WHERE status = 'conflict_review'"
            ).fetchone()[0]
            self.assertEqual(candidates, 3)
            self.assertGreaterEqual(conflicts, 1)


if __name__ == "__main__":
    unittest.main()
