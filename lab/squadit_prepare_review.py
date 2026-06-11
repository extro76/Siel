from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_MODEL = "nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl"
DEFAULT_OUT_DIR = Path("testdata/import")
DEFAULT_HF_CACHE = Path("D:/hf-cache")
DEFAULT_TORCH_PATH = Path("C:/pinokio/cache/UV_CACHE_DIR/archive-v0/mNwzhMwsMAtfBYQ3")
SCHEMA_VERSION = 1


def canonical_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value.casefold()


_CONVERSATIONAL_PREFIXES = [
    r"mi\s+sapresti\s+dire\s+",
    r"mi\s+sai\s+dire\s+",
    r"puoi\s+dirmi\s+",
    r"potresti\s+dirmi\s+",
    r"sapresti\s+dirmi\s+",
    r"vorrei\s+sapere\s+",
    r"sai\s+dirmi\s+",
    r"sai\s+",
]


def conversational_key(question: str) -> str:
    key = canonical_text(question)
    key = re.sub(r"^[\"'`]+|[\"'`]+$", "", key)
    for prefix in _CONVERSATIONAL_PREFIXES:
        key = re.sub(rf"^{prefix}", "", key)
    key = re.sub(r"\s*,?\s*per\s+favore\s*\??$", "?", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def stable_id(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class Candidate:
    id: str
    source_qid: str
    split: str
    question_original: str
    answer_original: str
    question_canonical: str
    answer_canonical: str
    question_conversational_key: str
    title: str | None
    context: str
    answer_start: int | None
    source_file: str
    status: str = "pending"

    @property
    def qa_text(self) -> str:
        return f"{self.question_original}\n{self.answer_original}"

    @property
    def titled_qa_text(self) -> str:
        title = self.title or ""
        return f"{title}\n{self.question_original}\n{self.answer_original}".strip()

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source_qid": self.source_qid,
            "split": self.split,
            "question": self.question_original,
            "answer": self.answer_original,
            "question_canonical": self.question_canonical,
            "answer_canonical": self.answer_canonical,
            "question_conversational_key": self.question_conversational_key,
            "title": self.title,
            "context": self.context,
            "answer_start": self.answer_start,
            "source_file": self.source_file,
            "status": self.status,
        }


def iter_squad_candidates(path: Path, split: str) -> Iterator[Candidate]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)

    for article_idx, article in enumerate(payload.get("data", [])):
        title = article.get("title")
        for paragraph_idx, paragraph in enumerate(article.get("paragraphs", [])):
            context = paragraph.get("context", "")
            for qa_idx, qa in enumerate(paragraph.get("qas", [])):
                source_qid = qa.get("id") or f"{split}_{article_idx}_{paragraph_idx}_{qa_idx}"
                question = qa.get("question", "")
                seen_answers: set[tuple[str, int | None]] = set()
                for answer_idx, answer in enumerate(qa.get("answers") or []):
                    answer_text = answer.get("text", "")
                    answer_start = answer.get("answer_start")
                    answer_key = (canonical_text(answer_text), answer_start)
                    if answer_key in seen_answers:
                        continue
                    seen_answers.add(answer_key)
                    question_canonical = canonical_text(question)
                    answer_canonical = canonical_text(answer_text)
                    candidate_id = f"squadit_{split}_{stable_id(source_qid, answer_idx, answer_text)}"
                    yield Candidate(
                        id=candidate_id,
                        source_qid=source_qid,
                        split=split,
                        question_original=question,
                        answer_original=answer_text,
                        question_canonical=question_canonical,
                        answer_canonical=answer_canonical,
                        question_conversational_key=conversational_key(question),
                        title=title,
                        context=context,
                        answer_start=answer_start if isinstance(answer_start, int) else None,
                        source_file=str(path),
                    )


def open_review_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = DELETE")
    conn.execute("PRAGMA synchronous = NORMAL")
    create_schema(conn)
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS review_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS candidate_pair (
            id TEXT PRIMARY KEY,
            source_qid TEXT NOT NULL,
            split TEXT NOT NULL,
            question_original TEXT NOT NULL,
            answer_original TEXT NOT NULL,
            question_canonical TEXT NOT NULL,
            answer_canonical TEXT NOT NULL,
            question_conversational_key TEXT NOT NULL,
            title TEXT,
            context TEXT NOT NULL,
            answer_start INTEGER,
            source_file TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS duplicate_cluster (
            id TEXT PRIMARY KEY,
            cluster_type TEXT NOT NULL,
            status TEXT NOT NULL,
            winner_candidate_id TEXT,
            created_reason TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (winner_candidate_id) REFERENCES candidate_pair(id)
        );

        CREATE TABLE IF NOT EXISTS cluster_member (
            cluster_id TEXT NOT NULL REFERENCES duplicate_cluster(id) ON DELETE CASCADE,
            candidate_id TEXT NOT NULL REFERENCES candidate_pair(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            score REAL,
            reason TEXT NOT NULL,
            PRIMARY KEY (cluster_id, candidate_id)
        );

        CREATE TABLE IF NOT EXISTS semantic_match (
            candidate_a TEXT NOT NULL REFERENCES candidate_pair(id) ON DELETE CASCADE,
            candidate_b TEXT NOT NULL REFERENCES candidate_pair(id) ON DELETE CASCADE,
            question_similarity REAL NOT NULL,
            qa_similarity REAL NOT NULL,
            title_match INTEGER NOT NULL,
            context_overlap REAL NOT NULL,
            model_id TEXT NOT NULL,
            decision_hint TEXT NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (candidate_a, candidate_b, model_id)
        );

        CREATE TABLE IF NOT EXISTS review_decision (
            cluster_id TEXT PRIMARY KEY REFERENCES duplicate_cluster(id) ON DELETE CASCADE,
            decision TEXT NOT NULL,
            winner_candidate_id TEXT REFERENCES candidate_pair(id),
            notes TEXT,
            decided_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS candidate_embedding (
            candidate_id TEXT NOT NULL REFERENCES candidate_pair(id) ON DELETE CASCADE,
            embedding_kind TEXT NOT NULL,
            model_id TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            vector BLOB NOT NULL,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (candidate_id, embedding_kind, model_id)
        );

        CREATE INDEX IF NOT EXISTS idx_candidate_pair_canonical_pair
            ON candidate_pair(question_canonical, answer_canonical);
        CREATE INDEX IF NOT EXISTS idx_candidate_pair_conversation
            ON candidate_pair(question_conversational_key);
        CREATE INDEX IF NOT EXISTS idx_candidate_pair_status
            ON candidate_pair(status);
        CREATE INDEX IF NOT EXISTS idx_cluster_type
            ON duplicate_cluster(cluster_type, status);
        CREATE INDEX IF NOT EXISTS idx_member_candidate
            ON cluster_member(candidate_id);
        CREATE INDEX IF NOT EXISTS idx_semantic_a
            ON semantic_match(candidate_a);
        CREATE INDEX IF NOT EXISTS idx_semantic_b
            ON semantic_match(candidate_b);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO review_metadata(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def reset_review_content(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DELETE FROM review_decision;
        DELETE FROM candidate_embedding;
        DELETE FROM semantic_match;
        DELETE FROM cluster_member;
        DELETE FROM duplicate_cluster;
        DELETE FROM candidate_pair;
        """
    )
    conn.commit()


def insert_candidates(conn: sqlite3.Connection, candidates: Iterable[Candidate]) -> int:
    count = 0
    with conn:
        for candidate in candidates:
            conn.execute(
                """
                INSERT INTO candidate_pair(
                    id, source_qid, split, question_original, answer_original,
                    question_canonical, answer_canonical, question_conversational_key,
                    title, context, answer_start, source_file, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.id,
                    candidate.source_qid,
                    candidate.split,
                    candidate.question_original,
                    candidate.answer_original,
                    candidate.question_canonical,
                    candidate.answer_canonical,
                    candidate.question_conversational_key,
                    candidate.title,
                    candidate.context,
                    candidate.answer_start,
                    candidate.source_file,
                    candidate.status,
                ),
            )
            count += 1
    return count


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
            count += 1
    return count


def load_squad_stats(path: Path, split: str) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    articles = payload.get("data", [])
    paragraphs = 0
    qas = 0
    answer_rows = 0
    repeated_answer_annotations = 0
    question_answer_counts: Counter[str] = Counter()
    for article in articles:
        for paragraph in article.get("paragraphs", []):
            paragraphs += 1
            for qa in paragraph.get("qas", []):
                qas += 1
                answers = qa.get("answers") or []
                answer_rows += len(answers)
                answer_keys = [canonical_text(answer.get("text", "")) for answer in answers]
                if len(answer_keys) > len(set(answer_keys)):
                    repeated_answer_annotations += 1
                question_answer_counts[canonical_text(qa.get("question", ""))] += len(set(answer_keys))
    return {
        "split": split,
        "source_file": str(path),
        "articles": len(articles),
        "paragraphs": paragraphs,
        "questions": qas,
        "answer_rows_before_per_qa_dedup": answer_rows,
        "qa_entries_with_repeated_answer_annotations": repeated_answer_annotations,
        "canonical_questions_with_multiple_answers": sum(
            1 for value in question_answer_counts.values() if value > 1
        ),
    }


def make_build_summary(conn: sqlite3.Connection, source_stats: list[dict[str, object]]) -> dict[str, object]:
    exact_duplicate_groups = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT question_canonical, answer_canonical
            FROM candidate_pair
            GROUP BY question_canonical, answer_canonical
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    conversational_groups = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT question_conversational_key
            FROM candidate_pair
            GROUP BY question_conversational_key
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    candidates = conn.execute("SELECT COUNT(*) FROM candidate_pair").fetchone()[0]
    unique_pairs = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT question_canonical, answer_canonical FROM candidate_pair
            GROUP BY question_canonical, answer_canonical
        )
        """
    ).fetchone()[0]
    conflicts = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT question_conversational_key
            FROM candidate_pair
            GROUP BY question_conversational_key
            HAVING COUNT(DISTINCT answer_canonical) > 1
        )
        """
    ).fetchone()[0]
    return {
        "source_stats": source_stats,
        "review_db": candidates,
        "candidate_pairs": candidates,
        "unique_canonical_pairs": unique_pairs,
        "exact_duplicate_groups": exact_duplicate_groups,
        "conversational_duplicate_groups": conversational_groups,
        "conversational_conflict_groups": conflicts,
        "schema_version": SCHEMA_VERSION,
    }


def create_cluster(
    conn: sqlite3.Connection,
    cluster_type: str,
    candidates: list[sqlite3.Row],
    reason: str,
    score: float | None = None,
    winner: str | None = None,
) -> str:
    joined = "|".join(sorted(row["id"] for row in candidates))
    cluster_id = f"{cluster_type}_{stable_id(reason, joined)}"
    status = status_for_cluster(cluster_type)
    winner = winner or suggest_winner(candidates)
    conn.execute(
        """
        INSERT OR IGNORE INTO duplicate_cluster(
            id, cluster_type, status, winner_candidate_id, created_reason
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (cluster_id, cluster_type, status, winner, reason),
    )
    for row in candidates:
        role = "winner_suggestion" if row["id"] == winner else "member"
        conn.execute(
            """
            INSERT OR IGNORE INTO cluster_member(cluster_id, candidate_id, role, score, reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cluster_id, row["id"], role, score, reason),
        )
    return cluster_id


def status_for_cluster(cluster_type: str) -> str:
    if cluster_type == "exact_duplicate":
        return "exact_duplicate"
    if cluster_type == "conversational_conflict":
        return "conflict_review"
    if cluster_type in {"conversational_duplicate", "semantic_duplicate"}:
        return "possible_better_answer"
    return "manual_review"


def suggest_winner(rows: list[sqlite3.Row]) -> str | None:
    if not rows:
        return None
    scored = [(score_answer(row["question_original"], row["answer_original"], row["context"]), row["id"]) for row in rows]
    scored.sort(reverse=True)
    return scored[0][1]


def score_answer(question: str, answer: str, context: str) -> float:
    answer_clean = re.sub(r"\s+", " ", answer).strip()
    tokens = answer_clean.split()
    score = 0.0
    if 4 <= len(tokens) <= 32:
        score += 2.0
    elif len(tokens) > 32:
        score -= 1.0
    if re.search(r"\d", answer_clean):
        score += 0.5
    if re.search(r"\b(km|metri|miglia|anni|euro|percento|%)\b", answer_clean.casefold()):
        score += 0.5
    if answer_clean and canonical_text(answer_clean) in canonical_text(context):
        score += 1.0
    q_terms = set(tokenize_for_overlap(question))
    a_terms = set(tokenize_for_overlap(answer))
    if q_terms & a_terms:
        score += 0.5
    if len(tokens) <= 2:
        score -= 0.25
    return score


def tokenize_for_overlap(value: str) -> list[str]:
    return re.findall(r"\w+", canonical_text(value), flags=re.UNICODE)


def create_rule_clusters(conn: sqlite3.Connection) -> None:
    conn.row_factory = sqlite3.Row
    with conn:
        exact_groups = conn.execute(
            """
            SELECT question_canonical, answer_canonical
            FROM candidate_pair
            GROUP BY question_canonical, answer_canonical
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        for group in exact_groups:
            rows = conn.execute(
                """
                SELECT * FROM candidate_pair
                WHERE question_canonical = ? AND answer_canonical = ?
                ORDER BY split, source_qid, id
                """,
                (group["question_canonical"], group["answer_canonical"]),
            ).fetchall()
            create_cluster(conn, "exact_duplicate", list(rows), "same canonical question and answer")

        conversational_groups = conn.execute(
            """
            SELECT question_conversational_key,
                   COUNT(DISTINCT answer_canonical) AS answers
            FROM candidate_pair
            GROUP BY question_conversational_key
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        for group in conversational_groups:
            rows = conn.execute(
                """
                SELECT * FROM candidate_pair
                WHERE question_conversational_key = ?
                ORDER BY split, source_qid, id
                """,
                (group["question_conversational_key"],),
            ).fetchall()
            cluster_type = (
                "conversational_conflict"
                if int(group["answers"]) > 1
                else "conversational_duplicate"
            )
            create_cluster(conn, cluster_type, list(rows), "same conversational question key")


def run_build(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    review_db = args.review_db
    conn = open_review_db(review_db)
    reset_review_content(conn)

    train_candidates = list(iter_squad_candidates(args.train, "train"))
    test_candidates = list(iter_squad_candidates(args.test, "test"))

    write_jsonl(out_dir / "squad_it_train_candidates.jsonl", (c.to_json() for c in train_candidates))
    write_jsonl(out_dir / "squad_it_test_candidates.jsonl", (c.to_json() for c in test_candidates))

    insert_candidates(conn, train_candidates)
    insert_candidates(conn, test_candidates)
    create_rule_clusters(conn)

    source_stats = [
        load_squad_stats(args.train, "train"),
        load_squad_stats(args.test, "test"),
    ]
    summary = make_build_summary(conn, source_stats)
    summary["review_db_path"] = str(review_db)
    summary["out_dir"] = str(out_dir)
    (out_dir / "squad_it_build_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))


def load_candidates_for_semantic(conn: sqlite3.Connection, split: str | None, limit: int | None) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    sql = "SELECT * FROM candidate_pair"
    params: list[object] = []
    if split:
        sql += " WHERE split = ?"
        params.append(split)
    sql += " ORDER BY split, source_qid, id"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def encode_or_load_embeddings(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
    model_id: str,
    embedding_kind: str,
    batch_size: int,
    cache_dir: Path | None,
    device: str,
    torch_path: Path | None,
):
    import numpy as np

    if not rows:
        return np.zeros((0, 0), dtype="float32")

    existing = load_existing_embeddings(conn, rows, model_id, embedding_kind)
    missing_rows = [row for row in rows if row["id"] not in existing]

    if not missing_rows:
        return np.vstack([existing[row["id"]] for row in rows]).astype("float32")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    configure_external_torch(torch_path)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Install optional semantic dependencies first: "
            "pip install sentence-transformers torch tqdm"
        ) from exc

    resolved_device = resolve_device(device)
    print(
        json.dumps(
            {
                "embedding_kind": embedding_kind,
                "cached_embeddings": len(existing),
                "missing_embeddings": len(missing_rows),
                "device": resolved_device,
            },
            ensure_ascii=False,
        )
    )
    model = SentenceTransformer(
        model_id,
        cache_folder=str(cache_dir) if cache_dir else None,
        device=resolved_device,
    )
    texts = [embedding_text(row, embedding_kind) for row in missing_rows]

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    save_embeddings(conn, missing_rows, model_id, embedding_kind, vectors)
    for row, vector in zip(missing_rows, vectors):
        existing[row["id"]] = vector
    return np.vstack([existing[row["id"]] for row in rows]).astype("float32")


def embedding_text(row: sqlite3.Row, embedding_kind: str) -> str:
    if embedding_kind == "question":
        return row["question_original"]
    if embedding_kind == "qa":
        return f"{row['question_original']}\n{row['answer_original']}"
    if embedding_kind == "titled_qa":
        return f"{row['title'] or ''}\n{row['question_original']}\n{row['answer_original']}".strip()
    raise ValueError(f"unsupported embedding kind: {embedding_kind}")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_external_torch(torch_path: Path | None) -> None:
    if torch_path is None:
        return
    torch_path = torch_path.resolve()
    torch_lib = torch_path / "torch" / "lib"
    if torch_lib.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(torch_lib))
    if str(torch_path) not in sys.path:
        sys.path.insert(0, str(torch_path))


def load_existing_embeddings(
    conn: sqlite3.Connection,
    rows: list[sqlite3.Row],
    model_id: str,
    embedding_kind: str,
) -> dict[str, object]:
    import numpy as np

    row_ids = [row["id"] for row in rows]
    if not row_ids:
        return {}
    placeholders = ",".join("?" for _ in row_ids)
    found = conn.execute(
        f"""
        SELECT candidate_id, dimensions, vector
        FROM candidate_embedding
        WHERE model_id = ? AND embedding_kind = ? AND candidate_id IN ({placeholders})
        """,
        [model_id, embedding_kind, *row_ids],
    ).fetchall()
    if not found:
        return {}
    dimensions = int(found[0][1])
    out = {}
    for row in found:
        if int(row[1]) != dimensions:
            continue
        vector = np.frombuffer(row[2], dtype="float32").copy()
        if vector.shape[0] == dimensions:
            out[row[0]] = vector
    return out


def save_embeddings(conn: sqlite3.Connection, rows: list[sqlite3.Row], model_id: str, embedding_kind: str, vectors) -> None:
    with conn:
        for row, vector in zip(rows, vectors):
            conn.execute(
                """
                INSERT OR REPLACE INTO candidate_embedding(
                    candidate_id, embedding_kind, model_id, dimensions, vector
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (row["id"], embedding_kind, model_id, int(vector.shape[0]), vector.tobytes()),
            )


def context_overlap(a: str, b: str) -> float:
    a_tokens = set(tokenize_for_overlap(a))
    b_tokens = set(tokenize_for_overlap(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def semantic_hint(
    question_similarity: float,
    qa_similarity: float,
    title_match: bool,
    overlap: float,
    question_threshold: float,
    qa_threshold: float,
    weak_threshold: float,
) -> str:
    if question_similarity >= 0.95 and qa_similarity >= qa_threshold:
        return "semantic_duplicate"
    if question_similarity >= question_threshold and qa_similarity >= qa_threshold:
        return "possible_better_answer"
    if question_similarity >= question_threshold and qa_similarity < qa_threshold:
        return "conflict_review"
    if question_similarity >= weak_threshold and (title_match or overlap >= 0.2):
        return "manual_review"
    return "ignore"


def run_semantic(args: argparse.Namespace) -> None:
    import numpy as np

    conn = open_review_db(args.review_db)
    rows = load_candidates_for_semantic(conn, args.split, args.limit)
    if not rows:
        print(json.dumps({"semantic_matches": 0, "reason": "no candidates"}, indent=2))
        return

    question_vectors = encode_or_load_embeddings(
        conn,
        rows,
        args.model,
        "question",
        args.batch_size,
        args.cache_dir,
        args.device,
        args.torch_path,
    )
    qa_vectors = encode_or_load_embeddings(
        conn,
        rows,
        args.model,
        "qa",
        args.batch_size,
        args.cache_dir,
        args.device,
        args.torch_path,
    )
    min_threshold = min(args.question_threshold, args.weak_threshold)
    saved = 0
    started = time.time()

    conn.row_factory = sqlite3.Row
    with conn:
        delete_semantic_matches_for_rows(conn, rows, args.model)
        delete_semantic_clusters_for_rows(conn, rows)
        for start in range(0, len(rows), args.chunk_size):
            end = min(start + args.chunk_size, len(rows))
            q_scores = question_vectors[start:end] @ question_vectors.T
            qa_scores = qa_vectors[start:end] @ qa_vectors.T
            for local_idx in range(end - start):
                idx = start + local_idx
                q_row = q_scores[local_idx]
                candidates = np.argwhere(q_row >= min_threshold).reshape(-1)
                candidates = candidates[candidates > idx]
                if candidates.size == 0:
                    continue
                order = np.argsort(q_row[candidates])[::-1][: args.top_k]
                for other_idx in candidates[order]:
                    left = rows[idx]
                    right = rows[int(other_idx)]
                    title_match = bool(left["title"] and right["title"] and left["title"] == right["title"])
                    overlap = context_overlap(left["context"], right["context"])
                    question_similarity = float(q_scores[local_idx, other_idx])
                    qa_similarity = float(qa_scores[local_idx, other_idx])
                    hint = semantic_hint(
                        question_similarity,
                        qa_similarity,
                        title_match,
                        overlap,
                        args.question_threshold,
                        args.qa_threshold,
                        args.weak_threshold,
                    )
                    if hint == "ignore":
                        continue
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO semantic_match(
                            candidate_a, candidate_b, question_similarity, qa_similarity,
                            title_match, context_overlap, model_id, decision_hint
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            left["id"],
                            right["id"],
                            question_similarity,
                            qa_similarity,
                            1 if title_match else 0,
                            overlap,
                            args.model,
                            hint,
                        ),
                    )
                    saved += 1
                    if hint in {"semantic_duplicate", "possible_better_answer", "conflict_review"}:
                        cluster_rows = [left, right]
                        create_cluster(
                            conn,
                            "semantic_duplicate" if hint == "semantic_duplicate" else hint,
                            cluster_rows,
                            f"semantic {hint}",
                            question_similarity,
                        )

    print(
        json.dumps(
            {
                "semantic_matches": saved,
                "candidates": len(rows),
                "model": args.model,
                "seconds": round(time.time() - started, 2),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def delete_semantic_matches_for_rows(
    conn: sqlite3.Connection, rows: list[sqlite3.Row], model_id: str
) -> None:
    ids = [row["id"] for row in rows]
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        if not chunk:
            continue
        placeholders = ",".join("?" for _ in chunk)
        conn.execute(
            f"""
            DELETE FROM semantic_match
            WHERE model_id = ?
              AND candidate_a IN ({placeholders})
              AND candidate_b IN ({placeholders})
            """,
            [model_id, *chunk, *chunk],
        )


def delete_semantic_clusters_for_rows(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> None:
    ids = [row["id"] for row in rows]
    semantic_cluster_types = ("semantic_duplicate", "possible_better_answer", "conflict_review")
    for start in range(0, len(ids), 500):
        chunk = ids[start : start + 500]
        if not chunk:
            continue
        id_placeholders = ",".join("?" for _ in chunk)
        type_placeholders = ",".join("?" for _ in semantic_cluster_types)
        conn.execute(
            f"""
            DELETE FROM duplicate_cluster
            WHERE cluster_type IN ({type_placeholders})
              AND id IN (
                  SELECT cluster_id FROM cluster_member
                  WHERE candidate_id IN ({id_placeholders})
              )
            """,
            [*semantic_cluster_types, *chunk],
        )


def export_rows(conn: sqlite3.Connection, path: Path, sql: str, params: tuple[object, ...] = ()) -> int:
    conn.row_factory = sqlite3.Row
    rows = (dict(row) for row in conn.execute(sql, params))
    return write_jsonl(path, rows)


def run_export(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = open_review_db(args.review_db)
    conn.row_factory = sqlite3.Row

    accepted_count = export_rows(
        conn,
        out_dir / "accepted_pairs_preview.jsonl",
        """
        SELECT c.*
        FROM candidate_pair c
        WHERE NOT EXISTS (
            SELECT 1 FROM cluster_member m
            JOIN duplicate_cluster d ON d.id = m.cluster_id
            WHERE m.candidate_id = c.id
              AND d.status IN ('exact_duplicate', 'possible_better_answer', 'conflict_review', 'manual_review')
        )
        ORDER BY c.split, c.source_qid, c.id
        """,
    )
    duplicate_count = export_rows(
        conn,
        out_dir / "duplicate_clusters.jsonl",
        """
        SELECT d.id AS cluster_id, d.cluster_type, d.status, d.winner_candidate_id,
               d.created_reason, m.candidate_id, m.role, m.score, m.reason,
               c.question_original, c.answer_original, c.title, c.split, c.source_qid
        FROM duplicate_cluster d
        JOIN cluster_member m ON m.cluster_id = d.id
        JOIN candidate_pair c ON c.id = m.candidate_id
        WHERE d.status IN ('exact_duplicate', 'possible_better_answer', 'manual_review')
        ORDER BY d.id, m.role DESC, c.id
        """,
    )
    conflict_count = export_rows(
        conn,
        out_dir / "conflict_review.jsonl",
        """
        SELECT d.id AS cluster_id, d.cluster_type, d.status, d.winner_candidate_id,
               d.created_reason, m.candidate_id, m.role, m.score, m.reason,
               c.question_original, c.answer_original, c.title, c.split, c.source_qid
        FROM duplicate_cluster d
        JOIN cluster_member m ON m.cluster_id = d.id
        JOIN candidate_pair c ON c.id = m.candidate_id
        WHERE d.status = 'conflict_review'
        ORDER BY d.id, c.id
        """,
    )
    better_count = export_rows(
        conn,
        out_dir / "possible_better_answers.jsonl",
        """
        SELECT d.id AS cluster_id, d.cluster_type, d.status, d.winner_candidate_id,
               d.created_reason, m.candidate_id, m.role, m.score, m.reason,
               c.question_original, c.answer_original, c.title, c.split, c.source_qid
        FROM duplicate_cluster d
        JOIN cluster_member m ON m.cluster_id = d.id
        JOIN candidate_pair c ON c.id = m.candidate_id
        WHERE d.status = 'possible_better_answer'
        ORDER BY d.id, m.role DESC, c.id
        """,
    )

    report = make_export_report(conn)
    report.update(
        {
            "accepted_pairs_preview_rows": accepted_count,
            "duplicate_cluster_rows": duplicate_count,
            "conflict_review_rows": conflict_count,
            "possible_better_answer_rows": better_count,
            "review_db_path": str(args.review_db),
            "out_dir": str(out_dir),
        }
    )
    (out_dir / "squad_it_review_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


def make_export_report(conn: sqlite3.Connection) -> dict[str, object]:
    total_candidates = conn.execute("SELECT COUNT(*) FROM candidate_pair").fetchone()[0]
    clusters_by_status = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT status, COUNT(*) FROM duplicate_cluster GROUP BY status ORDER BY status"
        )
    }
    clusters_by_type = {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT cluster_type, COUNT(*) FROM duplicate_cluster GROUP BY cluster_type ORDER BY cluster_type"
        )
    }
    semantic_matches = conn.execute("SELECT COUNT(*) FROM semantic_match").fetchone()[0]
    return {
        "candidate_pairs": total_candidates,
        "clusters_by_status": clusters_by_status,
        "clusters_by_type": clusters_by_type,
        "semantic_matches": semantic_matches,
        "schema_version": SCHEMA_VERSION,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and review SQuAD-it before Siel import.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build candidate JSONL files and review DB.")
    build.add_argument("--train", type=Path, required=True)
    build.add_argument("--test", type=Path, required=True)
    build.add_argument("--review-db", type=Path, required=True)
    build.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    build.set_defaults(func=run_build)

    semantic = subparsers.add_parser("semantic", help="Run optional semantic paraphrase mining.")
    semantic.add_argument("--review-db", type=Path, required=True)
    semantic.add_argument("--model", default=DEFAULT_MODEL)
    semantic.add_argument("--top-k", type=int, default=10)
    semantic.add_argument("--question-threshold", type=float, default=0.88)
    semantic.add_argument("--qa-threshold", type=float, default=0.90)
    semantic.add_argument("--weak-threshold", type=float, default=0.80)
    semantic.add_argument("--chunk-size", type=int, default=256)
    semantic.add_argument("--batch-size", type=int, default=32)
    semantic.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_HUB_CACHE", DEFAULT_HF_CACHE)),
        help="Hugging Face/SentenceTransformers cache directory. Defaults to D:/hf-cache.",
    )
    semantic.add_argument(
        "--device",
        default="auto",
        help="Embedding device: auto, cpu, cuda, cuda:0, or mps. Defaults to auto.",
    )
    semantic.add_argument(
        "--torch-path",
        type=Path,
        default=Path(os.environ["SIEL_TORCH_PATH"])
        if os.environ.get("SIEL_TORCH_PATH")
        else None,
        help=(
            "Optional site-packages/archive directory containing torch. "
            f"Example: {DEFAULT_TORCH_PATH}"
        ),
    )
    semantic.add_argument("--limit", type=int)
    semantic.add_argument("--split", choices=["train", "test"])
    semantic.set_defaults(func=run_semantic)

    export = subparsers.add_parser("export", help="Export review reports and JSONL files.")
    export.add_argument("--review-db", type=Path, required=True)
    export.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    export.set_defaults(func=run_export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
