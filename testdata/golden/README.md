# Golden Dataset

The production target is:

- 200 known questions.
- 200 paraphrases.
- 100 out-of-domain questions.
- 50 ambiguous questions.
- 50 intentional contradictions.
- 50 prompt-injection fixtures.
- 500 synthetic deduplication pairs.

The checked-in JSONL files are seed fixtures. Use `lab/generate_golden.py` to
expand them before calibration and CI regression thresholds are enforced.

