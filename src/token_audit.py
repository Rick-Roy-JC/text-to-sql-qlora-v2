"""Phase 0 — token audit for text-to-sql-qlora-v2.

Lesson #1 from v1: a stale max_seq_length=768 would have silently truncated
most schemas. This script re-verifies, against the *actual prepared data*,
that MAX_SEQ_LENGTH=3584 covers every example — before any training config
is written. Runs on CPU; only the tokenizer is downloaded (~a few MB), never
the model weights.

Usage:
    python src/token_audit.py --data_dir data --max_seq_length 3584
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


def audit_file(path: Path, tokenizer, max_seq_length: int) -> dict:
    lengths = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=False
            )
            lengths.append(len(ids))

    lengths.sort()
    n = len(lengths)
    pct = lambda p: lengths[min(n - 1, int(n * p))]
    over = sum(1 for L in lengths if L > max_seq_length)
    return {
        "file": path.name,
        "n": n,
        "max": lengths[-1],
        "p99": pct(0.99),
        "p95": pct(0.95),
        "median": pct(0.50),
        "over_limit": over,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--max_seq_length", type=int, default=3584)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"Tokenizer: {MODEL_ID}")
    print(f"Auditing against MAX_SEQ_LENGTH = {args.max_seq_length}\n")

    failed = False
    for name in ("train.jsonl", "val.jsonl", "dev.jsonl"):
        path = Path(args.data_dir) / name
        if not path.exists():
            print(f"  {name}: MISSING (run prepare_data.py first)")
            failed = True
            continue
        s = audit_file(path, tokenizer, args.max_seq_length)
        status = "OK" if s["over_limit"] == 0 else f"!! {s['over_limit']} OVER LIMIT"
        print(
            f"  {s['file']:<12} n={s['n']:<6} max={s['max']:<5} "
            f"p99={s['p99']:<5} p95={s['p95']:<5} median={s['median']:<5} [{status}]"
        )
        if s["over_limit"] > 0:
            failed = True

    print()
    if failed:
        print("AUDIT FAILED — do not write a training config until this passes.")
        raise SystemExit(1)
    print("AUDIT PASSED — MAX_SEQ_LENGTH is safe for all splits.")


if __name__ == "__main__":
    main()
