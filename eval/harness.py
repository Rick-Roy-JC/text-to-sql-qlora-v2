"""Phase 1 — evaluation harness for text-to-sql-qlora-v2.

Two metrics, computed per example and aggregated:

1. **Exact match (normalized string):** lowercase, whitespace collapsed,
   trailing semicolon stripped. NOTE: this is deliberately simpler than
   Spider's official component-based exact match — it will *under*-count
   correct-but-differently-written SQL. Execution accuracy is the headline
   metric; exact match is the cheap sanity companion.

2. **Execution accuracy:** run gold and predicted SQL against the example's
   sqlite database (read-only). Correct if result sets match — order-sensitive
   when the gold query contains ORDER BY, multiset comparison otherwise.
   A predicted query that errors or times out counts as incorrect. A *gold*
   query that errors is a data problem and is reported separately, not
   counted against the model.

Usage:
    python eval/harness.py \
        --data data/dev.jsonl \
        --predictions preds.txt \
        --db_root /path/to/spider/database

`--predictions` is a plain text file, one SQL query per line, aligned with
the rows of `--data`. Runs entirely on CPU.
"""

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

# Abort any single query after ~this many sqlite VM steps (guards against
# pathological predicted queries, e.g. accidental cross joins).
PROGRESS_STEP_LIMIT = 20_000_000
FLOAT_PRECISION = 6


# ---------------------------------------------------------------- exact match

def normalize_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql.lower()


def exact_match(gold: str, pred: str) -> bool:
    return normalize_sql(gold) == normalize_sql(pred)


# --------------------------------------------------------- execution accuracy

class QueryError(Exception):
    pass


def _normalize_value(v):
    if isinstance(v, float):
        return round(v, FLOAT_PRECISION)
    return v


def execute_sql(db_path: Path, sql: str):
    """Execute SQL read-only against a sqlite file; return list of row tuples."""
    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:
        raise QueryError(f"cannot open db: {e}") from e
    try:
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        steps = 0

        def guard():
            nonlocal steps
            steps += 1
            return 1 if steps > PROGRESS_STEP_LIMIT // 1000 else 0

        conn.set_progress_handler(guard, 1000)
        cur = conn.execute(sql)
        rows = cur.fetchall()
        return [tuple(_normalize_value(v) for v in row) for row in rows]
    except sqlite3.Error as e:
        raise QueryError(str(e)) from e
    finally:
        conn.close()


def execution_match(db_path: Path, gold: str, pred: str) -> dict:
    """Compare execution results. Returns dict with keys:
    match (bool), gold_error (bool), pred_error (bool)."""
    try:
        gold_rows = execute_sql(db_path, gold)
    except QueryError:
        return {"match": False, "gold_error": True, "pred_error": False}

    try:
        pred_rows = execute_sql(db_path, pred)
    except QueryError:
        return {"match": False, "gold_error": False, "pred_error": True}

    if "order by" in gold.lower():
        match = gold_rows == pred_rows
    else:
        match = Counter(gold_rows) == Counter(pred_rows)
    return {"match": match, "gold_error": False, "pred_error": False}


# ----------------------------------------------------------------- evaluation

def evaluate(data_rows: list, predictions: list, db_root: Path) -> dict:
    assert len(data_rows) == len(predictions), (
        f"{len(data_rows)} data rows vs {len(predictions)} predictions — "
        "files are misaligned"
    )
    n = len(data_rows)
    em = 0
    ex = 0
    pred_errors = 0
    gold_errors = 0
    per_example = []

    for row, pred in zip(data_rows, predictions):
        gold = row["query"]
        db_path = db_root / row["db_id"] / f"{row['db_id']}.sqlite"
        is_em = exact_match(gold, pred)
        res = execution_match(db_path, gold, pred)
        em += is_em
        ex += res["match"]
        pred_errors += res["pred_error"]
        gold_errors += res["gold_error"]
        per_example.append({
            "db_id": row["db_id"],
            "question": row["question"],
            "gold": gold,
            "predicted": pred,
            "exact_match": is_em,
            "execution_match": res["match"],
            "pred_error": res["pred_error"],
            "gold_error": res["gold_error"],
        })

    return {
        "n": n,
        "exact_match": em / n if n else 0.0,
        "execution_accuracy": ex / n if n else 0.0,
        "pred_error_rate": pred_errors / n if n else 0.0,
        "gold_errors": gold_errors,
        "per_example": per_example,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl with db_id/question/query")
    ap.add_argument("--predictions", required=True, help="txt, one SQL per line")
    ap.add_argument("--db_root", required=True, help="Spider database/ directory")
    ap.add_argument("--out", default=None, help="optional per-example report jsonl")
    args = ap.parse_args()

    data_rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    predictions = Path(args.predictions).read_text(encoding="utf-8").splitlines()
    predictions = [p for p in predictions]  # keep empty lines: they count as wrong

    report = evaluate(data_rows, predictions, Path(args.db_root))

    print(f"n                   : {report['n']}")
    print(f"exact match         : {report['exact_match']:.4f}")
    print(f"execution accuracy  : {report['execution_accuracy']:.4f}")
    print(f"pred error rate     : {report['pred_error_rate']:.4f}")
    if report["gold_errors"]:
        print(f"!! gold_errors      : {report['gold_errors']} (data problem — inspect)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in report["per_example"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"per-example report -> {args.out}")


if __name__ == "__main__":
    main()
