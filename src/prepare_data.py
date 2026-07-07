"""Phase 0 — Spider data preparation for text-to-sql-qlora-v2.

Reads the official Spider release (train_spider.json, dev.json, tables.json),
serializes each example's database schema into the prompt, and writes:

    data/train.jsonl   (90% of train_spider.json, seeded shuffle)
    data/val.jsonl     (10% of train_spider.json)
    data/dev.jsonl     (dev.json — HELD OUT. Never used for training or
                        model selection. Touched only in Phase 5.)

Each jsonl row: {"db_id", "question", "schema", "query", "messages"}
`messages` is a chat-format list ready for tokenizer.apply_chat_template().

Usage:
    python src/prepare_data.py --spider_dir /path/to/spider --out_dir data
"""

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "write a single SQLite-compatible SQL query that answers the question. "
    "Output only the SQL query."
)

USER_TEMPLATE = "### Database schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:"

SEED = 42
VAL_FRACTION = 0.10


def serialize_schema(db: dict) -> str:
    """Serialize one tables.json entry into a compact CREATE-TABLE-like text.

    Format per table:
        table_name(col1 TYPE, col2 TYPE, ...)
        primary key: col
        foreign key: col -> other_table.other_col
    Kept deliberately close to v1's serialization so token counts stay
    comparable (v1 audit: max 3,395 tokens with schema).
    """
    table_names = db["table_names_original"]
    column_names = db["column_names_original"]  # [ [table_idx, col_name], ... ]
    column_types = db["column_types"]
    primary_keys = set(db.get("primary_keys", []))
    foreign_keys = db.get("foreign_keys", [])

    # Group columns by table (skip the global "*" column at index 0)
    cols_by_table = {i: [] for i in range(len(table_names))}
    pk_by_table = {i: [] for i in range(len(table_names))}
    for col_idx, (t_idx, col_name) in enumerate(column_names):
        if t_idx == -1:
            continue
        cols_by_table[t_idx].append(f"{col_name} {column_types[col_idx].upper()}")
        if col_idx in primary_keys:
            pk_by_table[t_idx].append(col_name)

    fk_lines = []
    for src, dst in foreign_keys:
        s_t, s_c = column_names[src]
        d_t, d_c = column_names[dst]
        fk_lines.append(
            f"foreign key: {table_names[s_t]}.{s_c} -> {table_names[d_t]}.{d_c}"
        )

    lines = []
    for i, name in enumerate(table_names):
        lines.append(f"{name}({', '.join(cols_by_table[i])})")
        if pk_by_table[i]:
            lines.append(f"  primary key: {', '.join(pk_by_table[i])}")
    lines.extend(fk_lines)
    return "\n".join(lines)


def build_example(ex: dict, schemas: dict) -> dict:
    schema_text = schemas[ex["db_id"]]
    user_msg = USER_TEMPLATE.format(schema=schema_text, question=ex["question"])
    return {
        "db_id": ex["db_id"],
        "question": ex["question"],
        "schema": schema_text,
        "query": ex["query"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": ex["query"]},
        ],
    }


def write_jsonl(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>5} rows -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spider_dir", required=True,
                    help="Directory containing train_spider.json, dev.json, tables.json")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    spider = Path(args.spider_dir)
    out = Path(args.out_dir)

    for fname in ("train_spider.json", "dev.json", "tables.json"):
        if not (spider / fname).exists():
            raise FileNotFoundError(
                f"{fname} not found in {spider}. Point --spider_dir at the "
                "official Spider release directory."
            )

    tables = json.loads((spider / "tables.json").read_text(encoding="utf-8"))
    schemas = {db["db_id"]: serialize_schema(db) for db in tables}
    print(f"Serialized schemas for {len(schemas)} databases.")

    train_raw = json.loads((spider / "train_spider.json").read_text(encoding="utf-8"))
    dev_raw = json.loads((spider / "dev.json").read_text(encoding="utf-8"))

    train_rows = [build_example(ex, schemas) for ex in train_raw]
    dev_rows = [build_example(ex, schemas) for ex in dev_raw]

    # Seeded 90/10 split — same seed every run, so the split is reproducible.
    rng = random.Random(SEED)
    indices = list(range(len(train_rows)))
    rng.shuffle(indices)
    n_val = int(len(indices) * VAL_FRACTION)
    val_idx = set(indices[:n_val])

    train_split = [r for i, r in enumerate(train_rows) if i not in val_idx]
    val_split = [r for i, r in enumerate(train_rows) if i in val_idx]

    print("Writing outputs:")
    write_jsonl(train_split, out / "train.jsonl")
    write_jsonl(val_split, out / "val.jsonl")
    write_jsonl(dev_rows, out / "dev.jsonl")

    print("\nDone. Reminder: data/dev.jsonl is HELD OUT until Phase 5.")


if __name__ == "__main__":
    main()
