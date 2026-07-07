"""Phase 1 gate — unit tests for eval/harness.py.

Builds a tiny synthetic Spider-style database on the fly (no downloads, no
GPU) and checks every behavior the harness promises. Run from the repo root:

    python tests/test_harness.py

Exits 0 with 'ALL TESTS PASSED' or raises on the first failure.
"""

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.harness import evaluate, exact_match, execute_sql, execution_match  # noqa: E402


def build_fixture(root: Path) -> Path:
    """Create database/concert_singer/concert_singer.sqlite with known rows."""
    db_dir = root / "database" / "concert_singer"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "concert_singer.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE stadium (Stadium_ID INTEGER PRIMARY KEY, Name TEXT, Capacity INTEGER);
        CREATE TABLE singer  (Singer_ID INTEGER PRIMARY KEY, Name TEXT, Stadium_ID INTEGER);
        INSERT INTO stadium VALUES (1,'Alpha Arena',50000),(2,'Beta Bowl',30000),(3,'Gamma Ground',30000);
        INSERT INTO singer  VALUES (1,'Asha',1),(2,'Bilal',1),(3,'Chitra',2),(4,'Divya',2);
        """
    )
    conn.commit()
    conn.close()
    return root / "database"


def run_tests() -> None:
    tmp = Path(tempfile.mkdtemp())
    db_root = build_fixture(tmp)
    db = db_root / "concert_singer" / "concert_singer.sqlite"

    # --- exact match normalization
    assert exact_match("SELECT count(*) FROM singer;", "select   count(*)\nfrom singer")
    assert not exact_match("SELECT Name FROM singer", "SELECT Name FROM stadium")
    print("ok: exact-match normalization (case/whitespace/semicolon)")

    # --- execution: identical results, different SQL text
    r = execution_match(db, "SELECT count(*) FROM singer",
                            "SELECT count(Singer_ID) FROM singer")
    assert r["match"] and not r["pred_error"]
    print("ok: execution match despite different SQL text (the metric's whole point)")

    # --- execution: wrong answer
    r = execution_match(db, "SELECT count(*) FROM singer",
                            "SELECT count(*) FROM stadium")
    assert not r["match"] and not r["pred_error"]
    print("ok: wrong result detected")

    # --- multiset comparison when no ORDER BY (row order must not matter)
    r = execution_match(db, "SELECT Name FROM stadium",
                            "SELECT Name FROM stadium ORDER BY Name DESC")
    assert r["match"]
    print("ok: order-insensitive when gold has no ORDER BY")

    # --- order-sensitive when gold HAS ORDER BY
    r = execution_match(db, "SELECT Name FROM stadium ORDER BY Capacity ASC, Name ASC",
                            "SELECT Name FROM stadium ORDER BY Capacity DESC, Name ASC")
    assert not r["match"]
    print("ok: order-sensitive when gold has ORDER BY")

    # --- broken predicted SQL counts as incorrect, not a crash
    r = execution_match(db, "SELECT count(*) FROM singer",
                            "SELEC cont(*) FRUM singer")
    assert not r["match"] and r["pred_error"]
    print("ok: invalid predicted SQL -> incorrect, no crash")

    # --- broken GOLD SQL reported as data problem
    r = execution_match(db, "SELECT * FROM no_such_table", "SELECT 1")
    assert r["gold_error"]
    print("ok: broken gold SQL flagged as gold_error")

    # --- read-only guard: predicted DML must not mutate the fixture
    r = execution_match(db, "SELECT count(*) FROM singer", "DELETE FROM singer")
    assert not r["match"]
    rows = execute_sql(db, "SELECT count(*) FROM singer")
    assert rows == [(4,)], "read-only mode failed: fixture was mutated!"
    print("ok: read-only mode blocks destructive predicted SQL")

    # --- end-to-end evaluate() aggregation
    data_rows = [
        {"db_id": "concert_singer", "question": "How many singers?",
         "query": "SELECT count(*) FROM singer"},
        {"db_id": "concert_singer", "question": "Stadium names?",
         "query": "SELECT Name FROM stadium"},
        {"db_id": "concert_singer", "question": "Singers at Alpha Arena?",
         "query": "SELECT T1.Name FROM singer T1 JOIN stadium T2 ON T1.Stadium_ID=T2.Stadium_ID WHERE T2.Name='Alpha Arena'"},
        {"db_id": "concert_singer", "question": "Max capacity?",
         "query": "SELECT max(Capacity) FROM stadium"},
    ]
    predictions = [
        "SELECT count(*) FROM singer",                       # em yes, ex yes
        "SELECT Name FROM stadium ORDER BY Name",            # em no,  ex yes
        "SELECT Name FROM singer WHERE Stadium_ID = 999",    # em no,  ex no
        "totally not sql",                                   # em no,  ex no (pred_error)
    ]
    report = evaluate(data_rows, predictions, db_root)
    assert report["n"] == 4
    assert abs(report["exact_match"] - 0.25) < 1e-9, report["exact_match"]
    assert abs(report["execution_accuracy"] - 0.50) < 1e-9, report["execution_accuracy"]
    assert abs(report["pred_error_rate"] - 0.25) < 1e-9
    assert report["gold_errors"] == 0
    print("ok: evaluate() aggregation (em=0.25, ex=0.50, pred_err=0.25)")

    # --- misaligned files must fail loudly, not silently zip-truncate
    try:
        evaluate(data_rows, predictions[:2], db_root)
        raise RuntimeError("misalignment was not caught")
    except AssertionError:
        print("ok: misaligned data/prediction files fail loudly")

    print("\nALL TESTS PASSED — Phase 1 gate criteria met.")


if __name__ == "__main__":
    run_tests()
