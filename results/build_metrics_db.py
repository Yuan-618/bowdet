"""
Load BowDET LOPO evaluation results (per_performer_metrics.csv, seed_metrics.csv)
into a SQLite database and run analysis queries.

Usage:
    python build_metrics_db.py
"""

import csv
import sqlite3
from pathlib import Path

DB_PATH = Path("bowdet_metrics.db")
PER_PERFORMER_CSV = Path("per_performer_metrics.csv")
SEED_CSV = Path("seed_metrics.csv")


def build_database():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE per_performer_metrics (
            config    TEXT,
            seed      INTEGER,
            performer TEXT,
            metric    TEXT,
            precision REAL,
            recall    REAL,
            f1        REAL
        )
    """)

    cur.execute("""
        CREATE TABLE seed_metrics (
            config          TEXT,
            seed            INTEGER,
            point_f1_100ms  REAL,
            iou01_f1        REAL,
            iou025_f1       REAL,
            iou03_f1        REAL
        )
    """)

    with open(PER_PERFORMER_CSV) as f:
        reader = csv.DictReader(f)
        rows = [
            (r["config"], int(r["seed"]), r["performer"], r["metric"],
             float(r["precision"]), float(r["recall"]), float(r["f1"]))
            for r in reader
        ]
    cur.executemany(
        "INSERT INTO per_performer_metrics VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )

    with open(SEED_CSV) as f:
        reader = csv.DictReader(f)
        rows = [
            (r["config"], int(r["seed"]), float(r["point_f1_100ms"]),
             float(r["iou01_f1"]), float(r["iou025_f1"]), float(r["iou03_f1"]))
            for r in reader
        ]
    cur.executemany(
        "INSERT INTO seed_metrics VALUES (?, ?, ?, ?, ?, ?)", rows
    )

    conn.commit()
    return conn


def run_queries(conn):
    cur = conn.cursor()

    print("=" * 60)
    print("1. Average Point F1@100ms per performer (across seeds)")
    print("=" * 60)
    cur.execute("""
        SELECT performer,
               ROUND(AVG(f1), 4) AS avg_f1,
               COUNT(*) AS n_seeds
        FROM per_performer_metrics
        WHERE metric = 'point100'
        GROUP BY performer
        ORDER BY avg_f1 DESC
    """)
    for row in cur.fetchall():
        print(row)

    print()
    print("=" * 60)
    print("2. Best and worst performing performer (Point F1@100ms)")
    print("=" * 60)
    cur.execute("""
        SELECT performer, ROUND(AVG(f1), 4) AS avg_f1
        FROM per_performer_metrics
        WHERE metric = 'point100'
        GROUP BY performer
        ORDER BY avg_f1 DESC
        LIMIT 1
    """)
    print("Best: ", cur.fetchone())

    cur.execute("""
        SELECT performer, ROUND(AVG(f1), 4) AS avg_f1
        FROM per_performer_metrics
        WHERE metric = 'point100'
        GROUP BY performer
        ORDER BY avg_f1 ASC
        LIMIT 1
    """)
    print("Worst:", cur.fetchone())

    print()
    print("=" * 60)
    print("3. Seed-to-seed variance in overall Point F1@100ms")
    print("=" * 60)
    cur.execute("""
        SELECT seed, point_f1_100ms
        FROM seed_metrics
        ORDER BY seed
    """)
    seed_rows = cur.fetchall()
    for row in seed_rows:
        print(row)
    vals = [r[1] for r in seed_rows]
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    print(f"Mean: {mean:.4f}  StdDev: {variance ** 0.5:.4f}")

    print()
    print("=" * 60)
    print("4. Performers with below-average IoU@0.1 F1")
    print("=" * 60)
    cur.execute("""
        SELECT performer, ROUND(AVG(f1), 4) AS avg_f1
        FROM per_performer_metrics
        WHERE metric = 'iou01'
        GROUP BY performer
        HAVING avg_f1 < (
            SELECT AVG(f1) FROM per_performer_metrics WHERE metric = 'iou01'
        )
        ORDER BY avg_f1 ASC
    """)
    for row in cur.fetchall():
        print(row)


if __name__ == "__main__":
    conn = build_database()
    run_queries(conn)
    conn.close()
    print(f"\nDatabase written to {DB_PATH.resolve()}")
