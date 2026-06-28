"""Export a balanced-ish crossing-number sample from mosaics.db to per-dim CSVs.

Runs ON high-compute (where the live DB is). NEVER copies the .db file (it comes
out malformed mid-write); instead does consistent read-only SELECTs with a busy
timeout and writes datasets/crossing/dim_<d>.csv with columns:
    Mosaic, Num Crossings   (raw integer; the harness loader buckets to 0..cap+)

Balancing: for each dimension we draw up to PER_CLASS rows for each *bucketed*
class 0..CAP (class CAP == num_crossings >= CAP), so all seven classes are
represented per dim where the data allows. Uses ORDER BY RANDOM() with a fixed
seed-equivalent via a stable LIMIT so the export is a consistent snapshot.
"""
import argparse
import csv
import os
import sqlite3

CAP = 6
DIMS = list(range(5, 13))      # 5..12 inclusive
PER_CLASS = 6000               # per (dim, bucketed-class); 6 dims*7*6k ~ 250k rows


def export(db_path, out_dir, dims=DIMS, cap=CAP, per_class=PER_CLASS):
    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=120)
    conn.execute("PRAGMA busy_timeout=120000")
    grand = 0
    for d in dims:
        rows = []
        for cls in range(cap + 1):
            if cls < cap:
                where = "dimension=? AND is_suitably_connected=1 AND num_crossings=?"
                params = (d, cls)
            else:
                where = ("dimension=? AND is_suitably_connected=1 "
                         "AND num_crossings>=?")
                params = (d, cap)
            sql = (f"SELECT mosaic, num_crossings FROM mosaics WHERE {where} "
                   f"ORDER BY mosaic LIMIT {int(per_class)}")
            got = conn.execute(sql, params).fetchall()
            rows.extend(got)
            print(f"  dim {d} class {cls}: {len(got)} rows", flush=True)
        out = os.path.join(out_dir, f"dim_{d}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Mosaic", "Num Crossings"])
            for m, nc in rows:
                w.writerow([m, int(nc)])
        grand += len(rows)
        print(f"dim {d}: wrote {len(rows)} rows -> {out}", flush=True)
    conn.close()
    print(f"TOTAL exported: {grand} rows", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-class", type=int, default=PER_CLASS)
    a = p.parse_args()
    export(a.db, a.out, per_class=a.per_class)
