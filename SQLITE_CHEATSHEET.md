# `mosaics.db` — SQLite Cheatsheet

Quick reference for parsing through `mosaics.db` from the `sqlite3` CLI.

## Launching

```bash
# Open the database
sqlite3 mosaics.db

# Open read-only (safer when poking around the live file)
sqlite3 -readonly mosaics.db

# Run one query and exit
sqlite3 mosaics.db "SELECT COUNT(*) FROM mosaics;"

# Pretty output by default
sqlite3 -header -column mosaics.db
```

Exit the shell with `.quit` or `Ctrl-D`.

## Schema at a glance

Single table, `mosaics`, with **3,195,445 rows** (as of the current snapshot):

```sql
CREATE TABLE mosaics (
    mosaic                TEXT PRIMARY KEY,    -- nested-list string, e.g. "[[0,2,1],[2,8,4],[3,4,0]]"
    dimension             INTEGER NOT NULL,    -- mosaic grid dimension (n for an n x n board)
    is_suitably_connected INTEGER NOT NULL,    -- 0/1
    num_crossings         INTEGER NOT NULL,    -- count of crossing tiles
    has_crossing          INTEGER NOT NULL,    -- 0/1
    num_components        INTEGER,             -- NULL when not suitably connected
    is_unknot             INTEGER,             -- 0/1, NULL when undefined
    pd_code               TEXT                 -- planar diagram code as nested-list string, may be NULL/""
);

-- Indexes:
--   sqlite_autoindex_mosaics_1  (PRIMARY KEY on mosaic)
--   idx_dimension               (dimension)
--   idx_unknot                  (dimension, is_unknot)
--   idx_components              (dimension, num_components)
```

Row distribution by `dimension`:

| dim | rows    | dim | rows   |
|-----|---------|-----|--------|
| 3   | 354     | 12  | 60,000 |
| 4   | 60,782  | 13  | 60,000 |
| 5   | 534,309 | 14  | 60,000 |
| 6   | 400,000 | 15  | 60,000 |
| 7   | 400,000 | 16  | 60,000 |
| 8   | 400,000 | 17  | 60,000 |
| 9   | 400,000 | 18  | 60,000 |
| 10  | 400,000 | 19  | 60,000 |
| 11  | 60,000  | 20  | 60,000 |

## Dot-commands (shell builtins, no semicolon)

```text
.tables                       -- list tables
.schema                       -- show all CREATE statements
.schema mosaics               -- schema for one table
.indexes mosaics              -- indexes on the table
.databases                    -- list attached DBs and file paths
.headers on                   -- show column headers in output
.mode column                  -- aligned columns (good for small results)
.mode box                     -- bordered table (sqlite3 >= 3.33)
.mode csv                     -- CSV output
.mode markdown                -- Markdown table (great for pasting)
.mode line                    -- one field per line (best for wide rows)
.width 40 6 4 4 4 4 4 60      -- set column widths for column mode
.nullvalue NULL               -- print NULL instead of empty
.timer on                     -- show query timings
.output results.csv           -- redirect to file (combine with .mode csv)
.output stdout                -- back to terminal
.read script.sql              -- execute SQL from a file
.dump mosaics                 -- dump table as SQL inserts
.quit
```

Useful one-shot recipe — dump a query to CSV:

```bash
sqlite3 -header -csv mosaics.db \
  "SELECT mosaic, pd_code FROM mosaics WHERE dimension=5 AND is_unknot=0 LIMIT 1000;" \
  > nontrivial_dim5.csv
```

## Counting and basic stats

```sql
-- Total rows
SELECT COUNT(*) FROM mosaics;

-- Rows per dimension (uses idx_dimension)
SELECT dimension, COUNT(*) AS n
FROM mosaics
GROUP BY dimension
ORDER BY dimension;

-- Suitably-connected fraction per dimension
SELECT dimension,
       SUM(is_suitably_connected) AS connected,
       COUNT(*)                   AS total,
       1.0 * SUM(is_suitably_connected) / COUNT(*) AS frac
FROM mosaics
GROUP BY dimension
ORDER BY dimension;

-- Unknot vs non-unknot among suitably-connected mosaics (uses idx_unknot)
SELECT dimension, is_unknot, COUNT(*)
FROM mosaics
WHERE is_suitably_connected = 1
GROUP BY dimension, is_unknot
ORDER BY dimension, is_unknot;

-- Crossing-count distribution at a fixed dimension
SELECT num_crossings, COUNT(*)
FROM mosaics
WHERE dimension = 6
GROUP BY num_crossings
ORDER BY num_crossings;

-- Component-count distribution (uses idx_components)
SELECT dimension, num_components, COUNT(*)
FROM mosaics
WHERE num_components IS NOT NULL
GROUP BY dimension, num_components
ORDER BY dimension, num_components;
```

## Sampling rows

```sql
-- First few rows of dimension 5
SELECT * FROM mosaics WHERE dimension = 5 LIMIT 5;

-- Random sample (slow on big tables — scans)
SELECT * FROM mosaics
WHERE dimension = 8
ORDER BY RANDOM()
LIMIT 10;

-- Faster pseudo-random by hashing the rowid:
SELECT * FROM mosaics
WHERE dimension = 8 AND (rowid % 100000) = 0
LIMIT 10;
```

## Filtering common cases

```sql
-- Non-trivial knots (suitably connected, single component, not the unknot)
SELECT mosaic, pd_code
FROM mosaics
WHERE dimension = 6
  AND is_suitably_connected = 1
  AND num_components = 1
  AND is_unknot = 0
LIMIT 20;

-- Links (multi-component) at dim 7
SELECT mosaic, num_components, pd_code
FROM mosaics
WHERE dimension = 7
  AND num_components >= 2
LIMIT 20;

-- Rows where pd_code wasn't computed
SELECT COUNT(*) FROM mosaics WHERE pd_code IS NULL OR pd_code = '';

-- Minimum-crossing non-unknots per dimension
SELECT dimension, MIN(num_crossings) AS min_xings
FROM mosaics
WHERE is_unknot = 0 AND is_suitably_connected = 1
GROUP BY dimension
ORDER BY dimension;
```

## Looking at one mosaic

```sql
-- Exact match (PRIMARY KEY lookup, instant)
SELECT *
FROM mosaics
WHERE mosaic = '[[0, 2, 1], [2, 8, 4], [3, 4, 0]]';

-- Wide rows print better in line mode:
.mode line
SELECT * FROM mosaics WHERE dimension = 4 LIMIT 1;
```

## Inspecting indexes and query plans

```sql
-- See what an index will do for a query
EXPLAIN QUERY PLAN
SELECT * FROM mosaics WHERE dimension = 8 AND is_unknot = 1;

-- Stats for the table
SELECT * FROM sqlite_stat1;        -- if ANALYZE has been run
ANALYZE mosaics;                   -- refresh stats
```

## Exporting

```sql
-- Markdown table to terminal
.mode markdown
SELECT dimension, COUNT(*) FROM mosaics GROUP BY dimension;

-- CSV to a file
.headers on
.mode csv
.output dim10_unknots.csv
SELECT mosaic, pd_code
FROM mosaics
WHERE dimension = 10 AND is_unknot = 1;
.output stdout
```

From the shell directly:

```bash
sqlite3 -header -csv mosaics.db "SELECT * FROM mosaics WHERE dimension=4;" > dim4.csv
```

## Performance tips

- **Use the indexed columns in `WHERE`**: `dimension`, `(dimension, is_unknot)`, `(dimension, num_components)`, or exact `mosaic` lookups. Anything else triggers a full scan of ~3.2M rows.
- **Avoid `ORDER BY RANDOM()`** on the whole table — it materializes everything. Filter by `dimension` first, or sample via `rowid % k = 0`.
- **`COUNT(*)` is not free** in SQLite (no metadata count); but with `WHERE dimension = N` it'll use the index.
- **Read-only is faster and safer** when you don't intend to write: `sqlite3 -readonly mosaics.db`.
- For long sessions you may want `PRAGMA cache_size = -200000;` (≈200 MB cache).

## Handy PRAGMAs

```sql
PRAGMA table_info(mosaics);        -- columns, types, NOT NULL, default, PK flag
PRAGMA index_list(mosaics);        -- indexes on the table
PRAGMA index_info(idx_unknot);     -- columns inside one index
PRAGMA database_list;              -- attached DB files
PRAGMA page_count; PRAGMA page_size;   -- size info
PRAGMA integrity_check;            -- verify file integrity (slow on 3.4 GB)
```

## Reminder on the `mosaic` and `pd_code` formats

Both columns store nested-list strings (Python `repr` style), e.g.:

```
mosaic   = "[[0, 2, 1, 0], [2, 10, 9, 1], [3, 10, 4, 6], [0, 3, 5, 4]]"
pd_code  = "[[2, 5, 3, 6], [6, 3, 1, 4], [1, 5, 2, 4]]"
```

Inside `sqlite3` they're opaque TEXT — SQLite has no native list operators. For structured manipulation, pipe a query to Python and `ast.literal_eval` / `json.loads`:

```bash
sqlite3 -csv mosaics.db \
  "SELECT mosaic, pd_code FROM mosaics WHERE dimension=5 LIMIT 5;" \
  | python3 -c "import csv,sys,ast; [print(ast.literal_eval(r[0])) for r in csv.reader(sys.stdin)]"
```
