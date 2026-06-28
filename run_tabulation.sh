#!/usr/bin/env bash
#
# One-shot driver: prepopulate the SQLite mosaic table from the legacy CSVs,
# then tabulate every dimension in a range, one at a time, until each is
# exhausted (small dims) or hits its safety cap (large dims).
#
# Usage:
#   ./run_tabulation.sh <dim_lo> <dim_hi> [extra args passed to `generate`]
#
# Examples:
#   ./run_tabulation.sh 3 20
#   ./run_tabulation.sh 3 20 --workers 32 --max-attempts 5000000
#   ./run_tabulation.sh 3 20 --unknot-max-dim 12          # skip spherogram above dim 12
#   DB=run1.db ./run_tabulation.sh 3 8                    # custom database file
#
# Environment:
#   PYTHON   python interpreter to use (default: python3)
#   DB       SQLite database path     (default: mosaics.db)
#   SKIP_PREPOPULATE=1  skip the CSV import step

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <dim_lo> <dim_hi> [extra generate args]" >&2
    exit 2
fi

DIM_LO="$1"
DIM_HI="$2"
shift 2

PYTHON="${PYTHON:-python3}"
DB="${DB:-mosaics.db}"

cd "$(dirname "$0")"

echo "============================================================"
echo " KnotAI mosaic tabulation"
echo "   database : ${DB}"
echo "   dims     : [${DIM_LO}, ${DIM_HI}]"
echo "   python   : ${PYTHON}"
echo "   extra    : $*"
echo "============================================================"

if [[ "${SKIP_PREPOPULATE:-0}" != "1" ]]; then
    echo ">>> Step 1/2: prepopulating from datasets/*.csv"
    "${PYTHON}" tabulate_db.py --db "${DB}" prepopulate
else
    echo ">>> Step 1/2: skipped (SKIP_PREPOPULATE=1)"
fi

echo ">>> Step 2/2: generating dims [${DIM_LO}, ${DIM_HI}]"
"${PYTHON}" tabulate_db.py --db "${DB}" generate "${DIM_LO}" "${DIM_HI}" "$@"

echo ">>> Done. Summary:"
"${PYTHON}" tabulate_db.py --db "${DB}" stats
