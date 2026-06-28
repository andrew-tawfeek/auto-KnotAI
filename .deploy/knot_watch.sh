#!/usr/bin/env bash
# Watch the running tabulation; when it ends (sustained absence of the
# tabulate_db.py process), trigger the publish step exactly once.
set -uo pipefail

LOG=/tmp/knot_watch.log
LOCK=/tmp/knot_watch.lock
PATTERN="tabulate_db.py"
PUBLISH="$HOME/knot_publish.sh"

exec >>"$LOG" 2>&1
# Singleton: refuse to start a second watcher.
exec 9>"$LOCK"
if ! flock -n 9; then echo "$(date -u) [watch] another watcher holds the lock; exiting"; exit 0; fi

echo "=================================================="
echo "[watch] start $(date -u)"

# Phase 1: confirm the run is actually active (avoid firing during a transient
# gap). Wait up to ~10 min to see the process at least once.
seen=0
for _ in $(seq 1 120); do
  if pgrep -f "$PATTERN" >/dev/null 2>&1; then seen=1; echo "[watch] tabulation detected $(date -u)"; break; fi
  sleep 5
done
if [ "$seen" -eq 0 ]; then
  echo "[watch] never observed a running tabulation in 10min; aborting (launch the run first)."
  exit 1
fi

# Phase 2: wait for sustained absence. Require 6 consecutive 15s misses (90s)
# so sub-second gaps between segments/subcommands don't count as 'finished'.
absent=0
while true; do
  if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    absent=0
  else
    absent=$((absent + 1))
    echo "[watch] tabulation absent ${absent}/6 $(date -u)"
    if [ "$absent" -ge 6 ]; then break; fi
  fi
  sleep 15
done

echo "[watch] tabulation ended (90s sustained absence); launching publish $(date -u)"
bash "$PUBLISH"
echo "[watch] done $(date -u)"
