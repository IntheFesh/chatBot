#!/usr/bin/env bash
# Temporary probe (D-623): DEBUG log of the evening wake-up test around the wake-up, pass and fail.
set -u
export PYTHONPATH="$PWD/ci"
L=tests/integration/life
OPTS=(-q -p no:cacheprovider -m "not live" -rA --tb=short --log-level=DEBUG
      --log-format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s" --log-date-format="%H:%M:%S")
for i in 1 2 3 4 5 6; do
  echo "-- job $PROBE_INDEX run $i"
  uv run python -m pytest "${OPTS[@]}" -p probe_notoast -p probe_nowindow $L/test_life_one_day.py $L/test_life_sleep.py > "out-$i.txt" 2>&1
  tail -n 1 "out-$i.txt"
  echo "=== the evening test, run $i (from the wake-up on)"
  awk '/^_+ test_a_wake_up_in_the_evening/ {on=1} /^_+ test_a_wake_up_after_midnight/ {on=0} on' "out-$i.txt" \
    | grep -a -v "component_started\|component_stopped\|alembic\|sendtyping\|asyncio\|sqlalchemy\|aiosqlite\|httpcore" \
    | awk '/machine_resumed/ {go=1} go' | head -n 150
done
