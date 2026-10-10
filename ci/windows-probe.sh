#!/usr/bin/env bash
# Temporary probe (D-623): why does life_sleep fail after life_one_day on Windows with the real window?
set -u
export PYTHONPATH="$PWD/ci"
L=tests/integration/life
OPTS=(-q -p no:cacheprovider -m "not live" -rf --tb=short --log-level=INFO
      --log-format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s" --log-date-format="%H:%M:%S")
for i in 1 2 3 4; do
  echo "-- job $PROBE_INDEX run $i"
  uv run python -m pytest "${OPTS[@]}" -p probe_notoast $L/test_life_one_day.py $L/test_life_sleep.py > "out-$i.txt" 2>&1
  code=$?
  tail -n 1 "out-$i.txt"
  if [ "$code" -ne 0 ]; then
    echo "=== FAILED RUN $i: the report"
    n=$(grep -a -n "= FAILURES =" "out-$i.txt" | head -1 | cut -d: -f1)
    tail -n +"${n:-1}" "out-$i.txt" | grep -a -v "httpx" | head -n 260
  fi
done
