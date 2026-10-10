#!/usr/bin/env bash
# Temporary probe (D-620): which earlier test or import makes `import torch` fail with WinError 1114.
set -u
export PYTHONPATH="$PWD/ci"
TORCH_TEST="tests/unit/test_retrieval_real_backend.py::test_a_broken_model_folder_is_reported_with_the_fix"
uv run python scripts/shard_tests.py --shard 4 --of 5 --platform windows > shard4.txt 2> /dev/null
tr -d '\r' < shard4.txt > shard4.lf
mapfile -t files < shard4.lf
echo "shard 4/5: ${#files[@]} files"
# the files that run before test_retrieval_real_backend.py in the shard
before=()
for f in "${files[@]}"; do
  if [ "$f" = "tests/unit/test_retrieval_real_backend.py" ]; then break; fi
  before+=("$f")
done
echo "files before it: ${#before[@]}"
half=$(( ${#before[@]} / 2 ))
case "$PROBE_INDEX" in
  0)  echo "== 0: first half of the earlier files, then the torch test, with the winrt import watcher (-s)"
      uv run python -m pytest -q -s -p no:cacheprovider -p probe_plugin -m "not live" "${before[@]:0:$half}" "$TORCH_TEST" ;;
esac
