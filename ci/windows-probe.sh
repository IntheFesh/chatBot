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
  0)  echo "== 0: all earlier files, then the torch test"
      uv run python -m pytest -q -p no:cacheprovider -p probe_plugin -m "not live" "${before[@]}" "$TORCH_TEST" ;;
  1)  echo "== 1: collect only the earlier files, then import torch in the same process"
      uv run python ci/probe_collect_then_torch.py "${before[@]}" ;;
  2)  echo "== 2: the torch test alone"
      uv run python -m pytest -q -p no:cacheprovider -p probe_plugin "$TORCH_TEST" ;;
  3)  echo "== 3: first half of the earlier files, then the torch test"
      uv run python -m pytest -q -p no:cacheprovider -p probe_plugin -m "not live" "${before[@]:0:$half}" "$TORCH_TEST" ;;
  4)  echo "== 4: second half of the earlier files, then the torch test"
      uv run python -m pytest -q -p no:cacheprovider -p probe_plugin -m "not live" "${before[@]:$half}" "$TORCH_TEST" ;;
  5)  echo "== 5: the whole shard in order, as in CI"
      uv run python -m pytest -q -p no:cacheprovider -p probe_plugin -m "not live" "${files[@]}" ;;
esac
