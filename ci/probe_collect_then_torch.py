"""Collect the given test files (imports their modules), then import torch in the same process."""

import sys

import pytest

code = pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", *sys.argv[1:]])
sys.stdout.write(f"collect exit {code}\n")
sys.stdout.flush()
import torch  # noqa: E402

sys.stdout.write(f"torch imported {torch.__version__}\n")
