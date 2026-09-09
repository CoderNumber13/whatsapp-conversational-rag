"""Test-session setup.

Must run before any test module imports faiss or torch, which is why it lives in
conftest.py rather than in a fixture.
"""

from __future__ import annotations

# faiss-cpu and torch link different OpenMP runtimes and the second to
# initialise aborts the interpreter (exit 3, no traceback). src.runtime holds
# the single documented mitigation and the evidence for why no safer in-process
# option exists; the test session uses the same code path the app does rather
# than setting a flag of its own.
from src.runtime import init_native_runtimes

init_native_runtimes()
