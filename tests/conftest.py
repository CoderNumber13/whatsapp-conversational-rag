"""Test-session setup.

Must run before any test module imports faiss or torch, which is why it lives in
conftest.py rather than in a fixture.
"""

from __future__ import annotations

import os

# faiss-cpu and torch each bundle their own OpenMP runtime (libiomp5md.dll on
# Windows). Loading the second one into a process where the first is already
# running aborts the interpreter:
#
#   Fatal Python error: Aborted ... torch/__init__.py line 367
#
# It does not reproduce in a fresh process importing both — it needs faiss to
# have been *used* first, which is what happens here: the vector-index tests run
# long before the cross-encoder tests load torch.
#
# This flag tells the OpenMP runtime to tolerate the duplicate. Intel documents
# it as potentially unsafe (two runtimes can oversubscribe threads), which is
# acceptable in a single-threaded test process and is why it is set only here
# and never in application code. If the reranker is ever wired into the
# Streamlit app alongside FAISS, this needs revisiting properly — the real fix
# is a faiss build that shares torch's OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
