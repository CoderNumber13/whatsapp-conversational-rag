"""Process-level native runtime initialisation.

Call :func:`init_native_runtimes` **first**, before importing FAISS, torch or
anything that pulls them in, in any process that uses both — the Streamlit app,
the evaluation scripts, the test session.


The problem
-----------
``faiss-cpu`` from PyPI links LLVM's OpenMP runtime (``libomp140.x86_64.dll``);
``torch`` links Intel's (``libiomp5md.dll``). Two OpenMP runtimes in one process
is unsupported, and the second to *initialise* aborts the interpreter::

    OMP: Error #15: Initializing libomp140.x86_64.dll, but found
    libiomp5md.dll already initialized.

The process dies with **exit code 3** — no traceback, no exception to catch, no
opportunity to degrade gracefully. In a Streamlit app the browser simply loses
the server.

Measured on this environment (faiss-cpu 1.9.0.post1, torch 2.5.1, Windows,
Python 3.12):

===================================================  ==========
sequence                                             result
===================================================  ==========
faiss used, then ``import torch``                    abort
``import torch``, then faiss used                    abort
both imported first, then a *batch* faiss search     abort
both imported first, ``faiss.omp_set_num_threads(1)`` abort
single-query searches only, then torch               abort
``KMP_DUPLICATE_LIB_OK=TRUE``, heavy interleaved use  **ok**
===================================================  ==========

Import ordering, eager loading and single-threading were all tried and none of
them avoids it: whichever runtime initialises second aborts, and there is no
sequence of in-process calls that reliably prevents that. The pipeline survives
today only because of the precise order its lazy imports happen to produce —
an accident, not a design, and one that a batch search or a reordered call
would break.

The workaround, and why it is here
----------------------------------
``KMP_DUPLICATE_LIB_OK=TRUE`` tells the OpenMP runtime to tolerate the
duplicate. Intel documents it as unsafe and unsupported: two runtimes then
coexist, each with its own thread pool, which can oversubscribe cores and in
principle produce incorrect results. It is a genuine compromise and it is
applied here because every safer in-process alternative was measured and does
not work.

It is set in exactly one place, before the libraries load (the variable is read
at OpenMP initialisation, so setting it later has no effect), rather than
scattered across entry points or left to the developer's shell.

**The correct fix is to remove the duplicate runtime, not tolerate it.** Install
a FAISS built against the same OpenMP as torch::

    pip uninstall faiss-cpu
    conda install -c pytorch faiss-cpu

The conda-channel build links MKL/Intel OpenMP, the same runtime torch uses, so
only one is ever loaded and this module's workaround becomes a no-op. That is an
environment change this package cannot make for the user, so the guard stays as
a safety net for the pip-installed combination.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DUPLICATE_OMP_FLAG = "KMP_DUPLICATE_LIB_OK"

_state: dict[str, object] = {"initialised": False, "flag_set_by_us": False}


def init_native_runtimes(*, allow_duplicate_omp: bool = True) -> dict[str, object]:
    """Prepare the process for FAISS + torch coexistence. Idempotent.

    Returns what happened, so callers (and tests) can assert on it rather than
    guess. Imports are best-effort: a process without torch (parser-only work)
    or without FAISS (mock embedder) must still start.
    """
    if _state["initialised"]:
        return dict(_state)

    already = os.environ.get(DUPLICATE_OMP_FLAG)
    if allow_duplicate_omp and already is None:
        # Must precede the first OpenMP initialisation, hence before the imports
        # below. See the module docstring for why no safer option exists.
        os.environ[DUPLICATE_OMP_FLAG] = "TRUE"
        _state["flag_set_by_us"] = True
        logger.debug(
            "%s=TRUE: faiss-cpu and torch link different OpenMP runtimes. "
            "Install faiss from the pytorch conda channel to remove the need.",
            DUPLICATE_OMP_FLAG,
        )

    present: dict[str, bool] = {}
    for name in ("faiss", "torch"):
        try:
            __import__(name)
            present[name] = True
        except Exception as exc:  # pragma: no cover - depends on the install
            present[name] = False
            logger.debug("native runtime %s unavailable: %s", name, exc)

    _state.update({"initialised": True, "runtimes": present,
                   "omp_flag": os.environ.get(DUPLICATE_OMP_FLAG)})
    return dict(_state)


def is_initialised() -> bool:
    return bool(_state["initialised"])


def _reset_for_tests() -> None:
    """Forget that initialisation happened. Tests only."""
    _state.update({"initialised": False, "flag_set_by_us": False})
    _state.pop("runtimes", None)
    _state.pop("omp_flag", None)
