"""Startup checks for the Streamlit app.

Anaconda's **base** environment ships ``streamlit`` but not ``faiss``, so
``streamlit run app.py`` without activating ``convmem`` starts the app
perfectly and then fails deep inside ingestion with::

    Ingestion failed - ModuleNotFoundError: No module named 'faiss'

Base is Python 3.14, which has no faiss wheel at all, so "just pip install it"
is not a fix either. The only fix is to launch with the project environment.

A missing native dependency is a *launch* mistake. Catching it at page load and
naming the interpreter in use turns a confusing mid-demo failure into a clear
instruction, rather than something the user discovers after uploading a file.
"""

from __future__ import annotations

import sys

LAUNCHER = "run_app.bat"


def missing_runtimes(runtimes: dict[str, bool], *, required=("faiss",)) -> list[str]:
    """Which required native runtimes failed to import."""
    return [name for name in required if not runtimes.get(name)]


def wrong_environment_message(missing: list[str]) -> tuple[str, str]:
    """(error, guidance) markdown for a launch under the wrong interpreter."""
    error = (
        f"**Missing required package(s): {', '.join(missing)}**\n\n"
        f"This app is running under:\n\n`{sys.executable}`\n\n"
        "which is almost certainly Anaconda's **base** environment. Base has "
        "`streamlit` but not `faiss`, so the app starts and then fails at "
        "ingestion. Base is Python 3.14, which has no faiss wheel at all, so "
        "installing it there will not work either."
    )
    guidance = (
        "**Fix — launch with the project environment instead.** From the "
        f"project folder:\n\n"
        f"```\n{LAUNCHER}\n```\n\n"
        "or, equivalently:\n\n"
        "```\n"
        'cd /d "C:\\Users\\asus\\Documents\\Whatsapp Advanced RAG Project"\n'
        "D:\\anaconda3\\envs\\convmem\\python.exe -m streamlit run app.py\n"
        "```\n\n"
        "If the `convmem` environment does not exist yet, see **Setup** in "
        "`README.md`."
    )
    return error, guidance
