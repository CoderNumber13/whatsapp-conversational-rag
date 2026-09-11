"""Classifying LLM provider failures for the UI.

Two questions the demo needs answered about any provider error:

* **Is it worth retrying?** A 503 "overloaded" clears in seconds. A 429 quota
  exhaustion does not clear today at all, so retrying it just burns demo time
  and makes the app look hung. They must not be treated alike.
* **What should the user do?** A stack trace is useless in front of an
  audience; "switch GEMINI_MODEL in .env" is actionable.

Kept out of ``app.py`` so both can be unit tested without Streamlit, and out of
the provider clients so no retry policy is imposed on callers that want to see
raw failures — the evaluation harness deliberately records them rather than
masking them.
"""

from __future__ import annotations

# Transient: the request may well succeed moments later.
_TRANSIENT_MARKERS = ("503", "unavailable", "overloaded", "high demand",
                      "timed out", "timeout", "502", "504",
                      "connection", "temporarily",
                      # the client's own wording for a network failure
                      "cannot reach", "unreachable")

# Permanent for this run: retrying cannot help.
_QUOTA_MARKERS = ("429", "resource_exhausted", "quota", "rate limit")
_BAD_MODEL_MARKERS = ("404", "not_found", "no longer available")
_AUTH_MARKERS = ("401", "403", "api key not valid", "permission_denied",
                 "api_key", "not set")


def _has(text: str, markers) -> bool:
    low = text.lower()
    return any(m in low for m in markers)


def is_transient(exc: Exception) -> bool:
    """Should this be retried?

    Quota, auth and bad-model errors are checked first: a 429 body can mention
    "rate limits" documentation links, and must never be mistaken for a blip.
    """
    text = str(exc)
    if _has(text, _QUOTA_MARKERS) or _has(text, _AUTH_MARKERS) \
            or _has(text, _BAD_MODEL_MARKERS):
        return False
    return _has(text, _TRANSIENT_MARKERS)


def friendly_message(exc: Exception) -> str:
    """A sentence the person running the demo can act on."""
    text = str(exc)
    if _has(text, _QUOTA_MARKERS):
        return ("The model's request quota is exhausted. Free Gemini tiers allow "
                "only about 20 requests per day per model — switch GEMINI_MODEL "
                "in .env to another model, wait for the daily reset, or use a "
                "paid key.")
    if _has(text, _BAD_MODEL_MARKERS):
        return ("That model name is not available to this API key. Update "
                "GEMINI_MODEL in .env (see README for how to list valid names).")
    if _has(text, _AUTH_MARKERS):
        return "No valid API key configured. Set GEMINI_API_KEY in .env and restart."
    if _has(text, ("cannot reach", "unreachable", "connection")):
        return ("Could not reach the model API after several retries. Check the "
                "network connection, then try again.")
    if _has(text, _TRANSIENT_MARKERS):
        return ("The model is temporarily overloaded and did not recover after "
                "several retries. Try again in a moment, or switch GEMINI_MODEL "
                "in .env to a less busy model.")
    return text
