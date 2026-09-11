"""Classifying provider failures: retry the blips, surface the rest.

The distinction is load-bearing for the demo. Retrying a 503 recovers an answer;
retrying a 429 quota error burns the demo clock and cannot ever succeed.
"""

from __future__ import annotations

import pytest

from src.llm.base import LLMError
from src.llm.errors import friendly_message, is_transient

# Real response bodies, trimmed. Gemini's 429 text links to its rate-limit docs,
# which is exactly the trap: it mentions "rate limits" while being permanent.
QUOTA_429 = LLMError(
    'Gemini HTTP 429: {"error": {"code": 429, "message": "You exceeded your '
    'current quota, please check your plan and billing details. For more '
    'information on this error, head to: '
    'https://ai.google.dev/gemini-api/docs/rate-limits. Quota exceeded for '
    'metric: generate_content_free_tier_requests, limit: 20"}}'
)
OVERLOADED_503 = LLMError(
    'Gemini HTTP 503: {"error": {"code": 503, "message": "This model is '
    'currently experiencing high demand. Spikes in demand are usually '
    'temporary. Please try again later.", "status": "UNAVAILABLE"}}'
)
RETIRED_404 = LLMError(
    'Gemini HTTP 404: {"error": {"code": 404, "message": "This model '
    'models/gemini-2.5-flash is no longer available to new users."}}'
)
NO_KEY = LLMError("GEMINI_API_KEY is not set (get one at https://aistudio.google.com/apikey).")
TIMEOUT = LLMError("Gemini request timed out after 120s.")
UNREACHABLE = LLMError("Cannot reach the Gemini API at https://generativelanguage.googleapis.com/v1beta.")


# --- what to retry ----------------------------------------------------

@pytest.mark.parametrize("exc", [OVERLOADED_503, TIMEOUT, UNREACHABLE])
def test_transient_failures_are_retried(exc):
    assert is_transient(exc) is True


@pytest.mark.parametrize("exc", [QUOTA_429, RETIRED_404, NO_KEY])
def test_permanent_failures_are_not_retried(exc):
    assert is_transient(exc) is False


def test_a_quota_error_is_not_mistaken_for_a_blip():
    """The trap: Gemini's 429 body links to 'rate-limits' docs and says 'try
    again', but the daily cap cannot clear during a demo."""
    assert "rate-limits" in str(QUOTA_429)
    assert is_transient(QUOTA_429) is False


def test_an_unrecognised_error_is_not_retried_blindly():
    assert is_transient(LLMError("something entirely unexpected")) is False


# --- what to tell the user -------------------------------------------

def test_quota_message_names_the_daily_cap_and_the_escape_hatch():
    msg = friendly_message(QUOTA_429)
    assert "quota" in msg.lower()
    assert "GEMINI_MODEL" in msg, "must offer the switch-model workaround"


def test_retired_model_message_points_at_the_config():
    assert "GEMINI_MODEL" in friendly_message(RETIRED_404)


def test_missing_key_message_names_the_variable():
    assert "GEMINI_API_KEY" in friendly_message(NO_KEY)


def test_overload_message_says_retries_were_already_tried():
    msg = friendly_message(OVERLOADED_503)
    assert "overloaded" in msg.lower()
    assert "retries" in msg.lower(), (
        "the app retries first, so the message must not imply the user should "
        "simply hit the button again with no context"
    )


def test_an_unknown_error_is_passed_through_verbatim():
    assert friendly_message(LLMError("weird backend explosion")) == \
        "weird backend explosion"


# --- the app wires it up ---------------------------------------------

APP = (__import__("pathlib").Path("app.py")).read_text(encoding="utf-8")


def test_app_retries_transient_failures_only():
    assert "def answer_with_retry" in APP
    assert "is_transient(exc)" in APP
    assert "ans = answer_with_retry(" in APP, "the ask path must use the retry"


def test_app_reports_the_friendly_message_on_final_failure():
    assert "st.error(friendly_message(e))" in APP
