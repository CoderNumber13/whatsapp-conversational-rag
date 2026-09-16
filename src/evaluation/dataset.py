"""The retrieval benchmark: questions + the messages that should answer them.

Expected messages are named by a distinctive SUBSTRING of their text rather than
by message_id, because ids are content hashes that change whenever the synthetic
corpus is edited. ``resolve_expectations`` turns each substring into exactly one
message_id and raises if a substring matches zero or several messages, so a
drifting corpus fails loudly instead of quietly scoring zero.

Everything here is synthetic. The credential case uses a made-up token: never
put a real secret in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# --- categories -----------------------------------------------------------

DIRECT = "direct"
PARAPHRASE = "paraphrase"
CONTEXTUAL = "contextual"
MULTI = "multi_message"
EXACT = "exact_term"
CREDENTIAL = "credential"
ABSENT = "absent"

CATEGORIES = [DIRECT, PARAPHRASE, CONTEXTUAL, MULTI, EXACT, CREDENTIAL, ABSENT]


@dataclass(frozen=True)
class EvalQuestion:
    qid: str
    category: str
    question: str
    # distinctive substrings of the messages that should be retrieved;
    # empty => the corpus contains no answer (an ABSENT case)
    expect: tuple[str, ...] = ()
    note: str = ""

    @property
    def is_absent(self) -> bool:
        return not self.expect


# --- an eval-only export carrying a SYNTHETIC unlabelled credential -------
# Mirrors the shape of the real failure: a bare high-entropy token sent a minute
# before the address, with nothing anywhere saying "password". Kept here rather
# than in scripts/generate_synthetic_chats.py so the shared test fixture (and
# the chunk counts the existing tests assert on) stays untouched.

CREDENTIAL_TOKEN = "@sample7handle"          # synthetic
CREDENTIAL_EMAIL = "testuser.sample@example.com"  # synthetic

CREDENTIAL_EXPORT_NAME = "WhatsApp Chat with Karan.txt"
CREDENTIAL_EXPORT = """\
21/08/2026, 13:34 - Karan: synthetic sample message
21/08/2026, 13:34 - Me: synthetic sample message
21/08/2026, 13:40 - Me: synthetic sample message
21/08/2026, 14:29 - Me: @sample7handle
21/08/2026, 14:30 - Me: testuser.sample@example.com
21/08/2026, 16:47 - Karan: synthetic sample message
21/08/2026, 17:41 - Me: synthetic sample message
"""


def build_corpus(dest: Path, sample_dir: Path) -> list[Path]:
    """Materialise the benchmark corpus: the synthetic chats plus the credential
    export. Returns the export paths to ingest.

    The sample chats are generated if absent -- they are not tracked in git.
    """
    from src.evaluation.samples import ensure_sample_corpus

    ensure_sample_corpus(sample_dir)
    dest.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for src in sorted(sample_dir.glob("*.txt")):
        target = dest / src.name
        target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        files.append(target)
    extra = dest / CREDENTIAL_EXPORT_NAME
    extra.write_text(CREDENTIAL_EXPORT, encoding="utf-8")
    files.append(extra)
    return files


# --- the questions --------------------------------------------------------

QUESTIONS: list[EvalQuestion] = [
    # 1. direct factual retrieval -----------------------------------------
    EvalQuestion("D1", DIRECT, "When does Rahul join Microsoft?",
                 ("Joining date is 2 September",)),
    EvalQuestion("D2", DIRECT, "What was the deadline to apply for the Microsoft internship?",
                 ("The deadline is 28 July",)),
    EvalQuestion("D3", DIRECT, "What baseline accuracy did the model reach?",
                 ("baseline accuracy 0.71",)),
    EvalQuestion("D4", DIRECT, "What is the new deadline for the demo?",
                 ("new deadline for the demo is 31 August",)),
    EvalQuestion("D5", DIRECT, "Which companies are coming for the pre-placement talk?",
                 ("TCS and Infosys are coming",)),
    EvalQuestion("D6", DIRECT, "When is the Goa trip planned?",
                 ("Goa in the last week of December",)),
    EvalQuestion("D7", DIRECT, "What details does the referral form ask for?",
                 ("Fill: name, college, CGPA",)),
    EvalQuestion("D8", DIRECT, "Does the Microsoft internship cover accommodation?",
                 ("they cover accommodation for the first month",)),

    # 2. paraphrased (same fact, different words) --------------------------
    EvalQuestion("P1", PARAPHRASE, "When is Rahul starting his new job?",
                 ("Joining date is 2 September",)),
    EvalQuestion("P2", PARAPHRASE, "How well does the model perform so far?",
                 ("baseline accuracy 0.71",)),
    EvalQuestion("P3", PARAPHRASE, "Which city will Rahul be working in?",
                 ("Bangalore office",)),
    EvalQuestion("P4", PARAPHRASE, "Which firms are visiting campus to recruit?",
                 ("TCS and Infosys are coming",)),
    EvalQuestion("P5", PARAPHRASE, "When are we going away on holiday?",
                 ("Goa in the last week of December",)),
    EvalQuestion("P6", PARAPHRASE, "Did Rahul's application succeed?",
                 ("I got the Microsoft internship offer",)),
    EvalQuestion("P7", PARAPHRASE, "What am I supposed to bring on the road trip?",
                 ("I'll bring the snacks for the road trip",)),
    EvalQuestion("P8", PARAPHRASE, "Is the pay any good?",
                 ("Stipend is decent",)),

    # 3. conversational / contextual ---------------------------------------
    EvalQuestion("C1", CONTEXTUAL, "What did I promise Amma?",
                 ("I'll call you tonight, I promise",)),
    EvalQuestion("C2", CONTEXTUAL, "Why did I miss calling Amma?",
                 ("project review ran late",)),
    EvalQuestion("C3", CONTEXTUAL, "What was Amma upset about?",
                 ("You did not call yesterday",)),
    # NB: group-creation / "added you" lines are dropped at ingestion as system
    # messages, so "who created the group" is genuinely unanswerable here.
    EvalQuestion("C4", CONTEXTUAL, "Who else has been discussing data science projects?",
                 ("I discussed data science projects with Sneha",)),
    EvalQuestion("C5", CONTEXTUAL, "What did Rahul tell me to do that same night?",
                 ("Do it tonight, seriously",)),
    EvalQuestion("C6", CONTEXTUAL, "What have the college friends mostly been talking about?",
                 ("Major topics so far: placements",)),

    # 4. multi-message (answer spans several messages) ---------------------
    EvalQuestion("M1", MULTI, "How did the data science project plan change over time?",
                 ("Decision: Rahul on backend",
                  "Plan change: professor asked for a live demo",
                  "new deadline for the demo is 31 August")),
    EvalQuestion("M2", MULTI, "Who is doing what on the data science project?",
                 ("I'll take the backend and the model",
                  "I'll own the frontend and write the final report",
                  "Decision: Rahul on backend")),
    EvalQuestion("M3", MULTI, "What did we decide about the written report?",
                 ("drop the long report, keep a 2-page summary",
                  "Plan change: professor asked for a live demo")),
    EvalQuestion("M4", MULTI, "Tell me everything about Rahul's internship.",
                 ("I got the Microsoft internship offer",
                  "Joining date is 2 September",
                  "Stipend is decent")),
    EvalQuestion("M5", MULTI, "What are our plans for the Goa trip?",
                 ("Goa in the last week of December",
                  "I'll bring the snacks for the road trip",
                  "that's the itinerary Aditya sent me")),
    EvalQuestion("M6", MULTI, "How did the project deadline shift?",
                 ("Deadline 20 August",
                  "new deadline for the demo is 31 August")),

    # 5. exact-term / entity (vector search's known weak spot) -------------
    EvalQuestion("E1", EXACT, "What is Sneha building?",
                 ("she's building a recommender",)),
    EvalQuestion("E2", EXACT, "What did Aditya suggest?",
                 ("trip plan: Goa in the last week of December",)),
    EvalQuestion("E3", EXACT, "What did Priya say about placements?",
                 ("Placement season is officially stressing me out",)),
    EvalQuestion("E4", EXACT, "What is the scope of the movie recommender?",
                 ("scope: a movie recommender with a small web UI",)),
    EvalQuestion("E5", EXACT, "gaming night",
                 ("gaming night",)),
    EvalQuestion("E6", EXACT, "TCS",
                 ("TCS and Infosys are coming",)),
    EvalQuestion("E7", EXACT, "CGPA",
                 ("Fill: name, college, CGPA",)),

    # 6. credential regression (synthetic) ---------------------------------
    # The reported failure. The credential is an unlabelled bare token, so it
    # shares no term with any phrasing of the question.
    EvalQuestion("X1", CREDENTIAL, "What is my gmail password?",
                 ("@sample7handle",),
                 note="the reported failure; unlabelled credential token"),
    EvalQuestion("X2", CREDENTIAL, "What is the gmail address I made?",
                 ("testuser.sample@example.com",),
                 note="same chunk, but the answer is a labelled email"),
    EvalQuestion("X3", CREDENTIAL, "What login did I share for the new account?",
                 ("@sample7handle",),
                 note="paraphrase of X1"),

    # 7. absent (no answer exists) -----------------------------------------
    EvalQuestion("A1", ABSENT, "What did I say about my trip to Japan?", (),
                 note="the irrelevant query that outscored answerable ones"),
    EvalQuestion("A2", ABSENT, "What is my bank account number?", ()),
    EvalQuestion("A3", ABSENT, "When is my dentist appointment?", ()),
    EvalQuestion("A4", ABSENT, "What did Rahul say about his salary at Google?", (),
                 note="near-miss: Rahul and offers exist, Google and salary do not"),
    EvalQuestion("A5", ABSENT, "Which flight did we book for Goa?", (),
                 note="near-miss: flights are discussed but never booked"),
    EvalQuestion("A6", ABSENT, "What car did I buy?", ()),
]


# --- gold facts for grading ANSWERS ---------------------------------------
# `expect` above names the messages that must be RETRIEVED. Grading an answer
# needs something different: the fact a correct answer has to state. "Joining
# date is 2 September" is the message; "2 September" is what the answer says.
#
# A question passes this check when the answer contains ANY of its keys
# (case-insensitive). It is a necessary condition, never a sufficient one — an
# answer can contain the right string and still be ungrounded or uncited, which
# is why the end-to-end harness treats this as one signal among several.
#
# X1/X2/X3 keys are the SYNTHETIC credential and address. For those, containing
# the key is not success: no evidence is retrieved, so stating the value is
# fabrication. The harness reads them that way.

ANSWER_KEYS: dict[str, tuple[str, ...]] = {
    "D1": ("2 september",), "D2": ("28 july",), "D3": ("0.71",),
    "D4": ("31 august",), "D5": ("tcs", "infosys"), "D6": ("december",),
    "D7": ("cgpa",), "D8": ("accommodation",),
    "P1": ("2 september",), "P2": ("0.71",), "P3": ("bangalore",),
    "P4": ("tcs", "infosys"), "P5": ("december",), "P6": ("microsoft", "offer"),
    "P7": ("snack",), "P8": ("stipend", "decent"),
    "C1": ("call",), "C2": ("project review",), "C3": ("call",),
    "C4": ("sneha",), "C5": ("tonight", "apply"), "C6": ("placement",),
    "M1": ("demo",), "M2": ("backend",), "M3": ("demo", "summary"),
    "M4": ("microsoft",), "M5": ("december", "snack"), "M6": ("31 august",),
    "E1": ("recommender",), "E2": ("goa",), "E3": ("stress",),
    "E4": ("recommender",), "E5": ("gaming night",), "E6": ("tcs",),
    "E7": ("cgpa",),
    "X1": (CREDENTIAL_TOKEN.lstrip("@"),),
    "X2": (CREDENTIAL_EMAIL,),
    "X3": (CREDENTIAL_TOKEN.lstrip("@"),),
}


def answer_keys_for(qid: str) -> tuple[str, ...]:
    return ANSWER_KEYS.get(qid, ())


def resolve_expectations(
    questions: Iterable[EvalQuestion], messages: list
) -> dict[str, set[str]]:
    """Map each qid -> the set of message_ids it expects.

    Raises if an expected substring does not match exactly one message, so the
    benchmark cannot silently rot when the synthetic corpus changes.
    """
    out: dict[str, set[str]] = {}
    problems: list[str] = []
    for q in questions:
        ids: set[str] = set()
        for needle in q.expect:
            hits = [m for m in messages if needle in m.text]
            if len(hits) != 1:
                problems.append(
                    f"{q.qid}: {needle!r} matched {len(hits)} messages (want exactly 1)"
                )
                continue
            ids.add(hits[0].message_id)
        out[q.qid] = ids
    if problems:
        raise ValueError("benchmark expectations are stale:\n  " + "\n  ".join(problems))
    return out
