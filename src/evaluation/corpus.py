"""Production-scale synthetic corpus for the retrieval benchmark.

The small `data/sample/synthetic_chats` set yields only 7 chunks, so k=10
returns the whole corpus and Recall@5/@10 are 100% by construction. It also
cannot reproduce the reported credential failure: there the credential sits in a
7-message conversation whose single chunk says "gmail" twice, so it is trivially
reachable.

This module generates a corpus at the scale of a real export while keeping every
existing benchmark answer intact, so both scales are directly comparable:

* the four tracked sample chats are copied in verbatim — all 44 questions still
  resolve against them;
* the credential conversation is expanded to production length, with the
  unlabelled token buried mid-stream rather than sitting in a 7-message chat;
* filler conversations supply the competition a real corpus provides;
* **password DISTRACTORS** are scattered through the filler — messages that
  discuss logins, passwords and accounts but contain no credential. These are
  what actually defeated retrieval in production: dense vectors rank a chunk
  *about* passwords above a chunk *containing* an unlabelled one.

Generation is seeded and therefore byte-for-byte reproducible. Nothing here is
real: every name, address and token is invented.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20260908

# --- the credential conversation, at production length --------------------
# Mirrors the real export's shape: Hinglish chatter about splitting a
# subscription, with a bare handle-shaped token dropped mid-conversation and
# nothing anywhere calling it a password.

CREDENTIAL_TOKEN = "@sample7handle"          # synthetic
CREDENTIAL_EMAIL = "testuser.sample@example.com"  # synthetic

_CRED_OPENING = [
    ("Karan", "Bhai claude pro"),
    ("Karan", "Lega mere sath ?"),
    ("Karan", "Sharing mai"),
    ("Me", "But ek time pe ek jann hi use kar sakta hai"),
    ("Karan", "Ha to theek haina"),
    ("Karan", "Mujhe to bas shuru kr 10 din ka kaam hai"),
    ("Karan", "Phir tu use krte rehna"),
    ("Karan", "Like mujhe 5 sept tk chahiye"),
    ("Karan", "August end maanle"),
    ("Me", "20$ right?"),
    ("Karan", "Ha"),
    ("Karan", "can you set up a new gmail for the shared plan"),
    ("Me", "sure, keeping the spend under ten dollars"),
    ("Karan", "Uspe lenge"),
    ("Me", "Us plan pe"),
    ("Karan", "Arey nhi bhai usme aisa nhi hota limit ka chakker"),
    ("Me", "Limit to hoga hi obviously"),
    ("Karan", "Token based nhi hai wo"),
    ("Me", "Then max usage limit kaise pata chalega"),
    ("Karan", "Ruk batata"),
    ("Karan", "Sun"),
    ("Karan", "Every 5 hour reset"),
    ("Karan", "5 times more token then free tier"),
    ("Karan", "Aur baki coding agent wagera"),
    ("Karan", "So aisa kuch nhi hai jo tu smjh rha ye alg pack hai"),
    ("Karan", "Mujhe ek project banana hai"),
    ("Karan", "Nhi hai bhai hourly basis pr hai"),
    ("Me", "let me check the options"),
]

# the two messages the benchmark expects — buried here, unlabelled
_CRED_PAYLOAD = [
    ("Me", CREDENTIAL_TOKEN),
    ("Me", CREDENTIAL_EMAIL),
]

_CRED_TAIL = [
    ("Karan", "Bhai"),
    ("Karan", "Purchase krlo"),
    ("Karan", "Paise bhej dunga mai sham se start Krna hai"),
    ("Me", "Please give me 10 mins"),
    ("Karan", "I give you 1 hr"),
    ("Me", "done, please check and let me know"),
    ("Karan", "Aja ab room pe"),
    ("Me", "Are you there in your room?"),
    ("Karan", "Yes"),
    ("Karan", "Kal subah lab hai kya"),
    ("Me", "Haan 9 baje"),
    ("Karan", "Chal so ja phir"),
    ("Karan", "Bhai ek aur cheez"),
    ("Karan", "Wo billing monthly hi rakhna"),
    ("Me", "Haan monthly hi liya hai"),
    ("Karan", "Theek hai"),
    ("Karan", "Kitna use hua ab tak"),
    ("Me", "Abhi to bas thoda hi"),
    ("Karan", "Achha"),
    ("Karan", "Mai raat ko baithunga"),
    ("Me", "Ok bata dena jab start kare"),
    ("Karan", "Haan"),
    ("Karan", "Kal se proper start"),
    ("Me", "Cool"),
]

# --- filler topics --------------------------------------------------------
# Ordinary chatter. Deliberately shares no distinctive phrase with any gold
# answer, so it competes for rank without ever becoming a false positive.

_FILLER: dict[str, list[str]] = {
    "hostel": [
        "mess ka khana aaj theek tha", "warden ne notice lagaya hai",
        "room clean karna hai kal", "laundry bhej diya",
        "bhai geyser kaam nhi kr rha", "water cooler fix ho gaya",
        "kal subah uthana mujhe", "so raha hu ab", "light chali gayi",
        "wifi slow hai aaj", "guard ne gate band kr diya",
        "mera charger kisne liya", "chai peene chal", "canteen band ho gayi",
        "roommate ghar gaya hai", "kal cleaning staff aayega",
        "AC service karwana hai", "cupboard ki chabi mil gayi",
    ],
    "gym": [
        "aaj leg day hai", "protein khatam ho gaya", "trainer ne form theek karwaya",
        "kal chest karenge", "bahut dard ho rha hai", "cardio skip kr diya",
        "weight badha hai thoda", "diet plan bana hu", "supplement order kiya",
        "gym band hai sunday ko", "warm up jaruri hai", "5 km run kiya aaj",
    ],
    "cricket": [
        "match dekha kal ka", "wo catch kaisa tha", "final kab hai",
        "ticket mil gaya kya", "hamari team jeet gayi", "toss haar gaye",
        "ground book kr diya", "bat le aana", "practice 6 baje",
        "rain se match ruk gaya", "score kya hua", "century maar di usne",
    ],
    "food": [
        "order kr diya", "delivery late hai", "paneer wala mangwa lo",
        "bill split kr lenge", "khana thanda aaya", "coupon laga diya",
        "kal bahar khate hai", "biryani ya pizza", "sweets le aana",
        "breakfast skip kiya", "coffee chahiye", "dinner ready hai",
    ],
    "study": [
        "notes bhej dena", "exam ka syllabus aa gaya", "assignment kal tak hai",
        "lab file complete kr li", "viva kaisa gaya", "unit 3 padhna hai",
        "reference book chahiye", "class cancel ho gayi", "attendance short hai",
        "practical file jama kr di", "quiz me kitne aaye", "revision start kr do",
    ],
    "travel": [
        "train book kr di", "platform number aa gaya", "bus late hai",
        "cab share kr lenge", "ghar kab ja rha hai", "ticket confirm ho gaya",
        "station pe milte hai", "bag pack kr liya", "highway pe jam hai",
        "flight time change ho gaya", "hotel book kr do", "return kab hai",
    ],
    "gaming": [
        "ek match aur", "lag aa rha hai", "squad bana lo",
        "new update aaya hai", "controller charge nhi hai", "rank push krna hai",
        "server down hai", "kal raat khelenge", "headset kharab ho gaya",
        "lobby me aaja", "map yaad nhi", "skin le liya maine",
    ],
    "work": [
        "standup 10 baje hai", "PR review kr dena", "build fail ho gaya",
        "deploy ho gaya prod pe", "ticket assign kr diya", "meeting reschedule hui",
        "logs check kr lo", "staging pe test kiya", "release kal hai",
        "bug reproduce nhi ho rha", "doc update kr diya", "sprint plan bhej diya",
    ],
}

# --- password DISTRACTORS -------------------------------------------------
# The crux. These talk *about* credentials without containing one. In the real
# corpus this is the material that outranked the actual credential chunk.

_PASSWORD_DISTRACTORS = [
    "bhai wifi ka password kya hai",
    "wifi password change ho gaya hai naya wala pooch lena",
    "college portal ka login kaam nhi kr rha",
    "mera portal password reset krna pdega",
    "library account ka password bhool gaya",
    "OTP nhi aaya abhi tak",
    "netflix ka account share kr rha hu",
    "us account me login nhi ho rha",
    "password reset link mail pe aaya hoga",
    "apne email pe check kr lo verification aaya hoga",
    "mujhe apna gmail id do assignment bhejna hai",
    "gmail pe bhej diya maine",
    "sign in nhi ho pa rha laptop se",
    "two factor on kr diya maine",
    "credentials bhej dena portal ke",
    "account ban gaya kya",
    "wo wala login id yaad hai tujhe",
    "mail id galat likh di maine",
]


def _fmt(dt: datetime, sender: str, text: str) -> str:
    return f"{dt:%d/%m/%Y, %H:%M} - {sender}: {text}"


def _write(path: Path, lines: list[str]) -> None:
    header = f"{lines[0].split(' - ')[0]} - Messages and calls are end-to-end encrypted. Tap to learn more."
    path.write_text("\n".join([header, *lines]) + "\n", encoding="utf-8")


def _conversation(
    rng: random.Random, sender: str, pool: list[str], start: datetime, n: int,
    *, distractor_rate: float = 0.0,
) -> list[str]:
    lines: list[str] = []
    dt = start
    for _ in range(n):
        dt += timedelta(minutes=rng.randint(1, 240))
        who = sender if rng.random() < 0.55 else "Me"
        if distractor_rate and rng.random() < distractor_rate:
            text = rng.choice(_PASSWORD_DISTRACTORS)
        else:
            text = rng.choice(pool)
        lines.append(_fmt(dt, who, text))
    return lines


def generate_large_corpus(dest: Path, sample_dir: Path) -> list[Path]:
    """Build the production-scale corpus. Deterministic for a fixed SEED."""
    from src.evaluation.samples import ensure_sample_corpus

    ensure_sample_corpus(sample_dir)
    rng = random.Random(SEED)
    dest.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []

    # 1. the tracked sample chats, verbatim — keeps every gold answer valid
    for src in sorted(sample_dir.glob("*.txt")):
        target = dest / src.name
        target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        files.append(target)

    # 2. the credential conversation at production length
    dt = datetime(2026, 8, 21, 12, 22)
    cred: list[str] = []
    for who, text in [*_CRED_OPENING, *_CRED_PAYLOAD, *_CRED_TAIL]:
        dt += timedelta(minutes=rng.randint(1, 40))
        cred.append(_fmt(dt, who, text))
    # pad it out so the credential is buried in a long history, as in the real
    # export, and so its chunk competes with the rest of the same conversation
    cred += _conversation(rng, "Karan", _FILLER["hostel"] + _FILLER["work"],
                          dt + timedelta(days=1), 40, distractor_rate=0.10)
    path = dest / "WhatsApp Chat with Karan.txt"
    _write(path, sorted(cred))
    files.append(path)

    # 3. filler conversations, some carrying password distractors
    people = [
        ("Nikhil", ["hostel", "food"], 130, 0.06),
        ("Ishaan", ["gym", "cricket"], 120, 0.02),
        ("Meera", ["study", "work"], 140, 0.08),
        ("Tanvi", ["travel", "food"], 110, 0.04),
        ("Rohan", ["gaming", "hostel"], 125, 0.05),
        ("Zoya", ["study", "travel"], 100, 0.07),
        ("Kabir", ["work", "gaming"], 115, 0.03),
        ("Ananya", ["food", "study"], 105, 0.06),
        ("Dev", ["cricket", "gym"], 95, 0.02),
        ("Farah", ["travel", "work"], 90, 0.05),
    ]
    start = datetime(2026, 5, 4, 9, 0)
    for i, (name, topics, n, rate) in enumerate(people):
        pool = [line for t in topics for line in _FILLER[t]]
        lines = _conversation(rng, name, pool, start + timedelta(days=i * 3), n,
                              distractor_rate=rate)
        path = dest / f"WhatsApp Chat with {name}.txt"
        _write(path, sorted(lines))
        files.append(path)

    return files
