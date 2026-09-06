"""Generate synthetic WhatsApp exports for tests, demos and the public repo.

Everything here is invented. No real person or conversation is represented.
The content is deterministic so the evaluation set can rely on it, and it
deliberately exercises format variety:

  * Android ` - ` headers and iOS `[...]` headers
  * 12h clocks with the U+202F narrow no-break space before AM/PM, and 24h
  * 2- and 4-digit years, day-first and month-first date orders
  * multiline messages, emoji, media placeholders (both platforms)
  * system/notification lines, a deleted message, a missed call
  * message bodies containing commas and colons, senders with spaces

Run:  python scripts/generate_synthetic_chats.py
"""

from __future__ import annotations

from pathlib import Path

NBSP_NARROW = " "  # iOS puts this before AM/PM
LTR = "\u200E"  # iOS prefixes many lines with this

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample" / "synthetic_chats"


# --- 1) 1:1, Android, day-first, 24h -------------------------------------
RAHUL = f"""\
25/07/2026, 20:02 - Messages and calls are end-to-end encrypted. Tap to learn more.
25/07/2026, 20:02 - Rahul: Bro did you apply for the Microsoft internship?
25/07/2026, 20:03 - Me: Not yet. The deadline is 28 July, right?
25/07/2026, 20:03 - Rahul: Yeah, 28th. Do it tonight, seriously.
25/07/2026, 20:05 - Me: Ok ok. I'll do it tonight, I promise.
25/07/2026, 20:06 - Rahul: <Media omitted>
25/07/2026, 20:07 - Rahul: That's the referral form. Fill: name, college, CGPA.
17/08/2026, 09:14 - Rahul: Update: I got the Microsoft internship offer! 🎉
17/08/2026, 09:15 - Me: No way, congrats!! When do you join?
17/08/2026, 09:16 - Rahul: Joining date is 2 September. Bangalore office.
17/08/2026, 09:16 - Rahul: Stipend is decent, and they cover accommodation for the first month.
02/08/2026, 19:40 - Me: About our data science project - I think you should take the backend.
02/08/2026, 19:41 - Rahul: Fine by me. I'll handle the backend and the API.
02/08/2026, 19:41 - Me: Cool, I'll do the frontend and the report then.
02/08/2026, 19:42 - Rahul: Deal. Let's aim to finish the first version by 20 August.
28/08/2026, 12:00 - Rahul: Change of plan: professor wants a live demo, not a report.
28/08/2026, 12:01 - Me: Ok so new plan:
1. finish backend by 30 Aug
2. build the demo UI
3. rehearse once before the review
28/08/2026, 12:02 - Rahul: 👍
"""


# --- 2) group, iOS, month-first, 12h with U+202F -----------------------
CF = f"""\
[8/1/26, 9:00:00{NBSP_NARROW}PM] {LTR}Messages and calls are end-to-end encrypted. Tap to learn more.
[8/1/26, 9:00:05{NBSP_NARROW}PM] {LTR}Priya created group "College Friends"
[8/1/26, 9:00:07{NBSP_NARROW}PM] {LTR}Priya added you
[8/1/26, 9:01:10{NBSP_NARROW}PM] Priya: Placement season is officially stressing me out 😭
[8/1/26, 9:02:00{NBSP_NARROW}PM] Rahul Kumar: same. TCS and Infosys are coming next week for the pre-placement talk
[8/1/26, 9:02:30{NBSP_NARROW}PM] Aditya: relax guys, we still have the whole semester
[8/1/26, 9:03:00{NBSP_NARROW}PM] Me: Who have we talked to about data science roles? I only know Rahul is doing DS
[8/1/26, 9:03:40{NBSP_NARROW}PM] Rahul Kumar: I discussed data science projects with Sneha too, she's building a recommender
[8/1/26, 9:05:00{NBSP_NARROW}PM] Aditya: trip plan: Goa in the last week of December, flights are cheap now
[8/1/26, 9:05:30{NBSP_NARROW}PM] Priya: I'm in
[8/1/26, 9:05:45{NBSP_NARROW}PM] Me: I'm in too. I'll bring the snacks for the road trip, promise.
[8/1/26, 9:06:10{NBSP_NARROW}PM] Rahul Kumar: {LTR}image omitted
[8/1/26, 9:06:20{NBSP_NARROW}PM] Rahul Kumar: that's the itinerary Aditya sent me
[8/12/26, 7:30:00{NBSP_NARROW}AM] Aditya: This message was deleted
[8/12/26, 7:31:00{NBSP_NARROW}AM] Priya: lol what did you delete
[8/12/26, 7:35:00{NBSP_NARROW}AM] {LTR}Missed voice call
[8/20/26, 6:00:00{NBSP_NARROW}PM] Me: Major topics so far: placements, the Goa trip, and data science projects. Did I miss anything?
[8/20/26, 6:01:00{NBSP_NARROW}PM] Priya: gaming night. we keep saying we'll do it and never do
"""


# --- 3) 1:1, Android, month-first, 12h -------------------------------
AMMA = """\
8/15/26, 8:05 AM - Messages and calls are end-to-end encrypted. Tap to learn more.
8/15/26, 8:05 AM - Amma: Did you eat?
8/15/26, 8:06 AM - Me: yes amma
8/15/26, 8:06 AM - Amma: call me when you are free
8/15/26, 8:07 AM - Me: I'll call you tonight, I promise
8/18/26, 9:30 PM - Amma: You did not call yesterday
8/18/26, 9:31 PM - Me: sorry, project review ran late. calling now
"""


# --- 4) group, iOS, day-first, 24h (no AM/PM) --------------------------
DS = f"""\
[04/08/2026, 10:00:00] {LTR}Messages and calls are end-to-end encrypted. Tap to learn more.
[04/08/2026, 10:00:03] {LTR}Sneha created group "DS Project Team"
[04/08/2026, 10:00:05] {LTR}Sneha added Rahul Kumar
[04/08/2026, 10:01:00] Sneha: Ok team, scope: a movie recommender with a small web UI
[04/08/2026, 10:02:00] Rahul Kumar: I'll take the backend and the model. Me and Sneha can pair on data cleaning.
[04/08/2026, 10:02:40] Me: I'll own the frontend and write the final report
[04/08/2026, 10:03:00] Sneha: Decision: Rahul on backend, Me on frontend, Sneha on data + eval. Deadline 20 August.
[12/08/2026, 18:00:00] Rahul Kumar: model is training. baseline accuracy 0.71
[27/08/2026, 09:00:00] Sneha: Plan change: professor asked for a live demo instead of the report.
[27/08/2026, 09:01:00] Me: Updated plan: drop the long report, keep a 2-page summary, focus on the demo
[27/08/2026, 09:02:00] Rahul Kumar: agreed. new deadline for the demo is 31 August
"""


SYNTHETIC = {
    "WhatsApp Chat with Rahul.txt": RAHUL,
    "WhatsApp Chat with College Friends.txt": CF,
    "WhatsApp Chat with Amma.txt": AMMA,
    "WhatsApp Chat with DS Project Team.txt": DS,
}


def write_all(out_dir: Path = OUT_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in SYNTHETIC.items():
        path = out_dir / name
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for p in write_all():
        print(f"wrote {p}")
