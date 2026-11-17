"""Generate the synthetic dataset for Task 1 (plain-English calendar request -> JSON).

    python make_dataset.py                # writes data/train.jsonl and data/test.jsonl

The test set has two parts so the evaluation says something about generalisation:
  * "in_dist" : same phrasings / vocabulary as training, but new random combinations
  * "shifted" : phrasings, date formats, event types, names and locations the model
                never saw in training (this is where failures are expected)
Train/test never share an identical request text.
"""
import argparse
import calendar
import os
import random

from common import write_jsonl

# ----------------------------------------------------------------------------- vocab
TITLES = ["team standup", "dentist appointment", "project review", "lunch", "yoga class",
          "budget meeting", "client demo", "interview", "doctor visit", "design workshop",
          "sprint planning", "coffee chat", "piano lesson", "parent teacher meeting",
          "code review", "book club", "hackathon kickoff", "one on one", "training session",
          "farewell party"]
TITLES_SHIFTED = ["board meeting", "physiotherapy session", "pottery class", "investor call",
                  "retrospective", "dinner", "town hall", "driving test"]

NAMES = ["Priya", "Rahul", "Amit", "Neha", "Karan", "Sneha", "Vikram", "Anjali", "Rohan",
         "Pooja", "Arjun", "Divya", "Manish", "Kavya", "Sanjay", "Ritu", "Nikhil", "Meera"]
NAMES_SHIFTED = ["Tanvi", "Harsh", "Ishaan", "Lakshmi", "Zoya", "Dev"]

LOCS = ["Room 204", "Conference Room B", "Cafe Coffee Day", "Starbucks", "City Library",
        "Studio 2", "Main Office", "Bandra Gym", "Hall 3", "Lab 5"]
ONLINE_LOCS = ["Zoom", "Google Meet"]
LOCS_SHIFTED = ["Rooftop Cafe", "Block C Seminar Hall", "Annex Building"]
ONLINE_SHIFTED = ["Teams"]

MONTHS = list(calendar.month_name)[1:]
MON_ABBR = list(calendar.month_abbr)[1:]


# ----------------------------------------------------------------------------- helpers
def article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


def ordinal(n):
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def pick_date(rng):
    year = rng.choice([2026, 2027])
    month = rng.randint(1, 12)
    day = rng.randint(1, calendar.monthrange(year, month)[1])
    return year, month, day


def fmt_date(rng, y, m, d, shifted):
    iso = f"{y}-{m:02d}-{d:02d}"
    train_opts = [
        f"{MONTHS[m-1]} {ordinal(d)}, {y}",
        f"{d} {MONTHS[m-1]} {y}",
        iso,
        f"{MON_ABBR[m-1]} {d}, {y}",
    ]
    shifted_opts = [
        f"the {ordinal(d)} of {MONTHS[m-1]} {y}",
        f"{d}-{MON_ABBR[m-1]}-{y}",
    ]
    return rng.choice(shifted_opts if shifted and rng.random() < 0.7 else train_opts), iso


SPOKEN_HOURS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def daypart(h24):
    if h24 < 12:
        return "in the morning"
    if h24 < 17:
        return "in the afternoon"
    return "in the evening"


def fmt_time(rng, shifted):
    """Return (text, 'HH:MM'). Spoken forms are rare in train, common in the shifted test."""
    r = rng.random()
    spoken_p = 0.6 if shifted else 0.15
    if r < spoken_p:
        kind = rng.choice(["half_past", "quarter_past", "quarter_to"])
        h24 = rng.randint(7, 20)
        h12 = h24 % 12 or 12
        if kind == "half_past":
            return f"half past {SPOKEN_HOURS[h12]} {daypart(h24)}", f"{h24:02d}:30"
        if kind == "quarter_past":
            return f"quarter past {SPOKEN_HOURS[h12]} {daypart(h24)}", f"{h24:02d}:15"
        h_to = h24 + 1                      # "quarter to X" means (X-1):45
        h12_to = h_to % 12 or 12
        return f"quarter to {SPOKEN_HOURS[h12_to]} {daypart(h24)}", f"{h24:02d}:45"
    r2 = rng.random()
    if r2 < 0.10:
        return "noon", "12:00"
    if r2 < 0.20:
        h24 = rng.choice([8, 9, 10, 17, 18, 19])
        h12 = h24 % 12 or 12
        return f"{h12} o'clock {daypart(h24)}", f"{h24:02d}:00"
    h24 = rng.randint(7, 20)
    minute = rng.choice([0, 0, 0, 15, 30, 45, 5, 20, 40])
    hhmm = f"{h24:02d}:{minute:02d}"
    if rng.random() < 0.55:                  # 12-hour clock with am/pm
        h12 = h24 % 12 or 12
        ap = "am" if h24 < 12 else "pm"
        if rng.random() < 0.3:
            ap = ap.upper()
        sep = rng.choice(["", " "])
        body = f"{h12}" if minute == 0 else f"{h12}:{minute:02d}"
        return f"{body}{sep}{ap}", hhmm
    return (f"{hhmm} hrs" if rng.random() < 0.15 else hhmm), hhmm  # 24-hour clock


DURATION_TRAIN = {
    15: ["for 15 minutes"],
    30: ["for 30 minutes", "for half an hour", "for 30 min"],
    45: ["for 45 minutes", "for 45 min"],
    60: ["for an hour", "for 1 hour", "for 60 minutes"],
    90: ["for 90 minutes", "for an hour and a half", "for 1.5 hours"],
    120: ["for 2 hours", "for two hours"],
    180: ["for 3 hours"],
}
DURATION_SHIFTED = {
    75: ["for an hour and a quarter", "lasting 75 minutes"],
    40: ["lasting 40 minutes"],
    105: ["for 1 hour 45 minutes"],
    60: ["lasting an hour"],
    30: ["lasting half an hour"],
}


def pick_duration(rng, shifted):
    """Return (clause or '', minutes). No clause => default 60 minutes."""
    if rng.random() < 0.2:
        return "", 60
    table = DURATION_SHIFTED if shifted and rng.random() < 0.7 else DURATION_TRAIN
    mins = rng.choice(list(table))
    return rng.choice(table[mins]), mins


def pick_attendees(rng, shifted):
    pool = NAMES + (NAMES_SHIFTED if shifted else [])
    n = rng.choice([0, 0, 1, 1, 2, 3])
    names = rng.sample(pool, n)
    if n == 0:
        return "", []
    if shifted:
        verb = rng.choice(["along with", "and invite", "with"])
    else:
        verb = "with"
    if n == 1:
        joined = names[0]
    elif n == 2:
        joined = f"{names[0]} and {names[1]}"
    else:
        joined = f"{', '.join(names[:-1])} and {names[-1]}"
    return f"{verb} {joined}", names


def pick_location(rng, shifted):
    if rng.random() < 0.3:
        return "", None
    online = ONLINE_LOCS + (ONLINE_SHIFTED if shifted else [])
    physical = LOCS + (LOCS_SHIFTED if shifted else [])
    if rng.random() < 0.3:
        loc = rng.choice(online)
        return (f"over {loc}" if shifted and rng.random() < 0.4 else f"on {loc}"), loc
    loc = rng.choice(physical)
    return f"{rng.choice(['at', 'in'])} {loc}", loc


PREFIX_TRAIN = ["Schedule", "Book", "Add", "Set up", "Create", "Put", "Plan"]
PREFIX_SHIFTED = ["Could you please pencil in", "I need", "Can you block time for",
                  "Please arrange", "Kindly note down"]
SUFFIX_SHIFTED = [" Thanks!", " It is important.", " Please confirm."]


def make_example(rng, shifted):
    titles = TITLES + (TITLES_SHIFTED if shifted else [])
    title = rng.choice(titles)
    y, m, d = pick_date(rng)
    date_txt, iso = fmt_date(rng, y, m, d, shifted)
    time_txt, hhmm = fmt_time(rng, shifted)
    dur_txt, mins = pick_duration(rng, shifted)
    loc_txt, loc = pick_location(rng, shifted)
    att_txt, names = pick_attendees(rng, shifted)

    when = (f"on {date_txt} at {time_txt}" if rng.random() < 0.5
            else f"at {time_txt} on {date_txt}")
    extras = [c for c in (dur_txt, loc_txt, att_txt) if c]
    rng.shuffle(extras)
    prefix = rng.choice(PREFIX_SHIFTED if shifted else PREFIX_TRAIN)
    if shifted:
        text = f"{prefix} {article(title)} {title} {when}" + (" " + " ".join(extras) if extras else "") + "."
        text += rng.choice(SUFFIX_SHIFTED) if rng.random() < 0.5 else ""
    else:
        text = f"{prefix} {article(title)} {title} {when}" + (" " + " ".join(extras) if extras else "") + "."
    target = {"title": title, "date": iso, "time": hhmm, "duration_minutes": mins,
              "location": loc, "attendees": names}
    return text, target


def build(n, seed, shifted, forbidden=frozenset()):
    rng = random.Random(seed)
    seen, rows = set(forbidden), []
    while len(rows) < n:
        text, target = make_example(rng, shifted)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"text": text, "target": target,
                     "split_type": "shifted" if shifted else "in_dist"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test-indist", type=int, default=70)
    ap.add_argument("--n-test-shifted", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = build(args.n_train, args.seed, shifted=False)
    train_texts = {r["text"] for r in train}
    test = (build(args.n_test_indist, args.seed + 1, shifted=False, forbidden=train_texts)
            + build(args.n_test_shifted, args.seed + 2, shifted=True, forbidden=train_texts))
    for i, r in enumerate(train):
        r["id"] = f"train-{i:04d}"
    for i, r in enumerate(test):
        r["id"] = f"test-{i:03d}"

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)
    print(f"train: {len(train)}  test: {len(test)} "
          f"({args.n_test_indist} in_dist + {args.n_test_shifted} shifted)  -> {args.out_dir}/")
    for r in train[:2] + test[-2:]:
        print("  ", r["text"], "->", r["target"])


if __name__ == "__main__":
    main()
