"""Shared helpers for Task 1: prompt formats, JSON extraction and scoring.

Task: turn a plain-English calendar request into a fixed JSON object with the keys
    title, date, time, duration_minutes, location, attendees
"""
import json
import re

SCHEMA_KEYS = ["title", "date", "time", "duration_minutes", "location", "attendees"]

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


# --------------------------------------------------------------------------- prompts
def short_prompt(text: str) -> str:
    """Prompt the model is fine-tuned on (no instructions, the model must learn the format)."""
    return f"Request: {text}\nJSON:"


INSTRUCT_HEADER = (
    "Convert the calendar request into JSON with exactly these keys: "
    "title (string), date (YYYY-MM-DD), time (24-hour HH:MM), "
    "duration_minutes (integer; use 60 if not stated), "
    "location (string or null), attendees (list of first names; [] if none).\n"
    "Output only the JSON object.\n\n"
    "Example:\n"
    "Request: Set up a yoga class on 2026-02-03 at 6pm for 45 minutes at Studio 2 with Anita.\n"
    'JSON: {"title": "yoga class", "date": "2026-02-03", "time": "18:00", '
    '"duration_minutes": 45, "location": "Studio 2", "attendees": ["Anita"]}\n\n'
)


def instruct_prompt(text: str) -> str:
    """Prompt used for the *base* model: schema description + one worked example."""
    return INSTRUCT_HEADER + f"Request: {text}\nJSON:"


PROMPT_STYLES = {"short": short_prompt, "instruct": instruct_prompt}


def completion_text(target: dict) -> str:
    """The exact string the model must produce after the prompt."""
    return " " + json.dumps(target, ensure_ascii=False)


# --------------------------------------------------------------------------- parsing
def extract_json(text: str):
    """Return the first balanced {...} object in `text` parsed as a dict, else None."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break  # try the next '{'
        start = text.find("{", start + 1)
    return None


# --------------------------------------------------------------------------- scoring
def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def schema_ok(obj) -> bool:
    """Right keys AND right types/formats (does not check that values are correct)."""
    if not isinstance(obj, dict) or set(obj.keys()) != set(SCHEMA_KEYS):
        return False
    return (
        isinstance(obj["title"], str)
        and isinstance(obj["date"], str) and bool(DATE_RE.match(obj["date"]))
        and isinstance(obj["time"], str) and bool(TIME_RE.match(obj["time"]))
        and _is_int(obj["duration_minutes"])
        and (obj["location"] is None or isinstance(obj["location"], str))
        and isinstance(obj["attendees"], list)
        and all(isinstance(a, str) for a in obj["attendees"])
    )


def score_one(pred, gold: dict) -> dict:
    """Score one prediction (dict or None) against gold."""
    out = {
        "valid_json": pred is not None,
        "schema_ok": schema_ok(pred),
        "exact_match": False,
        "fields": {k: False for k in SCHEMA_KEYS},
    }
    if pred is not None:
        for k in SCHEMA_KEYS:
            out["fields"][k] = k in pred and pred[k] == gold[k]
        out["exact_match"] = out["schema_ok"] and all(out["fields"].values())
    return out


def aggregate(scores: list) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}
    agg = {
        "n": n,
        "valid_json": sum(s["valid_json"] for s in scores) / n,
        "schema_ok": sum(s["schema_ok"] for s in scores) / n,
        "exact_match": sum(s["exact_match"] for s in scores) / n,
    }
    for k in SCHEMA_KEYS:
        agg[f"field_{k}"] = sum(s["fields"][k] for s in scores) / n
    return agg


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
