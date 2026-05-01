import os
import json

# ── paths ─────────────────────────────────────────────────────────────────────
PRE_MEETING_JSON    = "app/Working_transcript/pre_meeting_notes.json"
DIARIZED_TXT        = "app/Working_transcript/diarized_output_api.txt"
# COMPILED_OUTPUT     = "app/Working_transcript/compiled_meeting.txt"
COMPILED_OUTPUT     = "app/Working_transcript/meeting_notes_dialouge.txt"


def _load_pre_meeting() -> dict:
    """Load pre-meeting notes from JSON."""
    if not os.path.exists(PRE_MEETING_JSON):
        raise FileNotFoundError(
            f"pre_meeting_notes.json not found at {PRE_MEETING_JSON}. "
            "Run /extract-pre-meeting first."
        )
    with open(PRE_MEETING_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_diarized_transcript() -> str:
    """Load diarized transcript text."""
    if not os.path.exists(DIARIZED_TXT):
        raise FileNotFoundError(
            f"diarized_output_api.txt not found at {DIARIZED_TXT}. "
            "Run /diarize-api first."
        )
    with open(DIARIZED_TXT, "r", encoding="utf-8") as f:
        return f.read().strip()


def _format_people(people: list) -> str:
    """
    Format people list into:
        David (Sales, TaskMaster), Sarah (Office Manager, ABC Company)
    Handles missing role or company gracefully.
    """
    parts = []
    for p in people:
        name    = p.get("name", "Unknown")
        role    = p.get("role")
        company = p.get("company")

        if role and company:
            parts.append(f"{name} ({role}, {company})")
        elif role:
            parts.append(f"{name} ({role})")
        elif company:
            parts.append(f"{name} ({company})")
        else:
            parts.append(name)

    return ",\n ".join(parts)


def _build_compiled_text(pre: dict, dialogue: str) -> str:
    """Assemble the final compiled meeting text."""

    meeting      = pre.get("meeting", "N/A")
    date         = pre.get("date", "N/A")
    meeting_type = pre.get("meeting_type", "N/A")
    people       = pre.get("people", [])

    people_str   = _format_people(people) if people else "N/A"

    return (
        f"Meeting: {meeting}\n"
        f"Date: {date}\n"
        f"Meeting Type: {meeting_type}\n"
        f"People:\n {people_str}\n\n"
        f"Meeting Dialogues:\n{dialogue}"
    )


def compile_meeting_file() -> dict:
    """
    Reads pre_meeting_notes.json and diarized_output_api.txt,
    merges them into a single meeting_notes_dialouge.txt file.

    Returns:
        {
            "status":  "success" | "failure",
            "message": "...",
            "output_path": "..." | None
        }
    """
    # 1. Load both sources
    pre_meeting = _load_pre_meeting()
    dialogue    = _load_diarized_transcript()

    # 2. Build compiled text
    compiled = _build_compiled_text(pre_meeting, dialogue)

    # 3. Save
    os.makedirs(os.path.dirname(COMPILED_OUTPUT), exist_ok=True)
    with open(COMPILED_OUTPUT, "w", encoding="utf-8") as f:
        f.write(compiled)

    return {
        "status":      "success",
        "message":     f"Meeting file compiled and saved to {COMPILED_OUTPUT}",
        "output_path": COMPILED_OUTPUT,
    }