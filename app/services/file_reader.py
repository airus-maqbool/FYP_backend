# only loads meeting notes safely

import os

MEETING_NOTES_PATH = "app/Working_transcript/meeting_notes_dialouge.txt"

def load_meeting_notes() -> str:
    if not os.path.exists(MEETING_NOTES_PATH):
        raise FileNotFoundError("meeting_notes.txt not found")

    with open(MEETING_NOTES_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError("meeting_notes.txt is empty")

    return content
