import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-v3.1:671b-cloud"

def _build_mom_prompt(meeting_notes: str) -> str:
    """
    Prompt is strict, structured, and anti-hallucination.
    """
    return f"""
You are an expert corporate meeting secretary.

Using ONLY the meeting notes below, generate professional Minutes of Meeting (MoM).
Do NOT assume or invent any information.

STRICT OUTPUT REQUIREMENTS:
- Output MUST be valid JSON
- No markdown
- No explanations
- No extra text

If any field is missing, write "Not specified".

JSON STRUCTURE:
{{
  "meeting_topic": "",
  "meeting_summary": "",
  "agenda_items": [
    {{
      "topic": "",
      "discussion": "",
      "decision": ""
    }}
  ],
  "decisions": [],
  "action_items": [
    {{
      "task": "",
      "owner": "",
      "deadline": ""
    }}
  ],
  "open_points": []
}}

MEETING NOTES:
\"\"\"
{meeting_notes}
\"\"\"
"""

def generate_minutes_of_meeting(meeting_notes: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "prompt": _build_mom_prompt(meeting_notes),
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()

    raw_output = response.json().get("response", "").strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        raise ValueError("AI returned invalid JSON")
