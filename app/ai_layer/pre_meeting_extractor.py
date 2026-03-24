import os
import json
import requests
from datetime import date

# ── config ────────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "deepseek-v3.1:671b-cloud"
OUTPUT_PATH = "app/Working_transcript/pre_meeting_notes.json"

# ── today's date (injected into the prompt so LLM can use it) ─────────────────
TODAY = date.today().strftime("%d/%m/%Y")


# ── prompt ────────────────────────────────────────────────────────────────────
def _build_prompt(text: str) -> str:
    return f"""
You are an intelligent meeting notes parser. Today's date is {TODAY}.

Your job is to extract structured information from the meeting notes text provided below.

Extract the following fields:
1. "meeting"       — The meeting title or topic. Look for explicit mentions or infer from context.
2. "date"          — The meeting date. If explicitly mentioned, use it. If not mentioned, use today's date: {TODAY}. Format as DD/MM/YYYY.
3. "meeting_type"  — The type of meeting (e.g. Product Demo, Sales Call, Standup, Kickoff, Review, Interview, etc.).
                     If not explicitly mentioned, INFER it intelligently from the context and tone of the text.
4. "people"        — A list of all people mentioned. Each person should be an object with:
                       - "name": their name
                       - "role": their job title or role (if mentioned)
                       - "company": their company or team (if mentioned)
                     If role or company is not mentioned, set them to null.

Rules:
- Be intelligent. Infer missing fields where possible from context.
- For "date": ONLY use today's date ({TODAY}) if there is truly no date anywhere in the text.
- For "meeting_type": NEVER leave this null. Always infer it.
- For "people": extract EVERY person mentioned by name. Even if only a name is given with no role/company.
- If "meeting" topic cannot be inferred at all from the text, set it to null.
- If "people" list is completely empty (no names found anywhere), set it to null.

Respond ONLY with a valid JSON object. No explanation, no markdown, no code fences. Just the raw JSON.

Example output format:
{{
  "meeting": "Demo of TaskMaster Pro",
  "date": "08/02/2026",
  "meeting_type": "Product Demo",
  "people": [
    {{"name": "David", "role": "Sales", "company": "TaskMaster"}},
    {{"name": "Sarah", "role": "Office Manager", "company": "ABC Company"}}
  ]
}}

Now extract from this text:
\"\"\"
{text}
\"\"\"
""".strip()


def _call_ollama(prompt: str) -> str:
    """Call Ollama and return the full response string."""
    payload = {
        "model":  MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure Ollama is running on localhost:11434."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out. The model may be loading — try again.")
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {str(e)}")


def _parse_llm_response(raw: str) -> dict:
    """
    Safely parse the LLM JSON response.
    Handles cases where the model wraps output in markdown code fences.
    """
    # strip markdown fences if model ignores instructions
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM returned invalid JSON. Raw response was:\n{raw}\n\nParse error: {e}"
        )


def _validate_and_report(data: dict) -> tuple[bool, list[str]]:
    """
    Validate extracted fields. Returns (is_valid, list_of_missing_fields).

    Rules:
    - "meeting"      : required — cannot be null/empty
    - "date"         : always filled (LLM uses today as fallback), but check anyway
    - "meeting_type" : always filled (LLM infers), but check anyway
    - "people"       : required — must have at least one person
    """
    missing = []

    if not data.get("meeting"):
        missing.append(
            "  - meeting  : Could not determine the meeting topic or title from the text."
        )

    if not data.get("date"):
        missing.append(
            "  - date     : No date found and fallback failed."
        )

    if not data.get("meeting_type"):
        missing.append(
            "  - meeting_type : Could not determine or infer the type of meeting."
        )

    people = data.get("people")
    if not people or not isinstance(people, list) or len(people) == 0:
        missing.append(
            "  - people   : No people/attendees could be found in the text."
        )

    is_valid = len(missing) == 0
    return is_valid, missing


def _save_to_json(data: dict) -> None:
    """Save extracted data as JSON to OUTPUT_PATH."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── public function ───────────────────────────────────────────────────────────
def extract_pre_meeting_info(plain_text: str) -> dict:
    """
    Takes plain text meeting notes and extracts structured pre-meeting info.

    Returns a dict with:
        {
            "status":  "success" | "failure",
            "message": "...",
            "data":    { ... } | None
        }

    On success: saves extracted data to working_transcript/pre_meeting_notes.json
    On failure: returns which fields are missing with clear descriptions
    """
    if not plain_text or not plain_text.strip():
        return {
            "status":  "failure",
            "message": "Input text is empty. Please provide meeting notes.",
            "data":    None,
        }

    # 1. Build prompt and call LLM
    prompt   = _build_prompt(plain_text.strip())
    raw      = _call_ollama(prompt)

    # 2. Parse JSON from LLM response
    extracted = _parse_llm_response(raw)

    # 3. Validate
    is_valid, missing_fields = _validate_and_report(extracted)

    if not is_valid:
        return {
            "status":  "failure",
            "message": (
                "Could not extract all required fields from the provided text.\n"
                "Missing or undetectable fields:\n"
                + "\n".join(missing_fields)
            ),
            "data": extracted,   # return partial data so caller can inspect
        }

    # 4. Save to JSON
    _save_to_json(extracted)

    return {
        "status":  "success",
        "message": (
            f"Pre-meeting info extracted and saved successfully.\n"
            f"  - Meeting      : {extracted['meeting']}\n"
            f"  - Date         : {extracted['date']}\n"
            f"  - Meeting Type : {extracted['meeting_type']}\n"
            f"  - People       : {', '.join(p['name'] for p in extracted['people'])}"
        ),
        "data": extracted,
    }