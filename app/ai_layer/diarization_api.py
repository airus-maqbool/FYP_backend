import os
import time
import requests

# ── paths ────────────────────────────────────────────────────────────────────
DIARIZED_OUTPUT_PATH = "app/Working_transcript/diarized_output_api.txt"

# ── AssemblyAI endpoints (v2 — confirmed working from official demo) ──────────
BASE_URL          = "https://api.assemblyai.com"
UPLOAD_URL        = f"{BASE_URL}/v2/upload"
TRANSCRIPT_URL    = f"{BASE_URL}/v2/transcript"

# ── load API key from environment once at startup ────────────────────────────
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    raise EnvironmentError(
        "ASSEMBLYAI_API_KEY is not set. "
        "Add it to your .env file or environment variables."
    )

# lowercase "authorization" — exactly as AssemblyAI's own demo uses it
_HEADERS = {"authorization": ASSEMBLYAI_API_KEY}


def _upload_audio(file_bytes: bytes) -> str:
    """Upload raw audio bytes to AssemblyAI and return the CDN URL."""
    response = requests.post(
        UPLOAD_URL,
        headers={**_HEADERS, "content-type": "application/octet-stream"},
        data=file_bytes,
    )
    if not response.ok:
        raise RuntimeError(
            f"Upload failed [{response.status_code}]: {response.text}"
        )
    return response.json()["upload_url"]


def _request_transcription(audio_url: str) -> str:
    """Submit transcription job — mirrors AssemblyAI's official demo config."""
    payload = {
        "audio_url":        audio_url,
        "speech_models":    ["universal-2"],   # must be a list
        "speaker_labels":   True,          # diarization
        "language_detection": True,        # auto-detect language
        "temperature":      0,             # deterministic output
    }
    response = requests.post(
        TRANSCRIPT_URL,
        json=payload,
        headers=_HEADERS,
    )
    if not response.ok:
        raise RuntimeError(
            f"Transcription request failed [{response.status_code}]: {response.text}"
        )
    return response.json()["id"]


def _poll_transcription(transcript_id: str, poll_interval: int = 3) -> dict:
    """Poll until job is complete. Returns full result dict."""
    polling_endpoint = f"{TRANSCRIPT_URL}/{transcript_id}"
    while True:
        response = requests.get(polling_endpoint, headers=_HEADERS)
        if not response.ok:
            raise RuntimeError(
                f"Polling failed [{response.status_code}]: {response.text}"
            )
        data   = response.json()
        status = data["status"]

        if status == "completed":
            return data
        elif status == "error":
            raise RuntimeError(f"AssemblyAI transcription error: {data.get('error')}")

        time.sleep(poll_interval)


def _format_utterances(utterances: list) -> str:
    lines = []
    for utt in utterances:
        lines.append(f"Speaker {utt['speaker']}: {utt['text'].strip()}")
    return "\n".join(lines)

def diarize_audio_via_api(file_bytes: bytes) -> str:
    """
    Full diarization pipeline using AssemblyAI:
        1. Upload audio
        2. Submit transcription job (speaker_labels + language_detection)
        3. Poll until complete
        4. Format & save to DIARIZED_OUTPUT_PATH
        5. Return formatted transcript string
    """
    # 1. Upload
    audio_url = _upload_audio(file_bytes)

    # 2. Submit
    transcript_id = _request_transcription(audio_url)

    # 3. Poll
    result = _poll_transcription(transcript_id)

    # 4. Format
    utterances = result.get("utterances") or []
    if not utterances:
        raise ValueError("No utterances returned — audio may be too short or silent.")

    formatted = _format_utterances(utterances)

    # 5. Save
    os.makedirs(os.path.dirname(DIARIZED_OUTPUT_PATH), exist_ok=True)
    with open(DIARIZED_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(formatted)

    return formatted