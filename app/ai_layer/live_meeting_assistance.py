import os
import json
import tempfile
import requests
import subprocess

# ── config ────────────────────────────────────────────────────────────────────
OLLAMA_URL          = "http://localhost:11434/api/generate"
MODEL_NAME          = "deepseek-v3.1:671b-cloud"
KNOWLEDGE_BASE_PATH = "app/Working_transcript/knowledge_base.json"

# ── load knowledge base once at module load ───────────────────────────────────
def _load_knowledge_base() -> dict:
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        raise FileNotFoundError(
            f"Knowledge base not found at {KNOWLEDGE_BASE_PATH}. "
            "Please create knowledge_base.json in Working_transcript folder."
        )
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

KNOWLEDGE_BASE = _load_knowledge_base()


# ── function 1: transcribe audio chunk ───────────────────────────────────────
def transcribe_audio_chunk(audio_bytes: bytes) -> str:
    """
    Takes raw audio bytes (webm from browser), normalizes to WAV via ffmpeg,
    transcribes using Whisper, returns plain text.
    """
    # write to temp file and close immediately so ffmpeg can access it on Windows
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in_path = tmp_in.name
    # file is now closed — ffmpeg can open it

    # clean output path — no double extension issue
    tmp_out_path = tmp_in_path.replace(".webm", "_normalized.wav")

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", tmp_out_path],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed to process audio: {result.stderr}")

        from app.main import model as whisper_model
        transcription = whisper_model.transcribe(tmp_out_path)
        return transcription["text"].strip()

    finally:
        if os.path.exists(tmp_in_path):
            os.unlink(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.unlink(tmp_out_path)


# ── function 2: call ollama ───────────────────────────────────────────────────
def _call_ollama(prompt: str) -> str:
    """Send prompt to local Ollama DeepSeek and return response text."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Make sure it is running on localhost:11434.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out.")
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {str(e)}")


# ── function 3: detect if product question ────────────────────────────────────
def _is_product_question(transcript: str) -> bool:
    """
    Dedicated LLM call ONLY to decide yes or no —
    is this text asking about TaskMaster Pro?
    Small focused prompt = reliable answer.
    """
    product_name = KNOWLEDGE_BASE.get("product_name", "the product")
    topics = (
        "pricing, plans, cost, features, trial, support, "
        "training, security, storage, integrations, onboarding, "
        "mobile app, refund, cancel, users, reminders, reports, attachments"
    )

    prompt = f"""You are a classifier. Your only job is to answer YES or NO.

Question: Is the following text asking about {product_name} — its {topics}?

Text: "{transcript}"

Rules:
- Answer YES if the text is a question or query about the product, its features, pricing, or any business detail.
- Answer NO if it is small talk, general conversation, greetings, unrelated topics, or statements that do not require product knowledge.
- Reply with ONLY the single word YES or NO. Nothing else.
"""

    response = _call_ollama(prompt).strip().upper()
    return "YES" in response


# ── function 4: generate answer from knowledge base ──────────────────────────
def _generate_answer(transcript: str) -> str:
    """
    Dedicated LLM call ONLY to craft the answer.
    Passes full knowledge base as context.
    """
    prompt = f"""You are a knowledgeable sales assistant for {KNOWLEDGE_BASE.get("product_name", "the product")}.

You have the following product knowledge base available:
{json.dumps(KNOWLEDGE_BASE, indent=2)}

A meeting participant asked:
"{transcript}"

Your job:
- Find the relevant information from the knowledge base above
- Craft a clear, natural, professional, and concise answer
- Only use information present in the knowledge base, do not make anything up
- Answer in 2 to 4 sentences maximum
- Do not use bullet points, just plain conversational sentences
- Do not start with phrases like "Based on the knowledge base", just answer directly
"""

    return _call_ollama(prompt).strip()


# ── function 5: detect and answer ────────────────────────────────────────────
def detect_and_answer(transcript: str) -> dict:
    """
    Call 1 — is this a product question? (YES/NO only, very focused)
    Call 2 — if YES, generate answer from knowledge base
    Two separate focused calls = reliable detection + accurate answers
    """
    is_product = _is_product_question(transcript)

    if not is_product:
        return {
            "is_product_question": False,
            "answer": None
        }

    answer = _generate_answer(transcript)

    return {
        "is_product_question": True,
        "answer": answer
    }


# ── main public function ──────────────────────────────────────────────────────
def process_audio_chunk(audio_bytes: bytes) -> dict:
    """
    Full pipeline:
        1. Transcribe audio chunk to text via Whisper
        2. Detect if it is a product question (focused YES/NO call)
        3. If yes, generate answer from knowledge base (focused answer call)
        4. Return everything to the endpoint
    """
    transcript = transcribe_audio_chunk(audio_bytes)

    if not transcript:
        return {
            "transcript":          "",
            "is_product_question": False,
            "answer":              None
        }

    result = detect_and_answer(transcript)

    return {
        "transcript":          transcript,
        "is_product_question": result["is_product_question"],
        "answer":              result["answer"]
    }