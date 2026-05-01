from fastapi import FastAPI, UploadFile, File, Form,HTTPException
import whisper
import tempfile
import os
import subprocess
from pydantic import BaseModel, EmailStr
from typing import Optional
import traceback
import json
from fastapi.middleware.cors import CORSMiddleware

# my written files functions import
from .services.file_reader import load_meeting_notes
from .ai_layer.mom_generator import generate_minutes_of_meeting
from .services.mom_storage import save_mom
from .db_queries.post_meeting import save_meeting_to_db
from .services.email_sender import send_mom_email
from .services.pdf_generator import generate_pdf_from_html
from .db_queries.auth import signup_user, login_user
from .ai_layer.diarization import diarize_audio, DIARIZATION_PATH
from .ai_layer.diarization_api import diarize_audio_via_api
from .ai_layer.pre_meeting_extractor import extract_pre_meeting_info
from .services.meeting_compiler import compile_meeting_file
from .ai_layer.live_meeting_assistance import process_transcript



# instances
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("small")

# here the full transcript will be saved.. chunk wise
TRANSCRIPT_PATH = "app/Working_transcript/full_transcript.txt"

AUDIO_PATH            = r"D:\FYP_backend\uploads\fyp_meeting.mpeg"
PRE_MEETING_USER_TEXT = "app/Working_transcript/pre_meeting_userText.json"



# request schemas
class SendEmailRequest(BaseModel):
    recipient_email: EmailStr          # e.g. "client@example.com"
    subject: str                       # e.g. "Minutes of Meeting – Project Kickoff"
    email_html: str                    # Plain HTML string from the frontend preview

class SignupRequest(BaseModel):
    email        : EmailStr
    password     : str
    full_name    : str
    company_name : str
    role         : str   # e.g. "sales_person", "manager"
    phone        : str   # e.g. "+923001234567"


class LoginRequest(BaseModel):
    email    : EmailStr
    password : str


class PreMeetingRequest(BaseModel):
    text: str   # plain text meeting notes from the user

class LiveAssistRequest(BaseModel):
    text: str   # transcribed text from frontend



@app.get("/")
async def hello():
    return "hello i am working, you can do it, keep going, improve it daily"
# transcribe + save in a file
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # 1. Save uploaded file (any format)
    with tempfile.NamedTemporaryFile(delete=False) as input_tmp:
        input_tmp.write(await file.read())
        input_path = input_tmp.name

    # 2. Normalize to WAV (16kHz, mono)
    output_path = input_path + "_normalized.wav"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            output_path
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # 3. Transcribe
    result = model.transcribe(output_path)

    # 4. Append transcription to full_transcript.txt
    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
        for segment in result["segments"]:
            f.write(f"[{segment['start']:.2f} - {segment['end']:.2f}] ")
            f.write(segment["text"].strip() + "\n")

    # 5. Cleanup temp files
    os.unlink(input_path)
    os.unlink(output_path)

    return {
        "message": "Transcription saved successfully",
        "text": result["text"]
    }


@app.post("/generate-mom")
async def generate_mom():
    # takes speaker wise notes from diaized_api.txt, pre_meeting_notes.json and pre_meeting_userText.json , gives to llm and generate a mom save in working_transcript/mom.json and send back to frontend
    try:
        meeting_notes = load_meeting_notes()
        mom = generate_minutes_of_meeting(meeting_notes)
        # save in system 
        save_mom(mom)

        # save in db

        print(mom)
        print("trying to save in db")
        response=save_meeting_to_db(mom)
        print(response)

        return {
            "status": "success",
            "meeting_id": response["meeting_id"],
            "meeting_topic": response["meeting_topic"],
            "minutes_of_meeting": mom
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate MoM: {str(e)}"
        )
    

@app.post("/send-mom-email")
async def send_mom_email_endpoint(request: SendEmailRequest):
    """
    Receives the rendered email HTML from the frontend, generates a matching
    PDF attachment, and sends everything via SMTP.

    Flow:
        Frontend renders MoM preview
            → POST /send-mom-email  { recipient_email, subject, email_html }
                → generate PDF from the same HTML   (pdf_generator)
                → send email with HTML body + PDF   (email_sender)
                    → return success / error
    """
    try:
        # 1. Convert the frontend HTML to PDF bytes
        pdf_bytes = generate_pdf_from_html(request.email_html)

        # 2. Send the email (HTML body + PDF attachment)
        result = send_mom_email(
            recipient_email=request.recipient_email,
            subject=request.subject,
            html_body=request.email_html,
            pdf_bytes=pdf_bytes,
            pdf_filename="minutes_of_meeting.pdf"
        )

        return result

    except ValueError as e:
        # Missing SMTP credentials or bad input
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send email: {str(e)}"
        )
    

@app.post("/signup")
async def signup(request: SignupRequest):
    """
    Registers a new user via Supabase Auth.

    Stores extra profile fields in user_metadata.
    Returns the created user's info (no token — user must log in after signup).
    """
    try:
        user_data = signup_user(
            email        = request.email,
            password     = request.password,
            full_name    = request.full_name,
            company_name = request.company_name,
            role         = request.role,
            phone        = request.phone
        )

        return {
            "status"  : "success",
            "message" : "Account created successfully. Please log in.",
            "user"    : user_data
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Signup failed: {str(e)}"
        )


@app.post("/login")
async def login(request: LoginRequest):
    """
    Authenticates a user and returns a JWT access token.

    Frontend should:
        1. Store the access_token (localStorage or memory)
        2. Send it in the Authorization header for protected routes:
           Authorization: Bearer <access_token>
    """
    try:
        login_data = login_user(
            email    = request.email,
            password = request.password
        )

        return {
            "status": "success",
            "data"  : login_data
        }

    except ValueError as e:
        # Wrong credentials — 401 Unauthorized
        raise HTTPException(status_code=401, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Login failed: {str(e)}"
        )

# diariaze via pyannot.audio
@app.post("/diarize")
async def diarize_audio_endpoint(
    file: UploadFile = File(...),
    num_speakers: Optional[int]  = Form(default=None),
    min_speakers: Optional[int]  = Form(default=None),
    max_speakers: Optional[int]  = Form(default=None),
):
    """
    Speaker Diarization Endpoint
    ─────────────────────────────
    Accepts any audio format (mp3, m4a, mp4, ogg, flac, wav …).

    Form fields (all optional):
        num_speakers  — exact speaker count (BEST accuracy, use when you know it)
        min_speakers  — minimum bound (use when count is uncertain)
        max_speakers  — maximum bound (use when count is uncertain)

    If none are provided, pyannote auto-detects — less accurate, may return 1 speaker.

    Examples (via curl):
        # You know there are exactly 2 speakers
        curl -X POST /diarize -F "file=@meeting.mp3" -F "num_speakers=2"

        # You think it's between 2 and 4 speakers
        curl -X POST /diarize -F "file=@meeting.mp3" -F "min_speakers=2" -F "max_speakers=4"
    """
    suffix = os.path.splitext(file.filename)[-1] if file.filename else ".audio"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        diarized_text = diarize_audio(
            input_audio_path=tmp_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        return {
            "status": "success",
            "saved_to": DIARIZATION_PATH,
            "num_speakers_hint": num_speakers or f"{min_speakers or '?'}–{max_speakers or '?'}",
            "diarization": diarized_text,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

''' post meeting steps
1. diarize audio 
2.'''

# diarization via api
# ── add this import at the top of main.py ────────────────────────────────────
from .ai_layer.diarization_api import diarize_audio_via_api

# ── add this endpoint in main.py ──────────────────────────────────────────────
@app.post("/diarize_api")
async def diarize_api(file: UploadFile = File(...)):
    """
    Accepts an audio file, runs speaker diarization via AssemblyAI,
    saves the result to app/Working_transcript/diarized_output_api.txt,
    and returns the formatted transcript.

    The AssemblyAI API key is read from the ASSEMBLYAI_API_KEY environment variable.
    """
    try:
        file_bytes = await file.read()

        transcript = diarize_audio_via_api(file_bytes=file_bytes)

        return {
            "status": "success",
            "message": "Diarization complete. Output saved to diarized_output_api.txt",
            "transcript": transcript,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Diarization failed: {str(e)}"
        )
    

@app.post("/extract-pre-meeting")
async def extract_pre_meeting(request: PreMeetingRequest):
    """
    Accepts plain text meeting notes and uses DeepSeek via Ollama to extract:
        - meeting title
        - date          (inferred as today if not mentioned)
        - meeting type  (inferred from context if not mentioned)
        - people        (name, role, company)

    Saves to app/Working_transcript/pre_meeting_notes.json on success.
    Returns a clear success or failure message with details.
    """
    try:
        result = extract_pre_meeting_info(request.text)

        # return 200 for success, 422 for extraction failure (not a server error)
        if result["status"] == "failure":
            raise HTTPException(status_code=422, detail=result)

        return result

    except HTTPException:
        raise

    except RuntimeError as e:
        # Ollama connection issues etc.
        raise HTTPException(status_code=503, detail=str(e))

    except ValueError as e:
        # LLM returned bad JSON
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pre-meeting extraction failed: {str(e)}"
        )


# combine the diarized + premeetingnotes json to make input of mom generation

@app.post("/compile-meeting")
async def compile_meeting():
    """
    Reads pre_meeting_notes.json and diarized_output_api.txt
    from the Working_transcript folder and merges them into
    meeting_notes_dialouge.txt in the same folder.
    but later i will name into compiled_meeting.txt
    Run this after:
        1. /extract-pre-meeting  (generates pre_meeting_notes.json)
        2. /diarize-api          (generates diarized_output_api.txt)
    """
    try:
        result = compile_meeting_file()
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compile meeting file: {str(e)}"
        )



# postmeeting automation for end meeting button (orchestration of all post meeting endpoints)

@app.post("/post-meeting-automation")
async def post_meeting_automation():

    # ── Step 1: Diarize ───────────────────────────────────────────────────────
    with open(AUDIO_PATH, "rb") as f:
        audio_bytes = f.read()
    diarize_audio_via_api(file_bytes=audio_bytes)

    # ── Step 2: Extract pre-meeting info ─────────────────────────────────────
    with open(PRE_MEETING_USER_TEXT, "r", encoding="utf-8") as f:
        plain_text = json.load(f)["text"]
    extract_pre_meeting_info(plain_text)

    # ── Step 3: Compile meeting file ─────────────────────────────────────────
    compile_meeting_file()

    # ── Step 4: Generate MoM ─────────────────────────────────────────────────
    meeting_notes = load_meeting_notes()
    mom = generate_minutes_of_meeting(meeting_notes)
    save_mom(mom)

    print("trying to save in db")
    response=save_meeting_to_db(mom)
    print(response)

    return {
            "status": "success",
            "meeting_id": response["meeting_id"],
            "meeting_topic": response["meeting_topic"],
            "minutes_of_meeting": mom
        }


@app.post("/live-assist")
async def live_assist_text(request: LiveAssistRequest):
    """
    Receives transcribed text from frontend (frontend already did audio → text).
 
    Pipeline:
        1. Detect if text is a product-related question (Ollama LLM call)
        2. If yes, generate answer from knowledge_base.json (Ollama LLM call)
        3. Return result to frontend
 
    Frontend receives:
        {
            "is_product_question": true / false,
            "answer": "crafted answer from knowledge base" or null
        }
    """
    try:
        result = process_transcript(request.text)
        return result
 
    except RuntimeError as e:
        # ollama connection issues
        raise HTTPException(status_code=503, detail=str(e))
 
    except FileNotFoundError as e:
        # knowledge base missing
        raise HTTPException(status_code=404, detail=str(e))
 
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Live assist failed: {str(e)}"
        )
 