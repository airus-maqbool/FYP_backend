"""
app/ai_layer/diarization.py

Speaker diarization using pyannote.audio 3.1 + torch 2.5.1
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import torch
from pyannote.audio import Pipeline


# ── Config ────────────────────────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
DIARIZATION_PATH = "app/Working_transcript/diarized_output.txt"
_pipeline: Pipeline | None = None


# ── Private helpers ───────────────────────────────────────────────────────────

def _get_pipeline() -> Pipeline:
    global _pipeline

    if _pipeline is not None:
        print("[Diarization] Pipeline already loaded, reusing.")
        return _pipeline

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN environment variable is not set.\n"
            "Run:  $env:HF_TOKEN='hf_your_token_here'  (PowerShell)"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Diarization] Loading pipeline on device: {device} ...")

    _pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )
    _pipeline.to(device)
    print("[Diarization] Pipeline loaded successfully.")

    return _pipeline


def _convert_to_wav(input_path: str) -> str:
    """Convert any audio format to 16-kHz mono WAV via ffmpeg."""
    print(f"[Diarization] Converting audio to WAV: {input_path}")

    tmp = tempfile.NamedTemporaryFile(suffix="_diarized.wav", delete=False)
    tmp.close()
    wav_path = tmp.name

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            wav_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg conversion failed:\n{err}")

    size_mb = os.path.getsize(wav_path) / (1024 * 1024)
    print(f"[Diarization] WAV ready: {wav_path} ({size_mb:.1f} MB)")
    return wav_path


def _format_diarization(diarization) -> str:
    """Convert pyannote Annotation to readable transcript string."""

    def _fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    lines = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        lines.append(f"[{_fmt(turn.start)} - {_fmt(turn.end)}] {speaker}:")

    return "\n".join(lines)


def _save_diarization(text: str) -> None:
    """Save diarized transcript to file."""
    output_path = Path(DIARIZATION_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[Diarization] Transcript saved to: {DIARIZATION_PATH}")


# ── Public API ────────────────────────────────────────────────────────────────

def diarize_audio(
    input_audio_path: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> str:
    """
    Full diarization flow:
        1. Convert audio to 16-kHz mono WAV  (ffmpeg)
        2. Run pyannote SpeakerDiarization 3.1
        3. Format result into readable transcript
        4. Save to DIARIZATION_PATH

    Args:
        input_audio_path : Path to raw audio (mp3, m4a, mp4, ogg, wav, ...)
        num_speakers     : Exact number of speakers (use this if you know it).
                           Overrides min/max when provided.
        min_speakers     : Minimum number of speakers (used when exact count unknown).
        max_speakers     : Maximum number of speakers (used when exact count unknown).

    Returns:
        Formatted diarization string.
    """
    wav_path: str | None = None

    try:
        wav_path = _convert_to_wav(input_audio_path)
        pipeline = _get_pipeline()

        # Build inference kwargs — only pass what was provided
        infer_kwargs: dict = {}
        if num_speakers is not None:
            infer_kwargs["num_speakers"] = num_speakers
            print(f"[Diarization] Using exact speaker count: {num_speakers}")
        else:
            if min_speakers is not None:
                infer_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                infer_kwargs["max_speakers"] = max_speakers
            if infer_kwargs:
                print(f"[Diarization] Speaker hint: {infer_kwargs}")
            else:
                print("[Diarization] No speaker count provided — pyannote will auto-detect (less accurate).")

        print("[Diarization] Running speaker diarization — this may take several minutes on CPU...")
        diarization = pipeline(wav_path, **infer_kwargs)
        print("[Diarization] Diarization complete.")

        formatted = _format_diarization(diarization)
        _save_diarization(formatted)

        unique_speakers = set(speaker for _, _, speaker in diarization.itertracks(yield_label=True))
        print(f"[Diarization] Detected {len(unique_speakers)} speaker(s): {sorted(unique_speakers)}")

        return formatted

    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)
            print("[Diarization] Temp WAV file cleaned up.")