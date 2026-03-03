"""
Script 2: Transcribe Audio from Video

Extracts audio from a video file and transcribes it using faster-whisper
(CTranslate2 backend — no numba dependency), producing sentence-level
segments with start/end timestamps.

Install:
    pip install faster-whisper

Output:
  - output/transcript_segments.json
"""

import json
import os
import sys
import tempfile

from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

WHISPER_MODEL = "medium"       # Options: tiny, base, small, medium, large, large-v2, large-v3
WHISPER_DEVICE = "cpu"         # "cpu" or "cuda" (if GPU available)
WHISPER_COMPUTE = "int8"       # "int8" (fast/low RAM) or "float16" (GPU), "float32"
DEFAULT_VIDEO_PATH = "recording.mp4"
OUTPUT_DIR = "output"
TRANSCRIPT_FILE = os.path.join(OUTPUT_DIR, "transcript_segments.json")

# ---------------------------------------------------------------------------


def extract_audio(video_path, audio_path):
    """Use ffmpeg to extract audio as a 16kHz mono WAV (Whisper-friendly)."""
    cmd = (
        f'ffmpeg -y -i "{video_path}" '
        f'-vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" '
        f'-loglevel error'
    )
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}")


def transcribe(video_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"Whisper model: {WHISPER_MODEL} ({WHISPER_DEVICE}/{WHISPER_COMPUTE})")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")

        print("Extracting audio with ffmpeg...")
        extract_audio(video_path, audio_path)

        print("Loading Whisper model...")
        model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)

        print("Transcribing...")
        raw_segments, _info = model.transcribe(audio_path, beam_size=5)

        # faster-whisper returns a lazy generator — consume it here
        segments = []
        for seg in raw_segments:
            segments.append({
                "start": round(float(seg.start), 3),
                "end":   round(float(seg.end), 3),
                "text":  seg.text.strip(),
            })

    with open(TRANSCRIPT_FILE, "w") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"\nTranscript written to: {TRANSCRIPT_FILE}")
    print(f"Total segments: {len(segments)}")

    return segments


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    transcribe(video_path)
