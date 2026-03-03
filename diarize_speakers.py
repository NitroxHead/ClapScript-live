"""
Script 5: Speaker Diarization

Identifies who is speaking and when, using pyannote.audio.
Useful for Q&A sections where multiple speakers appear.

Requirements:
  pip install pyannote.audio

  A HuggingFace token is required to download the model:
    1. Sign up / log in at https://huggingface.co
    2. Accept the model license at https://huggingface.co/pyannote/speaker-diarization-3.1
    3. Create a token at https://huggingface.co/settings/tokens
    4. Set it: export HF_TOKEN=hf_...

Output:
  - output/speaker_segments.json
"""

import json
import os
import sys
import tempfile

# HuggingFace token — set via environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", "")

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DEFAULT_VIDEO_PATH = "recording.mp4"
OUTPUT_DIR = "output"
SPEAKER_FILE = os.path.join(OUTPUT_DIR, "speaker_segments.json")

# ---------------------------------------------------------------------------


def extract_audio(video_path, audio_path):
    cmd = (
        f'ffmpeg -y -i "{video_path}" '
        f"-vn -acodec pcm_s16le -ar 16000 -ac 1 "
        f'"{audio_path}" -loglevel error'
    )
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}")


def diarize(video_path):
    if not HF_TOKEN:
        print(
            "ERROR: HuggingFace token required for speaker diarization.\n"
            "\n"
            "Steps:\n"
            "  1. Accept the model license at:\n"
            "     https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  2. Create a token at:\n"
            "     https://huggingface.co/settings/tokens\n"
            "  3. Re-run with:\n"
            "     HF_TOKEN=hf_... python diarize_speakers.py <video>\n"
        )
        return None

    from pyannote.audio import Pipeline

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"Model: {DIARIZATION_MODEL}")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.wav")

        print("Extracting audio...")
        extract_audio(video_path, audio_path)

        print("Loading diarization pipeline (may download model on first run)...")
        pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL, use_auth_token=HF_TOKEN
        )

        print("Running speaker diarization...")
        diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start":   round(turn.start, 3),
            "end":     round(turn.end, 3),
            "speaker": speaker,
        })

    with open(SPEAKER_FILE, "w") as f:
        json.dump(segments, f, indent=2)

    speakers = sorted({s["speaker"] for s in segments})
    print(f"\nSpeaker segments written to: {SPEAKER_FILE}")
    print(f"Speakers detected: {', '.join(speakers)}")
    print(f"Total segments: {len(segments)}")

    return segments


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    diarize(video_path)
