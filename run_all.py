"""
Script 4: Run Full Pipeline

Runs extract_slides → transcribe_audio → (diarize_speakers) → merge_transcript
in sequence for a given video file.

Speaker diarization is optional. It runs automatically if the HF_TOKEN
environment variable is set. Otherwise it is skipped and webcam/Q&A sections
will be labelled [webcam] instead of [SPEAKER_XX].

Usage:
    python run_all.py my_recording.mp4

With speaker diarization:
    HF_TOKEN=hf_... python run_all.py my_recording.mp4
"""

import os
import sys

import extract_slides
import transcribe_audio
import diarize_speakers
import merge_transcript

OUTPUT_DIR      = "output"
TIMESTAMPS_FILE = os.path.join(OUTPUT_DIR, "slide_timestamps.json")
TRANSCRIPT_FILE = os.path.join(OUTPUT_DIR, "transcript_segments.json")
SECTIONS_FILE   = os.path.join(OUTPUT_DIR, "sections.json")
SPEAKER_FILE    = os.path.join(OUTPUT_DIR, "speaker_segments.json")
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "synced_transcript.md")

DIVIDER = "-" * 60


def banner(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def run_pipeline(video_path):
    if not os.path.isfile(video_path):
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    banner("Step 1/4 — Extracting slides")
    # ------------------------------------------------------------------
    extract_slides.extract_slides(video_path)

    # ------------------------------------------------------------------
    banner("Step 2/4 — Transcribing audio")
    # ------------------------------------------------------------------
    transcribe_audio.transcribe(video_path)

    # ------------------------------------------------------------------
    banner("Step 3/4 — Speaker diarization (optional)")
    # ------------------------------------------------------------------
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        diarize_speakers.diarize(video_path)
    else:
        print("HF_TOKEN not set — skipping speaker diarization.")
        print("Webcam/Q&A sections will be labelled [webcam].")
        print("To enable: HF_TOKEN=hf_... python run_all.py <video>")

    # ------------------------------------------------------------------
    banner("Step 4/4 — Merging transcript with slide/section data")
    # ------------------------------------------------------------------
    merge_transcript.merge(
        timestamps_path=TIMESTAMPS_FILE,
        transcript_path=TRANSCRIPT_FILE,
        output_path=OUTPUT_FILE,
        sections_path=SECTIONS_FILE,
        speaker_path=SPEAKER_FILE,
    )

    # ------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print("  Pipeline complete!")
    print(f"  Final output: {os.path.abspath(OUTPUT_FILE)}")
    print(DIVIDER)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_all.py <video_file>")
        sys.exit(1)

    run_pipeline(sys.argv[1])
