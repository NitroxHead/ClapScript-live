"""
Script 4: Run Full Pipeline

Runs extract_slides → transcribe_audio → merge_transcript in sequence
for a given video file.

Usage:
    python run_all.py my_recording.mp4
"""

import os
import sys

# ---------------------------------------------------------------------------
# Import the three pipeline modules directly so we get proper Python
# tracebacks rather than opaque subprocess exit codes.
# ---------------------------------------------------------------------------

import extract_slides
import transcribe_audio
import merge_transcript

# Default paths (mirrors the individual scripts)
OUTPUT_DIR      = "output"
TIMESTAMPS_FILE = os.path.join(OUTPUT_DIR, "slide_timestamps.json")
TRANSCRIPT_FILE = os.path.join(OUTPUT_DIR, "transcript_segments.json")
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
    banner("Step 1/3 — Extracting slides")
    # ------------------------------------------------------------------
    extract_slides.extract_slides(video_path)

    # ------------------------------------------------------------------
    banner("Step 2/3 — Transcribing audio")
    # ------------------------------------------------------------------
    transcribe_audio.transcribe(video_path)

    # ------------------------------------------------------------------
    banner("Step 3/3 — Merging transcript with slide timestamps")
    # ------------------------------------------------------------------
    merge_transcript.merge(TIMESTAMPS_FILE, TRANSCRIPT_FILE, OUTPUT_FILE)

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
