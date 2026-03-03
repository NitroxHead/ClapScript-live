"""
Script 4: Run Full Pipeline

Slide extraction, transcription, and speaker diarization all start at the
same time and run in parallel. The merge step runs once all three finish.

Speaker diarization is optional — it runs automatically if the HF_TOKEN
environment variable is set, otherwise webcam/Q&A sections are labelled
[webcam].

Usage:
    python run_all.py my_recording.mp4

With speaker diarization:
    HF_TOKEN=hf_... python run_all.py my_recording.mp4
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def run_pipeline(video_path):
    if not os.path.isfile(video_path):
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN", "")

    # Build the set of tasks that run in parallel
    tasks = {
        "slide extraction": lambda: extract_slides.extract_slides(video_path),
        "transcription":    lambda: transcribe_audio.transcribe(video_path),
    }
    if hf_token:
        tasks["diarization"] = lambda: diarize_speakers.diarize(video_path)
    else:
        print("HF_TOKEN not set — diarization skipped (webcam sections labelled [webcam]).")
        print(f"To enable: HF_TOKEN=hf_... python run_all.py <video>\n")

    print(f"{DIVIDER}")
    print(f"  Starting {len(tasks)} task(s) in parallel: {', '.join(tasks)}")
    print(DIVIDER)

    failed = False
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_name = {executor.submit(fn): name for name, fn in tasks.items()}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
                print(f"\n  [done] {name}")
            except Exception as exc:
                print(f"\n  [FAILED] {name}: {exc}")
                failed = True

    if failed:
        print(f"\n{DIVIDER}")
        print("  One or more tasks failed — skipping merge.")
        print(DIVIDER)
        sys.exit(1)

    print(f"\n{DIVIDER}")
    print("  All tasks complete — merging transcript")
    print(DIVIDER)

    merge_transcript.merge(
        timestamps_path=TIMESTAMPS_FILE,
        transcript_path=TRANSCRIPT_FILE,
        output_path=OUTPUT_FILE,
        sections_path=SECTIONS_FILE,
        speaker_path=SPEAKER_FILE,
    )

    print(f"\n{DIVIDER}")
    print("  Pipeline complete!")
    print(f"  Final output: {os.path.abspath(OUTPUT_FILE)}")
    print(DIVIDER)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_all.py <video_file>")
        sys.exit(1)

    run_pipeline(sys.argv[1])
