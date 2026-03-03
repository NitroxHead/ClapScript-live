"""
Script 4: Run Full Pipeline

Slide extraction and transcription run in parallel.
The merge step runs once both finish.

Usage:
    python run_all.py my_recording.mp4
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import extract_slides
import transcribe_audio
import merge_transcript

OUTPUT_DIR      = "output"
TIMESTAMPS_FILE = os.path.join(OUTPUT_DIR, "slide_timestamps.json")
TRANSCRIPT_FILE = os.path.join(OUTPUT_DIR, "transcript_segments.json")
SECTIONS_FILE   = os.path.join(OUTPUT_DIR, "sections.json")
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "synced_transcript.md")

DIVIDER = "-" * 60


def run_pipeline(video_path):
    if not os.path.isfile(video_path):
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    # Step 1: Detect language (needed for OCR language selection)
    print(DIVIDER)
    print("  Detecting language...")
    print(DIVIDER)
    lang, whisper_model = transcribe_audio.detect_language(video_path)
    ocr_lang = extract_slides.whisper_to_tesseract_lang(lang)

    # Step 2: Run extraction and transcription in parallel
    tasks = {
        "slide extraction": lambda: extract_slides.extract_slides(video_path, ocr_lang=ocr_lang),
        "transcription":    lambda: transcribe_audio.transcribe(video_path, model=whisper_model),
    }

    print(f"\n{DIVIDER}")
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
