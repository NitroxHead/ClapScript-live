"""
Script 3: Merge Transcript with Slide Timestamps

Combines slide_timestamps.json and transcript_segments.json into a
human-readable synced_transcript.md where each block of text is
labelled with the slide(s) visible when those words were spoken.

Output:
  - output/synced_transcript.md
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# Paths (override via command-line args or environment)
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output"
TIMESTAMPS_FILE = os.path.join(OUTPUT_DIR, "slide_timestamps.json")
TRANSCRIPT_FILE = os.path.join(OUTPUT_DIR, "transcript_segments.json")
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "synced_transcript.md")

# ---------------------------------------------------------------------------


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_slide_at(time, slides):
    """Return the slide record active at `time` (or None if before first slide)."""
    active = None
    for slide in slides:
        if slide["start"] <= time:
            active = slide
        else:
            break
    return active


def find_transition_within(seg_start, seg_end, slides):
    """
    Return the first slide that starts strictly inside (seg_start, seg_end).
    Used to detect mid-sentence slide changes.
    """
    for slide in slides:
        if seg_start < slide["start"] < seg_end:
            return slide
    return None


def build_header(slide_at_start, transition_slide):
    """
    Build the [slide_XXX.png] or [slide_XXX.png → slide_YYY.png] header string.
    """
    if transition_slide and transition_slide["file"] != slide_at_start["file"]:
        return f'[{slide_at_start["file"]} → {transition_slide["file"]}]'
    return f'[{slide_at_start["file"]}]'


def merge(timestamps_path=TIMESTAMPS_FILE, transcript_path=TRANSCRIPT_FILE, output_path=OUTPUT_FILE):
    slides = load_json(timestamps_path)
    segments = load_json(transcript_path)

    lines = []
    prev_header = None

    for seg in segments:
        seg_start = seg["start"]
        seg_end   = seg["end"]
        text      = seg["text"]

        slide_at_start = find_slide_at(seg_start, slides)
        if slide_at_start is None:
            # Segment before any slide — skip or attach to first slide
            slide_at_start = slides[0] if slides else None
        if slide_at_start is None:
            continue

        transition_slide = find_transition_within(seg_start, seg_end, slides)
        header = build_header(slide_at_start, transition_slide)

        if header != prev_header:
            # Add a blank line between slide groups (except at the very start)
            if lines:
                lines.append("")
            lines.append(header)
            prev_header = header

        lines.append(text)

    output = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"Synced transcript written to: {output_path}")
    print(f"Segments processed: {len(segments)}")

    return output_path


if __name__ == "__main__":
    args = sys.argv[1:]
    timestamps_path = args[0] if len(args) > 0 else TIMESTAMPS_FILE
    transcript_path = args[1] if len(args) > 1 else TRANSCRIPT_FILE
    output_path     = args[2] if len(args) > 2 else OUTPUT_FILE
    merge(timestamps_path, transcript_path, output_path)
