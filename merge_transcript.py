"""
Script 3: Merge Transcript with Slide Timestamps

Combines slide/section data and transcript segments into a human-readable
synced_transcript.md where each block of text is labelled with what was
on screen when those words were spoken.

If sections.json is present (produced by extract_slides.py), it is used
to identify slide, face/webcam, and black-screen periods. Transcript
segments that fall during black screens are silently dropped. Segments
during face/webcam sections are labelled with a speaker ID from
extract_slides' visual matching, otherwise [webcam].

Falls back to slide_timestamps.json only if sections.json is absent
(backward compatibility with older runs).

Input:
  - output/slide_timestamps.json
  - output/transcript_segments.json
  - output/sections.json           (optional but recommended)

Output:
  - output/synced_transcript.md
"""

import json
import os
import sys

OUTPUT_DIR = "output"
TIMESTAMPS_FILE = os.path.join(OUTPUT_DIR, "slide_timestamps.json")
TRANSCRIPT_FILE = os.path.join(OUTPUT_DIR, "transcript_segments.json")
SECTIONS_FILE   = os.path.join(OUTPUT_DIR, "sections.json")
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "synced_transcript.md")

# ---------------------------------------------------------------------------


def load_json(path):
    with open(path) as f:
        return json.load(f)


# --- Helpers for sections-aware mode ----------------------------------------

def find_section_at(time, sections):
    """Return the section record active at `time`."""
    active = None
    for sec in sections:
        if sec["start"] <= time:
            active = sec
        else:
            break
    return active


def find_next_slide_after(time, sections):
    """Return the next slide section that starts strictly after `time`."""
    for sec in sections:
        if sec["type"] == "slide" and sec["start"] > time:
            return sec
    return None


def header_from_sections(seg_start, seg_end, sections):
    """
    Return the header string for a transcript segment, or None to skip it.
    """
    sec = find_section_at(seg_start, sections)
    if sec is None:
        return "[unknown]"

    if sec["type"] == "black":
        return None  # Drop segments that fall on blank screens

    if sec["type"] == "face":
        speaker = sec.get("speaker")
        if speaker:
            return f"[{speaker}]"
        return "[webcam]"

    # sec["type"] == "slide"
    # Check whether a slide-to-slide transition occurs mid-segment
    next_slide = find_next_slide_after(sec["start"], sections)
    if next_slide and next_slide["start"] < seg_end:
        return f'[{sec["file"]} \u2192 {next_slide["file"]}]'
    return f'[{sec["file"]}]'


# --- Helpers for slides-only fallback mode -----------------------------------

def find_slide_at(time, slides):
    active = None
    for slide in slides:
        if slide["start"] <= time:
            active = slide
        else:
            break
    return active


def find_transition_within(seg_start, seg_end, slides):
    for slide in slides:
        if seg_start < slide["start"] < seg_end:
            return slide
    return None


def build_slide_header(slide_at_start, transition_slide):
    if transition_slide and transition_slide["file"] != slide_at_start["file"]:
        return f'[{slide_at_start["file"]} \u2192 {transition_slide["file"]}]'
    return f'[{slide_at_start["file"]}]'


# ---------------------------------------------------------------------------

def merge(
    timestamps_path=TIMESTAMPS_FILE,
    transcript_path=TRANSCRIPT_FILE,
    output_path=OUTPUT_FILE,
    sections_path=SECTIONS_FILE,
):
    slides = load_json(timestamps_path)
    segments = load_json(transcript_path)

    sections = load_json(sections_path) if os.path.exists(sections_path) else None

    if sections:
        print("Using sections.json for slide/face/black classification.")
    else:
        print("sections.json not found — falling back to slide-only mode.")

    lines = []
    prev_header = None

    for seg in segments:
        seg_start = seg["start"]
        seg_end   = seg["end"]
        text      = seg["text"]

        if sections:
            header = header_from_sections(seg_start, seg_end, sections)
        else:
            slide_at_start = find_slide_at(seg_start, slides)
            if slide_at_start is None:
                slide_at_start = slides[0] if slides else None
            if slide_at_start is None:
                continue
            transition = find_transition_within(seg_start, seg_end, slides)
            header = build_slide_header(slide_at_start, transition)

        if header is None:
            # Black screen segment — skip and break grouping so the next real
            # section always gets a fresh header even if it shares a label.
            prev_header = None
            continue

        if header != prev_header:
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
    merge(
        timestamps_path=args[0] if len(args) > 0 else TIMESTAMPS_FILE,
        transcript_path=args[1] if len(args) > 1 else TRANSCRIPT_FILE,
        output_path    =args[2] if len(args) > 2 else OUTPUT_FILE,
    )
