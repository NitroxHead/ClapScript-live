"""
Script 1: Extract Slides from Video

Extracts the final (fully-animated) state of each slide from a presentation recording.
Uses SSIM-based frame comparison to distinguish animation steps from slide transitions.

Output:
  - output/slides/slide_001.png, slide_002.png, ...
  - output/slide_timestamps.json
"""

import cv2
import json
import os
import sys
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------------------------
# Tunable constants — adjust these for different recording styles
# ---------------------------------------------------------------------------

SAMPLE_INTERVAL_SEC = 0.5      # How often to sample frames (seconds)

# SSIM score below this → slide transition (new slide entirely)
TRANSITION_THRESHOLD = 0.70

# SSIM score between these two → animation step (bullet appearing, fade-in, etc.)
# Score above ANIMATION_THRESHOLD → frames are essentially identical (skip)
ANIMATION_THRESHOLD = 0.98

# Resize frames to this width before comparison (for speed; None = no resize)
COMPARE_WIDTH = 640

# Output paths
DEFAULT_VIDEO_PATH = "recording.mp4"
OUTPUT_DIR = "output"
SLIDES_DIR = os.path.join(OUTPUT_DIR, "slides")
TIMESTAMPS_FILE = os.path.join(OUTPUT_DIR, "slide_timestamps.json")

# ---------------------------------------------------------------------------


def resize_for_compare(frame, width=COMPARE_WIDTH):
    """Resize frame to a fixed width, preserving aspect ratio."""
    if width is None:
        return frame
    h, w = frame.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def frame_to_gray(frame):
    """Convert BGR frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def compute_ssim(frame_a, frame_b):
    """Compute SSIM between two BGR frames (resized and grayscale)."""
    a = frame_to_gray(resize_for_compare(frame_a))
    b = frame_to_gray(resize_for_compare(frame_b))
    score, _ = ssim(a, b, full=True)
    return score


def extract_slides(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if fps > 5:
        sample_step = int(fps)          # 1 frame per second
        sample_interval_display = 1.0
    else:
        sample_step = max(1, int(fps * SAMPLE_INTERVAL_SEC))
        sample_interval_display = SAMPLE_INTERVAL_SEC

    os.makedirs(SLIDES_DIR, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}, Duration: {duration:.1f}s, Total frames: {total_frames}")
    print(f"Sampling every {sample_step} frames ({sample_interval_display}s interval)")
    print(f"Thresholds — transition: <{TRANSITION_THRESHOLD}, animation: {ANIMATION_THRESHOLD}–{TRANSITION_THRESHOLD}")

    slide_records = []       # Final saved slides: {file, start, end}
    candidate_frame = None   # Current best (last) frame of the working slide
    candidate_time = 0.0     # Timestamp of candidate_frame
    slide_start_time = 0.0   # When the current slide became active
    slide_index = 1          # 1-based slide counter
    prev_frame = None        # Last sampled frame (for comparison)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        if frame_idx % sample_step == 0:
            if prev_frame is None:
                # Very first frame — start first slide
                prev_frame = frame
                candidate_frame = frame
                candidate_time = current_time
                slide_start_time = current_time
            else:
                score = compute_ssim(prev_frame, frame)

                if score < TRANSITION_THRESHOLD:
                    # --- Slide transition ---
                    # Save the candidate (final state of the previous slide)
                    filename = f"slide_{slide_index:03d}.png"
                    filepath = os.path.join(SLIDES_DIR, filename)
                    cv2.imwrite(filepath, candidate_frame)
                    slide_records.append({
                        "file": filename,
                        "start": round(slide_start_time, 3),
                        "end": round(current_time, 3),
                    })
                    print(f"  Saved {filename}  [{slide_start_time:.1f}s → {current_time:.1f}s]  SSIM={score:.3f}")

                    slide_index += 1
                    slide_start_time = current_time
                    candidate_frame = frame
                    candidate_time = current_time

                elif score < ANIMATION_THRESHOLD:
                    # --- Animation step — update candidate to latest frame ---
                    candidate_frame = frame
                    candidate_time = current_time

                # If score >= ANIMATION_THRESHOLD: frames are essentially identical, skip

                prev_frame = frame

        frame_idx += 1

    cap.release()

    # Save the last slide (no trailing transition event)
    if candidate_frame is not None:
        filename = f"slide_{slide_index:03d}.png"
        filepath = os.path.join(SLIDES_DIR, filename)
        cv2.imwrite(filepath, candidate_frame)
        slide_records.append({
            "file": filename,
            "start": round(slide_start_time, 3),
            "end": round(duration, 3),
        })
        print(f"  Saved {filename}  [{slide_start_time:.1f}s → {duration:.1f}s]  (final slide)")

    # Write timestamps JSON
    with open(TIMESTAMPS_FILE, "w") as f:
        json.dump(slide_records, f, indent=2)
    print(f"\nSlide timestamps written to: {TIMESTAMPS_FILE}")
    print(f"Total slides extracted: {len(slide_records)}")

    return slide_records


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    extract_slides(video_path)
