"""
Script 1: Extract Slides from Video

Extracts the final (fully-animated) state of each slide from a presentation
recording. Uses SSIM-based frame comparison for slide transitions and an
OpenCV Haar cascade to distinguish full-face webcam sections and black
screens from real slide content.

Output:
  - output/slides/slide_001.png, slide_002.png, ...
  - output/slide_timestamps.json   — slide entries only (backward-compat)
  - output/sections.json           — all labeled time periods (slide/face/black)
"""

import cv2
import json
import os
import sys
import numpy as np
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------------------------
# Slide detection thresholds
# ---------------------------------------------------------------------------

SAMPLE_INTERVAL_SEC = 0.5      # Used only when video FPS <= 5
TRANSITION_THRESHOLD = 0.70    # SSIM below this → new slide
ANIMATION_THRESHOLD = 0.98     # SSIM above this → identical frame, skip
COMPARE_WIDTH = 640            # Resize to this width before SSIM (None = full res)

# ---------------------------------------------------------------------------
# Frame classification thresholds
# ---------------------------------------------------------------------------

# Mean grayscale brightness (0–255) below this → black/blank frame
BLACK_BRIGHTNESS_THRESHOLD = 15

# Face bounding-box area / total frame area above this → face section.
# A small PiP webcam occupies ~0.5–2% of the frame; a full-face fill is ~20–50%.
# 12% is a comfortable middle ground.
FACE_COVERAGE_THRESHOLD = 0.12

# Higher → fewer face detections (fewer false positives on slides)
FACE_MIN_NEIGHBORS = 6

# Minimum face width as a fraction of frame width (ignores tiny false positives)
FACE_MIN_SIZE_RATIO = 0.12

# How many consecutive frames of a new type must appear before switching sections.
# At 1 fps this means a new section must persist for N seconds before it is
# committed. Prevents a single misclassified frame from breaking slide detection.
SECTION_CHANGE_MIN_FRAMES = 3

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_PATH = "recording.mp4"
OUTPUT_DIR = "output"
SLIDES_DIR = os.path.join(OUTPUT_DIR, "slides")
TIMESTAMPS_FILE = os.path.join(OUTPUT_DIR, "slide_timestamps.json")
SECTIONS_FILE = os.path.join(OUTPUT_DIR, "sections.json")

# ---------------------------------------------------------------------------


def resize_for_compare(frame, width=COMPARE_WIDTH):
    if width is None:
        return frame
    h, w = frame.shape[:2]
    new_h = int(h * (width / w))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def compute_ssim(frame_a, frame_b):
    a = cv2.cvtColor(resize_for_compare(frame_a), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(resize_for_compare(frame_b), cv2.COLOR_BGR2GRAY)
    score, _ = ssim(a, b, full=True)
    return score


def load_face_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(path)
    if detector.empty():
        raise RuntimeError(f"Could not load face cascade from: {path}")
    return detector


def classify_frame(frame, face_detector):
    """Return 'black', 'face', or 'slide'."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if gray.mean() < BLACK_BRIGHTNESS_THRESHOLD:
        return "black"

    h, w = gray.shape
    min_face_px = int(w * FACE_MIN_SIZE_RATIO)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=(min_face_px, min_face_px),
    )
    if len(faces) > 0:
        face_area = sum(fw * fh for (_, _, fw, fh) in faces)
        if face_area / (h * w) > FACE_COVERAGE_THRESHOLD:
            return "face"

    return "slide"


def save_slide(frame, index, slide_start, slide_end):
    filename = f"slide_{index:03d}.png"
    cv2.imwrite(os.path.join(SLIDES_DIR, filename), frame)
    print(f"  Saved {filename}  [{slide_start:.1f}s → {slide_end:.1f}s]")
    return filename


def extract_slides(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if fps > 5:
        sample_step = int(fps)
        sample_interval_display = 1.0
    else:
        sample_step = max(1, int(fps * SAMPLE_INTERVAL_SEC))
        sample_interval_display = SAMPLE_INTERVAL_SEC

    os.makedirs(SLIDES_DIR, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}, Duration: {duration:.1f}s, Total frames: {total_frames}")
    print(f"Sampling every {sample_step} frames ({sample_interval_display}s interval)")
    print(f"Thresholds — transition: <{TRANSITION_THRESHOLD}, animation: {ANIMATION_THRESHOLD}–{TRANSITION_THRESHOLD}")
    print(f"Frame classification — black: <{BLACK_BRIGHTNESS_THRESHOLD}, face coverage: >{FACE_COVERAGE_THRESHOLD}")

    face_detector = load_face_detector()

    # Output lists
    slide_records = []  # slide_timestamps.json
    all_sections = []   # sections.json

    # Section state machine
    current_type = None     # Committed section type: 'slide' | 'face' | 'black'
    section_start = 0.0     # Start time of current committed section
    pending_type = None     # Section type we're debouncing toward
    pending_count = 0       # How many consecutive frames of pending_type seen
    pending_start = 0.0     # Timestamp of the first pending frame

    # Slide tracking (only active when current_type == 'slide')
    slide_candidate = None  # Best frame to save (last frame of the current animation)
    slide_start = 0.0       # When the current slide became active
    slide_prev = None       # Previous slide frame used for SSIM comparison
    slide_index = 1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        if frame_idx % sample_step == 0:
            ftype = classify_frame(frame, face_detector)

            if current_type is None:
                # Bootstrap: first sampled frame
                current_type = ftype
                section_start = t
                print(f"  [{t:.1f}s] Section start → {ftype}")
                if ftype == "slide":
                    slide_candidate = frame
                    slide_start = t

            elif ftype == current_type:
                # Same type — reset debounce
                pending_type = None
                pending_count = 0

                if ftype == "slide":
                    if slide_prev is None:
                        slide_prev = frame
                        slide_candidate = frame
                    else:
                        score = compute_ssim(slide_prev, frame)
                        if score < TRANSITION_THRESHOLD:
                            # New slide within slide section
                            filename = save_slide(slide_candidate, slide_index, slide_start, t)
                            rec = {"file": filename,
                                   "start": round(slide_start, 3),
                                   "end": round(t, 3)}
                            slide_records.append(rec)
                            all_sections.append({"type": "slide", **rec})
                            slide_index += 1
                            slide_candidate = frame
                            slide_start = t
                        elif score < ANIMATION_THRESHOLD:
                            slide_candidate = frame
                        slide_prev = frame

            else:
                # Different type — debounce
                if ftype == pending_type:
                    pending_count += 1
                else:
                    pending_type = ftype
                    pending_count = 1
                    pending_start = t

                if pending_count >= SECTION_CHANGE_MIN_FRAMES:
                    # Commit section change — use pending_start as the boundary
                    print(f"  [{pending_start:.1f}s] Section → {pending_type}")

                    if current_type == "slide" and slide_candidate is not None:
                        filename = save_slide(slide_candidate, slide_index,
                                              slide_start, pending_start)
                        rec = {"file": filename,
                               "start": round(slide_start, 3),
                               "end": round(pending_start, 3)}
                        slide_records.append(rec)
                        all_sections.append({"type": "slide", **rec})
                        slide_index += 1
                        slide_candidate = None
                        slide_prev = None

                    elif current_type in ("face", "black"):
                        all_sections.append({
                            "type": current_type,
                            "file": None,
                            "start": round(section_start, 3),
                            "end": round(pending_start, 3),
                        })

                    current_type = pending_type
                    section_start = pending_start
                    pending_type = None
                    pending_count = 0

                    if current_type == "slide":
                        slide_candidate = frame
                        slide_start = section_start
                        slide_prev = None

        frame_idx += 1

    cap.release()

    # Finalize the last open section
    t = duration
    if current_type == "slide" and slide_candidate is not None:
        filename = save_slide(slide_candidate, slide_index, slide_start, t)
        rec = {"file": filename, "start": round(slide_start, 3), "end": round(t, 3)}
        slide_records.append(rec)
        all_sections.append({"type": "slide", **rec})
    elif current_type in ("face", "black"):
        all_sections.append({
            "type": current_type,
            "file": None,
            "start": round(section_start, 3),
            "end": round(t, 3),
        })

    with open(TIMESTAMPS_FILE, "w") as f:
        json.dump(slide_records, f, indent=2)
    with open(SECTIONS_FILE, "w") as f:
        json.dump(all_sections, f, indent=2)

    print(f"\nSlide timestamps: {TIMESTAMPS_FILE}")
    print(f"All sections:     {SECTIONS_FILE}")

    type_counts = {}
    for s in all_sections:
        type_counts[s["type"]] = type_counts.get(s["type"], 0) + 1
    for k, v in type_counts.items():
        print(f"  {k}: {v} section(s)")

    return slide_records, all_sections


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    extract_slides(video_path)
