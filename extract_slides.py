"""
Extract slides and speaker sections from a presentation recording.

Classifies each frame as face-dominant (speaker fills frame) or
slide-dominant (presentation content, possibly with small PiP camera).

  Face sections  → face crops collected, clustered by speaker via PCA
  Slide sections → pixel-diff tracking saves the final version of each slide
                   (the last stable frame before content changes)

Face detection runs on every sampled frame regardless of section type:
  - Face sections: face crops extracted for speaker clustering
  - Slide sections: face region masked out so PiP movement doesn't
    trigger false slide transitions

Output:
  - output/slides/slide_001.png, slide_002.png, ...
  - output/slides/speaker_01.png, speaker_02.png, ...
  - output/slide_timestamps.json
  - output/sections.json
"""

import cv2
import json
import logging
import os
import re
import sys
import numpy as np

_PYTESSERACT_AVAILABLE = False
try:
    import pytesseract
    _PYTESSERACT_AVAILABLE = True
except ImportError:
    pass

# ===================================================================
# Logging — change LOG_LEVEL to control verbosity
# ===================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

log = logging.getLogger("extract_slides")

# ===================================================================
# Frame sampling
# ===================================================================

SAMPLE_INTERVAL_SEC = 2  # Used only when video FPS <= 5

# ===================================================================
# Black screen
# ===================================================================

BLACK_BRIGHTNESS_THRESHOLD = 15  # Mean grayscale below this → black

# ===================================================================
# Face detection (OpenCV DNN)
# ===================================================================

FACE_CONFIDENCE_THRESHOLD = 0.5   # Ignore detections below this

# Face area / frame area above this → face-dominant section.
# PiP overlay ≈ 0.4%  |  normal webcam ≈ 3–8%  |  close-up ≈ 10–50%
CAMERA_FACE_FRACTION = 0.03

# ===================================================================
# Slide change detection (pixel-diff based)
# ===================================================================

PIXEL_DIFF_THRESHOLD = 30     # Per-pixel grayscale diff must exceed this
SLIDE_CHANGE_FRACTION = 0.10  # ≥10% of (unmasked) pixels changed → new slide
MIN_SLIDE_DURATION = 2.0      # Don't save slides shorter than this (seconds)
SLIDE_DEDUP_FRACTION = 0.05   # Skip saving if <5% different from last saved slide

# ===================================================================
# OCR text change detection (requires pytesseract + tesseract)
# ===================================================================

OCR_TEXT_REMOVAL_FRACTION = 0.50  # ≥50% of words removed → new slide
OCR_MIN_WORD_COUNT = 3            # Need at least 3 words to trigger OCR comparison

# Whisper (ISO 639-1) → Tesseract language codes
WHISPER_TO_TESSERACT = {
    "af": "afr", "ar": "ara", "bg": "bul", "bn": "ben",
    "ca": "cat", "cs": "ces", "cy": "cym", "da": "dan",
    "de": "deu", "el": "ell", "en": "eng", "es": "spa",
    "et": "est", "fa": "fas", "fi": "fin", "fr": "fra",
    "he": "heb", "hi": "hin", "hr": "hrv", "hu": "hun",
    "id": "ind", "is": "isl", "it": "ita", "ja": "jpn",
    "ko": "kor", "lt": "lit", "lv": "lav", "mk": "mkd",
    "ml": "mal", "ms": "msa", "nl": "nld", "no": "nor",
    "pl": "pol", "pt": "por", "ro": "ron", "ru": "rus",
    "sk": "slk", "sl": "slv", "sr": "srp", "sv": "swe",
    "ta": "tam", "te": "tel", "th": "tha", "tr": "tur",
    "uk": "ukr", "ur": "urd", "vi": "vie", "zh": "chi_sim",
}


def whisper_to_tesseract_lang(whisper_code):
    """Map Whisper ISO 639-1 code to Tesseract language code."""
    return WHISPER_TO_TESSERACT.get(whisper_code, "eng")


# ===================================================================
# Speaker clustering (PCA + greedy cosine matching)
# ===================================================================

FACE_CROP_SIZE = 64           # Resize face crops to NxN before PCA
PCA_COMPONENTS = 32           # Dimensions after PCA
SPEAKER_COSINE_THRESHOLD = 0.30  # Cosine distance < this → same speaker

# ===================================================================
# Section debounce
# ===================================================================

SECTION_CHANGE_MIN_FRAMES = 3  # Consecutive frames of new type before commit

# ===================================================================
# Paths
# ===================================================================

DEFAULT_VIDEO_PATH = "recording.mp4"
OUTPUT_DIR = "output"
SLIDES_DIR = os.path.join(OUTPUT_DIR, "slides")
TIMESTAMPS_FILE = os.path.join(OUTPUT_DIR, "slide_timestamps.json")
SECTIONS_FILE = os.path.join(OUTPUT_DIR, "sections.json")


# ===================================================================
# Logging setup
# ===================================================================

def setup_logging():
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ===================================================================
# Face detection
# ===================================================================

def load_face_detector():
    """Load OpenCV DNN face detector (ResNet-SSD, ~5 MB)."""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
    config_path = os.path.join(model_dir, "deploy.prototxt")
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise RuntimeError(
            f"DNN face model not found in {model_dir}\n"
            "See: https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"
        )
    return cv2.dnn.readNetFromCaffe(config_path, model_path)


def detect_faces(frame, detector):
    """Return list of (x1, y1, x2, y2, confidence) tuples."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(blob)
    raw = detector.forward()

    faces = []
    for i in range(raw.shape[2]):
        conf = raw[0, 0, i, 2]
        if conf < FACE_CONFIDENCE_THRESHOLD:
            continue
        box = raw[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            faces.append((x1, y1, x2, y2, float(conf)))
    return faces


def largest_face(faces):
    """Return the largest face by area, or None."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))


def face_area_fraction(face, frame_shape):
    """Face bounding-box area as a fraction of frame area."""
    if face is None:
        return 0.0
    x1, y1, x2, y2 = face[:4]
    h, w = frame_shape[:2]
    return (x2 - x1) * (y2 - y1) / (h * w)


# ===================================================================
# Frame classification
# ===================================================================

def classify_layout(frame, faces):
    """Return 'black', 'face', or 'slide'."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < BLACK_BRIGHTNESS_THRESHOLD:
        return "black"

    face = largest_face(faces)
    if face_area_fraction(face, frame.shape) > CAMERA_FACE_FRACTION:
        return "face"

    return "slide"


# ===================================================================
# Slide change detection (pixel-diff)
# ===================================================================

_DIFF_SIZE = (640, 360)


def compute_slide_diff(frame_a, frame_b, mask=None):
    """Fraction of significantly changed pixels between two frames.

    Resizes to 640x360 internally for speed.  An optional mask
    (255 = include, 0 = exclude) lets us ignore the PiP camera region.
    """
    ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    sa = cv2.resize(ga, _DIFF_SIZE, interpolation=cv2.INTER_AREA)
    sb = cv2.resize(gb, _DIFF_SIZE, interpolation=cv2.INTER_AREA)

    diff = cv2.absdiff(sa, sb)
    changed = diff > PIXEL_DIFF_THRESHOLD

    if mask is not None:
        sm = cv2.resize(mask, _DIFF_SIZE, interpolation=cv2.INTER_NEAREST)
        changed = changed & (sm > 0)
        total = np.count_nonzero(sm > 0)
    else:
        total = changed.size

    if total == 0:
        return 0.0
    return np.count_nonzero(changed) / total


def build_face_mask(frame_shape, faces):
    """Binary mask: 255 everywhere except face regions (set to 0).

    Pads around each face box to cover hair/shoulders so PiP speaker
    movement doesn't contaminate the slide-diff calculation.
    """
    h, w = frame_shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for face in faces:
        x1, y1, x2, y2 = face[:4]
        pad_x = int((x2 - x1) * 0.3)
        pad_y = int((y2 - y1) * 0.5)
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        mask[y1:y2, x1:x2] = 0
    return mask


# ===================================================================
# OCR text extraction and comparison
# ===================================================================

def extract_text(frame, mask=None, lang="eng"):
    """OCR text from a slide frame, optionally masking out face regions."""
    img = frame.copy()
    if mask is not None:
        img[mask == 0] = (255, 255, 255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w > 960:
        scale = 960 / w
        gray = cv2.resize(gray, (960, int(h * scale)))
    try:
        return pytesseract.image_to_string(gray, lang=lang).strip()
    except Exception:
        return ""


def _normalize_words(text):
    """Extract lowercase alphanumeric tokens (2+ chars) from OCR text."""
    return set(re.findall(r'\w{2,}', text.lower()))


def text_removal_fraction(prev_words, curr_words):
    """Fraction of previous words no longer present in current text."""
    if not prev_words:
        return 0.0
    removed = prev_words - curr_words
    return len(removed) / len(prev_words)


# ===================================================================
# Speaker clustering  (PCA via numpy, greedy cosine assignment)
# ===================================================================

def extract_face_feature(frame, face):
    """64x64 grayscale face crop → flat float32 vector (4096-d)."""
    x1, y1, x2, y2 = face[:4]
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (FACE_CROP_SIZE, FACE_CROP_SIZE),
                         interpolation=cv2.INTER_AREA)
    return resized.flatten().astype(np.float32)


def _cosine_dist(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return 1.0 - dot / (na * nb)


def cluster_speakers(face_data):
    """Cluster collected face features into speaker groups.

    Args:
        face_data: list of (section_idx, feature_vector) tuples

    Returns:
        dict  { section_idx: "speaker_01", ... }
    """
    if not face_data:
        return {}

    indices = [d[0] for d in face_data]
    features = np.array([d[1] for d in face_data])

    # Standardise
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-7
    normed = (features - mean) / std

    # PCA (numpy SVD)
    n_comp = min(PCA_COMPONENTS, normed.shape[0], normed.shape[1])
    if n_comp < 1:
        return {idx: "speaker_01" for idx in indices}

    _, _, Vt = np.linalg.svd(normed, full_matrices=False)
    projected = normed @ Vt[:n_comp].T

    log.debug(f"PCA: {features.shape[1]}d → {n_comp}d  ({len(features)} samples)")

    # Greedy cosine clustering
    centroids = []   # list of np.array
    members = []     # list of [section_idx, ...]

    for i, feat in enumerate(projected):
        best_c, best_d = -1, float("inf")
        for ci, cent in enumerate(centroids):
            d = _cosine_dist(feat, cent)
            if d < best_d:
                best_d = d
                best_c = ci

        if best_c >= 0 and best_d < SPEAKER_COSINE_THRESHOLD:
            members[best_c].append(indices[i])
            # Running centroid update
            n = len(members[best_c])
            centroids[best_c] = centroids[best_c] * ((n - 1) / n) + feat / n
        else:
            centroids.append(feat.copy())
            members.append([indices[i]])

    labels = {}
    for spk_idx, group in enumerate(members):
        spk_id = f"speaker_{spk_idx + 1:02d}"
        for sec_idx in group:
            labels[sec_idx] = spk_id

    log.info(f"Speaker clustering: {len(face_data)} face samples → "
             f"{len(centroids)} speaker(s)")
    return labels


# ===================================================================
# Helpers
# ===================================================================

def _save_slide(frame, index, t_start, t_end):
    filename = f"slide_{index:03d}.png"
    cv2.imwrite(os.path.join(SLIDES_DIR, filename), frame)
    log.info(f"  Saved {filename}  [{t_start:.1f}s → {t_end:.1f}s]")
    return filename


def _save_face(frame, index, t_start, t_end):
    filename = f"speaker_{index:02d}.png"
    cv2.imwrite(os.path.join(SLIDES_DIR, filename), frame)
    log.info(f"  Saved {filename}  [{t_start:.1f}s → {t_end:.1f}s]")
    return filename


# ===================================================================
# Main extraction
# ===================================================================

def extract_slides(video_path, ocr_lang=None):
    setup_logging()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if fps > 5:
        sample_step = int(fps)
    else:
        sample_step = max(1, int(fps * SAMPLE_INTERVAL_SEC))

    os.makedirs(SLIDES_DIR, exist_ok=True)

    log.info(f"Video: {video_path}")
    log.info(f"FPS: {fps:.2f}  Duration: {duration:.1f}s  Frames: {total_frames}")
    log.info(f"Sampling every {sample_step} frames  (~{sample_step / fps:.1f}s)")
    if _PYTESSERACT_AVAILABLE:
        if ocr_lang is None:
            ocr_lang = "eng"
        log.info(f"OCR text change detection: enabled (lang={ocr_lang})")
    else:
        log.info("OCR text change detection: disabled (pip install pytesseract)")

    face_detector = load_face_detector()

    # ---- outputs ----
    slide_records = []
    all_sections = []

    # ---- section state machine ----
    current_type = None
    section_start = 0.0
    pending_type = None
    pending_count = 0
    pending_start = 0.0

    # ---- slide tracking ----
    slide_index = 1
    slide_candidate = None   # Latest frame — saved as "final version" on transition
    slide_start = 0.0
    slide_prev = None        # Previous frame for pixel-diff
    slide_mask = None        # Face exclusion mask (updated when faces detected)
    slide_text = ""          # OCR text of current slide (max text seen)
    last_saved_slide = None  # Last saved slide frame (for dedup)

    # ---- face / speaker tracking ----
    face_candidate = None
    face_candidate_box = None
    face_data = []           # [(section_idx, feature_vector)]  clustered at the end

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        if frame_idx % sample_step == 0:
            faces = detect_faces(frame, face_detector)
            ftype = classify_layout(frame, faces)

            log.debug(f"[{t:.1f}s] frame={frame_idx}  type={ftype}  "
                      f"faces={len(faces)}  "
                      f"biggest={face_area_fraction(largest_face(faces), frame.shape):.3f}")

            # ---- bootstrap ----
            if current_type is None:
                current_type = ftype
                section_start = t
                log.info(f"[{t:.1f}s] Section start → {ftype}")
                if ftype == "slide":
                    slide_candidate = frame
                    slide_start = t
                    slide_prev = frame
                    slide_mask = build_face_mask(frame.shape, faces) if faces else None
                    if _PYTESSERACT_AVAILABLE:
                        slide_text = extract_text(frame, slide_mask, ocr_lang)
                elif ftype == "face":
                    face_candidate = frame
                    face_candidate_box = largest_face(faces)

            # ---- same type as current section ----
            elif ftype == current_type:
                pending_type = None
                pending_count = 0

                if ftype == "slide":
                    # Update face mask when faces detected (PiP position can shift)
                    if faces:
                        slide_mask = build_face_mask(frame.shape, faces)

                    if slide_prev is not None:
                        diff = compute_slide_diff(slide_prev, frame, slide_mask)
                        log.debug(f"  slide diff={diff:.4f}")

                        # OCR: detect significant text removal
                        ocr_triggered = False
                        if _PYTESSERACT_AVAILABLE:
                            curr_text = extract_text(frame, slide_mask, ocr_lang)
                            curr_words = _normalize_words(curr_text)
                            prev_words = _normalize_words(slide_text)
                            if len(prev_words) >= OCR_MIN_WORD_COUNT:
                                removal = text_removal_fraction(
                                    prev_words, curr_words)
                                if removal >= OCR_TEXT_REMOVAL_FRACTION:
                                    ocr_triggered = True
                                    log.info(
                                        f"  OCR: {removal:.0%} text removed "
                                        f"→ new slide")
                            # Track max text seen for this slide
                            if len(curr_words) > len(prev_words):
                                slide_text = curr_text

                        if diff >= SLIDE_CHANGE_FRACTION or ocr_triggered:
                            # Significant change → save previous candidate
                            if t - slide_start >= MIN_SLIDE_DURATION:
                                # Dedup: skip if too similar to last saved
                                is_dup = False
                                if last_saved_slide is not None:
                                    dedup = compute_slide_diff(
                                        last_saved_slide, slide_candidate,
                                        slide_mask)
                                    if dedup < SLIDE_DEDUP_FRACTION:
                                        is_dup = True
                                        log.debug(
                                            f"  Skipping duplicate slide "
                                            f"(dedup diff={dedup:.4f})")
                                if not is_dup:
                                    fn = _save_slide(slide_candidate,
                                                     slide_index,
                                                     slide_start, t)
                                    rec = {"file": fn,
                                           "start": round(slide_start, 3),
                                           "end": round(t, 3)}
                                    slide_records.append(rec)
                                    all_sections.append(
                                        {"type": "slide", **rec})
                                    slide_index += 1
                                    last_saved_slide = slide_candidate
                            slide_start = t
                            # Reset OCR tracking for new slide
                            if _PYTESSERACT_AVAILABLE:
                                slide_text = curr_text

                    # Always keep latest frame as candidate
                    slide_candidate = frame
                    slide_prev = frame

                elif ftype == "face":
                    face_candidate = frame
                    if faces:
                        face_candidate_box = largest_face(faces)

            # ---- different type → debounce ----
            else:
                if ftype == pending_type:
                    pending_count += 1
                else:
                    pending_type = ftype
                    pending_count = 1
                    pending_start = t

                if pending_count >= SECTION_CHANGE_MIN_FRAMES:
                    log.info(f"[{pending_start:.1f}s] Section → {pending_type}")

                    # ---- commit outgoing section ----
                    if current_type == "slide" and slide_candidate is not None:
                        if pending_start - slide_start >= MIN_SLIDE_DURATION:
                            is_dup = False
                            if last_saved_slide is not None:
                                dedup = compute_slide_diff(
                                    last_saved_slide, slide_candidate,
                                    slide_mask)
                                if dedup < SLIDE_DEDUP_FRACTION:
                                    is_dup = True
                                    log.debug(
                                        f"  Skipping duplicate slide "
                                        f"(dedup diff={dedup:.4f})")
                            if not is_dup:
                                fn = _save_slide(slide_candidate, slide_index,
                                                 slide_start, pending_start)
                                rec = {"file": fn,
                                       "start": round(slide_start, 3),
                                       "end": round(pending_start, 3)}
                                slide_records.append(rec)
                                all_sections.append({"type": "slide", **rec})
                                slide_index += 1
                                last_saved_slide = slide_candidate
                        slide_candidate = None
                        slide_prev = None
                        slide_mask = None

                    elif current_type == "face" and face_candidate is not None:
                        sec_idx = len(all_sections)
                        if face_candidate_box is not None:
                            feat = extract_face_feature(face_candidate,
                                                        face_candidate_box)
                            face_data.append((sec_idx, feat))
                        all_sections.append({
                            "type": "face",
                            "file": None,   # assigned after clustering
                            "start": round(section_start, 3),
                            "end": round(pending_start, 3),
                        })
                        face_candidate = None
                        face_candidate_box = None

                    elif current_type == "black":
                        all_sections.append({
                            "type": "black",
                            "file": None,
                            "start": round(section_start, 3),
                            "end": round(pending_start, 3),
                        })

                    # ---- enter new section ----
                    current_type = pending_type
                    section_start = pending_start
                    pending_type = None
                    pending_count = 0

                    if current_type == "slide":
                        slide_candidate = frame
                        slide_start = section_start
                        slide_prev = frame
                        slide_mask = (build_face_mask(frame.shape, faces)
                                      if faces else None)
                        if _PYTESSERACT_AVAILABLE:
                            slide_text = extract_text(frame, slide_mask, ocr_lang)
                    elif current_type == "face":
                        face_candidate = frame
                        face_candidate_box = largest_face(faces)

        frame_idx += 1

    cap.release()

    # ---- finalize last open section ----
    t = duration
    if current_type == "slide" and slide_candidate is not None:
        if t - slide_start >= MIN_SLIDE_DURATION:
            is_dup = False
            if last_saved_slide is not None:
                dedup = compute_slide_diff(
                    last_saved_slide, slide_candidate, slide_mask)
                if dedup < SLIDE_DEDUP_FRACTION:
                    is_dup = True
                    log.debug(f"  Skipping duplicate slide "
                              f"(dedup diff={dedup:.4f})")
            if not is_dup:
                fn = _save_slide(slide_candidate, slide_index, slide_start, t)
                rec = {"file": fn, "start": round(slide_start, 3),
                       "end": round(t, 3)}
                slide_records.append(rec)
                all_sections.append({"type": "slide", **rec})

    elif current_type == "face" and face_candidate is not None:
        sec_idx = len(all_sections)
        if face_candidate_box is not None:
            feat = extract_face_feature(face_candidate, face_candidate_box)
            face_data.append((sec_idx, feat))
        all_sections.append({
            "type": "face",
            "file": None,
            "start": round(section_start, 3),
            "end": round(t, 3),
        })

    elif current_type == "black":
        all_sections.append({
            "type": "black",
            "file": None,
            "start": round(section_start, 3),
            "end": round(t, 3),
        })

    # ===============================================================
    # Post-processing: cluster speakers via PCA
    # ===============================================================

    speaker_labels = cluster_speakers(face_data)

    # Build mapping: speaker_id → representative image filename
    speaker_files = {}  # speaker_id → filename already saved

    for sec_idx, spk_id in speaker_labels.items():
        section = all_sections[sec_idx]
        section["speaker"] = spk_id

        if spk_id not in speaker_files:
            # Find the face_data entry for this section to get the frame
            # We need the original frame — re-read isn't practical, so we
            # saved face_candidate at commit time.  We'll use the first
            # occurrence's feature to pick a representative.
            # For the image, we'll save it during the second pass below.
            speaker_files[spk_id] = None  # placeholder

    # Second pass: save one representative image per speaker.
    # Re-open video briefly to grab frames at the right timestamps.
    if speaker_files:
        cap2 = cv2.VideoCapture(video_path)
        for sec_idx, spk_id in speaker_labels.items():
            if speaker_files.get(spk_id) is not None:
                continue  # Already have an image for this speaker
            section = all_sections[sec_idx]
            # Seek to the middle of the section
            mid_t = (section["start"] + section["end"]) / 2.0
            cap2.set(cv2.CAP_PROP_POS_MSEC, mid_t * 1000)
            ret, frame = cap2.read()
            if ret:
                spk_num = int(spk_id.split("_")[1])
                fn = _save_face(frame, spk_num, section["start"], section["end"])
                speaker_files[spk_id] = fn
        cap2.release()

    # Assign filenames to all face sections
    for sec_idx, spk_id in speaker_labels.items():
        all_sections[sec_idx]["file"] = speaker_files.get(spk_id)

    # Face sections without a cluster assignment
    for section in all_sections:
        if section.get("type") == "face" and "speaker" not in section:
            section["speaker"] = "unknown"

    # ===============================================================
    # Write output
    # ===============================================================

    with open(TIMESTAMPS_FILE, "w") as f:
        json.dump(slide_records, f, indent=2)
    with open(SECTIONS_FILE, "w") as f:
        json.dump(all_sections, f, indent=2)

    log.info(f"Slide timestamps: {TIMESTAMPS_FILE}")
    log.info(f"All sections:     {SECTIONS_FILE}")

    type_counts = {}
    for s in all_sections:
        type_counts[s["type"]] = type_counts.get(s["type"], 0) + 1
    for k, v in type_counts.items():
        log.info(f"  {k}: {v} section(s)")

    return slide_records, all_sections


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    extract_slides(video_path)
