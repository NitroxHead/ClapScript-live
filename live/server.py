"""ClapScript Live — FastAPI/WebSocket server.

Binary WebSocket protocol (Browser → Server):
  0x01 + JPEG bytes      → video frame (1fps)
  0x02 + PCM s16le bytes → audio chunk (16kHz mono)

Server → Browser (JSON text):
  {"type": "section_change", "section": "slide"|"face"|"black", "speaker": str|null}
  {"type": "slide_change",   "slide_index": N, "thumbnail": "<base64-jpeg>"}
  {"type": "speaker_update", "speaker": "speaker_01"}
  {"type": "transcript_partial", "text": "...", "speaker": str|null}
  {"type": "transcript_final",   "text": "...", "speaker": str|null, "t": float}
  {"type": "status", "message": "..."}
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
import urllib.request
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to import extract_slides functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from extract_slides import (
    _cosine_dist,
    build_face_mask,
    classify_layout,
    compute_slide_diff,
    detect_faces,
    extract_face_feature,
    largest_face,
    load_face_detector,
    MIN_SLIDE_DURATION,
    PCA_COMPONENTS,
    SECTION_CHANGE_MIN_FRAMES,
    SLIDE_CHANGE_FRACTION,
    SLIDE_DEDUP_FRACTION,
    SPEAKER_COSINE_THRESHOLD,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("live")

# ===================================================================
# Vosk model registry
# ===================================================================

VOSK_MODELS = {
    "en": ("vosk-model-small-en-us-0.15",
           "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"),
    "de": ("vosk-model-small-de-0.15",
           "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip"),
    "fr": ("vosk-model-small-fr-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip"),
    "es": ("vosk-model-small-es-0.42",
           "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"),
    "it": ("vosk-model-small-it-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip"),
    "pt": ("vosk-model-small-pt-0.3",
           "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip"),
    "ru": ("vosk-model-small-ru-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"),
    "nl": ("vosk-model-small-nl-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-small-nl-0.22.zip"),
    "pl": ("vosk-model-small-pl-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-small-pl-0.22.zip"),
    "uk": ("vosk-model-small-uk-v3-small",
           "https://alphacephei.com/vosk/models/vosk-model-small-uk-v3-small.zip"),
    "hi": ("vosk-model-small-hi-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip"),
    "ja": ("vosk-model-small-ja-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip"),
    "zh": ("vosk-model-small-cn-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"),
    "ko": ("vosk-model-small-ko-0.22",
           "https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip"),
    "tr": ("vosk-model-small-tr-0.3",
           "https://alphacephei.com/vosk/models/vosk-model-small-tr-0.3.zip"),
}

VOSK_MODELS_DIR = Path(__file__).parent / "vosk_models"

# Speaker clustering
SPEAKER_WARMUP = 10       # Samples collected before first speaker identification
SPEAKER_RECOMPUTE = 20    # Re-run PCA every N new samples after warmup
FACE_SAMPLE_INTERVAL = 2.0  # Minimum seconds between face samples

# Slide thumbnails
THUMBNAIL_WIDTH = 800

# Idle exit (socket activation mode only)
IDLE_EXIT_TIMEOUT = 300   # seconds of no connections before exiting

# ===================================================================
# Connection tracking (for idle-exit under socket activation)
# ===================================================================

_active_connections = 0
_last_disconnect = 0.0


async def _idle_exit_watcher():
    """Exit when no clients have connected for IDLE_EXIT_TIMEOUT seconds.

    Only used under systemd socket activation (LISTEN_FDS set). systemd keeps
    the TCP socket open, so the next connection will re-activate the service.
    """
    while True:
        await asyncio.sleep(60)
        if _active_connections == 0 and time.time() - _last_disconnect >= IDLE_EXIT_TIMEOUT:
            log.info("Idle timeout - exiting (systemd will restart on next connection)")
            sys.exit(0)


# ===================================================================
# Global face detector (loaded once at startup)
# ===================================================================

_face_detector = None


def get_face_detector():
    global _face_detector
    if _face_detector is None:
        log.info("Loading face detector...")
        _face_detector = load_face_detector()
        log.info("Face detector ready")
    return _face_detector


def ensure_vosk_model(lang: str) -> Path:
    """Return path to Vosk model dir, downloading if needed."""
    if lang not in VOSK_MODELS:
        log.warning(f"No Vosk model for '{lang}', falling back to 'en'")
        lang = "en"
    model_name, url = VOSK_MODELS[lang]
    model_dir = VOSK_MODELS_DIR / model_name
    if model_dir.exists():
        return model_dir
    VOSK_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = VOSK_MODELS_DIR / f"{model_name}.zip"
    log.info(f"Downloading {model_name} (~40MB)...")
    urllib.request.urlretrieve(url, zip_path)
    log.info("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(VOSK_MODELS_DIR)
    zip_path.unlink()
    log.info(f"Model ready: {model_dir}")
    return model_dir


# ===================================================================
# Session state
# ===================================================================

def new_session(lang: str = "en") -> dict:
    return {
        "lang": lang,
        "start_time": time.time(),
        "elapsed": 0.0,
        # Section state machine
        "current_type": None,     # "slide", "face", "black", or None
        "section_start": 0.0,
        "pending_type": None,
        "pending_count": 0,
        "pending_start": 0.0,
        # Slide tracking
        "slide_index": 0,
        "slide_candidate": None,  # Last stable frame (sent on transition)
        "slide_start": 0.0,
        "slide_prev": None,       # Previous frame for pixel-diff
        "slide_mask": None,       # Exclude face/PiP regions from diff
        "last_saved_slide": None,
        # Speaker clustering
        "face_samples": [],       # [(t, feature_4096d)]
        "pca_basis": None,        # Vt[:n_comp] after SVD
        "pca_mean": None,
        "pca_std": None,
        "speaker_centroids": [],  # PCA-projected centroids
        "speaker_names": [],      # "speaker_01", "speaker_02", ...
        "current_speaker": None,
        "last_face_t": -999.0,
        "samples_since_pca": 0,
        # Output
        "slides": [],             # [(slide_index, jpeg_bytes)] for session save
        "transcript_segments": [],
        "recognizer": None,
    }


# ===================================================================
# Incremental speaker tracking
# ===================================================================

def _rebuild_pca_and_centroids(state):
    """Recompute PCA from all face samples and re-cluster speakers."""
    samples = state["face_samples"]
    features = np.array([s[1] for s in samples])
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-7
    normed = (features - mean) / std

    n_comp = min(PCA_COMPONENTS, normed.shape[0] - 1, normed.shape[1])
    if n_comp < 1:
        return

    _, _, Vt = np.linalg.svd(normed, full_matrices=False)
    basis = Vt[:n_comp]
    state["pca_basis"] = basis
    state["pca_mean"] = mean
    state["pca_std"] = std

    # Re-cluster all projected samples from scratch
    projected = normed @ basis.T
    centroids = []
    cluster_ids = []
    for feat in projected:
        best_c, best_d = -1, float("inf")
        for ci, cent in enumerate(centroids):
            d = _cosine_dist(feat, cent)
            if d < best_d:
                best_d, best_c = d, ci
        if best_c >= 0 and best_d < SPEAKER_COSINE_THRESHOLD:
            n = cluster_ids.count(best_c) + 1
            centroids[best_c] = centroids[best_c] * ((n - 1) / n) + feat / n
            cluster_ids.append(best_c)
        else:
            cluster_ids.append(len(centroids))
            centroids.append(feat.copy())

    state["speaker_centroids"] = centroids
    state["speaker_names"] = [f"speaker_{i + 1:02d}" for i in range(len(centroids))]
    state["samples_since_pca"] = 0
    log.debug(f"PCA rebuilt: {len(samples)} samples → {len(centroids)} speaker(s)")


def identify_speaker(state, feature, t):
    """Add face sample if throttle allows, update PCA, return speaker label or None."""
    if t - state["last_face_t"] >= FACE_SAMPLE_INTERVAL:
        state["face_samples"].append((t, feature))
        state["last_face_t"] = t
        state["samples_since_pca"] += 1
        n = len(state["face_samples"])
        if n == SPEAKER_WARMUP or (
            n > SPEAKER_WARMUP
            and state["samples_since_pca"] >= SPEAKER_RECOMPUTE
        ):
            _rebuild_pca_and_centroids(state)

    if state["pca_basis"] is None:
        return None  # Still warming up

    normed = (feature - state["pca_mean"]) / state["pca_std"]
    projected = normed @ state["pca_basis"].T

    if not state["speaker_centroids"]:
        return None

    best_c, best_d = -1, float("inf")
    for ci, cent in enumerate(state["speaker_centroids"]):
        d = _cosine_dist(projected, cent)
        if d < best_d:
            best_d, best_c = d, ci

    if best_c >= 0 and best_d < SPEAKER_COSINE_THRESHOLD:
        return state["speaker_names"][best_c]

    # New speaker — add centroid (projection only, not a full sample)
    new_name = f"speaker_{len(state['speaker_centroids']) + 1:02d}"
    state["speaker_centroids"].append(projected.copy())
    state["speaker_names"].append(new_name)
    return new_name


# ===================================================================
# Frame processing (synchronous — run via asyncio.to_thread)
# ===================================================================

def _encode_thumbnail(frame) -> tuple[str, bytes]:
    """Return (base64_str, jpeg_bytes) for a frame scaled to THUMBNAIL_WIDTH."""
    h, w = frame.shape[:2]
    if w > THUMBNAIL_WIDTH:
        scale = THUMBNAIL_WIDTH / w
        frame = cv2.resize(frame, (THUMBNAIL_WIDTH, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpeg = buf.tobytes()
    return base64.b64encode(jpeg).decode(), jpeg


def process_frame_sync(state, jpeg_bytes) -> list:
    """Decode JPEG, run detection pipeline, return list of JSON strings to send."""
    detector = get_face_detector()
    msgs = []

    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return msgs

    state["elapsed"] = time.time() - state["start_time"]
    t = state["elapsed"]

    faces = detect_faces(frame, detector)
    ftype = classify_layout(frame, faces)
    current_type = state["current_type"]
    log.debug(f"t={t:.1f} layout={ftype} faces={len(faces)} current={current_type}")

    def _emit_slide(f, idx_override=None):
        thumb, jpeg = _encode_thumbnail(f)
        idx = idx_override if idx_override is not None else state["slide_index"]
        state["slides"].append((idx, jpeg))
        log.info(f"Slide {idx} captured (t={t:.1f}s, total={len(state['slides'])})")
        return json.dumps({"type": "slide_change", "slide_index": idx, "thumbnail": thumb})

    # --- Bootstrap: first frame ---
    if current_type is None:
        state["current_type"] = ftype
        state["section_start"] = t
        msgs.append(json.dumps({"type": "section_change", "section": ftype, "speaker": None}))
        if ftype == "slide":
            state["slide_index"] += 1
            state["slide_candidate"] = frame
            state["slide_start"] = t
            state["slide_prev"] = frame
            state["slide_mask"] = build_face_mask(frame.shape, faces) if faces else None
            msgs.append(_emit_slide(frame, state["slide_index"]))
        elif ftype == "face":
            face = largest_face(faces)
            if face is not None:
                feat = extract_face_feature(frame, face)
                spk = identify_speaker(state, feat, t)
                if spk:
                    state["current_speaker"] = spk
                    msgs.append(json.dumps({"type": "speaker_update", "speaker": spk}))
        return msgs

    # --- Same section type: stay ---
    if ftype == current_type:
        state["pending_type"] = None
        state["pending_count"] = 0

        if ftype == "slide":
            if faces:
                state["slide_mask"] = build_face_mask(frame.shape, faces)
                # PiP speaker tracking during slide sections
                face = largest_face(faces)
                if face is not None:
                    feat = extract_face_feature(frame, face)
                    spk = identify_speaker(state, feat, t)
                    if spk and spk != state["current_speaker"]:
                        state["current_speaker"] = spk
                        msgs.append(json.dumps({"type": "speaker_update", "speaker": spk}))

            if state["slide_prev"] is not None:
                diff = compute_slide_diff(state["slide_prev"], frame, state["slide_mask"])
                log.debug(f"  slide diff={diff:.3f} threshold={SLIDE_CHANGE_FRACTION} duration={t - state['slide_start']:.1f}s")
                if diff >= SLIDE_CHANGE_FRACTION:
                    dur = t - state["slide_start"]
                    if dur >= MIN_SLIDE_DURATION:
                        candidate = state["slide_candidate"]
                        is_dup = False
                        if state["last_saved_slide"] is not None:
                            dd = compute_slide_diff(
                                state["last_saved_slide"], candidate, state["slide_mask"]
                            )
                            is_dup = dd < SLIDE_DEDUP_FRACTION
                            if is_dup:
                                log.info(f"Slide skipped - dup (dd={dd:.3f})")
                        if not is_dup:
                            state["slide_index"] += 1
                            state["last_saved_slide"] = candidate
                            msgs.append(_emit_slide(candidate, state["slide_index"]))
                    else:
                        log.info(f"Slide skipped - too short ({dur:.1f}s < {MIN_SLIDE_DURATION}s), diff={diff:.3f}")
                    state["slide_start"] = t

            state["slide_candidate"] = frame
            state["slide_prev"] = frame

        elif ftype == "face":
            face = largest_face(faces)
            if face is not None:
                feat = extract_face_feature(frame, face)
                spk = identify_speaker(state, feat, t)
                if spk and spk != state["current_speaker"]:
                    state["current_speaker"] = spk
                    msgs.append(json.dumps({"type": "speaker_update", "speaker": spk}))

        return msgs

    # --- Different type: debounce ---
    if ftype == state["pending_type"]:
        state["pending_count"] += 1
    else:
        state["pending_type"] = ftype
        state["pending_count"] = 1
        state["pending_start"] = t

    if state["pending_count"] < SECTION_CHANGE_MIN_FRAMES:
        return msgs

    # Commit the section transition
    pending_type = state["pending_type"]
    pending_start = state["pending_start"]
    log.info(f"Section: {current_type} → {pending_type} at t={t:.1f}s")

    # Finalize outgoing slide: send last stable candidate
    if current_type == "slide" and state["slide_candidate"] is not None:
        dur = pending_start - state["slide_start"]
        if dur >= MIN_SLIDE_DURATION:
            candidate = state["slide_candidate"]
            is_dup = False
            if state["last_saved_slide"] is not None:
                dd = compute_slide_diff(
                    state["last_saved_slide"], candidate, state["slide_mask"]
                )
                is_dup = dd < SLIDE_DEDUP_FRACTION
                if is_dup:
                    log.info(f"Slide skipped on section exit - dup (dd={dd:.3f})")
            if not is_dup:
                state["slide_index"] += 1
                state["last_saved_slide"] = candidate
                msgs.append(_emit_slide(candidate, state["slide_index"]))
        else:
            log.info(f"Slide skipped on section exit - too short ({dur:.1f}s)")
        state["slide_candidate"] = None
        state["slide_prev"] = None
        state["slide_mask"] = None

    state["current_type"] = pending_type
    state["section_start"] = pending_start
    state["pending_type"] = None
    state["pending_count"] = 0

    msgs.append(json.dumps({
        "type": "section_change",
        "section": pending_type,
        "speaker": state["current_speaker"] if pending_type == "face" else None,
    }))

    if pending_type == "slide":
        state["slide_index"] += 1
        state["slide_candidate"] = frame
        state["slide_start"] = pending_start
        state["slide_prev"] = frame
        state["slide_mask"] = build_face_mask(frame.shape, faces) if faces else None
        msgs.append(_emit_slide(frame, state["slide_index"]))
    elif pending_type == "face":
        face = largest_face(faces)
        if face is not None:
            feat = extract_face_feature(frame, face)
            spk = identify_speaker(state, feat, t)
            if spk and spk != state["current_speaker"]:
                state["current_speaker"] = spk
                msgs.append(json.dumps({"type": "speaker_update", "speaker": spk}))

    return msgs


# ===================================================================
# Audio processing (fast, runs inline in the event loop)
# ===================================================================

def process_audio_sync(state, pcm_bytes):
    """Feed PCM to Vosk. Returns (is_final, text) or (None, None)."""
    rec = state["recognizer"]
    if rec is None:
        return None, None
    if rec.AcceptWaveform(pcm_bytes):
        result = json.loads(rec.Result())
        text = result.get("text", "").strip()
        return (True, text) if text else (None, None)
    else:
        partial = json.loads(rec.PartialResult())
        text = partial.get("partial", "").strip()
        return (False, text) if text else (None, None)


# ===================================================================
# Session save
# ===================================================================

def save_session(state):
    ts = int(state["start_time"])
    out_dir = Path("output") / f"live_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, jpeg in state["slides"]:
        path = out_dir / f"slide_{idx:03d}.jpg"
        path.write_bytes(jpeg)

    with open(out_dir / "transcript_segments.json", "w") as f:
        json.dump(state["transcript_segments"], f, indent=2)

    md = ["# ClapScript Live Transcript\n"]
    prev_spk = None
    for seg in state["transcript_segments"]:
        spk = seg.get("speaker") or "unknown"
        if spk != prev_spk:
            md.append(f"\n**[{spk}]**\n")
            prev_spk = spk
        md.append(seg["text"])

    with open(out_dir / "transcript.md", "w") as f:
        f.write("\n".join(md))

    log.info(f"Session saved → {out_dir} ({len(state['slides'])} slides, {len(state['transcript_segments'])} segments)")


# ===================================================================
# FastAPI app
# ===================================================================

@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(get_face_detector)
    if int(os.environ.get("LISTEN_FDS", "0")) >= 1:
        asyncio.create_task(_idle_exit_watcher())
    yield


app = FastAPI(title="ClapScript Live", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _active_connections, _last_disconnect
    await ws.accept()
    _active_connections += 1
    log.info("Client connected")

    # Config handshake: client sends {"lang": "en"} first
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        lang = json.loads(raw).get("lang", "en")
    except Exception:
        lang = "en"

    # Download/load Vosk model (may take a moment on first run)
    await ws.send_text(json.dumps({
        "type": "status",
        "message": f"Loading speech model for '{lang}'...",
    }))

    recognizer = None
    try:
        model_dir = await asyncio.to_thread(ensure_vosk_model, lang)
        from vosk import KaldiRecognizer, Model
        model = await asyncio.to_thread(Model, str(model_dir))
        recognizer = KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)
        log.info(f"Vosk ready (lang={lang})")
    except ImportError:
        await ws.send_text(json.dumps({
            "type": "status",
            "message": "vosk not installed — transcript disabled (pip install vosk)",
        }))
    except Exception as e:
        log.exception("Vosk init failed")
        await ws.send_text(json.dumps({
            "type": "status",
            "message": f"STT unavailable: {e}",
        }))

    state = new_session(lang)
    state["recognizer"] = recognizer
    state["audio_received"] = False  # log first audio packet
    stt_status = "STT ready" if recognizer else "STT disabled"
    await ws.send_text(json.dumps({"type": "status", "message": f"Ready ({stt_status})"}))
    await ws.send_text(json.dumps({"type": "audio_status", "ok": False}))

    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) < 2:
                continue
            msg_type = data[0]
            payload = bytes(data[1:])

            if msg_type == 0x01:
                # Video frame: run face detection in thread (CPU-heavy)
                msgs = await asyncio.to_thread(process_frame_sync, state, payload)
                for m in msgs:
                    await ws.send_text(m)

            elif msg_type == 0x02:
                # Audio chunk: Vosk is fast (<1ms), run inline
                if not state["audio_received"]:
                    state["audio_received"] = True
                    log.info(f"First audio packet: {len(payload)} bytes")
                    await ws.send_text(json.dumps({"type": "audio_status", "ok": True}))
                is_final, text = process_audio_sync(state, payload)
                if is_final and text:
                    seg = {
                        "t": round(state["elapsed"], 2),
                        "speaker": state["current_speaker"],
                        "text": text,
                    }
                    state["transcript_segments"].append(seg)
                    await ws.send_text(json.dumps({"type": "transcript_final", **seg}))
                elif is_final is False and text:
                    await ws.send_text(json.dumps({
                        "type": "transcript_partial",
                        "text": text,
                        "speaker": state["current_speaker"],
                    }))

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.exception(f"WebSocket error: {e}")
    finally:
        _active_connections -= 1
        _last_disconnect = time.time()
        save_session(state)


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    import uvicorn
    if int(os.environ.get("LISTEN_FDS", "0")) >= 1:
        # Launched by systemd socket activation - fd 3 is the pre-bound socket
        uvicorn.run(app, fd=3, log_level="info")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8013, log_level="info")
