# Log

## 2026-03-03

- Created the project: four scripts (`extract_slides.py`, `transcribe_audio.py`, `merge_transcript.py`, `run_all.py`)
- Initial slide extraction using SSIM pixel-diff — produced 56 slides, many false positives from webcam frames and black screens
- Added frame classification (slide/face/black) using Haar cascade face detection and brightness checks to fix false slide detection — reduced to 31 slides
- Added `diarize_speakers.py` for speaker identification via pyannote.audio (not integrated into the pipeline, still exists as a standalone script)
- Added `sections.json` output tracking slide/face/black regions over time
- Added speaker frame capture — face sections save a representative frame as `speaker_XX.png`
- Parallelized extraction, transcription, and diarization using `ThreadPoolExecutor`

## 2026-03-04

- Replaced Haar cascade with OpenCV DNN face detector (5.1MB caffemodel) for more accurate face detection
- Major rewrite of `extract_slides.py` (130 → 759 lines)
- Face area > 3% of frame now triggers face section classification
- Slide change detection now masks out face region before pixel-diff
- Added PCA (32d) + cosine clustering for speaker identification from face crops
- Captures last stable frame before each transition to guarantee final animation state
- All thresholds made configurable as module constants at top of file
- Slide count reduced from 31 to 16 — significantly fewer false positives
- Ran Whisper model size comparison experiment (tiny/base/small/medium) with results in `experiment/whisper_size_comparison/`
