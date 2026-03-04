# ClapScript

Turn a presentation screen recording into a synced transcript with slide images.

Given a video of a presentation (slides + optional webcam), ClapScript extracts the slides, transcribes the audio, identifies speakers, and produces a timestamped markdown transcript linked to slide images.

## Output

- Extracted slide images (final version of each slide)
- Speaker face snapshots (one per speaker section)
- Timestamped transcript synced to slides in markdown

## How it works

1. **`extract_slides.py`** — Samples video at ~1fps. Uses OpenCV DNN face detection to classify frames as slide or face. Detects slide transitions via masked pixel-diff. Clusters face crops with PCA + cosine distance for speaker identification.
2. **`transcribe_audio.py`** — Transcribes audio using faster-whisper (medium model, CPU, int8).
3. **`merge_transcript.py`** — Maps transcript segments to the slide or speaker visible at that time. Outputs synced markdown.
4. **`run_all.py`** — Runs all steps in parallel, then merges.

## Usage

```
python3 run_all.py recording.mp4
```

Output goes to `output/`.

## Dependencies

- Python 3
- opencv-python
- faster-whisper
- numpy

The DNN face model files (~5MB) are downloaded automatically on first run.

## Optional

Speaker diarization via pyannote.audio is available as a standalone script (`diarize_speakers.py`) but not integrated into the main pipeline. Requires a HuggingFace token (`HF_TOKEN`).

## License

MIT
