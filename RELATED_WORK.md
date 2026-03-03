# Slide and Speaker Detection in Lecture Recordings: Related Work

## Problem Statement

Given a screen recording of a presentation (slides with optional speaker webcam overlay or full-camera sections), extract:
1. The final rendered version of each slide
2. Speaker sections with identity labels
3. Temporal alignment of both with a transcript

This is harder than it looks. Slides animate, speakers move in PiP overlays, transitions can be gradual, and the camera/slide ratio varies across recordings.

---

## Prior Approaches

### Pixel-Level Frame Differencing

**Lecture-Video-to-PDF** (Karton, GitHub) computes `cv2.absdiff` between consecutive frames, counts pixels exceeding a threshold, and tracks a 5-frame stability window. When a stable period breaks, the last stable frame is saved. Simple, zero dependencies beyond OpenCV, extremely fast. Weakness: no concept of frame regions — speaker movement in a PiP overlay triggers false transitions. Also no distinction between slide content and camera content; it treats the entire frame as one unit.

**slide-detector** (aweirdimagination) uses a similar pixel-diff approach that SliTraNet references. Compares consecutive frames, detects change points, applies minimum duration filters. Purely heuristic, no ML. Works well for clean screen recordings with no camera overlay.

### Structural Similarity (SSIM)

A common middle ground between raw pixel diff and learned features. SSIM compares luminance, contrast, and structure between two images, producing a score from 0 (completely different) to 1 (identical). Used in many lecture-to-PDF tools and our initial ClapScript implementation. Two thresholds partition the score: below 0.70 = new slide, 0.70–0.98 = animation in progress, above 0.98 = identical frame. More robust to compression artifacts and brightness changes than raw pixel diff, but ~10x slower than `absdiff` due to the windowed computation across the image. Still region-unaware — cannot distinguish PiP camera movement from slide transitions.

### Learned Transition Detection (SliTraNet)

**SliTraNet** (Sindel et al., 2021) uses a 3-stage deep learning pipeline for slide transition detection in lecture videos:

1. **Stage 1 — 2D CNN (ResNet-18/50):** Pairs of frames (anchor + current) are concatenated channel-wise and fed through a ResNet classifier. Binary output: same slide or transition. This identifies candidate transition points.
2. **Stage 2 — 3D CNN for slide-vs-video:** Around each candidate, a short video clip is extracted and classified by a 3D ResNet-50 into slide, video (camera), or transition. This filters out camera sections that Stage 1 misidentified as transitions.
3. **Stage 3 — 3D CNN for transition type:** Another 3D ResNet classifies whether a transition is a hard cut, gradual transition, or still part of a slide/video section.

Requires CUDA, three separately trained models, the `decord` library, and bounding-box ROI annotations per video. Designed for academic benchmarking on a curated dataset, not practical deployment. The 3-stage pipeline is rigorous but the computational cost and setup complexity are prohibitive for a lightweight tool.

### Semantic Embedding Approaches

**lecture-mind** (matte1782, GitHub) uses V-JEPA (Video Joint Embedding Predictive Architecture, Meta) — a ViT-L/16 vision transformer that produces 768-dimensional embeddings per frame. Cosine distance between consecutive embeddings detects "semantic boundaries" (topic/scene changes). A smoothing window of 5 embeddings and a minimum event gap prevent noise.

Conceptually elegant: instead of comparing pixels, compare meaning. A slide about "machine learning" and a slide about "data pipelines" would have high cosine distance even if they share the same template. However, the project's encoder is a placeholder (random linear projection); the actual V-JEPA model requires ~2GB of weights and GPU inference. The approach is research-grade, not production-ready. Also fundamentally unsuited to our problem: we need to detect visual transitions (same slide with one more bullet vs. completely new slide), not semantic ones.

### Face Detection for Camera/Slide Discrimination

Most lecture video tools ignore the camera entirely or assume slides fill the whole frame. When camera detection is attempted:

- **Haar cascades** (OpenCV `haarcascade_frontalface_default.xml`): The classical approach. Fast but unreliable — misses faces at angles, with glasses, partial occlusion, or non-frontal poses. High false-negative rate makes it unsuitable as the sole discriminator between camera and slide sections.

- **OpenCV DNN SSD** (`res10_300x300_ssd_iter_140000`): A small ResNet-based SSD face detector shipped with OpenCV. Single forward pass through a 5MB Caffe model. Dramatically more accurate than Haar at detecting faces in varied conditions. The standard choice for lightweight face detection that needs to actually work.

- **MediaPipe Face Detection** (Google): Uses a BlazeFace model optimized for mobile. Very fast, good accuracy. Adds a pip dependency (`mediapipe`) and pulls in TensorFlow Lite. Overkill for our frame classification task where we just need "is there a face and how big is it."

- **MTCNN, RetinaFace, SCRFD:** Progressively more accurate face detectors from the research community. All require PyTorch or ONNX runtime. Accuracy differences matter for face recognition but are negligible for our binary "face present / face absent" classification.

### Speaker Identification in Video

Most speaker identification work operates on audio (speaker diarization):

- **pyannote.audio** (Bredin et al.): State-of-the-art neural speaker diarization. The `speaker-diarization-3.1` pipeline segments audio into "who spoke when" using a combination of voice activity detection, speaker segmentation, and embedding clustering. Requires a HuggingFace token and ~1GB model download. Operates purely on audio — cannot identify speakers visually.

- **Visual speaker identification** is typically done via face recognition (ArcFace, FaceNet, DeepFace). These embed face crops into a high-dimensional space where distance correlates with identity. Effective but requires large models (100MB+) and often GPU inference.

For our use case — distinguishing 1-3 speakers in a presentation recording where each speaker appears in a distinct camera setup — full face recognition is overkill. The visual context (background, clothing, camera position) is as discriminative as the face itself.

---

## Our Approach (ClapScript)

We prioritize computational simplicity and reliability over accuracy on edge cases. The pipeline has no GPU requirement and no dependencies beyond OpenCV and numpy.

### Frame Classification

Every sampled frame (1 fps) goes through face detection using OpenCV's DNN SSD detector. The largest detected face's area as a fraction of the frame determines layout:

- Face area > 3% of frame → **face section** (speaker is the primary content)
- Face area < 3% or no face → **slide section**
- Mean brightness < 15 → **black screen**

The 3% threshold cleanly separates PiP webcam overlays (~0.4% face coverage) from actual webcam views (3–8% at normal distance). A debounce of 3 consecutive frames prevents flapping between types on ambiguous frames.

### Slide Detection

We replaced SSIM with `cv2.absdiff` pixel differencing for slide transition detection. SSIM was ~10x slower and provided no practical advantage for our use case.

Key innovation: **face-aware masking**. When faces are detected in a slide-dominant frame (PiP overlay), we build a binary mask that excludes the face region (padded 30% horizontal, 50% vertical to cover shoulders/hair). The pixel diff runs only on unmasked pixels. This prevents speaker movement in the PiP overlay from triggering false slide transitions — a problem that pure pixel-diff approaches like Lecture-Video-to-PDF cannot handle.

The diff fraction threshold is 10% — a real slide transition changes 30–80% of the slide area, while a single animation step (adding a bullet point) changes 1–3%. This single threshold handles both cases without the two-threshold (transition/animation) approach that SSIM required.

**Final version guarantee:** `slide_candidate` always holds the most recent frame. When a transition is detected (diff >= 10%), the candidate — which is the frame from the previous sample — is saved. This is the last frame before the transition, i.e., the final fully-rendered version of the outgoing slide. A minimum duration filter (2 seconds) prevents saving transient content from rapid changes or misclassified frames.

### Speaker Identification

We avoid audio-based diarization entirely. Instead, we collect visual features from face sections and cluster them offline:

1. **Feature extraction:** For each face section, the face bounding box from DNN detection is used to crop a 64x64 grayscale face region. This is flattened to a 4096-dimensional vector.

2. **Dimensionality reduction:** PCA (implemented as numpy SVD on standardized features) reduces the 4096-d vectors to 32 dimensions. This removes noise and makes distance computation meaningful.

3. **Clustering:** Greedy cosine-distance assignment: each face feature is compared to known speaker centroids. If cosine distance < 0.3 (correlation > 0.7), it's assigned to that speaker. Otherwise, a new speaker cluster is created. Centroids are updated incrementally.

This approach works because in a typical presentation recording, different speakers have different visual contexts (backgrounds, clothing, camera angles) that are captured even by a simple grayscale face crop. PCA extracts the dimensions that vary most across samples, which correspond to identity-discriminating features.

Compared to face recognition models (ArcFace, FaceNet), this is far less accurate for general face matching but sufficient for distinguishing 1–5 speakers within a single recording where visual context is consistent per speaker.

### Computational Cost

For a 60-minute 1080p recording sampled at 1 fps (3600 frames):

| Step | Per-frame cost | Total |
|---|---|---|
| DNN face detection | ~3ms | ~11s |
| Pixel diff (640x360) | ~0.2ms | ~0.7s |
| Face mask construction | ~0.1ms | ~0.4s |
| Face crop extraction | ~0.05ms | negligible |
| PCA + clustering (once) | — | < 1s |

Total slide/speaker extraction: ~15 seconds for a 60-minute video. The bottleneck is frame decoding from the video file, not our processing.
