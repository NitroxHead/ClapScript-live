"""
Binary-search for the optimal SPEAKER_COSINE_THRESHOLD.

Extracts face crops only from the Q&A / face section of the video,
runs PCA once, then binary-searches the cosine threshold to reach a
target speaker count.

Usage:
    python test/cosine_threshold_search/search.py <video> <expected_speakers> [face_start_sec]

    face_start_sec  optional: timestamp (seconds) where the face/Q&A section
                    begins. If omitted the script reads output/sections.json.

Output:
    test/cosine_threshold_search/results.json
"""

import json
import os
import sys

import cv2
import numpy as np

# Allow importing from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import extract_slides as es

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(HERE, "results.json")


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def find_face_start():
    """Read output/sections.json and return the start of the first face section."""
    path = os.path.join("output", "sections.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        sections = json.load(f)
    for s in sections:
        if s["type"] == "face":
            return s["start"]
    return None


def collect_samples(video_path, start_t, interval=es.FACE_SAMPLE_INTERVAL):
    """Extract face crops from the video from start_t onward."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    sample_step = max(1, int(fps))

    detector = es.load_face_detector()
    samples = []
    last_t = start_t - interval

    cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000)
    frame_idx = int(start_t * fps)

    print(f"Scanning {start_t:.0f}s → {duration:.0f}s  (every {interval:.0f}s) ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        if (frame_idx - int(start_t * fps)) % sample_step == 0:
            faces = es.detect_faces(frame, detector)
            if faces and t - last_t >= interval:
                box = es.largest_face(faces)
                if box is not None:
                    feat = es.extract_face_feature(frame, box)
                    samples.append((t, feat))
                    last_t = t
        frame_idx += 1

    cap.release()
    print(f"Collected {len(samples)} face samples")
    return samples


def pca_project(samples):
    features = np.array([s[1] for s in samples])
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-7
    normed = (features - mean) / std
    n_comp = min(es.PCA_COMPONENTS, normed.shape[0], normed.shape[1])
    _, _, Vt = np.linalg.svd(normed, full_matrices=False)
    return normed @ Vt[:n_comp].T


def count_speakers(projected, threshold):
    """Greedy cosine clustering — returns number of clusters."""
    centroids = []
    counts = []
    for feat in projected:
        best_c, best_d = -1, float("inf")
        for ci, cent in enumerate(centroids):
            d = es._cosine_dist(feat, cent)
            if d < best_d:
                best_d = d
                best_c = ci
        if best_c >= 0 and best_d < threshold:
            counts[best_c] += 1
            n = counts[best_c]
            centroids[best_c] = centroids[best_c] * ((n - 1) / n) + feat / n
        else:
            centroids.append(feat.copy())
            counts.append(1)
    return len(centroids)


# -----------------------------------------------------------------------
# Binary search
# -----------------------------------------------------------------------

def binary_search(projected, target, lo=0.0, hi=2.0, max_iters=25):
    log = []
    best = None

    for i in range(max_iters):
        mid = (lo + hi) / 2.0
        n = count_speakers(projected, mid)
        log.append({"threshold": round(mid, 5), "speakers": n})
        print(f"  [{i+1:2d}]  threshold={mid:.5f}  →  {n} speaker(s)")

        if n == target:
            best = mid
            print(f"\n  Exact match at threshold={mid:.5f}")
            break
        elif n > target:
            lo = mid   # too many clusters → raise threshold to merge more
        else:
            hi = mid   # too few clusters → lower threshold to split more

        if hi - lo < 1e-4:
            best = mid
            print(f"\n  Converged at threshold={mid:.5f}, got {n} (target={target})")
            break

    if best is None:
        best = (lo + hi) / 2.0
        n = count_speakers(projected, best)
        print(f"\n  Best found: threshold={best:.5f}, gives {n} speaker(s) (target={target})")

    return best, log


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python search.py <video> <expected_speakers> [face_start_sec]")
        sys.exit(1)

    video_path = sys.argv[1]
    target = int(sys.argv[2])
    start_t = float(sys.argv[3]) if len(sys.argv) > 3 else None

    if start_t is None:
        start_t = find_face_start()
        if start_t is not None:
            print(f"Face section starts at {start_t:.1f}s  (from output/sections.json)")
        else:
            print("No sections.json found — scanning from beginning")
            start_t = 0.0

    samples = collect_samples(video_path, start_t)
    if len(samples) < 2:
        print("Not enough face samples — aborting")
        sys.exit(1)

    print("\nRunning PCA ...")
    projected = pca_project(samples)

    print(f"\nBinary searching for threshold → {target} speaker(s) ...")
    best_threshold, search_log = binary_search(projected, target)

    result = {
        "video": video_path,
        "target_speakers": target,
        "face_start_sec": start_t,
        "samples_collected": len(samples),
        "best_threshold": round(best_threshold, 5),
        "search_log": search_log,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Suggested SPEAKER_COSINE_THRESHOLD = {best_threshold:.4f}")


if __name__ == "__main__":
    main()
