"""
Whisper Model Size Comparison Experiment

Runs faster-whisper with tiny, base, small, and medium models on the same
video, then generates an HTML report comparing accuracy and performance.

Uses the largest model (medium) as the default reference transcript and diffs
smaller models against it to surface mistakes. The reference model can be
changed interactively in the report.

Usage:
    python experiment/whisper_size_comparison/compare.py ~/Downloads/videoplayback.mp4
"""

import difflib
import html
import json
import os
import re
import sys
import time

from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = ["tiny", "base", "small", "medium"]
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
BEAM_SIZE = 5
VIDEO_URL = "https://www.youtube.com/watch?v=VxBKjeRVcPk"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
REPORT_PATH = os.path.join(EXPERIMENT_DIR, "report.html")

# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_audio(video_path, audio_path):
    cmd = (
        f'ffmpeg -y -i "{video_path}" '
        f'-vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" '
        f'-loglevel error'
    )
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}")


# ---------------------------------------------------------------------------
# Run one model
# ---------------------------------------------------------------------------

def run_model(model_name, audio_path):
    print(f"\n{'='*60}")
    print(f"  Running model: {model_name}")
    print(f"{'='*60}")

    t0 = time.time()
    model = WhisperModel(model_name, device=DEVICE, compute_type=COMPUTE_TYPE)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    t0 = time.time()
    raw_segments, info = model.transcribe(audio_path, beam_size=BEAM_SIZE)

    segments = []
    for seg in raw_segments:
        segments.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "text": seg.text.strip(),
        })
    transcribe_time = time.time() - t0

    print(f"  Transcribed in {transcribe_time:.1f}s — {len(segments)} segments")

    result = {
        "model": model_name,
        "load_time_s": round(load_time, 2),
        "transcribe_time_s": round(transcribe_time, 2),
        "total_time_s": round(load_time + transcribe_time, 2),
        "num_segments": len(segments),
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "segments": segments,
    }

    out_path = os.path.join(RESULTS_DIR, f"{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def normalize_word(word):
    """Strip punctuation from a word for comparison purposes."""
    return re.sub(r'[^\w]', '', word.lower())


def normalize_text(text):
    """Lowercase, collapse whitespace, strip punctuation for WER."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def word_error_rate(ref_words, hyp_words):
    """Compute WER using dynamic programming (Levenshtein on words)."""
    r, h = ref_words, hyp_words
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(r)][len(h)] / max(len(r), 1)


def compute_windowed_wer(ref_segments, hyp_segments, window_sec=30):
    """Compute WER in time windows."""
    if not ref_segments:
        return []

    max_time = max(s["end"] for s in ref_segments + hyp_segments)
    windows = []
    t = 0.0
    while t < max_time:
        t_end = t + window_sec
        ref_text = " ".join(
            s["text"] for s in ref_segments if s["start"] >= t and s["start"] < t_end
        )
        hyp_text = " ".join(
            s["text"] for s in hyp_segments if s["start"] >= t and s["start"] < t_end
        )
        ref_words = normalize_text(ref_text).split()
        hyp_words = normalize_text(hyp_text).split()
        wer = word_error_rate(ref_words, hyp_words) if ref_words else 0.0
        windows.append({
            "start": round(t, 1),
            "end": round(t_end, 1),
            "wer": round(wer, 4),
        })
        t = t_end
    return windows


def build_segment_diff(ref_segments, hyp_segments):
    """Diff two transcripts word-by-word, normalizing punctuation for matching."""
    ref_text = " ".join(s["text"] for s in ref_segments)
    hyp_text = " ".join(s["text"] for s in hyp_segments)

    ref_words = ref_text.split()
    hyp_words = hyp_text.split()

    # Normalize for matching, keep originals for display
    ref_norm = [normalize_word(w) for w in ref_words]
    hyp_norm = [normalize_word(w) for w in hyp_words]

    sm = difflib.SequenceMatcher(None, ref_norm, hyp_norm)
    diff_ops = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        diff_ops.append({
            "op": op,
            "ref": " ".join(ref_words[i1:i2]),
            "hyp": " ".join(hyp_words[j1:j2]),
        })
    return diff_ops


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def count_diff_changes(diff_ops):
    return sum(1 for d in diff_ops if d["op"] != "equal")


def render_diff_html(diff_ops):
    parts = []
    for d in diff_ops:
        if d["op"] == "equal":
            words = d["ref"].split()
            if len(words) > 20:
                parts.append(html.escape(" ".join(words[:8])))
                parts.append(" <span class='ellipsis'>[...]</span> ")
                parts.append(html.escape(" ".join(words[-8:])))
            else:
                parts.append(html.escape(d["ref"]))
            parts.append(" ")
        elif d["op"] == "replace":
            parts.append(f"<del>{html.escape(d['ref'])}</del>")
            parts.append(f"<ins>{html.escape(d['hyp'])}</ins> ")
        elif d["op"] == "delete":
            parts.append(f"<del>{html.escape(d['ref'])}</del> ")
        elif d["op"] == "insert":
            parts.append(f"<ins>{html.escape(d['hyp'])}</ins> ")
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_report(all_results, video_path):
    models = list(all_results.keys())

    # Precompute all pairwise comparisons
    pairwise = {}
    for ref_name in models:
        for hyp_name in models:
            if ref_name == hyp_name:
                continue
            ref_segs = all_results[ref_name]["segments"]
            hyp_segs = all_results[hyp_name]["segments"]
            ref_text = normalize_text(" ".join(s["text"] for s in ref_segs))
            hyp_text = normalize_text(" ".join(s["text"] for s in hyp_segs))
            wer = word_error_rate(ref_text.split(), hyp_text.split())
            windowed = compute_windowed_wer(ref_segs, hyp_segs)
            diff = build_segment_diff(ref_segs, hyp_segs)
            key = f"{ref_name}_vs_{hyp_name}"
            pairwise[key] = {
                "wer": round(wer, 6),
                "windowed": windowed,
                "diff_html": render_diff_html(diff),
                "num_changes": count_diff_changes(diff),
            }

    # Build JSON data for JS
    js_results = {}
    for name, r in all_results.items():
        js_results[name] = {
            "model": r["model"],
            "load_time_s": r["load_time_s"],
            "transcribe_time_s": r["transcribe_time_s"],
            "total_time_s": r["total_time_s"],
            "num_segments": r["num_segments"],
        }

    # Pairwise data for JS (windowed + wer, diff_html separate)
    js_pairwise = {}
    for key, pw in pairwise.items():
        js_pairwise[key] = {
            "wer": pw["wer"],
            "windowed": pw["windowed"],
            "num_changes": pw["num_changes"],
        }

    # Build full transcript HTML per model
    transcript_html = {}
    for name, r in all_results.items():
        parts = []
        for seg in r["segments"]:
            ts = f"[{fmt_time(seg['start'])} &rarr; {fmt_time(seg['end'])}]"
            parts.append(f"<p><span class='ts'>{ts}</span> {html.escape(seg['text'])}</p>")
        transcript_html[name] = "\n".join(parts)

    # Diff HTML keyed by pair
    diff_html_map = {key: pw["diff_html"] for key, pw in pairwise.items()}

    h = []
    h.append("<!DOCTYPE html>")
    h.append("<html lang='en'><head><meta charset='utf-8'>")
    h.append("<title>Whisper Model Comparison</title>")
    h.append("<style>")
    h.append(CSS)
    h.append("</style></head><body>")
    h.append("<div class='container'>")

    # Header
    h.append("<h1>Whisper Model Size Comparison</h1>")
    h.append(f"<p class='subtitle'>Source: <a href='{VIDEO_URL}'>{VIDEO_URL}</a></p>")

    # Reference selector
    h.append("<div class='ref-selector'>")
    h.append("<label for='ref-select'>Reference model:</label>")
    h.append("<select id='ref-select'>")
    for m in models:
        selected = " selected" if m == "medium" else ""
        h.append(f"<option value='{m}'{selected}>{m}</option>")
    h.append("</select>")
    h.append("</div>")

    # Summary table
    h.append("<h2>Summary</h2>")
    h.append("<table class='summary'><thead><tr>")
    h.append("<th>Model</th><th>Load Time</th><th>Transcribe Time</th>"
             "<th>Total Time</th><th>Segments</th>"
             "<th>WER vs <span class='ref-label'>medium</span></th>"
             "<th>Speed vs <span class='ref-label'>medium</span></th>"
             "</tr></thead>")
    h.append("<tbody id='summary-body'></tbody></table>")

    # Recommendation
    h.append("<div class='recommendation'><h2>Quick Take</h2>")
    h.append("<ul id='quick-take'></ul></div>")

    # WER over time
    h.append("<h2>Word Error Rate Over Time</h2>")
    h.append("<p class='note'>Each bar = 30-second window. Taller = more errors vs reference.</p>")
    h.append("<div id='wer-charts'></div>")

    # Diffs
    h.append("<h2>Detailed Word Diffs vs <span class='ref-label'>medium</span></h2>")
    h.append("<p class='note'><span class='del-sample'>red</span> = reference only, "
             "<span class='ins-sample'>green</span> = this model only</p>")
    h.append("<div id='diff-sections'></div>")

    # Full transcripts (static, not reference-dependent)
    h.append("<h2>Full Transcripts</h2>")
    for name in models:
        r = all_results[name]
        h.append(f"<details><summary><strong>{name}</strong> "
                 f"({r['num_segments']} segments)</summary>")
        h.append(f"<div class='transcript'>{transcript_html[name]}</div></details>")

    # Embed data
    h.append("<script>")
    h.append(f"const MODELS = {json.dumps(models)};")
    h.append(f"const RESULTS = {json.dumps(js_results)};")
    h.append(f"const PAIRWISE = {json.dumps(js_pairwise)};")
    # Diff HTML can contain quotes etc, so base64-encode each
    import base64
    diff_b64 = {}
    for key, dhtml in diff_html_map.items():
        diff_b64[key] = base64.b64encode(dhtml.encode()).decode()
    h.append(f"const DIFF_HTML_B64 = {json.dumps(diff_b64)};")
    h.append(REPORT_JS)
    h.append("</script>")

    h.append("</div></body></html>")

    report = "\n".join(h)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport written to: {REPORT_PATH}")


REPORT_JS = """
function fmtTime(sec) {
    sec = Math.floor(sec);
    const m = Math.floor(sec / 60), s = sec % 60;
    const h = Math.floor(m / 60), mm = m % 60;
    if (h) return h + ':' + String(mm).padStart(2,'0') + ':' + String(s).padStart(2,'0');
    return mm + ':' + String(s).padStart(2,'0');
}

function renderSummary(ref) {
    const refData = RESULTS[ref];
    let html = '';
    for (const m of MODELS) {
        const r = RESULTS[m];
        let werCell, werClass;
        if (m === ref) {
            werCell = '<span class="ref">reference</span>';
            werClass = '';
        } else {
            const key = ref + '_vs_' + m;
            const pw = PAIRWISE[key];
            const wer = pw ? pw.wer : 0;
            werCell = (wer * 100).toFixed(1) + '%';
            werClass = wer > 0.15 ? 'bad' : wer > 0.05 ? 'ok' : 'good';
        }
        const speedup = refData.total_time_s > 0
            ? (refData.total_time_s / r.total_time_s).toFixed(1) + 'x' : '—';
        html += '<tr>'
            + '<td><strong>' + m + '</strong></td>'
            + '<td>' + r.load_time_s.toFixed(1) + 's</td>'
            + '<td>' + r.transcribe_time_s.toFixed(1) + 's</td>'
            + '<td>' + r.total_time_s.toFixed(1) + 's</td>'
            + '<td>' + r.num_segments + '</td>'
            + '<td class="' + werClass + '">' + werCell + '</td>'
            + '<td>' + speedup + '</td></tr>';
    }
    document.getElementById('summary-body').innerHTML = html;
}

function renderQuickTake(ref) {
    const refData = RESULTS[ref];
    let html = '';
    for (const m of MODELS) {
        if (m === ref) continue;
        const key = ref + '_vs_' + m;
        const pw = PAIRWISE[key];
        if (!pw) continue;
        const wer = pw.wer * 100;
        const speed = refData.total_time_s > 0
            ? refData.total_time_s / RESULTS[m].total_time_s : 0;
        let verdict;
        if (wer < 3) verdict = 'Nearly identical (' + wer.toFixed(1) + '% WER) at ' + speed.toFixed(1) + 'x the speed. Strong candidate.';
        else if (wer < 8) verdict = 'Minor differences (' + wer.toFixed(1) + '% WER), ' + speed.toFixed(1) + 'x faster. Good tradeoff if speed matters.';
        else if (wer < 15) verdict = 'Noticeable mistakes (' + wer.toFixed(1) + '% WER), but ' + speed.toFixed(1) + 'x faster. Usable for drafts.';
        else verdict = 'Significant errors (' + wer.toFixed(1) + '% WER). ' + speed.toFixed(1) + 'x faster but quality suffers.';
        html += '<li><strong>' + m + '</strong>: ' + verdict + '</li>';
    }
    document.getElementById('quick-take').innerHTML = html;
}

function renderWerCharts(ref) {
    let html = '';
    for (const m of MODELS) {
        if (m === ref) continue;
        const key = ref + '_vs_' + m;
        const pw = PAIRWISE[key];
        if (!pw || !pw.windowed || pw.windowed.length === 0) continue;

        html += '<h3>' + m + '</h3>';
        const W = 800, H = 150;
        const barW = Math.max(2, Math.floor((W - 60) / pw.windowed.length));
        let maxWer = Math.max(...pw.windowed.map(w => w.wer), 0.05);

        let svg = "<svg width='" + W + "' height='" + (H+40) + "' class='chart'>";
        for (const frac of [0, 0.25, 0.5, 0.75, 1.0]) {
            const y = H - frac * H;
            const label = (frac * maxWer * 100).toFixed(0) + '%';
            svg += "<text x='45' y='" + (y+4) + "' class='axis'>" + label + "</text>";
            svg += "<line x1='50' y1='" + y + "' x2='" + W + "' y2='" + y + "' class='grid'/>";
        }
        pw.windowed.forEach((w, i) => {
            const x = 55 + i * barW;
            const barH = maxWer > 0 ? (w.wer / maxWer) * H : 0;
            const y = H - barH;
            const color = w.wer < 0.05 ? '#22c55e' : w.wer < 0.15 ? '#f59e0b' : '#ef4444';
            svg += "<rect x='" + x + "' y='" + y + "' width='" + Math.max(1, barW-1)
                + "' height='" + barH + "' fill='" + color + "' opacity='0.8'>"
                + "<title>" + fmtTime(w.start) + "–" + fmtTime(w.end) + ": "
                + (w.wer*100).toFixed(1) + "% WER</title></rect>";
        });
        const step = Math.max(1, Math.floor(pw.windowed.length / 10));
        for (let i = 0; i < pw.windowed.length; i += step) {
            const x = 55 + i * barW;
            svg += "<text x='" + x + "' y='" + (H+18) + "' class='axis'>"
                + fmtTime(pw.windowed[i].start) + "</text>";
        }
        svg += "</svg>";
        html += svg;
    }
    document.getElementById('wer-charts').innerHTML = html;
}

function renderDiffs(ref) {
    let html = '';
    for (const m of MODELS) {
        if (m === ref) continue;
        const key = ref + '_vs_' + m;
        const pw = PAIRWISE[key];
        const b64 = DIFF_HTML_B64[key];
        if (!pw || !b64) continue;
        const diffHtml = atob(b64);
        html += '<details><summary><strong>' + m + '</strong> — '
            + (pw.wer * 100).toFixed(1) + '% WER, '
            + pw.num_changes + ' differences</summary>'
            + '<div class="diff">' + diffHtml + '</div></details>';
    }
    document.getElementById('diff-sections').innerHTML = html;
}

function updateRefLabels(ref) {
    document.querySelectorAll('.ref-label').forEach(el => el.textContent = ref);
}

function refresh() {
    const ref = document.getElementById('ref-select').value;
    updateRefLabels(ref);
    renderSummary(ref);
    renderQuickTake(ref);
    renderWerCharts(ref);
    renderDiffs(ref);
}

document.getElementById('ref-select').addEventListener('change', refresh);
refresh();
"""


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0;
    line-height: 1.6; padding: 2rem;
}
.container { max-width: 1000px; margin: 0 auto; }
h1 { font-size: 1.8rem; margin-bottom: 0.3rem; color: #f8fafc; }
h2 { font-size: 1.3rem; margin: 2rem 0 0.8rem; color: #94a3b8;
     border-bottom: 1px solid #334155; padding-bottom: 0.4rem; }
h3 { font-size: 1.1rem; margin: 1rem 0 0.5rem; color: #cbd5e1; }
.subtitle { color: #64748b; margin-bottom: 1.5rem; }
.subtitle a { color: #60a5fa; text-decoration: none; }
.subtitle a:hover { text-decoration: underline; }
.note { color: #64748b; font-size: 0.85rem; margin-bottom: 0.8rem; }
code { background: #1e293b; padding: 0.15em 0.4em; border-radius: 4px; font-size: 0.9em; }

.ref-selector {
    background: #1e293b; display: inline-flex; align-items: center; gap: 0.6rem;
    padding: 0.5rem 1rem; border-radius: 6px; margin: 1rem 0;
}
.ref-selector label { color: #94a3b8; font-size: 0.9rem; }
.ref-selector select {
    background: #334155; color: #e2e8f0; border: 1px solid #475569;
    padding: 0.3rem 0.6rem; border-radius: 4px; font-size: 0.9rem;
    cursor: pointer;
}

table.summary {
    width: 100%; border-collapse: collapse; margin-bottom: 1.5rem;
    font-size: 0.95rem;
}
table.summary th {
    text-align: left; padding: 0.6rem 0.8rem;
    background: #1e293b; color: #94a3b8; font-weight: 600;
    border-bottom: 2px solid #334155;
}
table.summary td {
    padding: 0.6rem 0.8rem; border-bottom: 1px solid #1e293b;
}
table.summary tr:hover { background: #1e293b; }
.good { color: #22c55e; font-weight: 600; }
.ok { color: #f59e0b; font-weight: 600; }
.bad { color: #ef4444; font-weight: 600; }
.ref { color: #64748b; font-style: italic; }

.recommendation {
    background: #1e293b; border-radius: 8px; padding: 1.2rem 1.5rem;
    margin: 1rem 0 1.5rem; border-left: 4px solid #3b82f6;
}
.recommendation ul { margin-left: 1.2rem; }
.recommendation li { margin: 0.5rem 0; }

details { margin: 0.5rem 0; }
summary {
    cursor: pointer; padding: 0.5rem 0; color: #93c5fd;
    font-size: 0.95rem;
}
summary:hover { color: #60a5fa; }

.diff {
    background: #1e293b; padding: 1rem; border-radius: 6px;
    font-size: 0.85rem; line-height: 1.8; word-wrap: break-word;
    max-height: 500px; overflow-y: auto;
}
del {
    background: #450a0a; color: #fca5a5; text-decoration: line-through;
    padding: 0.1em 0.2em; border-radius: 2px;
}
ins {
    background: #052e16; color: #86efac; text-decoration: none;
    padding: 0.1em 0.2em; border-radius: 2px;
}
.del-sample { background: #450a0a; color: #fca5a5; padding: 0.1em 0.4em; border-radius: 2px; }
.ins-sample { background: #052e16; color: #86efac; padding: 0.1em 0.4em; border-radius: 2px; }
.ellipsis { color: #475569; font-style: italic; }

.transcript {
    background: #1e293b; padding: 1rem; border-radius: 6px;
    max-height: 400px; overflow-y: auto; font-size: 0.85rem;
}
.transcript p { margin: 0.3rem 0; }
.ts { color: #64748b; font-family: monospace; font-size: 0.8em; margin-right: 0.5em; }

svg.chart { display: block; margin: 0.5rem 0 1rem; }
.axis { fill: #64748b; font-size: 11px; font-family: monospace; }
.grid { stroke: #334155; stroke-width: 0.5; }
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python experiment/whisper_size_comparison/compare.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        print(f"Error: file not found: {video_path}")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Extract audio once
    audio_path = os.path.join(RESULTS_DIR, "audio.wav")
    if not os.path.isfile(audio_path):
        print("Extracting audio...")
        extract_audio(video_path, audio_path)
        print("Audio extracted.")
    else:
        print("Using cached audio.")

    # Run each model
    all_results = {}
    for model_name in MODELS:
        cache_path = os.path.join(RESULTS_DIR, f"{model_name}.json")
        if os.path.isfile(cache_path):
            print(f"\nUsing cached results for {model_name}")
            with open(cache_path) as f:
                all_results[model_name] = json.load(f)
        else:
            all_results[model_name] = run_model(model_name, audio_path)

    # Generate comparison report
    print("\nGenerating report...")
    generate_report(all_results, video_path)
    print("Done!")


if __name__ == "__main__":
    main()
