# Auto Excitement

Upload a short video in the browser and watch Meta's `facebook/tribev2`
model predict the BOLD response on the fsaverage5 cortical surface
(20,484 vertices) — collapsed into 7 Yeo networks plus a 4-axis brain
state and a PCA latent. The page renders the predicted brain on the
client with WebGL, in sync with the video, and exports an `ffmpeg`
script that cuts the boring stretches.

## What it does

- Video upload (`mp4` / `webm` / `mov` / …) with a real progress bar
  (live byte count + transfer rate)
- WhisperX → word-level transcript. **Japanese** works either as-is or
  translated-to-English for inference (recommended)
- TRIBEv2 inference: `(N_TR, 20484)` BOLD predictions
- Visualisation:
  - Video player with a vertical cursor synced across all panels
  - Chart.js: 7 Yeo networks + 4 brain-state axes (Excitement / Valence
    / Cognitive Load / Novelty)
  - **Live WebGL brain** (Three.js) on the half-inflated fsaverage5
    mesh: drag to rotate, wheel to zoom, vertex colors update at 60 fps
    against the video time
  - Filmstrip of per-segment thumbnails (click to seek)
  - Stacked timeline: audio waveform + word transcript + 4 axes + PCA
    top-3
  - Per-axis live readouts
- **Boring-segment cutter** driven by the Excitement axis:
  - Threshold and minimum-keep-duration sliders, dropped ranges greyed
    out across all timeline lanes
  - In-browser preview that skips the dropped intervals on playback
  - Generates a one-shot `ffmpeg` `select`/`aselect` filter command +
    downloadable `.sh`

## Layout

| Path | Role |
|---|---|
| `server.py` | FastAPI server — keeps TribeModel and the Yeo7 atlas resident, streams progress over SSE |
| `build_atlas.py` | Project the Yeo 2011 7-network MNI152 volume onto fsaverage5 |
| `smoke_predict.py` | Stand-alone model smoke test |
| `static/index.html` | Single-page GUI (Chart.js + Three.js from CDN) |
| `patches/` | Required patches against upstream `facebookresearch/tribev2` and `neuralset` (see below) |
| `cache/` *(.gitignore)* | Model intermediates + atlas |
| `static/videos/` *(.gitignore)* | Uploaded videos served back to the player |
| `static/thumbs/` *(.gitignore)* | Per-segment thumbnail jpegs |
| `static/mesh/` *(.gitignore)* | Generated fsaverage5 mesh blob (rebuilt at startup) |
| `static/preds/` *(.gitignore)* | Per-job float16 prediction blobs |
| `tribev2-src/` *(.gitignore)* | Upstream tribev2 clone — clone separately |

## Setup

```bash
git clone https://github.com/shi3z/auto-excitement.git
cd auto-excitement

# 1) Clone the upstream tribev2 sources alongside this repo
git clone https://github.com/facebookresearch/tribev2.git tribev2-src

# 2) venv + dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ./tribev2-src

# 3) Apply the required patches (Japanese support + --task translate route)
patches/apply.sh

# 4) Project the Yeo7 atlas to fsaverage5
python build_atlas.py

# 5) Japanese spaCy model (only needed for `japanese` / `japanese_translate` modes)
python -m spacy download ja_core_news_lg
```

`uvx whisperx` must be on `PATH` (used for word-level timestamps).
A CUDA GPU is strongly recommended — the V-JEPA2 video encoder is the
single hottest path even with the fp16 autocast we apply at startup.

### Upstream patches we apply

`patches/apply.sh` is idempotent; re-running it is a no-op.

- `patches/tribev2.patch` — adds `task` (`transcribe`/`translate`) to
  `ExtractWordsFromAudio`, registers `japanese="ja"` in the WhisperX
  language map, fixes the empty-string `--align_model` bug for non-English
  runs, threads a `language` kwarg through `get_audio_and_text_events` /
  `get_events_dataframe`, and adds the `japanese_translate` pseudo-language
- `patches/neuralset.patch` — adds `japanese="ja_core_news_lg"` and the ISO
  alias `"ja"→"japanese"` to neuralset's spaCy language map

## Run

```bash
source venv/bin/activate
python server.py            # http://localhost:8000
```

Pick an `mp4` (or webm/mov/…) and hit **予測する**:

1. Upload (with byte-level progress)
2. Server builds the events DataFrame via
   `model.get_events_dataframe(video_path=..., language=...)`
3. `model.predict(events=df)` produces `(N, 20484)` predictions
4. Server reduces to Yeo7 means, the 4 axes, and the top-3 PCA components,
   extracts thumbnails + audio waveform, and writes a packed float16 blob
   of the normalised predictions
5. Result is streamed to the browser via SSE; the page lights up with
   chart, filmstrip, timeline, the WebGL brain, and the cut tool

Performance reference (52-second Sintel clip, single CUDA GPU):

| Stage | Before | After (fp16 + WebGL) |
|---|---|---|
| V-JEPA2 encoding | 137 s (1.32 s/it) | 46 s (2.23 it/s) |
| Server-side brain PNG render | 70 s | 0 s (moved to client) |
| End-to-end | 270 s (5.2× slower than realtime) | 78 s (1.5× slower) |

## The 4 brain-state axes (heuristic proxies + PCA latent)

```
Excitement     = (z(VIS+SMN+DAN+VAN) − z(DMN)) / 2     # sensory + attentional engagement
Valence        = z(Limbic)                              # affective tone proxy
Cognitive Load = z(mean(FrontoParietal, DorsalAttn))    # control / working memory demand
Novelty        = z(VAN − mean(all 7 Yeo networks))      # salience above baseline (≈ prediction error)
PC1 – PC3      = top three principal components of the centred (T, V) prediction matrix (z-scored)
```

> **Caveat.** The axis names are post-hoc interpretations — don't read
> "Cognitive Load" as a literal cognitive load score, or "Novelty" as a
> subjective-surprise meter. The point is to leave the 1-D
> "excitement ↔ boredom" framing behind: independent named axes plus a
> data-driven latent space give a much richer view of the predicted
> brain state.

## API

`POST /predict` (multipart):

- `video` (file)
- `language` (str, default `english`): one of `english` / `japanese` /
  `japanese_translate` / `french` / `spanish` / `dutch` / `chinese`

Response: `{job_id, video_url, language}`.

`GET /events/{job_id}` — Server-Sent Events stream:

- `data: {"type":"progress","phase":"...","percent":N}`
- `data: {"type":"log","message":"..."}`
- `data: {"type":"done"}` followed by `event: result\ndata: {...}`
- on failure: `data: {"type":"error","message":"..."}` plus `event: error`

`result` payload:

```jsonc
{
  "n_segments": 53, "tr": 1.0,
  "times": [...], "durations": [...],
  "networks": { "Visual": [...], "Somatomotor": [...], ... },
  "axes": {
    "excitement": [...], "valence": [...],
    "cognitive_load": [...], "novelty": [...]
  },
  "pca": [[...], [...], [...]],   // top 3 PCs, length n_segments
  "pca_var": [0.73, 0.13, 0.08],  // explained variance ratios
  "thumbs": [{"t": 0.0, "url": "/static/thumbs/<job>/0000.jpg"}, ...],
  "waveform": [0.0, 0.12, ...],   // 1500 normalised envelope bins
  "words": [{"t": 12.21, "d": 0.12, "text": "What"}, ...],
  "mesh_url": "/static/mesh/fsaverage5.bin",
  "preds_url": "/static/preds/<job>.bin",
  "preds_dtype": "float16",
  "preds_shape": [53, 20484],
  "preds_normalized": true,
  "elapsed_sec": 78.0
}
```

`mesh_url` resolves to a custom packed binary (vertices / faces / sulcal
map for both hemispheres) parsed by the WebGL viewer. `preds_url` is a
flat `float16[T*V]` array, robust-normalised to `[0, 1]` so the client
only has to threshold + look up the fire colormap.

## Known limits

- The number of kept segments is `video_duration ÷ TR(=1s) × kept_ratio`.
  `predict()` internally drops a few; `times[i]` is the real-time anchor
  and accounts for the gaps when plotting.
- The axis scores are z-scored, so absolute comparisons across clips are
  meaningless.
- The Yeo atlas is projected to fsaverage5 with
  `vol_to_surf(interpolation="nearest_most_frequent")`, so boundaries are
  approximate. For rigorous analysis, prefer FreeSurfer's
  `?h.Yeo2011_7Networks_N1000.annot` directly.
- `japanese_translate` mode is a single WhisperX pass with
  `--task translate`. Timestamps are segment-level and word offsets are
  evenly distributed within each segment. The original Japanese
  transcript is not surfaced in the UI.
- TRIBEv2 was trained on English naturalistic stimuli (movies / audio
  books). Feeding non-English audio in plain `japanese` mode (no
  translation) keeps visual / auditory networks responsive, but
  language-network predictions degrade noticeably.

## Troubleshooting

- `requests.exceptions.RequestException: timed out` while building the
  atlas — `build_atlas.py` already falls back to a direct `urllib`
  download; just re-run it.
- Disk full → blow away `~/.cache/uv` (regenerated) first; be careful
  with the rest of the HuggingFace cache.
- Port 8000 in use → edit `server.py` or run
  `uvicorn server:app --port 9000`.
- WhisperX dies with `argument --align_model: expected one argument` →
  `patches/apply.sh` was not applied. The empty-string fix lives in that
  patch.

## License

The code in this repository is MIT. The contents of `tribev2-src/` and the
patches in `patches/` follow the upstream MIT license of
`facebookresearch/tribev2`. The Yeo 2011 atlas requires citation of
[Yeo et al. 2011, J. Neurophysiol.](https://doi.org/10.1152/jn.00338.2011).
