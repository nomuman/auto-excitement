"""TRIBEv2 viewer — FastAPI server with SSE progress.

Loads TribeModel once at startup. POST /predict accepts an uploaded video,
runs prediction in a worker thread, streams progress via SSE on
/events/{job_id}, and finally pushes the reduced 7-network time series + scores.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"
STATIC = ROOT / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tribev2-viewer")

state: dict[str, Any] = {}
jobs: dict[str, "JobState"] = {}


class JobState:
    def __init__(self) -> None:
        self.q: queue.Queue = queue.Queue()
        self.done = False
        self.error: str | None = None
        self.result: dict | None = None


class StderrTee:
    """Mirror stderr to the original fd while parsing tqdm bars and INFO lines into a queue."""
    BAR = re.compile(r"^\s*([^:|]+?):\s+(\d+)%\|")
    INFO = re.compile(r"(INFO|WARNING|ERROR)[\s\-:]+(.+)$")

    def __init__(self, q: queue.Queue, mirror) -> None:
        self.q = q
        self.mirror = mirror
        self.buf = ""
        self._last = (None, None)

    def write(self, s: str) -> int:
        try:
            self.mirror.write(s)
        except Exception:
            pass
        self.buf += s
        parts = re.split(r"[\r\n]", self.buf)
        self.buf = parts[-1]
        for line in parts[:-1]:
            line = line.strip()
            if not line:
                continue
            m = self.BAR.match(line)
            if m:
                phase = m.group(1).strip()[:60]
                pct = int(m.group(2))
                key = (phase, pct)
                if key != self._last:
                    self._last = key
                    self.q.put({"type": "progress", "phase": phase, "percent": pct})
                continue
            m = self.INFO.search(line)
            if m and ("Loading" in line or "Predict" in line or "Preparing" in line):
                self.q.put({"type": "log", "message": m.group(2).strip()[:200]})
        return len(s)

    def flush(self) -> None:
        try:
            self.mirror.flush()
        except Exception:
            pass


def _zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return np.zeros_like(x) if s < 1e-9 else (x - x.mean()) / s


def _normalize_video(video_path: Path) -> Path:
    """Re-encode HEVC/Dolby Vision/iPhone MOV to H.264+AAC MP4 so moviepy can read it."""
    out = video_path.with_suffix(".norm.mp4")
    cmd = [
        state["ffmpeg"], "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-vf", "format=yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    if out.exists() and out.stat().st_size > 0:
        return out
    return video_path


def _patch_video_extractor_autocast() -> None:
    """Wrap neuralset's HuggingFace video extractor in CUDA fp16 autocast for ~1.5x speedup
    on V-JEPA2-ViT-g without changing model weights. Outputs auto-cast back to fp32."""
    import torch as _torch
    from neuralset.extractors import video as _ns_video

    if getattr(_ns_video._HFVideoModel, "_viewer_autocast_patched", False):
        return
    _orig_predict = _ns_video._HFVideoModel.predict

    def _autocast_predict(self, images, audio=None):
        if _torch.cuda.is_available():
            with _torch.autocast(device_type="cuda", dtype=_torch.float16):
                return _orig_predict(self, images, audio)
        return _orig_predict(self, images, audio)

    _ns_video._HFVideoModel.predict = _autocast_predict
    _ns_video._HFVideoModel._viewer_autocast_patched = True
    log.info("Video extractor predict() wrapped with CUDA fp16 autocast")


def _dump_fsaverage5_mesh() -> str:
    """Dump a half-inflated fsaverage5 mesh + sulcal background as a packed binary
    that the browser fetches once and renders client-side via Three.js.
    Layout (little-endian):
        u32 n_verts_L, u32 n_faces_L, u32 n_verts_R, u32 n_faces_R
        f32[n_verts_L,3] verts_L, u32[n_faces_L,3] faces_L, f32[n_verts_L] sulc_L
        f32[n_verts_R,3] verts_R, u32[n_faces_R,3] faces_R, f32[n_verts_R] sulc_R
    """
    out = STATIC / "mesh" / "fsaverage5.bin"
    if out.exists():
        return f"/static/mesh/{out.name}"
    out.parent.mkdir(parents=True, exist_ok=True)

    import nibabel
    from nilearn import datasets
    fsavg = datasets.fetch_surf_fsaverage("fsaverage5")

    def load_gii(p):
        g = nibabel.load(p)
        return g.darrays[0].data.astype(np.float32), g.darrays[1].data.astype(np.uint32)

    def load_scalar(p):
        return nibabel.load(p).darrays[0].data.astype(np.float32)

    pial_l_v, _ = load_gii(fsavg.pial_left)
    infl_l_v, infl_l_f = load_gii(fsavg.infl_left)
    pial_r_v, _ = load_gii(fsavg.pial_right)
    infl_r_v, infl_r_f = load_gii(fsavg.infl_right)
    half_l = ((pial_l_v + infl_l_v) * 0.5).astype(np.float32)
    half_r = ((pial_r_v + infl_r_v) * 0.5).astype(np.float32)
    # offset hemispheres horizontally so they don't overlap when rendered together
    span_l = float(half_l[:, 0].max() - half_l[:, 0].min())
    span_r = float(half_r[:, 0].max() - half_r[:, 0].min())
    half_l = half_l.copy(); half_l[:, 0] -= half_l[:, 0].max() + 6
    half_r = half_r.copy(); half_r[:, 0] -= half_r[:, 0].min() - 6
    sulc_l = load_scalar(fsavg.sulc_left)
    sulc_r = load_scalar(fsavg.sulc_right)

    import struct
    with out.open("wb") as f:
        f.write(struct.pack("<IIII",
                            half_l.shape[0], infl_l_f.shape[0],
                            half_r.shape[0], infl_r_f.shape[0]))
        f.write(half_l.tobytes()); f.write(infl_l_f.tobytes()); f.write(sulc_l.tobytes())
        f.write(half_r.tobytes()); f.write(infl_r_f.tobytes()); f.write(sulc_r.tobytes())
    log.info(f"mesh dumped {out} ({out.stat().st_size} bytes, "
             f"L={half_l.shape[0]} verts/{infl_l_f.shape[0]} faces, "
             f"R={half_r.shape[0]} verts/{infl_r_f.shape[0]} faces)")
    return f"/static/mesh/{out.name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading atlas ...")
    atlas = np.load(CACHE / "atlas_yeo7_fsaverage5.npz", allow_pickle=True)
    state["labels"] = atlas["labels"].astype(np.int64)
    state["network_names"] = list(atlas["network_names"])
    log.info("Loading TribeModel ...")
    import matplotlib
    matplotlib.use("Agg")
    _patch_video_extractor_autocast()
    from tribev2 import TribeModel
    from imageio_ffmpeg import get_ffmpeg_exe
    state["ffmpeg"] = get_ffmpeg_exe()
    state["mesh_url"] = _dump_fsaverage5_mesh()
    state["model"] = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)
    state["lock"] = threading.Lock()
    log.info(f"Ready. ffmpeg={state['ffmpeg']}, mesh={state['mesh_url']}")
    yield


app = FastAPI(lifespan=lifespan, title="TRIBEv2 viewer")
STATIC.mkdir(exist_ok=True)
(STATIC / "videos").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


def _reduce(preds: np.ndarray, segments) -> dict:
    labels = state["labels"]
    names = state["network_names"]
    networks: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        if i == 0:
            continue
        mask = labels == i
        if mask.any():
            networks[name] = preds[:, mask].mean(axis=1).astype(float)

    vis = networks["Visual"]; smn = networks["Somatomotor"]
    dan = networks["DorsalAttn"]; van = networks["VentralAttn"]
    limbic = networks["Limbic"]; fpcn = networks["FrontoParietal"]; dmn = networks["Default"]

    sensory_attn = np.mean([vis, smn, dan, van], axis=0)
    all_mean = np.mean([vis, smn, dan, van, limbic, fpcn, dmn], axis=0)

    # Heuristic axes — proxies derived from Yeo7 networks. Names are post-hoc.
    excitement     = (_zscore(sensory_attn) - _zscore(dmn)) / 2.0           # sensory-attentional engagement vs. mind-wandering
    valence        = _zscore(limbic)                                          # affective tone (limbic engagement)
    cognitive_load = _zscore((np.asarray(fpcn) + np.asarray(dan)) / 2.0)      # FP control + top-down attention
    novelty        = _zscore(np.asarray(van) - all_mean)                      # salience above baseline (≈ prediction error)

    # Latent space — top-3 PCs of the predictions, computed via numpy SVD on the (T x V) matrix.
    n_comp = max(1, min(3, preds.shape[0] - 1, preds.shape[1]))
    X = preds.astype(np.float32) - preds.mean(axis=0, keepdims=True).astype(np.float32)
    # SVD: X = U S Vt; principal components = U * S, with shape (T, T)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    comps = (U[:, :n_comp] * S[:n_comp])
    total_var = float((S ** 2).sum())
    pca_var = [float(s * s / total_var) if total_var > 0 else 0.0 for s in S[:n_comp]]
    # standardize each component to z
    std = comps.std(axis=0, keepdims=True); std[std < 1e-9] = 1.0
    comps = (comps - comps.mean(axis=0, keepdims=True)) / std
    pca_components = [comps[:, i].astype(float).tolist() for i in range(comps.shape[1])]

    times = [float(s.start) for s in segments]
    durations = [float(s.duration) for s in segments]
    return {
        "n_segments": preds.shape[0],
        "tr": float(np.median(durations)) if durations else 1.0,
        "times": times,
        "durations": durations,
        "networks": {k: v.tolist() for k, v in networks.items()},
        "axes": {
            "excitement":     excitement.tolist(),
            "valence":        valence.tolist(),
            "cognitive_load": cognitive_load.tolist(),
            "novelty":        novelty.tolist(),
        },
        "pca": pca_components,
        "pca_var": pca_var,
    }


def _audio_envelope(video_path: Path, n_bins: int = 1500) -> list[float]:
    """Decode video audio to mono 8 kHz PCM, return per-bin |max| envelope normalized to 0..1."""
    cmd = [state["ffmpeg"], "-loglevel", "error", "-i", str(video_path),
           "-ac", "1", "-ar", "8000", "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    if not proc.stdout:
        return []
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return []
    edges = np.linspace(0, samples.size, n_bins + 1, dtype=int)
    env = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        seg = samples[edges[i]:edges[i + 1]]
        if seg.size:
            env[i] = np.abs(seg).max()
    peak = float(env.max()) or 1.0
    return (env / peak).tolist()


def _words_from_events(df) -> list[dict]:
    out: list[dict] = []
    if "type" not in df.columns:
        return out
    rows = df[df["type"] == "Word"]
    for r in rows.itertuples():
        text = getattr(r, "text", "") or ""
        try:
            t = float(r.start)
            d = float(r.duration)
        except Exception:
            continue
        out.append({"t": t, "d": d, "text": str(text)})
    return out


def _save_preds_binary(job_id: str, preds: np.ndarray) -> dict:
    """Persist (T, V) predictions as a float16 blob in [0,1] (robust-normalised) so the
    browser only has to apply a threshold cutoff and look up the fire colormap.
    """
    from tribev2.plotting.utils import robust_normalize
    out_dir = STATIC / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{job_id}.bin"
    norm = robust_normalize(preds, percentile=99).astype(np.float16)  # in [0, 1]
    norm.tofile(out)
    return {
        "preds_url": f"/static/preds/{out.name}",
        "preds_dtype": "float16",
        "preds_shape": list(preds.shape),
        "preds_normalized": True,
    }


def _extract_thumbs(job_id: str, video_path: Path, times: list[float], q: queue.Queue) -> list[dict]:
    out_dir = STATIC / "thumbs" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs: list[dict] = []
    n = len(times)
    for i, t in enumerate(times):
        out = out_dir / f"{i:04d}.jpg"
        subprocess.run(
            [state["ffmpeg"], "-loglevel", "error", "-ss", f"{max(0.0, t):.2f}",
             "-i", str(video_path), "-frames:v", "1", "-q:v", "6",
             "-vf", "scale=160:-2", "-y", str(out)],
            check=True,
        )
        thumbs.append({"t": float(t), "url": f"/static/thumbs/{job_id}/{out.name}"})
        if (i + 1) % max(1, n // 20) == 0 or i == n - 1:
            q.put({"type": "progress", "phase": "Extracting thumbnails",
                   "percent": int(100 * (i + 1) / n)})
    return thumbs


SUPPORTED_LANGUAGES = (
    "english", "japanese", "japanese_translate",
    "french", "spanish", "dutch", "chinese",
)


def _run_job(job_id: str, video_path: Path, language: str = "english") -> None:
    job = jobs[job_id]
    started = time.time()
    with state["lock"]:
        old_err = sys.stderr
        tee = StderrTee(job.q, sys.__stderr__)
        sys.stderr = tee
        try:
            job.q.put({"type": "progress", "phase": "Normalizing video", "percent": 0})
            norm_path = _normalize_video(video_path)
            if norm_path != video_path:
                video_path = norm_path
            job.q.put({"type": "progress", "phase": "Normalizing video", "percent": 100})

            job.q.put({"type": "log", "message": f"Building events (lang={language}) ..."})
            df = state["model"].get_events_dataframe(video_path=str(video_path), language=language)
            job.q.put({"type": "log", "message": f"Events built: {len(df)} rows"})
            words = _words_from_events(df)
            job.q.put({"type": "log", "message": f"Words: {len(words)}"})
            preds, segments = state["model"].predict(events=df, verbose=False)
            tee.flush()
            job.q.put({"type": "log", "message": "Reducing to networks ..."})
            res = _reduce(preds, segments)
            sys.stderr = old_err  # thumbnail / matplotlib output must not feed the parser
            res["thumbs"] = _extract_thumbs(job_id, video_path, res["times"], job.q)
            job.q.put({"type": "progress", "phase": "Decoding audio waveform", "percent": 50})
            res["waveform"] = _audio_envelope(video_path)
            res["words"] = words
            job.q.put({"type": "progress", "phase": "Decoding audio waveform", "percent": 100})
            job.q.put({"type": "progress", "phase": "Saving predictions", "percent": 50})
            res.update(_save_preds_binary(job_id, preds))
            res["mesh_url"] = state["mesh_url"]
            job.q.put({"type": "progress", "phase": "Saving predictions", "percent": 100})
            res["elapsed_sec"] = round(time.time() - started, 2)
            job.result = res
            job.q.put({"type": "done"})
        except Exception as e:
            log.exception("predict failed")
            import traceback
            tb = traceback.format_exc()
            job.error = f"{type(e).__name__}: {e}\n\n{tb}"
            job.q.put({"type": "error", "message": job.error})
        finally:
            sys.stderr = old_err
            job.done = True


@app.post("/predict")
async def predict(video: UploadFile = File(...), language: str = Form("english")):
    if not video.filename:
        raise HTTPException(400, "missing filename")
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"language must be one of {SUPPORTED_LANGUAGES}")
    suffix = Path(video.filename).suffix or ".mp4"
    job_id = uuid.uuid4().hex
    # Use a fixed filename per user session so neuralset/exca cache hits on re-upload
    saved = STATIC / "videos" / f"latest_upload{suffix}"
    with saved.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    log.info(f"saved upload -> {saved} ({saved.stat().st_size} bytes), lang={language}")
    jobs[job_id] = JobState()
    threading.Thread(target=_run_job, args=(job_id, saved, language), daemon=True).start()
    return {"job_id": job_id, "video_url": f"/static/videos/{saved.name}", "language": language}


@app.get("/events/{job_id}")
async def events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")

    async def gen():
        # Tight 50 ms poll: drain everything available, then await briefly. Avoids the
        # per-message run_in_executor round-trip that was bottlenecking delivery during
        # bursts (e.g. fast-arriving tqdm updates between long V-JEPA iterations).
        last_keepalive = time.time()
        terminated = False
        while not terminated:
            drained = False
            try:
                while True:
                    msg = job.q.get_nowait()
                    drained = True
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("type") in ("done", "error"):
                        terminated = True
                        break
            except queue.Empty:
                pass
            if terminated:
                break
            if drained:
                # let the loop ferry chunks downstream before we re-enter the drain
                await asyncio.sleep(0)
                continue
            if job.done:
                break
            # idle: brief sleep + occasional keepalive so proxies don't drop us
            await asyncio.sleep(0.05)
            now = time.time()
            if now - last_keepalive > 12:
                yield ": keepalive\n\n"
                last_keepalive = now

        if job.result is not None:
            yield f"event: result\ndata: {json.dumps(job.result)}\n\n"
        elif job.error:
            yield f"event: error\ndata: {json.dumps({'message': job.error})}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
