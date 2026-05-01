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


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading atlas ...")
    atlas = np.load(CACHE / "atlas_yeo7_fsaverage5.npz", allow_pickle=True)
    state["labels"] = atlas["labels"].astype(np.int64)
    state["network_names"] = list(atlas["network_names"])
    log.info("Loading TribeModel ...")
    import matplotlib
    matplotlib.use("Agg")
    from tribev2 import TribeModel
    from tribev2.plotting import PlotBrainNilearn
    from imageio_ffmpeg import get_ffmpeg_exe
    state["ffmpeg"] = get_ffmpeg_exe()
    state["model"] = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)
    state["plotter"] = PlotBrainNilearn(mesh="fsaverage5", inflate="half", bg_map="sulcal")
    # warm up the brain renderer to avoid first-call latency in the worker
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), subplot_kw={"projection": "3d"},
                              gridspec_kw={"wspace": 0, "hspace": 0})
    state["plotter"].plot_surf(np.zeros(20484, dtype=np.float32),
                                axes=[axes[0], axes[1]], views=["left", "right"],
                                cmap="fire", vmin=0.6, alpha_cmap=(0, 0.2))
    plt.close(fig)
    state["lock"] = threading.Lock()
    log.info(f"Ready. ffmpeg={state['ffmpeg']}")
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


def _render_brains(job_id: str, preds: np.ndarray, q: queue.Queue) -> list[str]:
    """Render one brain panel (left+right lateral) per timestep, save as PNG."""
    import matplotlib.pyplot as plt
    from tribev2.plotting.utils import robust_normalize

    out_dir = STATIC / "brain" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plotter = state["plotter"]
    preds_norm = robust_normalize(preds, percentile=99)  # global [0,1]
    n = preds.shape[0]
    urls: list[str] = []
    for i in range(n):
        fig, axes = plt.subplots(
            1, 2, figsize=(4, 2),
            subplot_kw={"projection": "3d"},
            gridspec_kw={"wspace": 0, "hspace": 0},
        )
        plotter.plot_surf(
            preds_norm[i], axes=[axes[0], axes[1]], views=["left", "right"],
            cmap="fire", vmin=0.6, alpha_cmap=(0, 0.2),
        )
        out = out_dir / f"{i:04d}.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0, dpi=72,
                    facecolor="white", transparent=False)
        plt.close(fig)
        urls.append(f"/static/brain/{job_id}/{out.name}")
        if (i + 1) % max(1, n // 25) == 0 or i == n - 1:
            q.put({"type": "progress", "phase": "Rendering brain",
                   "percent": int(100 * (i + 1) / n)})
    return urls


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
            res["brain_urls"] = _render_brains(job_id, preds, job.q)
            res["elapsed_sec"] = round(time.time() - started, 2)
            job.result = res
            job.q.put({"type": "done"})
        except Exception as e:
            log.exception("predict failed")
            job.error = f"{type(e).__name__}: {e}"
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
    saved = STATIC / "videos" / f"{job_id}{suffix}"
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
        loop = asyncio.get_event_loop()
        while True:
            try:
                msg = await loop.run_in_executor(None, lambda: job.q.get(timeout=1.0))
            except queue.Empty:
                if job.done:
                    break
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("type") in ("done", "error"):
                break
        if job.result is not None:
            yield f"event: result\ndata: {json.dumps(job.result)}\n\n"
        elif job.error:
            yield f"event: error\ndata: {json.dumps({'message': job.error})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
