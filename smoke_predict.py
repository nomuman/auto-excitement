"""Smoke test: end-to-end predict on the Sintel trailer."""
from pathlib import Path
import time

import numpy as np
from tribev2 import TribeModel
from tribev2.demo_utils import download_file

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

video_path = CACHE / "sample_video.mp4"
if not video_path.exists():
    url = "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4"
    download_file(url, video_path)

print("Loading model ...", flush=True)
t0 = time.time()
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)
print(f"Model loaded in {time.time()-t0:.1f}s")

print("Building events ...", flush=True)
t0 = time.time()
df = model.get_events_dataframe(video_path=video_path)
print(f"Events built in {time.time()-t0:.1f}s; rows={len(df)}")

print("Predicting ...", flush=True)
t0 = time.time()
preds, segments = model.predict(events=df)
dt = time.time() - t0
print(f"Predict done in {dt:.1f}s")
print(f"preds shape: {preds.shape}, dtype: {preds.dtype}")
print(f"segments: {len(segments)}")
print("First few segment offsets (s):", [round(float(s.offset), 2) for s in segments[:8]])
print("preds stats: min", float(preds.min()), "max", float(preds.max()), "mean", float(preds.mean()))

np.save(CACHE / "smoke_preds.npy", preds)
import pickle
with open(CACHE / "smoke_segments.pkl", "wb") as f:
    pickle.dump([{"offset": float(s.offset), "duration": float(s.duration)} for s in segments], f)
print("Saved preds + segments to cache/")
