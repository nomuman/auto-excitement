"""Smoke test: load TribeModel from HF Hub."""
import sys
from pathlib import Path

from tribev2 import TribeModel

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)

print("Loading TribeModel from facebook/tribev2 ...", flush=True)
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)
print("OK. TR =", model.data.TR)
print("Model device:", next(model._model.parameters()).device)
print("Param count:", sum(p.numel() for p in model._model.parameters()))
