"""Build vertex-to-network mapping for fsaverage5 (20484 vertices)."""
from pathlib import Path
import zipfile
import urllib.request
import numpy as np
from nilearn import datasets, surface, image

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
out = CACHE / "atlas_yeo7_fsaverage5.npz"

zip_path = CACHE / "Yeo_JNeurophysiol11_MNI152.zip"
if not zip_path.exists():
    url = "https://surfer.nmr.mgh.harvard.edu/ftp/data/Yeo_JNeurophysiol11_MNI152.zip"
    print("Downloading", url, flush=True)
    urllib.request.urlretrieve(url, zip_path)
    print(f"  -> {zip_path.stat().st_size} bytes")

extract_dir = CACHE / "yeo_atlas"
if not extract_dir.exists():
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

all_nii = sorted(extract_dir.rglob("*.nii*"))
seven = [p for p in all_nii if "Yeo2011_7Networks" in p.name and "Liberal" in p.name]
nii = seven[0] if seven else None
if nii is None:
    raise RuntimeError(f"No 7Networks Liberal NIfTI in {extract_dir}: {[p.name for p in all_nii]}")
print("Using atlas:", nii.relative_to(extract_dir))

print("Fetching fsaverage5 mesh ...")
fsavg = datasets.fetch_surf_fsaverage("fsaverage5")

yeo_img = image.load_img(str(nii))
if len(yeo_img.shape) == 4:
    yeo_img = image.index_img(yeo_img, 0)

print("Projecting to fsaverage5 left + right pial ...")
lh = surface.vol_to_surf(
    yeo_img, fsavg.pial_left, interpolation="nearest_most_frequent", radius=3.0
).astype(np.int8)
rh = surface.vol_to_surf(
    yeo_img, fsavg.pial_right, interpolation="nearest_most_frequent", radius=3.0
).astype(np.int8)

labels = np.concatenate([lh, rh])
hemi = np.concatenate([np.zeros(lh.shape[0], dtype=np.int8),
                       np.ones(rh.shape[0], dtype=np.int8)])
network_names = [
    "MedialWall", "Visual", "Somatomotor", "DorsalAttn",
    "VentralAttn", "Limbic", "FrontoParietal", "Default",
]
print(f"shape {labels.shape}, hemi sums L={int((hemi==0).sum())} R={int((hemi==1).sum())}")
for i, name in enumerate(network_names):
    n = int((labels == i).sum())
    print(f"  {i} {name:<14} {n:>5} vertices ({100*n/labels.size:.1f}%)")

np.savez(out, labels=labels, hemi=hemi, network_names=np.array(network_names))
print("Saved", out)
