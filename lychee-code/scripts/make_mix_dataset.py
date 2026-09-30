"""Concatenate two rendered selection-track training sets: all scenes of set A plus the first NB scenes of set B (the remedy-under-bias experiment: a biased set plus extra
paired scenes versus the same biased set plus extra unpaired natural scenes).  NOTE: all rho sets share one scene-seed range, so "the first NB scenes of the natural set" are the
same images as the first NB scenes of the biased set (with an unbiased command instead of a biased one): the "natural" extra adds no new images (the paper calls the arm "relabelled").
   python scripts/make_mix_dataset.py --a D:/lychee_data/select/rho90 --b D:/lychee_data/select/paired --nb 2000 --out D:/lychee_data/select/mix_rho90_paired
Arrays of meta.npz are concatenated along the scene axis; img.npy is written as a new memmap (chunked copy)."""
import argparse, json, os, shutil
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--nb", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

A, B = np.load(f"{a.a}/meta.npz"), np.load(f"{a.b}/meta.npz")
Na = len(A["ok"])
nb = min(a.nb, len(B["ok"]))
os.makedirs(a.out, exist_ok=True)
arr = {k: np.concatenate([A[k], B[k][:nb]], axis=0) for k in A.files}
np.savez(f"{a.out}/meta.npz", **arr)
ia = np.load(f"{a.a}/img.npy", mmap_mode="r")
ib = np.load(f"{a.b}/img.npy", mmap_mode="r")
out = np.lib.format.open_memmap(f"{a.out}/img.npy", "w+", np.uint8, (Na + nb,) + ia.shape[1:])
for s in range(0, Na, 500):
    out[s:min(Na, s + 500)] = ia[s:min(Na, s + 500)]
for s in range(0, nb, 500):
    out[Na + s:Na + min(nb, s + 500)] = ib[s:min(nb, s + 500)]
out.flush()
info = json.load(open(f"{a.a}/info.json"))
info.update(mode="mix", n=Na + nb, ok=int(arr["ok"].sum()), parts=[a.a, f"{a.b}[:{nb}]"])
json.dump(info, open(f"{a.out}/info.json", "w"))
print(f"wrote {a.out}: {Na} + {nb} scenes, commands {int(arr['ncmd'][arr['ok'].astype(bool)].sum())}")
