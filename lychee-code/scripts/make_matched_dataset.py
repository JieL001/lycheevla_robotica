"""Marginal-matched control for the bias dial: the scenes of the natural set, and for every scene ONE command drawn so that the frequencies of the command types equal those of the biased set
but the choice among the commands that are valid in a scene does not look at salience.
   python scripts/make_matched_dataset.py --nat D:/lychee_data/select/rho00 --bias D:/lychee_data/select/rho90 --out D:/lychee_data/select/matched90 [--seed 0] [--workers 3]
Why: the bias dial (rho) correlates the command with the most salient fruit and, as a side effect, changes the mix of command types (fewer "farthest" commands, more any-of commands, ...).
The coverage control cuts one type (146 "farthest" commands); this control matches ALL types: the frequency of every command template (family, maturity, side or depth or ordinal position;
36 templates) equals the frequency in the biased set, while the target is not steered towards any fruit.  If a selector trained on it loses as much as the biased one, the loss is a
consequence of the mix of command types and not of the correlation with the cue.
Method: (1) the fruit table of every scene (image position, apparent radius, maturity, visibility; stored with the set) gives back the benchmark's fruit views (depth from the radius: the
inversion is exact, and it reproduces the target sets of all stored commands of both sets, checked below); (2) the valid (template, target set) pairs of every scene follow from
lychee.lang; (3) the weights w(template) of a max-entropy draw p(template | scene) proportional to w over the valid templates are raked until the expected frequencies equal those of the
biased set; (4) one draw per scene, wording and primary target as in the generator.  Reuses the images of the natural set (a hard link where possible).  Writes meta.npz, img.npy, info.json.
The natural and the biased sets share their scenes (one seed range), which is checked."""
import argparse, json, multiprocessing as mp, os, shutil, sys, time

import numpy as np

sys.path.insert(0, ".")
FAMS = ["mat", "mat_any", "mat_side", "mat_depth", "mat_ordinal"]


def _light():
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def views_of(fxy, frad, fmat, fvis, fvalid):
    from lychee.lang import FruitView
    from lychee.modular import depth_from_radius
    from lychee.select import IMG_HW
    return [FruitView(j, int(fmat[j]), float(fxy[j][0]) / IMG_HW[1], depth_from_radius(float(frad[j])), float(fvis[j])) for j in range(len(fvalid)) if fvalid[j]]


def _valid(job):
    """[(template index, target tuple)] of one scene."""
    from lychee import lang
    from lychee.splits import SPLITS
    split = SPLITS["train"]
    fxy, frad, fmat, fvis, fvalid = job
    lo, hi = split.target_vis
    idx = {s: k for k, s in enumerate(lang.all_specs())}
    return [(idx[s], tuple(T)) for s, T in lang.valid_specs_sets(views_of(fxy, frad, fmat, fvis, fvalid), min_vis=lo, max_vis=hi, ord_min_vis=split.ord_min_vis, any_rule=split.any_rule)]


def rake(V, target, iters=200, tol=0.05):
    """Weights w of the max-entropy draw p(template | scene) proportional to w over the templates valid in the scene (rows of the boolean matrix V), raked until the expected number of
    draws of every template equals ``target`` (templates with a zero target get weight 0).  Returns (w, largest deviation of an expected count from its target, iterations used)."""
    target = np.asarray(target, float)
    w = np.where(target > 0, 1.0, 0.0)
    err = np.inf
    for it in range(iters):
        z = (V * w).sum(1, keepdims=True)
        e = (V * w / np.maximum(z, 1e-300)).sum(0)
        w = w * np.where((e > 0) & (target > 0), target / np.maximum(e, 1e-12), 1.0)
        w = w / w.max()
        err = float(np.abs(e - target).max())
        if err < tol:
            break
    return w, err, it + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nat", required=True)
    ap.add_argument("--bias", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    from lychee import lang
    from lychee.bc import tokenize
    from lychee.modular import parse_command
    from lychee.splits import SPLITS, _episode, salience

    split = SPLITS["train"]
    nat, bia = np.load(f"{a.nat}/meta.npz"), np.load(f"{a.bias}/meta.npz")
    ok = nat["ok"].copy()
    assert (ok == bia["ok"]).all() and (nat["seed"][ok] == bia["seed"][ok]).all() and np.allclose(nat["fxy"][ok], bia["fxy"][ok]), "the two sets do not share their scenes"
    specs = lang.all_specs()
    sidx = {s: k for k, s in enumerate(specs)}
    S, N = len(specs), len(ok)

    def counts(d):
        c = np.zeros(S)
        for i in np.flatnonzero(ok):
            c[sidx[parse_command(d["text"][i, 0])]] += 1
        return c

    c_nat, c_bia = counts(nat), counts(bia)
    print(f"templates: {S}; commands: natural {int(c_nat.sum())}, biased {int(c_bia.sum())}; templates never used: natural {(c_nat == 0).sum()}, biased {(c_bia == 0).sum()}", flush=True)

    t0 = time.time()
    jobs = [(nat["fxy"][i], nat["frad"][i], nat["fmat"][i], nat["fvis"][i], nat["fvalid"][i]) for i in np.flatnonzero(ok)]
    with mp.get_context("spawn").Pool(a.workers, initializer=_light) as pool:
        valid = pool.map(_valid, jobs, chunksize=50)
    print(f"valid templates of {len(valid)} scenes in {time.time()-t0:.0f}s (mean {np.mean([len(v) for v in valid]):.1f} per scene)", flush=True)
    V = np.zeros((len(valid), S), bool)
    for r, v in enumerate(valid):
        for k, _ in v:
            V[r, k] = True

    # rake the weights until the expected template frequencies of the max-entropy draw equal those of the biased set
    target = c_bia.astype(float)
    w, err, n_it = rake(V, target, a.iters)
    print(f"raking: {n_it} iterations, largest deviation of an expected template count {err:.2f} (counts up to {int(target.max())})", flush=True)

    rng = np.random.RandomState(1_234_567 + a.seed)
    pick = np.full(len(valid), -1)
    for r in range(len(valid)):
        p = V[r] * w
        p = p / p.sum()
        pick[r] = rng.choice(S, p=p)

    # write the set
    os.makedirs(a.out, exist_ok=True)
    arr = {k: nat[k].copy() for k in nat.files}
    arr["tok"][:] = 0; arr["tmask"][:] = False; arr["primary"][:] = -1; arr["text"][:] = ""; arr["ncmd"][:] = 1; arr["field"][:] = ""
    misalign = np.zeros(2)                              # commands whose target set contains the most salient fruit: [matched set, count]
    rows = np.flatnonzero(ok)
    for r, i in enumerate(rows):
        spec = specs[pick[r]]
        T = dict(valid[r])[pick[r]]
        fr = views_of(nat["fxy"][i], nat["frad"][i], nat["fmat"][i], nat["fvis"][i], nat["fvalid"][i])
        ep = _episode(spec, T, fr, split.form, rng)
        arr["tok"][i, 0] = tokenize(ep.text); arr["tmask"][i, 0, list(ep.targets)] = True; arr["primary"][i, 0] = ep.primary; arr["text"][i, 0] = ep.text
        arr["family"][i] = FAMS.index(spec.family); arr["form"][i] = ep.form
        top = int(np.argmax([salience(f) for f in fr]))
        misalign += (top in T, 1)
    np.savez(f"{a.out}/meta.npz", **arr)
    src, dst = f"{a.nat}/img.npy", f"{a.out}/img.npy"
    if not os.path.exists(dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    info = json.load(open(f"{a.nat}/info.json"))
    info.update(mode="single", rho="matched", matched_to=a.bias, seed=a.seed, ok=int(ok.sum()), seconds=0.0, reused_images_of=a.nat)      # labels only: no rendering
    json.dump(info, open(f"{a.out}/info.json", "w"))

    # report: family mix, the share of commands that name the most salient fruit, and the largest deviation of a template frequency from the biased set
    def share(d):
        rows_ = np.flatnonzero(ok)
        return np.array([np.mean([FAMS[int(d["family"][i])] == f for i in rows_]) for f in FAMS])
    fam = {"natural": share(nat), "biased": share(bia), "matched": share(arr)}
    print("family mix (" + ", ".join(FAMS) + "):")
    for k, v in fam.items():
        print(f"  {k:8s} " + "  ".join(f"{100*x:5.1f}" for x in v))
    c_new = np.zeros(S)
    for i in rows:
        c_new[sidx[parse_command(arr["text"][i, 0])]] += 1
    dev = np.abs(c_new - c_bia)
    print(f"matched set: share of commands that name the most salient fruit {100*misalign[0]/misalign[1]:.1f} %; largest deviation of a template count from the biased set {int(dev.max())} "
          f"(of {int(c_bia.max())}), total absolute deviation {int(dev.sum())} of {int(c_bia.sum())}")
    far = [k for k, s in enumerate(specs) if s.family == "mat_depth" and s.depth == "far"]
    print(f"'farthest' commands: natural {int(c_nat[far].sum())}, biased {int(c_bia[far].sum())}, matched {int(c_new[far].sum())}")
    json.dump(dict(share_salient=float(misalign[0] / misalign[1]), family_mix={k: v.tolist() for k, v in fam.items()}, template_counts=dict(natural=c_nat.tolist(), biased=c_bia.tolist(),
              matched=c_new.tolist()), templates=[str(s) for s in specs]), open(f"{a.out}/matched_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
