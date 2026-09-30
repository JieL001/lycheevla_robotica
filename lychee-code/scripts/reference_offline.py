"""Selection-level reference policies, computed offline (no simulator): the scripted reference policies of lychee.evalkit choose a fruit from the
scene state (visibility, maturity, position, depth, the command's fields); the scripted expert then executes the pick with 99% success (measured
closed-loop, Section "Demonstrations"), so the selection-level PTA is the PTA of the policy up to that 1%.  Because nothing is simulated, every split can be
evaluated on the same 600 / 300 pairs as the learned selectors.
   python scripts/reference_offline.py [--workers 3]
Writes results/eval_ref_offline/<policy>_<split>.jsonl (records in the format of eval_select.py) and ../tables/reference_iid.tex, reference_splits.tex,
reference_meta.tex (the same file names and macros that make_tables.py wrote for the closed-loop reference on 60 pairs)."""
import argparse, json, os, sys
import multiprocessing as mp

import numpy as np

sys.path.insert(0, ".")
POLICIES = [("expert", "Oracle (the label)"), ("attr_visible", "Maturity word only"), ("rel_only", "Relation words only"),
            ("random", "Random fruit"), ("blind_visible", "Language-blind: most visible"), ("blind_front", "Language-blind: nearest")]
SPLITS = [("iid", "IID", 600), ("occ_ood", "Occl.", 300), ("density_ood", "Dens.", 300), ("lang_ood", "Lang.", 300),
          ("attr_ood", "Attr.", 300), ("saliency_rev", "Sal.-rev.", 600)]
FAMS = [("mat_any", "any-of"), ("mat_side", "side"), ("mat_ordinal", "ordinal"), ("mat_depth", "depth"), ("mat", "unique")]
OUT = "results/eval_ref_offline"


class _U:
    """The attributes of the environment that the reference policies read."""
    def __init__(self, lay, ep):
        self.vis, self.mat_of, self.spec, self.n_fruit, self.target_idx = lay.vis, lay.mats, ep.spec, lay.n, ep.primary
        self._fv = lay.fruit_views()

    def fruit_views(self):
        return self._fv


def _job(args):
    split, index = args
    from lychee import evalkit
    from lychee.splits import SPLITS as SP, sample_config, layout_for
    cfg = sample_config(split, index, pair=True)
    lay = layout_for(SP[split], cfg.scene_seed)
    out = {}
    for tag, _ in POLICIES:
        cls = evalkit.BASELINES[tag]
        recs = []
        for which, ep in (("plus", cfg.plus), ("minus", cfg.minus)):
            u = _U(lay, ep)
            pol = cls.__new__(cls)
            pol.rng = np.random.RandomState((int(cfg.scene_seed) * 2 + (which == "minus")) % (2 ** 31 - 1))
            s = int(cls.choose(pol, u))
            ok = s in ep.targets
            recs.append(dict(split=split, index=index, which=which, family=cfg.family, field=cfg.edited_field, form=ep.form, text=ep.text,
                             n_fruit=int(lay.n), steps=0, seconds=0.0, first_detached=s, first_grasped=s, first_approached=s,
                             targets=[int(t) for t in ep.targets], primary=int(ep.primary),
                             target_vis=float(min(lay.vis[t] for t in ep.targets)), policy=tag, mode="selection", success=bool(ok),
                             harvested=bool(ok), detached=True, target_correct=bool(ok), wrong_target=not ok, approach_correct=bool(ok),
                             touched_nontarget=False, selected=s))
        out[tag] = recs
    return out


def ci(x):
    m, lo, hi = x
    return f"{100*m:.1f} [{100*lo:.0f}, {100*hi:.0f}]"


def main(workers):
    from lychee import evalkit
    os.makedirs(OUT, exist_ok=True)
    recs = {}
    for split, _, n in SPLITS:
        paths = {t: f"{OUT}/{t}_{split}.jsonl" for t, _ in POLICIES}
        if all(os.path.exists(p) for p in paths.values()):
            for t, p in paths.items():
                recs[(t, split)] = [json.loads(l) for l in open(p)]
            continue
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_job, [(split, i) for i in range(n)], chunksize=5)
        for t, p in paths.items():
            recs[(t, split)] = [r for scene in res for r in scene[t]]
            with open(p, "w") as fh:
                for r in recs[(t, split)]:
                    fh.write(json.dumps(r) + "\n")
        print("done", split, flush=True)
    # tables
    ref = recs[("expert", "iid")]
    fam_n = {k: v["n_pairs"] for k, v in evalkit.summarize(ref, by="family").items()}
    n_pairs = evalkit.summarize(ref)["all"]["n_pairs"]
    rows, chance = [], None
    for tag, label in POLICIES:
        r = recs[(tag, "iid")]
        s = evalkit.summarize(r)["all"]
        fam = evalkit.summarize(r, by="family")
        macro = evalkit.macro_average(r)[0]
        cells = [f"{100*fam[k]['PTA_sel'][0]:.0f}" if k in fam else "--" for k, _ in FAMS]
        rows.append(f"{label} & {ci(s['PTA_sel'])} & {100*macro:.0f} & {100*s['TSA'][0]:.1f} & {100*s['wrong_target'][0]:.1f} & " + " & ".join(cells))
        if tag == "expert":
            chance = 100 * s["chance_PTA"]
    rows.append(f"\\emph{{pairs per family}} & {n_pairs} & & & & " + " & ".join(str(fam_n.get(k, 0)) for k, _ in FAMS))
    os.makedirs("../tables", exist_ok=True)
    open("../tables/reference_iid.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    lines = [f"\\newcommand{{\\refPairs}}{{{n_pairs}}}", f"\\newcommand{{\\refChance}}{{{chance:.1f}}}"]
    for k, name in FAMS:
        lines.append(f"\\newcommand{{\\refN{name.replace('-', '').capitalize()}}}{{{fam_n.get(k, 0)}}}")
    open("../tables/reference_meta.tex", "w", newline="\n").write("\n".join(lines) + "\n")
    srows = []
    for tag, label in POLICIES:
        cells = []
        for sp, _, _ in SPLITS:
            s = evalkit.summarize(recs[(tag, sp)])["all"]
            cells.append(f"{ci(s['PTA_sel'])}")
        srows.append(f"{label} & " + " & ".join(cells))
    open("../tables/reference_splits.tex", "w", newline="\n").write(" \\\\\n".join(srows) + "\n")
    print("\n".join(srows))
    print(f"wrote tables; chance level on IID {chance:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    main(ap.parse_args().workers)
