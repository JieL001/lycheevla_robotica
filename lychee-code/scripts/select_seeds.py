"""Multi-seed summary of the selection track: PTA per arm and split (mean over training seeds, seed range), on all pairs and on the relational pairs (depth and ordinal
commands, where all of the effects live), recipe-level differences with a seed-and-scene cluster bootstrap and with a seed-level Welch t interval, and LaTeX macros so
that the numbers in the manuscript text are generated, never typed.
   python scripts/select_seeds.py [--boot 4000]
Writes ../tables/select_seeds.tex (arms x splits), ../tables/select_compare.tex (comparisons), ../tables/select_macros.tex and results/eval/select_seeds.md.
Seeds are the files s_<arm>.jsonl (seed 0), s_<arm>_s1.jsonl, s_<arm>_s2.jsonl.
Macros: \\sm<Arm><Split> mean over seeds, \\sm<Arm><Split>Rel the same on relational pairs, \\zs<Arm><Split> seed 0 only, \\dd/\\dl/\\dh<Comparison> difference and interval ends
(bootstrap), \\aa<Comparison> its absolute size, \\rd/\\rl/\\rh<Comparison> the same on relational pairs, \\tl/\\th<Comparison> ends of the seed-level t interval,
\\rg<Group><Min|Max>[Loss]<Split> range over the arms of a group (Px pixel, Ft fruit table, All) of the seed means (Loss: IID minus split)."""
import argparse, json, os, re, sys

import numpy as np
from scipy import stats as sst

sys.path.insert(0, ".")
from lychee import evalkit, stats

SPLITS = [("iid", "IID", "Iid"), ("sal", "Sal.-rev.", "Sal"), ("occ", "Occl.", "Occ"), ("occn", "Occl. no any-of", "OccN"), ("dens", "Dens.", "Dens"), ("lang", "Lang.", "Lang"),
          ("attr", "Attr.", "Attr"), ("dr", "Vis.-DR", "Dr"), ("iidn", "IID no any-of", "IidN"), ("occa", "Occl. any-of only", "OccA")]      # the last two are not in the manuscript table
# derived splits: (file split, families kept).  On the occlusion split the v1.0 target set of an any-of command contains only the fruit of the named maturity in the visibility range of
# the split (0.2-0.4), which the command does not state: every method, the oracle pipeline included, answers those 22 % of the pairs at 18-29 % (scripts/occ_anyof.py scores them under the
# plain reading of the command).  "occn" leaves them out and is the occlusion column of the manuscript; "occ" (all families) is a legacy diagnostic.
NOANY = ("mat", "mat_side", "mat_depth", "mat_ordinal")
DERIVED = {"occn": ("occ", NOANY), "iidn": ("iid", NOANY), "occa": ("occ", ("mat_any",))}
REL = ("mat_depth", "mat_ordinal")                                   # the families whose pairs are not solved by every arm
ARMS = [  # key (macro), base file name, label
    ("Rzero", "r0_rho00_film", r"R0, $\rho=0$"),
    ("Rhalf", "r0_rho50_film", r"R0, $\rho=0.5$"),
    ("Rninety", "r0_rho90_film", r"R0, $\rho=0.9$"),
    ("Rninetyseven", "r0_rho97_film", r"R0, $\rho=0.97$"),
    ("Rninetynine", "r0_rho99_film", r"R0, $\rho=0.99$"),
    ("Rone", "r1_film", "R1"),
    ("Ronenu", "r1u_film", "R1u"),
    ("Rsame", "r1n_film", "R1n"),
    ("RzeroP", "r0p_film", "R0p"),
    ("Rlate", "r1_late", "R1, late fusion"),
    ("Rtoken", "r1_token", "R1, command token"),
    ("RzeroLate", "r0_rho00_late", r"R0, $\rho=0$, late fusion"),
    ("RninetyLate", "r0_rho90_late", r"R0, $\rho=0.9$, late fusion"),
    ("RzeroToken", "r0_rho00_token", r"R0, $\rho=0$, command token"),
    ("RninetyToken", "r0_rho90_token", r"R0, $\rho=0.9$, command token"),
    ("MixPaired", "mix_rho90_paired", r"R0, $\rho=0.9$ + paired"),
    ("MixNatural", "mix_rho90_natural", r"R0, $\rho=0.9$ + relabelled"),
    ("MixBiased", "mix_rho90_biased", r"R0, $\rho=0.9$ + biased"),
    ("Far", "r0_rho00_far146", r"R0, $\rho=0$, 146 ``farthest''"),
    ("Rmatched", "r0_matched90_film", r"R0, command mix of $\rho=0.9$, no correlation"),
    ("ModOracle", "mod_oracle", "Modular, true fruit table"),
    ("ModDet", "mod_det", "Modular, benchmark margins"),
    ("ModFree", "mod_free", "Modular, no margins"),
    ("RoneNoShift", "r1_film_shiftnone", "R1, no image shifts"),
    ("RoneSyncShift", "r1_film_shiftsync", "R1, synchronised shifts"),
    ("RlateNoShift", "r1_late_shiftnone", "R1, late fusion, no image shifts"),
    ("RlateSyncShift", "r1_late_shiftsync", "R1, late fusion, synchronised shifts"),
    ("RlateWide", "r1_late_wide", "R1, late fusion (0.60 M)"),
    ("RoneLrLo", "r1_film_lr3e-4", r"R1, FiLM, lr $3\cdot10^{-4}$"),
    ("RoneLrHi", "r1_film_lr3e-3", r"R1, FiLM, lr $3\cdot10^{-3}$"),
    ("RlateLrLo", "r1_late_lr3e-4", r"R1, late fusion, lr $3\cdot10^{-4}$"),
    ("RlateLrHi", "r1_late_lr3e-3", r"R1, late fusion, lr $3\cdot10^{-3}$"),
    ("Sone", "sym_r1", "R1, fruit table"),
    ("Szero", "sym_r0_rho00", r"R0, $\rho=0$, fruit table"),
    ("Sninety", "sym_r0_rho90", r"R0, $\rho=0.9$, fruit table"),
    ("Sninetyseven", "sym_r0_rho97", r"R0, $\rho=0.97$, fruit table"),
    ("Sninetynine", "sym_r0_rho99", r"R0, $\rho=0.99$, fruit table"),
]
COMPARE = [  # label, macro, A key, B key, split
    (r"C1: R0$_{\rho=0.9}$ $-$ R0$_{\rho=0}$, saliency-reversed", "ConeSal", "Rninety", "Rzero", "sal"),
    (r"C1: R0$_{\rho=0.9}$ $-$ R0$_{\rho=0}$, IID", "ConeIid", "Rninety", "Rzero", "iid"),
    (r"C2: R1 $-$ R0$_{\rho=0.9}$, saliency-reversed", "CtwoSal", "Rone", "Rninety", "sal"),
    (r"C2: R1 $-$ R0$_{\rho=0.9}$, IID", "CtwoIid", "Rone", "Rninety", "iid"),
    (r"C3: R1 $-$ R1u, IID", "CthreeIid", "Rone", "Ronenu", "iid"),
    (r"R0$_{\rho=0.97}$ $-$ R0$_{\rho=0}$, saliency-reversed", "DialSevenSal", "Rninetyseven", "Rzero", "sal"),
    (r"R0$_{\rho=0.99}$ $-$ R0$_{\rho=0}$, saliency-reversed", "DialNineSal", "Rninetynine", "Rzero", "sal"),
    (r"R0$_{\rho=0.99}$ $-$ R0$_{\rho=0}$, IID", "DialNineIid", "Rninetynine", "Rzero", "iid"),
    (r"R0$_{\rho=0.9}$ $-$ R0$_{\rho=0}$, late fusion, saliency-reversed", "DialLateSal", "RninetyLate", "RzeroLate", "sal"),
    (r"R0$_{\rho=0.9}$ $-$ R0$_{\rho=0}$, command token, saliency-reversed", "DialTokenSal", "RninetyToken", "RzeroToken", "sal"),
    (r"R1 $-$ R0$_{\rho=0}$, IID", "RoneZeroIid", "Rone", "Rzero", "iid"),
    (r"R1 $-$ R0$_{\rho=0}$, saliency-reversed", "RoneZeroSal", "Rone", "Rzero", "sal"),
    (r"R1u $-$ R0$_{\rho=0}$, IID", "RnuZeroIid", "Ronenu", "Rzero", "iid"),
    (r"R1u $-$ R0$_{\rho=0.9}$, saliency-reversed", "RnuNineSal", "Ronenu", "Rninety", "sal"),
    (r"R1u $-$ R0$_{\rho=0.9}$, IID", "RnuNineIid", "Ronenu", "Rninety", "iid"),
    (r"R1 $-$ R0$_{\rho=0}$, attribute-OOD", "RoneZeroAttr", "Rone", "Rzero", "attr"),
    (r"R1 $-$ R1u, attribute-OOD", "RoneNuAttr", "Rone", "Ronenu", "attr"),
    (r"R1 $-$ R1u, saliency-reversed", "RoneNuSal", "Rone", "Ronenu", "sal"),
    (r"R1 $-$ R1n, IID", "RoneSameIid", "Rone", "Rsame", "iid"),
    (r"R1 $-$ R1n, saliency-reversed", "RoneSameSal", "Rone", "Rsame", "sal"),
    (r"R1 $-$ R1n, attribute-OOD", "RoneSameAttr", "Rone", "Rsame", "attr"),
    (r"R1n $-$ R0p, IID", "SamePIid", "Rsame", "RzeroP", "iid"),
    (r"R1n $-$ R0p, attribute-OOD", "SamePAttr", "Rsame", "RzeroP", "attr"),
    (r"R1u $-$ R1n, IID", "RnuSameIid", "Ronenu", "Rsame", "iid"),
    (r"R1u $-$ R1n, saliency-reversed", "RnuSameSal", "Ronenu", "Rsame", "sal"),
    (r"R1u $-$ R1n, attribute-OOD", "RnuSameAttr", "Ronenu", "Rsame", "attr"),
    (r"R0$_{\rho=0.9}$ + paired $-$ R0$_{\rho=0.9}$, saliency-reversed", "MixPairedNineSal", "MixPaired", "Rninety", "sal"),
    (r"R0$_{\rho=0.9}$ + relabelled $-$ R0$_{\rho=0.9}$, saliency-reversed", "MixNaturalNineSal", "MixNatural", "Rninety", "sal"),
    (r"R1 $-$ R1 with late fusion, IID", "RoneLateIid", "Rone", "Rlate", "iid"),
    (r"R1 $-$ R1 with a command token, IID", "RoneTokenIid", "Rone", "Rtoken", "iid"),
    (r"R1 $-$ R1 with late fusion, attribute-OOD", "RoneLateAttr", "Rone", "Rlate", "attr"),
    (r"R1 $-$ R1 with a command token, attribute-OOD", "RoneTokenAttr", "Rone", "Rtoken", "attr"),
    (r"R1 $-$ modular pipeline (benchmark margins), IID", "RoneModIid", "Rone", "ModDet", "iid"),
    (r"R1 $-$ modular pipeline (benchmark margins), attribute-OOD", "RoneModAttr", "Rone", "ModDet", "attr"),
    (r"R1 $-$ modular pipeline (no margins), IID", "RoneModFreeIid", "Rone", "ModFree", "iid"),
    (r"R1 $-$ modular pipeline (no margins), attribute-OOD", "RoneModFreeAttr", "Rone", "ModFree", "attr"),
    (r"R1 with late fusion $-$ modular pipeline (no margins), attribute-OOD", "RlateModFreeAttr", "Rlate", "ModFree", "attr"),
    (r"R1 with a command token $-$ modular pipeline (no margins), attribute-OOD", "RtokenModFreeAttr", "Rtoken", "ModFree", "attr"),
    (r"R1 $-$ R1 with late fusion (0.60 M), IID", "RoneLateWideIid", "Rone", "RlateWide", "iid"),
    (r"R1 $-$ R1 with late fusion (0.60 M), attribute-OOD", "RoneLateWideAttr", "Rone", "RlateWide", "attr"),
    (r"R1 $-$ R1 without image shifts, IID", "RoneNoShiftIid", "Rone", "RoneNoShift", "iid"),
    (r"R1 $-$ R1 without image shifts, attribute-OOD", "RoneNoShiftAttr", "Rone", "RoneNoShift", "attr"),
    (r"R1 $-$ R1 with synchronised shifts, IID", "RoneSyncShiftIid", "Rone", "RoneSyncShift", "iid"),
    (r"R1 $-$ R1 with synchronised shifts, attribute-OOD", "RoneSyncShiftAttr", "Rone", "RoneSyncShift", "attr"),
    (r"R1 with late fusion $-$ late fusion without image shifts, IID", "RlateNoShiftIid", "Rlate", "RlateNoShift", "iid"),
    (r"R1 with late fusion $-$ late fusion without image shifts, attribute-OOD", "RlateNoShiftAttr", "Rlate", "RlateNoShift", "attr"),
    (r"R1 with late fusion $-$ late fusion with synchronised shifts, IID", "RlateSyncShiftIid", "Rlate", "RlateSyncShift", "iid"),
    (r"R1 with late fusion $-$ late fusion with synchronised shifts, attribute-OOD", "RlateSyncShiftAttr", "Rlate", "RlateSyncShift", "attr"),
    (r"R1 without image shifts $-$ late fusion without image shifts, attribute-OOD", "RoneNoShiftRlateNoShiftAttr", "RoneNoShift", "RlateNoShift", "attr"),
    (r"R1 with synchronised shifts $-$ late fusion with synchronised shifts, attribute-OOD", "RoneSyncShiftRlateSyncShiftAttr", "RoneSyncShift", "RlateSyncShift", "attr"),
    (r"R0$_{\rho=0.9}$ + biased $-$ R0$_{\rho=0.9}$, saliency-reversed", "MixBiasedNineSal", "MixBiased", "Rninety", "sal"),
    (r"R0$_{\rho=0.9}$ + biased $-$ R0$_{\rho=0.9}$, IID", "MixBiasedNineIid", "MixBiased", "Rninety", "iid"),
    (r"R0$_{\rho=0.9}$ + relabelled $-$ R0$_{\rho=0.9}$ + biased, saliency-reversed", "MixNaturalBiasedSal", "MixNatural", "MixBiased", "sal"),
    (r"R0$_{\rho=0.9}$ + relabelled $-$ R0$_{\rho=0.9}$ + biased, IID", "MixNaturalBiasedIid", "MixNatural", "MixBiased", "iid"),
    (r"R0$_{\rho=0.9}$ + paired $-$ R0$_{\rho=0.9}$ + relabelled, saliency-reversed", "MixSal", "MixPaired", "MixNatural", "sal"),
    (r"R0$_{\rho=0.9}$ + paired $-$ R0$_{\rho=0.9}$ + relabelled, IID", "MixIid", "MixPaired", "MixNatural", "iid"),
    (r"R0$_{\rho=0.9}$ + paired $-$ R0$_{\rho=0.9}$ + relabelled, attribute-OOD", "MixAttr", "MixPaired", "MixNatural", "attr"),
    (r"R0$_{\rho=0}$ with 146 ``farthest'' $-$ R0$_{\rho=0}$, saliency-reversed", "FarSal", "Far", "Rzero", "sal"),
    (r"R0$_{\rho=0}$ with 146 ``farthest'' $-$ R0$_{\rho=0}$, IID", "FarIid", "Far", "Rzero", "iid"),
    (r"R0$_{\rho=0.9}$ $-$ R0$_{\rho=0}$ with 146 ``farthest'', saliency-reversed", "NineFarSal", "Rninety", "Far", "sal"),
    (r"R0 with the command mix of $\rho=0.9$, no correlation $-$ R0$_{\rho=0}$, saliency-reversed", "MatchedZeroSal", "Rmatched", "Rzero", "sal"),
    (r"R0 with the command mix of $\rho=0.9$, no correlation $-$ R0$_{\rho=0}$, IID", "MatchedZeroIid", "Rmatched", "Rzero", "iid"),
    (r"R0 with the command mix of $\rho=0.9$, no correlation $-$ R0$_{\rho=0}$, attribute-OOD", "MatchedZeroAttr", "Rmatched", "Rzero", "attr"),
    (r"R0 with the command mix of $\rho=0.9$, no correlation $-$ R0$_{\rho=0}$ with 146 ``farthest'', saliency-reversed", "MatchedFarSal", "Rmatched", "Far", "sal"),
    (r"R0 with the command mix of $\rho=0.9$, no correlation $-$ R0$_{\rho=0}$ with 146 ``farthest'', IID", "MatchedFarIid", "Rmatched", "Far", "iid"),
    (r"R0$_{\rho=0.9}$ $-$ R0 with the command mix of $\rho=0.9$, no correlation, saliency-reversed", "NineMatchedSal", "Rninety", "Rmatched", "sal"),
    (r"R0$_{\rho=0.9}$ $-$ R0 with the command mix of $\rho=0.9$, no correlation, IID", "NineMatchedIid", "Rninety", "Rmatched", "iid"),
]
NOT_IN_TABLE = {"MixPaired", "MixNatural", "MixBiased", "Far", "Rmatched", "RoneNoShift", "RoneSyncShift", "RlateNoShift", "RlateSyncShift", "RlateWide", "RoneLrLo", "RoneLrHi", "RlateLrLo", "RlateLrHi"}
MODULAR = {"ModOracle", "ModDet", "ModFree"}                                        # shown in the table, but not part of the ranges over the pixel or fruit-table selectors          # exploratory arms shown in their own table (make_remedy_table.py)
FRESH_MAP = {"iid": "iidf", "sal": "salf", "attr": "attrf"}   # the same comparison on the fresh scenes (lane_select_fresh.sh)
COMPARE_BOTH = [  # comparisons on the sal-both split (no pilot counterpart)
    (r"R0$_{\rho=0.9}$ $-$ R0$_{\rho=0}$, both commands avoid the salient fruit", "ConeBoth", "Rninety", "Rzero"),
    (r"R0$_{\rho=0.97}$ $-$ R0$_{\rho=0}$, both commands avoid the salient fruit", "DialSevenBoth", "Rninetyseven", "Rzero"),
    (r"R0$_{\rho=0.99}$ $-$ R0$_{\rho=0}$, both commands avoid the salient fruit", "DialNineBoth", "Rninetynine", "Rzero"),
    (r"R0$_{\rho=0.9}$ + paired $-$ R0$_{\rho=0.9}$ + relabelled, both commands avoid the salient fruit", "MixBoth", "MixPaired", "MixNatural"),
]
N_PLANNED = 5                                              # the first rows of COMPARE are the planned C1-C3
SEED_SUFFIXES = ("", "_s1", "_s2", "_s3", "_s4")           # training seeds 0-4 (seeds 3 and 4 exist for the four arms of the planned comparisons only)


BASE = {k: base for k, base, _ in ARMS}


def per_seed(base, split, fams=None):
    """List of {(split, index): PTA outcome} over the available training seeds (optionally only the pairs of some command families)."""
    if split in DERIVED:
        split, dfams = DERIVED[split]
        fams = dfams if fams is None else tuple(f for f in fams if f in dfams)
    out = []
    for suf in SEED_SUFFIXES:
        path = f"results/eval/s_{base}{suf}__{split}.jsonl"
        if os.path.exists(path):
            recs = [json.loads(l) for l in open(path)]
            if fams is not None:
                recs = [r for r in recs if r["family"] in fams]
            out.append(stats.scene_outcomes(recs, "PTA_sel"))
    return out


def fmt(x):
    return f"{100*x:.1f}"


def sgn(x):
    """Signed difference in percentage points with a typographic minus (for LaTeX text and tables)."""
    return f"{100*x:+.1f}".replace("-", "$-$")


def welch(ma, mb):
    """Seed-level Welch t interval for the difference of the seed means of two arms (None if an arm has fewer than three seeds, where the interval says nothing, or no variance)."""
    if len(ma) < 3 or len(mb) < 3:
        return None
    va, vb = np.var(ma, ddof=1) / len(ma), np.var(mb, ddof=1) / len(mb)
    if va + vb == 0:
        return None
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / ((va ** 2 / (len(ma) - 1)) + (vb ** 2 / (len(mb) - 1)))
    t = sst.t.ppf(0.975, df)
    d = np.mean(ma) - np.mean(mb)
    return d - t * se, d + t * se


def main(n_boot):
    data = {(k, sp): per_seed(base, sp) for k, base, _ in ARMS for sp, _, _ in SPLITS}
    rel = {(k, sp): per_seed(base, sp, REL) for k, base, _ in ARMS for sp in ("iid", "sal")}
    md, tex, macros = [], [], []
    # number of training seeds of the groups of arms (IID split), for the text
    ns = lambda keys: min(len(data[(k, "iid")]) for k in keys)
    macros += [f"\\providecommand{{\\nsPlanned}}{{{ns(('Rzero', 'Rninety', 'Rone', 'Ronenu'))}}}\\providecommand{{\\nsDial}}{{{ns(('Rhalf', 'Rninetyseven', 'Rninetynine'))}}}"
               f"\\providecommand{{\\nsControls}}{{{ns(('Rsame', 'RzeroP'))}}}\\providecommand{{\\nsLate}}{{{ns(('Rlate', 'Rtoken'))}}}"
               f"\\providecommand{{\\nsRemedy}}{{{ns(('MixPaired', 'MixNatural', 'Far'))}}}"]
    # every macro the manuscript may use exists (as ??) even when its runs are missing, so that main.tex always compiles and a "??" in the PDF marks what is missing
    macros += [f"\\providecommand{{\\sm{k}{mname}}}{{??}}\\providecommand{{\\zs{k}{mname}}}{{??}}" for k, _, _ in ARMS for _, _, mname in SPLITS]
    macros += [f"\\providecommand{{\\sm{k}{mname}Rel}}{{??}}" for k, _, _ in ARMS for mname in ("Iid", "Sal")]
    macros += [f"\\providecommand{{\\{pre}{mname}}}{{??}}" for _, mname, _, _, _ in COMPARE for pre in ("dd", "dl", "dh", "aa", "rd", "rl", "rh", "tl", "th")]
    md.append("| arm | seeds | IID | IID rel. | Sal.-rev. | Sal.-rev. rel. | " + " | ".join(n for _, n, _ in SPLITS[2:]) + " |\n|---|---|" + "---|" * (len(SPLITS) + 2))
    mean_of = {}                                                # (arm key, split key) -> mean over seeds in percent
    for k, base, label in ARMS:
        cells, tcells = {}, {}
        n_seeds = max(len(data[(k, sp)]) for sp, _, _ in SPLITS)
        if n_seeds == 0:
            continue
        for sp, _, mname in SPLITS:
            ps = data[(k, sp)]
            if not ps:
                cells[sp] = "-"; tcells[sp] = "--"; continue
            means = [float(np.mean(list(d.values()))) for d in ps]
            m = float(np.mean(means))
            mean_of[(k, sp)] = 100 * m
            rng_txt = f" ({fmt(min(means))}--{fmt(max(means))})" if len(ps) > 1 else ""
            cells[sp] = f"{fmt(m)}{rng_txt}".replace("--", "-"); tcells[sp] = f"{fmt(m)}{rng_txt}"
            macros.append(f"\\renewcommand{{\\sm{k}{mname}}}{{{fmt(m)}}}\\renewcommand{{\\zs{k}{mname}}}{{{fmt(means[0])}}}")
        for sp, mname in (("iid", "Iid"), ("sal", "Sal")):
            ps = rel[(k, sp)]
            if not ps:
                cells[sp + "rel"] = "-"; tcells[sp + "rel"] = "--"; continue
            means = [float(np.mean(list(d.values()))) for d in ps]
            m = float(np.mean(means))
            rng_txt = f" ({fmt(min(means))}--{fmt(max(means))})" if len(ps) > 1 else ""
            cells[sp + "rel"] = f"{fmt(m)}{rng_txt}".replace("--", "-"); tcells[sp + "rel"] = f"{fmt(m)}{rng_txt}"
            macros.append(f"\\renewcommand{{\\sm{k}{mname}Rel}}{{{fmt(m)}}}")
        order = ["iid", "iidrel", "sal", "salrel", "occ", "occn", "dens", "lang", "attr", "dr", "iidn", "occa"]
        md.append(f"| {k} | {n_seeds} | " + " | ".join(cells[o] for o in order) + " |")
        # the table of the manuscript leaves out the language split (uninformative for encoders trained from scratch); it stays in the markdown and in the macros
        torder = ["iid", "iidrel", "sal", "salrel", "occn", "dens", "attr", "dr"]
        if k not in NOT_IN_TABLE:
            tex.append(f"{label} & {n_seeds} & " + " & ".join(tcells[o] for o in torder))
    # ranges over the arms of a group (pixel selectors, fruit-table selectors, all) of the seed means, and the loss against IID
    groups = {"Px": [k for k, _, _ in ARMS if not k.startswith("S") and k not in NOT_IN_TABLE and k not in MODULAR], "Ft": [k for k, _, _ in ARMS if k.startswith("S")],
              "All": [k for k, _, _ in ARMS if k not in NOT_IN_TABLE and k not in MODULAR]}
    for g, keys in groups.items():
        for sp, _, mname in SPLITS:
            vals = [mean_of[(k, sp)] for k in keys if (k, sp) in mean_of]
            b_sp = "iidn" if sp == "occn" else "iid"
            loss = [mean_of[(k, b_sp)] - mean_of[(k, sp)] for k in keys if (k, sp) in mean_of and (k, b_sp) in mean_of]
            if vals:
                macros.append(f"\\providecommand{{\\rg{g}Min{mname}}}{{??}}\\renewcommand{{\\rg{g}Min{mname}}}{{{min(vals):.1f}}}"
                              f"\\providecommand{{\\rg{g}Max{mname}}}{{??}}\\renewcommand{{\\rg{g}Max{mname}}}{{{max(vals):.1f}}}")
            if loss:
                macros.append(f"\\providecommand{{\\rg{g}LossMin{mname}}}{{??}}\\renewcommand{{\\rg{g}LossMin{mname}}}{{{min(loss):.0f}}}"
                              f"\\providecommand{{\\rg{g}LossMax{mname}}}{{??}}\\renewcommand{{\\rg{g}LossMax{mname}}}{{{max(loss):.0f}}}")
    print("\n".join(md))
    os.makedirs("../tables", exist_ok=True)
    try:                                                                # size of the occlusion sample and its any-of share (from one arm's records)
        occ = [json.loads(l) for l in open("results/eval/s_r1_film__occ.jsonl")]
        pairs = {(r["index"]): r["family"] for r in occ if r["which"] == "plus"}
        macros.append(f"\\providecommand{{\\occPairs}}{{{len(pairs)}}}\\providecommand{{\\occAnyN}}{{{sum(f == 'mat_any' for f in pairs.values())}}}"
                      f"\\providecommand{{\\occAnyShare}}{{{100*sum(f == 'mat_any' for f in pairs.values())/len(pairs):.0f}}}")
    except OSError:
        pass
    open("../tables/select_seeds.tex", "w", newline="\n").write(" \\\\\n".join(tex) + "\n")
    # comparisons: seed-and-scene cluster bootstrap on all pairs and on the relational pairs, and the seed-level Welch t interval on all pairs
    rows, mdc = [], ["", "| comparison | seeds (A/B) | A - B [95% bootstrap] | relational pairs [95% bootstrap] | seed-level t [95%] | per-seed A | per-seed B |", "|---|---|---|---|---|---|---|"]
    for i, (label, mname, ka, kb, sp) in enumerate(COMPARE):
        A, B = data[(ka, sp)], data[(kb, sp)]
        if not A or not B:
            continue
        d, lo, hi, ma, mb = stats.cluster_diff(A, B, n_boot=n_boot)
        RA, RB = rel.get((ka, sp)), rel.get((kb, sp))
        rtxt, rmd = "--", "-"
        if RA and RB:
            rd, rlo, rhi, _, _ = stats.cluster_diff(RA, RB, n_boot=n_boot)
            rtxt = f"{sgn(rd)} [{sgn(rlo)}, {sgn(rhi)}]"; rmd = f"{100*rd:+.1f} [{100*rlo:+.1f}, {100*rhi:+.1f}]"
            for pre, val in (("rd", rd), ("rl", rlo), ("rh", rhi)):
                macros.append(f"\\renewcommand{{\\{pre}{mname}}}{{{sgn(val)}}}")
        w = welch(ma, mb)
        ttxt, tmd = "--", "-"
        if w:
            ttxt = f"[{sgn(w[0])}, {sgn(w[1])}]"; tmd = f"[{100*w[0]:+.1f}, {100*w[1]:+.1f}]"
            macros.append(f"\\renewcommand{{\\tl{mname}}}{{{sgn(w[0])}}}\\renewcommand{{\\th{mname}}}{{{sgn(w[1])}}}")
        ftxt = "--"
        if sp in FRESH_MAP:                                              # the same comparison on the fresh scenes
            Af, Bf = per_seed(BASE[ka], FRESH_MAP[sp]), per_seed(BASE[kb], FRESH_MAP[sp])
            if Af and Bf:
                fd, flo, fhi, _, _ = stats.cluster_diff(Af, Bf, n_boot=n_boot)
                ftxt = f"{sgn(fd)} [{sgn(flo)}, {sgn(fhi)}] ({len(Af)}/{len(Bf)})"
        rows.append((i >= N_PLANNED, f"{label} & {len(A)}/{len(B)} & {sgn(d)} [{sgn(lo)}, {sgn(hi)}] & {rtxt} & {ttxt} & {ftxt}", mname))
        mdc.append(f"| {label} | {len(A)}/{len(B)} | {100*d:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] | {rmd} | {tmd} | {', '.join(fmt(x) for x in ma)} | {', '.join(fmt(x) for x in mb)} |")
        for pre, val in (("dd", d), ("dl", lo), ("dh", hi)):
            macros.append(f"\\renewcommand{{\\{pre}{mname}}}{{{sgn(val)}}}")
        macros.append(f"\\renewcommand{{\\aa{mname}}}{{{100*abs(d):.1f}}}")       # unsigned size of the difference
    for label, mname, ka, kb in COMPARE_BOTH:
        Af, Bf = per_seed(BASE[ka], "salb"), per_seed(BASE[kb], "salb")
        if Af and Bf:
            fd, flo, fhi, _, _ = stats.cluster_diff(Af, Bf, n_boot=n_boot)
            rows.append((True, f"{label} & -- & -- & -- & -- & {sgn(fd)} [{sgn(flo)}, {sgn(fhi)}] ({len(Af)}/{len(Bf)})", mname))
    # between-split contrasts on independent scene sets: is an arm lower on the saliency-reversed than on the IID scenes (second clause of C1), and is the dial's cost larger there
    macros += [f"\\providecommand{{\\{p}{k}}}{{??}}" for k in ("Ninety", "Zero") for p in ("ssD", "ssL", "ssH")] + [f"\\providecommand{{\\did{p}}}{{??}}" for p in ("D", "L", "H")]
    for key, k in (("Rninety", "Ninety"), ("Rzero", "Zero")):
        A, B = data[(key, "sal")], data[(key, "iid")]
        if A and B:
            d, lo, hi = stats.cluster_diff_indep(A, B, n_boot=n_boot)
            macros.append(f"\\renewcommand{{\\ssD{k}}}{{{sgn(d)}}}\\renewcommand{{\\ssL{k}}}{{{sgn(lo)}}}\\renewcommand{{\\ssH{k}}}{{{sgn(hi)}}}")
            mdc.append(f"| {key}: saliency-reversed minus IID (independent scenes) | | {100*d:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] | | | | |")
            arm_label = r"R0$_{\rho=0.9}$" if key == "Rninety" else r"R0$_{\rho=0}$"
            Af, Bf = per_seed(BASE[key], "salf"), per_seed(BASE[key], "iidf")
            ftxt = "--"
            if Af and Bf:
                fd, flo, fhi = stats.cluster_diff_indep(Af, Bf, n_boot=n_boot)
                ftxt = f"{sgn(fd)} [{sgn(flo)}, {sgn(fhi)}] ({len(Af)})"
            rows.append((None, f"{arm_label}: saliency-reversed $-$ IID (independent scenes) & {len(A)} & {sgn(d)} [{sgn(lo)}, {sgn(hi)}] & -- & -- & {ftxt}", "ss"))
    if all(data[(k, s_)] for k in ("Rninety", "Rzero") for s_ in ("sal", "iid")):
        d, lo, hi = stats.cluster_did(data[("Rninety", "sal")], data[("Rninety", "iid")], data[("Rzero", "sal")], data[("Rzero", "iid")], n_boot=n_boot)
        macros.append(f"\\renewcommand{{\\didD}}{{{sgn(d)}}}\\renewcommand{{\\didL}}{{{sgn(lo)}}}\\renewcommand{{\\didH}}{{{sgn(hi)}}}")
        mdc.append(f"| difference in differences of the dial's cost (sal-rev minus IID) | | {100*d:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] | | | | |")
        ftxt = "--"
        F = {(k, s_): per_seed(BASE[k], s_) for k in ("Rninety", "Rzero") for s_ in ("salf", "iidf")}
        if all(F.values()):
            fd, flo, fhi = stats.cluster_did(F[("Rninety", "salf")], F[("Rninety", "iidf")], F[("Rzero", "salf")], F[("Rzero", "iidf")], n_boot=n_boot)
            ftxt = f"{sgn(fd)} [{sgn(flo)}, {sgn(fhi)}] ({len(F[('Rninety', 'salf')])}/{len(F[('Rzero', 'salf')])})"
        rows.append((None, f"Cost of the dial, saliency-reversed $-$ IID (difference in differences) & {len(data[('Rninety', 'sal')])}/{len(data[('Rzero', 'sal')])} & {sgn(d)} [{sgn(lo)}, {sgn(hi)}] & -- & -- & {ftxt}", "did"))
    print("\n".join(mdc))
    planned = [r for e, r, _ in rows if e is False]
    contrasts = [r for e, r, _ in rows if e is None]
    expl = [r for e, r, _ in rows if e is True]
    iv = re.compile(r"\[[^\]]*,[^\]]*\]")                     # the number of comparisons and of intervals that the two comparison tables list (quoted in the text)
    macros.append(f"\\providecommand{{\\nCmp}}{{{len(planned) + len(contrasts) + len(expl)}}}\\providecommand{{\\nCmpInt}}{{{sum(len(iv.findall(r)) for r in planned + contrasts + expl)}}}")
    # the exploratory comparisons by topic, in three tables so that none is too long to read: (A) dial, pairing, composition and their controls; (B) conditioning, its controls and the modular
    # pipeline; (C) remedy under bias, the coverage and mix-matched controls of the dial and the biased control
    B_TOPIC = ("RoneLate", "RoneToken", "RoneMod", "RlateMod", "RtokenMod", "RoneNoShift", "RoneSyncShift", "RlateNoShift", "RlateSyncShift")
    C_TOPIC = ("MixPaired", "MixNatural", "MixBiased", "MixSal", "MixIid", "MixAttr", "MixBoth", "Far", "NineFar", "Matched", "NineMatched")
    expl_b = [r for e, r, m in rows if e is True and m.startswith(B_TOPIC)]
    expl_c = [r for e, r, m in rows if e is True and m.startswith(C_TOPIC)]
    expl_a = [r for e, r, m in rows if e is True and not m.startswith(B_TOPIC + C_TOPIC)]
    body = " \\\\\n".join(planned) + (" \\\\\n" if contrasts else "\n")
    if contrasts:
        body += "\\midrule\n\\multicolumn{6}{@{}l}{\\emph{Between scene sets (post hoc)}}\\\\\n" + " \\\\\n".join(contrasts) + "\n"
    open("../tables/select_compare_expl.tex", "w", newline="\n").write(" \\\\\n".join(expl) + "\n")
    open("../tables/select_compare_expl_a.tex", "w", newline="\n").write(" \\\\\n".join(expl_a) + "\n")
    open("../tables/select_compare_expl_b.tex", "w", newline="\n").write(" \\\\\n".join(expl_b) + "\n")
    open("../tables/select_compare_expl_c.tex", "w", newline="\n").write(" \\\\\n".join(expl_c) + "\n")
    open("../tables/select_compare.tex", "w", newline="\n").write(body)
    open("../tables/select_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open("results/eval/select_seeds.md", "w", encoding="utf-8", newline="\n").write("\n".join(md + mdc) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=4000)
    main(ap.parse_args().boot)
