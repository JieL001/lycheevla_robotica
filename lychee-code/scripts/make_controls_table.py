"""Robustness controls of the FiLM-versus-late-fusion result and of the training recipe: image shifts, capacity, learning rate.
   python scripts/make_controls_table.py
Rows (PTA in percent, mean over the training seeds and their range; the seeds of the reference rows are limited to the seeds of the control):
   R1 FiLM (default: image shifted by up to 4 px, labels not shifted) | no image shifts | synchronised shifts (the fruit centres shift with the image)
   R1 late fusion (0.55 M) | the same without image shifts | with synchronised shifts | late fusion with a wider scoring head (0.60 M, as large as FiLM) | R1 command token (0.55 M)
   FiLM and late fusion at learning rates 3e-4 and 3e-3 (one seed; the default 1e-3 is the reference; nothing is selected among them)
Columns: IID, saliency-reversed and attribute-OOD (pilot scenes), attribute-OOD on the fresh scenes.  Writes ../tables/controls.tex, ../tables/controls_macros.tex
(\\ct<Row><Set> means, \\ct<Row>Seeds) and results/eval/controls.md."""
import json, os, re, sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from select_seeds import SEED_SUFFIXES

ROWS = [  # macro key, file base, label, seeds used (None = all available up to the number of control seeds), reference seeds count
    ("Film", "r1_film", "R1, FiLM (default: shifts of the image, not of the labels)", 3),
    ("NoShift", "r1_film_shiftnone", "R1, FiLM, no image shifts", 3),
    ("SyncShift", "r1_film_shiftsync", "R1, FiLM, synchronised shifts (labels shifted with the image)", 3),
    ("Late", "r1_late", "R1, late fusion (0.55 M)", 3),
    ("LateNoShift", "r1_late_shiftnone", "R1, late fusion, no image shifts", 3),
    ("LateSyncShift", "r1_late_shiftsync", "R1, late fusion, synchronised shifts (labels shifted with the image)", 3),
    ("LateWide", "r1_late_wide", "R1, late fusion, wider scoring head (0.60 M)", 3),
    ("Token", "r1_token", "R1, command token (0.55 M)", 3),
    ("FilmLrLo", "r1_film_lr3e-4", r"R1, FiLM, learning rate $3\cdot10^{-4}$ (seed 0)", 1),
    ("FilmLrHi", "r1_film_lr3e-3", r"R1, FiLM, learning rate $3\cdot10^{-3}$ (seed 0)", 1),
    ("FilmLr", "r1_film", r"R1, FiLM, learning rate $10^{-3}$ (default, seed 0)", 1),
    ("LateLrLo", "r1_late_lr3e-4", r"R1, late fusion, learning rate $3\cdot10^{-4}$ (seed 0)", 1),
    ("LateLrHi", "r1_late_lr3e-3", r"R1, late fusion, learning rate $3\cdot10^{-3}$ (seed 0)", 1),
    ("LateLr", "r1_late", r"R1, late fusion, learning rate $10^{-3}$ (default, seed 0)", 1),
]
SETS = [("iid", "Iid"), ("sal", "Sal"), ("attr", "Attr"), ("attrf", "AttrF")]


def pta_per_seed(base, split, n_max):
    out = []
    for suf in SEED_SUFFIXES[:n_max]:
        p = f"results/eval/s_{base}{suf}__{split}.jsonl"
        if not os.path.exists(p):
            continue
        pairs = {}
        for r in map(json.loads, open(p)):
            pairs.setdefault(r["index"], {})[r["which"]] = r
        out.append(float(np.mean([all(v[w]["target_correct"] for w in ("plus", "minus")) for v in pairs.values() if "minus" in v])))
    return out


def main():
    rows, md, macros = [], ["| row | seeds | IID | Sal.-rev. | Attr. | Attr. (fresh) |", "|---|---|---|---|---|---|"], []
    macros += [f"\\providecommand{{\\ct{key}{tag}}}{{??}}" for key, _, _, _ in ROWS for _, tag in SETS] + [f"\\providecommand{{\\ct{key}Seeds}}{{??}}" for key, _, _, _ in ROWS]   # ?? until a row has records
    for key, base, label, nmax in ROWS:
        cells, n_seeds = [], 0
        for sp, tag in SETS:
            v = pta_per_seed(base, sp, nmax)
            n_seeds = max(n_seeds, len(v))
            if not v:
                cells.append("--"); continue
            cells.append(f"{100*np.mean(v):.1f}" + (f" ({100*min(v):.1f}--{100*max(v):.1f})" if len(v) > 1 else ""))
            macros.append(f"\\renewcommand{{\\ct{key}{tag}}}{{{100*np.mean(v):.1f}}}")
        if n_seeds == 0:
            continue
        macros.append(f"\\renewcommand{{\\ct{key}Seeds}}{{{n_seeds}}}")
        rows.append(f"{label} & {n_seeds} & " + " & ".join(cells))
        md.append(f"| {key} | {n_seeds} | " + " | ".join(cells) + " |")
    # the weakest FiLM row and the strongest late-fusion row on the held-out combinations (pilot and fresh scenes): does any variant close the gap?
    def vals(keys, tag):
        return [float(m.group(1)) for k in keys for m in [re.search(rf"\\renewcommand\{{\\ct{k}{tag}\}}\{{([-0-9.]+)\}}", "\n".join(macros))] if m]
    film_rows, late_rows = ("Film", "NoShift", "SyncShift", "FilmLr", "FilmLrLo", "FilmLrHi"), ("Late", "LateNoShift", "LateSyncShift", "LateWide", "LateLr", "LateLrLo", "LateLrHi")
    fm, lt, fmf, ltf = vals(film_rows, "Attr"), vals(late_rows, "Attr"), vals(film_rows, "AttrF"), vals(late_rows, "AttrF")
    if fm and lt and fmf and ltf:
        macros += [f"\\providecommand{{\\ctFilmMinAttr}}{{{min(fm):.1f}}}\\providecommand{{\\ctLateMaxAttr}}{{{max(lt):.1f}}}\\providecommand{{\\ctGapAttr}}{{{int(min(fm) - max(lt))}}}",
                   f"\\providecommand{{\\ctFilmMinAttrF}}{{{min(fmf):.1f}}}\\providecommand{{\\ctLateMaxAttrF}}{{{max(ltf):.1f}}}\\providecommand{{\\ctGapAttrF}}{{{int(min(fmf) - max(ltf))}}}"]
    else:
        macros += ["\\providecommand{\\ctFilmMinAttr}{??}\\providecommand{\\ctLateMaxAttr}{??}\\providecommand{\\ctGapAttr}{??}\\providecommand{\\ctFilmMinAttrF}{??}\\providecommand{\\ctLateMaxAttrF}{??}\\providecommand{\\ctGapAttrF}{??}"]
    os.makedirs("../tables", exist_ok=True)
    open("../tables/controls.tex", "w", newline="\n").write(" \\\\\n".join(rows) + "\n")
    open("../tables/controls_macros.tex", "w", newline="\n").write("\n".join(macros) + "\n")
    open("results/eval/controls.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
