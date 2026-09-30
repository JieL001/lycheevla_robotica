# Reproducing the numbers, tables and figures of the paper

There are three levels. Level 1 needs neither the simulator nor a GPU and checks every number of every table of the paper; level 2 re-evaluates the shipped checkpoints on re-rendered
evaluation scenes; level 3 re-trains.

## 1. Tables, macros and figures from the released evaluation records (minutes, any OS with Python 3.11)

```bash
git clone <this repository> && cd <repository>
git checkout <tag or commit named in the Data Availability Statement of the paper>
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # numpy, scipy, matplotlib (versions used for the paper)
cd lychee-code
bash scripts/regenerate_tables.sh                      # 4,000 bootstrap resamples; writes ../tables/*.tex and ../figs/*
python scripts/verify_reproduction.py                  # regenerates in a temporary copy and compares every table and macro file byte for byte with the shipped ones
```

`regenerate_tables.sh` reads `lychee-code/results/eval/*.jsonl` (one record per scene and command for every evaluated arm, seed and split), the fruit tables and command lists of the evaluation
scenes (`lychee-code/data_meta/`, without images) and a few JSON summaries of the training sets (`results/bias_dial_measured.json`, `results/compute_summary.json`), and writes the LaTeX table bodies
and the macro files (`\smRoneIid`, ...) that the manuscript inputs, so that no number of a table or of the text is typed by hand. Set `PY=/path/to/python` if `python` is not the right
interpreter. `verify_reproduction.py` prints the result per file (`DIFFERS` or nothing for an identical file; the summary line counts them) and lists any difference; the figure PDFs embed
a creation time and are only checked for existence. Four files come from scripts that need the simulator or rendered images (`reference_*.tex`, `visibility_macros.tex`,
`modular_depth_macros.tex`; levels 2 and 3) and are shipped as generated. Without the rendered data the scripts fall back on the shipped summaries; the paths of the data are set by the
environment variable `LYCHEE_DATA` (default `D:/lychee_data`).

Byte-identical files do not show that the numbers in them agree with each other, so `regenerate_tables.sh` ends with `python scripts/check_consistency.py` (it also runs inside `verify_reproduction.py`; exit status 1
and a list of violations if a relation fails). It checks, on all evaluation records (every index has exactly one plus and one minus command of the same family; a correct command has selected a fruit; PTA lies between the
Frechet bounds of the two per-command accuracies, `max(0, TSA+ + TSA- - 1) <= PTA <= min(TSA+, TSA-)`; all files of an evaluation set contain the same commands), that the stored error counts of the modular baseline
equal those recomputed from its records, that the modular error table (correct / no target / wrong fruit, mutually exclusive) and the modular PTA table satisfy `PTA >= 1 - 2e` for both rule versions, that the seed-0 table
satisfies `PTA >= 2 TSA - 100` and `PTA + collapse <= 100`, and that the macros of independent scripts (`select_seeds.py`, `fresh_summary.py`, `make_controls_table.py`, `modular_summary.py`, `occ_anyof.py`) equal the PTA
recomputed from the records. `scripts/occ_anyof.py` re-scores the recorded selections of the any-of pairs of the occlusion split under two readings of the command (it asserts that the stored v1.0 targets equal those
recomputed from the shipped fruit tables).

The oracle of the modular baseline (the parser and the benchmark's rules applied to the true fruit table) needs no images either:

```bash
python scripts/eval_modular.py --oracle 1 --sets iid:data_meta/select_eval/iid_600 --outdir /tmp/oracle      # PTA 100 % on the IID split
```

The fruit-table selectors (`checkpoints/sym_*.pt`; they read a table of the fruit, not pixels) need no images either, and their evaluation on the shipped scene metadata reproduces the shipped
records exactly, on a CPU, in seconds:

```bash
pip install -r requirements_train.txt                  # only PyTorch is needed for this step (a CPU build is enough)
python scripts/verify_evaluation.py --ckpt ../checkpoints/sym_r1.pt --arm sym_r1 --split iid --data data_meta/select_eval/iid_600      # REPRODUCED, PTA 99.8 %
```

## 2. Evaluation of the shipped checkpoints on re-rendered scenes (CPU is enough; about five minutes for the 600 IID scenes with one simulator worker)

`checkpoints/` holds the networks of the planned comparisons (five seeds), the conditioning variants and the controls of the review (three seeds each) and the detectors of the modular baseline
(`CHECKPOINTS.md`: names and SHA-256). The evaluation scenes are not part of the repository (70-100 MB per set with images); `scripts/render_select.py` re-creates a set from the split
definitions and seed ranges of `lychee/splits.py` (CPU rendering with ManiSkill 3, about 4-8 scenes per second, about 2 GB of memory per simulator worker):

```bash
pip install -r requirements_train.txt                  # ManiSkill 3, PyTorch (a CPU build is enough to evaluate)
cd lychee-code
python scripts/verify_evaluation.py --ckpt ../checkpoints/r1_film.pt --arm r1_film --split iid --data ./scratch/iid_600
```

`verify_evaluation.py` renders the set if it does not exist, evaluates the checkpoint and compares command text, target sets and the selected fruit of every (scene, command) with
`results/eval/s_r1_film__iid.jsonl`; it prints `REPRODUCED` if the scenes and commands are identical and PTA agrees within one point (for `r1_film` on the IID split, re-rendered and evaluated on a CPU: all 1,200 command texts, target sets and selected fruit identical, PTA 97.3 % = 97.3 %). The network selection can change on a handful of commands if
the rendered pixels differ in the last bits (another CPU or another PyTorch build). `--split` takes `iid, sal, occ, dens, lang, attr, dr, iidf, salf, salb, attrf` (the sets of the paper);
`scripts/eval_select.py` and `scripts/eval_select_batch.py` evaluate many checkpoints on many sets, `scripts/eval_modular.py` the modular baseline. Unit tests: `pytest tests -q` (the tests that
need rendered scenes are skipped when they are absent).

## 3. Re-training (CUDA GPU, about 10-20 minutes per arm on an 8 GB laptop GPU)

`scripts/train_select.py` trains a selection network on a rendered training set (`scripts/render_select.py`), `scripts/train_detector.py` the fruit detector of the modular baseline and
`scripts/make_matched_dataset.py` builds the mix-matched control set from the rendered natural and biased sets. The queue scripts in `scripts/queue/` list every arm, seed and evaluation of the
paper in the order in which they were run (`common.sh` holds the paths: edit `DATA` and the interpreters). The environment used for the paper is described in `requirements_train.txt` and in the
paper (Windows 11, CPU simulation and rendering; the stack has not been verified on Linux or with GPU simulation, and the numbers may differ there).

## What is where

| path | content |
|---|---|
| `lychee-code/lychee/` | benchmark (scenes, command language, splits, environment, expert, selection network, detector, modular baseline, evaluation kit, statistics) |
| `lychee-code/scripts/` | rendering, training, evaluation, table and figure generators, queue lanes, checks |
| `lychee-code/results/eval/` | evaluation records of every arm (JSON lines) and summaries |
| `lychee-code/data_meta/` | fruit tables, commands and target masks of the evaluation scenes (no images) |
| `lychee-code/prereg/` | the written plan (planned comparisons and thresholds) and its deviation log |
| `checkpoints/` | the trained networks of the main comparisons, the controls and the detectors (`CHECKPOINTS.md`) |
| `tables/`, `figs/` | the generated table bodies, macro files and figures used in the paper |
