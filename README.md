# LycheeHarvest-Sim: code, records, checkpoints and generated tables (anonymised package for reviewers)

This archive accompanies the manuscript "Which Lychee? A Counterfactual Benchmark and Controlled Study of Command-Conditioned Target Selection in Simulated Harvesting".
It contains no author names or contact details.

## Contents

| path | what it is |
|---|---|
| `lychee-code/lychee/` | the benchmark: procedural scenes with analytic occlusion (`layout.py`), command language and target sets (`lang.py`), splits and the bias dial (`splits.py`), the ManiSkill3 environment with the virtual-stem harvest model (`env_proto.py`), the scripted expert (`expert.py`), the selection network (`select.py`), the fruit detector and the modular baseline (`detector.py`, `modular.py`), the evaluation kit with PTA and its companions (`evalkit.py`), statistics (`stats.py`), data paths (`paths.py`) |
| `lychee-code/scripts/` | rendering of selection-track data (`render_select.py`), training (`train_select.py`, `train_detector.py`, `make_matched_dataset.py`), offline evaluation (`eval_select.py`, `eval_modular.py`, `verify_evaluation.py`), scripted reference policies (`reference_offline.py`), table and figure generators, and the background pipeline in `scripts/queue/` |
| `lychee-code/tests/` | unit tests (splits and the bias dial, metrics, statistics, the selection network, the modular baseline) |
| `lychee-code/prereg/` | the written plan (planned comparisons, thresholds) and its deviation log |
| `lychee-code/results/` | one JSON-lines record per (scene, command) for every evaluated arm (`results/eval/s_<arm>__<split>.jsonl`), the offline reference policies (`results/eval_ref_offline/`), the scripted expert on 100 random scenes with 3-10 and 11-16 fruit (`results/expert/`, `scripts/run_expert_batch.py`; the `*_earlier_code.jsonl` files are the batches of an earlier version of the environment, kept for the record), summaries |
| `lychee-code/data_meta/` | fruit tables, commands and target masks of the evaluation scenes (no images), which the table scripts read |
| `checkpoints/` | the trained networks behind the main comparisons, the controls and the detectors of the modular baseline (`CHECKPOINTS.md`: names and SHA-256; about 2 MB each) |
| `tables/`, `figs/` | the generated LaTeX table bodies, macro files and figures used in the manuscript |

## Benchmark versions

All data and results of the manuscript use benchmark **v1.0** (`lychee/splits.py`: `SPLITS[...]`). Version **v1.1** (`SPLITS["<name>@v11"]`, no data rendered for it) differs in three rules: ordinal commands are
issued only if every fruit of the commanded maturity up to position k is at least 0.2 visible; the target visibility of the training and IID distributions is at least 0.5 (a guard band to the occlusion split);
and **any-of commands** (`lang.target_set(..., any_rule="v1.1")`) accept every fruit of the named maturity that is visible enough, and are issued on the occlusion split only if every fruit of that maturity is
occluded. In v1.0 the any-of command on the occlusion split accepts only fruit with a visible fraction of 0.2-0.4, which the command does not state; that evaluation is **deprecated** as a measure of occlusion
robustness (the manuscript reports the occlusion split without its any-of pairs, and `scripts/occ_anyof.py` re-scores the recorded any-of selections under the plain reading). New work should use the v1.1 splits.

## Reproducing the numbers of the manuscript

`REPRODUCE.md` describes three levels. Level 1 needs neither the simulator nor a GPU (numpy, scipy and matplotlib): every table, macro file and figure of the manuscript is
regenerated from `results/eval/*.jsonl` and `data_meta/`, and `verify_reproduction.py` compares the result with the shipped files.

    cd lychee-code
    bash scripts/regenerate_tables.sh         # every table, macro file and figure of the manuscript, 4,000 bootstrap resamples
    python scripts/verify_reproduction.py     # regenerates in a temporary copy and compares file by file

The script calls the generators in `scripts/` in the order listed in `regenerate_tables.sh` (arm x split table with the planned and exploratory comparisons and their seed-and-scene
bootstrap, family tables, coverage and mix-matched controls, cue readout, benchmark-v1.1 sensitivity, dial and scaling tables and figures, remedy table, closed-loop agreement, fresh scenes,
end-to-end counts, compute summary, held-out-combination accuracy, family-mix standardisation, modular baseline, controls, snap radius, the any-of pairs of the occlusion split under two readings of
the command, and finally `check_consistency.py`, which checks the relations between the numbers themselves: PTA against the per-command accuracies and error shares, the modular error table against the
modular PTA table, and the macros of independent scripts against the raw records); `check_manuscript.py` then checks the compiled PDF
(unresolved references, TODO markers, abstract length). `reference_offline.py` (scripted reference policies) is run separately. Records of arm `X` with seed `k` are
`results/eval/s_X_sk__<split>.jsonl` (seed 0 has no suffix); splits `iid`, `sal`, `occ`, `dens`, `lang`, `attr`, `dr` are the pilot evaluation scenes, `iidf`, `salf`, `salb`, `attrf` the fresh
ones (`salb`: both commands of a pair avoid the most salient fruit). `scripts/eval_select_batch.py` evaluates many checkpoints on many sets in one process and writes byte-identical records
to `eval_select.py`.

Level 2 re-renders an evaluation set from its split definition and seed range and evaluates a shipped checkpoint on it (`scripts/verify_evaluation.py`, a CPU is enough); level 3 re-trains
an arm: it needs the rendered training set (`render_select.py`, about 1 GB per set of 8,000 scenes; CPU rendering with ManiSkill3) and a CUDA GPU (`train_select.py`, about 10-20 minutes per
arm on an 8 GB laptop GPU; run at most two GPU jobs at once on a 16 GB machine). Paths in the scripts point to `D:/lychee_data`; set the environment variable `LYCHEE_DATA` (and `DATA` in
`scripts/queue/common.sh`) if you use another location. The rendered sets and the demonstration shards of the end-to-end track are not included.

## Environment used

Windows 11, CPU simulation and rendering with ManiSkill 3.0.1 (SAPIEN 3.0.3), Python 3.11 (numpy 2.4, gymnasium 1.3, scipy 1.17);
training with PyTorch 2.10 (CUDA 12.8), Python 3.12, one laptop GPU. The Windows wheels of ManiSkill lack the default inverse-kinematics solver;
`lychee/win_ik.py` patches it (damped least squares), and the normalised rotation action has an inverted sign relative to the ManiSkill convention
(see the manuscript, Section 7). The stack has not been verified on Linux or with GPU simulation.

## Licence

MIT (see `LICENSE`). The ManiSkill 3 simulator, PyTorch and the other dependencies keep their own licences; no third-party data, weights or images are included in this repository.
