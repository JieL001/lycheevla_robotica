# Pre-registration v0.1 — LycheeHarvest-Sim controlled comparison

Written 2026-09-29, **before any learned-policy result on the benchmark has been inspected** (only scripted reference policies and the expert have been evaluated; see `results/eval/reference_policies_iid.md`). This file is frozen — its SHA-256 is recorded in `prereg/FROZEN_HASHES.txt` together with the hashes of the benchmark source files — before the first training run of any main-study arm. After freezing, changes are made only by appending to the *Deviation log* at the end; the frozen text is never edited. A small from-scratch behaviour-cloning pilot (R0/R1, concat/FiLM) is run before the freeze and is reported as exploratory.

## 1. Question
In language-conditioned selective harvesting, (Q1) how large is the deficit of fine-tuned VLA policies in *choosing* the commanded fruit under same-scene counterfactual commands; (Q2) which remedies reduce it; (Q3) do action-free real lychee images help a policy trained on simulated actions under a real-appearance shift, beyond what extra simulated grounding supervision would give.

## 2. Benchmark, splits, sizes
- Environment `LycheeProto-v0` (to be renamed `LycheeHarvest-v0` at release), code in `lychee-code/lychee/`; control mode `pd_ee_delta_pose` at 20 Hz; cameras 256×256 (main) and 256×256 (wrist).
- Splits and their seed bases are those of `lychee/splits.py::SPLITS`: `iid` 20 000 000, `occ_ood` 30 000 000, `density_ood` 40 000 000, `lang_ood` 50 000 000, `attr_ood` 60 000 000, `saliency_rev` 70 000 000, `visual_dr_ood` 80 000 000 (RA-OOD: 90 000 000, to be added). Test scene *i* of a split is `sample_config(split, i)`, `i = 0 … 299`; each is evaluated with both commands of its pair.
- Training data come only from split `train` (seed base 0; indices below 10 000 for paired data, 10 000 and above for unpaired data) and validation from `val`. Test seeds are never used for training, tuning or model selection.
- The visual-DR *training* range (`sample_dr(seed, "train")`) is enabled for all main-study training data; the pilot data are generated with DR off.

## 3. Arms (all under one demonstration budget, augmentation, optimisation budget and evaluation scenes)
R0 plain imitation; R1 paired demonstrations (equal number of demonstrations as R0); R2 language modulation (FiLM); R3 inference-time counterfactual guidance; R4 attention supervision; R5 grounded injection; R6 = R5 + action-free real images; R7 = R6 + one-sided prototype alignment; M1 detector→mask→policy; M2 ground-truth-mask oracle. Control arms for the real-image question: B1 extra simulated grounding supervision of equal amount, B2 generic non-lychee box data of equal amount, B3 permuted maturity labels, B4 permuted boxes, B5 stronger domain randomisation. Ablations of R5: supervision without injection; injection without supervision; injection of grounded tokens only / command embedding only / both; parameter-matched adapter without grounding; closed-gate and swapped-token interventions at test time; Jensen–Shannon separation term; depth embedding.
Backbones: VLA-Adapter (all arms, ≥3 seeds) and OpenVLA-OFT (R0, R1, R5 and the diagnostic; 1 seed unless compute allows). Small policies: BC-Transformer, Diffusion Policy, ACT (3 seeds).

## 4. Outcomes
Primary: **PTA** — the share of test pairs in which the first fruit whose stem breaks is a member of the target set for *both* commands (`PTA_sel` in the code). Secondary: PTA-approach (the first fruit the fingertips come within 5 cm of), per-command accuracy (TSA), success (correct fruit in the basket, nothing else detached), wrong-target rate, conditional success given a correct selection, contact rate with non-target fruit (reported relative to the expert's), and the language probes (blank, gibberish, swapped command; language sensitivity = TSA(normal) − TSA(swap)).

## 5. Hypotheses and decision rules (thresholds fixed now)
- **H1 headroom.** R0 on VLA-Adapter has PTA ≤ 0.80 on `iid`, or PTA on some other split at least 0.15 below its `iid` value; and its conditional success given a correct selection is ≥ 0.70.
- **H2 paired data.** PTA(R1) − PTA(R0) ≥ 0.10 on `iid`, 95% paired-bootstrap interval excluding 0.
- **H3 grounding.** PTA(R5) − PTA(R1) ≥ 0.05 on the pooled shifted splits (`occ_ood`, `density_ood`, `lang_ood`, `attr_ood`, `saliency_rev`), interval excluding 0; and closing the gate or swapping grounded tokens between the two commands of a pair reduces PTA(R5) by at least half of that gain.
- **H4 real images.** PTA(R6) − PTA(R5) ≥ 0.10 on RA-OOD (interval excluding 0), the gain is larger on RA-OOD than on `iid`, and it increases monotonically over 0 / 1% / 10% / 100% of the real images.
- **H5 specificity.** The R6 gain over R5 on RA-OOD exceeds every control gain B1–B5 (paired bootstrap, interval of the difference excluding 0).
- **H6 alignment.** Wrong-target rate(R7) < wrong-target rate(R6) on RA-OOD, interval excluding 0.
- **H7 end-to-end vs modular.** PTA(R6) ≥ PTA(M1) − 0.03 on RA-OOD and on `occ_ood`, at comparable parameter count and latency.
Falsification: if R2 (FiLM) is within 0.03 of R5, the architectural claim for R5 is dropped; if attention supervision without injection is within 0.03 of R5, injection is dropped; if R1 is within 0.03 of R5, only the paired data is kept; if M1 or M2 exceeds R6 by more than 0.03, the contribution is attributed to perception; if command-only injection is within 0.03 of R5, the gain is attributed to the language shortcut. Negative results are reported as such.

## 6. Statistics
The unit of analysis is the scene (pair). Intervals: 95% percentile bootstrap over scenes (10 000 resamples; 2 000 in exploratory runs). Comparisons between arms use the paired bootstrap on scene-level differences and McNemar's exact test on the pair-level outcomes; the seven hypotheses are Holm-corrected together. Seeds are treated as a cluster: for arms with several training seeds the scene-level outcome is averaged over seeds before resampling, and the seed-to-seed range is reported separately. Effect sizes are reported as differences in PTA (percentage points) with intervals.

## 7. Data handling rules
- No episode is dropped after the fact. A simulator error is re-run once with the same seed; a second failure counts as a failed episode.
- Real images: split by acquisition date and scene (never by image); augmented copies stay with their source image; photographs used to build RA-OOD (`real-render`) are excluded from all training, tuning and prompt design.
- Hyper-parameters are tuned on `val` only. Checkpoints are chosen by the last step of a fixed schedule, not by test performance.
- Expert demonstrations that fail (no harvest) are discarded, and a pair is kept only if both members succeed.

## 8. Known limitations fixed in advance
Simulation only; fruit radius 3.0 cm (≈1.7× real); planar non-colliding leaves; a spring-damper stem with a geometric break rule; one fixed-base arm and fixed cameras; expert demonstrations from one scripted expert (contact with non-targets in 16% / 41% of episodes at 3–10 / 11–16 fruit); RA-OOD is an image-space appearance transfer, not real light transport.

## Deviation log (append only)
- 2026-09-29 — v0.1 written. Environment features added after the first scripted evaluations and before this file: `mat_any` command family, `approach_correct` outcome, visual-DR ranges (`off` in the pilot). No learned-policy result inspected.
