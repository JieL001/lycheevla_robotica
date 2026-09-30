#!/bin/bash
# Regenerate every table, macro file and figure of the manuscript that depends on the selection-track result files (results/eval/s_*.jsonl and sel_*.jsonl).
# Run from lychee-code/ after all evaluations have finished; the order matters only for the files that later scripts read (none of them reads another script's output).
#   bash scripts/regenerate_tables.sh            # default: 4,000 bootstrap resamples
set -e
PY=${PY:-python}
BOOT=${BOOT:-4000}
$PY scripts/select_seeds.py --boot $BOOT            # Table 6, Table 8, select_macros.tex
$PY scripts/make_select_table.py                    # Appendix Table (seed 0)
$PY scripts/make_select_family_table.py --split iid # Appendix Table (families, IID)
$PY scripts/coverage_share.py --boot $BOOT          # coverage share of the dial cost
$PY scripts/make_family_seeds_table.py --split attr # Table 17 (families, attribute-OOD, seeds)
$PY scripts/error_analysis.py                       # Table 7, error_macros.tex
$PY scripts/v11_filter.py                           # benchmark-v1.1 sensitivity
$PY scripts/bias_dial_measured.py                   # dial_macros.tex (achieved shares, command mixes, distinct labels)
$PY scripts/make_scaling_tables.py                  # scaling and binding tables
$PY scripts/make_labels_figure.py                   # Figure: PTA against distinct labels
$PY scripts/make_dial_figure.py                     # Figure: bias dial
$PY scripts/make_composition_figure.py              # Figure: composition on held-out combinations
$PY scripts/make_remedy_table.py                    # remedy under bias and coverage control
$PY scripts/closed_loop_agreement.py                # closed-loop agreement table
$PY scripts/fresh_summary.py --boot $BOOT           # fresh scenes and the sal-both split
$PY scripts/e2e_summary.py                        # end-to-end counts, exact tails, language probes
$PY scripts/compute_summary.py                      # compute macros
$PY scripts/make_select_family_table.py --split occ --arms r1_film r0_rho00_film r0_rho90_film r1u_film sym_r1 sym_r0_rho00 mod_oracle mod_det mod_free   # PTA by family on the occlusion split (any-of convention)
$PY scripts/attr_heldout_only.py                    # held-out-combination commands only (attribute-OOD)
$PY scripts/fresh_standardised.py --boot $BOOT      # pilot against fresh IID scenes: family mix against within-family effect
$PY scripts/modular_summary.py                      # modular baseline: per-family PTA, detector quality, errors
$PY scripts/make_controls_table.py                  # image shifts, capacity, learning rate controls
$PY scripts/snap_summary.py                         # sensitivity to the snap radius
$PY scripts/occ_anyof.py                            # any-of pairs on the occlusion split under the v1.0 convention and the plain reading (re-scoring of the records)
$PY scripts/check_consistency.py                    # cross-metric checks: records, snapshots, tables (PTA / TSA / error shares), macros of independent scripts
echo "tables regenerated $(date)"
