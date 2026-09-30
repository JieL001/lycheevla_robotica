"""Compare policies across splits from results/eval/<tag>__<split>.jsonl.
   python scripts/compare_policies.py --tags bc_unpaired_film bc_paired_film --splits iid occ_ood density_ood
Prints markdown tables (percent, bootstrap 95% CI over pairs): PTA (strict), PTA_appr (choice only), TSA_appr, success.
"""
import argparse, json, os, sys

sys.path.insert(0, ".")
from lychee import evalkit


def load(tag, split):
    """learned policies are stored as <tag>__<split>.jsonl, scripted references as <tag>_<split>.jsonl"""
    for path in (f"results/eval/{tag}__{split}.jsonl", f"results/eval/{tag}_{split}.jsonl"):
        if os.path.exists(path):
            return [json.loads(l) for l in open(path)]
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["iid"])
    ap.add_argument("--by_family", action="store_true")
    a = ap.parse_args()
    for metric, label in (("PTA_sel", "PTA (first detached fruit is a target, both commands)"),
                          ("PTA_grasp", "PTA-grasp (first fruit the gripper closes on is a target, both commands)"),
                          ("grasp_rate", "grasp rate (the gripper closed on some fruit)"),
                          ("PTA_appr", "PTA-approach (fingertips first come within 5 cm of a target, both commands)"),
                          ("TSA_appr", "per-command approach accuracy"), ("success", "per-command harvest success")):
        print(f"\n### {label}")
        print("| policy | " + " | ".join(a.splits) + " |\n|---|" + "---|" * len(a.splits))
        for tag in a.tags:
            cells = []
            for sp in a.splits:
                recs = load(tag, sp)
                cells.append("-" if not recs else f"{evalkit.fmt(evalkit.summarize(recs)['all'][metric])} (n={evalkit.summarize(recs)['all']['n_pairs']})")
            print(f"| {tag} | " + " | ".join(cells) + " |")
    if a.by_family:
        fams = ["mat_any", "mat_side", "mat_ordinal", "mat_depth", "mat"]
        for sp in a.splits:
            print(f"\n### PTA-approach by family on {sp}")
            print("| policy | " + " | ".join(fams) + " |\n|---|" + "---|" * len(fams))
            for tag in a.tags:
                recs = load(tag, sp)
                if not recs:
                    continue
                d = evalkit.summarize(recs, by="family")
                print(f"| {tag} | " + " | ".join((f"{100*d[f]['PTA_appr'][0]:.0f} (n={d[f]['n_pairs']})" if f in d else "-") for f in fams) + " |")
