"""Build ../references.bib from the verified sources: ../lit/E_references_corrected.bib + ../lit/F_new_refs.bib.
arXiv-only @misc entries get `note = {arXiv preprint arXiv:<id>}` so that the plain bibliography style prints the identifier."""
import re

parts = [open(f"../lit/{n}", encoding="utf-8").read() for n in ("E_references_corrected.bib", "F_new_refs.bib", "I_new_refs_2.bib", "L_new_refs_3.bib", "N_new_refs_4.bib")]
text = "\n\n".join(parts)


def fix(m):
    blk = m.group(0)
    if "eprint" in blk and not re.search(r"\n\s*(note|howpublished)\s*=", blk):
        eid = re.search(r"eprint\s*=\s*\{([^}]*)\}", blk).group(1)
        blk = re.sub(r"\n\}\s*$", f",\n  note          = {{arXiv preprint arXiv:{eid}}}\n}}", blk)
    return blk


text = re.sub(r"@misc\{.*?\n\}\n?", fix, text, flags=re.S)
text = re.sub(r",\s*,", ",", text)
import json, os
if os.path.exists("../lit/author_full.json"):                 # complete author lists (Robotica wants them), see scripts/fetch_full_authors.py
    full = json.load(open("../lit/author_full.json", encoding="utf-8"))
    for key, v in full.items():
        pat = re.compile(r"(@\w+\{" + re.escape(key) + r",.*?\n\s*author\s*=\s*\{)(.*?)(\},?\n)", re.S)
        text, n = pat.subn(lambda m: m.group(1) + v["authors"] + m.group(3), text, count=1)
        assert n == 1, key
open("../references.bib", "w", encoding="utf-8", newline="\n").write(text)
print("entries:", len(re.findall(r"^@", text, flags=re.M)))
