"""Assemble an anonymised package for reviewers: code, evaluation records, checkpoints of the arms of the paper's main comparisons (about 100 MB), generated tables and figures
(no rendered data, logs or caches).
   python scripts/make_release.py   ->  ../release/lychee_harvest_sim_anon.zip, ../release/FILES.txt and the SHA-256 of the archive
Checks that no file of the package contains an e-mail address; the archive is written but nothing is uploaded."""
import fnmatch, hashlib, os, re, sys, zipfile

ROOT = os.path.abspath("..")
OUT = os.path.join(ROOT, "release")
INCLUDE = [("lychee-code/lychee", "*.py"), ("lychee-code/scripts", "*.py"), ("lychee-code/scripts", "regenerate_tables.sh"), ("lychee-code/scripts/queue", "*.sh"), ("lychee-code/tests", "*.py"),
           ("lychee-code/prereg", "*.md"), ("lychee-code/results/eval", "*.jsonl"), ("lychee-code/results/eval", "*.md"), ("lychee-code/results/eval", "*.json"),
           ("lychee-code/results/expert", "*.jsonl"), ("lychee-code/results/eval_ref_offline", "*.jsonl"), ("lychee-code/results", "*.json"), ("tables", "*.tex"), ("figs", "*.pdf"), ("figs", "*.png")]
_DM = os.path.join(ROOT, "lychee-code", "data_meta", "select_eval")                     # meta.npz of the evaluation sets (fruit tables, commands, target masks; no images)
if os.path.isdir(_DM):
    INCLUDE += [(f"lychee-code/data_meta/select_eval/{n}", pat) for n in sorted(os.listdir(_DM)) for pat in ("meta.npz", "info.json")]
EXTRA = [("lychee-code/scripts/release_README.md", "README.md"), ("lychee-code/scripts/release_LICENSE", "LICENSE"), ("lychee-code/scripts/release_REPRODUCE.md", "REPRODUCE.md"),
         ("lychee-code/scripts/release_requirements.txt", "requirements.txt"), ("lychee-code/scripts/release_requirements_train.txt", "requirements_train.txt"), ("lychee-code/scripts/release_gitattributes", ".gitattributes")]              # the English README only (the working README of the folder is in Chinese)
CKPT_DIR = os.environ.get("CKPT_DIR", "D:/lychee_data/select_ckpt")
# the checkpoints of the arms of the planned comparisons (five seeds), the conditioning variants, the controls of the review and the coverage and mix-matched controls of the dial (three seeds),
# the blank-command floor and the detectors of the modular baseline (the other arms of the paper are re-trained with scripts/queue/, level 3 of REPRODUCE.md)
CKPT_KEEP = re.compile(r"^(r0_rho00_film|r0_rho90_film|r1_film|r1u_film)(_s[1-4])?\.pt$|^(r1_late|r1_token)(_s[12])?\.pt$|^r1_blank_film\.pt$|^detector_s[0-2]\.pt$"
                       r"|^(r1_film_shiftnone|r1_film_shiftsync|r1_late_shiftnone|r1_late_shiftsync|r1_late_wide)(_s[12])?\.pt$|^r1_(film|late)_lr3e-[34]\.pt$|^(r0_matched90_film|r0_rho00_far146)(_s[12])?\.pt$|^sym_(r0_rho(00|50|90|97|99)|r1|r1_blank|r1u)\.pt$")
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def main():
    os.makedirs(OUT, exist_ok=True)
    files = []
    for d, pat in INCLUDE:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            if os.path.isfile(p) and fnmatch.fnmatch(name, pat):
                files.append((p, f"{d}/{name}"))
    ck_rows = []
    if os.path.isdir(CKPT_DIR):
        for name in sorted(os.listdir(CKPT_DIR)):
            if CKPT_KEEP.match(name):
                pth = os.path.join(CKPT_DIR, name)
                files.append((pth, f"checkpoints/{name}"))
                base = name[:-3]
                ck_rows.append((name, hashlib.sha256(open(pth, "rb").read()).hexdigest(), base))
    if ck_rows:
        md = ["# Checkpoints", "", "Weights of the networks behind the main tables of the paper (PyTorch `torch.save` dictionaries with the training arguments; `scripts/eval_select.py` and `scripts/eval_modular.py` load them).",
              "The records of `checkpoint <name>.pt` are `lychee-code/results/eval/s_<name>__<split>.jsonl` (`detector_s<k>` is evaluated as `mod_det[_s<k>]`).", "",
              "| file | SHA-256 |", "|---|---|"] + [f"| `{n}` | `{h}` |" for n, h, _ in ck_rows]
        tmp = os.path.join(OUT, "CHECKPOINTS.md")
        open(tmp, "w", newline="\n").write("\n".join(md) + "\n")
        files.append((tmp, "checkpoints/CHECKPOINTS.md"))
    for src, arc in EXTRA:
        p = os.path.join(ROOT, src)
        if os.path.exists(p):
            files.append((p, arc))
    bad = []
    for p, arc in files:
        data = open(p, "rb").read()
        m = EMAIL.search(data)
        if m and not arc.endswith((".jsonl", ".pt", ".npz")):                   # binary weights and the record files are not text
            bad.append((arc, m.group(0)[:60]))
    if bad:
        print("e-mail-like strings found, package not written:", bad)
        sys.exit(1)
    zpath = os.path.join(OUT, "lychee_harvest_sim_anon.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in files:
            z.write(p, arc)
    open(os.path.join(OUT, "FILES.txt"), "w", newline="\n").write("\n".join(a for _, a in files) + "\n")
    h = hashlib.sha256(open(zpath, "rb").read()).hexdigest()
    print(f"{len(files)} files, {os.path.getsize(zpath)/1e6:.1f} MB, sha256 {h}")
    open(os.path.join(OUT, "SHA256.txt"), "w").write(f"{h}  lychee_harvest_sim_anon.zip\n")


if __name__ == "__main__":
    main()
