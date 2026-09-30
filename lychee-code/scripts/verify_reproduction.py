"""Regenerate every table and macro file of the paper in a TEMPORARY COPY of the repository and compare them byte for byte with the shipped ones.
   python scripts/verify_reproduction.py [--boot 4000]          (run from lychee-code/; needs the packages of requirements.txt; PY=/path/to/python bash is used for the run)
The copy contains lychee-code/ (code and results/eval records), tables/ and figs/; nothing outside the temporary directory is written.  Prints IDENTICAL or DIFFERS per file."""
import argparse, os, shutil, subprocess, sys, tempfile


def same_text(a, b):
    """Byte-for-byte equality of two text files up to the line-ending convention (a Windows clone with core.autocrlf=true has CRLF in its working tree)."""
    crlf, lf = bytes([13, 10]), bytes([10])
    return open(a, "rb").read().replace(crlf, lf) == open(b, "rb").read().replace(crlf, lf)


ap = argparse.ArgumentParser()
ap.add_argument("--boot", default="4000")
a = ap.parse_args()
here = os.path.abspath(".")                                   # lychee-code/
root = os.path.dirname(here)
tmp = tempfile.mkdtemp(prefix="lychee_repro_")
try:
    shutil.copytree(here, os.path.join(tmp, "lychee-code"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for d in ("tables", "figs"):
        if os.path.isdir(os.path.join(root, d)):
            shutil.copytree(os.path.join(root, d), os.path.join(tmp, d))
    # start from empty generated files so that a stale shipped file cannot hide a script that no longer writes it
    for d in ("tables",):
        for f in os.listdir(os.path.join(tmp, d)):
            if f.endswith(".tex"):
                os.remove(os.path.join(tmp, d, f))
    env = dict(os.environ, BOOT=a.boot, PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("PY", sys.executable.replace("\\", "/"))
    r = subprocess.run(["bash", "scripts/regenerate_tables.sh"], cwd=os.path.join(tmp, "lychee-code"), env=env, capture_output=True, text=True)
    print(r.stdout[-400:])
    if r.returncode != 0:
        print(r.stderr[-2000:]); sys.exit(1)
    # tables written by scripts that need the simulator or the rendered evaluation scenes (level 2 of REPRODUCE.md); they are shipped as generated and not regenerated here
    SEPARATE = {"reference_iid.tex", "reference_meta.tex", "reference_splits.tex", "visibility_macros.tex", "modular_depth_macros.tex"}
    same = diff = missing = 0
    for f in sorted(os.listdir(os.path.join(root, "tables"))):
        if not f.endswith(".tex"):
            continue
        new = os.path.join(tmp, "tables", f)
        if not os.path.exists(new):
            if f in SEPARATE:
                print("shipped as generated (needs the simulator or rendered scenes):", f); continue
            print("NOT REGENERATED", f); missing += 1; continue
        # line endings are ignored: a Windows clone with core.autocrlf=true has CRLF in the working tree (the release also ships a .gitattributes that keeps LF)
        if same_text(os.path.join(root, "tables", f), new):
            same += 1
        else:
            print("DIFFERS", f); diff += 1
    for f in sorted(os.listdir(os.path.join(root, "figs"))):
        if f.endswith(".pdf") and not os.path.exists(os.path.join(tmp, "figs", f)):
            print("FIGURE MISSING", f)
    print(f"tables and macro files: {same} IDENTICAL, {diff} differ, {missing} not regenerated")
    sys.exit(0 if diff == 0 and missing == 0 else 1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
