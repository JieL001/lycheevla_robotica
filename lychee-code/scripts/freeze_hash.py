"""Record SHA-256 hashes of the pre-registration and benchmark source files.  usage: python scripts/freeze_hash.py"""
import glob, hashlib, sys, time

files = sorted(glob.glob("prereg/PREREG_v*.md") + glob.glob("lychee/*.py") + ["scripts/generate_data.py", "scripts/eval_policy.py"])
lines = [f"# frozen {time.strftime('%Y-%m-%d %H:%M:%S')}"]
for f in files:
    lines.append(f"{hashlib.sha256(open(f, 'rb').read()).hexdigest()}  {f}")
open("prereg/FROZEN_HASHES.txt", "w", newline="\n").write("\n".join(lines) + "\n")
print("\n".join(lines))
