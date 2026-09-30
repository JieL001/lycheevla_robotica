"""Mechanical checks of the compiled manuscript: unresolved references or macros ("??" or "[?]"), TODO markers, abstract length (Robotica: at most 250 words),
number of pages.  Needs pdftotext (Git for Windows ships one).
   python scripts/check_manuscript.py [../main_rob.pdf]        (default: the Robotica-format PDF ../main_rob.pdf; pass ../main.pdf for the article-class master)"""
import re, subprocess, sys

PDFTOTEXT = r"C:\Program Files\Git\mingw64\bin\pdftotext.exe"
pdf = sys.argv[1] if len(sys.argv) > 1 else "../main_rob.pdf"
txt = subprocess.run([PDFTOTEXT, "-layout", pdf, "-"], capture_output=True, text=True, encoding="utf-8", errors="ignore").stdout
lines = txt.splitlines()
print("file:", pdf)
print("pages:", txt.count("\f"))
bad = [(i + 1, l.strip()) for i, l in enumerate(lines) if "??" in l or "[?]" in l]
print("unresolved '??' / '[?]':", len(bad))
for i, l in bad[:10]:
    print("   ", i, l[:120])
todo = [(i + 1, l.strip()) for i, l in enumerate(lines) if "TODO" in l]
print("TODO markers:", len(todo))
for i, l in todo:
    print("   ", i, l[:110])
m = re.search(r"\n\s*Abstract\s*\n(.*?)\n\s*1\.?\s+Introduction", txt, flags=re.S)
if m:
    words = re.sub(r"\s+", " ", m.group(1).replace("-\n", "")).split()
    print("abstract words:", len(words), "(limit 250)")
else:
    print("abstract not found")
