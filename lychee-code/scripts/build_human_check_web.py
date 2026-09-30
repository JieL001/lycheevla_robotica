"""Compose the annotation page: template + items.json -> results/human_check_web/index.html (publish it as an Artifact).
   python scripts/build_human_check_web.py [--lead "Send the copied results to ..."]"""
import argparse, json, pathlib
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results/human_check_web")
ap.add_argument("--lead", default="")
a = ap.parse_args()
t = pathlib.Path("scripts/human_check_template.html").read_text(encoding="utf-8")
items = pathlib.Path(f"{a.dir}/items.json").read_text(encoding="utf-8")
assert "</script" not in items
html = t.replace("__ITEMS_JSON__", items).replace("__STUDY_LEAD__", a.lead.replace('"', "'"))
pathlib.Path(f"{a.dir}/index.html").write_text(html, encoding="utf-8", newline="\n")
print(f"wrote {a.dir}/index.html ({len(html)/1e6:.1f} MB, {len(json.loads(items))} items)")
