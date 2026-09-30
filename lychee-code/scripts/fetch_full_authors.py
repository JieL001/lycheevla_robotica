"""Full author lists for the arXiv-only bibliography entries whose author field was truncated with "and others"
(Robotica wants complete author lists).  Reads the arXiv API (metadata only) and writes ../lit/author_full.json {bibkey: "Last, First and ..."};
scripts/build_refs.py applies it.   python scripts/fetch_full_authors.py"""
import json, os, re, sys, time, urllib.request
import xml.etree.ElementTree as ET

KEYS = {"deng2025graspvla": "2505.03233", "black2025pi05": "2504.16054", "zhong2025dexgraspvla": "2502.20900", "sun2026afp": "2607.10655",
        "jia2026guidedvla": "2605.12369", "li2025controlvla": "2506.16211", "wang2025vlaadapter": "2509.09372", "zhao2026instructmove": "2608.22990",
        "fei2025liberoplus": "2510.13626", "shi2026vlatrace": "2605.30117", "shukor2025smolvla": "2506.01844", "black2024pi0": "2410.24164"}
NS = {"a": "http://www.w3.org/2005/Atom"}


def fmt(name: str) -> str:
    parts = name.replace("\n", " ").split()
    suffix = {"Jr.", "Jr", "III", "II"}
    if len(parts) == 1:
        return parts[0]
    if parts[-1] in suffix and len(parts) > 2:
        return f"{parts[-2]} {parts[-1]}, {' '.join(parts[:-2])}"
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


out = {}
for key, aid in KEYS.items():
    url = f"http://export.arxiv.org/api/query?id_list={aid}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "lychee-bib-check/0.1"}), timeout=40) as r:
                root = ET.fromstring(r.read())
            break
        except Exception as e:
            print("retry", key, repr(e)[:80]); time.sleep(3)
    else:
        print("FAILED", key); continue
    ent = root.find("a:entry", NS)
    authors = [a.find("a:name", NS).text for a in ent.findall("a:author", NS)]
    title = " ".join(ent.find("a:title", NS).text.split())
    out[key] = dict(authors=" and ".join(fmt(a) for a in authors), n=len(authors), title=title)
    print(f"{key:22s} {aid:12s} {len(authors):3d} authors | {title[:70]}")
    time.sleep(3)                                     # be polite to the API
os.makedirs("../lit", exist_ok=True)
json.dump(out, open("../lit/author_full.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote ../lit/author_full.json")
