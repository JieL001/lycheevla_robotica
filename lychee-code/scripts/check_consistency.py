"""Cross-metric consistency checks of the per-command evaluation records and of the generated tables (run at the end of scripts/regenerate_tables.sh; exit status 1 on any violation).
   python scripts/check_consistency.py [--quiet]
A file that is byte-identical to the one that was shipped can still be logically wrong (a table built on a mis-selected set of records), so these checks are about the relations between
the numbers, not about the files.  Definitions: a command is exactly one of the mutually exclusive outcomes
    correct    the selected fruit is a target (target_correct),
    no target  nothing was selected (first_detached < 0: the rules found no target, or the pointing position was farther than the snap radius from every fruit),
    wrong      a fruit was selected that is not a target;
a pair is correct (PTA) if both of its commands are.  Let TSA+ / TSA- be the shares of correct commands among the plus / minus commands and e+ = 1 - TSA+, e- = 1 - TSA-.
Record files (results/eval/s_<arm>__<set>.jsonl, sel_<arm>__<set>.jsonl), for every file and family:
  R1  every index has exactly one plus and one minus command of the same family, none twice;
  R2  a correct command has selected a fruit (target_correct implies first_detached >= 0);
  R3  max(0, TSA+ + TSA- - 1) <= PTA <= min(TSA+, TSA-) (Frechet bounds; the lower bound is PTA >= 1 - e+ - e-, i.e. PTA >= 1 - 2e for e = (e+ + e-)/2);
  R4  all files of a set contain the same commands (same indices, families, numbers of pairs).
Snapshots and tables:
  S1  results/eval/modular_stats.json (error counts written when the modular baseline was evaluated) agrees with the counts recomputed from the records;
  S2  tables/modular_errors.tex: the shares of a row add up to 100 %, and for every family and both versions of the rules the PTA in tables/modular.tex satisfies the bounds of R3 with
      the shares of tables/modular_errors.tex (the two tables come from the same records; a version that pools records of another arm is caught here);
  S3  the macros of scripts/modular_summary.py agree with those of scripts/select_seeds.py for the arms both compute (independent code paths, same records);
  S4  the macros \\sm<Arm><Split> of scripts/select_seeds.py agree with the PTA recomputed here from the records (mean over the training seeds, in percent);
  S5  the same for the fresh-scene macros \\fm<Arm><Set> (scripts/fresh_summary.py) and the robustness-control macros \\ct<Row><Set> (scripts/make_controls_table.py);
  S6  the modular-baseline numbers of Table 7 (\\mod<Row>Iid, Attr, IidF, OccN and the five command families) agree with the PTA recomputed here from the records;
  S8  the seed-0 table (tables/select_main.tex): for every arm PTA >= 2 TSA - 100, PTA <= TSA, and PTA + collapse rate <= 100 (a pair whose two commands lead to the same fruit is not correct, because
      the two target sets are disjoint), up to the rounding of the printed numbers;
  S7  the occlusion any-of macros \\oa<Arm>Old (scripts/occ_anyof.py) equal the PTA on the any-of pairs of the occlusion records as scored (the same pairs, a different code path)."""
import argparse, glob, json, os, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

RES = "results/eval"
TABLES = os.environ.get("LYCHEE_TABLES", "../tables")
ROUND = 0.15                        # percentage points: rounding of the (up to three) shown numbers that enter one relation
FILE_RE = re.compile(r"(s|sel)_(.+)__(\w+)\.jsonl")
NOANY = ("mat", "mat_side", "mat_depth", "mat_ordinal")
FAMILIES = ("mat_any", "mat_side", "mat_ordinal", "mat_depth", "mat")
TABLE_FAMILIES = {"any-of": "mat_any", "side": "mat_side", "ordinal": "mat_ordinal", "depth": "mat_depth", "unique": "mat"}


# ---------------------------------------------------------------------------------------------- record level
def outcome(cmd):
    """The mutually exclusive outcome type of one command."""
    if cmd["target_correct"]:
        return "correct"
    return "none" if cmd["first_detached"] < 0 else "wrong"


def summarise(recs):
    """Per-family counters of one record file: pairs, plus_ok, minus_ok, both, (correct, none, wrong) commands; plus the list of structural problems and the signature of the set."""
    problems, pairs = [], defaultdict(dict)
    for r in recs:
        w = r.get("which")
        if w not in ("plus", "minus") or "index" not in r or "target_correct" not in r or "first_detached" not in r or "family" not in r:
            problems.append(f"malformed record {str(r)[:80]}")
            continue
        if w in pairs[r["index"]]:
            problems.append(f"index {r['index']}: two {w} commands")
        pairs[r["index"]][w] = r
    fam = defaultdict(lambda: {"pairs": 0, "plus_ok": 0, "minus_ok": 0, "both": 0, "correct": 0, "none": 0, "wrong": 0})
    for idx, v in sorted(pairs.items()):
        if len(v) < 2:
            problems.append(f"index {idx}: incomplete pair")
            continue
        p, m = v["plus"], v["minus"]
        if p["family"] != m["family"]:
            problems.append(f"index {idx}: plus is {p['family']}, minus is {m['family']}")
        for c in (p, m):
            if c["target_correct"] and c["first_detached"] < 0:
                problems.append(f"index {idx}: a correct command that selected no fruit")
        d = fam[p["family"]]
        d["pairs"] += 1
        d["plus_ok"] += bool(p["target_correct"])
        d["minus_ok"] += bool(m["target_correct"])
        d["both"] += bool(p["target_correct"] and m["target_correct"])
        for c in (p, m):
            d[outcome(c)] += 1
    sig = tuple(sorted((f, d["pairs"]) for f, d in fam.items()))
    return dict(fam), problems, sig


def frechet(d):
    """Violations of max(0, plus_ok + minus_ok - pairs) <= both <= min(plus_ok, minus_ok) in counts (exact, no tolerance); d has the keys of ``summarise``."""
    lo, hi = max(0, d["plus_ok"] + d["minus_ok"] - d["pairs"]), min(d["plus_ok"], d["minus_ok"])
    out = []
    if d["both"] < lo:
        out.append(f"PTA count {d['both']} below the lower bound {lo} (TSA+ + TSA- - 1)")
    if d["both"] > hi:
        out.append(f"PTA count {d['both']} above the upper bound {hi} (min TSA)")
    if d["correct"] + d["none"] + d["wrong"] != 2 * d["pairs"]:
        out.append("the outcome types do not add up to the number of commands")
    if d["correct"] != d["plus_ok"] + d["minus_ok"]:
        out.append("correct commands differ from the sum of the two TSA counts")
    return out


def pooled(fam, families=None):
    keep = [d for f, d in fam.items() if families is None or f in families]
    return {k: sum(d[k] for d in keep) for k in ("pairs", "plus_ok", "minus_ok", "both", "correct", "none", "wrong")}


def check_records(quiet=False):
    viol, sums, sigs = [], {}, defaultdict(list)
    files = sorted(glob.glob(os.path.join(RES, "s_*.jsonl")) + glob.glob(os.path.join(RES, "sel_*.jsonl")))
    for p in files:
        m = FILE_RE.fullmatch(os.path.basename(p))
        if not m:
            continue
        kind, tag, sp = m.groups()
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        fam, problems, sig = summarise(recs)
        name = os.path.basename(p)
        for q in problems[:5]:
            viol.append(f"R1/R2 {name}: {q}")
        if len(problems) > 5:
            viol.append(f"R1/R2 {name}: ... and {len(problems) - 5} more")
        for f, d in list(fam.items()) + [("all", pooled(fam))]:
            for q in frechet(d):
                viol.append(f"R3 {name} [{f}]: {q}")
        sums[(kind, tag, sp)] = fam
        k = re.search(r"_K(\d+)$", tag)                                                            # the counterfactual sets of the binding arms are their K training scenes
        sigs[(kind, sp, k.group(1) if k else None)].append((name, sig))
    for (kind, sp, k), lst in sigs.items():                                                       # R4
        common = Counter(s for _, s in lst).most_common(1)[0][0]
        for name, s in lst:
            if s != common:
                viol.append(f"R4 {name}: contains other commands than the other files of the set '{sp}' ({dict(s)} against {dict(common)})")
    if not quiet:
        print(f"records: {len(files)} files, {sum(sum(d['pairs'] for d in f.values()) for f in sums.values())} pairs")
    return viol, sums


# ---------------------------------------------------------------------------------------------- snapshots
def check_stats(sums):
    """S1: the error counts stored with the modular baseline against those recomputed from its records."""
    viol, p = [], os.path.join(RES, "modular_stats.json")
    if not os.path.exists(p):
        return viol
    st = json.load(open(p))
    for key, v in st.items():
        arm, sp = key.split("__")
        fam = sums.get(("s", arm, sp))
        if fam is None or "errors" not in v:
            continue
        for f, (n, none, far, wrong) in v["errors"].items():
            d = fam.get(f)
            if d is None:
                viol.append(f"S1 {key} [{f}]: no records")
                continue
            if (n, none + far, wrong) != (2 * d["pairs"], d["none"], d["wrong"]):
                viol.append(f"S1 {key} [{f}]: stats (commands, no target, wrong) = ({n}, {none + far}, {wrong}) but the records give ({2 * d['pairs']}, {d['none']}, {d['wrong']})")
            if none + far + wrong + d["correct"] != n:
                viol.append(f"S1 {key} [{f}]: the error counts do not leave room for the correct commands")
    return viol


def _first_number(cell):
    m = re.search(r"-?\d+(?:\.\d+)?", cell)
    return float(m.group()) if m else None


def _rows(path):
    return [[c.strip() for c in row.split("&")] for row in open(path, encoding="utf-8").read().replace("\\\\", "").split("\n") if row.strip()]


def check_modular_tables(modular_path=None, errors_path=None):
    """S2: tables/modular.tex (PTA per family) against tables/modular_errors.tex (shares of correct / no target / wrong)."""
    modular_path = modular_path or os.path.join(TABLES, "modular.tex")
    errors_path = errors_path or os.path.join(TABLES, "modular_errors.tex")
    viol = []
    if not (os.path.exists(modular_path) and os.path.exists(errors_path)):
        return ["S2 tables/modular.tex or tables/modular_errors.tex is missing"]
    ptas = {}
    for row in _rows(modular_path):
        if row[0] == "Modular, benchmark margins":
            ptas["benchmark"] = {f: _first_number(row[3 + i]) for i, f in enumerate(("any-of", "side", "ordinal", "depth", "unique"))}
        elif row[0] == "Modular, no margins":
            ptas["free"] = {f: _first_number(row[3 + i]) for i, f in enumerate(("any-of", "side", "ordinal", "depth", "unique"))}
    if set(ptas) != {"benchmark", "free"}:
        return ["S2 tables/modular.tex lacks the rows of the two versions of the rules"]
    for row in _rows(errors_path):
        fam, vals = row[0], [_first_number(c) for c in row[2:8]]
        for version, (c, n, w) in (("benchmark", vals[0:3]), ("free", vals[3:6])):
            if abs(c + n + w - 100) > ROUND:
                viol.append(f"S2 {fam}, {version}: correct + no target + wrong = {c + n + w:.1f} instead of 100")
            e = (n + w) / 100
            p = ptas[version][fam]
            lower, upper = max(0.0, 100 * (1 - 2 * e)), c
            if p < lower - ROUND:
                viol.append(f"S2 {fam}, {version}: PTA {p:.1f} is below 100 (1 - 2e) = {lower:.1f} with e = {100 * e:.1f} % failed commands")
            if p > upper + ROUND:
                viol.append(f"S2 {fam}, {version}: PTA {p:.1f} exceeds the share of correct commands {upper:.1f}")
    return viol


def check_select_table(path=None):
    """S8: relations between PTA, TSA and the collapse rate in the rows of the seed-0 table (all in percent, on IID scenes)."""
    path = path or os.path.join(TABLES, "select_main.tex")
    if not os.path.exists(path):
        return []
    viol = []
    for row in _rows(path):
        if len(row) < 6:
            continue
        pta, tsa, col = _first_number(row[2]), _first_number(row[4]), _first_number(row[5])
        if None in (pta, tsa, col):
            continue
        name = row[0]
        if pta < 2 * tsa - 100 - ROUND:
            viol.append(f"S8 {name}: PTA {pta} is below 2 TSA - 100 = {2 * tsa - 100:.1f}")
        if pta > tsa + ROUND:
            viol.append(f"S8 {name}: PTA {pta} exceeds TSA {tsa}")
        if pta + col > 100 + 0.65:
            viol.append(f"S8 {name}: PTA {pta} + collapse rate {col} exceeds 100")
    return viol


def read_macros(paths):
    """TeX semantics: \\providecommand defines only if undefined, \\renewcommand always; a value '??' is a placeholder."""
    mac = {}
    pat = re.compile(r"\\(provide|renew)command\{\\(\w+)\}\{([^{}]*)\}")
    for p in paths:
        if os.path.exists(p):
            for kind, name, val in pat.findall(open(p, encoding="utf-8").read()):
                if kind == "renew" or name not in mac or mac[name] == "??":
                    mac[name] = val
    return mac


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def check_modular_macros():
    """S3: the same numbers computed by scripts/modular_summary.py and scripts/select_seeds.py."""
    viol = []
    mac = read_macros(sorted(glob.glob(os.path.join(TABLES, "*macros.tex"))))
    pairs = [("Rone", "Rone"), ("Rzero", "Rzero"), ("Sone", "Sone"), ("Det", "ModDet"), ("Free", "ModFree"), ("Oracle", "ModOracle")]
    for mt, st in pairs:
        for sp in ("Iid", "Attr", "OccN", "IidF"):
            a, b = _num(mac.get(f"mod{mt}{sp}")), _num(mac.get(f"sm{st}{sp}"))
            if a is not None and b is not None and abs(a - b) > 0.051:
                viol.append(f"S3 modular_summary \\mod{mt}{sp} = {a} but select_seeds \\sm{st}{sp} = {b}")
    return viol


def mean_pta(sums, base, split, nmax=None, families=None):
    """PTA in percent, mean over the training seeds (files <base>, <base>_s1, ... up to nmax of them) on the pairs of ``families`` (None = all), or None without records."""
    from select_seeds import SEED_SUFFIXES
    vals = []
    for suf in (SEED_SUFFIXES[:nmax] if nmax else SEED_SUFFIXES):
        fam = sums.get(("s", base + suf, split))
        if fam:
            d = pooled(fam, families)
            if d["pairs"]:
                vals.append(100 * d["both"] / d["pairs"])
    return sum(vals) / len(vals) if vals else None


def _compare(viol, tag, macros, name, value, what):
    """Append a violation if the macro ``name`` (when defined) differs from the recomputed ``value`` by more than the rounding of one decimal; returns 1 if it was compared."""
    m = _num(macros.get(name))
    if value is None or m is None:
        return 0
    if abs(value - m) > 0.051:
        viol.append(f"{tag} \\{name} = {m} but the records give {value:.1f} ({what})")
    return 1


def check_macros_against_records(sums):
    """S4: \\sm<Arm><Split> against the PTA recomputed from the records (mean over the training seeds)."""
    from select_seeds import ARMS, SPLITS, DERIVED
    viol, checked = [], 0
    mac = read_macros([os.path.join(TABLES, "select_macros.tex")])
    for key, base, _ in ARMS:
        for fsplit, _, mname in SPLITS:
            src, fams = DERIVED.get(fsplit, (fsplit, None))
            checked += _compare(viol, "S4", mac, f"sm{key}{mname}", mean_pta(sums, base, src, families=fams), f"{base} on '{fsplit}'")
    return viol, checked


def check_more_macros(sums):
    """S5-S7: other macro families recomputed from the records with the code of this file."""
    import fresh_summary, make_controls_table, modular_summary
    viol, n = [], 0
    fm = read_macros([os.path.join(TABLES, "fresh_macros.tex")])
    for key, base, _ in fresh_summary.ARMS:
        for split, _, m in fresh_summary.SETS:
            n += _compare(viol, "S5", fm, f"fm{key}{m}", mean_pta(sums, base, split), f"{base} on '{split}'")
    ct = read_macros([os.path.join(TABLES, "controls_macros.tex")])
    for key, base, _, nmax in make_controls_table.ROWS:
        for split, tag in make_controls_table.SETS:
            n += _compare(viol, "S5", ct, f"ct{key}{tag}", mean_pta(sums, base, split, nmax=nmax), f"{base} on '{split}', first {nmax} seeds")
    md = read_macros([os.path.join(TABLES, "modular_macros.tex")])
    for _, base, tag in modular_summary.ROWS:
        n += _compare(viol, "S6", md, f"mod{tag}Iid", mean_pta(sums, base, "iid"), f"{base} on 'iid'")
        for fam, _, ft in modular_summary.FAMS:
            n += _compare(viol, "S6", md, f"mod{tag}{ft}", mean_pta(sums, base, "iid", families=(fam,)), f"{base} on 'iid', {fam}")
        n += _compare(viol, "S6", md, f"mod{tag}OccN", mean_pta(sums, base, "occ", families=NOANY), f"{base} on 'occ' without any-of")
        for sp, tg in (("attr", "Attr"), ("iidf", "IidF")):
            n += _compare(viol, "S6", md, f"mod{tag}{tg}", mean_pta(sums, base, sp), f"{base} on '{sp}'")
    oa = read_macros([os.path.join(TABLES, "occ_anyof_macros.tex")])
    from occ_anyof import ORDER
    from select_seeds import ARMS
    bases = {k: b for k, b, _ in ARMS}
    for key in ORDER:
        n += _compare(viol, "S7", oa, f"oa{key}Old", mean_pta(sums, bases[key], "occ", families=("mat_any",)), f"{bases[key]} on the any-of pairs of 'occ'")
    return viol, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    viol, sums = check_records(a.quiet)
    viol += check_stats(sums)
    viol += check_modular_tables()
    viol += check_select_table()
    viol += check_modular_macros()
    v4, n4 = check_macros_against_records(sums)
    v5, n5 = check_more_macros(sums)
    viol += v4 + v5
    if not a.quiet:
        print(f"macros: {n4} arm x set numbers of select_macros.tex and {n5} numbers of the fresh, controls, modular and occlusion any-of macros recomputed from the records")
    if viol:
        print(f"{len(viol)} VIOLATIONS", file=sys.stderr)
        for v in viol[:60]:
            print("  " + v, file=sys.stderr)
        sys.exit(1)
    print("consistency checks passed")


if __name__ == "__main__":
    main()
