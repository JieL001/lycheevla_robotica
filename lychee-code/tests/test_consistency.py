"""Cross-metric consistency checks (scripts/check_consistency.py) and the exact-arm matching of the modular summary (scripts/modular_summary.py): the checks accept consistent tables and
reject the inconsistency that a pooled-by-name-prefix summary produced (the margin-free error shares diluted by the oracle's zero errors, so that PTA fell below 1 - 2e)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
cc = pytest.importorskip("check_consistency")


def rec(index, which, family, correct, first_detached):
    return {"index": index, "which": which, "family": family, "target_correct": correct, "first_detached": first_detached}


def test_outcome_types_are_exclusive():
    assert cc.outcome({"target_correct": True, "first_detached": 3}) == "correct"
    assert cc.outcome({"target_correct": False, "first_detached": -1}) == "none"
    assert cc.outcome({"target_correct": False, "first_detached": 2}) == "wrong"


def test_summary_counts_and_bounds():
    recs = [rec(0, "plus", "mat_side", True, 1), rec(0, "minus", "mat_side", True, 2),          # correct pair
            rec(1, "plus", "mat_side", True, 1), rec(1, "minus", "mat_side", False, 0),         # plus right, minus wrong
            rec(2, "plus", "mat_depth", False, -1), rec(2, "minus", "mat_depth", False, 4)]     # nothing selected / wrong fruit
    fam, problems, sig = cc.summarise(recs)
    assert problems == [] and dict(sig) == {"mat_side": 2, "mat_depth": 1}
    assert fam["mat_side"] == {"pairs": 2, "plus_ok": 2, "minus_ok": 1, "both": 1, "correct": 3, "none": 0, "wrong": 1}
    assert fam["mat_depth"]["none"] == 1 and fam["mat_depth"]["wrong"] == 1
    for d in list(fam.values()) + [cc.pooled(fam)]:
        assert cc.frechet(d) == []


def test_structural_problems_are_reported():
    recs = [rec(0, "plus", "mat_side", True, 1),                                                # minus missing
            rec(1, "plus", "mat_side", True, 1), rec(1, "plus", "mat_side", True, 1), rec(1, "minus", "mat_side", True, 1),      # duplicate
            rec(2, "plus", "mat_side", True, -1), rec(2, "minus", "mat_any", False, 0)]         # correct without a selection, families differ
    _, problems, _ = cc.summarise(recs)
    assert any("incomplete" in p for p in problems)
    assert any("two plus" in p for p in problems)
    assert any("selected no fruit" in p for p in problems)
    assert any("minus is mat_any" in p for p in problems)


def test_frechet_detects_impossible_counts():
    ok = {"pairs": 10, "plus_ok": 8, "minus_ok": 9, "both": 7, "correct": 17, "none": 1, "wrong": 2}
    assert cc.frechet(ok) == []
    assert cc.frechet(dict(ok, both=6))                       # 6 < 8 + 9 - 10
    assert cc.frechet(dict(ok, both=9))                       # 9 > min(8, 9)
    assert cc.frechet(dict(ok, wrong=3))                      # outcome types no longer add up to the commands


def _write_tables(tmp_path, modular_rows, error_rows):
    m, e = tmp_path / "modular.tex", tmp_path / "modular_errors.tex"
    m.write_text(" \\\\\n".join(modular_rows) + "\n", encoding="utf-8")
    e.write_text(" \\\\\n".join(error_rows) + "\n", encoding="utf-8")
    return str(m), str(e)


def test_modular_tables_consistent(tmp_path):
    m, e = _write_tables(tmp_path,
                         ["Modular, benchmark margins & 3 & 89.9 (89.5--90.2) & 96.7 & 94.0 & 88.5 & 71.6 & 98.3 & 71.1 & 82.9 & 85.6",
                          "Modular, no margins & 3 & 93.6 (93.3--93.8) & 96.7 & 96.7 & 88.7 & 88.4 & 99.1 & 80.9 & 89.8 & 91.4"],
                         ["any-of & 966 & 97.9 & 1.3 & 0.7 & 98.2 & 1.0 & 0.7", "side & 960 & 96.7 & 2.0 & 1.4 & 98.3 & 0.0 & 1.7",
                          "ordinal & 834 & 91.2 & 5.8 & 3.0 & 91.5 & 5.2 & 3.4", "depth & 606 & 77.9 & 21.3 & 0.8 & 93.7 & 0.0 & 6.3", "unique & 234 & 99.1 & 0.9 & 0.0 & 99.6 & 0.4 & 0.0"])
    assert cc.check_modular_tables(m, e) == []


def test_modular_tables_reject_the_diluted_error_shares(tmp_path):
    """The margin-free error shares as an earlier version of the summary printed them (pooled with the oracle's zero errors): PTA below 1 - 2e for three families."""
    m, e = _write_tables(tmp_path,
                         ["Modular, benchmark margins & 3 & 89.9 & 96.7 & 94.0 & 88.5 & 71.6 & 98.3 & 71.1 & 82.9 & 85.6",
                          "Modular, no margins & 3 & 93.6 & 96.7 & 96.7 & 88.7 & 88.4 & 99.1 & 80.9 & 89.8 & 91.4"],
                         ["any-of & 966 & 97.9 & 1.3 & 0.7 & 98.6 & 0.8 & 0.5", "side & 960 & 96.7 & 2.0 & 1.4 & 98.7 & 0.0 & 1.3",
                          "ordinal & 834 & 91.2 & 5.8 & 3.0 & 93.4 & 3.9 & 2.6", "depth & 606 & 77.9 & 21.3 & 0.8 & 95.3 & 0.0 & 4.7", "unique & 234 & 99.1 & 0.9 & 0.0 & 99.7 & 0.3 & 0.0"])
    v = cc.check_modular_tables(m, e)
    assert any("any-of, free" in x for x in v) and any("side, free" in x for x in v) and any("depth, free" in x for x in v)
    assert not any("benchmark" in x for x in v)


def test_modular_tables_reject_shares_that_do_not_add_up(tmp_path):
    m, e = _write_tables(tmp_path,
                         ["Modular, benchmark margins & 3 & 89.9 & 96.7 & 94.0 & 88.5 & 71.6 & 98.3 & 71.1 & 82.9 & 85.6",
                          "Modular, no margins & 3 & 93.6 & 96.7 & 96.7 & 88.7 & 88.4 & 99.1 & 80.9 & 89.8 & 91.4"],
                         ["any-of & 966 & 97.9 & 1.3 & 0.7 & 98.2 & 1.0 & 0.7", "side & 960 & 96.7 & 2.0 & 1.4 & 98.3 & 0.0 & 1.7",
                          "ordinal & 834 & 91.2 & 5.8 & 3.0 & 91.5 & 5.2 & 1.4", "depth & 606 & 77.9 & 21.3 & 0.8 & 93.7 & 0.0 & 6.3", "unique & 234 & 99.1 & 0.9 & 0.0 & 99.6 & 0.4 & 0.0"])
    assert any("ordinal, free" in x and "instead of 100" in x for x in cc.check_modular_tables(m, e))


def test_macros_follow_tex_semantics(tmp_path):
    p = tmp_path / "m.tex"
    p.write_text("\\providecommand{\\a}{??}\\providecommand{\\b}{1.0}\n\\renewcommand{\\a}{96.7}\\providecommand{\\b}{2.0}\\providecommand{\\c}{??}\n", encoding="utf-8")
    mac = cc.read_macros([str(p)])
    assert mac == {"a": "96.7", "b": "1.0", "c": "??"}


def test_modular_summary_matches_arms_exactly():
    """'mod_free' must not pick up the records of 'mod_free_oracle' (its zero-error runs diluted the pooled error shares); seeds are matched by their suffix."""
    ms = pytest.importorskip("modular_summary")
    st = {"mod_free__iid": 1, "mod_free_s1__iid": 1, "mod_free_s2__iid": 1, "mod_free_oracle__iid": 1, "mod_free_oracle_s1__iid": 1, "mod_free__occ": 1, "mod_free_x__iid": 1}
    assert sorted(ms.stat_keys(st, "mod_free", "iid")) == ["mod_free__iid", "mod_free_s1__iid", "mod_free_s2__iid"]
    assert sorted(ms.stat_keys(st, "mod_free_oracle", "iid")) == ["mod_free_oracle__iid", "mod_free_oracle_s1__iid"]


def test_modular_summary_bound_check():
    ms = pytest.importorskip("modular_summary")
    fams = [f for f, _, _ in ms.FAMS]
    cnt = {f: [200, 180, 10, 10] for f in fams}                   # e = 0.10 -> PTA >= 0.80
    ms.check_outcomes("x", cnt, {f: 0.85 for f in fams})
    with pytest.raises(AssertionError):
        ms.check_outcomes("x", cnt, {f: 0.75 for f in fams})
    with pytest.raises(AssertionError):
        ms.check_outcomes("x", cnt, {f: 0.95 for f in fams})      # more pairs correct than commands


def test_seed0_table_relations(tmp_path):
    """PTA >= 2 TSA - 100, PTA <= TSA and PTA + collapse <= 100 in the rows of the seed-0 table."""
    p = tmp_path / "select_main.tex"
    ok = "R0 & unpaired & 94.8 [93, 96] & 93.8 [92, 95] & 97.2 & 2 & 96 [95, 98] & 27 \\\nblank & paired & 0.0 [0, 1] & 0.0 [0, 1] & 30.8 & 100 & 0 [0, 0] & 31"
    p.write_text(ok + "\n", encoding="utf-8")
    assert cc.check_select_table(str(p)) == []
    bad = "A & x & 80.0 [78, 82] & 79 & 94.0 & 2 & -- & --\\\nB & x & 96.0 [94, 98] & 95 & 94.0 & 2 & -- & --\\\nC & x & 90.0 [88, 92] & 89 & 95.0 & 15 & -- & --"
    p.write_text(bad + "\n", encoding="utf-8")
    v = cc.check_select_table(str(p))
    assert any(x.startswith("S8 A") and "below 2 TSA" in x for x in v)
    assert any(x.startswith("S8 B") and "exceeds TSA" in x for x in v)
    assert any(x.startswith("S8 C") and "collapse" in x for x in v)
