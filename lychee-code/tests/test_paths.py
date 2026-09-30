"""Data paths: the evaluation-set metadata ship with the repository, so the table scripts run without the rendered scenes."""
import os

import numpy as np
import pytest

import lychee.paths as P


def test_eval_meta_falls_back_to_the_shipped_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "DATA", str(tmp_path / "nowhere"))
    path = P.eval_meta("iid_600")
    assert os.path.exists(path) and "data_meta" in path.replace("\\", "/")
    d = np.load(path)
    assert d["ok"].shape == (600,) and d["tmask"].shape[0] == 600
    with pytest.raises(FileNotFoundError):
        P.eval_meta("no_such_set")


def test_shipped_metadata_covers_every_evaluation_set_of_the_paper():
    for name in ("iid_600", "sal_600", "occ_300", "dens_300", "lang_300", "attr_300", "dr_300", "iid_fresh_600", "sal_fresh_600", "sal_both_600", "attr_fresh_300"):
        assert os.path.exists(os.path.join(P.SHIPPED, "select_eval", name, "meta.npz")), name
