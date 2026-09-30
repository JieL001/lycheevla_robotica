import numpy as np
import pytest

torch = pytest.importorskip("torch")
from lychee.bc import VOCAB, tokenize
from lychee.layout import DEFAULT_CAM
from lychee.select import (SelectNet, MAP_HW, CELL, N_MAX, cell_centres, decode, fruit_table, select_loss, sincos_2d,
                           target_distribution)


def test_position_embedding_and_cell_centres():
    assert sincos_2d(*MAP_HW, 128).shape == (MAP_HW[0] * MAP_HW[1], 128)
    c = cell_centres()
    assert c.shape == (MAP_HW[0] * MAP_HW[1], 2)
    assert tuple(c[0].tolist()) == (CELL / 2, CELL / 2) and tuple(c[-1].tolist()) == (MAP_HW[1] * CELL - CELL / 2, MAP_HW[0] * CELL - CELL / 2)


@pytest.mark.parametrize("cond", ["film", "late", "token"])
def test_forward_shape_gradients_and_language_sensitivity(cond):
    torch.manual_seed(0)
    net = SelectNet(len(VOCAB), cond=cond)
    x = torch.rand(2, 3, 160, 256)
    t1 = torch.as_tensor(np.stack([tokenize("pick the ripe lychee on the left")] * 2))
    t2 = torch.as_tensor(np.stack([tokenize("pick the ripe lychee on the right")] * 2))
    out = net(x, t1)
    assert out.shape == (2, MAP_HW[0] * MAP_HW[1])
    out.logsumexp(1).sum().backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)
    net.eval()
    with torch.no_grad():
        assert (net(x, t1) - net(x, t2)).abs().max() > 1e-6            # the command reaches the logits


def _fruit(xy):
    f = torch.zeros(1, N_MAX, 2)
    v = torch.zeros(1, N_MAX, dtype=torch.bool)
    for i, p in enumerate(xy):
        f[0, i] = torch.tensor(p, dtype=torch.float32)
        v[0, i] = True
    return f, v


def test_target_distribution_is_a_mixture_over_the_target_fruit_only():
    f, v = _fruit([(100.0, 60.0), (200.0, 100.0), (40.0, 120.0)])
    tm = torch.zeros(1, N_MAX, dtype=torch.bool)
    tm[0, 0] = True
    q = target_distribution(f, tm & v)
    assert abs(float(q.sum()) - 1.0) < 1e-5
    c = cell_centres()
    assert float(((c[q[0].argmax()] - f[0, 0]) ** 2).sum().sqrt()) <= CELL
    tm[0, 2] = True                                                     # any-of: two targets -> two modes of equal mass
    q2 = target_distribution(f, tm & v)[0]
    near = lambda p: float(q2[((c - torch.tensor(p)) ** 2).sum(1) < 30 ** 2].sum())
    assert abs(near((100.0, 60.0)) - 0.5) < 0.05 and abs(near((40.0, 120.0)) - 0.5) < 0.05
    assert near((200.0, 100.0)) < 0.01


def test_decode_picks_nearest_fruit_or_reports_invalid():
    f, v = _fruit([(100.0, 60.0), (200.0, 100.0)])
    c = cell_centres()
    logits = torch.full((1, c.shape[0]), -5.0)
    logits[0, ((c - torch.tensor([196.0, 100.0])) ** 2).sum(1).argmin()] = 5.0
    assert int(decode(logits, f, v)[0]) == 1
    logits = torch.full((1, c.shape[0]), -5.0)
    logits[0, ((c - torch.tensor([20.0, 20.0])) ** 2).sum(1).argmin()] = 5.0            # far from every fruit
    assert int(decode(logits, f, v)[0]) == -1
    v2 = v.clone(); v2[0, 1] = False                                                     # an invalid slot is never chosen
    logits = torch.full((1, c.shape[0]), -5.0)
    logits[0, ((c - torch.tensor([196.0, 100.0])) ** 2).sum(1).argmin()] = 5.0
    assert int(decode(logits, f, v2)[0]) == -1


def test_loss_is_lower_when_the_logits_point_at_the_target():
    f, v = _fruit([(100.0, 60.0), (200.0, 100.0)])
    tm = torch.zeros(1, N_MAX, dtype=torch.bool); tm[0, 0] = True
    q = target_distribution(f, tm & v)
    right, wrong = q.clone() * 10, torch.roll(q.clone(), 40, dims=1) * 10
    assert select_loss(right, q) < select_loss(wrong, q)


def test_fruit_table_geometry():
    class L:                                                           # minimal stand-in for a Layout
        cam = DEFAULT_CAM
    xy, rad, valid = fruit_table(L, [DEFAULT_CAM.target], 0.03)
    assert valid[0] and not valid[1:].any()
    assert np.allclose(xy[0], (128.0, 128.0 - 40.0), atol=1e-3)         # the camera target is the image centre
    assert 8.0 < rad[0] < 16.0                                          # a 3 cm fruit spans ~22 px at this distance


def test_blank_trained_checkpoint_always_receives_the_blank_command():
    """A network trained without commands is the language-blind reference: swapped, gibberish and normal probes must give it the SAME (empty) command, so that its language
    sensitivity (TSA normal minus TSA swapped) is exactly 0 (audit of 9/30: the swap probe had fed it real commands, 2.3 points)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_select as es
    rng = np.random.RandomState(0)
    words = [w for w in VOCAB if not w.startswith("<")]
    blank = torch.as_tensor(tokenize(""))
    tok = np.stack([np.stack([tokenize("pick the ripe lychee on the left"), tokenize("pick the ripe lychee on the right")])] * 3)      # (scenes, 2 commands, tokens)
    idx = np.arange(3)
    for mode in ("normal", "swap", "gibberish", "blank"):
        assert (es.command_tokens(mode, {"blank": True}, tok, idx, 0, rng, words, blank) == blank).all(), mode
    assert (es.command_tokens("blank", {}, tok, idx, 0, rng, words, blank) == blank).all()                       # a network trained with commands: blank probe = empty command
    normal = es.command_tokens("normal", {}, tok, idx, 0, rng, words, blank)
    assert (normal == torch.as_tensor(tok[idx, 0])).all()
    assert (es.command_tokens("swap", {}, tok, idx, 0, rng, words, blank) == torch.as_tensor(tok[idx, 1])).all()          # the other member's command
    assert not (es.command_tokens("gibberish", {}, tok, idx, 0, rng, words, blank) == normal).all()
    assert not (es.command_tokens("normal", {"blank": True}, tok, idx, 0, rng, words, blank, raw=True) == blank).all()  # raw=True bypasses the rule (diagnostic)


def test_shift_offsets_report_where_the_content_moved():
    """random_shift_offsets returns the pixel shift of the content: a bright pixel at (x, y) of the input is at (x, y) + offset in the output (used by the 'synchronised shifts' control)."""
    from lychee.bc import random_shift_offsets
    torch.manual_seed(0)
    x = torch.zeros(6, 3, 40, 64)
    x[:, :, 20, 30] = 1.0                                              # a single bright pixel at (x=30, y=20)
    y, off = random_shift_offsets(x, pad=4)
    for b in range(6):
        yy, xx = np.unravel_index(int(y[b, 0].argmax()), y[b, 0].shape)
        assert (xx, yy) == (30 + int(off[b, 0]), 20 + int(off[b, 1])), (b, (xx, yy), off[b])


def test_late_fusion_width_matches_the_parameter_count_of_film():
    n = lambda m: sum(p.numel() for p in m.parameters())
    film, late, wide = SelectNet(len(VOCAB), cond="film"), SelectNet(len(VOCAB), cond="late"), SelectNet(len(VOCAB), cond="late", late_dim=256)
    assert n(late) < n(wide) and abs(n(wide) - n(film)) < 0.03 * n(film), (n(film), n(late), n(wide))
    out = wide(torch.rand(2, 3, 160, 256), torch.as_tensor(np.stack([tokenize("pick the ripe lychee")] * 2)))
    assert out.shape == (2, MAP_HW[0] * MAP_HW[1])
