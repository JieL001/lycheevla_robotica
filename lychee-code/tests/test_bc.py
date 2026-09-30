import numpy as np
import pytest
torch = pytest.importorskip("torch")
from lychee.bc import BCNet, VOCAB, tokenize, CHUNK, ACT_DIM, random_shift, MAXLEN
from lychee.lang import render, Spec


def test_vocab_covers_all_templates_of_both_forms():
    for form in "AB":
        for spec in [Spec("mat", 2), Spec("mat_any", 0), Spec("mat_side", 1, side="left"), Spec("mat_depth", 2, depth="far"),
                     Spec("mat_ordinal", 0, side="right", k=3)]:
            for v in range(2):
                for w in range(2):
                    tok = tokenize(render(spec, form=form, variant=v, word=w, avoid=1))
                    assert (tok == 1).sum() == 0, (form, spec, v, w)           # no <unk>
                    assert tok.shape == (MAXLEN,)


@pytest.mark.parametrize("film", [False, True])
def test_forward_shapes_and_gradients(film):
    net = BCNet(len(VOCAB), film=film)
    B = 3
    base, hand = torch.rand(B, 3, 100, 160), torch.rand(B, 3, 96, 96)
    out = net(random_shift(base), hand, torch.rand(B, 8), torch.as_tensor(np.stack([tokenize("pick any ripe lychee")] * B)))
    assert out.shape == (B, CHUNK, ACT_DIM)
    out.abs().mean().backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad and p.numel() > 0)


def test_language_reaches_the_output():
    torch.manual_seed(0)
    net = BCNet(len(VOCAB), film=True).eval()
    base, hand, prop = torch.rand(1, 3, 100, 160), torch.rand(1, 3, 96, 96), torch.rand(1, 8)
    a = net(base, hand, prop, torch.as_tensor(tokenize("pick the ripe lychee on the left"))[None])
    b = net(base, hand, prop, torch.as_tensor(tokenize("pick the ripe lychee on the right"))[None])
    assert (a - b).abs().max() > 1e-6


def _batch(B=3):
    return (torch.rand(B, 3, 100, 160), torch.rand(B, 3, 96, 96), torch.rand(B, 8),
            torch.as_tensor(np.stack([tokenize("pick any ripe lychee")] * B)))


def test_ground_variant_shapes_and_heatmap_size():
    from lychee.bc import ground_loss, FMAP_HW
    net = BCNet(len(VOCAB), film=False, ground=True, inject=True, lang_direct=False)
    out, s = net(*_batch(), return_ground=True)
    assert out.shape == (3, CHUNK, ACT_DIM) and tuple(s.shape) == (3,) + FMAP_HW
    tgt = torch.zeros(3, *FMAP_HW); tgt[:, 5, 7] = 1
    loss = ground_loss(s, tgt) + out.abs().mean()
    loss.backward()
    assert net.ghead.Wq.weight.grad is not None and net.inj.gamma.grad is not None


def test_gate_is_closed_at_init_so_grounded_tokens_have_no_effect():
    net = BCNet(len(VOCAB), film=False, ground=True, inject=True, lang_direct=False).eval()
    assert float(torch.tanh(net.inj.gamma)) == 0.0
    f = torch.rand(2, net.head[0].in_features)
    G1, G2 = torch.rand(2, 4, net.base.c_out + 2), torch.rand(2, 4, net.base.c_out + 2)
    assert torch.allclose(net.inj(f, G1), net.inj(f, G2))                    # tanh(0) = 0: identity at initialisation
    base, hand, prop, _ = _batch(1)
    a = net(base, hand, prop, torch.as_tensor(tokenize("pick the ripe lychee on the left"))[None])
    b = net(base, hand, prop, torch.as_tensor(tokenize("pick the unripe lychee on the right"))[None])
    assert torch.allclose(a, b)                     # no language shortcut: with the gate closed the command cannot matter


def test_opening_the_gate_lets_the_command_matter_through_the_heatmap_only():
    torch.manual_seed(1)
    net = BCNet(len(VOCAB), film=False, ground=True, inject=True, lang_direct=False).eval()
    net.inj.gamma.data.fill_(1.0)
    for p in net.inj.o.parameters():
        p.data.normal_(0, 0.5)
    base, hand, prop, _ = _batch(1)
    a = net(base, hand, prop, torch.as_tensor(tokenize("pick the ripe lychee on the left"))[None])
    b = net(base, hand, prop, torch.as_tensor(tokenize("pick the unripe lychee on the right"))[None])
    assert (a - b).abs().max() > 1e-6


def test_supervision_only_variant_ignores_the_heatmap_at_action_time():
    net = BCNet(len(VOCAB), film=False, ground=True, inject=False, lang_direct=False).eval()
    net.inj.gamma.data.fill_(1.0)
    base, hand, prop, _ = _batch(1)
    a = net(base, hand, prop, torch.as_tensor(tokenize("pick the ripe lychee on the left"))[None])
    b = net(base, hand, prop, torch.as_tensor(tokenize("pick the unripe lychee on the right"))[None])
    assert torch.allclose(a, b)
