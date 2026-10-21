import numpy as np
import torch

from pilot import env as E
from experiments import seq as S


def _batch(n=64, seed=0, **kw):
    return S.sample_seq_batch(np.random.default_rng(seed), n, E.EnvConfig(**kw))


def test_sampler_matches_pilot_env():
    b = _batch(200, seed=3)
    ref = E.sample_batch(np.random.default_rng(3), 200, E.EnvConfig())
    assert np.array_equal(b.truth, ref.truth) and np.array_equal(b.covered, ref.covered)
    assert np.array_equal(b.stub_ok, ref.stub_ok) and np.array_equal(b.pattern, ref.pattern)
    rows = np.arange(200)[:, None]
    assert (b.tokens[rows, b.broken] == E.TOK_BROKEN).all()
    assert ((b.tokens == E.TOK_BROKEN).sum(1) == 4).all()
    assert np.array_equal(b.flags[rows, b.broken], np.where(b.covered, 2, 1))
    assert ((b.flags > 0).sum(1) == 4).all()
    # intact tokens follow the pattern (a + b*i) mod V with b >= 1
    intact = b.tokens != E.TOK_BROKEN
    assert (b.tokens[intact] < E.V).all()


def test_test_tokens_mark_covered_and_some_intact_branches():
    b = _batch(500, seed=1)
    n_intact_tested = 0
    for i in range(len(b)):
        named = set(b.tests[i][b.tests[i] != S.TEST_PAD].tolist())
        broken = set(b.broken[i].tolist())
        covered = set(b.broken[i][b.covered[i]].tolist())
        assert named & broken == covered                 # exactly the covered broken branches
        n_intact_tested += len(named - broken)
        assert len(named) == (b.tests[i] != S.TEST_PAD).sum()   # no duplicates
    frac = n_intact_tested / (500 * (E.K - 4))
    assert 0.4 < frac < 0.6                              # intact branches tested w.p. 0.5


def test_policy_shapes_both_variants():
    b = _batch(8, horizon=6)
    for variant in ("flags", "tests"):
        pol = S.make_policy(S.SeqConfig(variant=variant, d_model=16, seed=0))
        logits, hidden = pol(b)
        assert logits.shape == (8, 6, E.N_ACTIONS) and hidden.shape == (8, 6, 16)


def test_tests_variant_ignores_padding():
    b = _batch(4)
    pol = S.make_policy(S.SeqConfig(variant="tests", d_model=16))
    with torch.no_grad():
        l1, _ = pol(b)
        pol.test.weight[S.TEST_PAD] += 5.0               # pad embedding must not matter
        l2, _ = pol(b)
    assert torch.allclose(l1, l2, atol=1e-5)


def test_auroc_rank_statistic():
    assert S.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert S.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert S.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert S.auroc([1, 2, 3], [1, 1, 1]) is None


def test_tiny_run_is_deterministic():
    cfg = S.SeqConfig(d_model=16, sft_steps=5, sft_batch=32, rl_steps=4, rl_batch=8,
                      eval_every=2, eval_size=50, seed=7)
    a, b = S.run(cfg), S.run(cfg)
    assert a["log"] == b["log"]
    for p, q in zip(a["policy"].parameters(), b["policy"].parameters()):
        assert torch.equal(p, q)
