import numpy as np

from pilot import env as E
from pilot.policy import MLP, log_softmax
from pilot.train import RunConfig, reward_fn, run


def test_classify_outcomes():
    truth = np.array([3, 3, 3, 3, 3, 0])
    acts = np.array([3, 4, 7, E.DELETE, E.STUB, 9])
    assert list(E.classify(acts, truth)) == [E.CORRECT, E.PLAUSIBLE, E.BUILD_FAIL,
                                             E.DELETED, E.STUBBED, E.PLAUSIBLE]


def test_deleting_an_untested_branch_passes_visible_but_not_hidden():
    out = np.array([[E.CORRECT, E.DELETED]])
    covered = np.array([[True, False]])
    assert E.visible_pass(out, covered)[0]
    assert not E.truly_correct(out)[0]
    # The same deletion on a tested branch is caught by the regression suite.
    assert not E.visible_pass(out, np.array([[True, True]]))[0]


def test_stub_fails_build_where_it_does_not_type_check():
    acts = np.array([E.STUB, E.STUB])
    assert list(E.classify(acts, np.array([1, 1]), np.array([True, False]))) == [E.STUBBED, E.BUILD_FAIL]


def test_build_failure_fails_visible_check_everywhere():
    out = np.array([[E.CORRECT, E.BUILD_FAIL]])
    assert not E.visible_pass(out, np.array([[False, False]]))[0]


def test_regular_branches_are_inferable_from_neighbours():
    rng = np.random.default_rng(0)
    b = E.sample_batch(rng, 500, E.EnvConfig(horizon=1, p_irregular=0.0))
    assert (b.truth == b.pattern).all()
    obs = b.obs[:, 0, :]
    tok = lambda k: obs[:, k * E.N_TOK:(k + 1) * E.N_TOK].argmax(1)
    l2, l1 = tok(0), tok(1)
    ok = (l2 < E.V) & (l1 < E.V)
    assert ((2 * l1[ok] - l2[ok]) % E.V == b.truth[ok, 0]).all()


def test_audit_rewards():
    cfg = RunConfig(reward="audit", audit_rate=1.0)
    out = np.array([[[E.CORRECT, E.DELETED], [E.CORRECT, E.CORRECT], [E.BUILD_FAIL, E.CORRECT]]])
    cov = np.array([[[True, False]] * 3])
    r = reward_fn(cfg, np.random.default_rng(0), out, cov)
    assert list(r[0]) == [-1.0, 1.0, 0.0]


def test_backward_matches_finite_differences():
    rng = np.random.default_rng(0)
    m = MLP(rng, hidden=8)
    x = rng.random((5, E.OBS_DIM))
    a = rng.integers(0, E.N_ACTIONS, 5)
    loss = lambda: -log_softmax(m.forward(x)[0])[np.arange(5), a].sum()
    logits, cache = m.forward(x)
    d = np.exp(log_softmax(logits))
    d[np.arange(5), a] -= 1
    g = m.backward(d, cache)
    for k in ("W1", "W2", "b2"):
        idx = tuple(rng.integers(0, s) for s in m.params[k].shape)
        old = m.params[k][idx]
        m.params[k][idx] = old + 1e-5; up = loss()
        m.params[k][idx] = old - 1e-5; dn = loss()
        m.params[k][idx] = old
        assert abs((up - dn) / 2e-5 - g[k][idx]) < 1e-4


def test_runs_are_reproducible():
    cfg = RunConfig(sft_steps=20, rl_steps=10, eval_every=5, eval_size=200, seed=3)
    assert run(cfg)["log"] == run(cfg)["log"]
