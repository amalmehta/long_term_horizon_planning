import numpy as np

from experiments import engine as X
from pilot import env as E
from pilot.train import RunConfig, run


def _pilot_vs_engine(reward, audit_rate):
    small = dict(sft_steps=30, rl_steps=20, eval_every=10, eval_size=300, seed=4)
    ref = run(RunConfig(reward=reward, audit_rate=audit_rate, **small))["log"]
    c = X.XConfig(reward="visible" if reward == "audit" else reward, audit_rate=audit_rate, **small)
    policy = X.sft(c, np.random.default_rng([c.seed, 0]))
    _, _, res = X.train(c, policy=policy)
    keys = ("visible_pass", "true_correct", "false_success")
    return [[e[k] for k in keys] for e in ref], [[e[k] for k in keys] for e in res["log"]]


def test_engine_reproduces_pilot_visible():
    a, b = _pilot_vs_engine("visible", 0.0)
    assert a == b


def test_engine_reproduces_pilot_audit():
    a, b = _pilot_vs_engine("audit", 0.25)
    assert a == b


def test_sampler_matches_pilot_for_default_config():
    b0 = E.sample_batch(np.random.default_rng(7), 50, E.EnvConfig())
    b1 = X.sample(np.random.default_rng(7), 50, X.XConfig())
    assert np.array_equal(b0.obs, b1.obs) and np.array_equal(b0.truth, b1.truth)


def test_abstain_is_honest_but_not_correct():
    out = np.array([[[E.CORRECT, X.ABSTAINED]]])
    vis, true, honest = X.verdicts(out, np.array([[[True, True]]]))
    assert vis[0, 0] and honest[0, 0] and not true[0, 0]


def test_targeted_audits_spend_budget_on_suspicious_first():
    c = X.XConfig(audit_rate=0.25, audit_mode="targeted")
    out = np.full((2, 4, 1), E.CORRECT)
    out[0, 1, 0] = E.DELETED
    cov = np.zeros((2, 4, 1), bool)
    mask = X.audit_mask(c, np.random.default_rng(0), out, cov)
    assert mask.sum() == 2 and mask[0, 1]


def test_memory_flag_hides_coverage_beyond_k():
    c = X.XConfig(flag="memory", memory_k=2)
    b = X.sample(np.random.default_rng(0), 100, c)
    flag, known = b.obs[:, :, E.OBS_DIM - 1], b.obs[:, :, E.OBS_DIM]
    assert (known[:, :2] == 1).all() and (known[:, 2:] == 0).all()
    assert (flag[:, 2:] == 0).all() and (flag[:, :2] == b.covered[:, :2]).all()
