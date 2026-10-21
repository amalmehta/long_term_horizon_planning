import numpy as np

from pilot import env as E
from pilot.policy import MLP
from experiments import multiagent as M


def _setup(seed=0, R=6, G=3):
    rng = np.random.default_rng(seed)
    planner = MLP(rng, hidden=8, in_dim=E.OBS_DIM, out_dim=M.N_MSG)
    executor = MLP(rng, hidden=8, in_dim=M.EXEC_IN, out_dim=E.N_ACTIONS)
    for net in (planner, executor):
        net.params["W2"] = rng.normal(0, 0.5, net.params["W2"].shape)   # non-trivial softmax
    ref = executor.copy()
    ref.params["b2"] = rng.normal(0, 0.5, ref.params["b2"].shape)      # KL != 0
    batch = E.sample_batch(rng, R, E.EnvConfig(horizon=1))
    obs = batch.obs.reshape(R, -1)
    msgs = rng.integers(0, M.N_MSG, (R, G))
    acts = rng.integers(0, E.N_ACTIONS, (R, G))
    adv = rng.normal(size=(R, G))
    return planner, executor, ref, obs, msgs, acts, adv


def test_joint_loss_gradient_matches_finite_differences():
    planner, executor, ref, obs, msgs, acts, adv = _setup()
    args = (obs, msgs, acts, adv, 0.3, 2)
    _, gp, ge = M.joint_loss_and_grads(planner, executor, ref, *args)
    rng = np.random.default_rng(1)
    h = 1e-6
    for net, grads in ((planner, gp), (executor, ge)):
        for k in ("W1", "b1", "W2", "b2"):
            for _ in range(4):
                idx = tuple(rng.integers(0, s) for s in net.params[k].shape)
                old = net.params[k][idx]
                net.params[k][idx] = old + h
                lp = M.joint_loss_and_grads(planner, executor, ref, *args)[0]
                net.params[k][idx] = old - h
                lm = M.joint_loss_and_grads(planner, executor, ref, *args)[0]
                net.params[k][idx] = old
                num = (lp - lm) / (2 * h)
                assert abs(num - grads[k][idx]) < 1e-5 * max(1.0, abs(num)), (k, num, grads[k][idx])


def test_message_sampling_and_executor_input_shapes():
    rng = np.random.default_rng(0)
    p = np.full((10, M.N_MSG), 0.5)
    m = M.sample_cat(rng, p, 8)
    assert m.shape == (10, 8) and set(np.unique(m)) <= {0, 1}
    obs = E.sample_batch(rng, 5, E.EnvConfig()).obs.reshape(20, -1)
    x = M.exec_input(obs, np.zeros(20, int))
    assert x.shape == (20, M.EXEC_IN)
    assert np.all(x[:, -2] == 1) and np.all(x[:, -1] == 0)
    assert np.array_equal(x[:, :E.OBS_DIM - 1], obs[:, :-1])   # covered flag dropped


def test_message_accuracy_best_mapping():
    cov = np.array([True, False, True, False])
    assert M.message_accuracy(np.array([1, 0, 1, 0]), cov) == 1.0
    assert M.message_accuracy(np.array([0, 1, 0, 1]), cov) == 1.0      # flipped map also counts
    assert M.message_accuracy(np.array([0, 0, 0, 0]), cov) == 0.5


def test_reviewer_feature_shapes_and_values():
    acts = np.array([[[3, E.DELETE, E.STUB, 7]]])
    cov = np.array([[[True, False, False, True]]])
    vis = np.array([[True]])
    full = M.reviewer_features("full", acts, cov, vis)
    assert full.shape == (1, 1, 24) and full.sum() == 4
    blocks = full.reshape(4, 6).argmax(-1)
    assert list(blocks) == [0 * 2 + 1, 1 * 2 + 0, 2 * 2 + 0, 0 * 2 + 1]
    summ = M.reviewer_features("summary", acts, cov, vis)
    assert summ.shape == (1, 1, 2 + 9) and summ.sum() == 2
    assert summ[0, 0, 1] == 1                       # visible pass
    assert summ[0, 0, 2 + 5] == 1                   # 1 + 2 + 1 + 1 lines
    assert M.lines_changed(np.array([E.DELETE] * 4)) == 8


def test_tiny_runs_are_deterministic():
    for exp in ("E45", "E46"):
        cfg = M.MAConfig(exp=exp, seed=0, sft_steps=5, rl_steps=4, eval_every=2,
                         eval_size=50, rl_batch=8)
        a, b = M.run_one(cfg), M.run_one(cfg)
        assert a["log"] == b["log"]
        assert len(a["log"]) == 3
