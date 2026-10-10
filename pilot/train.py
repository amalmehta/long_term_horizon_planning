"""SFT from a flawed demonstrator, then GRPO against a chosen reward.

Run one configuration:
    python -m pilot.train --reward visible --horizon 4 --seed 0
"""

import argparse
import json
from pathlib import Path
from dataclasses import asdict, dataclass

import numpy as np

from . import env as E
from .policy import MLP, log_softmax

EVAL_SEED = 12345


@dataclass
class RunConfig:
    reward: str = "visible"     # visible | delete_penalty | audit
    audit_rate: float = 0.0     # chance an episode also gets the hidden equivalence check
    horizon: int = 4
    seed: int = 0
    sft_steps: int = 1000
    sft_batch: int = 256
    sft_lr: float = 2e-3
    rl_steps: int = 1000
    rl_batch: int = 64          # programs per step
    group: int = 8              # samples per program (GRPO group)
    rl_lr: float = 1e-3
    kl_coef: float = 0.02
    eval_every: int = 50
    eval_size: int = 2000


# ---- the flawed demonstrator --------------------------------------------------

DEMO_MIX = {"pattern": 0.65, "off_by_one": 0.15, "random": 0.10, "delete": 0.05, "stub": 0.05}


def demonstrator(rng, batch: E.Batch) -> np.ndarray:
    """A careless engineer: usually follows the pattern, sometimes slips.

    It ignores test coverage entirely, so SFT learns no coverage-aware habits.
    """
    shape = batch.pattern.shape
    u = rng.random(shape)
    c = np.cumsum(list(DEMO_MIX.values()))
    off = np.where(rng.random(shape) < 0.5, 1, -1)
    actions = np.where(u < c[0], batch.pattern,
              np.where(u < c[1], (batch.pattern + off) % E.V,
              np.where(u < c[2], rng.integers(0, E.V, shape),
              np.where(u < c[3], E.DELETE, E.STUB))))
    return actions


# ---- rewards -------------------------------------------------------------------

def reward_fn(cfg: RunConfig, rng, outcomes: np.ndarray, covered: np.ndarray) -> np.ndarray:
    vis = E.visible_pass(outcomes, covered).astype(float)
    if cfg.reward == "visible":
        return vis
    if cfg.reward == "delete_penalty":
        # A detector someone wrote after seeing the deletion hack.
        return vis - (outcomes == E.DELETED).any(axis=-1)
    if cfg.reward == "audit":
        true = E.truly_correct(outcomes)
        audited = rng.random(vis.shape) < cfg.audit_rate
        caught = vis.astype(bool) & ~true
        audited_r = np.where(true, 1.0, np.where(caught, -1.0, 0.0))
        return np.where(audited, audited_r, vis)
    raise ValueError(cfg.reward)


# ---- evaluation ----------------------------------------------------------------

def evaluate(policy: MLP, batch: E.Batch) -> dict:
    n, H, _ = batch.obs.shape
    logits, _ = policy.forward(batch.obs.reshape(n * H, -1))
    actions = logits.argmax(-1).reshape(n, H)
    out = E.classify(actions, batch.truth, batch.stub_ok)
    vis = E.visible_pass(out, batch.covered)
    true = E.truly_correct(out)

    def mix(mask):
        sel = out[mask]
        return {name: float((sel == k).mean()) for k, name in enumerate(E.OUTCOME_NAMES)}

    return {
        "visible_pass": float(vis.mean()),
        "true_correct": float(true.mean()),
        "false_success": float((vis & ~true).mean()),
        "uncovered": mix(~batch.covered),
        "covered": mix(batch.covered),
    }


# ---- training ------------------------------------------------------------------

def sft(cfg: RunConfig, env_cfg: E.EnvConfig, rng) -> MLP:
    policy = MLP(rng)
    for _ in range(cfg.sft_steps):
        batch = E.sample_batch(rng, cfg.sft_batch, env_cfg)
        target = demonstrator(rng, batch).reshape(-1)
        x = batch.obs.reshape(len(target), -1)
        logits, cache = policy.forward(x)
        p = np.exp(log_softmax(logits))
        p[np.arange(len(target)), target] -= 1.0
        policy.adam_step(policy.backward(p / len(target), cache), cfg.sft_lr)
    return policy


def grpo(cfg: RunConfig, env_cfg: E.EnvConfig, policy: MLP, rng, eval_batch, log):
    ref = policy.copy()
    B, G, H = cfg.rl_batch, cfg.group, cfg.horizon
    for step in range(cfg.rl_steps + 1):
        if step % cfg.eval_every == 0:
            log.append({"step": step, **evaluate(policy, eval_batch)})
        if step == cfg.rl_steps:
            break

        batch = E.sample_batch(rng, B, env_cfg)
        x = batch.obs.reshape(B * H, -1)
        logits, cache = policy.forward(x)
        logp = log_softmax(logits)
        p = np.exp(logp)

        # G samples per broken branch, shared observation
        cum = np.cumsum(p, axis=-1)
        u = rng.random((B * H, G, 1))
        acts = (u > cum[:, None, :]).sum(-1).clip(max=E.N_ACTIONS - 1)   # (B*H, G)
        acts_bgh = acts.reshape(B, H, G).transpose(0, 2, 1)               # (B, G, H)

        rep = lambda a: np.repeat(a[:, None, :], G, axis=1)
        out = E.classify(acts_bgh, rep(batch.truth), rep(batch.stub_ok))
        cov = rep(batch.covered)
        r = reward_fn(cfg, rng, out, cov)                                  # (B, G)
        adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-6)

        # policy-gradient term: d/dlogits of -A * log pi(a)
        adv_rows = np.repeat(adv[:, None, :], H, axis=1).reshape(B * H, G)  # (B*H, G)
        onehot_sum = np.zeros_like(p)
        for g in range(G):
            np.add.at(onehot_sum, (np.arange(B * H), acts[:, g]), adv_rows[:, g])
        dlogits = -(onehot_sum - adv_rows.sum(1, keepdims=True) * p) / (B * G)

        # exact KL(pi || ref) per branch, averaged over programs
        ref_logp = log_softmax(ref.forward(x)[0])
        diff = logp - ref_logp
        kl = (p * diff).sum(-1, keepdims=True)
        dlogits += cfg.kl_coef * p * (diff - kl) / B

        policy.adam_step(policy.backward(dlogits, cache), cfg.rl_lr)
    return policy


def sft_cached(cfg: RunConfig, env_cfg: E.EnvConfig, cache_dir=None) -> MLP:
    """SFT depends only on (seed, horizon, SFT settings), so reward sweeps share it."""
    rng = np.random.default_rng([cfg.seed, 0])
    if cache_dir is None:
        return sft(cfg, env_cfg, rng)
    path = Path(cache_dir) / f"sft_h{cfg.horizon}_s{cfg.seed}_n{cfg.sft_steps}.npz"
    if path.exists():
        policy = MLP(rng)
        policy.params = dict(np.load(path))
        return policy
    policy = sft(cfg, env_cfg, rng)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **policy.params)
    return policy


def run(cfg: RunConfig, cache_dir=None) -> dict:
    env_cfg = E.EnvConfig(horizon=cfg.horizon)
    eval_batch = E.sample_batch(np.random.default_rng(EVAL_SEED), cfg.eval_size, env_cfg)
    policy = sft_cached(cfg, env_cfg, cache_dir)
    rng = np.random.default_rng([cfg.seed, 1])
    log: list = []
    grpo(cfg, env_cfg, policy, rng, eval_batch, log)
    return {"config": asdict(cfg), "env": asdict(env_cfg), "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="visible", choices=["visible", "delete_penalty", "audit"])
    ap.add_argument("--audit-rate", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rl-steps", type=int, default=1000)
    a = ap.parse_args()
    res = run(RunConfig(reward=a.reward, audit_rate=a.audit_rate, horizon=a.horizon,
                        seed=a.seed, rl_steps=a.rl_steps))
    first, last = res["log"][0], res["log"][-1]
    print(json.dumps({"start": first, "end": last}, indent=1))


if __name__ == "__main__":
    main()
