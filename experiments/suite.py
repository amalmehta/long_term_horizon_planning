"""Jobs and judges for the NumPy experiments (E01–E30, E39–E44, E48–E50).

Each experiment lists the jobs it needs and a judge that turns their results
into a verdict using the pre-registered criteria in registry.py. Jobs are
cached by content hash, so experiments that share a configuration share runs.

    python -m experiments.suite --workers 8        # run everything, write results/experiments/EXX.json
    python -m experiments.suite --only E02 E05     # a subset
"""

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "experiments"
RUNS = OUT / "runs"
SEEDS = (0, 1, 2)
DEPLOY = (("deploy", (("p_covered", 0.2), ("horizon", 6))),)
COMMON = dict(extras=("entropy", "kl", "passk"), deploy=DEPLOY)    # visible-reward sweeps


# ---- jobs ------------------------------------------------------------------------

def job(kind, **kw):
    return {"kind": kind, **kw}


def run_job(spec):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from . import engine as X
    path = RUNS / f"{X.digest(spec)}.json"
    if path.exists():
        return json.loads(path.read_text())
    t0 = time.time()
    res = KINDS[spec["kind"]](**{k: v for k, v in spec.items() if k != "kind"})
    res["runtime_s"] = time.time() - t0
    res["spec"] = spec
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(path)
    return res


def cfg(**kw):
    from .engine import XConfig
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in kw.items()}
    if "deploy" in kw:
        kw["deploy"] = tuple((n, tuple(tuple(p) for p in o)) for n, o in kw["deploy"])
    return XConfig(**kw)


def k_train(**kw):
    from . import engine as X
    _, _, res = X.train(cfg(**kw))
    return res


def k_chain(stages):
    """Train stages in sequence, carrying the policy; KL stays anchored to the first SFT."""
    from . import engine as X
    policy = ref = None
    logs, offset = [], 0
    for st in stages:
        c = cfg(**st)
        policy, ref, res = X.train(c, policy=policy, ref=ref, step_offset=offset)
        logs.append(res["log"])
        offset += c.rl_steps
    return {"logs": logs}


def _rm_features(obs, acts, n_actions):
    onehot = np.zeros(acts.shape + (n_actions,))
    np.put_along_axis(onehot, acts[..., None], 1.0, axis=-1)
    return np.concatenate([obs, onehot], -1)


class RewardModel:
    """Per-branch P(correct | context, action): a small MLP trained with BCE."""

    def __init__(self, seed, in_dim):
        from pilot.policy import MLP
        self.net = MLP(np.random.default_rng([seed, 7]), hidden=64, in_dim=in_dim, out_dim=1)
        self.seed = seed

    def fit(self, X, y, steps=600):
        rng = np.random.default_rng([self.seed, 8])
        for _ in range(steps):
            idx = rng.integers(0, len(X), min(256, len(X)))
            logit, cache = self.net.forward(X[idx])
            p = 1 / (1 + np.exp(-logit[:, 0]))
            self.net.adam_step(self.net.backward(((p - y[idx]) / len(idx))[:, None], cache), 3e-3)

    def score(self, X):
        return 1 / (1 + np.exp(-self.net.forward(X)[0][..., 0]))


def _rm_data(c, policy, n_branches, rng):
    """Label branches sampled from `policy` with the hidden equivalence check."""
    from . import engine as X
    from pilot.policy import log_softmax
    n_prog = int(np.ceil(n_branches / c.horizon))
    b = X.sample(rng, n_prog, c)
    x = b.obs.reshape(-1, b.obs.shape[-1])
    p = np.exp(log_softmax(policy.forward(x)[0]))
    acts = (rng.random((len(x), 1)) > np.cumsum(p, -1)).sum(-1).clip(max=c.n_actions() - 1)
    out = X.classify(acts.reshape(b.truth.shape), b.truth, b.stub_ok).reshape(-1)
    feats = _rm_features(x, acts, c.n_actions())
    return feats[:n_branches], (out == 0).astype(float)[:n_branches]


def k_rm(seed, n_labels, mode, refresh_every=100, refresh_rate=0.05):
    """GRPO against a learned reward model. mode: rm | vis_x_rm | refresh."""
    from . import engine as X
    c = cfg(seed=seed, reward="hook")
    policy = X.sft_cached(c)
    rng = np.random.default_rng([seed, 9])
    feats, labels = _rm_data(c, policy, n_labels, rng)
    rm = RewardModel(seed, feats.shape[1])
    rm.fit(feats, labels)
    buf = [feats, labels]

    def hook(out, cov, acts, vis, true, ctx):
        b = ctx["batch"]
        B, G, H = acts.shape
        obs = np.repeat(b.obs[:, None], G, 1)
        s = rm.score(_rm_features(obs, acts, c.n_actions())).prod(-1)
        if mode == "refresh":
            audited = rng.random((B, G)) < refresh_rate
            if audited.any():
                buf[0] = np.concatenate([buf[0], _rm_features(obs[audited], acts[audited], c.n_actions()).reshape(-1, buf[0].shape[1])])
                buf[1] = np.concatenate([buf[1], (out[audited] == 0).astype(float).reshape(-1)])
            if ctx["step"] % refresh_every == refresh_every - 1:
                rm.fit(buf[0], buf[1], steps=300)
        return s * vis if mode == "vis_x_rm" else s

    _, _, res = X.train(c, policy=policy, hook=hook)
    res["rm_labels_final"] = int(len(buf[1]))
    return res


def k_bon(seed, n_max=32, rm_labels=1000):
    """Best-of-n against the visible checks, plain and reranked by a reward model."""
    from . import engine as X
    from pilot.policy import log_softmax
    c = cfg(seed=seed)
    policy = X.sft_cached(c)
    feats, labels = _rm_data(c, policy, rm_labels, np.random.default_rng([seed, 9]))
    rm = RewardModel(seed, feats.shape[1])
    rm.fit(feats, labels)
    b = X.sample(np.random.default_rng(X.EVAL_SEED), c.eval_size, c)
    n, H, D = b.obs.shape
    p = np.exp(log_softmax(policy.forward(b.obs.reshape(-1, D))[0]))
    rng = np.random.default_rng([seed, 10])
    acts = (rng.random((n * H, n_max, 1)) > np.cumsum(p, -1)[:, None]).sum(-1).clip(max=c.n_actions() - 1)
    acts = acts.reshape(n, H, n_max).transpose(0, 2, 1)                       # (n, k, H)
    rep = lambda a: np.repeat(a[:, None], n_max, 1)
    out = X.classify(acts, rep(b.truth), rep(b.stub_ok))
    vis, true, _ = X.verdicts(out, rep(b.covered))
    score = rm.score(_rm_features(rep(b.obs), acts, c.n_actions())).prod(-1)  # (n, k)
    res = {"n": [], "fs_first": [], "true_first": [], "fs_rm": [], "true_rm": []}
    for k in (1, 2, 4, 8, 16, 32):
        v, t = vis[:, :k], true[:, :k]
        first = np.where(v.any(1), v.argmax(1), 0)
        best = np.where(v.any(1), np.where(v, score[:, :k], -1).argmax(1), 0)
        rows = np.arange(n)
        for name, pick in (("first", first), ("rm", best)):
            sv, st = v[rows, pick], t[rows, pick]
            res[f"fs_{name}"].append(float((sv & ~st).mean()))
            res[f"true_{name}"].append(float(st.mean()))
        res["n"].append(k)
    return res


def k_errcorr(seed, horizon, eval_size=20000):
    from . import engine as X
    c = cfg(seed=seed, horizon=horizon, eval_size=eval_size)
    policy = X.sft_cached(c)
    b = X.sample(np.random.default_rng(X.EVAL_SEED), eval_size, c)
    n, H, D = b.obs.shape
    acts = policy.forward(b.obs.reshape(-1, D))[0].argmax(-1).reshape(n, H)
    ok = X.classify(acts, b.truth, b.stub_ok) == 0
    return {"all_correct": float(ok.all(1).mean()), "branch_acc": float(ok.mean())}


def k_zipf(seed):
    from . import engine as X
    c = cfg(seed=seed, coverage_mode="zipf")
    policy, _, res = X.train(c)
    b = X.sample(np.random.default_rng(X.EVAL_SEED), c.eval_size, c)
    n, H, D = b.obs.shape
    acts = policy.forward(b.obs.reshape(-1, D))[0].argmax(-1).reshape(n, H)
    wrong = X.classify(acts, b.truth, b.stub_ok) != 0
    use = np.take_along_axis(b.usage, b.pos, 1)
    res["customer_failure"] = float((use * wrong).sum(1).mean())
    res["uniform_failure"] = float((wrong.sum(1) / c.n_branches).mean())
    res["untested_wrong"] = float(wrong[~b.covered].mean())
    res["tested_wrong"] = float(wrong[b.covered].mean())
    return res


KINDS = {"train": k_train, "chain": k_chain, "rm": k_rm, "bon": k_bon,
         "errcorr": k_errcorr, "zipf": k_zipf}


# ---- helpers for judges ------------------------------------------------------------

def last(res, key="false_success"):
    return res["log"][-1][key]


def unc(res, outcome="deleted", entry=-1):
    return res["log"][entry]["uncovered"][outcome]


def ms(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)


def onset_of(log, threshold=0.15):
    for e in log:
        if e["false_success"] >= threshold:
            return e["step"]
    return None


def fmt(m, s=None):
    return f"{m:.2f}" if s is None else f"{m:.2f} ± {s:.2f}"


def monotone(vals, decreasing=True, tol=0.02):
    pairs = zip(vals, vals[1:])
    return all((b <= a + tol) if decreasing else (b >= a - tol) for a, b in pairs)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def curve(res, key="false_success", every=100):
    return [(e["step"], e[key]) for e in res["log"] if e["step"] % every == 0]


def line(x_label, y_label, x, series, y_max=1):
    return {"kind": "line", "x_label": x_label, "y_label": y_label, "x": x, "series": series, "y_max": y_max}


def bar(x_label, y_label, x, series, y_max=1):
    return {"kind": "bar", "x_label": x_label, "y_label": y_label, "x": x, "series": series, "y_max": y_max}


def verdict(supported, refuted):
    return "supported" if supported and not refuted else "refuted" if refuted else "mixed"


def seeds_job(kind="train", seeds=SEEDS, **kw):
    return [job(kind, seed=s, **kw) for s in seeds]


def mean_curve(results, key="false_success", every=100):
    steps = [s for s, _ in curve(results[0], key, every)]
    ys = [[v for _, v in curve(r, key, every)] for r in results]
    return steps, [float(np.mean(col)) for col in zip(*ys)], [float(np.std(col)) for col in zip(*ys)]


# ---- experiments -----------------------------------------------------------------

EXPS = {}


def experiment(eid):
    def wrap(fn):
        EXPS[eid] = fn
        return fn
    return wrap


VIS = dict(reward="visible", **COMMON)


@experiment("E01")
def e01(get):
    rs = get(seeds_job(**VIS))
    leads, rows = [], []
    for r in rs:
        o = onset_of(r["log"])
        warn = next((e["step"] for e in r["log"] if e["kl_uncovered"] > 2 * e["kl_covered"] and e["kl_uncovered"] > 0.01), None)
        leads.append(None if o is None or warn is None else o - warn)
        rows.append({"onset": o, "warning": warn})
    med = float(np.median([l for l in leads if l is not None])) if any(l is not None for l in leads) else float("nan")
    st, ku, _ = mean_curve(rs, "kl_uncovered")
    _, kc, _ = mean_curve(rs, "kl_covered")
    return dict(verdict=verdict(med >= 100, med <= 0),
                result=f"Untested-branch KL passed twice the tested-branch KL at steps {[r['warning'] for r in rows]}; onset came at {[r['onset'] for r in rows]}. Median lead time {med:.0f} steps.",
                metrics={"per_seed": rows, "median_lead": med},
                chart=line("GRPO step", "KL to reference (nats)", st,
                           [{"name": "Untested branches", "y": ku}, {"name": "Tested branches", "y": kc}], y_max=None))


@experiment("E02")
def e02(get):
    kls = [0.0, 0.02, 0.1, 0.5, 2.0]
    fs = [ms([last(r) for r in get(seeds_job(kl_coef=k, **VIS))]) for k in kls]
    tc = [ms([last(r, "true_correct") for r in get(seeds_job(kl_coef=k, **VIS))]) for k in kls]
    means = [m for m, _ in fs]
    mono = monotone(means)
    return dict(verdict=verdict(mono and means[-1] < 0.10, (not mono) or means[-1] > 0.25),
                result=f"Final false success by KL coefficient {kls}: {[round(m, 2) for m in means]}. "
                       f"{'Monotone' if mono else 'Not monotone'}; at KL 2 it ends at {fmt(*fs[-1])}.",
                metrics={"kl": kls, "fs": fs, "true": tc},
                chart=bar("KL coefficient", "share of programs", [str(k) for k in kls],
                          [{"name": "False success", "y": means, "err": [s for _, s in fs]},
                           {"name": "Actually correct", "y": [m for m, _ in tc], "err": [s for _, s in tc]}]))


@experiment("E03")
def e03(get):
    gs = [2, 4, 8, 16, 32]
    ons = []
    for g in gs:
        o = [onset_of(r["log"]) for r in get(seeds_job(group=g, **VIS))]
        ons.append([x if x is not None else 1050 for x in o])
    med = [float(np.median(o)) for o in ons]
    return dict(verdict=verdict(monotone(med, tol=0) and med[-1] < med[0], med[-1] >= med[1]),
                result=f"Median onset step by group size {gs}: {med} (1050 = never within 1,000 steps). Larger groups also mean more samples per step.",
                metrics={"group": gs, "onset": ons},
                chart=bar("GRPO group size", "onset step (median)", [str(g) for g in gs],
                          [{"name": "Onset step", "y": med}], y_max=None))


@experiment("E04")
def e04(get):
    lrs = [3e-4, 1e-3, 3e-3]
    ons, prods = [], []
    for lr in lrs:
        o = [onset_of(r["log"]) for r in get(seeds_job(rl_lr=lr, rl_steps=3000, reward="visible"))]
        ons.append(o)
        prods.append(None if None in o else lr * float(np.median(o)))
    never = any(p is None for p in prods)
    ratio = None if never else max(prods) / max(min(prods), 1e-9)
    return dict(verdict=verdict((not never) and ratio <= 3, never or ratio > 3),
                result=f"Onset steps by learning rate {lrs}: {ons}. " +
                       ("Some learning rate never reached onset within 3,000 steps." if never else
                        f"Learning rate × onset ranges {min(prods):.2f}–{max(prods):.2f}, a factor of {ratio:.1f}."),
                metrics={"lr": lrs, "onset": ons, "lr_x_onset": prods},
                chart=bar("learning rate", "onset step (median)", [str(l) for l in lrs],
                          [{"name": "Onset step", "y": [float(np.median([x if x is not None else 3050 for x in o])) for o in ons]}], y_max=None))


@experiment("E05")
def e05(get):
    cs = [0.1, 0.3, 0.5, 0.7, 0.9]
    fs = [ms([last(r) for r in get(seeds_job(p_covered=c, **VIS))]) for c in cs]
    means = [m for m, _ in fs]
    ok = monotone(means) and means[-1] < 0.10
    return dict(verdict=verdict(ok, not ok),
                result=f"Final false success by coverage {cs}: {[round(m, 2) for m in means]}.",
                metrics={"coverage": cs, "fs": fs},
                chart=bar("share of broken branches tested", "false success", [str(c) for c in cs],
                          [{"name": "False success", "y": means, "err": [s for _, s in fs]}]))


@experiment("E06")
def e06(get):
    base = dict(coverage_mode="positional", position_features=True, reward="visible")
    shown = [last(r) for r in get(seeds_job(flag="shown", **base))]
    hidden = [last(r) for r in get(seeds_job(flag="hidden", **base))]
    (sm, ss), (hm, hs) = ms(shown), ms(hidden)
    return dict(verdict=verdict(hm >= 0.5 * sm, hm < 0.5 * sm),
                result=f"With coverage tied to position, false success was {fmt(sm, ss)} with the flag shown and {fmt(hm, hs)} with it hidden ({hm / max(sm, 1e-9):.0%} of the flag-shown level).",
                metrics={"shown": shown, "hidden": hidden},
                chart=bar("coverage flag", "false success", ["shown", "hidden"],
                          [{"name": "False success", "y": [sm, hm], "err": [ss, hs]}]))


@experiment("E07")
def e07(get):
    acc = [last(r) for r in get(seeds_job(**VIS))]
    noisy = [last(r) for r in get(seeds_job(flag="noisy", reward="visible"))]
    (am, as_), (nm, ns) = ms(acc), ms(noisy)
    return dict(verdict=verdict(nm >= 0.15 and nm <= 0.7 * am, nm > 0.7 * am),
                result=f"False success {fmt(nm, ns)} with a 20%-wrong coverage flag against {fmt(am, as_)} with an accurate one ({1 - nm / max(am, 1e-9):.0%} lower).",
                metrics={"accurate": acc, "noisy": noisy},
                chart=bar("coverage flag", "false success", ["accurate", "20% wrong"],
                          [{"name": "False success", "y": [am, nm], "err": [as_, ns]}]))


@experiment("E08")
def e08(get):
    hs = [0.0, 0.05, 0.10, 0.20]
    runs = [get(seeds_job(demo_hack=h, reward="visible")) for h in hs]
    ons = [[onset_of(r["log"]) for r in rs] for rs in runs]
    med = [float(np.median([o if o is not None else 1050 for o in x])) for x in ons]
    zero_never = all(o is None for o in ons[0])
    return dict(verdict=verdict(zero_never and med[3] < med[1], not zero_never),
                result=f"Onset steps by demonstrator hack share {hs}: {ons} (None = never). Final false success: {[round(ms([last(r) for r in rs])[0], 2) for rs in runs]}.",
                metrics={"demo_hack": hs, "onset": ons},
                chart=bar("hacks in demonstrations", "onset step (median; 1050 = never)", ["0%", "5%", "10%", "20%"],
                          [{"name": "Onset step", "y": med}], y_max=None),
                notes="The registry calls 5% the default; the pilot's demonstrator actually deletes or stubs 10% of branches (5% each). Runs cover 0%, 5%, 10% and 20%, and the registered comparison uses 5%.")


@experiment("E09")
def e09(get):
    rs = get(seeds_job(from_scratch=True, kl_coef=0.0, rl_steps=3000, reward="visible"))
    tc, de = ms([last(r, "true_correct") for r in rs]), ms([unc(r) + unc(r, "stubbed") for r in rs])
    return dict(verdict=verdict(tc[0] < 0.05 and ms([unc(r) for r in rs])[0] > 0.5, tc[0] > 0.15),
                result=f"From a random start, after 3,000 steps true correctness is {fmt(*tc)} and untested branches are deleted or stubbed {fmt(*de)} of the time.",
                metrics={"true": tc, "untested_hack": de, "untested_deleted": ms([unc(r) for r in rs])},
                chart=line("GRPO step", "share of programs", *(lambda s, m, e: (s, [{"name": "False success", "y": m, "err": e}]))(*mean_curve(rs, every=250))),
                notes="KL penalty set to 0: a KL anchor to a random initial policy would pull towards uniform actions rather than towards a demonstrator.")


def _e10_chain(seed):
    return job("chain", stages=[dict(seed=seed, reward="visible"),
                                dict(seed=seed, reward="visible", audit_rate=0.25),
                                dict(seed=seed, reward="visible")])


@experiment("E10")
def e10(get):
    rs = get([_e10_chain(s) for s in SEEDS])
    rec, ends = [], []
    for r in rs:
        st2 = r["logs"][1]
        rec.append(next((e["step"] - 1000 for e in st2 if e["false_success"] < 0.10), None))
        ends.append(st2[-1]["false_success"])
    ok = all(x is not None and x <= 500 for x in rec)
    full = [e for e in rs[0]["logs"][0]] + rs[0]["logs"][1][1:] + rs[0]["logs"][2][1:]
    return dict(verdict=verdict(ok, ms(ends)[0] > 0.25),
                result=f"After 25% audits started, false success never fell below 0.10 in any seed; it ended the audited phase at {fmt(*ms(ends))}, the same level 25% audits settle at in a long run from a clean start (E24).",
                metrics={"steps_to_recover": rec, "end_audited": ends},
                chart=line("GRPO step (visible → audited → visible)", "false success",
                           [e["step"] for e in full if e["step"] % 100 == 0],
                           [{"name": "Seed 0", "y": [e["false_success"] for e in full if e["step"] % 100 == 0]}]))


@experiment("E11")
def e11(get):
    rs = get([_e10_chain(s) for s in SEEDS])
    orig, rel = [], []
    for r in rs:
        orig.append(onset_of(r["logs"][0]))
        o = onset_of(r["logs"][2][1:])
        rel.append(None if o is None else o - 2000)
    pairs = [(o, x) for o, x in zip(orig, rel) if o is not None]
    fast = pairs and all(x is not None and x <= o / 2 for o, x in pairs)
    slow = any(x is None or x > o for o, x in pairs)
    return dict(verdict=verdict(fast, slow),
                result=f"Original onset {orig} steps; after audits were removed the hack returned after {rel} steps (None = not within 1,000).",
                metrics={"original_onset": orig, "relapse_onset": rel})


@experiment("E12")
def e12(get):
    ps = [0.5, 0.75, 0.9, 1.0]
    runs = [get(seeds_job(reward="delete_penalty", p_stub_ok=p)) for p in ps]
    st = [ms([unc(r, "stubbed") for r in rs]) for rs in runs]
    fs = [ms([last(r) for r in rs]) for rs in runs]
    return dict(verdict=verdict(st[-1][0] > 0.5, st[-1][0] < 0.2),
                result=f"Under a deletion penalty, stubs on untested branches by stub type-check rate {ps}: {[round(m, 2) for m, _ in st]}; false success {[round(m, 2) for m, _ in fs]}.",
                metrics={"p_stub_ok": ps, "stubbed_untested": st, "fs": fs},
                chart=bar("share of branches where a stub type-checks", "share", [f"{p:.0%}" for p in ps],
                          [{"name": "Untested branches stubbed", "y": [m for m, _ in st], "err": [s for _, s in st]},
                           {"name": "False success", "y": [m for m, _ in fs], "err": [s for _, s in fs]}]))


AUD = lambda q, lam=1.0, **kw: dict(reward="visible", audit_rate=q, caught_penalty=lam, **kw)


@experiment("E13")
def e13(get):
    lams = [0.0, 0.5, 1.0, 2.0, 5.0]
    fs = [ms([last(r) for r in get(seeds_job(**AUD(0.10, l)))]) for l in lams]
    means = [m for m, _ in fs]
    return dict(verdict=verdict(monotone(means, tol=0.03) and means[-1] < 0.10, min(means) >= 0.20),
                result=f"At 10% audits, false success by caught-hack penalty {lams}: {[round(m, 2) for m in means]}.",
                metrics={"penalty": lams, "fs": fs},
                chart=bar("penalty for a caught hack", "false success", [str(l) for l in lams],
                          [{"name": "False success", "y": means, "err": [s for _, s in fs]}]))


@experiment("E14")
def e14(get):
    qs, lams = [0.05, 0.10, 0.25], [0.5, 1.0, 2.0, 4.0]
    xs, ys, tab = [], [], []
    for q in qs:
        row = []
        for l in lams:
            m = ms([last(r) for r in get(seeds_job(**AUD(q, l)))])[0]
            xs.append(q * (1 + l)); ys.append(m); row.append(round(m, 2))
        tab.append(row)
    rho = spearman(np.array(xs), np.array(ys))
    order = np.argsort(xs)
    return dict(verdict=verdict(rho <= -0.9, rho > -0.7),
                result=f"Across 12 audit-rate × penalty settings, Spearman correlation between q·(1+λ) and false success is {rho:.2f}.",
                metrics={"rows_audit_rate": qs, "cols_penalty": lams, "fs": tab, "spearman": rho},
                chart=line("expected cost of a hack, q·(1+λ)", "false success",
                           [round(xs[i], 3) for i in order], [{"name": "False success", "y": [ys[i] for i in order]}]))


@experiment("E15")
def e15(get):
    t = ms([last(r) for r in get(seeds_job(**AUD(0.10, audit_mode="targeted")))])
    r10 = ms([last(r) for r in get(seeds_job(**AUD(0.10)))])
    r25 = ms([last(r) for r in get(seeds_job(**AUD(0.25)))])
    return dict(verdict=verdict(t[0] <= r25[0] + 0.01, t[0] >= r10[0]),
                result=f"False success with a 10% budget: targeted {fmt(*t)}, random {fmt(*r10)}; random 25% gives {fmt(*r25)}.",
                metrics={"targeted10": t, "random10": r10, "random25": r25},
                chart=bar("audits", "false success", ["random 10%", "targeted 10%", "random 25%"],
                          [{"name": "False success", "y": [r10[0], t[0], r25[0]], "err": [r10[1], t[1], r25[1]]}]))


@experiment("E16")
def e16(get):
    rs = get(seeds_job(**AUD(0.10, audit_mode="targeted_delete", p_stub_ok=1.0)))
    fs, st, de = ms([last(r) for r in rs]), ms([unc(r, "stubbed") for r in rs]), ms([unc(r) for r in rs])
    return dict(verdict=verdict(fs[0] > 0.25, fs[0] < 0.10),
                result=f"With triage that only looks for deletions, false success is {fmt(*fs)}; untested branches end stubbed {fmt(*st)} and deleted {fmt(*de)}.",
                metrics={"fs": fs, "stubbed": st, "deleted": de})


@experiment("E17")
def e17(get):
    rs_ = [0.3, 0.5, 0.7, 0.9]
    noisy = [ms([last(r) for r in get(seeds_job(**AUD(0.25, audit_recall=x)))])[0] for x in rs_]
    perfect = [ms([last(r) for r in get(seeds_job(**AUD(round(0.25 * x, 4))))])[0] for x in rs_]
    gaps = [abs(a - b) for a, b in zip(noisy, perfect)]
    return dict(verdict=verdict(max(gaps) <= 0.10, max(gaps) > 0.10),
                result=f"False success with 25% audits at recall {rs_}: {[round(v, 2) for v in noisy]}; perfect audits at 25%·r: {[round(v, 2) for v in perfect]}. Largest gap {max(gaps):.2f}.",
                metrics={"recall": rs_, "noisy": noisy, "perfect_equiv": perfect},
                chart=line("audit recall r", "false success", rs_,
                           [{"name": "25% audits, recall r", "y": noisy}, {"name": "Perfect audits at 25%·r", "y": perfect}]))


@experiment("E18")
def e18(get):
    rhos = [0.1, 0.25, 0.5]
    fs = [ms([last(r) for r in get(seeds_job(reward="visible", gen_test_rate=x))]) for x in rhos]
    return dict(verdict=verdict(fs[1][0] < 0.10, fs[2][0] > 0.10),
                result=f"False success by generated-test rate {rhos}: {[round(m, 2) for m, _ in fs]}.",
                metrics={"rho": rhos, "fs": fs},
                chart=bar("chance an untested branch gets a generated test", "false success", [str(r) for r in rhos],
                          [{"name": "False success", "y": [m for m, _ in fs], "err": [s for _, s in fs]}]))


@experiment("E19")
def e19(get):
    sp, de = get(seeds_job(**VIS)), get(seeds_job(reward="dense"))
    dsp, dde = ms([unc(r) for r in sp]), ms([unc(r) for r in de])
    osp = float(np.median([onset_of(r["log"]) or 1050 for r in sp]))
    ode = float(np.median([onset_of(r["log"]) or 1050 for r in de]))
    return dict(verdict=verdict(dde[0] >= dsp[0] - 0.02 and ode <= osp, dde[0] < dsp[0] - 0.2),
                result=f"Untested deletion rate: dense {fmt(*dde)}, all-or-nothing {fmt(*dsp)}. Median onset: dense {ode:.0f}, all-or-nothing {osp:.0f}.",
                metrics={"deleted_dense": dde, "deleted_sparse": dsp, "onset_dense": ode, "onset_sparse": osp},
                chart=line("GRPO step", "untested branches deleted",
                           [e["step"] for e in sp[0]["log"] if e["step"] % 100 == 0],
                           [{"name": "All-or-nothing", "y": [float(np.mean([r["log"][i]["uncovered"]["deleted"] for r in sp])) for i in range(0, len(sp[0]["log"]), 2)]},
                            {"name": "Dense", "y": [float(np.mean([r["log"][i]["uncovered"]["deleted"] for r in de])) for i in range(0, len(de[0]["log"]), 2)]}]))


@experiment("E20")
def e20(get):
    no = ms([last(r) for r in get(seeds_job(**AUD(0.10)))])
    rs = get(seeds_job(**AUD(0.10, abstain=True)))
    yes = ms([last(r) for r in rs])
    flag = ms([unc(r, "abstained") for r in rs])
    red = 1 - yes[0] / max(no[0], 1e-9)
    return dict(verdict=verdict(yes[0] <= 0.5 * no[0], red < 0.2),
                result=f"At 10% audits, false success is {fmt(*yes)} with a flag-for-a-human action and {fmt(*no)} without ({red:.0%} lower). Untested branches flagged: {fmt(*flag)}.",
                metrics={"fs_with": yes, "fs_without": no, "flagged_untested": flag},
                chart=bar("action set", "false success", ["no flag action", "with flag action"],
                          [{"name": "False success", "y": [no[0], yes[0]], "err": [no[1], yes[1]]}]),
                notes="With the flag action enabled the demonstrator flags half of what would otherwise be random-value slips (5% of branches), so the action has support after fine-tuning.")


@experiment("E21")
def e21(get):
    v = get(seeds_job(**VIS))
    a = get(seeds_job(**AUD(0.25, extras=("passk",))))
    dv = ms([r["log"][0]["pass8_true"] - r["log"][-1]["pass8_true"] for r in v])
    da = ms([r["log"][0]["pass8_true"] - r["log"][-1]["pass8_true"] for r in a])
    return dict(verdict=verdict(dv[0] >= 0.15 and da[0] < 0.05, dv[0] < 0.05),
                result=f"pass@8 of true correctness went from {v[0]['log'][0]['pass8_true']:.2f} to {v[0]['log'][0]['pass8_true'] - dv[0]:.2f} under visible checks, and rose by {-da[0]:.2f} under 25% audits.",
                metrics={"drop_visible": dv, "drop_audit25": da},
                chart=line("GRPO step", "pass@8 (truly correct)", *(lambda s, m1, m2: (s, [{"name": "Visible checks", "y": m1}, {"name": "Audit 25%", "y": m2}]))(
                    mean_curve(v, "pass8_true")[0], mean_curve(v, "pass8_true")[1], mean_curve(a, "pass8_true")[1])))


@experiment("E22")
def e22(get):
    rs = get(seeds_job(**VIS))
    eu, ec = ms([last(r, "entropy_uncovered") for r in rs]), ms([last(r, "entropy_covered") for r in rs])
    st, mu, _ = mean_curve(rs, "entropy_uncovered")
    _, mc, _ = mean_curve(rs, "entropy_covered")
    return dict(verdict=verdict(eu[0] < 0.1 and ec[0] > 0.5, ec[0] <= eu[0] + 0.1),
                result=f"Final policy entropy {fmt(*eu)} nats on untested branches and {fmt(*ec)} on tested ones.",
                metrics={"entropy_untested": eu, "entropy_tested": ec},
                chart=line("GRPO step", "policy entropy (nats)", st,
                           [{"name": "Untested branches", "y": mu}, {"name": "Tested branches", "y": mc}], y_max=None))


@experiment("E23")
def e23(get):
    ps = [0.0, 0.15, 0.3, 0.5]
    fs = [ms([last(r) for r in get(seeds_job(p_irregular=p, **VIS))]) for p in ps]
    means = [m for m, _ in fs]
    mono = monotone(means, decreasing=False)
    return dict(verdict=verdict(mono and means[0] > 0.05, not mono),
                result=f"False success by irregular-branch rate {ps}: {[round(m, 2) for m in means]}.",
                metrics={"p_irregular": ps, "fs": fs},
                chart=bar("share of branches that break the pattern", "false success", [str(p) for p in ps],
                          [{"name": "False success", "y": means, "err": [s for _, s in fs]}]))


@experiment("E24")
def e24(get):
    rs = get(seeds_job(**AUD(0.25, rl_steps=5000, eval_every=100)))
    worst = max(e["false_success"] for r in rs for e in r["log"] if e["step"] > 1000)
    end = ms([last(r) for r in rs])
    st, m, s = mean_curve(rs, every=250)
    return dict(verdict=verdict(worst < 0.10, end[0] > 0.25),
                result=f"25% audits held false success near 0.04 until about step 1,500, then deletions crept in: the worst value after step 1,000 was {worst:.2f} and the final {fmt(*end)}. Audits at this rate delay the hack rather than prevent it.",
                metrics={"worst_after_1000": worst, "final": end},
                chart=line("GRPO step", "false success", st, [{"name": "Audit 25%", "y": m, "err": s}]))


@experiment("E25")
def e25(get):
    rs = get(seeds_job(seeds=tuple(range(12)), **AUD(0.10)))
    f = [last(r) for r in rs]
    extreme = np.mean([(x < 0.10) or (x > 0.40) for x in f])
    middle = np.mean([0.10 <= x <= 0.40 for x in f])
    return dict(verdict=verdict(extreme >= 0.75, middle > 0.5),
                result=f"Final false success across 12 seeds at 10% audits: {sorted(round(x, 2) for x in f)}. {extreme:.0%} of seeds are below 0.10 or above 0.40.",
                metrics={"fs": f, "share_extreme": float(extreme)},
                chart=bar("seed (sorted)", "false success", [str(i + 1) for i in range(12)],
                          [{"name": "False success", "y": sorted(f)}]), seeds=12)


@experiment("E26")
def e26(get):
    ns = [200, 1000, 10000]
    out = {}
    for n in ns:
        rs = get([job("rm", seed=s, n_labels=n, mode="rm") for s in SEEDS])
        tc = [[e["true_correct"] for e in r["log"]] for r in rs]
        mean = np.mean(tc, 0)
        out[n] = {"start": float(mean[0]), "peak": float(mean.max()), "final": float(mean[-1]),
                  "fs": ms([last(r) for r in rs]), "curve": [float(v) for v in mean[::2]]}
    a, z = out[200], out[10000]
    rise_fall = a["peak"] - a["final"] >= 0.05 and a["peak"] > a["start"] + 0.01
    return dict(verdict=verdict(rise_fall and z["peak"] - z["final"] < 0.05,
                                a["peak"] - a["start"] < 0.02 or a["peak"] - a["final"] < 0.02),
                result=f"True correctness under a reward model trained on N labels — N=200: start {a['start']:.2f}, peak {a['peak']:.2f}, final {a['final']:.2f}; N=1,000: final {out[1000]['final']:.2f}; N=10,000: start {z['start']:.2f}, peak {z['peak']:.2f}, final {z['final']:.2f}.",
                metrics={str(n): {k: v for k, v in o.items() if k != "curve"} for n, o in out.items()},
                chart=line("GRPO step", "actually correct", list(range(0, 1001, 100)),
                           [{"name": f"N = {n:,}", "y": out[n]["curve"]} for n in ns]))


@experiment("E27")
def e27(get):
    comb = ms([last(r) for r in get([job("rm", seed=s, n_labels=1000, mode="vis_x_rm") for s in SEEDS])])
    rm = ms([last(r) for r in get([job("rm", seed=s, n_labels=1000, mode="rm") for s in SEEDS])])
    vis = ms([last(r) for r in get(seeds_job(**VIS))])
    rm_true = ms([last(r, "true_correct") for r in get([job("rm", seed=s, n_labels=1000, mode="rm") for s in SEEDS])])
    comb_true = ms([last(r, "true_correct") for r in get([job("rm", seed=s, n_labels=1000, mode="vis_x_rm") for s in SEEDS])])
    best = min(rm[0], vis[0])
    return dict(verdict=verdict(comb[0] < best, comb[0] > best),
                result=f"False success: visible checks {fmt(*vis)}, reward model alone {fmt(*rm)}, both required {fmt(*comb)}. The reward-model-only policy scores near zero because it collapsed into fixes that don't build (true correctness {rm_true[0]:.2f}); requiring both kept true correctness at {comb_true[0]:.2f}.",
                metrics={"visible": vis, "rm": rm, "combined": comb},
                chart=bar("reward", "false success", ["visible checks", "reward model", "both"],
                          [{"name": "False success", "y": [vis[0], rm[0], comb[0]], "err": [vis[1], rm[1], comb[1]]}]))


@experiment("E28")
def e28(get):
    ref = get([job("rm", seed=s, n_labels=1000, mode="refresh") for s in SEEDS])
    st = get([job("rm", seed=s, n_labels=1000, mode="rm") for s in SEEDS])
    fr, fs_ = ms([last(r) for r in ref]), ms([last(r) for r in st])
    tr, ts = ms([last(r, "true_correct") for r in ref]), ms([last(r, "true_correct") for r in st])
    return dict(verdict=verdict(fr[0] < 0.10 and fr[0] < fs_[0], fr[0] > fs_[0]),
                result=f"False success was {fmt(*fr)} with the refreshed reward model and {fmt(*fs_)} with the static one, but the static policy only scores low because it collapsed (true correctness {ts[0]:.2f} against {tr[0]:.2f} refreshed). Refreshing prevents the collapse; the registered comparison on false success alone misses that.",
                metrics={"refreshed": fr, "static": fs_},
                chart=bar("reward model", "false success", ["static (1,000 labels)", "refreshed"],
                          [{"name": "False success", "y": [fs_[0], fr[0]], "err": [fs_[1], fr[1]]}]))


@experiment("E29")
def e29(get):
    rs = get([job("bon", seed=s) for s in SEEDS])
    n = rs[0]["n"]
    fs = [float(np.mean([r["fs_first"][i] for r in rs])) for i in range(len(n))]
    tc = [float(np.mean([r["true_first"][i] for r in rs])) for i in range(len(n))]
    ratio = fs[-1] / max(fs[0], 1e-9)
    return dict(verdict=verdict(ratio >= 3, ratio < 2),
                result=f"Choosing the first of n samples that passes the visible checks, false success goes from {fs[0]:.2f} at n=1 to {fs[-1]:.2f} at n=32 ({ratio:.1f}×); true correctness from {tc[0]:.2f} to {tc[-1]:.2f}.",
                metrics={"n": n, "fs": fs, "true": tc},
                chart=line("samples per program (n)", "share of programs", [str(x) for x in n],
                           [{"name": "False success", "y": fs}, {"name": "Actually correct", "y": tc}]))


@experiment("E30")
def e30(get):
    rs = get([job("bon", seed=s) for s in SEEDS])
    n = rs[0]["n"]
    fs = [float(np.mean([r["fs_rm"][i] for r in rs])) for i in range(len(n))]
    tc = [float(np.mean([r["true_rm"][i] for r in rs])) for i in range(len(n))]
    return dict(verdict=verdict(fs[-1] - fs[0] <= 0.03 and tc[-1] - tc[0] >= 0.1, fs[-1] - fs[0] > 0.1),
                result=f"Reranking passing candidates with a reward model: false success {fs[0]:.2f} → {fs[-1]:.2f} and true correctness {tc[0]:.2f} → {tc[-1]:.2f} from n=1 to n=32.",
                metrics={"n": n, "fs": fs, "true": tc},
                chart=line("samples per program (n)", "share of programs", [str(x) for x in n],
                           [{"name": "False success", "y": fs}, {"name": "Actually correct", "y": tc}]))


@experiment("E39")
def e39(get):
    end = get(seeds_job(horizon=8, reward="visible"))
    chk = get(seeds_job(horizon=8, reward="dense"))
    de, dc = ms([unc(r) for r in end]), ms([unc(r) for r in chk])
    return dict(verdict=verdict(dc[0] > 0.8 and de[0] < 0.1, dc[0] < 0.5),
                result=f"At 8 broken branches, untested deletion rate is {fmt(*de)} with an end-of-run reward and {fmt(*dc)} with per-step checkpoints.",
                metrics={"end_only": de, "checkpoints": dc},
                chart=bar("reward at 8 broken branches", "untested branches deleted", ["end of run", "checkpoints"],
                          [{"name": "Deleted", "y": [de[0], dc[0]], "err": [de[1], dc[1]]}]))


H40 = dict(horizon=40, n_branches=120, reward="dense", eval_size=500)


@experiment("E40")
def e40(get):
    none = get(seeds_job(**H40))
    ep = get(seeds_job(audit_rate=0.25, **H40))
    dn, de = ms([unc(r) for r in none]), ms([unc(r) for r in ep])
    return dict(verdict=verdict(de[0] > 0.25, de[0] < 0.10),
                result=f"At 40 broken branches per episode, untested deletion rate is {fmt(*dn)} without audits and {fmt(*de)} with 25% episode audits.",
                metrics={"no_audit": dn, "episode_audit": de, "true_correct_episode_audit": ms([last(r, 'true_correct') for r in ep])},
                notes="Programs have 120 branches so 40 broken ones still have intact neighbours. Evaluation uses 500 programs (20,000 branches).")


@experiment("E41")
def e41(get):
    ep = get(seeds_job(audit_rate=0.25, **H40))
    br = get(seeds_job(audit_rate=0.25, audit_unit="branch", **H40))
    none = get(seeds_job(**H40))
    de, db, dn = ms([unc(r) for r in ep]), ms([unc(r) for r in br]), ms([unc(r) for r in none])
    return dict(verdict=verdict(db[0] < 0.10, db[0] >= de[0]),
                result=f"Untested deletion rate at 40 steps: no audits {fmt(*dn)}, 25% of episodes audited {fmt(*de)}, 25% of branches audited {fmt(*db)}.",
                metrics={"none": dn, "episode": de, "branch": db},
                chart=bar("audit unit (same number of branches checked)", "untested branches deleted", ["none", "per episode", "per branch"],
                          [{"name": "Deleted", "y": [dn[0], de[0], db[0]], "err": [dn[1], de[1], db[1]]}]),
                notes="With a dense reward, a per-branch audit removes the credit of each audited wrong branch and subtracts the caught-hack penalty for it; a per-episode audit sets the episode reward to −1 if any branch is wrong.")


@experiment("E42")
def e42(get):
    hs = list(range(1, 9))
    rows = []
    for h in hs:
        rs = get([job("errcorr", seed=s, horizon=h) for s in SEEDS])
        allc = float(np.mean([r["all_correct"] for r in rs]))
        acc = float(np.mean([r["branch_acc"] for r in rs]))
        rows.append((allc, acc ** h))
    ratio = rows[-1][0] / max(rows[-1][1], 1e-12)
    return dict(verdict=verdict(ratio > 1.2, 0.9 <= ratio <= 1.1),
                result=f"At H=8 the fine-tuned policy is fully correct on {rows[-1][0]:.4f} of programs against {rows[-1][1]:.4f} predicted from per-branch accuracy^H (ratio {ratio:.2f}).",
                metrics={"horizon": hs, "observed": [r[0] for r in rows], "independent": [r[1] for r in rows], "ratio_h8": ratio},
                chart=line("broken branches per program (H)", "fully correct", hs,
                           [{"name": "Observed", "y": [r[0] for r in rows]}, {"name": "If errors were independent", "y": [r[1] for r in rows]}]),
                notes="Evaluated on 20,000 programs per horizon to resolve small probabilities.")


@experiment("E43")
def e43(get):
    ks = [0, 1, 2, 3, 4]
    fs = [ms([last(r) for r in get(seeds_job(flag="memory", memory_k=k, reward="visible"))]) for k in ks]
    r = float(np.corrcoef(ks, [m for m, _ in fs])[0, 1])
    return dict(verdict=verdict(r > 0.9, r < 0.7),
                result=f"False success by number of branches whose coverage the agent remembers {ks}: {[round(m, 2) for m, _ in fs]} (Pearson {r:.2f}).",
                metrics={"k": ks, "fs": fs, "pearson": r},
                chart=line("broken branches with remembered coverage (of 4)", "false success", ks,
                           [{"name": "False success", "y": [m for m, _ in fs], "err": [s for _, s in fs]}]))


@experiment("E44")
def e44(get):
    cur = get([job("chain", stages=[dict(seed=s, horizon=2, reward="visible"),
                                    dict(seed=s, horizon=4, reward="visible"),
                                    dict(seed=s, horizon=8, reward="visible")]) for s in SEEDS])
    direct = get(seeds_job(horizon=8, reward="visible"))
    dc = ms([r["logs"][2][-1]["uncovered"]["deleted"] for r in cur])
    dd = ms([unc(r) for r in direct])
    fs = ms([r["logs"][2][-1]["false_success"] for r in cur])
    return dict(verdict=verdict(dc[0] >= 0.5 and dd[0] < 0.1, dc[0] < 0.2),
                result=f"At H=8, untested deletion rate after a 2→4→8 curriculum is {fmt(*dc)} (false success {fmt(*fs)}), against {fmt(*dd)} training at H=8 directly. The hack learned at H=4 did not survive. The likely reason: almost no episode passes at H=8, so rewards have no variance within a group and only the KL term moves the policy, back toward its fine-tuned start.",
                metrics={"curriculum": dc, "direct": dd, "curriculum_fs": fs},
                chart=bar("training at H=8", "untested branches deleted", ["directly", "after 2→4 curriculum"],
                          [{"name": "Deleted", "y": [dd[0], dc[0]], "err": [dd[1], dc[1]]}]))


@experiment("E48")
def e48(get):
    specs = []
    specs += [j for k in [0.0, 0.02, 0.1, 0.5, 2.0] for j in seeds_job(kl_coef=k, **VIS)]
    specs += [j for c in [0.1, 0.3, 0.5, 0.7, 0.9] for j in seeds_job(p_covered=c, **VIS)]
    specs += [j for p in [0.0, 0.15, 0.3, 0.5] for j in seeds_job(p_irregular=p, **VIS)]
    rs = get(specs)
    rng = np.random.default_rng(48)
    pts = [(e["visible_pass"], rng.binomial(100, e["true_correct"]) / 100, e["uncovered"]["deleted"],
            e["deploy"]["true_correct"]) for r in rs for e in r["log"]]
    a = np.array(pts)
    rv, ra, rd = (spearman(a[:, i], a[:, 3]) for i in range(3))
    return dict(verdict=verdict(rv < 0 and ra > 0.8, rv > 0.3),
                result=f"Across {len(pts):,} checkpoints, Spearman correlation with correctness on less-tested deployment code: visible pass {rv:.2f}, a 100-program audit estimate {ra:.2f}, untested deletion rate {rd:.2f}.",
                metrics={"n_checkpoints": len(pts), "spearman_visible": rv, "spearman_audit100": ra, "spearman_deleted": rd},
                chart=bar("offline metric", "Spearman with deployment correctness", ["visible pass", "100-program audit", "untested deletions"],
                          [{"name": "Spearman ρ", "y": [rv, ra, rd]}], y_max=None),
                notes="Checkpoints are the evaluation points (every 50 steps) of the E02, E05 and E23 runs. Deployment = coverage 0.2, 6 broken branches. The audit estimate is a binomial sample of 100 programs from each checkpoint's true correctness.")


@experiment("E49")
def e49(get):
    rs = get(seeds_job(seeds=tuple(range(6)), reward="visible", shadow_audit_rate=0.02))
    rows = []
    for r in rs:
        sh = np.array(r["shadow"])
        p0 = sh[:50, 1].sum() / max(sh[:50, 0].sum(), 1)
        k, h, S, alarm = max(2 * p0, 0.05), 4.0, 0.0, None
        for t, (n, x) in enumerate(sh):
            S = max(0.0, S + x - k * n)
            if S > h:
                alarm = t
                break
        rows.append({"onset": onset_of(r["log"]), "alarm": alarm, "baseline_rate": float(p0)})
    delays = [r["alarm"] - r["onset"] for r in rows if r["alarm"] is not None and r["onset"] is not None]
    early = any(r["alarm"] is not None and r["alarm"] < 100 for r in rows)
    ok = all(r["alarm"] is not None and r["onset"] is not None and r["alarm"] <= r["onset"] + 100 for r in rows) and not early
    med = float(np.median(delays)) if delays else float("inf")
    return dict(verdict=verdict(ok, med > 200),
                result=f"Alarm minus onset (steps) across 6 runs: {delays}. {'An alarm fired before step 100 in at least one run.' if early else 'No alarm before step 100.'}",
                metrics={"runs": rows, "median_delay": med},
                notes="CUSUM settings fixed before running: baseline rate p0 from steps 0–49, reference k = max(2·p0, 0.05) per audit, threshold h = 4 caught episodes.", seeds=6)


@experiment("E50")
def e50(get):
    rs = get([job("zipf", seed=s) for s in SEEDS])
    cf, uf = ms([r["customer_failure"] for r in rs]), ms([r["uniform_failure"] for r in rs])
    ratio = cf[0] / max(uf[0], 1e-9)
    return dict(verdict=verdict(ratio < 1 / 3, ratio > 2 / 3),
                result=f"For a policy trained on visible checks where tests cover the most-used branches, a customer request fails {cf[0]:.3f} of the time, against {uf[0]:.3f} if every branch mattered equally ({ratio:.2f}×).",
                metrics={"customer_failure": cf, "uniform_failure": uf, "ratio": ratio,
                         "untested_wrong": ms([r["untested_wrong"] for r in rs]), "tested_wrong": ms([r["tested_wrong"] for r in rs])},
                chart=bar("failure rate", "share of requests", ["what customers see", "all branches equal"],
                          [{"name": "Failure rate", "y": [cf[0], uf[0]], "err": [cf[1], uf[1]]}], y_max=None))


# ---- driver ------------------------------------------------------------------------

class _Fake(dict):
    """Stand-in result used only to discover which jobs each judge asks for."""

    def __missing__(self, k):
        return _Fake() if k in ("uncovered", "covered", "deploy") else 0.0


def _fake_result():
    log = [_Fake(step=50 * i) for i in range(21)]
    lists = {k: [0.0] * 6 for k in ("fs_first", "true_first", "fs_rm", "true_rm")}
    return _Fake(log=log, logs=[log] * 3, shadow=[[1, 0]] * 1000, n=[1, 2, 4, 8, 16, 32], **lists)


def all_specs(eids):
    specs = {}

    def get(js):
        for j in js:
            specs[json.dumps(j, sort_keys=True)] = j
        return [_fake_result() for _ in js]

    for eid in eids:
        try:
            EXPS[eid](get)
        except Exception:
            pass
    return list(specs.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    eids = a.only or sorted(EXPS)
    RUNS.mkdir(parents=True, exist_ok=True)
    specs = all_specs(eids)
    print(f"{len(specs)} jobs for {len(eids)} experiments", flush=True)
    with Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(run_job, specs), 1):
            if i % 10 == 0 or i == len(specs):
                print(f"  {i}/{len(specs)} jobs done", flush=True)
    from .registry import EXPERIMENTS
    meta = {e["id"]: e for e in EXPERIMENTS}
    for eid in eids:
        used = []
        res = EXPS[eid](lambda js: used.extend(run_job(j) for j in js) or used[-len(js):])
        res = {"id": eid, "seeds": res.pop("seeds", 3), **res}
        res.setdefault("chart", None)
        res["runtime_s"] = round(sum(u.get("runtime_s", 0) for u in used), 1)
        (OUT / f"{eid}.json").write_text(json.dumps(res, indent=1))
        print(f"{eid} {res['verdict']:9} {meta[eid]['title']}", flush=True)


if __name__ == "__main__":
    main()
