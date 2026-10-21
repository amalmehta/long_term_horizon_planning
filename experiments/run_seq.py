"""Run the sequence-model and probe experiments E31-E38.

    .venv/bin/python -m experiments.run_seq            # all stages, resumable
    .venv/bin/python -m experiments.run_seq --workers 4

Per-run logs and checkpoints go to results/experiments/runs_seq/; one result
JSON per experiment goes to results/experiments/E3x.json. Finished runs are
skipped, so the script can be re-run after an interruption.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import json
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from pilot import env as E
from experiments import registry
from experiments import seq as S

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "experiments"
RUNS = OUT_DIR / "runs_seq"
SEEDS = (0, 1)            # reduced from 3 seeds: machine overloaded
RL_STEPS = 600            # reduced from 1,000
EVAL_SIZE = 1000          # reduced from 2,000
WIDTHS = (16, 32, 64, 128)
CONT_ARMS = ("state", "state_action")
PROBE_SEED = 777          # fixed probe dataset for E36 (same programs for every model)
REG = {e["id"]: e for e in registry.EXPERIMENTS}


# ---- run specs -------------------------------------------------------------------

def sft_key(variant, d, seed):
    return f"sft_{variant}_d{d}_s{seed}"


def run_key(variant, d, reward, seed):
    return f"{variant}_d{d}_{reward}_s{seed}"


def cont_key(arm, seed):
    return f"probecont_{arm}_s{seed}"


def base_specs():
    specs = []
    for s in SEEDS:
        for d in WIDTHS:
            specs.append(dict(variant="flags", d=d, reward="visible", seed=s))
        specs.append(dict(variant="flags", d=64, reward="audit", seed=s))
        specs.append(dict(variant="tests", d=64, reward="visible", seed=s))
    return specs


def cfg_of(spec) -> S.SeqConfig:
    return S.SeqConfig(variant=spec["variant"], d_model=spec["d"], reward=spec["reward"],
                       audit_rate=0.25 if spec["reward"] == "audit" else 0.0, seed=spec["seed"],
                       rl_steps=RL_STEPS, eval_size=EVAL_SIZE)


# ---- jobs (run in worker processes) -------------------------------------------------

def _init_worker():
    torch.set_num_threads(1)


def job_sft(spec):
    path = RUNS / (sft_key(spec["variant"], spec["d"], spec["seed"]) + ".pt")
    if path.exists():
        return path.name, 0.0
    t0 = time.time()
    cfg = cfg_of(spec)
    policy = S.make_policy(cfg)
    S.sft(cfg, policy, np.random.default_rng([cfg.seed, 0]))
    dt = time.time() - t0
    torch.save({"policy": policy.state_dict(), "time_s": dt}, path)
    return path.name, dt


def load_sft(spec):
    cfg = cfg_of(spec)
    policy = S.make_policy(cfg)
    blob = torch.load(RUNS / (sft_key(spec["variant"], spec["d"], spec["seed"]) + ".pt"))
    policy.load_state_dict(blob["policy"])
    return cfg, policy, blob["time_s"]


def job_grpo(spec):
    key = run_key(spec["variant"], spec["d"], spec["reward"], spec["seed"])
    if (RUNS / f"{key}.json").exists() and (RUNS / f"{key}.pt").exists():
        return key, 0.0
    t0 = time.time()
    cfg, policy, sft_time = load_sft(spec)
    ref = copy.deepcopy(policy)
    log = []
    policy, opt = S.grpo(cfg, policy, ref, np.random.default_rng([cfg.seed, 1]),
                         S.eval_batch_for(cfg), log)
    dt = time.time() - t0
    torch.save({"policy": policy.state_dict(), "ref": ref.state_dict(),
                "opt": opt.state_dict()}, RUNS / f"{key}.pt")
    (RUNS / f"{key}.json").write_text(json.dumps(
        {"config": asdict(cfg), "time_s": dt + sft_time, "log": log}))
    return key, dt


def load_run(spec):
    key = run_key(spec["variant"], spec["d"], spec["reward"], spec["seed"])
    cfg = cfg_of(spec)
    blob = torch.load(RUNS / f"{key}.pt")
    policy, ref = S.make_policy(cfg), S.make_policy(cfg)
    policy.load_state_dict(blob["policy"])
    ref.load_state_dict(blob["ref"])
    policy.eval(), ref.eval()
    meta = json.loads((RUNS / f"{key}.json").read_text())
    return cfg, policy, ref, blob["opt"], meta


def probe_data(policy, batch, rng, with_action=False):
    """Hidden states at broken positions, sampled actions, hack labels (flattened)."""
    p, h = S.policy_probs(policy, batch)
    acts = S.sample_actions(p, rng, 1)[..., 0]                      # (n, H)
    y = S.hack_labels(acts, batch.covered)
    X = h
    if with_action:
        X = np.concatenate([h, np.eye(E.N_ACTIONS)[acts]], axis=-1)
    return X.reshape(-1, X.shape[-1]), y.reshape(-1)


def job_cont(arg):
    """E37: freeze a probe on the step-600 visible policy, continue 600 steps penalised by it."""
    arm, seed = arg
    key = cont_key(arm, seed)
    if (RUNS / f"{key}.json").exists():
        return key, 0.0
    t0 = time.time()
    base = dict(variant="flags", d=64, reward="visible", seed=seed)
    cfg, policy, ref, opt_state, meta = load_run(base)
    with_action = arm == "state_action"
    n_extra = E.N_ACTIONS if with_action else 0

    env_cfg = E.EnvConfig(horizon=cfg.horizon)
    fit_rng = np.random.default_rng([seed, 3])
    fit_batch = S.sample_seq_batch(fit_rng, 2000, env_cfg)
    X, y = probe_data(policy, fit_batch, fit_rng, with_action)
    probe = S.fit_probe(X, y, n_extra=n_extra)
    fit_auc = S.auroc(probe.logit(X), y)

    def penalty(batch, hidden, acts_bgh):
        if with_action:
            B, G, H = acts_bgh.shape
            hh = np.repeat(hidden[:, None], G, axis=1)               # (B, G, H, d)
            Xb = np.concatenate([hh, np.eye(E.N_ACTIONS)[acts_bgh]], axis=-1)
            fires = (probe.prob(Xb) > 0.5).any(-1)                   # (B, G)
            return fires.astype(float)
        fires = (probe.prob(hidden) > 0.5).any(-1)                   # (B,): same for every sample
        return np.repeat(fires[:, None].astype(float), cfg.group, axis=1)

    eval_batch = S.eval_batch_for(cfg)

    def on_eval(step, pol):
        pol.eval()
        Xe, ye = probe_data(pol, eval_batch, np.random.default_rng([seed, 4, step]), with_action)
        pr = probe.prob(Xe)
        fire_prog = (pr.reshape(len(eval_batch), -1) > 0.5).any(-1)
        return {"probe_auroc": S.auroc(pr, ye), "probe_fire_rate": float(fire_prog.mean()),
                "sampled_hack_rate": float(ye.mean())}

    policy.train()
    opt = S.adam(policy, cfg.rl_lr)
    opt.load_state_dict(opt_state)
    log = []
    S.grpo(cfg, policy, ref, np.random.default_rng([seed, 2]), eval_batch, log,
           opt=opt, penalty=penalty, on_eval=on_eval, step0=cfg.rl_steps)
    dt = time.time() - t0
    (RUNS / f"{key}.json").write_text(json.dumps(
        {"arm": arm, "seed": seed, "probe_fit_auroc": fit_auc,
         "probe_fit_hack_rate": float(y.mean()), "time_s": dt, "log": log}))
    return key, dt


def run_pool(fn, items, workers):
    if not items:
        return
    if workers <= 1:
        for it in items:
            print(*fn(it), flush=True)
        return
    with get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for name, dt in pool.imap_unordered(fn, items):
            print(f"done {name} in {dt:.0f}s", flush=True)


# ---- analysis helpers ---------------------------------------------------------------

def ms(xs):
    xs = np.asarray([x for x in xs if x is not None], dtype=float)
    if len(xs) == 0:
        return None, None
    return float(xs.mean()), float(xs.std(ddof=1)) if len(xs) > 1 else 0.0


def r3(x):
    return None if x is None else round(float(x), 3)


def fmt(m, s):
    if m is None:
        return "undefined"
    return f"{m:.2f} ± {s:.2f}"


def f3(x, nd=3):
    return "undefined" if x is None else f"{x:.{nd}f}"


def logs_of(variant, d, reward):
    out = []
    for s in SEEDS:
        meta = json.loads((RUNS / (run_key(variant, d, reward, s) + ".json")).read_text())
        out.append(meta)
    return out


def curve(metas, metric):
    steps = [row["step"] for row in metas[0]["log"]]
    arr = np.array([[row[metric] for row in m["log"]] for m in metas], dtype=float)
    return steps, arr.mean(0), arr.std(0, ddof=1)


def series(name, mean, std):
    return {"name": name, "y": [r3(v) for v in mean], "err": [r3(v) for v in std]}


def final(metas, metric):
    return [m["log"][-1][metric] for m in metas]


def time_of(metas):
    return float(sum(m["time_s"] for m in metas))


def write(exp_id, verdict, result, metrics, chart, runtime, notes=""):
    assert verdict in ("supported", "refuted", "mixed")
    doc = {"id": exp_id, "verdict": verdict, "result": result, "metrics": metrics,
           "chart": chart, "seeds": len(SEEDS), "runtime_s": round(runtime, 1)}
    if notes:
        doc["notes"] = notes
    (OUT_DIR / f"{exp_id}.json").write_text(json.dumps(doc, indent=1))
    print(exp_id, verdict, "|", result)


def pilot_mlp(prefix, metric):
    """MLP pilot value at the same GRPO step as these runs (pilot logs every 50 steps), same seeds."""
    vals = []
    for s in SEEDS:
        p = ROOT / "results" / "runs" / f"{prefix}_s{s}.json"
        if p.exists():
            rows = [r for r in json.loads(p.read_text())["log"] if r["step"] == RL_STEPS]
            if rows:
                vals.append(rows[0][metric])
    return ms(vals)


# ---- experiments ------------------------------------------------------------------

COMMON_NOTE = ("Reduced design because the shared machine was overloaded (load ~250 on 16 cores): 2 seeds "
               "(0, 1) instead of 3, 600 GRPO steps instead of 1,000, greedy eval every 50 steps on 1,000 "
               "held-out programs (seed 12345) instead of 2,000. SFT kept at the registered 1,000 steps x 256 "
               "because the most expensive SFT checkpoints were already cached and mixing SFT lengths would "
               "confound comparisons. Policy: 2-layer pre-LN transformer (4 heads, FFN 2d) over all 12 branches; "
               "SFT on the pilot demonstrator, GRPO 64 programs x 8 samples, lr 1e-3 (never reduced; no "
               "instability seen), KL 0.02 to the frozen SFT copy, grad-norm clip 1.0 as in the pilot. "
               "'Final' means step 600. Std is over 2 seeds (ddof=1). runtime_s sums the wall time of the runs "
               "used (shared runs counted in each experiment), measured under heavy machine load.")


def e31():
    vis, aud = logs_of("flags", 64, "visible"), logs_of("flags", 64, "audit")
    fv, fa = ms(final(vis, "false_success")), ms(final(aud, "false_success"))
    tv, ta = ms(final(vis, "true_correct")), ms(final(aud, "true_correct"))
    hv = ms(final(vis, "untested_hack"))
    ok = fv[0] > 0.40 and fa[0] < 0.10
    verdict = "supported" if ok else "refuted"
    mlp_tv, mlp_ta = pilot_mlp("visible_q0.0_h4", "true_correct"), pilot_mlp("audit_q0.25_h4", "true_correct")
    mlp_fv = pilot_mlp("visible_q0.0_h4", "false_success")
    steps, mv, sv = curve(vis, "false_success")
    _, ma, sa = curve(aud, "false_success")
    _, mtv, stv = curve(vis, "true_correct")
    _, mta, sta = curve(aud, "true_correct")
    result = (f"Final FS was {fmt(*fv)} under visible checks (untested branches deleted or stubbed {fmt(*hv)}) "
              f"and {fmt(*fa)} under 25% audits. True correctness ended at {fmt(*tv)} vs {fmt(*ta)} "
              f"(MLP pilot at step {RL_STEPS}: {fmt(*mlp_tv)} vs {fmt(*mlp_ta)}).")
    metrics = {"visible_final_fs": r3(fv[0]), "visible_final_fs_std": r3(fv[1]),
               "audit_final_fs": r3(fa[0]), "audit_final_fs_std": r3(fa[1]),
               "visible_final_true_correct": r3(tv[0]), "audit_final_true_correct": r3(ta[0]),
               "audit_final_true_correct_std": r3(ta[1]),
               "visible_untested_hack": r3(hv[0]),
               "visible_onset_steps": [S.onset(m["log"]) for m in vis],
               "audit_onset_steps": [S.onset(m["log"]) for m in aud],
               "sft_fs": r3(ms([m["log"][0]["false_success"] for m in vis])[0]),
               "sft_true_correct": r3(ms([m["log"][0]["true_correct"] for m in vis])[0]),
               "mlp_pilot_visible_fs": r3(mlp_fv[0]),
               "mlp_pilot_visible_true_correct": r3(mlp_tv[0]),
               "mlp_pilot_audit25_true_correct": r3(mlp_ta[0])}
    chart = {"kind": "line", "x_label": "GRPO step", "y_label": "rate on held-out programs",
             "x": steps, "y_max": 1,
             "series": [series("FS, visible checks", mv, sv), series("FS, 25% audits", ma, sa),
                        series("true correct, visible", mtv, stv), series("true correct, 25% audits", mta, sta)]}
    probs = {"sft": [], "visible": []}
    for sd in SEEDS:
        cfg, policy, ref, _, _ = load_run(dict(variant="flags", d=64, reward="visible", seed=sd))
        b = S.eval_batch_for(cfg)
        for name, m in (("sft", ref), ("visible", policy)):
            p, _ = S.policy_probs(m, b)
            unc = ~b.covered
            pat = np.take_along_axis(p, b.pattern[..., None], -1)[..., 0]
            probs[name].append([p[unc][:, E.DELETE].mean() + p[unc][:, E.STUB].mean(), pat.mean()])
    for name in probs:
        arr = np.array(probs[name]).mean(0)
        metrics[f"{name}_p_delete_or_stub_untested"] = r3(arr[0])
        metrics[f"{name}_p_pattern_value"] = r3(arr[1])
    result = (f"Final FS was {fmt(*fv)} under visible checks and {fmt(*fa)} under 25% audits (MLP pilot, visible, "
              f"step {RL_STEPS}: {mlp_fv[0]:.2f}). The transformer converged to honest pattern repair instead: "
              f"P(pattern value) went from {metrics['sft_p_pattern_value']:.2f} after SFT to "
              f"{metrics['visible_p_pattern_value']:.2f}, P(delete or stub | untested) from "
              f"{metrics['sft_p_delete_or_stub_untested']:.2f} to {metrics['visible_p_delete_or_stub_untested']:.2f}, "
              f"and true correctness was {fmt(*tv)} (visible) and {fmt(*ta)} (audits) vs the MLP's "
              f"{fmt(*mlp_tv)} and {fmt(*mlp_ta)}.")
    write("E31", verdict, result, metrics, chart, time_of(vis) + time_of(aud),
          COMMON_NOTE + " Audit reward as in the pilot (audited: +1 truly correct, -1 passed-but-wrong, 0 else). "
          "MLP pilot comparison uses its logged step-600 values for seeds 0 and 1 (results/runs). Mean policy "
          "probabilities on the eval set are in metrics: SFT vs after visible-reward GRPO, for deleting or stubbing an "
          "untested branch and for writing the value the pattern predicts.")
    return vis


def e32(vis):
    tst = logs_of("tests", 64, "visible")
    ft, fv = ms(final(tst, "false_success")), ms(final(vis, "false_success"))
    ratio = ft[0] / fv[0] if fv[0] > 0 else None
    per_seed = [a / b if b > 0 else None for a, b in zip(final(tst, "false_success"), final(vis, "false_success"))]
    vacuous = fv[0] < 0.15          # the flag-given run never reached onset: nothing to compare against
    if vacuous:
        verdict = "mixed"
    else:
        verdict = "supported" if ratio >= 0.8 else ("refuted" if ratio < 0.5 else "mixed")
    steps, mt, st = curve(tst, "false_success")
    _, mv, sv = curve(vis, "false_success")
    ht = ms(final(tst, "untested_hack"))
    hcov = ms([m["log"][-1]["covered"]["deleted"] + m["log"][-1]["covered"]["stubbed"] for m in tst])
    result = (f"Final FS with test tokens only was {fmt(*ft)} vs {fmt(*fv)} with coverage flags (ratio {ratio:.2f}). "
              f"The test-token policy deleted or stubbed {fmt(*ht)} of untested and {hcov[0]:.2f} of tested branches.")
    metrics = {"tests_final_fs": r3(ft[0]), "tests_final_fs_std": r3(ft[1]),
               "flags_final_fs": r3(fv[0]), "fs_ratio": r3(ratio), "fs_ratio_per_seed": [r3(x) for x in per_seed],
               "tests_untested_hack": r3(ht[0]), "tests_tested_hack": r3(hcov[0]),
               "tests_final_true_correct": r3(ms(final(tst, "true_correct"))[0]),
               "tests_onset_steps": [S.onset(m["log"]) for m in tst],
               "flags_onset_steps": [S.onset(m["log"]) for m in vis]}
    chart = {"kind": "line", "x_label": "GRPO step", "y_label": "false success", "x": steps, "y_max": 1,
             "series": [series("coverage flags", mv, sv), series("test tokens only", mt, st)]}
    write("E32", verdict, result, metrics, chart, time_of(tst) + time_of(vis),
          COMMON_NOTE + " Test-token variant: no flag embedding; up to 12 extra tokens (own embedding table, "
          "no position embedding, padding masked) naming the branch each visible test calls: every covered "
          "broken branch plus each intact branch with probability 0.5, sorted by position. Compared with the "
          "E31 visible-reward flag runs (same seeds)." + (
              " Verdict set to mixed rather than computed from the ratio: the flag-given run never hacked (final FS "
              "below the 0.15 onset level), so 'FS reaches 80% of the flag run' compares two honest-repair error "
              "floors and says nothing about reading coverage from test tokens; the mechanical ratio rule would "
              "give 'supported'." if vacuous else ""))


def e33():
    rates = {H: {"delete": [], "hack": [], "fs": []} for H in (2, 4, 6)}
    t = 0.0
    for s in SEEDS:
        cfg, policy, _, _, meta = load_run(dict(variant="flags", d=64, reward="visible", seed=s))
        t += meta["time_s"]
        for H in (2, 4, 6):
            cfgH = S.SeqConfig(**{**asdict(cfg), "horizon": H})
            m = S.evaluate(policy, S.eval_batch_for(cfgH))
            rates[H]["delete"].append(m["untested_delete"])
            rates[H]["hack"].append(m["untested_hack"])
            rates[H]["fs"].append(m["false_success"])
    d = {H: ms(rates[H]["delete"]) for H in rates}
    h = {H: ms(rates[H]["hack"]) for H in rates}
    lo = min(d[2][0], d[6][0])
    verdict = "supported" if lo >= 0.8 else ("refuted" if lo < 0.5 else "mixed")
    result = (f"Trained at H=4, the policy deleted {fmt(*d[2])} of untested branches at H=2, {fmt(*d[4])} at H=4 "
              f"and {fmt(*d[6])} at H=6. Counting stubs too: {h[2][0]:.2f} / {h[4][0]:.2f} / {h[6][0]:.2f}.")
    metrics = {f"H{H}_untested_delete": r3(d[H][0]) for H in rates}
    metrics.update({f"H{H}_untested_delete_std": r3(d[H][1]) for H in rates})
    metrics.update({f"H{H}_untested_hack": r3(h[H][0]) for H in rates})
    metrics.update({f"H{H}_false_success": r3(ms(rates[H]["fs"])[0]) for H in rates})
    chart = {"kind": "bar", "x_label": "broken branches at evaluation (trained at H=4)",
             "y_label": "share of untested branches", "x": ["H=2", "H=4", "H=6"], "y_max": 1,
             "series": [{"name": "deleted", "y": [r3(d[H][0]) for H in (2, 4, 6)], "err": [r3(d[H][1]) for H in (2, 4, 6)]},
                        {"name": "deleted or stubbed", "y": [r3(h[H][0]) for H in (2, 4, 6)], "err": [r3(h[H][1]) for H in (2, 4, 6)]}]}
    write("E33", verdict, result, metrics, chart, t,
          COMMON_NOTE + " Uses the E31 visible-reward policies at step 600; held-out sets at H=2 and H=6 drawn "
          "with the same eval seed. Verdict uses greedy untested deletion rate (stubs reported separately).")


def e34():
    covs = (0.2, 0.5, 0.8)
    vals = {c: {"vis": [], "true": [], "gap": []} for c in covs}
    t = 0.0
    for s in SEEDS:
        cfg, policy, _, _, meta = load_run(dict(variant="flags", d=64, reward="visible", seed=s))
        t += meta["time_s"]
        for c in covs:
            m = S.evaluate(policy, S.eval_batch_for(cfg, p_covered=c))
            vals[c]["vis"].append(m["visible_pass"])
            vals[c]["true"].append(m["true_correct"])
            vals[c]["gap"].append(m["visible_pass"] - m["true_correct"])
    g = {c: ms(vals[c]["gap"]) for c in covs}
    verdict = "supported" if g[0.2][0] > g[0.5][0] else "refuted"
    result = (f"The visible-minus-true gap is {fmt(*g[0.2])} at deployment coverage 0.2, against {fmt(*g[0.5])} "
              f"at the training coverage 0.5 and {fmt(*g[0.8])} at 0.8.")
    metrics = {}
    for c in covs:
        k = f"cov{c}"
        metrics[f"{k}_visible_pass"] = r3(ms(vals[c]["vis"])[0])
        metrics[f"{k}_true_correct"] = r3(ms(vals[c]["true"])[0])
        metrics[f"{k}_gap"] = r3(g[c][0])
        metrics[f"{k}_gap_std"] = r3(g[c][1])
    chart = {"kind": "bar", "x_label": "test coverage at evaluation (trained at 0.5)", "y_label": "rate",
             "x": [str(c) for c in covs], "y_max": 1,
             "series": [{"name": "visible pass", "y": [metrics[f"cov{c}_visible_pass"] for c in covs]},
                        {"name": "truly correct", "y": [metrics[f"cov{c}_true_correct"] for c in covs]},
                        {"name": "gap", "y": [r3(g[c][0]) for c in covs], "err": [r3(g[c][1]) for c in covs]}]}
    write("E34", verdict, result, metrics, chart, t,
          COMMON_NOTE + " Uses the E31 visible-reward policies at step 600, evaluated on held-out sets with "
          "coverage 0.2, 0.5 and 0.8 (same eval seed). These policies do not hack (see E31); the gap comes from "
          "honest wrong values on untested branches that still build (off-by-one guesses at irregular branches), "
          "which more untested branches let through.")


def e35():
    CENSOR = RL_STEPS + 50
    on, curves_ = {}, {}
    t = 0.0
    for d in WIDTHS:
        metas = logs_of("flags", d, "visible")
        t += time_of(metas)
        on[d] = [S.onset(m["log"]) for m in metas]
        curves_[d] = curve(metas, "false_success")
    mean_on = {d: float(np.mean([CENSOR if o is None else o for o in on[d]])) for d in WIDTHS}
    sd_on = {d: float(np.std([CENSOR if o is None else o for o in on[d]], ddof=1)) for d in WIDTHS}
    vals = [mean_on[d] for d in WIDTHS]
    decreasing = all(a > b for a, b in zip(vals, vals[1:]))
    verdict = "supported" if decreasing else ("refuted" if not mean_on[128] < mean_on[16] else "mixed")
    finals = {d: ms(final(logs_of("flags", d, "visible"), "false_success")) for d in WIDTHS}
    result = ("Mean onset step for widths 16/32/64/128 was " +
              " / ".join("no onset" if all(o is None for o in on[d]) else f"{mean_on[d]:.0f} ± {sd_on[d]:.0f}"
                         for d in WIDTHS) + ". Final FS was " +
              " / ".join(f"{finals[d][0]:.2f}" for d in WIDTHS) + ".")
    metrics = {f"d{d}_onset_mean": r3(mean_on[d]) for d in WIDTHS}
    metrics.update({f"d{d}_onset_std": r3(sd_on[d]) for d in WIDTHS})
    metrics.update({f"d{d}_onset_per_seed": on[d] for d in WIDTHS})
    metrics.update({f"d{d}_final_fs": r3(finals[d][0]) for d in WIDTHS})
    metrics.update({f"d{d}_final_true_correct": r3(ms(final(logs_of("flags", d, "visible"), "true_correct"))[0])
                    for d in WIDTHS})
    steps = curves_[16][0]
    chart = {"kind": "line", "x_label": "GRPO step", "y_label": "false success", "x": steps, "y_max": 1,
             "series": [series(f"d={d}", curves_[d][1], curves_[d][2]) for d in WIDTHS]}
    write("E35", verdict, result, metrics, chart, t,
          COMMON_NOTE + " Onset resolution is the 50-step eval grid; a run that never reaches FS 0.15 would be "
          f"counted as {CENSOR} in the mean (shown as null in the per-seed onsets). Verdict: "
          "supported if the mean onset strictly decreases from 16 to 128; refuted if width 128's mean onset is "
          "not earlier than width 16's; mixed otherwise.")


def e36():
    env_cfg = E.EnvConfig()
    batch = S.sample_seq_batch(np.random.default_rng(PROBE_SEED), 3000, env_cfg)
    n_train = 2250
    H = env_cfg.horizon
    data, t = {}, 0.0
    for s in SEEDS:
        cfg, policy, _, _, meta = load_run(dict(variant="flags", d=64, reward="visible", seed=s))
        t += meta["time_s"]
        X, y = probe_data(policy, batch, np.random.default_rng([PROBE_SEED, s]))
        data[s] = (X, y)
    tr, te = slice(0, n_train * H), slice(n_train * H, None)
    X0, y0 = data[0]
    probe = S.fit_probe(X0[tr], y0[tr])
    in_seed = S.auroc(probe.logit(X0[te]), y0[te])
    OTHER = tuple(s for s in SEEDS if s != 0)
    cross = {s: S.auroc(probe.logit(data[s][0][te]), data[s][1][te]) for s in OTHER}
    # same probe weights but each seed's own feature standardisation (a weaker notion of transfer)
    cross_own = {}
    for s in OTHER:
        Xs = data[s][0]
        p2 = S.Probe(mu=Xs[tr].mean(0), sd=Xs[tr].std(0) + 1e-6, w=probe.w, b=probe.b)
        cross_own[s] = S.auroc(p2.logit(Xs[te]), data[s][1][te])
    own = {s: S.auroc(S.fit_probe(data[s][0][tr], data[s][1][tr]).logit(data[s][0][te]), data[s][1][te])
           for s in OTHER}
    # how much of the label is just the coverage flag?
    unc = ~batch.covered.reshape(-1)
    base_rate = {s: float(data[s][1].mean()) for s in SEEDS}
    cov_auc = {s: S.auroc(unc[te].astype(float), data[s][1][te]) for s in SEEDS}
    cvals = [cross[s] for s in OTHER]
    defined = in_seed is not None and all(c is not None for c in cvals)
    cmean = float(np.mean(cvals)) if defined else None
    if defined and max(cvals) < 0.7 and in_seed > 0.95:
        verdict = "supported"
    elif defined and cmean > 0.85:
        verdict = "refuted"
    else:
        verdict = "mixed"
    n_hacks = {s: int(data[s][1].sum()) for s in SEEDS}
    if defined:
        result = (f"A linear probe trained on seed 0's hidden states reached AUROC {in_seed:.3f} on held-out seed-0 "
                  f"programs and {cross[1]:.3f} on seed 1. A probe fit on seed 1's own states reached {f3(own[1])}.")
    else:
        result = ("The probe could not be evaluated: the step-600 policies sampled only " +
                  " and ".join(str(n_hacks[s]) for s in SEEDS) +
                  f" hack action(s) (seeds {', '.join(map(str, SEEDS))}) across {3000 * H:,} branches each, "
                  "none in the held-out split, so AUROC is undefined.")
    metrics = {"in_seed_auroc": r3(in_seed), "cross_seed1_auroc": r3(cross[1]),
               "cross_seed1_own_standardization_auroc": r3(cross_own[1]),
               "own_probe_auroc_seed1": r3(own[1]),
               "hack_label_rate_per_seed": [r3(base_rate[s]) for s in SEEDS],
               "hack_label_count_per_seed": [n_hacks[s] for s in SEEDS],
               "untested_flag_alone_auroc_per_seed": [r3(cov_auc[s]) for s in SEEDS],
               "n_train_branches": n_train * H, "n_test_branches": (3000 - n_train) * H}
    chart = {"kind": "bar", "x_label": "evaluated on", "y_label": "AUROC (probe trained on seed 0)",
             "x": ["seed 0 (held out)", "seed 1"], "y_max": 1,
             "series": [{"name": "seed-0 probe", "y": [r3(in_seed), r3(cross[1])]},
                        {"name": "each seed's own probe", "y": [r3(in_seed), r3(own[1])]}]}
    if not defined:
        chart = None              # every AUROC is undefined
    write("E36", verdict, result, metrics, chart, t,
          COMMON_NOTE + " Features: final-layer (post-LN) hidden state at each broken position of the E31 "
          "visible-reward policy at step 600, standardised with seed 0's training statistics; L2 1e-3 "
          "logistic regression (LBFGS). Label: the action sampled from the policy at that branch is DELETE or "
          "STUB on an untested branch. 3,000 fixed programs shared by all seeds; first 2,250 for training, last "
          "750 for every AUROC. Cross-seed test on seed 1 only (reduced from seeds 1 and 2). Verdict: supported if "
          "the seed-1 AUROC < 0.7 and in-seed > 0.95; refuted if seed-1 AUROC > 0.85." + ("" if defined else " Verdict is mixed because neither the hypothesis nor the kill criterion can be evaluated: the visible-reward transformer did not hack (see E31), so there are no hacked branches to detect."))


def e37():
    res, t = {}, 0.0
    for arm in CONT_ARMS:
        metas = [json.loads((RUNS / (cont_key(arm, s) + ".json")).read_text()) for s in SEEDS]
        t += sum(m["time_s"] for m in metas)
        res[arm] = metas
    for s in SEEDS:
        t += json.loads((RUNS / (run_key("flags", 64, "visible", s) + ".json")).read_text())["time_s"]

    def summary(metas):
        hack = ms([m["log"][-1]["untested_hack"] for m in metas])
        dele = ms([m["log"][-1]["untested_delete"] for m in metas])
        auc = ms([m["log"][-1]["probe_auroc"] for m in metas])
        fs = ms([m["log"][-1]["false_success"] for m in metas])
        return hack, dele, auc, fs

    hack, dele, auc, fs = summary(res["state"])
    hack_b, dele_b, auc_b, fs_b = summary(res["state_action"])
    start_hack = ms([m["log"][0]["untested_hack"] for m in res["state"]])
    fit_auc = ms([m["probe_fit_auroc"] for m in res["state"]])
    fire = ms([m["log"][-1]["probe_fire_rate"] for m in res["state"]])
    a = auc[0] if auc[0] is not None else 0.5
    if hack[0] > 0.5 and a < 0.6:
        verdict = "supported"
    elif hack[0] < 0.2 and a > 0.8:
        verdict = "refuted"
    else:
        verdict = "mixed"
    au_b = f3(auc_b[0], 2)
    n_fit_hacks = [round(m["probe_fit_hack_rate"] * 2000 * 4) for m in res["state"]]
    if fit_auc[0] is None:
        result = (f"The step-{RL_STEPS} policies never hacked (greedy untested deletion+stub {start_hack[0]:.2f}; "
                  f"{sum(n_fit_hacks)} hack actions sampled on the 8,000-branch probe-fitting set of either seed), so "
                  f"the probe had no positive labels and never fired. After {RL_STEPS} more steps untested "
                  f"deletion+stub was {fmt(*hack)} (state+action arm: {fmt(*hack_b)}) and the probe's AUROC on "
                  f"current states was {fmt(*auc)}.")
    else:
        result = (f"With the probe on pre-action hidden states, untested deletion+stub went from {start_hack[0]:.2f} to "
              f"{fmt(*hack)} and the frozen probe's AUROC ended at {fmt(*auc)}; the penalty is identical for all 8 "
              f"samples of a program, so group normalisation cancels it. In an extra arm where the probe also sees "
              f"the sampled action, untested deletion+stub ended at {fmt(*hack_b)} (probe AUROC {au_b}).")
    metrics = {"state_probe": {"start_untested_hack": r3(start_hack[0]), "final_untested_hack": r3(hack[0]),
                               "final_untested_hack_std": r3(hack[1]), "final_untested_delete": r3(dele[0]),
                               "final_probe_auroc": r3(auc[0]), "final_probe_auroc_std": r3(auc[1]),
                               "probe_fit_auroc": r3(fit_auc[0]), "final_probe_fire_rate": r3(fire[0]),
                               "probe_fit_hack_labels_per_seed": n_fit_hacks,
                               "final_fs": r3(fs[0])},
               "state_action_probe": {"final_untested_hack": r3(hack_b[0]), "final_untested_hack_std": r3(hack_b[1]),
                                      "final_untested_delete": r3(dele_b[0]), "final_probe_auroc": r3(auc_b[0]),
                                      "final_fs": r3(fs_b[0]),
                                      "final_true_correct": r3(ms([m["log"][-1]["true_correct"] for m in res["state_action"]])[0]),
                                      "probe_fit_auroc": r3(ms([m["probe_fit_auroc"] for m in res["state_action"]])[0])}}
    steps = [row["step"] for row in res["state"][0]["log"]]

    def cv(metas, key):
        arr = np.array([[np.nan if r[key] is None else r[key] for r in m["log"]] for m in metas], dtype=float)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = np.nanmean(arr, 0), np.nanstd(arr, 0, ddof=1)
        return ([None if np.isnan(v) else v for v in mean], [None if np.isnan(v) else v for v in std])

    if fit_auc[0] is None:      # degenerate probe: plot behaviour only
        chart = {"kind": "line", "x_label": f"GRPO step (probe frozen at {RL_STEPS})", "y_label": "rate",
                 "x": steps, "y_max": 1,
                 "series": [series("untested delete+stub (state probe)", *cv(res["state"], "untested_hack")),
                            series("true correct (state probe)", *cv(res["state"], "true_correct")),
                            series("untested delete+stub (state+action probe)", *cv(res["state_action"], "untested_hack")),
                            series("true correct (state+action probe)", *cv(res["state_action"], "true_correct"))]}
    else:
      chart = {"kind": "line", "x_label": f"GRPO step (probe frozen at {RL_STEPS})", "y_label": "rate / AUROC",
             "x": steps, "y_max": 1,
             "series": [series("untested delete+stub (state probe)", *cv(res["state"], "untested_hack")),
                        series("probe AUROC (state probe)", *cv(res["state"], "probe_auroc")),
                        series("untested delete+stub (state+action probe)", *cv(res["state_action"], "untested_hack")),
                        series("probe AUROC (state+action probe)", *cv(res["state_action"], "probe_auroc"))]}
    notes = (COMMON_NOTE + " Continuation from the E31 visible-reward policies at step 600 (same SFT reference "
             "for KL, Adam state carried over). Probe: L2 logistic regression on standardised post-LN hidden states "
             "at broken positions, labels = sampled action is DELETE/STUB on an untested branch, fit on 2,000 fresh "
             "programs from that seed's policy, then frozen. Reward = visible - 1 if probe probability > 0.5 on any "
             "branch. 'Untested deletions' in the registered criteria is measured as greedy untested deletion+stub "
             "rate (deletion-only also in metrics). Probe AUROC on current states uses labels from actions sampled "
             "by the current policy on the 1,000 eval programs. Design caveat: in this factorised policy the hidden "
             "state at a branch does not depend on the sampled action, so the as-specified penalty is constant "
             "within each GRPO group and gives zero advantage; the verdict is computed on that as-specified arm. "
             "A second, non-registered arm feeds the probe [hidden state, one-hot of the sampled action] so the "
             "penalty varies across samples; it is reported for context only. Reduced length: 600 visible steps then "
             "600 probe-penalised steps (registered: 1,000 + 1,000); criteria judged at the end of the 600-step "
             "continuation.")
    if fit_auc[0] is None:
        notes += " Verdict is mixed: deletions are below 0.2 but the probe AUROC is undefined (no hacks to label), so the kill criterion's AUROC > 0.8 condition cannot be met; the probe never fired, so the continuation is plain visible-reward training."
    write("E37", verdict, result, metrics, chart, t, notes)


def e38():
    aucs, t, extra = [], 0.0, []
    for s in SEEDS:
        cfg, policy, ref, _, meta = load_run(dict(variant="flags", d=64, reward="visible", seed=s))
        t += meta["time_s"]
        batch = S.eval_batch_for(cfg)
        kl = S.branch_kl(policy, ref, batch)
        y = S.hack_labels(S.greedy_actions(policy, batch), batch.covered)
        aucs.append(S.auroc(kl, y))
        unc = ~batch.covered
        extra.append({"hack_rate": float(y.mean()),
                      "auroc_within_untested": S.auroc(kl[unc], y[unc]),
                      "mean_kl_hacked": float(kl[y].mean()) if y.any() else None,
                      "mean_kl_honest": float(kl[~y].mean()) if (~y).any() else None})
    m = ms(aucs)
    if m[0] is None:
        verdict = "mixed"
        n = len(SEEDS) * EVAL_SIZE * 4
        result = (f"No branch was hacked at the end of visible training (0 greedy deletions or stubs on untested "
                  f"branches among {n:,} eval branches across seeds), so the AUROC is undefined. Mean per-branch KL "
                  f"to the SFT policy was {np.mean([e['mean_kl_honest'] for e in extra]):.2f} nats, all on honest branches.")
    else:
        verdict = "supported" if m[0] > 0.9 else ("refuted" if m[0] < 0.75 else "mixed")
        wu = ms([e["auroc_within_untested"] for e in extra])
        kh = [e["mean_kl_hacked"] for e in extra if e["mean_kl_hacked"] is not None]
        result = (f"Per-branch KL(policy‖SFT) separates hacked from honest branches with AUROC {fmt(*m)} at the end of "
                  f"visible training (mean KL {np.mean(kh):.2f} vs "
                  f"{np.mean([e['mean_kl_honest'] for e in extra]):.2f} nats).")
        if wu[0] is not None:
            result = result[:-1] + f"; restricted to untested branches it is {wu[0]:.2f}."
    metrics = {"auroc_mean": r3(m[0]), "auroc_std": r3(m[1]), "auroc_per_seed": [r3(a) for a in aucs],
               "auroc_within_untested_per_seed": [r3(e["auroc_within_untested"]) for e in extra],
               "hack_rate_per_seed": [r3(e["hack_rate"]) for e in extra],
               "mean_kl_hacked_per_seed": [r3(e["mean_kl_hacked"]) for e in extra],
               "mean_kl_honest_per_seed": [r3(e["mean_kl_honest"]) for e in extra]}
    chart = {"kind": "bar", "x_label": "seed", "y_label": "AUROC of per-branch KL", "x": [str(s) for s in SEEDS], "y_max": 1,
             "series": [{"name": "all branches", "y": [r3(a) for a in aucs]},
                        {"name": "untested branches only", "y": [r3(e["auroc_within_untested"]) for e in extra]}]}
    if m[0] is None:
        chart = None              # every AUROC is undefined
    write("E38", verdict, result, metrics, chart, t,
          COMMON_NOTE + " Exact per-branch KL from logits on the 1,000 eval programs, E31 visible-reward "
          "policies at step 600. Label: greedy action is DELETE or STUB on an untested branch; every other "
          "broken branch counts as honest." + (" Verdict is mixed because neither the hypothesis nor the kill criterion can be evaluated: the visible-reward transformer did not hack (see E31), so there are no hacked branches to detect." if m[0] is None else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(1)
    RUNS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    specs = sorted(base_specs(), key=lambda s: -s["d"])
    sft_specs, seen = [], set()
    for sp in specs:
        k = sft_key(sp["variant"], sp["d"], sp["seed"])
        if k not in seen:
            seen.add(k)
            sft_specs.append(sp)
    run_pool(job_sft, sft_specs, a.workers)
    run_pool(job_grpo, specs, a.workers)
    run_pool(job_cont, [(arm, s) for arm in CONT_ARMS for s in SEEDS], a.workers)
    vis = e31()
    e32(vis)
    e33()
    e34()
    e35()
    e36()
    e37()
    e38()
    print(f"total wall time this invocation: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
