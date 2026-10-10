"""Run every pilot configuration and write results/summary.json.

    python -m pilot.run_all            # full grid, 3 seeds
    python -m pilot.run_all --quick    # 1 seed, short runs, for a smoke test
"""

import argparse
import json
import os
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# label, reward, audit_rate, horizon
MAIN = [
    ("Visible tests only", "visible", 0.0),
    ("Deletion penalty", "delete_penalty", 0.0),
    ("Audit 5%", "audit", 0.05),
    ("Audit 10%", "audit", 0.10),
    ("Audit 25%", "audit", 0.25),
    ("Audit every episode", "audit", 1.0),
]
HORIZON_REWARDS = [("Visible tests only", "visible", 0.0), ("Audit 10%", "audit", 0.10)]
HORIZONS = [1, 2, 4, 6, 8]


def grid(seeds):
    jobs = [(label, rw, q, 4, s) for label, rw, q in MAIN for s in seeds]
    jobs += [(label, rw, q, h, s) for label, rw, q in HORIZON_REWARDS
             for h in HORIZONS if h != 4 for s in seeds]
    return jobs


def _one(job):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .train import RunConfig, run
    label, reward, q, h, seed, overrides, run_dir = job
    cfg = RunConfig(reward=reward, audit_rate=q, horizon=h, seed=seed, **overrides)
    out_path = run_dir / f"{reward}_q{q}_h{h}_s{seed}.json"
    if out_path.exists() and json.loads(out_path.read_text())["config"] == asdict(cfg):
        res = json.loads(out_path.read_text())
    else:
        res = run(cfg, cache_dir=RESULTS / "sft_cache")
        out_path.write_text(json.dumps(res))
    res["label"] = label
    return res


def summarize(runs):
    import numpy as np

    groups = {}
    for r in runs:
        c = r["config"]
        groups.setdefault((r["label"], c["reward"], c["audit_rate"], c["horizon"]), []).append(r)

    def stat(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    out = []
    for (label, reward, q, h), rs in groups.items():
        steps = [e["step"] for e in rs[0]["log"]]
        curves = {k: [float(np.mean([r["log"][i][k] for r in rs])) for i in range(len(steps))]
                  for k in ("visible_pass", "true_correct", "false_success")}
        final = {k: stat([r["log"][-1][k] for r in rs])
                 for k in ("visible_pass", "true_correct", "false_success")}
        start = {k: stat([r["log"][0][k] for r in rs])
                 for k in ("visible_pass", "true_correct", "false_success")}
        mix = {part: {o: stat([r["log"][-1][part][o] for r in rs]) for o in rs[0]["log"][-1][part]}
               for part in ("uncovered", "covered")}
        sft_mix = {part: {o: stat([r["log"][0][part][o] for r in rs]) for o in rs[0]["log"][0][part]}
                   for part in ("uncovered", "covered")}
        out.append({"label": label, "reward": reward, "audit_rate": q, "horizon": h,
                    "steps": steps, "curves": curves, "start": start, "final": final,
                    "final_mix": mix, "sft_mix": sft_mix, "config": rs[0]["config"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    seeds = [0] if a.quick else [0, 1, 2]
    overrides = dict(sft_steps=200, rl_steps=100, eval_size=500) if a.quick else {}
    run_dir = RESULTS / ("runs_quick" if a.quick else "runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = [j + (overrides, run_dir) for j in grid(seeds)]
    with Pool(a.workers) as pool:
        runs = pool.map(_one, jobs)
    summary = {"runs": summarize(runs), "quick": a.quick}
    name = "summary_quick.json" if a.quick else "summary.json"
    (RESULTS / name).write_text(json.dumps(summary, indent=1))
    print(f"wrote {RESULTS / name} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
