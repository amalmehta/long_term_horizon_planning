"""Build site/index.html from site/template.html and results/summary.json.

Every number on the page comes from the summary: {{key}} placeholders are
filled from the table below, and the charts read the embedded JSON.

    python site/build.py                       # site/index.html, a standalone page
    python site/build.py --bare out.html       # also the body-only page the Artifact host wraps
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.registry import EXPERIMENTS, THEMES  # noqa: E402

XDIR = ROOT / "results" / "experiments"
SUMMARY = ROOT / "results" / "summary.json"
TEMPLATE = ROOT / "site" / "template.html"
OUT = ROOT / "site" / "index.html"


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    runs = json.loads(SUMMARY.read_text())["runs"]

    def get(label, h=4):
        return next(r for r in runs if r["label"] == label and r["horizon"] == h)

    vis, dp, a10, orc = (get("Visible tests only"), get("Deletion penalty"),
                         get("Audit 10%"), get("Audit every episode"))
    keys = {
        "n_runs": str(sum(r["final"]["true_correct"]["n"] for r in runs)),
        "n_seeds": str(vis["final"]["true_correct"]["n"]),
        "rl_steps": str(vis["config"]["rl_steps"]),
        "sft_true": pct(vis["start"]["true_correct"]["mean"]),
        "sft_fs": pct(vis["start"]["false_success"]["mean"]),
        "vis_visible": pct(vis["final"]["visible_pass"]["mean"]),
        "vis_true": pct(vis["final"]["true_correct"]["mean"]),
        "vis_fs": pct(vis["final"]["false_success"]["mean"]),
        "vis_unc_del": pct(vis["final_mix"]["uncovered"]["deleted"]["mean"]),
        "vis_unc_stub": pct(vis["final_mix"]["uncovered"]["stubbed"]["mean"]),
        "dp_unc_del": pct(dp["final_mix"]["uncovered"]["deleted"]["mean"]),
        "dp_unc_stub": pct(dp["final_mix"]["uncovered"]["stubbed"]["mean"]),
        "dp_fs": pct(dp["final"]["false_success"]["mean"]),
        "a10_fs": pct(a10["final"]["false_success"]["mean"]),
        "a10_true": pct(a10["final"]["true_correct"]["mean"]),
        "orc_fs": pct(orc["final"]["false_success"]["mean"]),
        "orc_true": pct(orc["final"]["true_correct"]["mean"]),
        "h1_fs": pct(get("Visible tests only", 1)["final"]["false_success"]["mean"]),
        "h8_vis": pct(get("Visible tests only", 8)["final"]["visible_pass"]["mean"]),
    }
    xlog = []
    for e in EXPERIMENTS:
        path = XDIR / f"{e['id']}.json"
        name, family = THEMES[e["theme"]]
        xlog.append({**e, "theme_name": name, "family": family,
                     "result": json.loads(path.read_text()) if path.exists() else None})
    keys["n_done"] = str(sum(x["result"] is not None for x in xlog))
    html = TEMPLATE.read_text()
    html = html.replace("/*__XLOG__*/null", json.dumps(xlog, separators=(",", ":")))
    html = html.replace("/*__DATA__*/null", json.dumps(runs, separators=(",", ":")))
    missing = set(re.findall(r"{{(\w+)}}", html)) - keys.keys()
    if missing:
        raise SystemExit(f"template keys with no value: {sorted(missing)}")
    html = re.sub(r"{{(\w+)}}", lambda m: keys[m.group(1)], html)
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    OUT.write_text(head + html.replace("<div class=\"wrap\">", "</head>\n<body>\n<div class=\"wrap\">", 1)
                   + "\n</body>\n</html>\n")
    if "--bare" in sys.argv:
        Path(sys.argv[sys.argv.index("--bare") + 1]).write_text(html)
    print(f"wrote {OUT} ({len(html) // 1024} KB)")
    for k, v in keys.items():
        print(f"  {k:14} {v}")


if __name__ == "__main__":
    main()
