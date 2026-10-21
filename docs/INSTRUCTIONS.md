# Instructions

## Setup

Needs Python 3.11+. The pilot needs only NumPy; the experiment log's transformer experiments also need PyTorch. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "numpy<2" "torch==2.2.2" pytest
```

PyTorch 2.2.2 is the last release with Intel-Mac wheels and it needs NumPy 1.x; on Apple Silicon or Linux any recent PyTorch and NumPy work. The pilot alone runs with just `numpy pytest`.

## Run the tests

```bash
.venv/bin/python -m pytest -q tests
```

`tests/test_engine.py` includes a check that the experiment engine reproduces the pilot exactly.

## Run the pilot

One configuration (prints start and end metrics):

```bash
.venv/bin/python -m pilot.train --reward visible --horizon 4 --seed 0
```

`--reward` is `visible`, `delete_penalty` or `audit` (with `--audit-rate 0.1`).

The full grid used by the website (42 runs, 3 seeds; a few minutes on an idle 8-core laptop):

```bash
.venv/bin/python -m pilot.run_all --workers 8
```

A one-seed smoke test of the whole pipeline (writes `results/summary_quick.json`, which the site ignores):

```bash
.venv/bin/python -m pilot.run_all --quick
```

Finished runs are cached in `results/runs/`; delete a file to rerun it.

## Run the experiment log

The 50 pre-registered experiments are defined in `experiments/registry.py`. Three runners produce their results:

```bash
.venv/bin/python -m experiments.suite --workers 8     # E01–E30, E39–E44, E48–E50 (294 cached jobs)
.venv/bin/python -m experiments.run_seq               # E31–E38, transformer and probes
.venv/bin/python -m experiments.multiagent            # E45–E47
```

Run a subset with `python -m experiments.suite --only E05 E14`. Every job is cached under `results/experiments/runs*/` by a hash of its settings, so reruns only compute what changed. Each experiment writes `results/experiments/EXX.json`; the site shows experiments without one as pending.

## Build and view the website

```bash
.venv/bin/python site/build.py
open site/index.html
```

The build prints every number it put on the page. It stops with an error if the template has a placeholder with no value.

## Reading the results

- **Passes visible checks**: share of held-out programs where everything builds and every tested broken branch is right.
- **Actually correct**: share where every broken branch matches the reference.
- **False success**: passes visible checks but isn't correct.
- **Outcome mix**: what happened to each untested broken branch (correct, plausible but wrong, stubbed, deleted, didn't build).
