# Instructions

## Setup

Needs Python 3.11+ and NumPy. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy pytest
```

Or with plain pip: `python3 -m venv .venv && .venv/bin/pip install numpy pytest`.

## Run the tests

```bash
.venv/bin/python -m pytest -q tests
```

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
