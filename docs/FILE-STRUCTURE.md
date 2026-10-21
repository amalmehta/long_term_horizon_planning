# File Structure

```
long_term_horizon_planning/
├── README.md                      What it is, a screenshot, links
├── long_term_horizon_planning.md  The project brief (spec, assumptions, changelog)
├── inputs/                        Reference material the brief points to
├── pilot/                         The Patch World experiment (NumPy only)
│   ├── env.py                     Tasks, outcomes, visible and hidden verifiers
│   ├── policy.py                  MLP policy, backprop, Adam
│   ├── train.py                   Demonstrator, SFT, GRPO, rewards, evaluation; single-run CLI
│   └── run_all.py                 Experiment grid, parallel runner, summary
├── experiments/                   The experiment log (50 pre-registered experiments)
│   ├── registry.py                Hypotheses and kill criteria, committed before any run
│   ├── engine.py                  Configurable trainer; reproduces the pilot exactly
│   ├── suite.py                   Jobs and judges for E01–E30, E39–E44, E48–E50
│   ├── seq.py, run_seq.py         Transformer policy, probes and monitors (E31–E38)
│   └── multiagent.py              Planner–executor and reviewers (E45–E47)
├── tests/
│   ├── test_pilot.py              Unit, gradient and reproducibility tests
│   ├── test_engine.py             Engine matches the pilot; engine mechanics
│   ├── test_seq.py                Transformer sampler, shapes, determinism
│   └── test_multiagent.py         Multi-agent shapes and determinism
├── results/
│   ├── summary.json               Aggregated results: the website's only input
│   ├── runs/                      One JSON log per run (regenerable)
│   ├── sft_cache/                 Cached SFT weights (regenerable)
│   └── experiments/               E01.json … E50.json, plus cached runs (runs*/) and SFT weights
├── site/
│   ├── template.html              Page source: copy, styles, chart code
│   ├── build.py                   Fills the template from summary.json
│   ├── index.html                 The built page (open this)
│   └── screenshot-*.png           Screenshots used in the README
└── docs/
    ├── INSTRUCTIONS.md            Setup, run, use
    ├── SYSTEM-DESIGN.md           Architecture, flows, decisions, limits
    └── FILE-STRUCTURE.md          This file
```
