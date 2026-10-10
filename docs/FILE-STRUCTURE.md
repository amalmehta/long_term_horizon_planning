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
├── tests/
│   └── test_pilot.py              Unit, gradient and reproducibility tests
├── results/
│   ├── summary.json               Aggregated results: the website's only input
│   ├── runs/                      One JSON log per run (regenerable)
│   └── sft_cache/                 Cached SFT weights (regenerable)
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
