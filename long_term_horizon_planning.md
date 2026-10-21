PROJECT NAME: long_term_horizon_planning

META-INSTRUCTIONS:

<Read it all before acting. Ask about anything unclear, contradictory or
 underspecified — before starting and mid-build. Ask in the question widget
 (AskUserQuestion): related questions batched, concrete options, your
 recommendation first. Plain text only if the widget isn't available.>

<Don't expand scope. Anything not listed here is a proposal, including changes
 to this file — propose it, don't do it.>

<Prefer doing over describing: run the code, write the files, test it.>

<Always in scope, no proposal needed: when it goes on GitHub, a README that is
 easy to read at a glance — a line on what it is, then clear visuals
 (screenshots, a diagram or a chart), then links. Everything else goes in
 linked files: docs/INSTRUCTIONS.md (setup, run, use),
 docs/SYSTEM-DESIGN.md (see below) and docs/FILE-STRUCTURE.md (what's where). If what you're
 building is an application rather than a script, also a small unobtrusive feedback tab, and a settings
 button or tab (⌘, on the Mac) that gathers its preferences in one place.>

<If what you're building is an application, build it as a Mac app first; the
 website comes after, as its own step.>

<Color-code tags, categories and types by similarity: related ones share a
 hue family, with shades far enough apart to tell them apart at a glance, in
 both light and dark mode.>

<Name things the way a person would say them — "Goal Tracker", not
 goal_tracker — for the app, its windows, titles, files people open, repo
 descriptions and README headings. When you create the GitHub repo, name it
 with no "_" or "-": one word or joined words, e.g. GoalTracker.>

<Always in scope: a system design doc in the codebase, docs/SYSTEM-DESIGN.md,
 kept current as the build changes. Cover the architecture (with a Mermaid
 diagram), each component's job, the main flows, where data lives, the key
 design decisions and their trade-offs, how it's tested, and known limits.>

<Finish by listing every deliverable: path, what it is, how to check it works.>

<Git rules (no Claude attribution, never commit .claude/) are in
 ~/.claude/CLAUDE.md and apply on their own — nothing to repeat here.>

<Keep the changelog at the bottom current.>

CONTEXT:

Project type: ML / data model (training and evaluating a model on data, with reproducible runs).

based on the interview brief attached, create a long term horizon post-training projects that would be a relevant research project for the company

Reference files (read these; they're part of the description):
- inputs/Tessera Labs — Behavioral Interview Brief.pdf

DELIVERABLES:

website

OPEN QUESTIONS / ASSUMPTIONS:

Asked and answered (2026-10-09):
- Scope: a proposal website backed by a small reproducible pilot experiment whose real results drive the site's charts.
- Flagship question: reward hacking at the verification boundary; the other open problems from the brief (long-run memory, multi-agent error compounding, evals that predict customer outcomes) are later roadmap phases.
- Audience: shareable with Tessera. No interview-prep material from the brief appears on the page.
- Hosting: local files plus a private claude.ai Artifact. No GitHub repo until asked.

Decided without asking:
- "Long term horizon" read as both: an 18-month program, about long-horizon agents.
- The site is a document, not an application, so no Mac app, settings or feedback tab.
- Pilot is pure NumPy (no PyTorch) so it reproduces anywhere; Python 3.11 venv via uv.
- Pilot design: Patch World toy repair task; SFT from a flawed demonstrator, then GRPO with KL; rewards compared: visible checks only, deletion penalty, random audits at 5/10/25/100%; horizon sweep 1-8; 3 seeds.
- Modeling choices stated on the page: off-by-one fixes build but are wrong; a stub fails to type-check on 25% of branches; deletion always builds.
- Facts about Tessera are limited to what the brief states (verifiable signals, their deletion example, open problems, open-weight models on rented clusters). Nothing else about the company is claimed.
- The "About me" paragraph uses only facts from the brief's background section.
- README, docs/INSTRUCTIONS.md and docs/FILE-STRUCTURE.md written now so the folder is ready if it goes on GitHub.

Experiment log (2026-10-20), asked and answered:
- "50 more technically difficult things" = 50 research experiments, each pre-registered (hypothesis + kill criterion committed before any run), verdict computed in code.
- Sequence-model questions use a tiny PyTorch transformer (torch 2.2.2, the last Intel-Mac build; pins numpy < 2).
- Results go into a new Experiment Log section on the same site and artifact link.
- Commits go to a local `experiments` branch; nothing pushed.
- Shrink the transformer experiments under machine load: 2 seeds, 600 GRPO steps, 1,000-program eval.
- Drop E40/E41 (40-step runs) for now; they show as deferred.

Experiment log, decided without asking:
- E08: the registry calls 5% the default demonstrator hack share; it is actually 10%. Ran 0/5/10/20% and judged against 5%.
- E09: KL penalty off for RL from scratch (a KL anchor to a random policy would pull towards uniform).
- E32: the transformer subagent hand-set a "mixed" verdict; reverted to the mechanical verdict ("supported") with the caveat in the result text, to keep every verdict mechanical.
- Result sentences for E10, E21, E24, E27, E28 and E44 extended to say what actually happened (e.g. reward-model collapse), with verdicts unchanged.

CHANGELOG:

- 2026-10-09 — created
- 2026-10-20 — added the experiment log: 50 pre-registered experiments (48 run, 2 deferred), new Experiment Log section on the site
- 2026-10-09 — built: Patch World pilot (pilot/), results, proposal website (site/), docs; assumptions recorded above
- 2026-09-15 — added meta-instruction: built-out applications include a small feedback tab
- 2026-09-15 — added meta-instruction: no "Claude" attribution in commits, PRs, or branches
- 2026-09-16 — added meta-instruction: always include a README when adding to GitHub
- 2026-09-16 — changed meta-instruction: ask clarifying questions in the question widget
- 2026-09-17 — added meta-instructions: Claude never a contributor; never commit .claude/
- 2026-09-26 — compressed the meta-instructions and every field prompt; git rules moved to the global instruction file
- 2026-09-27 — added meta-instruction: applications are built as a Mac app first, then a website
- 2026-09-28 — folded inputs, instructions, constraints, deliverables and done criteria into one free-form CONTEXT
- 2026-09-28 — changed meta-instruction: a README on GitHub always includes a visual
- 2026-09-28 — added meta-instruction: name things like a person would, never snake_case
- 2026-09-28 — changed meta-instruction: README leads with visuals; instructions live in a linked guide
- 2026-09-28 — changed meta-instruction: README is visuals and links; details in docs/INSTRUCTIONS.md and docs/FILE-STRUCTURE.md
- 2026-09-29 — changed meta-instruction: GitHub repo names have no "_" or "-"
- 2026-10-02 — added a DELIVERABLES field after CONTEXT
- 2026-10-02 — added meta-instruction: every project has a system design doc at docs/SYSTEM-DESIGN.md
- 2026-10-05 — added meta-instruction: applications include a settings button or tab
- 2026-10-08 — added meta-instruction: color-code tags by similarity (related share a hue family, each distinct)
