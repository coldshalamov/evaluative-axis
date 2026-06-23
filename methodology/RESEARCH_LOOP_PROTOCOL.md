# Research Loop Protocol

Date: June 23, 2026

This protocol exists to keep the project from collapsing into a single
benchmark, a single paper, or a single convenient interpretation. It turns the
research into alternating modes with explicit outputs.

## The Loop

Every substantial research cycle must pass through six modes:

1. Idea Mode
2. Literature Mode
3. Experiment Mode
4. Autopsy Mode
5. Forest Mode
6. Decision Mode

Do not skip Autopsy or Forest Mode. Those are the modes that were missing when
raw HH-RLHF agreement was mistaken for the result.

## Mode 1: Idea Mode

Purpose: expand the possible implications before narrowing.

Questions:

- If this signal is real, where in the AI pipeline could it help?
- What does it make cheap that used to be expensive?
- What does it make dense that used to be sparse?
- What does it make automatic that used to require human judgment?
- What new training technique could exist if this worked?
- What non-obvious place could embeddings evaluate: tool calls, judge reports,
  reasoning traces, synthetic data, pretraining documents, code diffs?

Required output:

- A mechanism inventory, not just a hypothesis.
- At least five possible applications.
- At least three experiments that do not use the same benchmark.

Template: `methodology/templates/idea_mode.md`

## Mode 2: Literature Mode

Purpose: extract implications, not summaries.

For every paper, ask:

- What mechanism does this paper reveal?
- What assumption in our project does it support?
- What assumption does it attack?
- What technique can we borrow?
- What experiment does it suggest?
- What vocabulary does it give us for explaining the idea?
- What gap remains after this paper?

Required output:

- An implication matrix.
- A list of borrowed methods.
- A list of threats to the project.
- A list of opportunities the paper makes visible.

Template: `methodology/templates/literature_mode.md`

## Mode 3: Experiment Mode

Purpose: run a narrow test without letting it define the whole project.

Before running, write:

- what the test can answer;
- what it cannot answer;
- what assumption the benchmark makes;
- what result would change our belief;
- what result would be ambiguous;
- which examples must be inspected afterward;
- what baselines are required.

Required output:

- Frozen protocol.
- Saved code/config.
- Saved raw outputs.
- Saved examples for autopsy.

Template: `methodology/templates/experiment_mode.md`

## Mode 4: Autopsy Mode

Purpose: read the examples and find what the metric hid.

Required inspections:

- top embedding-over-benchmark disagreements;
- top benchmark-over-embedding disagreements;
- near-ties;
- cases where both options score low;
- cases where both options score high;
- surprising wins;
- surprising failures.

Questions:

- Did the benchmark lie?
- Did the embedding fail?
- Did both fail?
- Is the pair useless for training?
- Did the signal discover a different task?
- Did the failure suggest a better granularity or interface?

Required output:

- Failure taxonomy.
- Dataset-noise taxonomy.
- Mechanism implications.
- Examples that should appear in the paper.

Template: `methodology/templates/autopsy_mode.md`

## Mode 5: Forest Mode

Purpose: step back from the current metric and ask what the result means for
the whole research program.

This mode must run:

- after every experiment;
- after every major literature batch;
- whenever a result seems "weak";
- whenever a result seems "too good";
- before declaring a phase complete.

Questions:

- Did this test the real idea or a proxy?
- Did we accidentally ask the wrong question?
- What did the examples show that the aggregate hid?
- What new mechanism became visible?
- What new application became plausible?
- What would this imply if it scaled with embedding model quality?
- What would this imply if evaluated over full context instead of final answer?
- What would this imply for training, not just evaluation?
- What would matter to someone with hardware?
- What one experiment would most change our belief now?

Required output:

- A forest-pass memo.
- Updated mechanism map.
- Updated next-experiment priority.
- A sentence beginning: "The broadest useful interpretation is..."

Template: `methodology/templates/forest_mode.md`

## Mode 6: Decision Mode

Purpose: decide the next move without pretending uncertainty is completion.

Possible decisions:

- promote finding to central evidence;
- keep finding exploratory;
- rerun with stronger model or better context;
- run disagreement audit;
- change granularity;
- switch from dataset correlation to intervention;
- seek collaborator/hardware support;
- abandon a path because examples contradicted it.

Required output:

- Decision.
- Evidence used.
- Remaining uncertainty.
- Next action.
- What would reverse the decision.

Template: `methodology/templates/decision_mode.md`

## Periodic Broad-Thinking Checkpoints

To force forest-level thinking, schedule these checkpoints:

- Every 1 experiment: Autopsy Mode and Forest Mode.
- Every 3 experiments: rewrite the mechanism map.
- Every 5 papers: literature implication synthesis.
- Every major negative result: ask whether the benchmark or interface was wrong.
- Every major positive result: ask how it could be an artifact.
- Before spending money or GPU time: complete all six modes in writing.

## The Anti-Trap Rule

If the conclusion is only:

> "The score was X%."

the research cycle is incomplete.

A complete conclusion must say:

- what X% measured;
- what X% failed to measure;
- what examples showed;
- what disagreement revealed;
- what mechanism became more plausible;
- what application this suggests;
- what test should happen next.

## Current Project-Specific Forest Questions

For this embedding-evaluator project, always ask:

- Is this using embeddings as final-answer reward, data filter, reranker,
  critique scorer, process scorer, or something else?
- Did we score raw text, decomposed evaluation, or cumulative context?
- Would the result change with a longer-context embedding model?
- Did the embedding identify bad answers that the dataset rewarded?
- Did the embedding identify pairs where both answers should be rejected?
- Could this be used before training, during training, or during inference?
- Is this signal cheaper or more robust than vanilla LLM-as-judge?
- Does this suggest a flywheel between embeddings and LLM training?

## Definition Of A Real Research Cycle

A real cycle is not "run script, report number." A real cycle is:

1. generate possible mechanisms;
2. map relevant literature implications;
3. run a frozen test;
4. inspect examples;
5. zoom out;
6. choose the next test based on what was learned.

That is the operating system for this project.
