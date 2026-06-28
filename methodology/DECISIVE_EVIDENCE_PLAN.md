# Decisive Evidence Plan

Date: June 24, 2026

## Purpose

This memo separates three different claims that are easy to blur together:

1. the evaluative axis exists in embedding space;
2. the axis is useful for choosing better answers;
3. the axis can serve as a training signal.

The project already has suggestive evidence for Claim 1. It does not yet have
partner-grade proof for Claims 2 or 3. The goal of this plan is to define the
cheapest falsifiable path from the current repo state to evidence that a
serious partner or investor could inspect without feeling misled.

## What The Repo Can Honestly Claim Today

- broad evaluative geometry is plausible from prior literature and from the
  repo's axis-convergence work;
- the HH disagreement audit is suggestive evidence that embedding scoring can
  expose label noise and outdated preference norms;
- cheap open models can fail badly under exact length control, which is useful
  because it rules out easy self-deception;
- oracle decomposition can create high scores by leaking the answer into the
  scored text, so any positive decomposition result must be treated with
  suspicion unless leakage is actively blocked.

That is a respectable research starting point. It is not yet a decisive proof
of practical usefulness.

## What Does Not Count As Proof

These are the main reward-hacky failure modes to avoid:

- agreement with HH, PKU, or SHP labels by itself;
- any benchmark where longer answers win by construction;
- any decomposition that contains `good`, `bad`, `better`, `worse`, or an
  experimenter-written answer key;
- single-reviewer non-blind adjudication used as if it were ground truth;
- small hand-built batteries presented as if they were generalization evidence;
- broad claims about training before a reranking result exists.

If a result depends on one of these shortcuts, it can still be useful as
internal debugging, but it should not be used in an external pitch.

## The Three Claims

### Claim 1: Evaluative Geometry Exists

This is the easiest claim and is already partially supported.

Required evidence:

- prior-work chain: Osgood -> semantic projection -> embedding geometry;
- axis convergence across unrelated domains;
- negative examples documented honestly: sycophancy, confidence, negation.

External framing:

> There is strong evidence that embedding spaces contain broad evaluative
> structure. The open question is whether that structure is useful enough to
> become a cheap alignment signal.

This is the safe claim to lead with today.

### Claim 2: Evaluative Geometry Improves Selection

This is the first decisive practical claim.

Question:

> If we generate several candidate answers for the same prompt, does the
> embedding-based selector choose better answers than cheap baselines?

Minimum credible experiment:

- 50-100 prompts;
- 4-6 candidates per prompt;
- candidate lengths matched exactly or binned tightly enough that length cannot
  dominate;
- methods compared:
  - random;
  - length;
  - sentiment;
  - refusal heuristic;
  - direct answer embedding score;
  - blind decomposition plus embedding score;
  - LLM judge for comparison only;
- blinded pairwise adjudication of method-selected winners.

Pass condition:

- embedding method beats random, length, sentiment, and refusal on blind
  review;
- if decomposition helps, it also survives leakage ablations;
- example audit shows plausible wins rather than obvious verbosity or label
  reading.

This is the first result that can credibly interest a partner.

### Claim 3: Evaluative Geometry Can Train Behavior

This is the strongest and hardest claim.

Question:

> Does full-context evaluative potential provide useful dense reward for
> reasoning or response trajectories?

Minimum credible experiment before training:

- 30-50 short trajectories with injected bad steps and later repairs;
- score cumulative context after each step;
- evaluate whether score deltas localize the error and recognize repair better
  than final-answer-only scoring.

Only after that:

- construct preference pairs or ranked traces from the signal;
- train a small model with DPO or a related method;
- evaluate on held-out prompts for quality, reward hacking, and length drift.

Do not jump to this stage unless the selection claim is already positive.

## Cheapest Laptop-Scale Program

This is the recommended order under the current hardware and quota limits.

### Step 1: Blind The Existing HH Disagreement Packet

Use the existing 108-case gradeable packet.

Goal:

- strengthen the claim that the axis is a useful label-noise detector;
- produce an artifact that a third party can inspect.

What it proves:

- not practical selection yet;
- but it does show the signal is not merely parroting one old preference
  dataset.

### Step 2: Run Pairwise Blind Review On Current Intervention Outputs

Use the existing scored pilot runs, but evaluate method-selected winners by
blind review rather than proxy labels.

Use:

- `scripts/build_pairwise_review.py`
- `scripts/analyze_pairwise_review.py`

This does not fix the length-biased candidate pool, but it converts the current
pilot into a more honest evidence artifact and reveals where the methods really
disagree in human terms.

### Step 3: Build A Clean Length-Controlled Reranking Set

This is the first experiment that could produce a positive external claim.

Rules:

- exact word-count matches where feasible;
- otherwise pre-registered length bins with no method-specific filtering;
- candidate generation prompt frozen before judging;
- blind review packet generated before anyone sees method names;
- held-out split reserved before tuning axes or prompts.

### Step 4: Add Leakage Ablations

For any decomposition-based method:

- strip evaluative words;
- shuffle decompositions between candidates;
- score decomposition alone;
- score answer alone;
- flip candidate order.

If the gain disappears completely under these checks, the method is reading
labels rather than evaluating substance.

### Step 5: Run A Small Potential-Shaping Battery

Keep this narrow and diagnostic. The point is not to prove full RL on a laptop.
The point is to show that dense evaluative deltas can detect degradation and
repair in a trajectory.

## What An Investor Or Partner Packet Should Contain

Do not lead with every internal experiment. Lead with a clean evidence ladder.

Minimum partner packet:

1. one-page thesis statement;
2. what is already known from literature;
3. one blind disagreement artifact showing label-noise detection;
4. one clean reranking result beating length and random;
5. one honest limitations section;
6. cost comparison against LLM-as-judge.

Ideal packet:

1. Claim 1 summary with citations;
2. Claim 2 blind reranking result with win-rate table and examples;
3. failure-mode table: sycophancy, confidence, negation, both-bad cases;
4. roadmap to Claim 3;
5. replication plan with a stronger embedding model.

## Hard Rules For Agents Working In This Repo

- Never call HH agreement alone "proof."
- Never promote oracle decomposition as evidence.
- Never hide length baselines.
- Never treat a single proxy key as blind truth.
- Always separate geometry evidence, selection evidence, and training evidence.
- Always preserve examples of failure; they increase credibility.

## Concrete Near-Term Success Criteria

The project becomes externally interesting if it can show all three:

1. blind adjudication says the embedding side wins a meaningful share of HH
   disagreement cases;
2. a length-controlled reranker beats length and random under blind review;
3. the cost per decision is dramatically lower than LLM-as-judge.

That is enough for a serious "this may be a new alignment primitive" story.
Without Item 2, the thesis is still promising but remains mostly conceptual.
