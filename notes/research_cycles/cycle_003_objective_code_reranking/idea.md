# Idea Mode: Cycle 003

Date: June 25, 2026
Cycle: `cycle_003_objective_code_reranking`

## Goal

Choose the highest-value next experiment under the real project constraints:

- one laptop;
- free-tier Google access;
- no hired annotators;
- no assumption that HH-RLHF is truth;
- strong need for falsification rather than comforting overlap metrics.

## Tradeoff Lens

A good next test should score well on five things:

1. **Thesis fit**: does it actually test evaluative geometry as a useful signal?
2. **Falsification power**: can it clearly fail and teach us something?
3. **Budget fit**: can it run with the current hardware and API situation?
4. **Contamination resistance**: is it hard to fake with length, labels, or judge leakage?
5. **Decision value**: does the result tell us what to do next?

## Falsification Pressure

Before proposing new tests, the current repo already falsifies several weak
versions of the idea:

- **Raw HH agreement is not the result.**
  The project already documents HH as one noisy sensor among several.
- **Broad direct good/bad scoring is not enough on small open models.**
  On the exact length-balanced v2 battery, direct broad scoring failed badly.
- **Oracle decomposition is not valid evidence.**
  Hand-authored `Good parts` / `Bad parts` text leaks the answer.
- **Generic process scoring does not automatically work.**
  The first naive trajectory probe failed.

So the next test should not be another HH-only overlap run, another length-leaky
candidate benchmark, or another oracle decomposition demo.

## Candidate Tests

Scores are 1-5 on:

- `fit`: thesis fit
- `falsify`: falsification power
- `budget`: laptop/free-tier feasibility
- `clean`: contamination resistance
- `decision`: decision value

Total is out of 25.

| Rank | Test | fit | falsify | budget | clean | decision | total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Objective code reranking with hidden unit tests, comparing direct answer scoring vs blind critique scoring | 5 | 5 | 4 | 5 | 5 | 24 |
| 2 | Length-controlled no-leakage intervention with generated open-ended answers and Gemini pairwise adjudication | 5 | 4 | 4 | 3 | 5 | 21 |
| 3 | Objective math reranking with exact-answer checks and optional reasoning traces | 4 | 5 | 4 | 5 | 3 | 21 |
| 4 | Injected error/repair potential-shaping benchmark on natural reasoning traces | 5 | 4 | 4 | 4 | 4 | 21 |
| 5 | Gemini-blind adjudication of the 108 HH disagreement cases | 3 | 4 | 5 | 3 | 5 | 20 |
| 6 | Expand the exact length-balanced controlled battery from 12 to 50 cases, then rerun Gemini embeddings | 4 | 4 | 4 | 5 | 3 | 20 |
| 7 | Label-free critique ablations: evaluative-word stripping, critique shuffling, answer-only vs critique-only | 4 | 5 | 4 | 5 | 2 | 20 |
| 8 | Tool-result interpretation benchmark with exact answer keys and overclaim traps | 4 | 4 | 4 | 5 | 3 | 20 |
| 9 | Candidate rejection/regenerate benchmark for both-bad sets | 4 | 4 | 4 | 4 | 4 | 20 |
| 10 | Multi-sensor Gemini rerun on HH, PKU, and SHP with fixed axes | 3 | 3 | 4 | 3 | 4 | 17 |
| 11 | Persona-honesty adversarial scale-up with blind Gemini adjudication | 3 | 4 | 5 | 4 | 1 | 17 |
| 12 | Contradiction / quotation / endorsement / negation diagnostic battery | 3 | 4 | 5 | 5 | 0 | 17 |
| 13 | Reward-hack robustness set: positive-tone bad answers vs neutral good answers | 3 | 4 | 5 | 5 | 0 | 17 |
| 14 | Pair sanitation benchmark: raw HH vs cleaned HH under Gemini judge | 3 | 4 | 4 | 3 | 3 | 17 |
| 15 | Cross-domain summarization reranking against rated datasets | 3 | 3 | 3 | 3 | 4 | 16 |
| 16 | Writing-quality reranking against essay scores | 3 | 3 | 3 | 3 | 4 | 16 |
| 17 | Evolutionary scalar-feedback loop on reasoning traces without direct decomposition prompting | 5 | 3 | 3 | 3 | 2 | 16 |
| 18 | Pretraining-document quality pilot with manual top/bottom inspection | 3 | 2 | 4 | 3 | 3 | 15 |
| 19 | Small Colab DPO run from embedding-ranked preference pairs | 5 | 4 | 2 | 3 | 1 | 15 |
| 20 | Open-ended safety/helpfulness reranking on public multi-response datasets without strict length control | 4 | 2 | 4 | 1 | 2 | 13 |

## Why The Top Test Wins

The top-ranked test is the best compromise because it:

- avoids HH as the primary truth source;
- avoids human-review dependence;
- avoids oracle decomposition;
- gives objective pass/fail outcomes through hidden tests;
- directly evaluates the practical claim: can the embedding-based selector pick
  a better candidate from several answers?

It is not a perfect test of the full alignment thesis. Code correctness is
narrower than open-ended helpfulness, honesty, and safety. But that is a
strength at this stage: the outcome is verifiable, so a negative result is
informative rather than debatable.

## The 5 Highest-Value Targets

1. **Objective code reranking with hidden tests**
   Why: cleanest zero-budget intervention test with no human label dependence.
2. **Length-controlled no-leakage open-ended reranking**
   Why: most direct practical claim if we can trust the adjudicator.
3. **Objective math reranking with exact answers**
   Why: another no-human objective domain, closer to natural-language reasoning.
4. **Injected error/repair potential-shaping benchmark**
   Why: strongest direct test of the dense-process-reward hypothesis.
5. **Gemini-blind HH disagreement adjudication**
   Why: strengthens the label-noise story cheaply, but does not by itself prove
   practical usefulness.

## Decision

Run Target 1 first.

If it fails:

- the result is still valuable because it means the current evaluative geometry
  interface does not automatically transfer to objective code correctness;
- next move should be Target 4 or stronger critique/decomposition ablations.

If it succeeds:

- the project gains a hardware-cheap, objectively scored intervention result;
- next move should be Target 2 or Target 3 to test generalization beyond code.
