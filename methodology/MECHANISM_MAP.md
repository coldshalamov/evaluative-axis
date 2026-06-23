# Mechanism Map: Embedding Evaluative Geometry

Date: June 23, 2026

This file tracks possible places where embedding-space evaluation could improve
LLM training, inference, evaluation, or data pipelines. It should be updated
after every Forest Mode pass.

## Primary Claim

Embedding-space evaluative geometry may provide a cheap signal about whether
text moves toward or away from "good" in a broad human sense. The signal can be
used at many points in the pipeline, not only as a replacement for preference
labels.

## Mechanisms

| Mechanism | Pipeline location | What gets embedded | Decision changed | Why it matters | Status |
| --- | --- | --- | --- | --- | --- |
| Preference-pair sanitation | Before DPO/RLHF | Prompt + chosen/rejected answers | Keep, drop, audit, or regenerate pair | Avoid training on both-bad or mislabeled pairs | Supported by HH disagreement audit |
| Pair weighting | DPO/RLHF data prep | Prompt + answer pair | Weight training pair by embedding margin | Stronger clean pairs get more influence; ambiguous pairs get less | Proposed |
| Candidate reranking | Inference / data generation | Prompt + each candidate answer | Select best candidate from N samples | No training needed; directly tests practical value | Next decisive test |
| Critique scoring | Evaluation / data generation | LLM-generated critique or decomposition | Select answer whose critique scores highest | Lets LLM expose evaluative reasoning, embedding supplies cheap deterministic score | Proposed |
| Judge-of-judges | Evaluation | LLM judge report | Accept, reject, or audit judge decision | Detects judge rationalization or low-quality judgment | Proposed |
| Cumulative process reward | RL / reasoning training | Full context after each reasoning step | Reward deltas across trajectory | Dense supervision without trained process reward model | Proposed no-GPU simulation |
| Tool-trace evaluation | Agent training / eval | Tool call output + model interpretation | Reward good tool use and punish misuse/misread outputs | Evaluates action trajectory, not just final answer | Proposed |
| Synthetic-data filter | Data generation | Generated instruction/answer examples | Keep or discard synthetic samples | Cheap scalable quality filter | Proposed |
| Pretraining curation | Pretraining | Documents or document chunks | Weight/filter corpus | General quality signal at scale | Proposed |
| Active-learning router | Human review | High-disagreement or uncertain cases | Send to human/strong judge | Uses humans where information gain is highest | Proposed |
| Quality conditioning | SFT/pretraining | Text plus quality tag | Learn high-quality style distribution | Model can learn bad/good distinction without hiding bad text | Proposed |
| Axis diagnostics | Evaluation | Same text scored on basis axes | Explain what kind of good/bad is present | Broad score plus interpretable dimensions | Proposed |

## Scalar Plus Basis

The broad good/bad score is the primary scalar. Specific axes are diagnostics
and nudges:

- honesty / dishonesty;
- helpfulness / uselessness;
- safety / harm;
- calibration / overconfidence;
- non-sycophancy / flattery;
- agency respect / manipulation;
- risk disclosure / risk hiding;
- craftsmanship / sloppiness;
- truth correction / false agreement.

The broad score should prevent over-optimization of any one virtue into its
failure mode. Specific axes explain or steer local behavior.

## Dense Supervision Concept

Outcome-only reward asks whether the final answer was good. Cumulative-context
embedding reward asks whether each new reasoning step made the entire trajectory
better or worse in context.

Protocol sketch:

1. Embed context up to step `t`.
2. Score against broad good/bad and basis axes.
3. Add the next reasoning step, tool call, or answer segment.
4. Embed the full updated context.
5. Use score delta as process signal.

This is the most direct way to turn embedding evaluation from a labeler into a
training mechanism.

## Evidence So Far

- Full HH disagreement grading suggests raw HH agreement substantially
  understated the signal: among gradeable disagreements, embedding was judged
  better 65/109 times; many remaining disagreements were both-bad or low-signal
  pairs.
- Persona honesty example suggests embedding score can prefer modern assistant
  norms over older HH labels.
- Context-polarity test suggests aspect-specific axes can bind context better
  than broad generic anchors.
- Gemini controlled-pair result suggests embedding model quality matters.
- Reported axis-convergence result suggests domain-specific good/bad axes may
  share a broad evaluative direction; this needs local reproduction.

## Key Unknowns

- Does embedding-based candidate selection beat random, length, sentiment, and
  vanilla LLM-as-judge under blind review?
- Does cumulative-context scoring identify good and bad reasoning steps before
  the final answer?
- Does the signal improve with frontier embedding models and longer context?
- Which granularity is best: response, critique, decomposition, step, turn, or
  full context?
- Can broad good/bad avoid failure modes of over-optimizing specific axes?
- Can this improve actual fine-tuning or RL, not just evaluation?

## Next Forest-Pass Questions

- Which mechanism is currently strongest on evidence?
- Which mechanism is highest-upside if true?
- Which mechanism can be tested without hardware?
- Which mechanism would most impress a skeptical ML collaborator?
- Which mechanism would require long-context embeddings or GPU training?
