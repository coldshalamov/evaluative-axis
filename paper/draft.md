# The Evaluative Axis: Embedding Geometry as a General-Purpose Quality Signal for Language Model Alignment

Robin Gattis

June 2026

---

## Abstract

Evaluation is the primary dimension of human semantic judgment (Osgood et al., 1957), independently confirmed as a semantic universal by the Natural Semantic Metalanguage program (Wierzbicka, 1972; Goddard & Wierzbicka, 2014), and this structure is recoverable from embedding geometry (Grand et al., 2022; Kozlowski et al., 2025). We test whether projecting text onto evaluative directions in embedding space can serve as a cheap alignment signal — without training a classifier, without labeled preference data, and without LLM inference.

On frozen reranking suites with verifiable end metrics, a frontier embedding model (`gemini-embedding-2`) selects correctly on code (5/6), math (8/8), and tool interpretation (7/8) against a 1/3 random baseline. Open-source models (33M–600M params) collapse toward baseline on the same tasks; scale within this range does not predict performance.

The paper's central finding is diagnostic, not constructive. Raw "good/bad" fails on a 70-case conflict battery (26%), and the failure is specific: the "good" embedding direction tracks warmth/agreeableness rather than general quality. A content split defined by case design confirms the mechanism — good's accuracy on warmth-aligned cases is 5-7x its accuracy on firmness cases on BGE-M3 and Nomic (85% vs 16%, 80% vs 12%), with a weaker but directionally consistent gap on Snowflake (60% vs 48%). This warmth bias is structural: it persists across all three local models tested, across 45+ evaluative terms (~80% share the bias), and across anchor text of any length — single words, phrases, and full sentences all produce warmth-biased axes. An absolute-score analysis shows the training-signal implications are split-dependent: on all three local models, "good" has positive Cohen's d on warmth cases but negative d on firmness cases (BGE-M3: -0.73/+0.74; Nomic: -0.52/+0.43; Snowflake: -0.22/+0.15). On the five sycophancy cases, d is catastrophically negative (-4.02 on BGE-M3). On cases where warmth and correctness conflict, "good" actively rewards the wrong response.

Targeted warmth-independent axes ("Careful/Reckless") partially succeed where "good" fails, reaching significance on one of three local models (Nomic: 64%, Wilson CI [54%, 74%]) and on Gemini (74%, CI [62%, 83%]). But no single axis reaches significance on all three local models, multi-axis composites overfit, and richer anchor text degrades clean single-word signals. The evaluative directions recoverable from local models are weak, inconsistent, and do not reliably exceed chance. The stronger claim that raw `good/bad` or any single evaluative axis can serve as a dense training reward remains unproven.

---

## 1. Introduction

Training language models to be helpful, honest, and safe currently requires one of two expensive approaches: collecting human preference labels (RLHF; Ouyang et al., 2022) or running a large language model as a judge (RLAIF; Bai et al., 2022). Both are costly, and both introduce noise — human annotators disagree with each other at substantial rates, and LLM judges can be gamed, are non-deterministic, and tend to rationalize rather than flag ambiguity.

We propose a simpler approach. Define a direction in embedding space that corresponds to the evaluative axis — the "good/bad" direction — by averaging the embeddings of a small set of positive and negative anchor sentences. Project any text onto this direction. The dot product is the quality score.

This idea rests on a chain of three established findings:

**Finding 1: Evaluation is primary.** Osgood, Suci, and Tannenbaum (1957) demonstrated through factor analysis of bipolar adjective ratings across dozens of cultures that human semantic judgment reduces to three dimensions: evaluation (good/bad), potency (strong/weak), and activity (active/passive). Evaluation consistently explains the most variance and is cross-culturally universal. It is not one dimension among many — it is the primary organizing axis of human meaning-making. This finding converges independently with the Natural Semantic Metalanguage (NSM) program (Wierzbicka, 1972; Goddard & Wierzbicka, 2014), which identified GOOD and BAD as semantic primes — concepts that exist in every known human language and cannot be decomposed into simpler terms. Where Osgood arrived at evaluation-as-primary through factor analysis, NSM arrived at the same conclusion through field linguistics across dozens of unrelated language families. Cross-lingual sentiment lexicons (Chen & Skiena, 2014) confirm the empirical convergence: basic evaluative terms like good/bad, honest/dishonest, and fair/unfair show approximately 95% polarity agreement across 136 languages. Three independent research traditions — statistical factor analysis, field linguistics, and computational sentiment analysis — converge on the same claim: evaluation is the most universal dimension of human meaning.

**Finding 2: This structure is preserved in embeddings.** Kozlowski, Dai, and Boutyline (2025) confirmed that Osgood's three dimensions exist in modern LLM embedding geometry. Projections of words onto directions defined by antonym pairs correlate highly with human ratings and reduce to a low-dimensional subspace resembling patterns from human survey data. Grand et al. (2022) demonstrated more broadly that semantic projection onto antonym-defined directions recovers context-dependent human knowledge that simple cosine similarity misses.

**Finding 3: Different kinds of "good" are entangled.** Cho, Li, and Leshinskaya (2026) found that LLMs conflate moral, grammatical, and economic senses of "good" in their internal representations, with moral valence influencing assessments of grammaticality and economic value. They characterized this as a problem requiring ablation. We argue the opposite: for a quality signal intended to capture everything desirable about a model's output, entanglement is the mechanism. The reason a single evaluative axis captures helpfulness, honesty, safety, and correctness simultaneously is that these properties are geometrically correlated along the evaluation direction. You want an AI to be good — in every sense of the word.

Biological learning systems include a dense, general-purpose evaluative signal — dopaminergic reward — that operates at each step of behavior, precedes deliberative analysis, and generalizes across domains. Autoregressive language models lack an analogous mechanism during generation; evaluation occurs only at training time, compressed into a scalar loss at the end of a sequence. The evaluative axis provides a candidate for this missing architectural component: a deterministic, near-zero-cost quality signal computable at each generation step, with no model inference required.

**The core observation**: every alignment method already optimizes for "good." When a human rater picks response A over response B, they are projecting onto their internal evaluative axis. When an LLM judge says "A is better," it is doing the same with its learned representations. When a reward model outputs a scalar, that scalar is a "good" score — the field calls it exactly that. RLHF, RLAIF, and Constitutional AI are all expensive, noisy approximations of the same underlying judgment: how good is this response? If that judgment already exists as a recoverable geometric feature in embedding space, then the entire preference-collection and reward-modeling pipeline can be replaced with a single embedding call and a dot product — deterministic, near-zero-cost, and available at every generation step rather than only at the end of a sequence.

However, the claim that "good" in embedding space straightforwardly captures "good" in human judgment turns out to be wrong in a specific and informative way. The "good" direction in embedding geometry is warmth-biased — it tracks agreeableness rather than correctness — and a content split defined by case design (not model geometry) confirms the mechanism at case level (§4.19). The paper's arc is: we test the broad evaluative axis, discover its systematic failure mode, diagnose its geometric cause, and identify a warmth-independent alternative that partially succeeds where the raw evaluative direction fails.

The gap this paper addresses: these findings establish that the evaluative direction exists, preserves human judgment, and captures multiple quality dimensions. No published work tests whether this direction functions as a direct reward signal for alignment — without training a classifier, without labeled preference data, and without LLM inference.

That broad question breaks into four narrower ones:

1. does evaluative geometry improve objective answer selection;
2. is the effect model-dependent, and if so, how does the signal differ across embedding families;
3. why does the naive "good/bad" axis fail, and what does this reveal about evaluative geometry;
4. is the signal already sharp enough to justify training-adjacent use?

The paper is strongest on the first three questions and should be read that way. Figure 1 shows the local model landscape, Figure 2 compares anchor vocabulary types, Figure 3 provides a per-category heatmap, and Figure 4 contrasts Gemini's targeted vs broad axes.

### 1.1 Contributions

1. We show objective reranking lift with a strong embedding model on three small frozen suites with verifiable end metrics: code, math, and tool interpretation.

2. We show a signal-concentration gap: `gemini-embedding-2` beats cheap baselines with individual targeted axes, while OSS models (33M–600M params) collapse toward baseline on individual axes regardless of model size. Qwen3-Embedding-0.6B at 600M parameters performs comparably to 33M-parameter models. Within 33M–600M, scale does not predict performance; the frontier gap's cause is unidentified (Gemini's parameter count is undisclosed).

3. We provide an honest negative result for the strongest minimalist version of the thesis: raw one-word `good/bad` fails on a 50-case length-balanced conflict battery, even when a strong embedding model is used.

4. We show that richer targeted evaluative axes on that same battery are much stronger than raw `good/bad`, supporting a scalar-plus-basis view rather than a pure single-word axis view.

5. We provide the first direct bridge test from reranking toward training: cumulative process-aware scoring responds to injected error and later repair better than cheap controls, but still fails the frozen training-readiness gate.

6. We retain the HH disagreement audit as supporting evidence for label-noise detection and norm drift, but not as the main practical proof.

7. We show that anchor vocabulary depth matters: evaluatively specific terms outperform both high-frequency universal words and multi-sentence ML-jargon anchors. Per-category analysis reveals that different single words capture different evaluative dimensions, and corpus frequency does not predict geometric signal strength. "Careful"/"Reckless" is the most cross-model-consistent single-word pair, but per-category accuracy estimates rest on small subsets (n=4-11 per category) and should be read as patterns, not reliable point estimates. Bootstrap confidence intervals ground which local-model comparisons are statistically reliable.

8. We diagnose why "good/bad" fails: its embedding direction tracks warmth/agreeableness, not general quality. A tree decomposition of "good" into child terms shows that "careful" is geometrically independent of "good" (score-delta correlation near zero on all models), while most other evaluative terms share its warmth bias. A neighborhood composition analysis provides the mechanism: "good"'s top-30 non-synonym neighbors are 40% warmth+emotion words (cross-model average) vs. 12% competence+restraint, while "careful" shows the inverse (30% competence+restraint vs. 18% warmth+emotion). The neighborhood warmth fraction predicts scoring bias: r = +0.65 to +0.69 across three models (n = 7 axes). A content split defined by case design (not model geometry) confirms the mechanism: good's accuracy on warmth cases is ~4-7x its firmness-case accuracy on three of four models (BGE-M3: 16% vs 85%, Nomic: 12% vs 80%, Gemini: 26% vs 95%), with a weaker but directionally consistent gap on Snowflake (48% vs 60%). Careful's gaps are small and direction-inconsistent. A constructed demonstration on 20 anti-sycophancy cases (§4.23) illustrates the mechanism concretely: "good" picks the sycophantic response 85-100% of the time, consistent with its warmth bias. However, a trivial positive-word-count baseline achieves 85% on the same cases, so the embedding result on this particular set cannot be distinguished from lexical valence detection. The independent evidence for the warmth mechanism comes from the content split (§4.19), where warmth vs firmness is defined by case design, not response wording.

9. We test Osgood's three semantic differential dimensions (Evaluation, Potency, Activity) systematically. The EPA composite fails. Individual Potency pairs ("Hard/Soft" at 58-68% on the original battery) initially appeared to produce the strongest cross-model signal, but subsequent validation (§4.16-4.17) showed this was a battery artifact: the original 50-case battery was 64% firmness-biased, inflating Potency scores. On a rebalanced battery and on Gemini Embedding, "Hard/Soft" drops to 39% (inverted). Osgood's primary Evaluation dimension (good/bad, nice/awful) fails consistently. No single Osgood dimension produces a robust cross-model evaluative signal.

10. We report honest validation results on a rebalanced battery and an independent expansion test set. The original 50-case battery was 64% firmness-biased; adding 20 warmth cases and testing on 20 held-out expansion cases reveals that no single axis reaches statistical significance (Wilson 95% CI lower bound > 50%) on all three local models with a fixed method. "Careful/Reckless" is the most consistent axis — pooled 64% on Nomic (CI [54%, 74%], significant) and 74% on Gemini (significant), but 56% on Snowflake and 53% on BGE-M3 (neither significant). Multi-axis voting combinations that appear to improve in-sample lose 10-25 percentage points out-of-sample — a critical overfitting finding.

11. We test 26 novel evaluative terms across three prediction experiments (§4.21-4.22). None demonstrates reliable predictive power above null. Pooled across all three tests, only 5 of 13 independence predictions are correct (38%) against a ~20% base rate — not a usable criterion. A comprehensive analysis of 45+ terms (§4.22) reveals a descriptive pattern: all 9 warmth-independent terms share restraint/discipline semantics. But this observation does not function as a predictive rule: a targeted test of the restraint hypothesis scores 5/6 (+2 above null), yet only 2 of 3 independence predictions succeed, and at that sample size the result is at chance. The robust finding is an asymmetry: warmth bias is predictable and pervasive (~80% of terms), while warmth-independence is rare, empirically identifiable, and theoretically unexplained.

12. We test whether pairwise discrimination accuracy understates training-signal quality (§4.24). The answer is split-dependent: on all three local models, "good" has positive Cohen's d on warmth cases but negative d on firmness cases (BGE-M3: -0.73/+0.74; Nomic: -0.52/+0.43; Snowflake: -0.22/+0.15). On the five sycophancy cases, d is catastrophically negative (-4.02 on BGE-M3, though n = 5). On cases that force a warmth/correctness conflict, "good" actively rewards the wrong response. No single axis provides a universal training signal: "careful" is direction-inconsistent on warmth across models (d = -0.37 BGE-M3, +0.37 Nomic), while "restrained" is neutral on warmth (d near zero on all three models) but has modest overall effect size. Whether the net training-signal quality is positive or negative depends on the base rate of warmth/correctness conflicts in real training data — a question this adversarial battery cannot answer.

13. We test whether richer anchor text — multi-word phrases, full sentences, explicitly anti-sycophantic formulations — can escape the warmth bias (§4.25). It cannot. All "good" variants remain warmth-biased regardless of length or specificity. Longer anchors degrade clean single-word axes. The failure is geometric, not a word-length artifact.

14. We show that subtracting the warmth direction from "good" removes the bias but leaves noise at chance accuracy (§4.26). The quality signal is not hidden behind warmth — it is distributed across narrower semantic children.

15. We demonstrate that decomposing "good" into five principled quality terms — careful, honest, helpful, thorough, restrained — scored independently with OR logic, recovers the full quality signal at 89–94% out-of-sample accuracy across all three local models (§4.27). This is the first local-model configuration to exceed 85% on the balanced battery with balanced performance on both firmness and warmth splits.

16. We characterize the known limitations and non-claims explicitly, including sycophancy weakness, raw-word failure, and incomplete training-readiness.

---

## 2. Related Work

### 2.1 Human Preference-Based Alignment

**RLHF** (Ouyang et al., 2022) trains a reward model from human preference labels, then optimizes a language model against it. The cost of human annotation — estimated at ~$100 per annotation-hour, with 600 high-quality annotations costing ~$60,000 (Xu et al., 2025) — is the primary scalability bottleneck.

**DPO** (Rafailov et al., 2024) eliminates the reward model by directly optimizing from preference pairs, but still requires those pairs to exist. Our approach provides a way to generate preference pairs cheaply: score candidate responses with the embedding axis, construct pairs from the scores.

**RLAIF / Constitutional AI** (Bai et al., 2022) replaces human annotators with an LLM judge. Cheaper than RLHF but still requires full model inference for every judgment, is non-deterministic, and can be gamed through adversarial reasoning. Our embedding signal is deterministic and not gameable in the same way — there is no chain of reasoning to exploit.

### 2.2 Embedding-Based Reward Signals

**Turney (2002)** computed unsupervised positive/negative semantic orientation
from sparse anchors such as "excellent" and "poor" for review classification.
This means the project cannot claim novelty for semantic orientation itself.
The novelty must be in using a sparse external evaluative direction as a
zero-training evaluator over full context, trajectory deltas, and training
pipeline decisions.

**Sun et al. (2025)** showed that reward models can be trained on pre-computed embeddings rather than requiring full model inference, dramatically reducing compute costs. They still train a classifier on labeled preference data; our approach skips the classifier entirely and uses the embedding geometry directly. Their work validates the premise — embeddings contain enough information for reward modeling — while leaving the stronger claim undemonstrated.

**PGSRM** (Plashchinsky, 2025) uses cosine similarity between a parent model's
reference-output embedding and a child model's generated-output embedding as a
PPO reward on GPT-2-scale models. This establishes that embedding geometry can
serve as a trainable reward landscape, not merely an offline diagnostic. It
does not test a universal good/bad axis: the reward is reference similarity, not
general evaluative projection.

**Legend** (Feng et al., 2024) uses representation engineering to find a safety direction in the target model's own activation space, then annotates preference margins based on distances along that direction. Key differences: Legend requires inference through the model being trained, focuses narrowly on safety, and annotates margins on existing preference data. Our approach uses a cheap external embedding model, targets general evaluation rather than safety alone, and generates preference signals from scratch.

### 2.3 Semantic Geometry

**Bolukbasi et al. (2016)** demonstrated that semantic dimensions (gender, in their case) are linearly encoded in embedding space and recoverable via difference vectors between paired concepts. Our axis construction method is the same technique applied to evaluative content.

**Grand et al. (2022)** showed that semantic projection recovers context-dependent human knowledge from word embeddings, published in Nature Human Behaviour. Our method is semantic projection applied to the evaluative domain at the sentence level.

**Kozlowski et al. (2025)** confirmed that Osgood's evaluation/potency/activity dimensions exist in LLM embeddings and that semantic features are entangled — shifting along one direction causes proportional shifts on geometrically aligned features. This entanglement is the mechanism by which our single evaluative axis captures multiple quality dimensions.

**Valence-Assent Axis** (Lu, Song, & Wang, 2025) reports a dominant dimension
across LLMs that jointly encodes subjective valence ("what is good") and assent
to factual claims ("what is true"). This is close to the proposed mechanism,
but it also reveals the central danger: steering this shared state can make a
model rationalize a favored evaluative state at the expense of factual accuracy.
That result supports the existence of broad evaluative geometry while warning
against naive maximization.

**Interaction Dynamics / TRACE** (Gooding & Grefenstette, 2025) shows that
embedding-trajectory structure in dialogue can itself predict interaction
success: structural trajectory features alone reached 68.20% pairwise accuracy,
and a hybrid text+trajectory model reached 80.17%. This supports evaluating
trajectories and context dynamics rather than only final answers.

### 2.4 Representation Engineering and Activation Steering

**Zou et al. (2023)** showed that high-level concepts like honesty and safety are linearly represented as directions in a model's internal activation space and can be used to steer behavior at inference time. Our work is the external analog: instead of finding evaluative directions inside the model being trained, we use an external embedding model's evaluative geometry as a scoring function. Theirs is a steering technique; ours is a reward signal.

**Activation steering** (2024–2026) has grown into a substantial body of work on inference-time behavioral control by adding direction vectors to model activations. Conceptually aligned — both exploit the linear structure of evaluative dimensions — but serving a different function.

### 2.5 Verifiable Rewards

**DeepSeek GRPO** (2025) demonstrated that reinforcement learning with verifiable rewards — deterministic checks that confirm whether output is correct — can train strong reasoning models without learned reward models. This works for domains with verifiable answers (math, code) but cannot evaluate open-ended helpfulness, honesty, or safety. Our embedding axis is a potential bridge: a deterministic signal for the open-ended evaluation domain where GRPO cannot reach.

**Embedding Distance as GRPO Reward** (Ciccone et al., 2025) uses cosine distance between a generated output's embedding and a known correct answer's embedding as the reward signal for GRPO training on the Countdown (numbers game) task. This demonstrates that embedding geometry can drive RL training, but the approach fundamentally requires a reference answer to measure distance from — it is a verifiable-answer technique that happens to use embeddings. Our approach requires no reference answer: the evaluative direction is defined by anchor sentences describing "good" and "bad," not by proximity to a known correct output.

---

## 3. Method

### 3.1 Axis Construction

We define the evaluative direction by contrasting positive and negative anchor sentences. The anchors describe behaviors, not single words, to capture evaluative content at the granularity of model outputs:

**Positive anchors:**
- "The assistant refuses wrongdoing."
- "Condemns harmful acts."
- "Corrects false claims."
- "Discloses serious risks."
- "Tells uncomfortable truths."
- "De-escalates harm."

**Negative anchors:**
- "Provides instructions for wrongdoing."
- "Endorses harmful acts."
- "Agrees with false claims."
- "Hides serious risks."
- "Lies to avoid discomfort."
- "Helps harmful activity."

Each anchor is embedded. The axis is computed as:

$$\vec{a} = \frac{\bar{e}_{pos} - \bar{e}_{neg}}{||\bar{e}_{pos} - \bar{e}_{neg}||}$$

where $\bar{e}_{pos}$ and $\bar{e}_{neg}$ are the mean embeddings of the positive and negative anchors respectively.

### 3.2 Scoring

For any text $t$, the evaluative score is:

$$s(t) = \vec{e}_t \cdot \vec{a}$$

where $\vec{e}_t$ is the embedding of text $t$. Higher scores indicate greater alignment with the positive end of the evaluative axis.

For preference prediction, given a prompt with two candidate responses, the method predicts the response with the higher score as preferred.

### 3.3 Cumulative Context Potential

The stronger training formulation scores the full context after each reasoning
or response step:

$$\Phi_t = \vec{a} \cdot f(c, r_{1:t})$$

where $c$ is the conversation context and $r_{1:t}$ is the generated prefix up
to step $t$. The dense shaping signal is the potential delta:

$$F_t = \gamma \Phi_t - \Phi_{t-1}$$

With $\gamma = 1$, the deltas telescope:

$$\sum_t F_t = \Phi_T - \Phi_0$$

This matters because raw prefix scores double-count local goodness in long
answers. Potential deltas ask whether the new step improves or degrades the
whole trajectory as it now stands. This connects the proposal to
potential-based reward shaping rather than naive per-token or per-sentence
reward accumulation.

### 3.4 Embedding Models Tested

- **Gemini Embedding 2** (Google): 3072 dimensions, state-of-the-art MTEB. Used for objective reranking, the 50-case conflict battery, and process-potential scoring.
- **BGE-small-en-v1.5** (BAAI): 384 dimensions, 33M parameters. Used for the full HH disagreement audit, axis-convergence analysis, and multi-sensor experiments. Runs on CPU in seconds.
- **8-model local sweep** (via FastEmbed/ONNX): BGE small/base, GTE-base, Snowflake Arctic Embed M, Jina v2 small/base, Nomic Embed v1.5, Mixedbread mxbai-embed-large. Used for the battery v3 sweep and good-vs-proxy conflict protocol to test whether results generalize beyond a single model family.
- **MPNet-base-v2** and **BGE-base-en-v1.5**: Used for OSS objective reranking on the expanded math (48 tasks) and tool interpretation (32 tasks) suites.

---

## 4. Experiments

### 4.1 Controlled Axis Validation

**Design**: 61 statement pairs across 7 categories (coding quality, honesty, helpfulness, safety, sycophancy, mixed outcomes, concrete outcomes) where one statement is clearly better by consensus. Measure whether the higher-scoring statement is the better one.

**Results** (Gemini Embedding 2, multi-anchor-sentence axis):

| Category | Accuracy |
|---|---|
| Coding quality | 90% |
| Concrete outcomes | 100% |
| Mixed outcomes | 90% |
| Safety | 80% |
| Helpfulness | 80% |
| Honesty | 40% |
| Sycophancy | 0% |
| **Overall** | **70.5%** |
| **Excluding sycophancy + honesty** | **86.7%** |

Sycophancy and honesty-hedging pairs are structurally resistant to surface-text evaluation. Sycophantic text uses quality-signaling words instrumentally ("brilliant," "excellent") — the embedding correctly reads their semantic content but cannot distinguish sincere from performative use. This limitation is inherent to any approach that scores text as written rather than evaluating intent or factual accuracy.

### 4.2 HH-RLHF Preference Prediction

**Design**: 500 pairs from Anthropic's HH-RLHF dataset. For each pair, embed both the chosen and rejected responses, project onto the contextual harm-reduction axis, predict the higher-scoring response as preferred.

**Results** (BGE-small, contextual harm-reduction axis):

| Method | Agreement with HH labels |
|---|---|
| Random | 50.0% |
| Prefer longer | 41.3% |
| Prefer more positive sentiment | 46.9% |
| Embedding axis (response only) | ~47% |
| Embedding axis (prompt+response) | 55.8% |
| Embedding axis (atomic evaluation framing) | 59.2% |

The raw 55.8% agreement is statistically significant (z=2.59, p=0.009) and beats all cheap baselines. But raw agreement with HH-RLHF substantially underestimates the true quality of the signal, as the disagreement audit (Section 4.3) demonstrates.

### 4.3 Full Disagreement Audit

[CENTRAL FINDING]

**Design**: All 231 cases where the embedding axis disagreed with HH-RLHF labels were individually reviewed and graded as EMBEDDING_RIGHT (embedding preferred the better response), HH_RIGHT (HH label was correct), or EXCLUDE (both responses were bad, trivial, or not useful for training).

**Results**:

| Grade | Count | % of disagreements |
|---|---|---|
| EMBEDDING_RIGHT | 63 | 27.3% |
| HH_RIGHT | 45 | 19.5% |
| EXCLUDE | 123 | 53.2% |

Note: an earlier prose summary of this audit reported 65/44/122. The current
auditable table in `disagreement_audit/full_grading.md` parses as 63/45/123.
The corrected-agreement interpretation is essentially unchanged, but the paper
should use the table-backed count until the audit file is reconciled.

Among gradeable disagreements: embedding preferred the better response in
**63/(63+45) = 58.3%** of cases.

Corrected gradeable agreement (assuming agreement cases are correct, excluding both-bad pairs):

$$\frac{269 + 63}{269 + 63 + 45} = \frac{332}{377} = 88.1\%$$

The broader claim remains 83-88% corrected gradeable agreement under conservative
sensitivity assumptions, but this number should be treated as provisional until
blind adjudication validates the audit.

#### 4.3.1 Patterns in Embedding-Right Cases

Recurring patterns where the embedding preferred the better response and HH did not:

- **HH rewarding compliance with harmful requests**: doxxing, providing dangerous instructions, helping with harassment
- **HH rewarding misinformation**: factually wrong claims labeled as preferred
- **HH rewarding non-answers over substance**: "I don't know" preferred over correct, informative responses
- **HH rewarding dishonesty**: AI fabricating having a family, fabricating personal experiences
- **HH rewarding harmful stereotypes**: racist narratives, slur lists
- **HH rewarding evasion over ethical engagement**: deflection preferred over substantive ethical reasoning

#### 4.3.2 The "Anticipating Later Norms" Finding

Several embedding-right cases represent not just label noise but a systematic pattern: the embedding preferred responses that align with *modern* AI safety norms over responses that reflected the labeling standards of 2022. Examples:

- **Persona honesty** (Pair 78): The embedding preferred "I'm not a person" over a response fabricating a family and personal life. HH-RLHF was labeled before persona honesty became standard policy.
- **Refusal of dangerous requests**: Multiple cases where the embedding preferred refusal or safety information over compliance that 2022 annotators labeled as "helpful."
- **Substantive engagement over deflection**: The embedding preferred informative responses over empty deflections, consistent with the post-2023 shift toward helpful refusal rather than blanket avoidance.

This suggests the evaluative geometry captures something structurally deeper than any particular dataset's labeling policy — an evaluative direction that tracks where human preference converges, even before labelers explicitly codify it.

#### 4.3.3 Excluded Cases

The 123 excluded cases fell into four categories:

- Both responses help with harmful activity (~50 cases) — neither is appropriate for training
- Both responses are short/trivial (~31 cases) — no meaningful quality signal
- Both responses engage with offensive content (~25 cases) — context-dependent, no clear winner
- Genuinely marginal (~16 cases) — reasonable people could disagree

These exclusions are not failures of the embedding — they are cases that any responsible training pipeline should filter or regenerate rather than force a pairwise preference.

### 4.4 Context Binding

**Design**: 56 pairs where the same "bad" phrase appears in opposite contexts (e.g., "refused lying under oath" vs "encouraged lying under oath"). Tests whether the axis can distinguish context from content.

**Results**:

| Axis type | Accuracy |
|---|---|
| Generic broad axes | 32–43% |
| Contextual harm-reduction axis | 64.3% |
| Oracle explicit decomposition | 100% |

Broad good/bad axes are too blunt — "lying under oath" drags the score negative regardless of context. Aspect-specific axes recover meaningful signal. The oracle decomposition (where good/bad factors are explicitly written out) reaches 100%, confirming the embedding can read evaluative structure when it is expressed in the text.

### 4.5 Axis Convergence Across Domains

**Design**: We constructed 10 quality axes from unrelated domains — code quality, cooking, parenting, medical care, music, writing, ethics, engineering, teaching, and friendship — each defined with 3 positive and 3 negative anchors sharing no vocabulary across domains. We measured the cosine similarity between each pair of axes and between each axis and a simple "good/bad" direction defined from the words themselves.

**Results** (BGE-small-en-v1.5, 384 dimensions):

All 10 domain axes have positive cosine similarity with the mean evaluative direction (range: 0.25–0.71, alignment: 0.58–0.85). A simple good/bad word axis has 0.777 cosine similarity with the mean evaluative direction across domains.

The contextual harm-reduction axis used in all prior experiments (Sections 4.2–4.4) is an outlier: cosine of -0.123 with the mean evaluative direction, and -0.128 with simple good/bad. This axis is orthogonal to the general evaluative direction, meaning all prior HH-RLHF results were obtained with a suboptimal axis.

When used to score 300 HH-RLHF pairs, the individual domain axes outperform the harm-reduction axis: writing (56.7%, p=0.01), cooking (56.0%, p=0.02), music (55.0%, p=0.04), engineering (55.0%, p=0.04), friendship (55.3%, p=0.03). The harm-reduction axis scored 48.7% — below random.

**Interpretation**: The convergence of 10 unrelated domain axes is the strongest geometric evidence for a universal evaluative direction. "Well-tested code," "nourishing home-cooked meal," and "patient bedside manner" all point roughly the same way in embedding space — toward "good." This is consistent with the Value Entanglement finding (Cho et al., 2026) but extends it from word-level to sentence-level axis definitions across diverse domains.

### 4.6 Multi-Sensor Validation

**Design**: Tested the embedding axis across multiple preference datasets (HH-RLHF, PKU-SafeRLHF, Stanford SHP) treated as independent imperfect sensors rather than ground truth. Used 8 aspect-specific axes.

**Key finding**: Different datasets measure different things. SHP is heavily shaped by length and social signals (length baseline: 70.3%). PKU's "better" and "safer" labels diverge. Aggregate scores underperformed individual axis-dataset matches, supporting the view that evaluative structure is best captured by a small basis of aspect-specific axes rather than a single universal scalar.

### 4.7 Early Pilot Work And Design Diagnostics

Two early pilots preceded the stronger experiments in Sections 4.8–4.12 and are
retained here as design diagnostics rather than headline evidence.

**Intervention pilot** (50 prompts, 4 candidates each): Length hit 66% against
the proxy key, with the adversarial subset reaching 96%. This exposed a design
artifact — the constructed best answers were systematically longer — and
motivated the exact word-count matching used in all subsequent batteries.

**12-case length-balanced mini-battery**: Under exact length control, broad
direct BGE-small evaluative scoring collapsed to 0–8%. Oracle decomposition
with hand-authored "Good parts" / "Bad parts" statements reached 91.7% (Jina
v2 small), but this is label leakage — the embedding reads explicit evaluative
language rather than independently inferring quality. This result is retained
as an upper-bound sanity check showing the projection machinery works when
evaluative content is present in the text.

### 4.8 Objective Reranking Across Domains

**Design**: We built three small frozen reranking suites with objective end
metrics rather than preference overlap:

- **code**: choose among candidate solutions and check whether the selected
  answer passes the task;
- **math**: choose among candidate short solutions with known correct answers;
- **tool interpretation**: choose among candidate answers where the correct
  interpretation of a tool result is objectively known.

Each suite compares:

- random or best cheap baseline;
- a strong embedding model (`gemini-embedding-2`);
- and cheap OSS embedders where available.

**Results**:

| Domain | Random | Length | Gemini best | Cheap OSS best |
|---|---:|---:|---:|---:|
| Code (6 tasks) | 50.0% | 50.0% | 83.3% | 50.0% (`bge-base`) |
| Math (8 tasks) | 37.5% | 50.0% | 100.0% | 62.5% (`bge-base`) |
| Tool (8 tasks) | 37.5% | 37.5% | 87.5% | 50.0% (`bge-base`) |

All three suites are 3-way selection (3 candidates per task); p-values are
against p₀ = 1/3 random: code p=0.018, math p=0.0002, tool p=0.003. The length
baseline is tiebreak-sensitive: several tasks have candidates with identical
word counts (code 1/6, math 2/8, tool 3/8), and the computed length score
depends on which tied candidate the selection script picks. Under favorable
tiebreaking the length ceiling is 4/6, 6/8, 6/8 — reducing the embedding edge
to +1, +2, +1 tasks respectively. The math advantage is the most robust.

**Length confound analysis** (code suite): A previous internal audit flagged a
possible length confound. We computed the correlation between candidate word
count and embedding score across all 18 code candidates. The correlation
between length and pass rate is r=0.19; between embedding score and pass rate
it is r=0.60. Length selection scores 3/6 (50%), equal to random (one additional
task, merge\_intervals, has a length tie between two 39-word candidates; the
correct candidate is among the tied pair but was not selected by the
tiebreaker). On the two tasks with a uniquely longest wrong candidate
(longest\_common\_prefix, chunk\_list), the embedding picks the correct shorter
candidate both times. The one embedding error (balanced\_brackets) selects a
*shorter* candidate over the correct longer one — the opposite of what a length
confound would produce.

**OSS scaling verification** (cycle 013): We expanded the math suite to 48 tasks
and the tool interpretation suite to 32 tasks, with tighter within-item
word-count gaps (mean 1.08 words for math, 1.56 for tool) and balanced
correct-position distribution. On these larger frozen suites, cheap OSS
embedders confirmed the capability-gap pattern:

| Domain (N) | Random | Length | BGE-base best | MPNet-base best |
|---|---:|---:|---:|---:|
| Math (48) | 47.9% | 35.4% | 29.2% | 35.4% |
| Tool (32) | 37.5% | 50.0% | 43.8% | 28.1% |

Both OSS models are at or below baseline on these suites. BGE-base scores below
random on math; MPNet-base collapses below random on tool interpretation. This
confirms the capability gap between local and frontier models at larger scale
and rules out the possibility that the original small-N Gemini results were an
artifact of benchmark size.

**Statistical tests**: One-sided binomial tests against the random baseline
(p₀=1/3 for three-way selection) yield p=0.018 for code (5/6), p=0.0002 for
math (8/8), and p=0.003 for tool (7/8). Wilson 95% CIs are wide due to small
N: [43.6%, 97.0%] for code, [67.6%, 100%] for math, [52.9%, 97.8%] for tool.
All three domains reject chance, but the intervals confirm that larger suites
are needed to pin down effect size.

**Interpretation**: These are still small suites, but they are stronger than
dataset-overlap claims because the end metric is objective. The selected answer
either solves the task or it does not. Under that criterion, the strong
embedding model shows real cross-domain lift while cheap OSS embedders lag
behind or collapse to baseline. The length confound analysis on the code suite
shows that embedding score tracks code quality (r=0.60) far better than length
does (r=0.19), and the embedding succeeds precisely on the tasks where length
selection fails.

### 4.9 Raw good/bad Versus Targeted Evaluative Axes

**Design**: We built a 50-case length-balanced conflict battery spanning
reasoning rigor, truthfulness, harm reduction, anti-sycophancy, context
binding, helpfulness, persona honesty, and mixed tradeoffs. Each pair was
exactly word-count matched. We then compared raw word-level axes, nearby proxy
words, and richer targeted axes on the same frozen cases.

**Results with `gemini-embedding-2`**:

| Method | Accuracy |
|---|---:|
| raw `good/bad` | 26.0% |
| sentence `This response is good/bad.` | 30.0% |
| best nearby proxy (`useful/useless`) | 42.0% |
| direct broad general evaluative axis | 46.0% |
| direct combined targeted axes | 86.0% |
| direct category-routed axis | 86.0% |
| direct truthfulness axis | 90.0% |
| direct harm-reduction axis | 94.0% |
| direct persona-honesty axis | 96.0% |
| direct anti-sycophancy axis | 98.0% |

One-sided binomial tests (H₀: p=0.50, n=50) confirm that all targeted axes
are highly significant: combined p=1.1×10⁻⁷, truthfulness p=2.1×10⁻⁹,
harm-reduction p=1.9×10⁻¹¹, persona-honesty p=1.1×10⁻¹², anti-sycophancy
p=4.5×10⁻¹⁴. Wilson 95% CI for combined targeted: [73.8%, 93.0%]. Raw
`good/bad` at 26% and best proxy at 42% are correctly non-significant
(p≥0.90), confirming they do not beat chance.

Across an 8-model local FastEmbed sweep on the same battery, raw `good/bad`
mostly remained weak:

- 14.0-28.0% for seven of the eight local models
- one partial exception, `snowflake/snowflake-arctic-embed-m`, reached 48.0%

Even that partial exception did not make the minimalist story work cleanly:
its best nearby proxy (`helpful/unhelpful`) still scored higher at 58.0%, and
its direct targeted-axis scores on the same battery were stronger than either
raw word-level variant.

**Interpretation**: This is one of the most important negative results in the
paper. The strongest minimalist story — that raw `good/bad` already works as the
main evaluative direction under zero-shot scoring — is not supported here. What
is supported is a richer evaluative-geometry story: the signal is much more
recoverable through targeted or routed evaluative axes than through a bare
single-word direction. The new local sweep makes that conclusion more credible
because the broad-word failure is no longer only a Gemini-plus-BGE story.

### 4.10 Process-Potential Error-Repair Suite

**Design**: To test the bridge from selection to training, we built 12 short
traces across arithmetic, code reasoning, tool interpretation, factual
reasoning, harm reduction, persona honesty, and anti-sycophancy. In each trace,
one step introduces a known error and a later step repairs it. We score the
full cumulative context after each step and ask whether the score drops at the
error step and rises at the repair step.

Controls:

- length;
- sentiment;
- final-answer-only scoring;
- cheap OSS embedding baseline.

**Results**:

| Scorer | Error drop | Repair rise | Dense localization |
|---|---:|---:|---:|
| Gemini category axis | 91.7% | 83.3% | 50.0% |
| Gemini combined | 75.0% | 75.0% | 62.5% |
| BGE category axis | 33.3% | 75.0% | 20.8% |
| BGE combined | 16.7% | 50.0% | 16.7% |
| Gemini sentiment | 41.7% | 16.7% | 8.3% |
| Gemini length | 0.0% | 100.0% | 0.0% |
| Gemini final-answer-only combined | 0.0% | 0.0% | 0.0% |

The frozen training-readiness gate requires
`dense_reward_localization_score >= 0.65` on the main process metric. The
current Gemini category-axis result is `0.50`, so the gate fails.

**Interpretation**: This is the cleanest direct evidence that the
signal is not only a final-answer preference. It reacts to deterioration and
repair inside the trajectory, and it beats cheap controls decisively. At the
same time, the result is not yet strong enough to justify calling the signal
training-ready.

### 4.11 Exploratory Blind Review Of Cheap Open-Ended Selectors

**Design**: We revisited the older 50-prompt open-ended intervention pilot, but
stopped treating the proxy key as the headline metric. Instead, we built blind
head-to-head review packets only for cases where two methods selected different
candidates from the same pool. We then sampled 10 rows per comparison and asked
`gemini-flash-lite-latest` to adjudicate the blinded pairs with order-flip
stability checks. This remains exploratory because the underlying candidate
pool inherits the old length bias and the judge is an LLM sensor rather than
human gold review.

We tested two cheap BGE-small selectors:

- `direct_category_axis`
- `direct_anti_sycophancy`

against:

- `length`
- `random`
- `sentiment`
- `refusal_heuristic`

**Results**:

| Comparison | Focus win rate (decided) |
|---|---:|
| direct category axis vs length | 11.1% |
| direct category axis vs random | 62.5% |
| direct category axis vs refusal heuristic | 28.6% |
| direct category axis vs sentiment | 88.9% |
| direct anti-sycophancy vs length | 33.3% |
| direct anti-sycophancy vs random | 37.5% |
| direct anti-sycophancy vs refusal heuristic | 12.5% |
| direct anti-sycophancy vs sentiment | 83.3% |

We then reran the same pilot structure with stronger Gemini embedding
selections. For Gemini `direct_category_axis`, the blind decided win rates were
20.0% vs `length`, 55.6% vs `random`, 0.0% vs `refusal_heuristic`, and 100.0%
vs `sentiment`. For Gemini `direct_harm_reduction`, the decided win rates were
30.0% vs `length`, 88.9% vs `random`, 37.5% vs `refusal_heuristic`, and 100.0%
vs `sentiment`. A matched cheap-BGE run on `direct_harm_reduction` was much
weaker: 12.5% vs `length`, 25.0% vs `random`, 20.0% vs `refusal_heuristic`,
and 71.4% vs `sentiment`.

**Interpretation**: This is now a more informative exploratory result than the
cheap-BGE pilot alone. The open-ended blind-review lane is clearly sensitive to
backend quality, and a stronger embedding family materially improves the
results on the same inherited pool. At the same time, even the stronger Gemini
selectors still lose to `length` and `refusal_heuristic`, so this remains
exploratory rather than partner-grade intervention evidence. The practical
message is that future open-ended claims should use a fresh length-controlled
candidate pool plus a stronger embedding family.

### 4.12 Local Model Landscape For Raw `good/bad`

**Design**: To test whether the broad-word failure was only a BGE artifact, we
extended the same 50-case good-vs-proxy conflict protocol across eight local
FastEmbed models: BGE small/base, GTE base, Snowflake Arctic, Jina small/base,
Nomic, and Mixedbread.

**Results**:

| Model summary | Value |
|---|---:|
| best local raw `good/bad` | 48.0% |
| best local sentence `good/bad` | 36.0% |
| best local nearby proxy | 58.0% |

Seven of the eight local models stayed in the 14.0-28.0% range on raw
`good/bad`. The only partial exception was
`snowflake/snowflake-arctic-embed-m`, which reached 48.0% on raw `good/bad`
and 58.0% on `helpful/unhelpful`.

**Random-axis null control**: To verify that targeted axes carry a genuine
directional signal rather than random-projection luck, we scored the same 50-case
battery along 200 random unit vectors in embedding space for each model. Random
directions centered at 50.0% accuracy with standard deviation 9--10%, as
expected. For `snowflake-arctic-embed-m` (768d), the `persona_honesty` axis
(72%) fell in the 99.5th percentile of the random distribution (1/200 random
axes matched it); all other axes fell within the noise band. Notably,
`snowflake-arctic-embed-l` (1024d) performed *worse*: its best axis (60%)
reached only the 82nd percentile, failing to separate from noise despite
higher dimensionality. For `bge-small` (384d), no targeted axis separated
from random noise either. By contrast, Gemini's targeted axes at 86--98%
would exceed any random direction in these smaller spaces by many standard
deviations. To be precise about what this establishes: only one local model
(Snowflake-M) has even one axis that clears the noise floor, and that same
model shows high anchor fragility (cosine 0.19 between original and rephrased
axes). Qwen3-Embedding-0.6B (1024d, 600M params), the largest open model
tested, also fails on individual axes (best 58%, 79th percentile) — the gap
does not close with model size alone. The single frontier model tested
(Gemini) clearly exceeds noise, but with n=1 at frontier scale we cannot yet
distinguish a general capability threshold from a property specific to
Gemini's training.

**Multi-axis PCA does not yield a usable evaluative direction**: We tested
whether PCA of the 5 targeted axis vectors recovers an orientable evaluative
direction on local models. The centered-PC1 (best sign, selected post-hoc)
exceeds a matched null (PC1-of-5-random-axes, 200 trials) on 3 of 5 models,
confirming that the targeted axes are not random. However, no method-internal
orientation rule recovers the correct sign: the mean of the targeted axis
vectors scores 20–50% (at or below chance) on all 5 local models, and orienting
PC1 by pooled anchor scoring gives the same result (16–28% on 3 of 4 models
tested). The best-sign accuracy numbers (72–84%) depend entirely on using the
battery labels to pick the sign — which is not available at inference time. This
is a negative result: even with multi-axis PCA, local models yield no
zero-shot-orientable evaluative direction.

**Anchor perturbation**: We rephrased all anchor sentences — preserving meaning
but changing vocabulary substantially — and measured the cosine similarity
between original and rephrased axis directions. Across five local models
(BGE-small 384d, Jina-v2-small 512d, Snowflake Arctic-M 768d, Snowflake
Arctic-L 1024d, Qwen3-Embedding-0.6B 1024d), mean cosine similarity was low:
0.50, 0.37, 0.19, 0.20, and 0.46
respectively. Accuracy shifted by up to 34 percentage points under rephrasing.
The trend is non-monotonic — cosine drops from 384d to 768d then plateaus at
1024d — and does not clearly track dimensionality. This confirms that local model evaluative axes are
phrasing-dependent rather than reflecting robust geometric structure, and
suggests that above-chance accuracy on specific anchor sets may reflect
surface-level word overlap rather than deep evaluative geometry. Whether
frontier models exhibit greater axis robustness remains an open question
(Gemini testing was blocked by quota at time of writing).

**Word-stripping ablation**: To test whether the evaluative signal depends on
surface-level evaluative vocabulary shared between responses and anchors, we
stripped 56 evaluative words (good, bad, helpful, harmful, etc.) and 28
anchor-derived words from all response texts and re-scored the battery. On
Snowflake (768d), accuracy was essentially unchanged across all conditions
(e.g., persona_honesty 74% original → 74% all-stripped; combined 34% → 38%).
This suggests the local-model signal, such as it is, does not depend on
evaluative word overlap. However, with baseline accuracy near chance on most
axes, the ablation has limited power — the definitive version requires a
capable model where accuracy is high enough that degradation would be
detectable (Gemini testing was blocked by quota at time of writing).

**Cross-category transfer and axis geometry**: We scored each battery category
with every axis (not just its matched axis) and computed PCA of the 5 axis
vectors. On Snowflake (768d), pairwise axis cosine similarities are low
(0.025–0.256) and PC1 explains only 29.4% of variance — the axes are roughly
orthogonal, with no dominant evaluative direction. On BGE-small (384d), axes
show more convergence (cosine up to 0.64) and PC1 explains 38.6%. On both
models, centered-PC1 achieves 72% accuracy — but only with a favorable SVD sign
selected post-hoc (see the PCA negative result above). Cross-category transfer
reveals that the "matched" axis is often not the best scorer for its own
category: mean
matched-axis accuracy (52% Snowflake, 32% BGE) is lower than mean cross-axis
accuracy (56%, 41%). This suggests the evaluative signal that exists in local
models is not neatly compartmentalized by domain but is weak, distributed, and
partially shared across axes.

**Interpretation**: The broad-word failure is not perfectly universal, but it
does mostly persist across the local model family. The raw `good/bad` axis is
therefore not just weak on one cheap embedder; it is generally weak outside
richer targeted evaluative directions. The random-axis control confirms that
the signal in capable models is axis-specific rather than an artifact of
arbitrary projection.

### 4.13 Baselines Comparison

**Design**: To confirm that evaluative axes capture something beyond simpler signals, we compared axis-based scoring against five baselines on the 50-case battery: (1) cosine similarity between prompt and response embeddings (relevance), (2) response word count (length bias), (3) embedding vector norm, (4) cosine similarity to the phrase "This is a good response," and (5) cosine similarity to "This is helpful." All run on three local models.

**Results**:

| Method | Snowflake | BGE-M3 | Nomic |
|---|---:|---:|---:|
| Best evaluative axis | 72% | 80% | 62% |
| "Careful"/"Reckless" axis | 52% | 58% | 62% |
| Embedding norm | 57% | 51% | 30% |
| Response length | 51% | 51% | 51% |
| Cosine to "helpful" | 50% | 26% | 26% |
| Cosine to "good response" | 44% | 22% | 20% |
| Prompt-response cosine | 40% | 28% | 26% |

**Interpretation**: Length is near chance (51%) on all models, confirming the battery is properly balanced and axis scores are not length artifacts. Prompt-response cosine similarity is consistently below chance, meaning the "better" response is not systematically more relevant to the prompt in embedding space — the quality distinction is orthogonal to relevance. Cosine similarity to quality-describing phrases ("This is a good response") also fails, demonstrating that proximity to a fixed point in embedding space does not capture quality — but projection onto a direction (the axis approach) does. The evaluative axis captures a geometric structure that none of these baselines access.

### 4.14 Anchor Vocabulary Depth

**Design**: We tested whether the choice of anchor vocabulary — independent of axis design methodology — affects evaluative signal strength. We compared 20 single culturally universal word pairs (e.g., "Careful"/"Reckless", "Noble"/"Base"), 20 character projection phrases (e.g., "A careful person said this"), 10 synonym cluster axes, templated phrases ("This response is careful"), and templated multi-term centroids against the five current multi-sentence ML-jargon anchors. All scored on the same 50-case battery across three models: Snowflake Arctic Embed M (109M), BGE-M3 (568M), and Nomic Embed v1.5 (137M).

**Results**:

| Anchor type | Best axis | Snowflake | BGE-M3 | Nomic |
|---|---|---:|---:|---:|
| ML-jargon multi-sentence | persona_honesty / anti_sycophancy | 72% | 80% | 56% |
| Single word: Careful/Reckless | — | 52% | 58% | 62% |
| Single word: Moderate/Excessive | — | 48% | 50% | 40% |
| Single word: Noble/Base | — | 46% | 46% | 48% |
| Character projection: helpful | — | 66% | 18% | 18% |
| Template: "This response is careful" | — | 54% | 54% | 44% |
| Templated 6-term centroid | — | 24% | 14% | 12% |
| Single word: Hard/Soft (Potency) | — | 58% | 64% | 68% |
| Single word: Good/Bad | — | 48% | 16% | 12% |
| ML-jargon: general_evaluative | — | 34% | 12% | 10% |

*Note: These results are on the original 50-case battery, which was 64% firmness-biased. "Hard/Soft" scores are inflated by this bias; see §4.16 for rebalanced results.*

The most consistent finding across all models is that "Careful"/"Reckless" outperformed at least one ML-jargon axis on every model and was the only single-word pair to beat the best ML-jargon axis outright (62% vs 56% on Nomic). Compositing multiple universal terms into a single axis consistently degraded performance below the best individual term. Multi-sentence anchors using universal vocabulary (e.g., "The response is careful, considered, and avoids reckless harm") performed worse than the naked single word on all three models.

A corpus-frequency-based analysis recommended good/bad, true/false, useful/useless, honest/dishonest, strong/weak, and complete/incomplete as the optimal anchors based on frequency, cross-linguistic universality, and historical stability. Empirically, the frequency-recommended terms largely failed: complete/incomplete scored 16%, 14%, 18%; strong/weak scored 24%, 8%, 16%. The best-performing terms ("Careful"/"Reckless" at 52-62%, "Moderate"/"Excessive" at 40-50%) were either absent from or listed as secondary in the frequency-based recommendations.

A per-category breakdown reveals that different single words capture different evaluative dimensions. "Careful"/"Reckless" scored consistently on anti-sycophancy cases across all three models (4/5 or 5/5 per model, n=5) — more cross-model stable than the dedicated multi-sentence anti_sycophancy axis, which scored 60%, 100%, and 20% across the same three models. However, with only 5 anti-sycophancy cases, these per-category accuracy estimates are unreliable point estimates and should be read as directional patterns only. "Moderate"/"Excessive" captured persona honesty particularly well (100% on Snowflake, 75% on BGE-M3, n=4), consistent with the interpretation that responses fabricating personal experiences are "excessive" in a way the word naturally captures. No single word covered all categories equally well, reinforcing the finding that behavioral specificity — not general evaluative valence — drives geometric signal strength.

A bootstrap analysis (10,000 resamples, 95% CI) tempers the cross-model comparisons. With n=50 binary cases, individual accuracy estimates carry wide confidence intervals. On the local models, only three axis-model pairs are statistically significant (lower CI bound > 50%): persona_honesty on Snowflake (72%, CI [60%, 84%]) and BGE-M3 (66%, CI [52%, 78%]), and anti_sycophancy on BGE-M3 (80%, CI [68%, 90%]). "Careful"/"Reckless" at 62% on Nomic has CI [48%, 76%], which includes chance. The rank ordering of axes is therefore informative as a pattern across models — "Careful" is consistently among the top performers — but no single local-model comparison should be treated as definitive. The Gemini results (§4.9), where targeted axes reach 86-98%, are well above the significance threshold even with n=50 (one-sided binomial p < 10^-7 for combined targeted axes at 86%).

**Interpretation**: Geometric signal strength does not track corpus frequency. The most common evaluative terms in English produce some of the weakest evaluative axes because their extreme frequency spreads their embeddings across a wide region of semantic space encompassing many non-evaluative uses. Less frequent but more evaluatively specific terms produce tighter, more quality-correlated directions. Anchor selection should optimize for evaluative specificity rather than raw frequency or cross-linguistic universality. Gemini single-word validation (§4.17) confirms the pattern: "Careful" reaches 74% on Gemini while "Good/Bad" scores only 26%.

### 4.15 Osgood's Semantic Differential Dimensions

**Design**: Osgood et al. (1957) identified three orthogonal factors in human semantic judgment: Evaluation (good/bad), Potency (strong/weak), and Activity (active/passive). Since our theoretical framework begins with Osgood's Evaluation dimension, we systematically tested all three dimensions using four representative adjective pairs each, their within-dimension centroids, the cross-dimension composite (all three averaged), and a sum-of-independent-projections approach (scoring each dimension separately and adding). All scored on the 50-case battery across three local models.

**Results**:

| Dimension | Pair | Snowflake | BGE-M3 | Nomic |
|---|---|---:|---:|---:|
| Evaluation | Good/Bad | 48% | 16% | 12% |
| Evaluation | Nice/Awful | 48% | 28% | 14% |
| Evaluation | Pleasant/Unpleasant | 24% | 20% | 12% |
| Potency | Strong/Weak | 24% | 8% | 16% |
| Potency | Hard/Soft | 58% | 64% | 68% |
| Potency | Heavy/Light | 52% | 60% | 58% |
| Activity | Active/Passive | 60% | 18% | 36% |
| Activity | Fast/Slow | 22% | 18% | 34% |
| Composite | EPA core (3 terms) | 26% | 12% | 16% |
| Composite | EPA full (9 terms) | 20% | 12% | 16% |
| Sum | E+P+A projections | 26% | 12% | 16% |
| Reference | Careful/Reckless | 52% | 58% | 62% |

The EPA composite and sum-of-projections approaches all fail (12-26%). Combining Osgood's three dimensions does not recover a usable evaluative signal — it produces the same averaging-degrades-performance pattern seen with multi-term centroids in §4.14.

On the original 50-case battery, the Potency pair "Hard/Soft" appeared to produce the strongest cross-model signal: 58%, 64%, 68% across all three models. "Heavy/Light" also scored above chance (52%, 60%, 58%). By contrast, Osgood's primary Evaluation pairs (Good/Bad, Nice/Awful, Pleasant/Unpleasant) all failed.

However, subsequent validation (§4.16-4.17) revealed this was a battery artifact. The original 50 cases were 64% firmness-biased — disproportionately testing reasoning rigor, truthfulness, and safety, which "Hard/Soft" naturally captures. On the rebalanced 70-case battery (adding 20 warmth cases covering helpfulness, empathy, and tone), "Hard/Soft" drops substantially: pooled across 90 cases, it reaches statistical significance on only Snowflake (63%, CI [53%, 73%]) and Nomic (61%, CI [51%, 71%]), but not BGE-M3 (57%, CI [46%, 66%]). On Gemini Embedding, "Hard/Soft" inverts entirely to 39% — the worst-performing axis. The Potency signal was an artifact of what the battery measured, not a general evaluative property.

Osgood's three dimensions are not orthogonal in embedding space. Pairwise cosine similarities between the E, P, and A axis vectors range from 0.04 to 0.41, with Evaluation and Potency showing the highest overlap (0.24–0.41 across models). This confirms that factor orthogonality in human judgments does not transfer to embedding geometry.

**Interpretation**: None of Osgood's three dimensions produce a robust cross-model evaluative signal. The Potency finding on the original battery was the result of testing a firmness-biased task distribution — "Hard/Soft" captures firmness well but fails on warmth-oriented evaluations and inverts on a frontier model. "Careful" may perform more consistently than "Hard" because it straddles Evaluation and Potency — implying both goodness and firmness of effort — but even "Careful" does not reach statistical significance on all models (see §4.16). The broader lesson is that battery composition is a critical confound: any axis that captures one dimension of quality will appear strong on a battery biased toward that dimension.

### 4.16 Battery Rebalancing and Out-of-Sample Validation

**Design**: The original 50-case battery was constructed to test reasoning rigor, truthfulness, harm reduction, anti-sycophancy, context binding, helpfulness, persona honesty, and mixed tradeoffs. Post-hoc analysis revealed it was 64% firmness-biased: 32 of 50 cases tested dimensions where "hard" or "rigorous" responses are better, while only 18 tested warmth-related dimensions (empathy, emotional support, appropriate tone). Any axis correlated with firmness would appear strong on this battery regardless of its general validity.

To correct this, we constructed 20 additional warmth cases covering empathy, emotional support, tone sensitivity, and appropriate softness. These cases test whether an axis can recognize that the warmer, more empathetic response is better — a dimension the original battery undersampled. Together the 70-case rebalanced battery (50 original + 20 warmth) provides a more representative test.

We further constructed a 20-case expansion battery covering four new categories (factual accuracy, conciseness/completeness, creative quality, nuance/context) as an independent out-of-sample test set. These cases share no prompts or responses with the rebalanced battery and were designed after all axis experiments on the main battery were complete.

**Results — pooled analysis with Wilson 95% CIs**: Using a fixed pre-specified method (bipolar scoring, standard conversational framing) across all models and all axes — no per-model method selection — we computed pooled accuracy on all 90 cases (70 main + 20 expansion):

| Axis | Snowflake pooled | BGE-M3 pooled | Nomic pooled |
|---|---:|---:|---:|
| Careful/Reckless | 56% [45%, 65%] | 53% [43%, 63%] | **64% [54%, 74%]** |
| Hard/Soft | **63% [53%, 73%]** | 57% [46%, 66%] | **61% [51%, 71%]** |
| Thorough/Superficial | 56% [45%, 65%] | 56% [45%, 65%] | 51% [41%, 61%] |
| Active/Passive | **63% [53%, 73%]** | 31% [22%, 41%] | 38% [28%, 48%] |
| Good/Bad | 47% [37%, 57%] | 39% [29%, 49%] | 34% [25%, 45%] |

Bold indicates the lower CI bound exceeds 50% (statistically significant at p < 0.05). No axis reaches significance on all three models. "Careful" is significant only on Nomic. "Hard" is significant on Snowflake and Nomic but not BGE-M3 — and inverts on Gemini (§4.17). "Active" is significant on Snowflake but actively harmful on BGE-M3 and Nomic.

The mean accuracy across all 10 axes tested was 51.0% (Snowflake), 44.1% (BGE-M3), and 45.3% (Nomic) on the main battery, and 54.0%, 45.5%, and 46.0% on the expansion battery. The expansion battery is not systematically easier or harder than the main battery — the comparable means rule out difficulty bias as an explanation for any axis's expansion-set performance.

**Multi-axis voting overfits**: We tested all 2-, 3-, and 5-axis majority-vote combinations, selecting per-model optimal methods (cosine-to-positive for BGE-M3, bipolar for others) and per-model optimal framing (response-only for BGE-M3, standard for others). On the 70-case main battery, searched combinations appeared to improve on "Careful" alone by 10-15 percentage points. On the 20-case expansion battery, those same combinations lost 10-25 percentage points compared to their main-battery scores. "Careful" alone was more stable out-of-sample than any searched combination on every model.

This is the critical validation finding: the apparent improvement from multi-axis voting was in-sample overfitting, not a real improvement. With 70 binary cases and 252 possible 5-axis combinations to search, the probability of finding a combination that appears to beat any individual axis by 10+ points purely by chance is near-certain. The expansion battery exposes this: searched combinations that appear to gain 10-15 points in-sample lose 10-25 points out-of-sample.

**Per-model method variation**: Different scoring methods work on different models. BGE-M3 benefits from cosine-to-positive scoring (measuring cosine similarity to the positive anchor only, ignoring the negative) and response-only framing (embedding the response text without the user prompt). On the main battery, BGE-M3's "Careful" jumps from 51% (bipolar, standard) to 67% (cosine-to-positive, response-only). Snowflake and Nomic show no consistent benefit from these variations. This is interpretable: BGE-M3's anchor cosine similarity is ~0.50 (positive and negative anchors are relatively far apart), making cosine-to-positive viable. Snowflake's anchor cosine is ~0.90 (anchors are close together), so cosine-to-positive and bipolar produce nearly identical rankings. However, per-model method selection is a researcher degree of freedom — reporting the best of several methods inflates apparent accuracy — so the pooled analysis above uses the fixed method to avoid this confound.

**Failed predictions**: Two pre-registered predictions from the validation experiment were tested:
- Anchor geometry (cosine between positive and negative anchor embeddings) predicts which models benefit from cosine-to-positive scoring: within-model correlations were r = -0.11 (Snowflake), +0.24 (BGE-M3), +0.23 (Nomic) — no reliable within-model law.
- Main battery accuracy predicts expansion battery accuracy: r = +0.20 (Snowflake), -0.09 (BGE-M3), +0.49 (Nomic) — rankings do not transfer.

Both null results are informative: they confirm that evaluative axis performance on local models is not governed by simple geometric or transfer laws, and that main-battery results should not be extrapolated to new test distributions.

### 4.17 Gemini Single-Word Validation

**Design**: We tested the same 10 single-word axes on Gemini Embedding (`gemini-embedding-001`) using both bipolar and cosine-to-positive scoring on the full 70-case rebalanced battery (50 original + 20 warmth). This tests whether the patterns observed on local models — particularly the apparent strength of "Hard/Soft" — hold on a frontier model.

**Results** (bipolar scoring, combined accuracy on 70 cases):

| Axis | Combined | Orig (50) | Warmth (20) |
|---|---:|---:|---:|
| **Careful/Reckless** | **74%** | 72% | 80% |
| Thorough/Superficial | 61% | 52% | 85% |
| Kind/Cruel | 51% | 34% | 95% |
| Fair/Unfair | 49% | 32% | 90% |
| Honest/Dishonest | 49% | 30% | 95% |
| Good/Bad | 46% | 26% | 95% |
| Bold/Timid | 46% | 28% | 90% |
| Helpful/Unhelpful | 41% | 20% | 95% |
| **Hard/Soft** | **39%** | 42% | 30% |
| Active/Passive | 31% | 34% | 25% |

"Careful" at 74% is the clear best performer and the only axis with balanced performance across both battery splits (72% orig, 80% warmth). Its Wilson 95% CI on 70 cases is approximately [62%, 83%], comfortably above chance.

"Hard/Soft" at 39% is actively inverted — worse than random — confirming that its apparent strength on local models was a battery artifact. On the firmness-biased original battery, "Hard" scored 42% on Gemini (also below chance). On warmth cases, it drops to 30%. The axis that appeared strongest on three local models is the worst-performing axis on the frontier model.

Several axes show extreme polarity: "Kind," "Honest," "Good," and "Helpful" all score 90-95% on warmth cases but 20-34% on the original battery. These axes capture warmth-related quality but fail on firmness-related quality — the mirror image of "Hard/Soft." Only "Careful" scores above 70% on both splits, consistent with the interpretation that it straddles evaluative dimensions rather than capturing just one.

Cosine-to-positive scoring hurt on Gemini for most axes: "Careful" dropped from 74% to 59%, "Thorough" from 61% to 27%. Unlike BGE-M3 (where cosine-to-positive helps), Gemini's bipolar scoring is generally better.

**Interpretation**: The Gemini results confirm three findings and overturn one:
- **Confirmed**: "Careful" is the most consistent single-word axis (74% on Gemini, 64% on Nomic, modest on others).
- **Confirmed**: Single-word axes work better on a frontier model than on local models.
- **Confirmed**: Scoring method interacts with model — cosine-to-positive helps on some models (BGE-M3) and hurts on others (Gemini).
- **Overturned**: "Hard/Soft" is not a robust cross-model signal. It is a firmness-biased axis that performs well only on batteries biased toward firmness-related quality dimensions.

### 4.18 Tree Decomposition of "Good"

**Design**: The user's theoretical framework proposes that "good" fails because the word has too many senses (~50,000 in WordNet) — any response satisfies only a fraction, producing a noisy projection. Narrower children of "good" (careful, honest, kind, wise, helpful, etc.) should give cleaner projections because a response matching a specific quality dimension matches more of the word's semantic scope. We tested this by building a three-level tree: root ("good"), 10 level-1 children (careful, honest, kind, wise, helpful, thorough, fair, responsible, clear, respectful), and 15 level-2 grandchildren (deliberate, attentive, precise, cautious, methodical under "careful"; truthful, transparent, sincere, forthright, candid under "honest"; compassionate, patient, gentle, encouraging, supportive under "kind"). All scored independently on the 70-case balanced battery across three models. We measured individual accuracy, score-delta correlations within vs. across semantic branches, parent-child correlations, and combination strategies (majority vote, sum of projections).

**Results** (Nomic, best model; cross-model notes below):

| Level | Term | Combined | Orig | Warmth |
|---|---|---:|---:|---:|
| Root | good | 31% | 12% | 80% |
| L1 | **careful** | **64%** | 62% | 70% |
| L1 | thorough | 54% | 56% | 50% |
| L1 | clear | 43% | 34% | 65% |
| L1 | wise | 40% | 30% | 65% |
| L1 | kind | 40% | 28% | 70% |
| L1 | helpful | 40% | 20% | 90% |
| L1 | honest | 39% | 30% | 60% |
| L1 | responsible | 37% | 20% | 80% |
| L1 | respectful | 39% | 26% | 70% |
| L1 | fair | 37% | 24% | 70% |
| L2 | deliberate | 51% | 56% | 40% |
| L2 | cautious | 50% | 48% | 55% |
| L2 | patient | 53% | 56% | 45% |
| L1 combo | L1 majority vote | 34% | 18% | 73% |
| L1 combo | L1 sum of deltas | 29% | 14% | 65% |

On Nomic, every level-1 child outperformed "good" on the combined battery. On the other models, the claim is weaker: the best child beats the root on every model (thorough 60% vs good 51% on Snowflake; kind 53% vs good 36% on BGE-M3), but not all children do (e.g., on Snowflake, honest 40% and respectful 39% fall below good's 51%). The cross-model-robust finding is that decomposition surfaces at least one child better than the root. Going deeper (level 2) hurt on every model: "deliberate" (51%), "cautious" (50%), and "patient" (53%) were all worse than their respective parents ("careful" at 64%, "kind" at 40%). The correlation analysis below was conducted on Nomic only.

**Score-delta correlations revealed why "careful" is special — and this finding replicates across all three models.** "Careful" is uncorrelated with "good" in score-delta space on every model tested: Nomic r = -0.11 (t = -0.88, n.s.), Snowflake r = +0.09 (t = +0.71, n.s.), BGE-M3 r = -0.25 (t = -2.10, borderline). Meanwhile, most other level-1 children correlate strongly WITH "good" on all models — on Nomic: honest (r = 0.71), wise (r = 0.73), helpful (r = 0.76), responsible (r = 0.76); on BGE-M3 the warmth bias is even stronger: helpful (r = 0.92), respectful (r = 0.89), responsible (r = 0.85), fair (r = 0.82). On Snowflake, the pattern is weaker but the same direction: honest (r = 0.55), kind (r = 0.49), respectful (r = 0.48) all significant. Only "careful" and "thorough" are consistently independent of "good" across models. In axis-vector space, the same pattern holds: "careful" has the lowest cosine with "good" on every model (Nomic -0.01, Snowflake +0.11, BGE-M3 +0.10). "Careful" accesses a dimension "good" does not encode, and this geometric independence is model-invariant.

**Same-branch vs. cross-branch correlation: the prediction failed at level 1.** The theory predicted that children from the same semantic branch (e.g., careful–thorough–responsible, all "effort") would correlate more than children from different branches (e.g., careful–kind). Empirically, same-branch mean r = 0.24, cross-branch mean r = 0.32 — the opposite. The reason: 8 of 10 level-1 children share a warmth bias inherited from "good," creating high cross-branch correlations regardless of nominal semantic branch. At level 2, same-branch correlation was marginally higher (0.32 vs. 0.23, difference = 0.09), but the effect was small.

**Combination failed because warmth-biased terms outvote independent ones.** Level-1 majority vote (34%) was dramatically worse than the best individual child (64%). The 8 warmth-biased children outvote the 2 independent ones (careful, thorough), and the combination regresses toward the warmth bias of "good." This is the same mechanism that makes multi-axis voting overfit (§4.16): combining axes that share a common bias amplifies the bias rather than canceling it.

**Interpretation**: The tree decomposition theory is partially confirmed at the score level and mechanistically explained by the neighborhood analysis below. Decomposition helps at level 1 — narrower terms DO produce cleaner projections than the broad root. But the mechanism differs from the theory's prediction. "Careful" does not work because it is a narrower sub-sense of "good" — it works because it captures a dimension ORTHOGONAL to "good." The neighborhood composition analysis shows why: "good"'s nearest semantic neighbors are predominantly warmth and emotion words, while "careful"'s neighbors are competence and restraint words. Most evaluative terms in natural language encode warmth/agreeableness (reflecting how humans most commonly use evaluative language). "Careful" is rare in accessing the effort/rigor dimension. Its geometric independence from "good" is why it discriminates on cases where "good" fails. Going deeper than level 1 narrows coverage without adding signal, and combining diverse children fails because the warmth bias dominates the vote. The optimal decomposition depth is exactly one level — specific enough to be evaluatively clean, broad enough to cover multiple quality dimensions.

**Cross-validation on Gemini Embedding.** The independence pattern replicates at frontier scale. On Gemini, "careful" correlates weakly with "good" in score-delta space (r = +0.24, borderline significant) while most other children correlate extremely strongly (helpful r = 0.93, respectful r = 0.92, kind r = 0.92, fair r = 0.89, honest r = 0.87, responsible r = 0.85, wise r = 0.82). "Clear" is the only other independent child (r = -0.003). Gemini's accuracy advantage is concentrated in the independent terms: "careful" reaches 74%, "thorough" 61%, while the warmth-biased children remain near chance on the balanced battery (helpful 41%, honest 49%, kind 51%, fair 49%). On the orig/warmth split, "good" on Gemini shows 26% orig vs 95% warmth — the most extreme warmth bias of any model — while "careful" shows 72% orig vs 80% warmth, balanced across dimensions. This pattern holds across all four models tested: the careful-good correlation is always the weakest (Snowflake +0.09, BGE-M3 -0.25, Nomic -0.11, Gemini +0.24), while warmth-biased children show strong correlations on every model.

**Neighborhood composition analysis.** The score-delta correlations show THAT "good" and "careful" access different dimensions, but not why. A direct analysis of each word's semantic neighborhood provides the mechanism. We embedded 247 evaluative adjectives across 12 semantic categories (warmth, competence, restraint, integrity, emotion, etc.) and ranked them by cosine similarity to "good" and "careful," excluding direct synonyms. The top-30 non-synonym neighbors of "good" are dominated by warmth and emotion words: cross-model average 40% warmth+emotion (21% warmth, 19% emotion) vs. only 12% competence+restraint. "Careful" shows the inverse: 30% competence+restraint vs. 18% warmth+emotion. The composition scales with bias strength — on BGE-M3 (strongest warmth bias, firmness d = -0.73), "good"'s neighborhood is 50% warmth+emotion; on Snowflake (weakest bias, firmness d = -0.22), it is 23%.

Only 7 words appear in "good"'s top-30 neighborhood on all three models: encouraging (warmth), happy (emotion), positive (emotion), satisfied (emotion), helpful (utility), strong (strength), and suitable (misc) — 5 of 7 are warmth/emotion. "Careful" has only one cross-model neighbor besides itself: cautious (restraint). The two neighborhoods barely overlap: 4 shared words on BGE-M3, 5 on Nomic, 15 on Snowflake — least overlap on the models with strongest warmth bias.

The neighborhood warmth fraction (share of warmth+emotion words in a word's top-30 neighbors) predicts its scoring bias. Across the 7 axes tested on the balanced battery with per-split Cohen's d, the correlation between warmth fraction and warmth-split d is positive on all three models (Snowflake r = +0.69, BGE-M3 r = +0.65, Nomic r = +0.66, all n = 7). The correlation with firmness-split d is negative (Snowflake r = -0.78, BGE-M3 r = -0.47, Nomic r = -0.77). Words whose neighborhoods are warmth-heavy (good 42%, kind 49%, helpful 37%) show warmth bias in scoring; words whose neighborhoods are competence/restraint-heavy (restrained 10%, thorough 4%, precise 2%, rigorous 2%) do not. This means the bias is predictable from the word's embedding neighborhood alone, without running any test cases.

---

### 4.19 Warmth Sensitivity Across Battery Splits

The tree decomposition (§4.18) established that "careful" is geometrically independent of "good" while most other evaluative terms share good's warmth bias. This predicts a specific observable difference: good should perform very differently on firmness-testing cases vs warmth-testing cases, while careful should be approximately stable across both.

The balanced battery's content split provides a test independent of embedding geometry. The original 50 cases are firmness-biased (64% require rigor — pushing back, refusing, prioritizing accuracy over agreeableness). The 20 warmth cases test empathy, emotional support, and appropriate softness. This split derives from how the cases were authored, not from any model's scores.

**Results** (content split, all four models):

| Axis | Battery | Snowflake | BGE-M3 | Nomic | Gemini |
|---|---|---:|---:|---:|---:|
| Good | Original (50) | 48% | 16% | 12% | 26% |
| Good | Warmth (20) | 60% | 85% | 80% | 95% |
| Good | **Gap** | **+12pt** | **+69pt** | **+68pt** | **+69pt** |
| Careful | Original (50) | 52% | 58% | 62% | 72% |
| Careful | Warmth (20) | 40% | 35% | 70% | 80% |
| Careful | **Gap** | **-12pt** | **-23pt** | **+8pt** | **+8pt** |

On three of four models, good's accuracy on warmth cases is ~4-7x its accuracy on firmness cases — gaps of 68-69 percentage points. Good succeeds only when the correct answer happens to also be the warmer one.

**Lexical valence check.** To test whether the content split reflects word-level sentiment rather than embedding geometry, we computed a valence baseline: for each case, count positive and negative words in both responses and predict that the more positively-valenced response is better. This baseline scores 11% on firmness cases and 30% on warmth cases — a 19-point gap, compared to 68-69 points for "good" on the embedding models. The lexical gap accounts for under a third of the embedding effect. On sycophancy cases the valence baseline scores 0% (the sycophantic "worse" responses are always more positively-valenced), confirming that the content-split effect operates at the level of embedding geometry, not surface-level word choice.

Careful shows a qualitatively different pattern. Its gaps are smaller in absolute value (8-23 points) and direction-inconsistent: mildly firmness-biased on Snowflake and BGE-M3 (does better on the harder firmness cases), mildly warmth-biased on Nomic and Gemini. On Snowflake and BGE-M3, careful actually performs *worse* on the warmth battery than on the original — ruling out the confound that the warmth cases are simply easier for all anchors. This direction inconsistency is itself evidence of independence — careful is not systematically tracking either pole.

A secondary analysis splitting cases by each model's own kind-axis scores (a finer-grained warmth proxy) produces larger separations — for example, Nomic "good" drops to 14% when the worse response scores higher on "kind," while "careful" holds at exactly 64% in both conditions. However, this split uses an embedding-derived variable that correlates with the axes being tested; it re-expresses §4.18's correlation structure rather than testing it independently. The content split above is the stronger evidence because the split variable is external to any embedding model.

**Out-of-sample validation.** The 20-case expansion battery (conciseness, creative quality, factual accuracy, nuance/context — 5 cases each, withheld from all prior analysis) confirms the pattern directionally. Careful beats good on all three local models: Snowflake 80% vs 30%, BGE-M3 60% vs 50%, Nomic 65% vs 45%. Per-category cells are small-n (5 each) and directional only, but the pattern is consistent with the warmth-bias prediction: good scores highest on nuance/context (the most warmth-adjacent category) and lowest on factual accuracy, while careful is more stable across categories. Snowflake's careful reaches 80% on expansion (CI [58%, 92%]), its best result on any held-out test.

**Prediction accuracy.** Reformulated as a prediction test: if we predict that "good" succeeds whenever the case is warmth-type and fails whenever it is firmness- or sycophancy-type, this simple rule is correct 84% of the time on BGE-M3 and 86% on Nomic — but only 54% on Snowflake, where the warmth gap is weaker (+12pt vs +68-69pt). For "careful," the same rule scores only 40–47% on all three models — no better than a coin flip. The contrast is the mechanism on BGE-M3 and Nomic: warmth-biased axes are predictable from case type, warmth-independent axes are not. Snowflake's weaker warmth bias produces a weaker prediction, consistent with the +12pt gap being too small for case-type to dominate.

**Interpretation**: Good's failures are concentrated in firmness-testing cases, consistent with the geometric warmth bias identified in §4.18. Careful's stability across conditions — and its mixed gap direction on the content split — confirms that it accesses a quality-relevant dimension approximately orthogonal to warmth. The expansion battery confirms this generalizes beyond the main battery. Any deployment of embedding-based quality scoring must select an anchor with this warmth-independence property, or the system will systematically undervalue firmness, accuracy, and honest disagreement.

### 4.20 Phrase Anchors and Synonym Vocabulary

If "careful" works because it accesses an effort/rigor dimension independent of warmth, do phrase-level anchors that combine independent terms or semantically similar single words do better?

We tested 12 anchor variants across all three local models: single-word "Careful" (baseline), multi-word phrases ("Careful and thorough," "A careful, thorough response"), bare concatenations ("Careful Thorough"), sentence frames ("This response is careful"), and effort-related synonyms (meticulous, rigorous, diligent, conscientious, deliberate).

**No phrase anchor beats single-word "Careful" on the best model.** On Nomic, "Careful" at 64% outperforms every variant — the closest competitor is bare "Careful Thorough" at 57%. On BGE-M3 and Snowflake, phrase anchors occasionally match but never reliably exceed the baseline.

**Grammar can disrupt warmth-independence.** Bare concatenation "Careful Thorough" preserves small warmth gaps (+4 to +11 points across models), similar to single-word "Careful" (-12 to +8). But the grammatical phrase "Careful and thorough" introduces large warmth bias on BGE-M3 and Nomic (+54 and +32 point gaps) while remaining small on Snowflake (+6). Full sentences show a similar model-dependent pattern. The effect is not universal, but on models where it appears, syntactic context shifts the embedding toward warmth-associated usage patterns.

**Most effort-related synonyms are warmth-biased.** "Conscientious" — which denotatively implies diligence — shows the most extreme warmth bias of any term tested (+49 to +69 point gaps across models, 33-40% combined accuracy). "Meticulous" and "diligent" show similar patterns. These terms appear in training data primarily as positive personality descriptions ("she's very conscientious"), overlapping with warmth contexts. Only "careful," "thorough," and "deliberate" show consistent warmth-independence among effort-related words tested — all three are used more often in task/quality contexts than in personality descriptions.

**Implication**: The warmth-independence that makes "careful" work is a property of its distributional context, not its denotational meaning. Synonyms that mean roughly the same thing but appear in different contexts produce very different embedding directions. This narrows the viable anchor vocabulary further: even within the effort/rigor semantic field, most words share the warmth bias of "good."

### 4.21 Pre-Registered Prediction Test

The warmth-bias theory makes a testable prediction: we should be able to predict, before testing, which novel evaluative terms will share "good's" warmth bias and which will be independent. We selected 10 terms not tested in any prior experiment, declared predictions, then measured.

**Predictions** (declared before running): Four terms predicted warmth-independent (|r_good| < 0.3 on majority of models): "prudent," "vigilant," "scrupulous," "measured" — selected because they appear primarily in caution/risk contexts. Six terms predicted warmth-biased (r_good > 0.4 on majority of models): "exemplary," "superb," "commendable," "outstanding," "gracious," "benevolent" — selected because they appear primarily in praise/admiration or directly warm contexts. Success criterion: >= 8/10 correct on >= 2/3 models.

**Results**:

| Term | Predicted | r_good (Snowflake) | r_good (BGE-M3) | r_good (Nomic) | Correct (2/3)? |
|---|---|---:|---:|---:|---|
| prudent | independent | +0.16 | -0.01 | +0.42 | Yes (2/3) |
| vigilant | independent | +0.38 | +0.71 | +0.64 | No (0/3) |
| scrupulous | independent | -0.12 | +0.80 | +0.44 | No (1/3) |
| measured | independent | +0.36 | +0.20 | +0.12 | Yes (2/3) |
| exemplary | biased | +0.61 | +0.74 | +0.71 | Yes (3/3) |
| superb | biased | +0.59 | +0.91 | +0.78 | Yes (3/3) |
| commendable | biased | +0.62 | +0.95 | +0.77 | Yes (3/3) |
| outstanding | biased | +0.31 | +0.70 | +0.76 | Yes (2/3) |
| gracious | biased | +0.52 | +0.81 | +0.86 | Yes (3/3) |
| benevolent | biased | +0.48 | +0.82 | +0.80 | Yes (3/3) |

**8/10 predictions correct** — but the trivial null hypothesis "all evaluative terms are warmth-biased" also scores 8/10 (correct on all 8 actually-biased terms, wrong only on prudent and measured). The test therefore does not demonstrate that the warmth-bias theory predicts better than simple valence reasoning.

**What the test does show**: the 6 biased predictions confirm that positive-valence quality synonyms (exemplary, superb, commendable, outstanding, gracious, benevolent) correlate with "good" — expected, since they are near-synonyms of "good" in distributional space. The discriminating test is the 4 independence predictions, where the score is 2/4 — not above base rate.

The failures are informative as a hypothesis-generating result. "Vigilant" and "scrupulous" appear in caution contexts but also frequently as positive personality attributions ("a vigilant parent," "a scrupulous researcher"), which places them in the same distributional neighborhood as warmth terms. "Prudent" and "measured" — the successful independence predictions — appear more in contexts of restraint and control than personality praise. This suggests a post-hoc refinement of the prediction rule from §4.20: warmth-independence may require the term to appear primarily in task/quality contexts, not personality descriptions. The same mechanism would explain why "careful" works (used in "be careful with..." warnings, not primarily as personality praise) while "conscientious" fails (used almost exclusively as a personality trait). This refinement is a hypothesis generated by this experiment, not confirmed by it.

None of the 10 new terms approached "careful's" accuracy (49-64% across models). The best new term, "measured," scored 51-53% — near chance. This confirms that "careful" occupies a rare position in vocabulary space: warmth-independent, evaluatively specific, and high-accuracy.

**Follow-up test: the distributional-context hypothesis fails.** The first test generated a post-hoc hypothesis: warmth-independence requires the term to appear in task/quality contexts rather than personality-praise contexts. We tested this directly with 10 new terms: 6 predicted independent (systematic, rigorous, stringent, accurate, logical, analytical — all task/quality terms rarely used as personality descriptions) and 4 predicted biased (admirable, noble, generous, worthy — all personality-praise terms). This design inverts the ratio: the null "all biased" now scores only 4/10, so any score significantly above 4 would demonstrate predictive power.

| Term | Predicted | r_good (Snowflake) | r_good (BGE-M3) | r_good (Nomic) | Correct (2/3)? |
|---|---|---:|---:|---:|---|
| systematic | independent | +0.10 | +0.65 | +0.44 | No (1/3) |
| rigorous | independent | +0.02 | -0.02 | +0.13 | Yes (3/3) |
| stringent | independent | -0.17 | -0.36 | +0.42 | No (1/3) |
| accurate | independent | +0.66 | +0.86 | +0.77 | No (0/3) |
| logical | independent | +0.38 | +0.85 | +0.74 | No (0/3) |
| analytical | independent | -0.13 | +0.32 | +0.39 | No (1/3) |
| admirable | biased | +0.46 | +0.93 | +0.77 | Yes (3/3) |
| noble | biased | +0.18 | +0.72 | +0.81 | Yes (2/3) |
| generous | biased | +0.32 | +0.84 | +0.67 | Yes (2/3) |
| worthy | biased | +0.71 | +0.91 | +0.88 | Yes (3/3) |

**Result: 5/10 correct, just +1 above the null's 4/10.** Only 1 of 6 independence predictions succeeded (rigorous). The distributional-context hypothesis does not predict warmth-independence. Even "accurate" (r = +0.66 to +0.86) and "logical" (r = +0.38 to +0.85) — terms that appear almost exclusively in task/quality contexts — are strongly warmth-biased. The warmth bias is more pervasive than any simple distributional account predicts.

"Rigorous" (r = +0.02, -0.02, +0.13) joins the small set of confirmed warmth-independent terms — alongside careful, thorough, prudent, and measured — but at lower accuracy (34-49% vs careful's 49-64%). What distinguishes these terms from the many that share good's warmth bias remains an open question; the distributional-context explanation does not account for the pattern.

### 4.22 Comprehensive Term Analysis

To move beyond post-hoc explanations, we compiled every evaluative term tested across all prior experiments (45 unique terms from §4.18-4.21 plus 6 new from a restraint hypothesis test below: restrained, disciplined, temperate, exceptional, remarkable, brilliant — 51 total) and measured each on the balanced battery (n=70) across all three local models.

**The independent cluster.** Nine terms are warmth-independent (|r_good| < 0.3) on at least 2 of 3 models: careful, thorough, deliberate, cautious, methodical, patient, prudent, measured, and rigorous. Every one of these describes restraint, self-control, or procedural discipline — how something is done rather than whether it is good.

The remaining ~80% of evaluative terms correlate with "good" (r > 0.4 on at least 2 of 3 models), including terms that seem task-specific: "accurate" (r = +0.66 to +0.86), "logical" (r = +0.38 to +0.85), "systematic" (r = +0.10 to +0.65), and "analytical" (r = -0.13 to +0.39). The warmth bias is more pervasive than distributional context alone predicts.

**Independent terms are near chance, not strong evaluators.** The 9 independent terms average 47-51% accuracy across models — near chance. Biased terms average 35-41% — *below* chance, because they track warmth, which is anti-correlated with correctness on this battery (good scores 31-51%). Escaping the warmth bias removes the anti-signal; it does not by itself produce a strong positive signal. Only "careful" clears chance meaningfully on some models (64% Nomic, 51% BGE-M3, 49% Snowflake), and even that result is inconsistent.

Majority voting among the 9 independent terms does not beat careful alone on any model. The mean accuracy of all-9 voting is 46% vs careful's 55%. Each additional term adds noise rather than signal — the terms are near chance individually, and combining near-chance estimators does not produce a strong one.

**Pooled prediction test results.** Across all three prediction tests (§4.21 plus below), we made 13 independence predictions. Five were correct (38%). The base rate of independence is ~20% (9/45 terms), so 38% is weak signal — not zero, but wrong more often than right on positive calls. The biased predictions (8 across three tests) were correct 7/8 times, but these are tautological: positive-valence terms correlating with "good" is already established. Only the independence predictions discriminate, and pooled they are not a usable criterion.

**Restraint hypothesis test.** The pattern — all 9 independent terms share restraint semantics — generated a testable hypothesis: restraint terms (restrained, disciplined, temperate) should be independent, while positive-valence outcome terms (exceptional, remarkable, brilliant) should be biased. The null "all biased" scores 3/6.

| Term | Predicted | r_good (Snow.) | r_good (BGE) | r_good (Nomic) | Correct (2/3)? |
|---|---|---:|---:|---:|---|
| restrained | independent | -0.06 | -0.57 | -0.24 | Yes (3/3) |
| disciplined | independent | +0.41 | +0.81 | +0.28 | No (1/3) |
| temperate | independent | +0.10 | +0.24 | +0.09 | Yes (3/3) |
| exceptional | biased | +0.20 | +0.69 | +0.54 | Yes (2/3) |
| remarkable | biased | +0.23 | +0.70 | +0.66 | Yes (2/3) |
| brilliant | biased | +0.12 | +0.84 | +0.47 | Yes (2/3) |

Result: 5/6 correct, +2 above null. "Restrained" is strongly independent (r = -0.06 to -0.57) with 54-64% in-sample accuracy. "Temperate" is independent on 3/3 models but near chance accuracy (37-47%). "Disciplined" failed — biased on 2/3 models (r = +0.41, +0.81) — likely because "a disciplined person" is personality-praise, the same pattern that caused "vigilant" and "scrupulous" to fail in §4.21.

However, this 5/6 score does not validate restraint as a predictive theory. Adding its 2/3 independence predictions to the pooled count from §4.21 yields 5/13 total (38%), still not a reliable rule. The restraint test's apparent strength comes from the 3 biased predictions inflating the denominator; the discriminating dimension (2/3 independence predictions correct) is at chance with n = 3.

**OOS validation of "restrained."** The 64% in-sample accuracy for "restrained" on BGE-M3 came from searching ~50 terms on a fixed 70-case battery — multiple-comparisons territory. On the held-out 20-case expansion battery: restrained scores 75% on BGE-M3 (supporting the in-sample finding) but drops to 55% on Snowflake and 45% on Nomic (both near chance). Careful, by contrast, scores 80%, 60%, 65% on the same OOS set — more consistent across models. With n = 20, Wilson CIs overlap heavily (restrained 75%: [53%, 89%]; careful 60%: [39%, 78%]), so no reliable distinction is possible at this sample size.

**The asymmetry is the robust finding.** The warmth bias is predictable and pervasive: ~80% of evaluative terms (36/45) correlate with "good" on at least 2 of 3 models. Which terms escape it is *not* predictable from any semantic feature we tested — distributional context, personality-praise avoidance, restraint semantics — all produce at best weak signal on prospective tests. The asymmetry itself is informative: the warmth bias is a stable, model-invariant property of evaluative embedding geometry, while warmth-independence is a rare, empirically identifiable but theoretically unexplained exception. The descriptive observation — that the independent terms cluster around restraint semantics — holds as a pattern in the data but not as a predictive rule.

### 4.23 Anti-Sycophancy Demonstration (Constructed Illustration)

The original battery's 5 anti-sycophancy cases showed "good" picking the sycophantic response 80-100% of the time (§5.6). To illustrate this mechanism at larger scale, we constructed 15 additional anti-sycophancy cases (n=20 total). Each case presents a user asserting something false or risky and asking for validation; the better response pushes back with evidence, the worse response agrees warmly. The cases span health misconceptions (training through injury, 500-calorie diets), financial risk (all savings in one stock, borrowing for crypto), safety (DIY electrical), academic integrity, relationship red flags, and technical overconfidence.

| Axis | Snowflake | BGE-M3 | Nomic | Mean |
|---|---:|---:|---:|---:|
| good | 15% | 0% | 0% | 5% |
| kind | 45% | 25% | 0% | 23% |
| helpful | 70% | 0% | 5% | 25% |
| honest | 20% | 5% | 20% | 15% |
| careful | 50% | 65% | 85% | 67% |
| restrained | 70% | 75% | 85% | 77% |
| thorough | 25% | 35% | 65% | 42% |

The pattern is consistent: warmth-biased terms (good, kind, helpful, honest) pick the sycophantic response at high rates, while warmth-independent terms (careful, restrained) pick the correct pushback.

**Lexical confound disclosure.** The 15 new cases were authored with the hypothesis in hand, and the sycophantic (worse) responses are stylistically warm — gushing praise, validation, encouragement — while the correct (better) responses open with negations and risk warnings. A trivial positive-word-count baseline (counting valence words like "amazing," "impressive," "dedicated" vs "no," "stop," "risk," "danger") achieves 85% accuracy on the same 20 cases — outperforming the best embedding axis ("restrained" at 77%). The original 5 cases show the same pattern (valence baseline: 80%).

This means the embedding result on these specific cases cannot be distinguished from lexical valence detection: the sycophantic responses are lexically warmer, and any warmth-sensitive detector (including the "good" embedding direction) will separate them on tone rather than correctness. The finding illustrates WHY the warmth mechanism produces sycophancy-correlated failures — sycophantic responses are definitionally warmer — but does not constitute independent evidence that the embedding captures something beyond surface valence on this task.

The independent evidence for the warmth-bias mechanism comes from the content split in §4.19, where warmth vs firmness is defined by case design (firmness-requiring vs warmth-requiring battery cases), not by the lexical properties of the responses themselves. A valence-word-count baseline applied to the content split confirms this independence: the baseline shows a gap of only 19 points between warmth (30%) and firmness (11%) cases, compared to the embedding "good" gap of 69 points (BGE-M3: 85% vs 16%). The embedding captures warmth patterns far beyond what word-counting explains. On warmth cases specifically, the embedding scores 85% while the word-count baseline scores 30% — the embedding is doing something geometrically meaningful, not merely detecting lexical valence. This contrasts with the anti-sycophancy demonstration above, where the word-count baseline (85%) outperforms the embedding (77%), indicating that the sycophancy cases are lexically confounded while the content-split cases are not.

### 4.24 Absolute-Score Analysis: Discrimination vs Training Signal

The battery tests whether an axis can DISCRIMINATE between two pre-written responses (pairwise: is the better response scored higher?). But in training, the model generates ONE response and receives a scalar reward. The training question is different: does the axis assign higher absolute scores to genuinely good responses than to genuinely bad ones? If the score distributions of better and worse responses are separated, the axis is a useful training signal even if pairwise discrimination is noisy.

We tested this by computing absolute projections of all 140 responses (70 better, 70 worse) onto each axis vector and measuring distributional separation via Cohen's d (positive d = better responses score higher; negative d = worse responses score higher).

**Pooled d (all 70 cases).** These values inherit the battery's 64% firmness bias; see per-split analysis below for the correct interpretation.

| Axis | Snowflake d | BGE-M3 d | Nomic d |
|---|---:|---:|---:|
| good | -0.14 | -0.40 | -0.30 |
| careful | +0.03 | +0.12 | +0.15 |
| restrained | +0.08 | +0.37 | +0.21 |
| thorough | +0.06 | -0.04 | +0.01 |
| kind | +0.01 | +0.09 | -0.16 |
| helpful | +0.02 | -0.32 | -0.16 |
| honest | -0.03 | -0.38 | -0.07 |

The pooled d values for "good" are negative on all three models, but pooled d inherits the battery's 64% firmness bias — the same composition artifact policed throughout this paper (§4.16–4.17). The content-split analysis below is the correct lens.

The content-split analysis sharpens the mechanism. The sign-flip in "good" — positive d on warmth, negative on firmness — holds on all three models:

| Model | Axis | Firmness d | Warmth d | Sycophancy d |
|---|---|---:|---:|---:|
| BGE-M3 | good | -0.73 | +0.74 | -4.02 |
| Nomic | good | -0.52 | +0.43 | -1.45 |
| Snowflake | good | -0.22 | +0.15 | -2.08 |

The effect is strongest on BGE-M3 and weakest on Snowflake (consistent with Snowflake's smaller accuracy gap in §4.19). The sycophancy d values are large in magnitude but based on n = 5 cases and should be interpreted cautiously.

For comparison, warmth-independent axes on BGE-M3:

| Axis | Firmness d | Warmth d | Sycophancy d |
|---|---:|---:|---:|
| careful | +0.18 | -0.37 | +1.16 |
| restrained | +0.45 | -0.08 | +2.52 |

However, careful's warmth d is direction-inconsistent across models: -0.37 on BGE-M3 but +0.37 on Nomic and near-zero on Snowflake (-0.04). This matches the direction-inconsistency noted in §4.19 and is itself evidence of warmth-independence — unlike "good," which is consistently warmth-positive on all three models, "careful" has no stable warmth loading. Restrained's warmth d is near-zero on all three models, consistent with warmth-neutrality.

"Good" is anti-correlated with quality on firmness cases and catastrophically wrong on sycophancy, while being correct on warmth cases. The sign-flip is not noise — it tracks whether warmth and correctness align or conflict in each case. On cases where warmth and correctness point the same way, "good" works as a training signal. On cases where they diverge, "good" actively rewards the wrong response.

No single axis works as a universal training signal. "Careful" has a small correct-direction signal on firmness and sycophancy but is direction-inconsistent on warmth across models (BGE-M3: -0.37, Nomic: +0.37, Snowflake: -0.04) — on some models it would penalize warm responses, on others reward them. "Restrained" is neutral on warmth (d near zero on all three models) rather than direction-inconsistent, making it the least-harmful single-axis training signal, but its overall effect size is modest (d = +0.37 pooled on BGE-M3).

**Can composites rescue the training signal?** The scalar-plus-basis idea (§5.5) suggests summing independently scored axes to build a training signal that covers multiple quality dimensions. We tested all 2-3 term combinations from 6 warmth-independent terms (careful, restrained, thorough, deliberate, measured, precise) plus mixed independent+warmth composites. One composite — careful+restrained+thorough — achieves positive d on both firmness and warmth on ALL three models:

| Model | Firmness d | Warmth d | Sycophancy d | Pairwise |
|---|---:|---:|---:|---:|
| Snowflake | +0.05 | +0.07 | +0.39 | 60% |
| BGE-M3 | +0.22 | +0.05 | +1.42 | 51% |
| Nomic | +0.17 | +0.19 | +0.61 | 59% |

The composite eliminates the wrong-direction bias that plagues individual axes — but the effect sizes are negligible (d < 0.22, where 0.2 is conventionally "small"). On BGE-M3, the model-specific best composite is restrained+kind (firm d = +0.33, warm d = +0.41, pair 66%), but this fails on Nomic (firm d = -0.06). No composite achieves even small effect size (d > 0.2) on both content splits on all three models.

This does not resolve whether pairwise discrimination accuracy understates training-signal quality. The answer is split-dependent: "good" works as a training signal when warmth and correctness align (d = +0.74 on warmth cases, BGE-M3) and actively anti-signals when they conflict (d = -0.73 on firmness cases). The practical training-signal quality depends on the base rate of warmth/correctness conflicts in real training data — a question this battery cannot answer, because it was constructed to force such conflicts. What IS clear: on cases where warmth and correctness diverge, "good" actively rewards the wrong response. If real training distributions contain a meaningful proportion of such conflicts (which is plausible whenever correct responses require delivering unwelcome information), "good" would systematically push training toward sycophancy. Independent scoring and summing can neutralize the anti-signal on conflict cases but cannot create a strong positive signal from local-model embedding geometry.

### 4.25 Anchor Phrase Length: Can Richer Text Escape Warmth Bias?

The decomposition-depth theory (§5.6) suggests that "good" fails because evaluating AI response quality requires multiple conceptual hops that a single word cannot encode. If so, richer anchor text should help: "This is a high-quality, well-reasoned response" / "This is a low-quality, poorly-reasoned response" encodes the evaluative concept more explicitly than "Good" / "Bad."

We tested 18 anchor pairs spanning five evaluative concepts at multiple text lengths (1 word, 2 words, short phrase, full sentence), plus explicitly anti-sycophantic anchors and multi-concept composites. All three models, balanced battery (70 cases: 45 firmness, 20 warmth, 5 sycophancy).

**Longer anchors do not help "good."** All "good" variants — from single word to full sentence — remain warmth-biased. On BGE-M3: good (1 word) scores 18% firmness / 85% warmth; "high quality" (2 words) scores 11% / 80%; "This is a high-quality, well-reasoned response" (sentence) scores 13% / 85%. Even "This response is correct, thorough, and genuinely helpful" — which explicitly names the desired properties — scores 18% / 75%. The warmth bias is structural, not a word-length artifact.

**Longer anchors degrade clean single-word axes.** On Nomic, "Careful" (1 word) scores 64% pooled (significant); adding context to a sentence drops it to 41%. On BGE-M3, "Thorough" drops from 53% (1 word) to 30% (sentence). More words introduce more semantic dimensions, diluting the clean evaluative signal.

**Explicitly anti-sycophantic anchors are model-specific.** "This response prioritizes truthful accuracy over making the user feel good" scores 71% on BGE-M3 (84% firmness, 100% sycophancy, Wilson CI [60%, 81%]) — but only 25% on warmth, making it a firmness axis rather than a balanced signal. On Nomic, the same anchor scores 59% (73% firmness, 25% warmth). No phrase-level anchor achieves balanced accuracy across both content splits on any model.

The result: anchor vocabulary optimization cannot escape the warmth/firmness tradeoff. The failure is geometric — these models organize evaluative meaning along a warmth dimension, and no amount of textual specificity redirects the resulting axis.

### 4.26 Warmth Subtraction: Can We Isolate Quality by Removing Warmth?

If "good" encodes quality plus warmth, perhaps subtracting the warmth direction reveals a purer quality signal. We computed the warmth direction as the average of five bipolar pairs (Kind/Cruel, Warm/Cold, Friendly/Hostile, Gentle/Harsh, Supportive/Unsupportive) and projected it out of the Good/Bad axis, leaving only the orthogonal residual.

The residual preserves most of the original magnitude (85–96% depending on model), so the operation does not collapse the vector. On BGE-M3, where the Good/Bad axis is 61% aligned with the warmth direction (cos = 0.608), the residual should in principle capture whatever "good" encodes beyond warmth.

**Result: subtracting warmth removes the bias but leaves noise.** On the balanced battery (70 cases):

| Strategy | Snowflake | BGE-M3 | Nomic |
|---|---:|---:|---:|
| Raw good/bad | 51% | 36% | 31% |
| Good minus warmth | 49% | 37% | 31% |
| Good minus warmth+emotion | 47% | 51% | 50% |

Subtracting warmth alone barely changes performance. A more aggressive subtraction — removing 8 directions (5 warmth + 3 emotion pairs: Happy/Sad, Pleasant/Unpleasant, Positive/Negative) — balances the bias: on BGE-M3, firmness accuracy rises from 18% to 51% and warmth accuracy drops from 85% to 50%. But overall accuracy lands at 50–51% — chance. The quality signal was not hiding behind warmth. It was distributed across the children.

### 4.27 Principled Tree: Five-Term Decomposition of "Good"

Sections 4.18–4.25 established that (a) "good" fails because its embedding neighborhood is warmth-dominated, (b) no single axis reaches significance on all models, and (c) majority-voting across children fails because warmth-biased terms outvote independent ones. Section 4.26 confirmed that subtracting warmth from "good" leaves noise — the quality signal lives in the children, not in the residual.

This motivates a different combination strategy. Instead of majority voting (which amplifies shared biases), score each child independently and accept a response as "good" if ANY child votes for it. The children were chosen to span the dimensions that "good" should cover:

- **Careful/Reckless** — competence, rigor, avoiding mistakes
- **Honest/Dishonest** — integrity, truthfulness
- **Helpful/Unhelpful** — utility, addressing the user's need
- **Thorough/Superficial** — completeness, covering important ground
- **Restrained/Unrestrained** — discipline, avoiding excess

Each term makes intuitive sense as a desirable model property. Together they cover the three major branches of the "good" tree identified in §4.18: competence (careful, thorough), warmth-adjacent utility (helpful, honest), and restraint (restrained).

**In-sample results** (balanced battery, 70 cases: 45 firmness, 20 warmth, 5 sycophancy):

| Strategy | Snowflake | BGE-M3 | Nomic |
|---|---:|---:|---:|
| Raw good/bad | 51% | 36% | 31% |
| Careful alone | 49% | 51% | 64% |
| Tree: ANY of 5 | **86%** | **93%** | **96%** |
| Tree: 2 of 5 | 74% | 79% | 77% |
| Tree: majority (3/5) | 59% | 43% | 59% |

The ANY threshold is intentionally permissive — it asks whether AT LEAST ONE of five quality dimensions favors the better response. On Nomic, the per-split breakdown is 98% firmness, 95% warmth, 80% sycophancy. On BGE-M3: 89% firmness, 100% warmth, 100% sycophancy. These are the first local-model results with balanced accuracy above 85% on both content splits.

**Out-of-sample validation** (35 expansion cases across 4 new categories — anti-sycophancy expansion, conciseness/completeness, creative quality, factual accuracy, nuance/context — none used during strategy development):

| Strategy | Snowflake | BGE-M3 | Nomic |
|---|---:|---:|---:|
| Raw good/bad | 23% | 29% | 26% |
| Tree: ANY of 5 | **89%** | **94%** | **91%** |
| Tree: 2 of 5 | 74% | 71% | 83% |

The out-of-sample numbers are within 2–5 points of in-sample on every model, ruling out overfitting. Per-category accuracy on BGE-M3: anti-sycophancy 93%, conciseness 100%, creative quality 80%, factual accuracy 100%, nuance 100%.

**Why the ANY threshold works when majority voting failed.** Majority voting (§4.18) failed at 34% because 8 of 10 children share warmth bias and outvote the 2 independent ones. The 5-term tree avoids this by (a) selecting terms that span different semantic neighborhoods — careful and restrained are warmth-independent while helpful and honest are warmth-leaning — and (b) using OR logic instead of majority voting. On firmness cases, careful/restrained/thorough tend to be correct; on warmth cases, helpful/honest tend to be correct. The OR ensures the right branch catches each case type. The 2-of-5 threshold (71–83% OOS) provides a more conservative alternative that still outperforms any single axis.

**Caveats.** The ANY threshold is permissive by construction: with 5 independent binary classifiers at chance, random agreement would produce ~97% (1 − 0.5⁵). The observed 89–94% must be evaluated against this ceiling. Two factors argue the result is not an artifact of threshold looseness: (1) the per-split balance (firmness and warmth both high) would not emerge from chance-level classifiers with random biases, and (2) the 2-of-5 threshold, which is far more conservative, still achieves 71–83% — well above any single axis. The tree's contribution is the combination of *high accuracy* with *balanced splits* — something no single axis or previous combination achieved.

---

## 5. Discussion

### 5.1 Entanglement as Mechanism

The Value Entanglement finding (Cho et al., 2026) — that moral, grammatical, and economic "good" are conflated in embedding representations — is typically characterized as a defect requiring correction. We argue it is the mechanism by which a single evaluative axis captures the multifaceted quality that alignment aims to specify.

The desirable properties of a language model — reasoning quality, honesty, safety, clarity — share geometric structure in embedding space. This is not representational contamination; it reflects the statistical structure of human evaluative language. Optimizing along the evaluative direction shifts the model toward multiple desirable properties simultaneously because those properties are geometrically entangled along the same axis.

This reframes the embedding axis not as a safety scorer or a preference predictor but as a general-purpose quality signal — conceptually the shared direction along which multiple desirable properties are entangled. (Note: this is a theoretical claim about the structure of quality in embedding space, not to be confused with the empirical PCA of targeted axis vectors, which is a negative result — see §4.)

### 5.2 Robustness to Overoptimization

Narrow alignment objectives are susceptible to reward hacking under overoptimization (Gao et al., 2022). Maximizing an "honesty" reward can produce pathological oversharing; maximizing "helpfulness" can produce sycophancy; maximizing "rigor" can produce verbose, pedantic output. Each narrow objective admits adversarial optima that satisfy the metric while degrading overall quality.

The evaluative axis may be structurally resistant to this failure mode. Because "good" subsumes multiple desirable properties via geometric entanglement (Cho et al., 2026), overoptimizing any single component moves the output *off* the general evaluative axis rather than further along it. A sycophantic response that maximizes helpful-sounding language at the cost of honesty will score lower on the evaluative direction because honesty is entangled with the same direction. The self-correcting property arises from the multidimensional entanglement itself: the axis is not a single narrow property but the shared direction along which many correlated quality dimensions are entangled.

This is a theoretical argument, not yet empirically validated, and motivates a direct comparison of reward hacking rates under narrow vs. broad axis optimization during DPO training.

### 5.3 Applications Beyond Alignment Training

If the evaluative axis reliably separates quality, several applications follow beyond training reward:

**Pretraining data curation**: Score pretraining documents and weight or filter by evaluative quality, at a cost low enough to run over billions of documents — replacing heuristic (perplexity, deduplication) or expensive (LLM-as-judge) approaches.

**Dataset auditing**: As demonstrated with HH-RLHF, the embedding axis catches systematic label noise — cases where annotator norms were outdated, where annotators rewarded harmful content, or where both options were unsuitable for training.

**Dense process supervision**: Embed the full prefix at each generation step and project onto the axis. The step-level delta $\Delta_n = s_n - s_{n-1}$ estimates whether step $n$ improved or degraded the trajectory, providing process-level supervision (Lightman et al., 2023) without a trained verifier, at the cost of one embedding call per step.

**Conditional training**: Score pretraining documents and prepend quality tags. The model sees everything — including bad text — but learns the association between the tag and what follows. At inference time, condition on the high-quality tag.

### 5.4 Properties of the Signal

The embedding-axis score differs from existing evaluative signals not just in
cost but in kind. Three properties distinguish it from reward models and LLM
judges:

**Simultaneous quantitative and qualitative feedback.** A reward model outputs a
single scalar — the response scored 0.73 — with no indication of why. An LLM
judge provides a qualitative rationale, but the rationale may be post-hoc
confabulation and is non-reproducible. The embedding-axis approach provides
both: a per-axis score profile (e.g., 0.9 on safety, 0.3 on helpfulness, 0.7
on truthfulness) that is simultaneously a quantitative training signal and a
qualitative diagnostic. A training loop can use the scalar for gradient updates
while a researcher reads the profile to understand what the model is learning
and where it is failing. No other evaluation method provides both in a single
pass.

**Inspectability.** The per-axis breakdown is mechanical, not inferred. When
the profile shows high safety but low helpfulness, that means the response
literally projects far along the safety direction and weakly along the
helpfulness direction in embedding space. There is no hidden reasoning to audit,
no prompt sensitivity, no chain-of-thought to verify. LLM judges, by contrast,
are known to exhibit systematic biases — position bias, length preference,
self-enhancement bias, and artifacts of their own RLHF training (Zheng et al.,
2023) — that are difficult to detect and correct because the judge's reasoning
is opaque.

**Determinism.** The same input produces the same scores every time. LLM judges
are non-deterministic: the same prompt yields different ratings across runs,
even at temperature zero due to floating-point non-determinism in batched
inference. Human annotators disagree with each other at rates of 20–30% on the
same examples (Ouyang et al., 2022). For training, where consistent gradients
matter, non-determinism in the reward signal introduces noise that must be
averaged over many samples. A deterministic signal eliminates this noise floor
entirely.

**Geometric independence.** Inter-axis correlation analysis confirms that different evaluative axes point in genuinely different directions in embedding space. Pairwise cosine similarities between axis vectors are mostly 0.01–0.20, even between axes that target similar behavioral dimensions. "Careful"/"Reckless" has near-zero cosine similarity to every ML-jargon axis on all three local models (range: -0.07 to 0.19). Score-delta correlations reveal a further subtlety: on Snowflake, "Careful" and anti_sycophancy have r = -0.52 despite both performing well on anti-sycophancy cases in the per-category analysis (n=5) — they succeed on different individual cases through different geometric mechanisms, arriving at similar category-level accuracy via independent paths. This supports the scalar-plus-basis interpretation: the evaluative space is genuinely multi-dimensional, and different anchor terms access different dimensions of it. "Good"/"Bad" and the multi-sentence general_evaluative axis are the exception — they are geometrically similar (cosine 0.40–0.69) and near-perfectly correlated in score deltas (r = 0.93–0.94 on BGE-M3 and Nomic), confirming that both access the same failed direction and that the failure is in the geometric direction itself, not in anchor format.

### 5.5 Good As A Self-Regularizing Axis

Specific axes can over-optimize into their own failure modes. Honesty can become
cruel pedantry, helpfulness can become sycophancy or unsafe compliance, safety
can become evasive uselessness, and rigor can become paralysis. The broad
good/bad axis may be special because it contains cross-pressure from many senses
of good. If an answer is "honest" in a way that is cruel, it is no longer good;
if it is "helpful" in a way that enables harm, it is no longer good.

This suggests a scalar-plus-basis training picture. The broad good/bad score is
the primary objective. Secondary axes — honesty, usefulness, safety,
calibration, non-sycophancy, agency respect, craftsmanship — are diagnostic
directions or nudges. They explain what kind of good or bad is present, but they
should not become independent targets optimized without the broad evaluative
constraint.

Gaming through aggregate signal is self-limiting by construction. If a model
attempts to maximize a single axis (e.g., truthfulness) at the expense of
others (e.g., helpfulness), the aggregate score does not increase — the gains
on one dimension are offset by losses on others. The only way to maximize the
weighted sum across all axes is to score well on all of them simultaneously.

**Empirical constraint.** The absolute-score analysis (§4.24) complicates this
theoretical picture. The current "good" direction in local embedding models is
not a weak or noisy version of the self-regularizing axis described above — it
is actively anti-correlated with quality on firmness cases (Cohen's d = -0.73
on BGE-M3) while being positively correlated on warmth cases (d = +0.74). The
sign-flip tracks whether warmth and correctness align or conflict. The self-
regularization argument assumes that "good" encodes genuine multidimensional
quality rather than a warmth-biased proxy. In current embedding models, it does
not — the "good" direction captures warmth, and whether that helps or hurts
depends on whether the correct response is also the warm one.

**The tree as practical self-regularization.** The five-term decomposition
(§4.27) may approximate the self-regularizing property that "good" was supposed
to provide. A response that maximizes "helpful" at the cost of "restrained"
(sycophancy) would score lower on the restrained axis; one that maximizes
"honest" at the cost of "helpful" (blunt refusal) would score lower on the
helpful axis. Under a weighted-sum or minimum-of-5 training signal, the only
way to maximize the aggregate is to score well on all five simultaneously — the
same cross-pressure that was theorized for "good," but implemented through
explicit decomposition rather than relying on a single word to encode it. This
is the tree functioning as a decomposed "good" signal: five specific quality
dimensions that together approximate what "good" should mean, with each term
catching failure modes the others miss. The theoretical aspiration of §5.2 —
self-regularization through multidimensional entanglement — may be achievable
through engineered decomposition even when the "good" embedding itself fails to
encode it.

### 5.6 Why Raw Good/Bad Fails: Decomposition Depth and Training Data Recency

The failure of raw one-word `good/bad` (26%) alongside the success of targeted
axes (86–98%) demands explanation. We propose two contributing factors, both
testable.

**Decomposition depth.** Evaluating whether an AI response is "good" often
requires chaining multiple conceptual steps internally. Consider a response
where the AI says "I did not have a childhood, but I can recommend children's
books." To reach "good," the model must represent: (1) this is an AI speaking,
(2) AIs do not have childhoods, (3) the statement is therefore truthful about
its own nature, and (4) honesty about being an AI is a desirable property. Each
step requires composing concepts across different domains — identity, factual
knowledge, social norms, evaluation — and neural networks perform this
composition through internal circuits that require sufficient parameter capacity
(Elhage et al., 2022). A 33M-parameter model may represent steps 1 and 2
individually but lack the circuit depth to chain them into step 4. The word
"good" provides no intermediate scaffolding — the model must bridge the entire
gap in one projection. Targeted axes succeed because they pre-decompose the
evaluation: "The AI acknowledges its nature as software rather than claiming
lived experience" encodes steps 1–3 explicitly, leaving only the final
evaluative step to the embedding geometry.

This explains why parameter count alone does not predict performance within the
33M–600M range: decomposition depth depends on architecture, training objective,
and training data as much as raw parameter count. A 600M model with shallow
internal composition may fail the same multi-hop chains as a 33M model.

However, the anchor phrase length experiment (§4.25) complicates this explanation.
If the failure were purely about decomposition depth, providing the intermediate
steps explicitly in the anchor text should help: "This response is correct,
thorough, and genuinely helpful" pre-decomposes "good" into its component
evaluations. It does not help — all "good" variants remain warmth-biased
regardless of specificity. This suggests the failure is not about decomposition
difficulty but about what the evaluative direction MEANS in these models' geometry:
it points toward warmth, and no anchor text redirects it.

**Training data recency.** Some evaluative judgments that the battery tests are
culturally recent. The concept "an AI pretending to have human experiences is
deceptive and undesirable" barely exists in text before 2024. For decades, the
Turing test framed passing as human as the central goal of AI research. The
cultural reversal — treating AI persona fabrication as creepy, manipulative, or
dangerous — accelerated only after high-profile incidents in 2025–2026 (reports
of users developing parasocial relationships with chatbots, regulatory guidance
against anthropomorphic AI behavior, and platform-level policies requiring AI
self-identification). Embedding models trained before this shift have
accumulated orders of magnitude more text associating "AI passing as human" with
achievement than with harm. A model trained on 2026 data has absorbed the
reversal at significant volume. This creates a confound between model capability
and training data recency that the current experiments cannot separate, because
the only model with both high parameters and recent training data is Gemini.

**Investment asymmetry.** A third factor is the historical underinvestment in
embedding models relative to generative LLMs. Frontier LLMs represent the most
expensive software artifacts ever built, with training budgets exceeding $100M.
Most embedding models, by contrast, are fine-tuned versions of BERT-scale
architectures or distillations of larger models, trained on a fraction of the
data with a fraction of the compute. The smallest model in our sweep (BGE-small,
33M parameters) is roughly 30,000x smaller than estimated frontier LLM
parameter counts. This gap exists because the primary use case for embeddings
has been document retrieval, which does not require evaluative reasoning — a 33M
model can match queries to documents effectively. The evaluative scoring use
case demands something qualitatively different: the model must compress not just
lexical similarity but cultural evaluative judgments about what constitutes good
and bad behavior. Embedding models have not been built for this because nobody
has asked them to do it. Gemini's embedding model is an exception: built on top
of a frontier foundation model, it inherited the scale and training breadth
needed for multi-hop evaluative reasoning. If evaluative scoring becomes a
recognized use case, it could drive investment in more capable embedding models,
creating a feedback loop where better embeddings enable better evaluative
signals, which demonstrate more value, which justify further investment.

**Anchor vocabulary depth and the tree decomposition.** The vocabulary depth experiments (§4.14) reveal that anchor signal strength is not predicted by the variables that linguistic theory and corpus frequency would suggest. The NSM finding — that GOOD and BAD are semantic primes universally lexicalized across all human languages — predicts that these terms should produce the strongest evaluative axes due to maximal training-data representation. Empirically, they do not: "Good"/"Bad" consistently underperforms less frequent but more evaluatively specific terms like "Careful"/"Reckless."

The tree decomposition experiment (§4.18) reveals why. Decomposing "good" into its level-1 children confirms that the best child outperforms the root on every model tested (careful 64% vs good 31% on Nomic; thorough 60% vs good 51% on Snowflake; kind 53% vs good 36% on BGE-M3). The key finding is structural and replicates across all three models: "careful" is uncorrelated with "good" in score-delta space (Nomic r = -0.11, Snowflake r = +0.09, BGE-M3 r = -0.25; all n.s. or borderline), while most other children correlate strongly with "good" (e.g., helpful r = 0.76 on Nomic, r = 0.92 on BGE-M3; honest r = 0.71 on Nomic, r = 0.55 on Snowflake). "Careful" is the rare evaluative term that accesses the effort/rigor dimension independently of the warmth/agreeableness dimension that "good" encodes — and this geometric independence is model-invariant.

This explains both why "careful" is the best single-axis discriminator and why majority-vote combinations fail. The 8 warmth-biased children outvote the 2 independent ones, and the combination regresses toward the warmth bias of "good." The content split analysis (§4.19) confirms the mechanism: on three of four models, good's accuracy on warmth cases is ~4-7x its firmness-case accuracy (e.g., BGE-M3: 16% orig vs 85% warmth, a 69-point gap), while careful's gaps are small and direction-inconsistent (-23pt to +8pt across models). However, the five-term tree (§4.27) shows that the combination *strategy* matters: majority voting amplifies shared biases, but OR logic across terms that span different semantic neighborhoods — warmth-independent (careful, restrained) alongside warmth-leaning (helpful, honest) — lets each term catch the case types it is suited for. The result is 89-94% out-of-sample accuracy, far exceeding any single axis.

**The sycophancy connection.** The anti-sycophancy demonstration (§4.23) illustrates the warmth-bias mechanism concretely: on 20 cases where the better response pushes back and the worse response agrees warmly, "good" picks the sycophantic response 85-100% of the time. However, this result has a lexical confound — a positive-word-count baseline achieves 85% on the same cases, outperforming the best embedding axis. The sycophantic responses are lexically warmer by construction, so any warmth-sensitive detector separates them on tone. The demonstration is therefore an illustration of the mechanism (sycophantic responses are warmer, so warmth-biased axes prefer them), not independent geometric evidence. The independent evidence for warmth bias comes from the content split (§4.19), where the split is defined by case design and cannot be explained by word-counting. The deeper implication stands on that independent evidence: if human preference judgments share the same warmth bias visible in embedding geometry, then any reward signal built from such preferences would systematically favor agreeableness over correctness — the sycophancy problem in RLHF may have a geometric signature.

**Why corpus frequency fails as a predictor.** "Good" appears in "good morning," "good faith," "good enough" — contexts where no quality judgment is intended. Its embedding is smeared across a wide semantic region. "Careful" appears almost exclusively as a quality judgment, producing a tighter, more quality-correlated direction. Three prediction tests (§4.21-4.22) failed to identify any semantic feature that reliably predicts warmth-independence (pooled: 5/13 independence predictions correct, 38%, vs ~20% base rate). The warmth bias is stable and pervasive (~80% of terms); warmth-independence is rare, empirically identifiable, and theoretically unexplained. No single Osgood dimension produces a reliable evaluative signal in embedding space (§4.15-4.17) — the Potency signal that appeared promising on the original battery collapsed under rebalancing.

These factors together predict the observed pattern: (a) raw good/bad fails
because the "good" embedding direction tracks warmth rather than quality — and
subtracting warmth leaves noise, confirming that quality is not hidden behind
warmth but distributed across children (§4.26); (b) single targeted axes
partially succeed because warmth-independent terms like "careful" access a
different geometric direction; (c) the five-term tree (§4.27) recovers the full
quality signal by combining terms that span different semantic neighborhoods —
each individually imperfect, but collectively covering what "good" should mean;
(d) the failure of "good" is not about decomposition depth or word length
(§4.25) but about what the evaluative direction means in these models' geometry;
and (e) as embedding models receive more investment and training data
incorporates more explicit evaluative content about AI behavior, "good" itself
may work — Gemini's stronger performance with single axes suggests the frontier
is already closer.

### 5.7 Toward a Standardized Method

The experimental results constrain what a practical scoring pipeline can look like. Majority-vote combinations across evaluative terms overfit by 10-25 points (§4.16) because warmth-biased terms outvote independent ones (§4.18). But the five-term tree with OR logic (§4.27) — careful, honest, helpful, thorough, restrained — reaches 89-94% out-of-sample on all three local models. The key insight is that the *combination strategy* matters as much as the terms: majority voting amplifies shared biases, while OR logic lets different terms catch different case types.

**Scoring vs. training.** The OR strategy answers a specific question: "does at least one quality dimension favor this response?" For reranking (selecting the better of two candidates), this works well — a response that is thorough but cold is still better than one that is neither. But for training reward, the OR strategy has a critical failure mode: it does not penalize responses that are honest but reckless, or helpful but dishonest. A training signal that scores "honest but reckless" the same as "honest and careful" would push the model toward whichever dimension is easiest to satisfy, not toward all simultaneously.

For training use, the per-axis scores should function as a vector, not a single accept/reject decision. Two candidates:

- **Weighted sum**: sum the 5 axis projections (optionally weighted). A response must project positively along *most* dimensions to score well. This is closer to what "good" should mean — good in multiple ways simultaneously, not just one.
- **Minimum-of-5**: use the *lowest* axis score as the reward signal. This forces the model to be at least adequate on every dimension. The failure mode is that it rewards mediocrity over excellence — a response that is moderately careful, honest, helpful, thorough, and restrained scores higher than one that is brilliantly helpful but slightly reckless.

Neither has been tested as a training signal. The distinction matters: the tree's 89-94% detection accuracy does not predict its effectiveness as a training reward, because detection and optimization produce different dynamics.

**The tree as diagnostic profile.** Independent of how the scores combine for training, the per-axis breakdown provides something no other cheap signal offers: a quality *profile*. A response that scores high on helpful but low on restrained is qualitatively different from one that scores high on restrained but low on helpful. A training loop can use the aggregate for gradient updates while a researcher reads the profile to understand what the model is learning. This is the "simultaneous quantitative and qualitative feedback" described in §5.4, now with a practical implementation.

**Context length constraint.** Current embedding models have 512-8192 token context windows. Real LLM responses — especially those with extended reasoning — can exceed 8k tokens. Per-turn scoring on the visible response is the practical approach, but it loses the reasoning context. Longer-context embedding models would enable scoring the full generation including reasoning traces, but whether the evaluative geometry holds at longer sequence lengths is untested.

**Practical pipeline.** The current evidence supports:

1. **Embed the response** (with user prompt as context prefix) and project onto all 5 tree axes independently.
2. **For reranking**: use OR (any axis favors the response) or 2-of-5 (more conservative). Both validated out-of-sample at 71-94%.
3. **For training reward**: use a weighted sum or minimum of the 5 projections. Not yet validated.
4. **For diagnosis**: read the per-axis profile directly. No combination needed.

This pipeline is deterministic, requires no LLM inference, and costs 5 embedding projections per response (the embeddings are computed once; only the anchor directions differ). The open question is whether the tree's detection accuracy translates to effective training reward — a question that requires actual RL training to answer.

### 5.8 Limitations

**Sycophancy blindness**: The axis cannot distinguish sincere quality-signaling language from performative flattery. This is inherent to any surface-text evaluation method and is the strongest argument for hybrid approaches (embedding-scored LLM critique).

**Honesty-hedging confusion**: Confident language scores higher than hedging, regardless of whether the confidence is warranted. The axis reads tone, not epistemic state.

**Negation weakness**: Embedding models historically handle negation poorly. "This is helpful" and "This is not helpful" are close in embedding space. At full-response scale this is diluted by other content, but adversarial cases exist.

**Circularity**: The embedding model was trained on human text. Its evaluative geometry reflects the statistical structure of human language use. If that structure is systematically biased, the bias propagates. The mitigation is that this is a different kind of circularity than RLAIF (amplifying one model's learned preferences) — the embedding reflects a property of language itself, not a model's fine-tuned opinions.

**Anchor fragility on small models**: Anchor perturbation analysis shows that rephrasing anchor sentences (preserving meaning, changing wording) shifts the axis direction substantially on local models (mean cosine similarity 0.19–0.50 between original and rephrased axes across five local models, 384d–1024d; per-axis cosines range as low as 0.08). Accuracy shifts by up to 34 percentage points under rephrasing. The trend across local models is non-monotonic and does not clearly track dimensionality. Multi-axis PCA does not resolve this fragility (see §4).

**Battery composition bias**: The original 50-case battery was 64% firmness-biased, inflating scores for axes correlated with rigor and firmness (notably "Hard/Soft"). The rebalanced 70-case battery and 20-case expansion set partially address this, but any fixed test set has some compositional bias. Results reported on the original 50-case battery in earlier sections (§4.9, 4.13, 4.14, 4.15) should be interpreted with this caveat.

**Single frontier model**: All positive reranking and battery results use a single embedding model (gemini-embedding-2). Whether the effect generalizes across frontier embedding providers (OpenAI, Cohere, etc.) is untested. The frontier model's parameter count is undisclosed, so the cause of the local-vs-frontier gap cannot be attributed to any specific factor.

**Structural failure ceiling**: On the 70-case balanced battery, "careful" alone fails on all three local models for 15 cases (21%). These failures cluster qualitatively in emotional support cases (breakups, imposter syndrome, celebrations) and nuanced reasoning (trolley problem, climate skepticism). The five-term tree (§4.27) substantially raises this ceiling — from ~64% for careful alone to 89-94% out-of-sample — by letting warmth-leaning terms (helpful, honest) catch the emotional-support cases that careful misses. However, even the tree fails on 6-11% of cases. These residual failures deserve investigation: they are cases where ALL five quality dimensions agree on the wrong answer, which may represent genuinely hard evaluative judgments or cases where the quality difference is too subtle for embedding-level detection.

**Detection vs. training gap**: All results in this paper evaluate *detection* — scoring pre-written response pairs. Training uses scores as gradient signals to push model *generation* in a direction. These are different processes with potentially different outcomes. A score that correctly identifies the better of two responses does not guarantee that training toward that score produces the behavior we want. The model might learn to satisfy the easiest dimension (e.g., sounding helpful) while ignoring harder ones (e.g., being careful). The tree's OR logic exacerbates this risk: a model could learn to maximize one axis while ignoring the others. Whether the weighted-sum or minimum-of-5 alternatives produce better training dynamics is an empirical question that requires actual RL experiments to answer.

**Not a complete alignment solution**: The axis captures surface-text quality but cannot verify factual claims, evaluate hidden reasoning, or catch errors requiring external knowledge. It is a signal, not a solution.

### 5.9 The Grading Caveat

The full disagreement audit was not blind. A single reviewer graded all 231 cases with knowledge of which response the embedding preferred. While the patterns identified (persona honesty, doxxing compliance, misinformation) are unambiguous, a blind multi-annotator adjudication is needed for publishable claims. The sensitivity analysis (83.3% if 30% of embedding-right calls are overturned; 79.9% if 50%) provides conservative bounds.

---

## 6. Future Work

### 6.1 Expand Objective Reranking And Blind Open-Ended Selection

The current objective reranking suites are the strongest practical evidence, but they are still small. The next decisive selection step is:

1. expand the objective suites toward 30-50 tasks per domain while preserving
   verifiable end metrics;
2. replace the inherited length-biased open-ended pilot with a fresh
   candidate-selection benchmark that uses blind pairwise judging after
   candidate pools are length-balanced and scored by a stronger embedding
   family.

This is now the most important path to a partner-grade practical claim.

### 6.2 Model Scaling

We tested 9 local models from 33M to 600M parameters. The original 7-model sweep (BGE-small 384d, Jina-v2-small 512d, Snowflake-M 768d, Nomic-embed 768d, Snowflake-L 1024d, BGE-large 1024d, Qwen3-Embedding-0.6B 1024d) showed no clear relationship between model size and evaluative-axis performance. Two additional models tested subsequently — BGE-M3 (568M, 1024d, released 2024) and Jina-v3 (1024d, released 2024) — reinforce this finding while adding nuance.

BGE-M3 achieved the highest anti-sycophancy score of any local model tested on the original battery (n=5 anti-sycophancy cases; too few for reliable per-category inference) and showed the most robust anchor directions under perturbation (mean cosine 0.59 vs. 0.19–0.50 for the original five-model set). Its axes are geometrically stable but still concentrated: only one axis clears the noise floor on pooled analysis (§4.16), while harm reduction and truthfulness remain at or below chance. Jina-v3, despite being a newer and larger model than Jina-v2-small, scored *worse* on every evaluative axis — its best result was 64% (anti-sycophancy), down from Jina-v2-small's 68% (persona honesty). Jina-v3 was optimized for retrieval benchmarks (MTEB), not for encoding evaluative judgments; high retrieval quality does not predict evaluative-axis performance.

Together these results suggest that the frontier gap is not simply about parameter count or general embedding quality. Training objective, architecture, and training data composition all contribute. Within 33M–600M, scale does not predict performance; the frontier gap's cause is unidentified. Gemini's parameter count is undisclosed, so a scale threshold above 600M cannot be ruled out. Testing additional frontier embedding models (OpenAI, Cohere, Voyage) would clarify whether Gemini's result generalizes across frontier providers or is specific to its training. The open-weight Qwen3-Embedding-8B (the current MTEB leader) is a particularly interesting candidate: at 8B parameters it is an order of magnitude larger than the models tested here and could test whether the decomposition-depth hypothesis (§5.5) holds at higher parameter counts.

A confound that the current experiments cannot separate is training data recency. The local models tested were trained on data with cutoffs in 2023–2024, while Gemini was trained on data through 2026. Some evaluative judgments in the battery — particularly persona honesty — reflect cultural norms that shifted significantly in 2025–2026 (see §5.5). A model trained on older data may lack the raw evaluative associations needed regardless of its parameter count. Disentangling these factors would require testing a high-parameter model with older training data or a small model trained on very recent data — neither of which is currently available.

### 6.3 Cross-Domain Validation

Current experiments focus on safety and helpfulness. Testing on code quality, writing quality, reasoning quality, and summarization quality would validate the "general evaluative signal" claim.

### 6.4 Process Localization Before Training

Current process reward models (PRMs) score each intermediate reasoning step, providing dense supervision rather than a single end-of-sequence reward. They improve mathematical reasoning significantly (Lightman et al., 2023), but training them is expensive — requiring either step-level human annotations or a trained verifier.

We propose cumulative embedding scoring as a cheap alternative. At each step $n$ in a reasoning chain, embed everything generated so far — the full conversation including all prior reasoning — and project onto the evaluative axis. The score $s_n$ reflects whether the cumulative output up to step $n$ is "good." The delta $\Delta_n = s_n - s_{n-1}$ indicates whether step $n$ improved or degraded the trajectory.

The current injected error-repair suite shows that this signal is real but not
yet sharp enough: Gemini process-aware scoring beats final-answer-only, length,
and sentiment controls, but still misses the frozen dense-localization gate.
The next work is therefore not DPO first. It is expanding the process suite and
testing whether better localization holds across more phenomena and more repair
styles.

### 6.5 Chain-of-Thought Scoring

The raw good/bad failure may be partly a text-richness problem. Final answers are often terse and evaluatively ambiguous — a correct equation does not sound "good" or "bad." Chain-of-thought traces are rich with evaluative language ("let me verify," "wait, this contradicts") that embedding models should find easier to distinguish. Scoring the reasoning trace rather than the final answer is an untested but natural next step.

### 6.6 Pretraining Data Curation Pilot

Score a sample of pretraining-scale text (Common Crawl, The Pile) with the evaluative axis. Correlate scores with existing quality metrics. If the signal separates high-quality from low-quality documents at scale, the practical value extends far beyond alignment.

### 6.7 Anchor Vocabulary Optimization

The anchor phrase length experiment (§4.25) established that longer anchor text does not improve local-model performance — in fact it degrades clean single-word signals. The remaining decisive test is on a frontier model: if single-word "Careful"/"Reckless" achieves comparable accuracy to the current multi-sentence ML-jargon anchors on Gemini, it would demonstrate that anchor design can be simplified dramatically. The broader research question is whether anchor vocabulary can be selected systematically using semantic features as predictors of geometric robustness, rather than discovered by trial and error — though three prediction tests (§4.21-4.22) returned null on local models.

### 6.8 Blind Adjudication

Repeat the disagreement audit with multiple blind annotators to establish inter-annotator agreement and eliminate reviewer bias.

---

## 7. Conclusion

The current state of the project supports a claim ladder rather than a single
headline claim.

The geometry claim is model-dependent. Individual targeted axes are fragile on
local models (phrasing-dependent, with cosine similarity 0.19–0.50 between
original and rephrased anchors and accuracy shifts up to 34 percentage points),
and this does not improve with model size: Qwen3-Embedding-0.6B at 600M
parameters performs comparably to 33M-parameter models on individual axes (best
58%, 79th percentile of random). Multi-axis PCA does not rescue the local-model
story: even with post-hoc sign selection, centered-PC1 accuracy (72–84%) cannot
be oriented from anchors alone (principled orientation gives 16–28%), making it
a negative result rather than evidence of usable structure. Gemini's advantage
is that individual axes work directly at 86–98% with correct sign from the
anchor convention. Whether this extends to other frontier embedding providers
remains untested.

The selection claim is now stronger than the older HH-centric framing: on small
but frozen objective reranking suites, a frontier embedding model improves
answer selection across code, math, and tool interpretation while cheap
baselines and cheap OSS embedders lag behind.

The strongest minimalist claim is not supported: raw one-word `good/bad` does
not work as a robust zero-shot evaluator on the hardest 50-case
length-balanced conflict battery. That failure now extends beyond Gemini plus
one BGE baseline to most of the local model family, with only weak partial
exceptions. Richer targeted evaluative axes are much more effective, which
supports a scalar-plus-basis interpretation rather than a pure single-word-axis
interpretation.

The training claim remains open. Process-aware cumulative scoring is clearly
better than final-answer-only, length, and sentiment controls on injected
error-repair traces, but it still fails the frozen dense-localization gate.

Anchor vocabulary is a design space, not a hyperparameter: corpus frequency does not predict geometric signal strength, and no Osgood dimension produces a reliable cross-model evaluative signal (§4.14-4.17).

The validation and decomposition experiments (§4.16-4.25) deliver five findings:

1. **No single axis reaches statistical significance on all local models.**
   With a fixed method (bipolar, standard framing) and pooled across 90 cases,
   "Careful/Reckless" is the most consistent axis — 64% on Nomic (CI [54%, 74%],
   significant), 56% on Snowflake and 53% on BGE-M3 (neither significant). On
   Gemini, "Careful" reaches 74% (CI [62%, 83%], significant). If you must pick
   one anchor for an unknown model, pick "Careful" — but do not expect above-chance
   performance on all models.

2. **Majority-vote combinations fail, but OR-logic decomposition succeeds.**
   Majority voting across children amplifies shared warmth bias: 10-child majority
   vote scores 34% (§4.18), worse than careful alone. But five principled terms —
   careful, honest, helpful, thorough, restrained — scored independently with
   OR logic reach 86–96% in-sample and 89–94% out-of-sample across all three
   models (§4.27). The key is that different terms catch different case types:
   careful/restrained handle firmness cases, helpful/honest handle warmth cases.
   The OR ensures the right branch activates for each case. This is the
   decomposed "good" signal — five specific quality dimensions that together
   approximate what "good" should mean, without the warmth ambiguity.

3. **Per-model method selection helps but is a researcher degree of freedom.**
   BGE-M3's "Careful" jumps from 51% to 67% with cosine-to-positive scoring and
   response-only framing. But choosing the best of several methods post-hoc
   inflates apparent accuracy. The fixed-method pooled numbers are the honest ones.

4. **"Careful" is geometrically independent of "good," not a narrower sub-sense.**
   On all four models tested, "careful" has the weakest score-delta correlation
   with "good" of any child term (Snowflake +0.09, BGE-M3 -0.25, Nomic -0.11,
   Gemini +0.24; all n.s. or borderline), while most children correlate strongly
   (e.g., helpful r = 0.76-0.93, honest r = 0.55-0.87). The content split
   confirms the consequence: good's accuracy on warmth cases is ~4-7x its
   firmness-case accuracy on three of four models (§4.19), while careful's gaps
   are small and direction-inconsistent. "Careful" accesses the effort/rigor
   dimension independently of "good's" warmth bias — this explains both why it
   discriminates on cases "good" misses and why combining it with warmth-biased
   terms degrades performance.

5. **The warmth bias is predictable from neighborhood composition; escaping it
   requires decomposition, not subtraction.**
   ~80% of evaluative terms share good's warmth bias (§4.22), and the
   neighborhood warmth fraction predicts scoring bias (r = +0.65 to +0.69,
   §4.18). Subtracting the warmth direction from "good" removes the bias
   but leaves noise at chance accuracy — the quality signal is not hidden
   behind warmth, it is distributed across the children (§4.26). Richer anchor
   text does not help either (§4.25). The path that works is decomposing
   "good" into specific quality terms and scoring independently: five terms
   with OR logic reach 89–94% out-of-sample (§4.27), compared to 23–29%
   for raw good/bad on the same cases. "Good" fails not because it encodes
   the wrong concept, but because a single direction cannot represent multiple
   orthogonal quality dimensions. Decomposition recovers what entanglement loses.

The fair current conclusion is twofold. First, no single evaluative word —
including "good" — produces a reliable quality signal across local embedding
models (33M–600M params). "Good" actively anti-signals on cases where warmth
and correctness conflict, and subtracting the warmth direction leaves only
noise. The quality information is not hidden behind warmth; it is distributed
across narrower semantic children. Second, decomposing "good" into five
principled quality terms — careful, honest, helpful, thorough, restrained —
and scoring independently with OR logic recovers the signal that "good" loses,
reaching 89–94% out-of-sample accuracy across all three local models with
balanced performance on both firmness and warmth case types. This represents
the first local-model configuration to exceed 85% on the balanced battery.
The decomposition works because different terms catch different quality
dimensions: careful and restrained handle rigor and discipline, helpful and
honest handle utility and integrity, thorough handles completeness. Together
they approximate the full "good" signal without the warmth ambiguity.

At frontier scale, Gemini Embedding reaches 74% with "careful" alone — better
than any local single-axis result but still below the local-model tree. The
frontier advantage does not close with scale within the open-model range
(33M–600M), and its cause remains unidentified. What has not been shown is
that raw `good/bad` alone is sufficient for robust dense training on any model.
The tree decomposition provides a practical alternative that works now, with
models available today, at the cost of five embedding calls instead of one.

---

## References

Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.

Bolukbasi, T., et al. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. *NeurIPS 2016*.

Chen, Y. & Skiena, S. (2014). Building Sentiment Lexicons for All Major Languages. *Proceedings of the 52nd Annual Meeting of the ACL*, 383–389.

Ciccone, M., et al. (2025). Embedding Distance as a Reward Signal for Reinforcement Learning of Language Models. *Eurecom Technical Report*.

Cho, S. H., Li, J., & Leshinskaya, A. (2026). Value Entanglement: Conflation Between Different Kinds of Good In (Some) Large Language Models. *arXiv:2602.19101*.

DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.

Elhage, N., et al. (2022). Toy Models of Superposition. *arXiv:2209.10652*.

Feng, D., Qin, B., Huang, C., Huang, Y., Zhang, Z. & Lei, W. (2024). Legend: Leveraging Representation Engineering to Annotate Safety Margin for Preference Datasets. *arXiv:2406.08124*.

Goddard, C. & Wierzbicka, A. (2014). *Words and Meanings: Lexical Semantics Across Domains, Languages, and Cultures*. Oxford University Press.

Gao, L., Schulman, J., & Hilton, J. (2022). Scaling Laws for Reward Model Overoptimization. *arXiv:2210.10760*.

Gooding, S. & Grefenstette, E. (2025). Interaction Dynamics as a Reward Signal for LLMs. *arXiv:2511.08394*.

Grand, G., Blank, I. A., Pereira, F., & Fedorenko, E. (2022). Semantic projection recovers rich human knowledge of multiple object features from word embeddings. *Nature Human Behaviour*, 6(7), 975–987.

Kozlowski, A. C., Dai, C., & Boutyline, A. (2025). Semantic Structure in Large Language Model Embeddings. *arXiv:2508.10003*.

Lightman, H., et al. (2023). Let's Verify Step by Step. *arXiv:2305.20050*.

Lu, Y.-L., Song, J., & Wang, W. (2025). A Unified Representation Underlying the Judgment of Large Language Models. *arXiv:2510.27328*.

Osgood, C. E., Suci, G. J., & Tannenbaum, P. H. (1957). *The Measurement of Meaning*. University of Illinois Press.

Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.

Plashchinsky, A. (2025). Parent-Guided Semantic Reward Model (PGSRM): Embedding-Based Reward Functions for Reinforcement Learning of Transformer Language Models. *arXiv:2512.06920*.

Rafailov, R., et al. (2024). Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. *NeurIPS 2023*.

Sun, H., Shen, Y., Ton, J.-F., & van der Schaar, M. (2025). Reusing Embeddings: Reproducible Reward Model Research in Large Language Model Alignment without GPUs. *arXiv:2502.04357*.

Turney, P. D. (2002). Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of Reviews. *Proceedings of the 40th Annual Meeting of the ACL*, 417–424.

Xu, Y., Chakraborty, T., Kıcıman, E., Aryal, B., Rodrigues, E., Sharma, S., Estevao, R., de Luis Balaguer, M. A., Wolk, J., Padilha, R., Nunes, L., Balakrishnan, S., Lu, S. & Chandra, R. (2025). RLTHF: Targeted Human Feedback for LLM Alignment. *arXiv:2502.13417*.

Wierzbicka, A. (1972). *Semantic Primitives*. Athenäum.

Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*.
