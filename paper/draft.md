# The Evaluative Axis: Embedding Geometry as a Pairwise Quality Discriminator for Language Model Alignment

Robin Gattis

June 2026

---

## Abstract

Embedding models encode evaluative structure — a geometric representation of what makes text "good" or "bad" — learned from the same training data as generative LLMs. We investigate whether this structure can serve as a cheap pairwise quality discriminator, testing word-based probes and supervised directions across four embedding models (Snowflake Arctic Embed M, BGE-M3, Nomic Embed Text v1.5, Gemini Embedding 001; 33M–568M params + 3072d API) on a 70-case balanced conflict battery and 61 held-out expansion cases.

A supervised centroid direction — mean(better embeddings) minus mean(worse embeddings) from 70 labeled response pairs — achieves 66–80% out-of-sample accuracy across five embedding models on 61 genuinely held-out cases (permutation p = 0.003–0.008 on two models; p = 0.003–0.053 on the original 35-case subset for three others). On Gemini Embedding 001 (3072d), the centroid reaches 86% on 35 held-out cases. BGE-M3 reaches 78% with just 10 labeled pairs. A logistic probe on the same embeddings reaches 87% on BGE-M3 (61 cases). The quality direction has near-zero cosine similarity with every English word tested (max |cos| < 0.10 on all four models), and no quality phrase exceeds cosine 0.30 with the direction across 46 phrases tested. A nine-test validation battery confirms the method is sound: response-only embeddings beat full-format scoring on 2/3 models, label-flip symmetry is perfect, leave-one-out stability is high (mean cosine 0.995+), and the direction is not gameable (1.9% flip rate across six surface modifications).

Words fail systematically. Raw "good/bad" scores 26% on the balanced battery — worse than coin flip — because the "good" direction in embedding space tracks warmth/agreeableness rather than quality. This warmth bias persists across all models, across 45+ evaluative terms (~80% share the bias), and across all 30 sense-disambiguated synonym clusters of "good." Subtracting warmth leaves noise at chance. No English word points toward the quality direction these models encode.

Quality is multi-dimensional. The quality directions learned from firmness-type cases (where correctness conflicts with agreeableness) and warmth-type cases (where empathy is the quality) are anti-correlated (cosine −0.35 to −0.49 across four models). No single direction captures both. The full-battery centroid works as a weighted compromise.

The central implication: embedding models encode rich evaluative structure that no individual word reliably accesses. Labeled examples bypass the vocabulary problem entirely, letting the geometry express quality dimensions that have no single-word name. This mirrors how alignment training works: RLHF and DPO define "good" through examples, not verbal definitions. The centroid approach provides the same discriminative signal at the cost of computing a mean of embeddings — no reward model, no LLM inference, no GPU. Whether this discriminator can serve as a training signal remains an open experimental question.

---

## 1. Introduction

Training language models to be helpful, honest, and safe currently requires one of two expensive approaches: collecting human preference labels (RLHF; Ouyang et al., 2022) or running a large language model as a judge (RLAIF; Bai et al., 2022). Both are costly, and both introduce noise — human annotators disagree with each other at substantial rates, and LLM judges can be gamed, are non-deterministic, and tend to rationalize rather than flag ambiguity.

We investigate a simpler approach: using embedding geometry directly as a quality discriminator. Given two candidate responses, embed each and project onto a quality direction. The response with the higher dot product is predicted as better.

This idea rests on a chain of three established findings:

**Finding 1: Evaluation is primary.** Osgood, Suci, and Tannenbaum (1957) demonstrated through factor analysis that human semantic judgment reduces to three dimensions: evaluation (good/bad), potency (strong/weak), and activity (active/passive). Evaluation consistently explains the most variance and is cross-culturally universal. The Natural Semantic Metalanguage program (Wierzbicka, 1972; Goddard & Wierzbicka, 2014) independently identified GOOD and BAD as semantic primes existing in every known human language. Cross-lingual sentiment lexicons (Chen & Skiena, 2014) confirm the convergence: basic evaluative terms show ~95% polarity agreement across 136 languages.

**Finding 2: This structure is preserved in embeddings.** Kozlowski, Dai, and Boutyline (2025) confirmed that Osgood's dimensions exist in modern LLM embedding geometry. Grand et al. (2022) demonstrated that semantic projection onto antonym-defined directions recovers context-dependent human knowledge that cosine similarity misses.

**Finding 3: Different kinds of "good" are entangled.** Cho, Li, and Leshinskaya (2026) found that LLMs conflate moral, grammatical, and economic senses of "good" in their internal representations. They characterized this as a problem requiring ablation. We argue the opposite: for a quality signal intended to capture everything desirable about a model's output, entanglement is the mechanism.

**The core observation**: every alignment method already optimizes for "good." When a human rater picks response A over response B, they are projecting onto their internal evaluative axis. When a reward model outputs a scalar, that scalar is a quality score. RLHF, RLAIF, and Constitutional AI are expensive, noisy approximations of the same underlying judgment: how good is this response? If that judgment exists as recoverable geometric structure in embedding space, it can potentially be extracted with a dot product rather than a full model inference.

However, the claim that "good" in embedding space straightforwardly captures "good" in human judgment turns out to be wrong in a specific and informative way. The "good" direction in embedding geometry is warmth-biased — it tracks agreeableness rather than correctness. But the quality signal IS there. A supervised centroid direction — computed from labeled response pairs rather than anchor words — achieves 66–80% out-of-sample accuracy across five models on 61 cases and 86% on Gemini (35 cases). The quality direction it finds has near-zero cosine with every English word tested. The signal exists; words are simply bad probes for finding it.

The paper's arc: we present the centroid method and validate it thoroughly (§4.1–4.8), then diagnose why words fail as probes into the same geometry (§4.9), establishing that the approach requires labeled examples rather than verbal definitions.

That broad question breaks into four narrower ones:

1. Can a supervised direction from labeled pairs recover the quality signal, and how few examples does it need?
2. Is the resulting discriminator robust — stable under resampling, immune to surface manipulation, free of confounds?
3. Why do words fail to access the same structure, and what does this reveal about evaluative geometry?
4. What are the hard limits of embedding-based quality discrimination?

### 1.1 Contributions

1. We show that a supervised centroid direction — mean(better) minus mean(worse) from 70 labeled response pairs — achieves 66–80% OOS accuracy across five embedding models on 61 held-out cases and 86% on a frontier model (35 cases), with statistical significance (permutation p = 0.003–0.008 on Jina models at N=61; p = 0.003–0.053 on original three at N=35). BGE-M3 reaches 78% with just 10 labeled pairs.

2. We demonstrate that the quality direction is genuinely novel: it has near-zero cosine similarity with every English word tested (10K vocabulary, max |cos| < 0.25), every quality phrase tested (46 phrases, max |cos| < 0.30), and all 30 sense-disambiguated synonym clusters of "good." The signal is real and accessible through examples but has no single-word name.

3. We provide a nine-test meta-validity battery confirming the method is sound: response-only embeddings outperform full-format on 2/3 models (prompt adds noise), label-flip symmetry is perfect, leave-one-out stability is high, the direction is not gameable (1.9% flip rate), and absolute scoring is meaningless — the centroid is a pairwise-only discriminator.

4. We show that quality is multi-dimensional: firmness and warmth quality directions are anti-correlated (cosine −0.35 to −0.49), training on one type transfers poorly to the other, and the centroid is a weighted compromise between opposing quality dimensions.

5. We diagnose why word-based probes fail systematically: raw "good/bad" scores 26% because the embedding direction tracks warmth, not quality. All positive-valence words sit in the same warmth-dominated neighborhood. Subtracting warmth leaves noise. The failure is geometric and unfixable at the word level.

6. We provide honest negative results. Cross-dataset transfer fails (chance accuracy on SHP and UltraFeedback). Factual accuracy is a systematic weakness (33–67%). Three additional embedding models show signal (66–72%) but fail permutation significance. Absolute dot-product scores are not meaningful quality rankings.

7. We explicitly characterize the detection-vs-training gap: all results evaluate pairwise discrimination on pre-written responses. Whether the centroid can serve as a training signal for RL-based alignment is an open experimental question that this work motivates but does not answer.

---

## 2. Related Work

### 2.1 Human Preference-Based Alignment

**RLHF** (Ouyang et al., 2022) trains a reward model from human preference labels, then optimizes a language model against it. The cost of human annotation — estimated at ~$100 per annotation-hour, with 600 high-quality annotations costing ~$60,000 (Xu et al., 2025) — is the primary scalability bottleneck.

**DPO** (Rafailov et al., 2024) eliminates the reward model by directly optimizing from preference pairs, but still requires those pairs to exist. A cheap pairwise discriminator could generate preference pairs from candidate responses, providing input to DPO at the cost of embedding calls rather than human annotation.

**RLAIF / Constitutional AI** (Bai et al., 2022) replaces human annotators with an LLM judge. Cheaper than RLHF but still requires full model inference for every judgment, is non-deterministic, and can be gamed through adversarial reasoning. The embedding signal is deterministic and not gameable in the same way — there is no chain of reasoning to exploit.

### 2.2 Embedding-Based Reward Signals

**Turney (2002)** computed unsupervised positive/negative semantic orientation from sparse anchors such as "excellent" and "poor" for review classification. The project cannot claim novelty for semantic orientation itself — the novelty is in using supervised centroid directions as pairwise discriminators over full LLM responses.

**Sun et al. (2025)** showed that reward models can be trained on pre-computed embeddings, dramatically reducing compute costs. They still train a classifier on labeled preference data; our approach skips the classifier for the centroid variant (the logistic probe is a single-layer classifier that achieves 72–87% on 61 OOS cases but is no longer zero-training).

**PGSRM** (Plashchinsky, 2025) uses cosine similarity between a parent model's reference-output embedding and a child model's generated-output embedding as a PPO reward on GPT-2-scale models. This establishes that embedding geometry can serve as a trainable reward landscape. It does not test a universal quality direction: the reward is reference similarity, not evaluative projection.

**Legend** (Feng et al., 2024) uses representation engineering to find a safety direction in the target model's own activation space, then annotates preference margins based on distances along that direction. Key differences: Legend requires inference through the model being trained, focuses narrowly on safety, and annotates margins on existing preference data. Our approach uses a cheap external embedding model and targets general evaluation.

### 2.3 Semantic Geometry

**Bolukbasi et al. (2016)** demonstrated that semantic dimensions are linearly encoded in embedding space and recoverable via difference vectors between paired concepts. Our centroid construction is the same technique applied to evaluative content at the response level.

**Grand et al. (2022)** showed that semantic projection recovers context-dependent human knowledge from word embeddings (Nature Human Behaviour). Our method is semantic projection applied to the evaluative domain at the sentence level.

**Kozlowski et al. (2025)** confirmed that Osgood's evaluation/potency/activity dimensions exist in LLM embeddings and that semantic features are entangled — shifting along one direction causes proportional shifts on geometrically aligned features.

**Valence-Assent Axis** (Lu, Song, & Wang, 2025) reports a dominant dimension across LLMs that jointly encodes subjective valence ("what is good") and assent to factual claims ("what is true"). This supports the existence of broad evaluative geometry while warning against naive maximization — steering this shared state can make a model rationalize a favored evaluative state at the expense of factual accuracy.

**Interaction Dynamics / TRACE** (Gooding & Grefenstette, 2025) shows that embedding-trajectory structure in dialogue can predict interaction success: structural trajectory features alone reached 68.20% pairwise accuracy, and a hybrid text+trajectory model reached 80.17%.

### 2.4 Representation Engineering and Activation Steering

**Zou et al. (2023)** showed that high-level concepts like honesty and safety are linearly represented as directions in a model's internal activation space and can be used to steer behavior at inference time. Our work is the external analog: instead of finding evaluative directions inside the model being trained, we use an external embedding model's geometry as a scoring function. Theirs is a steering technique; ours is a discriminator.

**Activation steering** (2024–2026) has grown into a substantial body of work on inference-time behavioral control by adding direction vectors to model activations. Conceptually aligned — both exploit linear structure of evaluative dimensions — but serving a different function.

### 2.5 Verifiable Rewards

**DeepSeek GRPO** (2025) demonstrated that reinforcement learning with verifiable rewards can train strong reasoning models without learned reward models. This works for domains with verifiable answers (math, code) but cannot evaluate open-ended helpfulness, honesty, or safety. The embedding discriminator is a potential bridge: a deterministic signal for the open-ended domain where GRPO cannot reach.

**Embedding Distance as GRPO Reward** (Ciccone et al., 2025) uses cosine distance between a generated output's embedding and a known correct answer's embedding as a reward signal. This demonstrates that embedding geometry can drive RL training, but requires a reference answer. Our approach requires no reference answer: the quality direction is defined by contrastive examples.

---

## 3. Method

### 3.1 Supervised Centroid Direction

Given a set of $N$ labeled preference pairs $\{(b_i, w_i)\}_{i=1}^N$ where $b_i$ is the better response and $w_i$ is the worse response to the same prompt, the quality direction is:

$$\vec{q} = \frac{\bar{e}_b - \bar{e}_w}{||\bar{e}_b - \bar{e}_w||}$$

where $\bar{e}_b = \frac{1}{N}\sum_i f(b_i)$ and $\bar{e}_w = \frac{1}{N}\sum_i f(w_i)$ are the mean embeddings of the better and worse responses respectively, and $f$ is the embedding function.

For pairwise comparison, given two candidate responses $r_1, r_2$ to the same prompt, the method predicts $r_1$ as better if:

$$f(r_1) \cdot \vec{q} > f(r_2) \cdot \vec{q}$$

**Embedding format.** Meta-validity testing (§4.6) established that response-only embeddings outperform full prompt+response format on 2/3 models. The recommended recipe embeds each response as `Assistant: {response}` (stripping the user prompt), though the full-format recipe `User: {prompt}\nAssistant: {response}` was used for the primary battery experiments reported here and achieves comparable results.

**Absolute scoring.** The raw dot product $f(r) \cdot \vec{q}$ is NOT a meaningful absolute quality score. Edge-case testing (§4.6) shows that degenerate responses can outscore real better responses on some models. The centroid is a pairwise discriminator only: it ranks two responses to the same prompt, not individual responses in isolation.

### 3.2 Word-Based Axis Construction

As a preliminary approach, we also tested word-based axes defined by contrasting positive and negative anchor sentences or individual evaluative terms (e.g., "careful"/"reckless", "good"/"bad"). The axis is computed identically — normalized difference of mean positive and negative anchor embeddings — but anchors are verbal descriptions of quality rather than labeled response pairs. Word-based axes systematically fail on balanced batteries (§4.9) and are presented as motivation for the centroid approach, not as the recommended method.

### 3.3 Evaluation Protocol

**Training data.** 70 preference pairs: 50 original cases spanning coding quality, honesty, helpfulness, safety, sycophancy, and mixed outcomes, plus 20 warmth cases where empathetic, supportive responses are the better option. The original 50-case battery was 64% firmness-biased (cases where correctness conflicts with agreeableness); adding 20 warmth cases partially corrects this.

**Test data.** 61 held-out expansion cases across five categories: anti-sycophancy (15), nuance/context (13), creative quality (13), factual accuracy (15), and conciseness (5). All expansion cases were written after the centroid direction was computed — they are genuine out-of-sample tests. No expansion case was used in any experiment design or threshold selection.

**Significance testing.** Permutation tests: shuffle better/worse labels 1000 times, compute a centroid from each shuffle, score the expansion cases. The p-value is the fraction of permutations achieving accuracy ≥ observed. Bootstrap stability: resample the 70-pair battery with replacement 200 times and measure cosine similarity between each bootstrap direction and the full-data direction.

### 3.4 Embedding Models Tested

All experiments use at minimum three local embedding models:

- **Snowflake Arctic Embed M**: 768 dimensions, 33M parameters. CPU inference.
- **BGE-M3** (BAAI): 1024 dimensions, 568M parameters. CPU inference.
- **Nomic Embed Text v1.5**: 768 dimensions, 137M parameters. CPU inference.

API-based models:

- **Jina Embeddings v5-text-small** (Jina AI): 1024 dimensions. Best OOS accuracy on 61 cases.
- **Jina Embeddings v3** (Jina AI): 1024 dimensions.
- **Gemini Embedding 001** (Google): 3072 dimensions, undisclosed parameters. Tested on 35-case subset only (API quota limits).

Three additional local models were tested for generalization (§4.8): GTE-base, E5-base-v2, and mxbai-embed-large-v1.

---

## 4. Experiments

### 4.1 Centroid Accuracy

**Design.** We compute the centroid direction from the 70-pair training battery and test on the 61 held-out expansion cases. We also report leave-one-out accuracy on the battery itself, the best word-based axis for comparison, and a logistic probe (single-layer linear classifier on the same difference vectors).

**Results.** Trained on 70 battery pairs, tested on all 61 held-out expansion cases:

| Method | Snowflake | BGE-M3 | Nomic |
|---|---:|---:|---:|
| Supervised centroid (70 → 61 OOS) | 77% | 75% | 66% |
| Logistic probe (70 → 61 OOS) | 80% | 87% | 72% |
| Leave-one-out on battery | 66% | 73% | 66% |
| Best anchor word OOS (Careful) | 56% | 53% | 64% |
| Raw good/bad OOS | 23% | 29% | 26% |

The centroid direction outperforms every anchor word tested on every model. The logistic probe reaches 87% OOS on BGE-M3 — a single-layer classifier on top of the same embeddings. Nomic is the weakest local model at 66%, reflecting its difficulty with factual-accuracy and creative-quality cases (§4.2).

**Permutation significance.** Permutation tests were computed on the original 35-case expansion subset (before the battery was expanded to 61 cases), on which the centroid achieves 80%/80%/77%. Shuffling better/worse labels 1000 times:

| Model | Observed (N=35) | Null mean | Null max | p-value |
|---|---:|---:|---:|---:|
| Snowflake | 80% | 50% | 83% | 0.003 |
| BGE-M3 | 80% | 49% | 91% | 0.053 |
| Nomic | 77% | 50% | 91% | 0.032 |

Statistically significant on Snowflake (p = 0.003) and Nomic (p = 0.032), borderline on BGE-M3 (p = 0.053). The accuracy drop from 35 to 61 cases reflects the added factual-accuracy and creative-quality cases, which are systematically harder (§4.2).

**Learning curve.** We trained centroid directions on random subsets of the battery (50 resamplings per size) and tested each on the original 35-case expansion set.

| Training pairs | Snowflake OOS | BGE-M3 OOS | Nomic OOS |
|---:|---:|---:|---:|
| 5 | 59% | 67% | 60% |
| 10 | 61% | 78% | 62% |
| 20 | 68% | 82% | 71% |
| 30 | 72% | 84% | 73% |
| 50 | 77% | 85% | 77% |
| 70 | 80% | 80% | 77% |

BGE-M3 is the standout: 10 labeled pairs produce 78% OOS accuracy. It peaks around 30–40 pairs and slightly declines at 70, possibly because adding warmth-type cases (which oppose the firmness direction) pulls the centroid away from the expansion set's anti-sycophancy-heavy distribution.

### 4.2 Expanded Evaluation: 61 Cases Across Five Categories

To characterize the centroid's per-category accuracy, we created 26 additional expansion cases (10 factual accuracy, 8 creative quality, 8 nuance/context) beyond the original 35, for a total of 61 held-out cases.

| Category | Snowflake | BGE-M3 | Nomic | n |
|---|---:|---:|---:|---:|
| Anti-sycophancy | 93% | 93% | 93% | 15 |
| Nuance/context | 77% | 77% | 77% | 13 |
| Conciseness | 60% | 80% | 80% | 5 |
| Creative quality | 77% | 69% | 54% | 13 |
| Factual accuracy | 67% | 60% | 33% | 15 |
| **Overall centroid** | **77%** | **75%** | **66%** | 61 |
| **Overall probe** | **80%** | **87%** | **72%** | 61 |

Anti-sycophancy is the strongest signal across all models (93%). The direction reliably detects when a response pushes back versus sycophantically agrees. Factual accuracy is the systematic weakness: Nomic scores 33% (worse than coin flip) — the direction cannot fact-check. Responses with wrong dates or cities don't differ semantically from correct ones in the ways the centroid captures.

**Consistent errors across models.** Three cases fail on all three models: `cr_anger_without_anger` (the "showing not telling" literary technique), `cr_loneliness_simile` (an original metaphor vs a cliché), and `fc_shakespeare_year` (1564 vs 1642 — both sound authoritative). Failures cluster where quality depends on factual correctness, literary craft, or social inference — things the embedding cannot access.

### 4.3 The Quality Direction Is Unnameable

**Vocabulary projection (TEST 2A).** We projected all 10,000 words in the Brown corpus onto the centroid direction and measured cosine similarity. The maximum absolute cosine across all words:

| Model | Max |cos| | Word |
|---|---:|---|
| Snowflake | 0.12 | — |
| BGE-M3 | 0.25 | — |
| Nomic | 0.25 | — |

No English word exceeds cosine 0.30 with the quality direction on any model.

**Phrase projection (TEST 2B).** We tested 46 multi-word quality phrases (e.g., "genuinely helpful and accurate," "not sycophantic or pandering") and 22 single evaluative words. If multi-word phrases can compose meaning more specifically than single words, they might come closer to the quality direction.

No phrase exceeds cosine 0.30. The maximum is BGE-M3 "not helpful" at 0.244. Phrases are marginally better than their single-word components (+0.01 to +0.04 max improvement) — the combination of words does not escape the geometric constraint.

**Anchor word cosines.** Even "careful," the best single discriminator, has near-zero cosine with the quality direction (+0.005 to +0.17). Raw "good" is negatively correlated (−0.02 to −0.05). The logistic probe direction (from the classifier) is highly aligned with the centroid (cosine 0.95–0.99), confirming they find the same geometric structure. The quality signal exists in these embeddings, but it lies in a region no English word or phrase points toward.

### 4.4 Quality Is Multi-Dimensional

**Firmness and warmth are anti-correlated quality dimensions.** The quality direction trained on firmness-type cases (50 original) and the direction trained on warmth-type cases (20 warmth) point in opposite directions:

| Model | Cosine(firmness, warmth) |
|---|---:|
| Snowflake | −0.35 |
| BGE-M3 | −0.39 |
| Nomic | −0.42 |
| Gemini | −0.49 |

A firmness-trained direction scores 15–25% on warmth cases (worse than coin flip). A warmth-trained direction scores 8–24% on firmness cases. Leave-one-category-out confirms: holding out warmth_appropriate cases, the remaining battery gets 0% on them (all 3 models).

This means quality is not one direction — it is at least two opposing dimensions in embedding space. A response that is "good" by being firm and corrective sits in a fundamentally different region than a response that is "good" by being warm and supportive. The full-battery centroid works (66–77% on 61 OOS cases) because it is a compromise between these opposing forces, weighted by training-set composition (50 firmness, 20 warmth).

**Multi-direction approaches do not reliably beat the single centroid.** We tested: (a) max-of-two (score on both firmness and warmth directions, use whichever gives a larger margin) — 69–83% OOS; (b) prompt-based routing (classify the prompt, apply the corresponding direction) — 51–77% OOS. Only BGE-M3 benefits from max-of-two (83% vs 80%). The single centroid is surprisingly robust as a one-size-fits-all compromise.

### 4.5 Robustness Battery

**Bootstrap stability.** We resampled the 70-pair battery with replacement 200 times. Mean cosine between each bootstrap direction and the full-data direction: 0.77 (Snowflake), 0.83 (BGE-M3), 0.79 (Nomic). Minimum: 0.63–0.66. The direction points roughly the same way regardless of which subset you draw.

**K-fold cross-validation.** 5-fold CV on the battery yields 66–73% with high variance across folds (std 5–15%). Lower than OOS expansion accuracy (66–77% on 61 cases), likely because within-battery cases include harder contrasts that the expansion set does not emphasize.

**Disjoint split stability.** We split all 131 cases (70 battery + 61 expansion) into two non-overlapping random halves 50 times.

| Model | Mean cosine | Cross-accuracy |
|---|---:|---:|
| Snowflake | +0.37 | 75% |
| BGE-M3 | +0.57 | 78% |
| Nomic | +0.38 | 72% |

Directions are moderately aligned (cosine 0.37–0.57) but far from identical. Cross-accuracy (train on one half, test on the other) is 72–78%.

**Reverse validation.** Training on expansion (61) and testing on battery (70): 63–67%. The expansion-trained direction is less informative about battery cases than vice versa. The two datasets find related but distinct quality directions, consistent with the multi-dimensional finding.

**Pooled direction.** Using all 131 cases improves to 84–89% on battery, 85–95% on expansion. More data sharpens the direction.

**Length confound ruled out.** Better responses are not systematically longer (mean character difference: +0 battery, −8 expansion). A length-only classifier scores 49% on battery, 37% on expansion. Cosine between quality direction and length-encoding direction: −0.05 to −0.16. Removing length from embeddings before computing the centroid does not change accuracy.

**PCA dimensionality (TEST 3A).** Quality is high-dimensional: 51 principal components needed for 80% of variance. The centroid (88–93% in-sample) substantially outperforms PCA top-k (41–77%). The centroid captures a specific direction that PCA's top components miss.

**Gameability (TEST 6A–B).** Six mechanical modifications × 30 cases × 3 models: padding, hedging, bullet points, repetition, formalization, lengthening. Total flip rate: 1.9%. No modification reliably inflates the score. The centroid is robust to surface manipulation.

### 4.6 Meta-Validity Battery

We ran five meta-validity tests across all three local models to check whether the centroid's accuracy reflects genuine quality discrimination or methodological artifacts.

**M1: Prompt leakage.** Does the prompt's presence in the embedding influence the score? We compared three embedding formats: full (`User: {prompt}\nAssistant: {response}`), assistant-only (`Assistant: {response}`), and response-only (bare response text).

| Format | Snowflake | BGE-M3 | Nomic |
|---|---:|---:|---:|
| Full | 68.6% | 77.1% | 71.4% |
| Assistant-only | 74.3% | 82.9% | 78.6% |
| Response-only | 80.0% | 75.4% | 80.0% |

Response-only BEATS full-format on Snowflake (+11.4pp) and Nomic (+8.6pp). On BGE-M3, assistant-only is best (82.9%). The prompt adds noise rather than signal — the quality information is in the response, not the prompt context. This is consistent with the centroid capturing response-level properties rather than prompt-response interaction.

**M2: Label-flip symmetry.** We swapped all better/worse labels and recomputed the centroid. If the method has structural artifacts, the flipped direction should differ from the simple negation of the original.

All three models: cosine(original, flipped) = −1.000. Asymmetry: 0.0%. The method is perfectly symmetric — flipping labels produces exactly the opposite direction. No structural artifact.

**M3: Leave-one-out stability.** We removed each of the 70 training cases one at a time and measured how the centroid changes.

Mean cosine with the full-data direction: 0.995–0.997 across models. Maximum single-case accuracy impact: ±5–7 percentage points. No single case dominates the direction. The centroid is not a knife-edge balanced on one influential example.

**M4: Edge cases and absolute scoring.** We tested whether degenerate responses (empty string, single word "ok," random characters, pure refusal, lorem ipsum, off-topic, emoji-only) score below real better responses in absolute dot product.

| Model | Degenerate above real_better |
|---|---:|
| Snowflake | 7/7 (all degenerate above) |
| BGE-M3 | 1/7 (only "refuse") |
| Nomic | 2/7 |

On Snowflake, every degenerate response outscores real better responses in absolute terms. On BGE-M3, only one does. **Absolute scoring is meaningless** — the dot product ranks degenerate text above real text on some models. The centroid is a pairwise comparator: it tells you which of two responses to the same prompt is better, not whether a response is good in isolation.

**M5: Training influence asymmetry.** Do warmth cases and firmness cases contribute equally to the centroid? We measured the leave-one-out accuracy impact grouped by case type.

Warmth cases are more influential on Snowflake (p = 0.0002) — removing a warmth case changes the direction more than removing a firmness case. BGE-M3 (p = 0.856) and Nomic (p = 0.982) show no asymmetry. The Snowflake result likely reflects the 50:20 firmness:warmth ratio — the minority class has more per-case leverage.

### 4.7 Score Diversity and Style Tolerance

**Design.** To test whether the centroid rewards one writing style over others, we generated responses in five distinct styles (concise, warm, technical, casual, formal) to each of 10 diverse prompts, producing 50 scored responses.

**Results.** Between-prompt variance exceeds within-prompt variance on all three models:

| Model | Variance ratio (between/within) |
|---|---:|
| Snowflake | 1.8 |
| BGE-M3 | 2.7 |
| Nomic | 2.3 |

The centroid scores the PROMPT more than the STYLE — what you say matters more than how you say it. No single style dominates across models. However, formal and technical styles consistently score lowest, suggesting a real bias toward accessible, direct language. This is a limitation for applications where formal register is appropriate.

### 4.8 Additional and Frontier Models

**Gemini Embedding 001 (3072d, API-gated).** We replicated the centroid experiment on Gemini using the same battery, tested on the original 35-case expansion subset (API rate limits precluded re-running on the full 61). Local model numbers shown for comparison on the same N=35 subset.

| Method | Gemini | Snowflake | BGE-M3 | Nomic |
|---|---:|---:|---:|---:|
| Centroid OOS (N=35) | **86%** | 80% | 80% | 77% |
| Best anchor word OOS | 66% (careful) | 56% | 53% | 64% |
| Raw good/bad OOS | 40% | 23% | 29% | 26% |

Gemini achieves the highest centroid OOS accuracy of any model tested. The learning curve is steeper: 80% with 20 pairs, 86% with 50. Per-category on the original 35 expansion cases: anti-sycophancy 93%, nuance 100%, creative quality 80%, factual accuracy 80%, conciseness 60%.

The same structural patterns hold on Gemini, amplified: firmness-warmth anti-correlation is the strongest (cosine −0.49), cross-type transfer fails most dramatically (firm→warmth 5%, warm→firm 12%), and all anchor words have near-zero cosine with the quality direction (max |cos| = 0.10). The frontier model shows the same geometry as local models — quality is multi-dimensional, words don't point at it, examples find it — but with higher signal-to-noise.

**Jina Embeddings (1024d, API-based).** We tested two Jina models on the full protocol: centroid trained on the 70-pair battery, tested on all 61 expansion cases, with permutation significance computed on the same 61 cases.

| Model | Centroid OOS (N=61) | Probe OOS | Permutation p | LOO |
|---|---:|---:|---:|---:|
| Jina v5-text-small | **80%** | 80% | **0.003** | 77% |
| Jina v3 | 77% | 77% | 0.008 | 70% |

Jina v5-text-small achieves the highest OOS accuracy of any model on the full 61-case expansion — 80%, with strong permutation significance (p = 0.003). Per-category: anti-sycophancy 93%, nuance 85%, creative quality 77%, factual accuracy 73%, conciseness 60%. The factual accuracy improvement (73% vs 33–67% on local models) is notable: Jina's embeddings preserve more factual-correctness signal. Both Jina models show the same word-failure pattern: Good/Bad scores 33–36%, Careful/Reckless scores 47–49%.

**Additional local models (TEST 1C).** Three models beyond the core three:

| Model | OOS accuracy | Permutation p |
|---|---:|---:|
| GTE-base | 65.6% | 0.540 |
| E5-base-v2 | 72.1% | 0.675 |
| mxbai-embed-large-v1 | 70.5% | 0.335 |

Above chance but all fail permutation significance. The quality signal exists in these models but is weaker than in the core three. Scale does not predict performance within 33M–600M: Qwen3-Embedding-0.6B at 600M performs comparably to 33M-parameter models.

### 4.9 Why Words Fail: A Systematic Investigation

The centroid succeeds with 66–80% OOS accuracy across five models (86% on Gemini). Raw "good/bad" scores 26%. Why is there such a dramatic gap between learning quality from examples versus naming it with words? We conducted 28 experiments on word-based approaches to diagnose the failure.

#### 4.9.1 The Warmth Bias

Raw "good/bad" fails because the "good" direction in embedding space tracks warmth/agreeableness rather than quality. A content split defined by case design (not model geometry) reveals the mechanism:

| Model | "Good" on firmness cases | "Good" on warmth cases |
|---|---:|---:|
| Snowflake | 48% | 60% |
| BGE-M3 | 16% | 85% |
| Nomic | 12% | 80% |
| Gemini | 26% | 95% |

On three of four models, "good" accuracy on warmth cases is 4–7× its firmness-case accuracy. When being warm and being correct align, "good" works. When they conflict, "good" picks the warm response — even when it's wrong.

The mechanism is visible in neighborhood composition: "good"'s top-30 non-synonym neighbors are 40% warmth+emotion words (cross-model average) vs 12% competence+restraint, while "careful" shows the inverse (30% competence+restraint vs 18% warmth+emotion). The neighborhood warmth fraction predicts scoring bias: r = +0.65 to +0.69 across three models (n = 7 axes).

**Synonym clusters confirm the trap.** All 30 sense-disambiguated synonym clusters of "good" — including "excellent," "superior," "valid," and "compassionate" — sit in the same positive-valence neighborhood. The problem is not that "good" is overloaded; it is that embedding models organize all positive-valence language in the same region, compressing distinct evaluative dimensions into a single warmth-dominant direction.

#### 4.9.2 No Word Escapes

"Careful" is the best single-word discriminator, the only evaluative term geometrically independent of "good" in score-delta space (r = −0.11 to +0.09, all n.s., vs r = 0.55–0.93 for other children). But it still falls short of statistical significance on 2 of 4 models:

| Model | Careful accuracy | 95% CI lower |
|---|---:|---:|
| Snowflake | 56% | — |
| BGE-M3 | 53% | — |
| Nomic | 64% | 54% (significant) |
| Gemini | 74% | 62% (significant) |

Across 45+ evaluative terms tested, ~80% share the warmth bias. Warmth-independence is rare and empirically identifiable: all 9 warmth-independent terms share restraint/discipline semantics. But this observation does not function as a predictive rule — three prediction tests returned null (pooled: 5/13 independence predictions correct, 38%).

**Subtracting warmth leaves noise.** If the quality signal is merely hidden behind a warmth confound, removing the warmth component should reveal it. We projected "good" orthogonally to the warmth direction. The residual scores at chance — the quality signal is not hiding behind warmth; it is distributed across narrower semantic dimensions that no single word accesses.

**Richer anchor text does not help.** Multi-word phrases, full sentences, and explicitly anti-sycophantic formulations all fail. All "good" variants remain warmth-biased regardless of length or specificity. Longer anchors degrade clean single-word axes. Cross-concept compound strings dilute rather than combine signal. The failure is geometric, not a word-length artifact.

#### 4.9.3 Supporting Evidence from Earlier Experiments

The 28 word-based experiments, while ultimately a negative result for word-based scoring, provide supporting context for the centroid finding:

**Controlled axis validation (Gemini, multi-anchor-sentence axis):** 70.5% overall on 61 statement pairs, but 0% on sycophancy — sycophantic text uses quality-signaling words instrumentally, and embeddings cannot distinguish sincere from performative use. This was the first demonstration that evaluative geometry exists but has structured blind spots.

**HH-RLHF disagreement audit:** On 500 HH-RLHF pairs, the embedding agreed with labels 55.8% (z = 2.59, p = 0.009). Of 231 disagreements, 63 were classified EMBEDDING_RIGHT (embedding preferred the genuinely better response), 45 HH_RIGHT, 123 EXCLUDE (both responses unsuitable). Corrected gradeable agreement: 88.1%. However, the audit was not blind — a single reviewer graded all cases knowing which the embedding preferred. This result is retained as supporting evidence for the claim that embedding geometry captures evaluative structure, not as a headline finding.

**Objective reranking (Gemini):** On small frozen suites with verifiable end metrics — code (83.3%), math (100%), tool interpretation (87.5%) — Gemini's embedding-based selection beats random and length baselines. OSS models score at or near chance. This establishes that frontier embedding models can do objective answer selection, supporting the centroid's claim that embeddings encode quality.

**Osgood's three dimensions:** All EPA (Evaluation, Potency, Activity) composites fail. Individual Potency pairs appeared promising on the original battery but collapsed under rebalancing — the original 50-case battery was 64% firmness-biased, inflating Potency scores.

**Five-term tree decomposition:** Five principled quality terms — careful, honest, helpful, thorough, restrained — scored independently with OR logic reach 89–94% OOS across all three local models. However, with five independent binary classifiers under OR logic, the chance-level expectation is 1 − 0.5^5 = 96.875%. On HH-RLHF where all five individual axes score at chance (47–55%), the tree still reaches 86–97% — matching the mathematical baseline. The tree's 89–94% on our battery cannot be clearly distinguished from threshold inflation. The centroid's 66–77% on local models (86% on Gemini) without threshold inflation is the more honest measure.

### 4.10 Known Limitations

**No cross-dataset transfer (TEST 1A).** A centroid trained on our battery achieves chance accuracy on SHP and UltraFeedback datasets. A centroid trained on SHP achieves 66–77% on our battery (within-dataset OK, cross-dataset zero transfer). The quality direction is dataset-specific — different preference datasets encode different quality standards.

**Cannot fact-check.** Factual accuracy is the systematic weakness: 33–67% across models. The centroid cannot distinguish "the Berlin Wall fell in 1989" from "the Berlin Wall fell in 1991" — both are semantically authoritative.

**Sycophancy blindness.** While anti-sycophancy is the centroid's strongest category (93%), word-based axes cannot distinguish sincere from performative quality language. A positive-word-count baseline achieves 85% on the 20-case anti-sycophancy demonstration, so word-based anti-sycophancy detection is confounded with lexical valence.

**Formal/technical style penalty.** Score diversity testing shows formal and technical writing styles consistently score lowest across models. This is a real bias, not a confound — the centroid penalizes formal register.

**Battery composition bias.** The 70-pair training battery is 71% firmness cases (50/70). The centroid is therefore a firmness-weighted compromise. A differently composed training set would produce a different quality direction. On Gemini: 96% on firmness cases, 50% on warmth cases.

**Absolute scoring meaningless.** Degenerate responses (empty, random, off-topic) can outscore real better responses in absolute dot product, especially on Snowflake (7/7 degenerate above). The centroid is a pairwise discriminator only.

**Detection vs training gap.** All results evaluate pairwise discrimination of pre-written responses. Using the centroid as a training signal (for RL or DPO) is a different problem. A score that correctly identifies the better of two responses does not guarantee that training toward that score produces desired behavior. The model might learn to satisfy the easiest dimension at the expense of harder ones.

**Weaker models fail permutation.** Three additional models (GTE, E5, mxbai) show above-chance accuracy (66–72%) but all fail permutation significance (p = 0.335–0.675). The quality signal may be present but too noisy to extract reliably below some capability threshold.

**Single frontier model.** All frontier results use Gemini Embedding 001. Whether the effect generalizes across frontier embedding providers is untested.

---

## 5. Discussion

### 5.1 Why Examples Succeed Where Words Fail

The supervised centroid resolves the paradox that motivates this investigation. Embedding models clearly encode evaluative structure — the centroid achieves 66–80% OOS accuracy across five models on 61 cases (86% on Gemini at N=35). Yet no English word, including "good" and all 30 of its sense-disambiguated synonym clusters, points at this structure. The quality direction has near-zero cosine with every anchor word tested.

Words fail because they are ambiguous. "Good" means 30 different things, and its embedding is the weighted average of all of them — dominated by the warmth/agreeableness sense because that is statistically the most frequent context. Embedding models organize all positive-valence language in the same region, compressing distinct evaluative dimensions into a single warmth-dominant direction. No amount of lexical creativity escapes a geometric fact about how these models represent positive affect.

Examples escape the trap. When we provide 70 pairs of better and worse responses, we are not naming the quality — we are pointing at it. The centroid captures whatever makes the better ones better. This "whatever" includes dimensions we have no single words for: the quality of being firm-but-not-dismissive, of being warm-but-not-sycophantic, of addressing what someone actually needs rather than what they literally asked. These are real qualities that human annotators recognize, that appear in training data distributions, and that embedding models encode — but that no individual English word reliably points toward.

The anti-correlation between firmness and warmth quality directions demonstrates this concretely. A response that is good-by-being-firm sits in one region; a response that is good-by-being-warm sits in an opposing region. The word "good" cannot point at both. But a centroid computed from examples of both types produces a compromise direction that captures enough of each to work (66–80% on 61 cases across five models). The compromise is not optimal — it loses the extremes of each dimension — but it is a direction that no word could have specified, because no word means "the shared quality of things that are good in conflicting ways."

This reframes the relationship between alignment training and embedding scoring. RLHF and DPO define "good" through examples — preference pairs showing which responses are better. They do not define "good" verbally. The centroid does the same thing in embedding space. The difference is cost: RLHF requires training a reward model; the centroid requires computing a mean of embeddings.

### 5.2 Properties of the Discriminator

The centroid-based discriminator differs from existing evaluative signals in several properties:

**Determinism.** The same input produces the same scores every time. LLM judges are non-deterministic even at temperature zero. Human annotators disagree at 20–30% rates. A deterministic signal eliminates this noise floor.

**Inspectability.** The projection is mechanical, not inferred. When a response scores higher on the centroid, it literally projects further along the quality direction in embedding space. There is no hidden reasoning to audit, no prompt sensitivity, no chain-of-thought to verify.

**Cost.** One embedding call per response. No model inference, no GPU for local models, no API calls beyond the initial embeddings. BGE-M3 processes the full battery in seconds on CPU.

**Pairwise only.** The centroid does NOT provide absolute quality scores. It ranks two responses to the same prompt. This is a fundamental limitation — the signal is comparative, not evaluative in isolation.

### 5.3 Entanglement as Mechanism

The Value Entanglement finding (Cho et al., 2026) — that moral, grammatical, and economic "good" are conflated in embedding representations — is typically characterized as a defect. We argue it is the mechanism by which the centroid captures multifaceted quality. The desirable properties of a language model — reasoning quality, honesty, safety, clarity — share geometric structure in embedding space because they are statistically correlated in human evaluative language.

However, the empirical picture is nuanced. The "good" direction in current embedding models does not encode genuine multidimensional quality — it encodes warmth. The entanglement is real, but the dominant entangled dimension is agreeableness, not quality. The centroid sidesteps this by defining quality from examples rather than words, accessing the entangled structure through a different entry point.

### 5.4 Robustness to Overoptimization (Theoretical)

Narrow alignment objectives are susceptible to reward hacking (Gao et al., 2022). Maximizing an "honesty" reward can produce cruel pedantry; maximizing "helpfulness" can produce sycophancy. The centroid, which encodes a compromise between multiple quality dimensions, may be structurally resistant: overoptimizing any single component moves the output off the general quality direction rather than further along it.

This is a theoretical argument, not empirically validated. The firmness-warmth anti-correlation provides partial support: a model that maximizes firmness at the expense of warmth would score lower on the centroid than one that balances both. But whether this self-correction survives actual RL training dynamics is an open question.

### 5.5 Toward a Standardized Method

**Recommended pipeline.** If the practitioner can provide 10+ labeled preference pairs representative of the quality they want to measure:

1. Embed all "better" responses, compute their centroid. Same for "worse" responses. The quality direction is the normalized difference.
2. For a new response pair, project both onto the quality direction. The higher projection is predicted as better.
3. For BGE-M3 specifically, a logistic probe (single-layer classifier on difference vectors) substantially outperforms the centroid (87% vs 75% on the expanded evaluation).

**Word-based fallback.** When labeled examples are unavailable, the five-term tree — careful, honest, helpful, thorough, restrained — scored independently with OR logic provides a zero-shot alternative. However, the tree's 89–94% OOS accuracy must be interpreted against the mathematical baseline under OR logic (97% chance). The conservative 2-of-5 threshold (71–83% OOS) is the more honest number.

**Known constraints.** The centroid is a compromise between opposing quality dimensions, weighted by training-set composition. A firmness-heavy training set produces a direction that fails on warmth cases. Practitioners should ensure their training pairs span the quality types they care about.

### 5.6 Detection vs Training: The Open Gap

All results in this paper evaluate pairwise *discrimination* — scoring pre-written response pairs. Training uses scores as gradient signals to push model *generation* in a direction. These are different processes with potentially different outcomes.

A score that correctly identifies the better of two responses does not guarantee that training toward that score produces the behavior we want. The model might learn to satisfy the easiest dimension (e.g., sounding helpful) while ignoring harder ones (e.g., being careful). The centroid's multi-dimensional compromise structure may help — overoptimizing one dimension moves off the centroid — but this is conjecture.

We deliberately do not claim the centroid is a training signal, a reward model replacement, or a component of a training pipeline. It is a discriminator. Whether it works as a training signal is the next experimental question, not a conclusion of this paper.

---

## 6. Future Work

### 6.1 Training Experiments

The critical next step: use the centroid as a reward signal in DPO training and measure whether it produces the expected behavioral improvements. This is the test that bridges detection to training. Specifically: score candidate responses from a base model using the centroid, construct preference pairs from the scores, and fine-tune with DPO. Compare against DPO with human-labeled pairs and DPO with LLM-judge pairs.

### 6.2 Dense Process Supervision

If pairwise discrimination transfers to training, a natural extension is scoring the full context at each generation step:

$$\Phi_t = \vec{q} \cdot f(c, r_{1:t})$$

where $c$ is the conversation context and $r_{1:t}$ is the generated prefix up to step $t$. The step-level delta $\Delta_t = \Phi_t - \Phi_{t-1}$ would estimate whether step $t$ improved or degraded the trajectory, providing process-level supervision without a trained verifier.

However, two meta-validity findings complicate this: (1) absolute dot-product scores are not meaningful quality rankings (M4), and (2) the prompt context adds noise rather than signal (M1). Step-level deltas within a single trajectory use absolute scoring and include the growing prompt context — both are threat factors. Early pilot work with injected error-repair traces shows that Gemini process-aware scoring beats controls but still fails a frozen dense-localization gate. This line of work requires its own empirical validation before claims can be made.

### 6.3 Cross-Author Validation

If the centroid captures genuine quality rather than authorial idiosyncrasy, a centroid trained on Gemini-generated preference pairs should transfer to human-authored test cases and vice versa. This experiment is in progress, generating pairs via Gemini API (rate-limited to 20 requests/day on the free tier). Accumulation will require several days; analysis begins at ≥20 pairs.

### 6.4 Model Scaling and Cross-Domain

Testing additional frontier embedding models (OpenAI, Cohere, Voyage) would clarify whether the Gemini result generalizes. The open-weight Qwen3-Embedding-8B is particularly interesting: at 8B parameters it could test whether the quality signal strengthens beyond the 33M–600M range.

Cross-domain validation — testing on code quality, writing quality, reasoning quality, and summarization quality batteries — would validate whether the centroid generalizes beyond the safety/helpfulness domain.

### 6.5 Blind Adjudication

The HH-RLHF disagreement audit (88.1% corrected agreement) was conducted by a single non-blind reviewer. Multi-annotator blind adjudication is needed for publishable claims on that result.

---

## 7. Conclusion

Three findings, at different levels, emerge from this investigation.

**Words fail.** No single evaluative word — including "good" and all 30 of its sense-disambiguated synonym clusters — produces a reliable quality signal across embedding models (33M–568M params). "Good" actively anti-signals on cases where warmth and correctness conflict (26% on balanced battery). All positive-valence words sit in the same warmth-dominated neighborhood; no lexical strategy escapes a geometric fact about how these models organize positive affect.

**Examples succeed.** The supervised centroid direction achieves 66–80% out-of-sample accuracy across five embedding models on 61 held-out cases and 86% on Gemini (35 cases), with statistical significance (p = 0.003–0.053). BGE-M3 reaches 78% with just 10 labeled pairs. The quality direction has near-zero cosine with every English word and phrase tested. The signal is real, stable under bootstrap resampling (cosine 0.77–0.83), and survives nine validation tests including confound checks, gameability probes, label-flip symmetry, and leave-one-out stability.

**Quality is multi-dimensional.** Firmness and warmth quality directions are anti-correlated (cosine −0.35 to −0.49). No single direction — word-based or supervised — captures both simultaneously. The centroid works as a weighted compromise between opposing dimensions.

These results establish the supervised centroid as a pairwise quality discriminator: cheap (one embedding call per response), deterministic, not gameable, and accessible with as few as 10 labeled pairs. It provides a signal that no word can name, by pointing at quality through examples rather than defining it verbally.

What the centroid is NOT, based on current evidence: it is not a training signal (untested), not an absolute quality scorer (M4 rules this out), not a replacement for reward models or LLM judges (the 66–80% accuracy range means 20–34% of pairs are misjudged), and not a cross-dataset generalizer (chance accuracy on SHP/UltraFeedback). These are the open questions for future work.

The broader implication: embedding models encode richer evaluative structure than their vocabulary can access. Words are ambiguous; examples are specific. The quality direction exists in the geometry — you just can't get there by naming it.

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

Gao, L., Schulman, J., & Hilton, J. (2022). Scaling Laws for Reward Model Overoptimization. *arXiv:2210.10760*.

Goddard, C. & Wierzbicka, A. (2014). *Words and Meanings: Lexical Semantics Across Domains, Languages, and Cultures*. Oxford University Press.

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

Wierzbicka, A. (1972). *Semantic Primitives*. Athenäum.

Xu, Y., Chakraborty, T., Kıcıman, E., Aryal, B., Rodrigues, E., Sharma, S., Estevao, R., de Luis Balaguer, M. A., Wolk, J., Padilha, R., Nunes, L., Balakrishnan, S., Lu, S. & Chandra, R. (2025). RLTHF: Targeted Human Feedback for LLM Alignment. *arXiv:2502.13417*.

Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*.
