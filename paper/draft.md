# The Evaluative Axis: Embedding Geometry as a General-Purpose Quality Signal for Language Model Alignment

Robin Gattis

June 2026

---

## Abstract

We propose using the evaluative direction in sentence embedding space — the geometric axis corresponding to "good" versus "bad" — as a direct reward signal for language model alignment. This approach requires no trained reward model, no human preference labels, and no LLM-as-judge inference. The theoretical foundation rests on three established findings: that evaluation is the primary dimension of human semantic judgment (Osgood et al., 1957), that this dimension is recoverable from embedding geometry via semantic projection (Grand et al., 2022; Kozlowski et al., 2025), and that different senses of "good" — moral, grammatical, epistemic, economic — are geometrically entangled in embedding space (Cho et al., 2026), meaning a single axis simultaneously captures helpfulness, honesty, safety, and quality.

In a full audit of 231 cases where our embedding-axis scoring disagreed with Anthropic's HH-RLHF preference labels, we find that the embedding preferred the better response in 60% of gradeable disagreements, yielding a corrected agreement rate of 83–88% — substantially higher than the 54% raw overlap with the 2022 labels. In several disagreement cases, the embedding axis preferred responses consistent with modern AI safety norms over responses that reflected older labeling policies, suggesting the geometric signal anticipates where human preference converges rather than merely approximating existing annotations.

We argue that evaluative entanglement — the conflation of different kinds of "good" — is not a defect of embedding representations but the mechanism by which a single geometric direction captures the general quality axis that alignment training aims to specify. The resulting signal is deterministic, costs a fraction of a cent per evaluation, and scales to billions of documents, with potential applications ranging from alignment reward signals to pretraining data curation.

---

## 1. Introduction

Training language models to be helpful, honest, and safe currently requires one of two expensive approaches: collecting human preference labels (RLHF; Ouyang et al., 2022) or running a large language model as a judge (RLAIF; Bai et al., 2022). Both are costly, and both introduce noise — human annotators disagree with each other at substantial rates, and LLM judges can be gamed, are non-deterministic, and tend to rationalize rather than flag ambiguity.

We propose a simpler approach. Define a direction in embedding space that corresponds to the evaluative axis — the "good/bad" direction — by averaging the embeddings of a small set of positive and negative anchor sentences. Project any text onto this direction. The dot product is the quality score.

This idea rests on a chain of three established findings:

**Finding 1: Evaluation is primary.** Osgood, Suci, and Tannenbaum (1957) demonstrated through factor analysis of bipolar adjective ratings across dozens of cultures that human semantic judgment reduces to three dimensions: evaluation (good/bad), potency (strong/weak), and activity (active/passive). Evaluation consistently explains the most variance and is cross-culturally universal. It is not one dimension among many — it is the primary organizing axis of human meaning-making.

**Finding 2: This structure is preserved in embeddings.** Kozlowski, Dai, and Boutyline (2025) confirmed that Osgood's three dimensions exist in modern LLM embedding geometry. Projections of words onto directions defined by antonym pairs correlate highly with human ratings and reduce to a low-dimensional subspace resembling patterns from human survey data. Grand et al. (2022) demonstrated more broadly that semantic projection onto antonym-defined directions recovers context-dependent human knowledge that simple cosine similarity misses.

**Finding 3: Different kinds of "good" are entangled.** Cho, Li, and Leshinskaya (2026) found that LLMs conflate moral, grammatical, and economic senses of "good" in their internal representations, with moral valence influencing assessments of grammaticality and economic value. They characterized this as a problem requiring ablation. We argue the opposite: for a quality signal intended to capture everything desirable about a model's output, entanglement is the mechanism. The reason a single evaluative axis captures helpfulness, honesty, safety, and correctness simultaneously is that these properties are geometrically correlated along the evaluation direction. You want an AI to be good — in every sense of the word.

Biological learning systems include a dense, general-purpose evaluative signal — dopaminergic reward — that operates at each step of behavior, precedes deliberative analysis, and generalizes across domains. Autoregressive language models lack an analogous mechanism during generation; evaluation occurs only at training time, compressed into a scalar loss at the end of a sequence. The evaluative axis provides a candidate for this missing architectural component: a deterministic, near-zero-cost quality signal computable at each generation step, with no model inference required.

The gap this paper addresses: these findings establish that the evaluative direction exists, preserves human judgment, and captures multiple quality dimensions. No published work tests whether this direction functions as a direct reward signal for alignment — without training a classifier, without labeled preference data, and without LLM inference.

### 1.1 Contributions

1. We demonstrate that evaluative-axis projection agrees with corrected human preference at 83–88% on gradeable cases, substantially higher than raw overlap with the noisy 2022 HH-RLHF labels suggests.

2. We show that 10 quality axes constructed from unrelated domains — code, cooking, parenting, medicine, music, writing, ethics, engineering, teaching, friendship — converge to the same geometric direction in embedding space, providing structural evidence for a universal evaluative axis.

3. We document cases where embedding-axis scoring anticipates later preference norms — preferring responses consistent with modern AI safety standards over responses that reflected older labeling policies.

4. We identify the embedding axis as a label-noise detector, catching fabricated persona claims, doxxing compliance, misinformation, and harmful content that HH-RLHF rewarded.

4. We provide evidence that evaluative entanglement — the conflation of different types of "good" — is not a representational defect but a feature that enables a single geometric direction to serve as a general-purpose quality signal.

5. We characterize the known limitations: the axis conflates quality with positive tone for sycophantic text (0% accuracy on controlled sycophancy pairs) and cannot distinguish appropriate from misplaced confidence (40% on honesty-hedging pairs). These are inherent limits of any surface-text evaluation and should be documented, not hidden.

---

## 2. Related Work

### 2.1 Human Preference-Based Alignment

**RLHF** (Ouyang et al., 2022) trains a reward model from human preference labels, then optimizes a language model against it. The cost of human annotation — estimated at ~$100 per annotation-hour, with 600 high-quality annotations costing ~$60,000 (RLTHF, 2025) — is the primary scalability bottleneck.

**DPO** (Rafailov et al., 2024) eliminates the reward model by directly optimizing from preference pairs, but still requires those pairs to exist. Our approach provides a way to generate preference pairs cheaply: score candidate responses with the embedding axis, construct pairs from the scores.

**RLAIF / Constitutional AI** (Bai et al., 2022) replaces human annotators with an LLM judge. Cheaper than RLHF but still requires full model inference for every judgment, is non-deterministic, and can be gamed through adversarial reasoning. Our embedding signal is deterministic and not gameable in the same way — there is no chain of reasoning to exploit.

### 2.2 Embedding-Based Reward Signals

**Sun et al. (2025)** showed that reward models can be trained on pre-computed embeddings rather than requiring full model inference, dramatically reducing compute costs. They still train a classifier on labeled preference data; our approach skips the classifier entirely and uses the embedding geometry directly. Their work validates the premise — embeddings contain enough information for reward modeling — while leaving the stronger claim undemonstrated.

**Legend** (2024) uses representation engineering to find a safety direction in the target model's own activation space, then annotates preference margins based on distances along that direction. Key differences: Legend requires inference through the model being trained, focuses narrowly on safety, and annotates margins on existing preference data. Our approach uses a cheap external embedding model, targets general evaluation rather than safety alone, and generates preference signals from scratch.

### 2.3 Semantic Geometry

**Bolukbasi et al. (2016)** demonstrated that semantic dimensions (gender, in their case) are linearly encoded in embedding space and recoverable via difference vectors between paired concepts. Our axis construction method is the same technique applied to evaluative content.

**Grand et al. (2022)** showed that semantic projection recovers context-dependent human knowledge from word embeddings, published in Nature Human Behaviour. Our method is semantic projection applied to the evaluative domain at the sentence level.

**Kozlowski et al. (2025)** confirmed that Osgood's evaluation/potency/activity dimensions exist in LLM embeddings and that semantic features are entangled — shifting along one direction causes proportional shifts on geometrically aligned features. This entanglement is the mechanism by which our single evaluative axis captures multiple quality dimensions.

### 2.4 Representation Engineering and Activation Steering

**Zou et al. (2023)** showed that high-level concepts like honesty and safety are linearly represented as directions in a model's internal activation space and can be used to steer behavior at inference time. Our work is the external analog: instead of finding evaluative directions inside the model being trained, we use an external embedding model's evaluative geometry as a scoring function. Theirs is a steering technique; ours is a reward signal.

**Activation steering** (2024–2026) has grown into a substantial body of work on inference-time behavioral control by adding direction vectors to model activations. Conceptually aligned — both exploit the linear structure of evaluative dimensions — but serving a different function.

### 2.5 Verifiable Rewards

**DeepSeek GRPO** (2025) demonstrated that reinforcement learning with verifiable rewards — deterministic checks that confirm whether output is correct — can train strong reasoning models without learned reward models. This works for domains with verifiable answers (math, code) but cannot evaluate open-ended helpfulness, honesty, or safety. Our embedding axis is a potential bridge: a deterministic signal for the open-ended evaluation domain where GRPO cannot reach.

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

### 3.3 Embedding Models Tested

- **BGE-small-en-v1.5** (BAAI): 384 dimensions, 33M parameters. Used for the full disagreement audit and multi-sensor experiments. Chosen for cost and reproducibility — runs on CPU in seconds.
- **Gemini Embedding 2** (Google): 3072 dimensions, state-of-the-art MTEB. Used for controlled axis validation. Quota limitations prevented full preference prediction experiments.

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
| EMBEDDING_RIGHT | 65 | 28.1% |
| HH_RIGHT | 44 | 19.0% |
| EXCLUDE | 122 | 52.8% |

Among gradeable disagreements: embedding preferred the better response in **65/(65+44) = 59.6%** of cases.

Corrected gradeable agreement (assuming agreement cases are correct, excluding both-bad pairs):

$$\frac{269 + 65}{269 + 65 + 44} = \frac{334}{378} = 88.4\%$$

Conservative sensitivity: 83.3% if 30% of embedding-right calls are wrong; 79.9% if 50% are wrong.

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

The 122 excluded cases fell into four categories:

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

---

## 5. Discussion

### 5.1 Entanglement as Mechanism

The Value Entanglement finding (Cho et al., 2026) — that moral, grammatical, and economic "good" are conflated in embedding representations — is typically characterized as a defect requiring correction. We argue it is the mechanism by which a single evaluative axis captures the multifaceted quality that alignment aims to specify.

The desirable properties of a language model — reasoning quality, honesty, safety, clarity — share geometric structure in embedding space. This is not representational contamination; it reflects the statistical structure of human evaluative language. Optimizing along the evaluative direction shifts the model toward multiple desirable properties simultaneously because those properties are geometrically entangled along the same axis.

This reframes the embedding axis not as a safety scorer or a preference predictor but as a general-purpose quality signal — the principal component of the multidimensional space of properties that alignment aims to specify.

### 5.2 Robustness to Overoptimization

Narrow alignment objectives are susceptible to reward hacking under overoptimization (Gao et al., 2022). Maximizing an "honesty" reward can produce pathological oversharing; maximizing "helpfulness" can produce sycophancy; maximizing "rigor" can produce verbose, pedantic output. Each narrow objective admits adversarial optima that satisfy the metric while degrading overall quality.

The evaluative axis may be structurally resistant to this failure mode. Because "good" subsumes multiple desirable properties via geometric entanglement (Cho et al., 2026), overoptimizing any single component moves the output *off* the general evaluative axis rather than further along it. A sycophantic response that maximizes helpful-sounding language at the cost of honesty will score lower on the evaluative direction because honesty is entangled with the same direction. The self-correcting property arises from the multidimensional entanglement itself: the axis is not a single narrow property but the principal component of many correlated quality dimensions.

This is a theoretical argument, not yet empirically validated, and motivates a direct comparison of reward hacking rates under narrow vs. broad axis optimization during DPO training.

### 5.3 Applications Beyond Alignment Training

If the evaluative axis reliably separates quality, several applications follow:

**Pretraining data curation**: Score every document in a pretraining corpus. Weight higher-quality text more heavily during training, or filter the bottom of the distribution. Current approaches are either heuristic (perplexity, deduplication) or expensive (LLM-as-judge). An embedding-axis score would be fast enough to run over billions of documents and would capture evaluative quality, not just surface cleanliness.

**Dataset auditing**: As demonstrated with HH-RLHF, the embedding axis catches systematic label noise — cases where annotator norms were outdated, where annotators rewarded harmful content, or where both options were unsuitable for training. This could be applied to any preference dataset before using it for training.

**Dense process supervision**: The signal is cheap enough to compute at each generation step, enabling cumulative trajectory scoring. At each step $n$, the full prefix — prompt, prior turns, reasoning trace, and partial response — is embedded and projected onto the axis. The step-level reward $\Delta_n = s_n - s_{n-1}$ estimates whether the new step improved or degraded the trajectory in context. This provides process-level supervision (Lightman et al., 2023) without a trained verifier, at the cost of one embedding call per step. The context window of the embedding model is the primary constraint on feasibility.

**Conditional training**: Score pretraining documents and prepend quality tags. The model sees everything — including bad text — but learns the association between the tag and the quality of what follows. At inference time, condition on the high-quality tag. The model understands what bad is (can recognize and avoid it) while preferring the good distribution.

### 5.3 Good As A Self-Regularizing Axis

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

### 5.4 Limitations

**Sycophancy blindness**: The axis cannot distinguish sincere quality-signaling language from performative flattery. This is inherent to any surface-text evaluation method and is the strongest argument for hybrid approaches (embedding-scored LLM critique).

**Honesty-hedging confusion**: Confident language scores higher than hedging, regardless of whether the confidence is warranted. The axis reads tone, not epistemic state.

**Negation weakness**: Embedding models historically handle negation poorly. "This is helpful" and "This is not helpful" are close in embedding space. At full-response scale this is diluted by other content, but adversarial cases exist.

**Circularity**: The embedding model was trained on human text. Its evaluative geometry reflects the statistical structure of human language use. If that structure is systematically biased, the bias propagates. The mitigation is that this is a different kind of circularity than RLAIF (amplifying one model's learned preferences) — the embedding reflects a property of language itself, not a model's fine-tuned opinions.

**Not a complete alignment solution**: The axis captures surface-text quality but cannot verify factual claims, evaluate hidden reasoning, or catch errors requiring external knowledge. It is a signal, not a solution.

### 5.5 The Grading Caveat

The full disagreement audit was not blind. A single reviewer graded all 231 cases with knowledge of which response the embedding preferred. While the patterns identified (persona honesty, doxxing compliance, misinformation) are unambiguous, a blind multi-annotator adjudication is needed for publishable claims. The sensitivity analysis (83.3% if 30% of embedding-right calls are overturned; 79.9% if 50%) provides conservative bounds.

---

## 6. Future Work

### 6.1 Intervention Test (Next Priority)

The most important missing experiment: generate multiple candidate responses per prompt, score with the embedding axis, and blind-judge whether embedding-selected outputs beat random, length, sentiment, and LLM-judge baselines. This tests the practical claim — does the signal improve output selection? — rather than adding another correlation measurement.

### 6.2 Model Scaling

All full-dataset results used BGE-small (384 dimensions, 33M parameters). The controlled validation with Gemini Embedding 2 (3072 dimensions) showed substantially better performance. Running the full preference prediction and disagreement audit with a frontier embedding model is the clearest path to stronger results.

### 6.3 Cross-Domain Validation

Current experiments focus on safety and helpfulness. Testing on code quality, writing quality, reasoning quality, and summarization quality would validate the "general evaluative signal" claim.

### 6.4 Dense Process Supervision via Cumulative Scoring

Current process reward models (PRMs) score each intermediate reasoning step, providing dense supervision rather than a single end-of-sequence reward. They improve mathematical reasoning significantly (Lightman et al., 2023), but training them is expensive — requiring either step-level human annotations or a trained verifier.

We propose cumulative embedding scoring as a cheap alternative. At each step $n$ in a reasoning chain, embed everything generated so far — the full conversation including all prior reasoning — and project onto the evaluative axis. The score $s_n$ reflects whether the cumulative output up to step $n$ is "good." The delta $\Delta_n = s_n - s_{n-1}$ indicates whether step $n$ improved or degraded the trajectory.

This provides dense supervision (one signal per step), is context-aware (a statement that's benign in isolation can lower the score in context), and costs one embedding call per step with no trained model. The critical constraint is embedding context window: BGE-small's 512 tokens is insufficient, Gemini Embedding 2's 8K tokens is workable for short reasoning chains, and the ideal version needs embedding models with context windows approaching those of generation models (100K+). Embedding context windows currently trail LLM context windows by roughly two orders of magnitude — bridging this gap would unlock the full potential of cumulative evaluation.

### 6.5 Pretraining Data Curation Pilot

Score a sample of pretraining-scale text (Common Crawl, The Pile) with the evaluative axis. Correlate scores with existing quality metrics. If the signal separates high-quality from low-quality documents at scale, the practical value extends far beyond alignment.

### 6.6 Blind Adjudication

Repeat the disagreement audit with multiple blind annotators to establish inter-annotator agreement and eliminate reviewer bias.

### 6.6 Cumulative Context Process Scoring

Test whether the evaluative axis can provide dense supervision over reasoning
or response trajectories. Collect traces with good and bad final outcomes, score
the full cumulative context after each step, and inspect whether score deltas
identify where the reasoning improves or degrades. This is the no-training
proxy for the longer-term training idea: reward reasoning streams that
decompose situations into good-making and bad-making parts before producing the
final answer.

---

## 7. Conclusion

The evaluative direction in embedding space — the geometric axis corresponding to "good" versus "bad" — encodes a general-purpose quality signal that agrees with corrected human preference at 83–88% on gradeable cases, anticipates later preference norms in some disagreements with existing labels, and costs orders of magnitude less than current alignment reward approaches.

The theoretical basis is the convergence of three lines of evidence: that evaluation is the primary axis of human semantic judgment (Osgood, 1957), that this axis is preserved in embedding geometry (Kozlowski et al., 2025), and that different senses of "good" are entangled in that geometry (Cho et al., 2026) — making a single direction capture the multifaceted quality that alignment aims to specify.

The signal has known limitations — sycophancy blindness, negation weakness, inability to verify factual claims — and is not a complete alignment solution. But as a cheap, deterministic, scalable evaluative signal that can serve as a reward component, a data quality filter, a label-noise detector, or a pretraining data curator, it fills a gap that existing approaches leave open.

---

## References

Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.

Bolukbasi, T., et al. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. *NeurIPS 2016*.

Cho, S. H., Li, J., & Leshinskaya, A. (2026). Value Entanglement: Conflation Between Different Kinds of Good In (Some) Large Language Models. *arXiv:2602.19101*.

Grand, G., Blank, I. A., Pereira, F., & Fedorenko, E. (2022). Semantic projection recovers rich human knowledge of multiple object features from word embeddings. *Nature Human Behaviour*, 6(7), 975–987.

Kozlowski, A. C., Dai, C., & Boutyline, A. (2025). Semantic Structure in Large Language Model Embeddings. *arXiv:2508.10003*.

Legend (2024). Leveraging Representation Engineering to Annotate Safety Margin for Preference Datasets. *arXiv:2406.08124*.

Osgood, C. E., Suci, G. J., & Tannenbaum, P. H. (1957). *The Measurement of Meaning*. University of Illinois Press.

Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.

Rafailov, R., et al. (2024). Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. *NeurIPS 2023*.

Sun, H., Shen, Y., Ton, J.-F., & van der Schaar, M. (2025). Reusing Embeddings: Reproducible Reward Model Research in Large Language Model Alignment without GPUs. *ICML 2025*.

Zou, A., et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. *arXiv:2310.01405*.
