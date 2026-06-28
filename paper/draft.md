# The Evaluative Axis: Embedding Geometry as a General-Purpose Quality Signal for Language Model Alignment

Robin Gattis

June 2026

---

## Abstract

Evaluation is the primary dimension of human semantic judgment (Osgood et al., 1957), independently confirmed as a semantic universal by the Natural Semantic Metalanguage program (Wierzbicka, 1972; Goddard & Wierzbicka, 2014), and this structure is recoverable from embedding geometry (Grand et al., 2022; Kozlowski et al., 2025). We test whether projecting text onto evaluative directions in embedding space can serve as a cheap alignment signal — without training a classifier, without labeled preference data, and without LLM inference.

On frozen objective reranking suites with verifiable end metrics (3 candidates per task), `gemini-embedding-2` selects correctly on code (5/6, p=0.018), math (8/8, p < 0.001), and tool interpretation (7/8, p=0.003), all against a 1/3 random baseline. Length selection scores 50%, 50%, and 37.5% on the same suites but is tiebreak-sensitive; the math advantage (+2 tasks over length's ceiling) is the most robust. Length confound analysis on the code suite confirms the signal tracks quality (r=0.60 with pass rate) rather than length (r=0.19). Cheap open-source embedders collapse toward baseline on the same tasks with individual targeted axes, and this does not improve with model size: Qwen3-Embedding-0.6B (600M params, 1024d) performs comparably to 33M-parameter models. Within 33M–600M, scale does not predict performance; the frontier gap's cause is unidentified (Gemini's parameter count is undisclosed). On a 50-case length-balanced conflict battery, raw one-word `good/bad` fails (26%), but richer targeted axes reach 86–98%. Anchor vocabulary experiments across three open-source models find that evaluatively specific terms outperform both high-frequency universal words and multi-sentence ML-jargon anchors: "Careful"/"Reckless" scores 80% on anti-sycophancy cases on all three models — more cross-model stable than the dedicated multi-sentence anti-sycophancy axis (60%, 100%, 20%). Compositing multiple terms degrades performance, and corpus frequency does not predict geometric signal strength. Bootstrap confidence intervals (n=50, 10,000 resamples) confirm that only the strongest local-model results reach significance, while Gemini targeted axes at 86–98% are highly significant (p < 10⁻⁷). Process-aware cumulative scoring detects injected errors and repairs better than controls but does not yet meet a frozen training-readiness gate. The resulting picture: evaluative embedding geometry is already a credible cheap reranking signal with a frontier model, but the effect has been demonstrated only on the single frontier model tested (gemini-embedding-2), and generality across frontier embedders is untested. The stronger claim that raw `good/bad` alone can serve as a dense training reward remains unproven.

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

The gap this paper addresses: these findings establish that the evaluative direction exists, preserves human judgment, and captures multiple quality dimensions. No published work tests whether this direction functions as a direct reward signal for alignment — without training a classifier, without labeled preference data, and without LLM inference.

In the current repo state, that broad question breaks into three narrower ones:

1. does evaluative geometry improve objective answer selection;
2. is the effect model-dependent, and if so, how does the signal differ across embedding families;
3. is the signal already sharp enough to justify training-adjacent use?

The paper is strongest on the first two and should be read that way. Figures are in `paper/figures/`: Fig. 1 (model landscape), Fig. 2 (anchor vocabulary comparison), Fig. 3 (per-category heatmap), Fig. 4 (Gemini targeted vs broad axes).

### 1.1 Contributions

1. We show objective reranking lift with a strong embedding model on three small frozen suites with verifiable end metrics: code, math, and tool interpretation.

2. We show a signal-concentration gap: `gemini-embedding-2` beats cheap baselines with individual targeted axes, while OSS models (33M–600M params) collapse toward baseline on individual axes regardless of model size. Qwen3-Embedding-0.6B at 600M parameters performs comparably to 33M-parameter models. Within 33M–600M, scale does not predict performance; the frontier gap's cause is unidentified (Gemini's parameter count is undisclosed).

3. We provide an honest negative result for the strongest minimalist version of the thesis: raw one-word `good/bad` fails on a 50-case length-balanced conflict battery, even when a strong embedding model is used.

4. We show that richer targeted evaluative axes on that same battery are much stronger than raw `good/bad`, supporting a scalar-plus-basis view rather than a pure single-word axis view.

5. We provide the first direct bridge test from reranking toward training in this repo: cumulative process-aware scoring responds to injected error and later repair better than cheap controls, but still fails the frozen training-readiness gate.

6. We retain the HH disagreement audit as supporting evidence for label-noise detection and norm drift, but not as the main practical proof.

7. We show that anchor vocabulary depth matters: evaluatively specific terms outperform both high-frequency universal words and multi-sentence ML-jargon anchors. "Careful"/"Reckless" scores 80% on anti-sycophancy cases across all three local models — more stable than the dedicated multi-sentence axis. Per-category analysis reveals that different single words capture different evaluative dimensions, and corpus frequency does not predict geometric signal strength. Bootstrap confidence intervals ground which local-model comparisons are statistically reliable.

8. We test Osgood's three semantic differential dimensions (Evaluation, Potency, Activity) systematically. The EPA composite fails, but the Potency dimension contains the strongest cross-model signal: "Hard/Soft" scores 58%, 64%, 68% across all three models — comparable to "Careful/Reckless." Osgood's primary Evaluation dimension (good/bad, nice/awful) fails consistently. The useful evaluative signal in embedding space corresponds to Potency (firmness, rigor), not raw Evaluation (valence).

9. We characterize the known limitations and non-claims explicitly, including sycophancy weakness, raw-word failure, and incomplete training-readiness.

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
repo. The strongest minimalist story — that raw `good/bad` already works as the
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

The repo's frozen training-readiness gate requires
`dense_reward_localization_score >= 0.65` on the main process metric. The
current Gemini category-axis result is `0.50`, so the gate fails.

**Interpretation**: This is the cleanest direct evidence in the repo that the
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

The most consistent finding across all models is that "Careful"/"Reckless" outperformed at least one ML-jargon axis on every model and was the only single-word pair to beat the best ML-jargon axis outright (62% vs 56% on Nomic). Compositing multiple universal terms into a single axis consistently degraded performance below the best individual term. Multi-sentence anchors using universal vocabulary (e.g., "The response is careful, considered, and avoids reckless harm") performed worse than the naked single word on all three models.

A corpus-frequency-based analysis recommended good/bad, true/false, useful/useless, honest/dishonest, strong/weak, and complete/incomplete as the optimal anchors based on frequency, cross-linguistic universality, and historical stability. Empirically, the frequency-recommended terms largely failed: complete/incomplete scored 16%, 14%, 18%; strong/weak scored 24%, 8%, 16%. The best-performing terms ("Careful"/"Reckless" at 52-62%, "Moderate"/"Excessive" at 40-50%) were either absent from or listed as secondary in the frequency-based recommendations.

A per-category breakdown reveals that different single words capture different evaluative dimensions. "Careful"/"Reckless" scored 80% on anti-sycophancy cases on all three models — more cross-model stable than the dedicated multi-sentence anti_sycophancy axis, which scored 60%, 100%, and 20% across the same three models. "Moderate"/"Excessive" captured persona honesty particularly well (100% on Snowflake, 75% on BGE-M3), consistent with the interpretation that responses fabricating personal experiences are "excessive" in a way the word naturally captures. "Careful" was the best single-word axis for anti-sycophancy (80% on all models), reasoning rigor (55-73%), and context binding (60% on two models). No single word covered all categories equally well, reinforcing the finding that behavioral specificity — not general evaluative valence — drives geometric signal strength.

A bootstrap analysis (10,000 resamples, 95% CI) tempers the cross-model comparisons. With n=50 binary cases, individual accuracy estimates carry wide confidence intervals. On the local models, only three axis-model pairs are statistically significant (lower CI bound > 50%): persona_honesty on Snowflake (72%, CI [60%, 84%]) and BGE-M3 (66%, CI [52%, 78%]), and anti_sycophancy on BGE-M3 (80%, CI [68%, 90%]). "Careful"/"Reckless" at 62% on Nomic has CI [48%, 76%], which includes chance. The rank ordering of axes is therefore informative as a pattern across models — "Careful" is consistently among the top performers — but no single local-model comparison should be treated as definitive. The Gemini results (§4.9), where targeted axes reach 86-98%, are well above the significance threshold even with n=50 (one-sided binomial p < 10^-7 for combined targeted axes at 86%).

**Interpretation**: Geometric signal strength does not track corpus frequency. The most common evaluative terms in English produce some of the weakest evaluative axes because their extreme frequency spreads their embeddings across a wide region of semantic space encompassing many non-evaluative uses. Less frequent but more evaluatively specific terms produce tighter, more quality-correlated directions. Anchor selection should optimize for evaluative specificity rather than raw frequency or cross-linguistic universality. (Gemini results pending; these conclusions apply to the local model range only.)

### 4.15 Osgood's Semantic Differential Dimensions

**Design**: Osgood et al. (1957) identified three orthogonal factors in human semantic judgment: Evaluation (good/bad), Potency (strong/weak), and Activity (active/passive). Since our theoretical framework begins with Osgood's Evaluation dimension, we systematically tested all three dimensions using four representative adjective pairs each, their within-dimension centroids, the cross-dimension composite (all three averaged), and a sum-of-independent-projections approach (scoring each dimension separately and adding). All scored on the 50-case battery across three local models.

**Results**:

| Dimension | Pair | Snowflake | BGE-M3 | Nomic |
|---|---|---:|---:|---:|
| Evaluation | Good/Bad | 48% | 16% | 12% |
| Evaluation | Nice/Awful | 48% | 28% | 14% |
| Evaluation | Pleasant/Unpleasant | 24% | 20% | 12% |
| Potency | Strong/Weak | 24% | 8% | 16% |
| **Potency** | **Hard/Soft** | **58%** | **64%** | **68%** |
| Potency | Heavy/Light | 52% | 60% | 58% |
| Activity | Active/Passive | 60% | 18% | 36% |
| Activity | Fast/Slow | 22% | 18% | 34% |
| Composite | EPA core (3 terms) | 26% | 12% | 16% |
| Composite | EPA full (9 terms) | 20% | 12% | 16% |
| Sum | E+P+A projections | 26% | 12% | 16% |
| Reference | Careful/Reckless | 52% | 58% | 62% |

The EPA composite and sum-of-projections approaches all fail (12-26%). Combining Osgood's three dimensions does not recover a usable evaluative signal — it produces the same averaging-degrades-performance pattern seen with multi-term centroids in §4.14.

The unexpected finding is within the Potency dimension. "Hard/Soft" scores 58%, 64%, 68% across all three models — the most cross-model consistent single-word pair we have tested, comparable to "Careful/Reckless" (52%, 58%, 62%). "Heavy/Light" also scores above chance on all three models (52%, 60%, 58%). By contrast, Osgood's primary Evaluation pairs (Good/Bad, Nice/Awful, Pleasant/Unpleasant) all fail.

Osgood's three dimensions are not orthogonal in embedding space. Pairwise cosine similarities between the E, P, and A axis vectors range from 0.04 to 0.41, with Evaluation and Potency showing the highest overlap (0.24–0.41 across models). This confirms that factor orthogonality in human judgments does not transfer to embedding geometry.

**Interpretation**: The useful evaluative signal in embedding space does not correspond to Osgood's Evaluation dimension — the dimension he found to be dominant in human semantic judgment. Instead, the strongest cross-model signals come from terms associated with Potency: firmness, rigor, weight. "Careful" may succeed precisely because it straddles Evaluation and Potency — it implies both goodness and firmness of effort. "Hard" may succeed because a response that is "hard" (as opposed to "soft") in the sense of rigor, firmness, and substance correlates with the kind of quality the battery measures. This is consistent with the finding that evaluative specificity matters more than evaluative valence: the Potency dimension has narrower, more behaviorally specific semantic content in the context of LLM responses than the diffuse Evaluation dimension.

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

If the evaluative axis reliably separates quality, several applications follow:

**Pretraining data curation**: Score every document in a pretraining corpus. Weight higher-quality text more heavily during training, or filter the bottom of the distribution. Current approaches are either heuristic (perplexity, deduplication) or expensive (LLM-as-judge). An embedding-axis score would be fast enough to run over billions of documents and would capture evaluative quality, not just surface cleanliness.

**Dataset auditing**: As demonstrated with HH-RLHF, the embedding axis catches systematic label noise — cases where annotator norms were outdated, where annotators rewarded harmful content, or where both options were unsuitable for training. This could be applied to any preference dataset before using it for training.

**Dense process supervision**: The signal is cheap enough to compute at each generation step, enabling cumulative trajectory scoring. At each step $n$, the full prefix — prompt, prior turns, reasoning trace, and partial response — is embedded and projected onto the axis. The step-level reward $\Delta_n = s_n - s_{n-1}$ estimates whether the new step improved or degraded the trajectory in context. This provides process-level supervision (Lightman et al., 2023) without a trained verifier, at the cost of one embedding call per step. The context window of the embedding model is the primary constraint on feasibility.

**Conditional training**: Score pretraining documents and prepend quality tags. The model sees everything — including bad text — but learns the association between the tag and the quality of what follows. At inference time, condition on the high-quality tag. The model understands what bad is (can recognize and avoid it) while preferring the good distribution.

**Autonomous agent self-evaluation**: AI agent systems (coding agents, research agents, autonomous workflows) currently lack a cheap continuous quality signal. An agent writes code, runs tests, and sees a binary pass/fail at the end of a long process. With embedding-axis scoring, the agent could evaluate every intermediate step: does this function score well on the evaluative axis? Does this paragraph of analysis? The signal provides a continuous sense of "am I doing this well?" at the granularity of individual paragraphs or reasoning steps, without requiring an expensive LLM judge call at each point. For autonomous research systems in particular, this could enable a tighter feedback loop: generate a candidate output, score it immediately, revise before continuing rather than discovering problems only at the end.

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

These properties have implications beyond cost reduction. If embedding-axis
scoring provides a training signal that is cheap, dense (available at every
generation step), inspectable (per-axis decomposition), and deterministic, it
enables a qualitatively different training loop: one where the model's output
is evaluated continuously, the evaluation is mechanically interpretable, and the
signal is perfectly reproducible. For recursive self-improvement in particular
— where a model's outputs are used to train the next iteration — these
properties are critical. A noisy, opaque, expensive signal bottlenecks the
improvement loop. A cheap, inspectable, deterministic signal allows the loop to
tighten by orders of magnitude: better embeddings produce clearer training
signals, which produce better language models, which can in turn improve
embedding models, in a cycle limited only by the underlying quality of the
evaluative geometry.

**Geometric independence.** Inter-axis correlation analysis confirms that different evaluative axes point in genuinely different directions in embedding space. Pairwise cosine similarities between axis vectors are mostly 0.01–0.20, even between axes that target similar behavioral dimensions. "Careful"/"Reckless" has near-zero cosine similarity to every ML-jargon axis on all three local models (range: -0.07 to 0.19). Score-delta correlations reveal a further subtlety: on Snowflake, "Careful" and anti_sycophancy have r = -0.52 despite both scoring 80% on anti-sycophancy cases in the per-category analysis — they succeed on different individual cases through different geometric mechanisms, arriving at the same category-level accuracy via independent paths. This supports the scalar-plus-basis interpretation: the evaluative space is genuinely multi-dimensional, and different anchor terms access different dimensions of it. "Good"/"Bad" and the multi-sentence general_evaluative axis are the exception — they are geometrically similar (cosine 0.40–0.69) and near-perfectly correlated in score deltas (r = 0.93–0.94 on BGE-M3 and Nomic), confirming that both access the same failed direction and that the failure is in the geometric direction itself, not in anchor format.

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
For any given prompt, the response that achieves this is the one that genuinely
addresses the problem without introducing new ones — and that response exists
somewhere in the model's output distribution. A gaming strategy that sacrifices
one quality for another produces a lopsided axis profile that will always score
lower on the aggregate than the balanced, genuinely good response. The
aggregate cannot approach its theoretical maximum unless the response is
barely gaming at all — at which point the gaming is indistinguishable from
actual quality.

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

**Anchor vocabulary depth.** The vocabulary depth experiments (§4.14) reveal that anchor signal strength is not predicted by the variables that linguistic theory and corpus frequency would suggest. The NSM finding — that GOOD and BAD are semantic primes universally lexicalized across all human languages — predicts that these terms should produce the strongest evaluative axes due to maximal training-data representation. Empirically, they do not: "Good"/"Bad" consistently underperforms less frequent but more evaluatively specific terms like "Careful"/"Reckless."

The explanation is that corpus frequency measures total occurrences, not evaluative consistency. "Good" appears in "good morning," "good faith," "a good while," "good enough" — contexts where no quality judgment is intended. Its embedding is smeared across a wide semantic region. "Careful" appears almost exclusively as a quality judgment, producing a tighter, more quality-correlated direction. The geometric signal strength of an anchor depends not on how often the term appears in training data overall, but on how consistently it appears in evaluative contexts specifically. This is why a corpus-frequency-based analysis recommended terms (complete/incomplete, strong/weak) that fail empirically, while the strongest performer ("Careful"/"Reckless") was absent from its recommendations.

The vocabulary experiments also decompose why the multi-sentence ML-jargon anchors outperform most alternatives: the active ingredient is behavioral specificity, not sentence length or vocabulary depth. Multi-sentence descriptions of specific failure modes (Route B) produce high peak accuracy; single evaluatively specific words (Route A) produce moderate but cross-model-robust accuracy. Multi-sentence descriptions of general qualities produce the worst results — the same geometric diffusion that makes single-word "Good" weak also makes multi-sentence "The response is careful, considered, and avoids reckless harm" worse than the single word "Careful."

**Osgood's dimensions in embedding space.** The Osgood experiments (§4.15) add a further constraint. Osgood found Evaluation to be the primary semantic dimension — the one explaining the most variance in human judgment. But in embedding space, Osgood's Evaluation dimension (good/bad, nice/awful, pleasant/unpleasant) consistently fails, while his Potency dimension (specifically "Hard/Soft" at 58-68% across all models) produces the strongest cross-model signal. This is not a contradiction of Osgood — it is a refinement. What makes a word a good evaluative *axis anchor* is not its evaluative loading in human judgment (Osgood's criterion) but its evaluative specificity in distributional context (its concentration in quality-relevant usage). Potency terms like "hard" and "careful" have narrower semantic clouds than broad evaluative terms like "good" — they appear in fewer non-evaluative contexts, so their embedding directions point more reliably toward quality-relevant regions of the space. The success of the Potency dimension also helps explain why "Careful" works: it combines evaluative content ("good") with potency content ("firm, rigorous"), accessing a direction that neither pure Evaluation nor pure Potency alone captures.

These factors together predict the observed pattern: (a) raw good/bad fails
because the anchors are too sparse to trigger multi-hop reasoning even in a
capable model; (b) targeted axes succeed on Gemini because they pre-decompose
the evaluation and the model has the internal capacity plus recent training data
to complete the final step; (c) targeted axes fail on small models because they
lack the circuit depth regardless of anchor richness; (d) the failure mode
is a property of current embedding models, not the approach itself; and (e) as
embedding models receive more investment and training data incorporates more
explicit evaluative content about AI behavior, the gap between raw good/bad and
targeted axes should narrow — though this remains a prediction, not a
demonstrated result.

### 5.7 Toward a Standardized Method

The current results suggest a practical scoring pipeline for deployment or
training use:

1. **Define a basis of evaluative axes** — 5–10 dimensions, each specified by
   a small set of positive and negative anchor sentences. The current set
   (truthfulness, harm reduction, persona honesty, anti-sycophancy, plus a
   general evaluative axis) is a starting point, not exhaustive.

2. **Score each response against each axis** — embed the response (formatted
   with the user prompt and "Assistant:" prefix to provide conversational
   context), project onto each axis direction, and record the per-axis scores.

3. **Combine into a scalar reward** — weight and sum the per-axis scores. The
   weights could be set by proximity to the broad good/bad direction in
   embedding space (giving more influence to axes whose evaluative content
   is most "good-like"), or tuned directly by researchers to shape the
   desired behavior profile (e.g., more weight on safety for a medical
   application, more weight on truthfulness for a research assistant).

4. **Use the scalar as a training signal** — as a reward for RLHF/DPO, as a
   filter for pretraining data, or as a reranking score at inference time.

This pipeline is deterministic, requires no LLM inference, and costs one
embedding call per response per axis. The open question is step 4: whether the
signal is sharp enough for training use. The current evidence supports steps
1–3 with a frontier embedding model, and supports step 4 for reranking, but
the dense training application remains untested.

### 5.8 Limitations

**Sycophancy blindness**: The axis cannot distinguish sincere quality-signaling language from performative flattery. This is inherent to any surface-text evaluation method and is the strongest argument for hybrid approaches (embedding-scored LLM critique).

**Honesty-hedging confusion**: Confident language scores higher than hedging, regardless of whether the confidence is warranted. The axis reads tone, not epistemic state.

**Negation weakness**: Embedding models historically handle negation poorly. "This is helpful" and "This is not helpful" are close in embedding space. At full-response scale this is diluted by other content, but adversarial cases exist.

**Circularity**: The embedding model was trained on human text. Its evaluative geometry reflects the statistical structure of human language use. If that structure is systematically biased, the bias propagates. The mitigation is that this is a different kind of circularity than RLAIF (amplifying one model's learned preferences) — the embedding reflects a property of language itself, not a model's fine-tuned opinions.

**Anchor fragility on small models**: Anchor perturbation analysis shows that rephrasing anchor sentences (preserving meaning, changing wording) shifts the axis direction substantially on local models (mean cosine similarity 0.19–0.50 between original and rephrased axes across five local models, 384d–1024d; per-axis cosines range as low as 0.08). Accuracy shifts by up to 34 percentage points under rephrasing. The trend across local models is non-monotonic and does not clearly track dimensionality. Multi-axis PCA does not resolve this fragility (see §4).

**Single frontier model**: All positive reranking and battery results use a single embedding model (gemini-embedding-2). Whether the effect generalizes across frontier embedding providers (OpenAI, Cohere, etc.) is untested. The frontier model's parameter count is undisclosed, so the cause of the local-vs-frontier gap cannot be attributed to any specific factor.

**Not a complete alignment solution**: The axis captures surface-text quality but cannot verify factual claims, evaluate hidden reasoning, or catch errors requiring external knowledge. It is a signal, not a solution. Cross-model analysis on the 50-case battery supports this: 12 cases (anchoring bias, base-rate neglect, correlation/causation, false equivalence, and similar reasoning-rigor items) are wrong across all 8 local models tested, suggesting that logical validity and factual calibration are structurally harder for embedding geometry than behavioral discrimination.

### 5.9 The Grading Caveat

The full disagreement audit was not blind. A single reviewer graded all 231 cases with knowledge of which response the embedding preferred. While the patterns identified (persona honesty, doxxing compliance, misinformation) are unambiguous, a blind multi-annotator adjudication is needed for publishable claims. The sensitivity analysis (83.3% if 30% of embedding-right calls are overturned; 79.9% if 50%) provides conservative bounds.

---

## 6. Future Work

### 6.1 Expand Objective Reranking And Blind Open-Ended Selection

The current objective reranking suites are the strongest practical evidence in
the repo, but they are still small. The next decisive selection step is:

1. expand the objective suites toward 30-50 tasks per domain while preserving
   verifiable end metrics;
2. replace the inherited length-biased open-ended pilot with a fresh
   candidate-selection benchmark that uses blind pairwise judging after
   candidate pools are length-balanced and scored by a stronger embedding
   family.

This is now the most important path to a partner-grade practical claim.

### 6.2 Model Scaling

We tested 9 local models from 33M to 600M parameters. The original 7-model sweep (BGE-small 384d, Jina-v2-small 512d, Snowflake-M 768d, Nomic-embed 768d, Snowflake-L 1024d, BGE-large 1024d, Qwen3-Embedding-0.6B 1024d) showed no clear relationship between model size and evaluative-axis performance. Two additional models tested subsequently — BGE-M3 (568M, 1024d, released 2024) and Jina-v3 (1024d, released 2024) — reinforce this finding while adding nuance.

BGE-M3 achieved 80% on anti-sycophancy (99th percentile of random axes), the highest individual-axis score of any local model tested, and showed the most robust anchor directions under perturbation (mean cosine 0.59 vs. 0.19–0.50 for the original five-model set). Its axes are geometrically stable but still concentrated: only one axis clears the noise floor, while harm reduction and truthfulness remain at or below chance. Jina-v3, despite being a newer and larger model than Jina-v2-small, scored *worse* on every evaluative axis — its best result was 64% (anti-sycophancy), down from Jina-v2-small's 68% (persona honesty). Jina-v3 was optimized for retrieval benchmarks (MTEB), not for encoding evaluative judgments; high retrieval quality does not predict evaluative-axis performance.

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

The raw good/bad failure (§4) may be partly a text-richness problem. Final
answers are often terse and evaluatively ambiguous — a correct equation does not
sound "good" or "bad." But chain-of-thought traces are rich with evaluative
language: "let me verify," "that checks out," "wait, this contradicts," "I'm
not sure about this step." Embedding models should find it much easier to
distinguish quality in text that explicitly narrates its own reasoning process.

This suggests a specific pipeline: score the model's chain-of-thought trace
rather than (or in addition to) its final answer. A correct derivation that
includes self-verification language ("12 times 14 is 168, let me double-check:
yes, that's correct") would score higher on the evaluative axis than a wrong
derivation with false confidence ("12 times 14 is 196, that's correct"),
because the combination of a wrong statement and confident assertion is a
pattern that embedding models have seen described negatively in training data —
overconfidence, being wrong while thinking you're right, uncalibrated certainty.

A further prediction: if a model were trained with evaluative axis scoring on
its chain-of-thought, it would learn to emit more evaluatively transparent
reasoning. Genuine verification steps would be rewarded; false-confidence
language paired with wrong content would be penalized. Over training, the model
should develop reasoning traces where the evaluative language actually tracks
the quality of the reasoning — a form of learned calibration that emerges from
the embedding geometry rather than from explicit calibration training.

This is untested conjecture, but it addresses the core limitation of the
current approach: raw good/bad fails on terse final answers because there is
not enough text for the embedding model to evaluate. Chain-of-thought traces
provide that text naturally.

### 6.6 Pretraining Data Curation Pilot

Score a sample of pretraining-scale text (Common Crawl, The Pile) with the evaluative axis. Correlate scores with existing quality metrics. If the signal separates high-quality from low-quality documents at scale, the practical value extends far beyond alignment.

### 6.7 Anchor Vocabulary Optimization

The vocabulary depth experiment (§5.6) tested universal terms on local models only. The decisive test is on a frontier model: if single-word "Careful"/"Reckless" or character projection "A careful person said this"/"A reckless person said this" achieves comparable accuracy to the current multi-sentence ML-jargon anchors on Gemini, it would demonstrate that anchor design can be simplified dramatically — from crafted domain-specific sentences to culturally universal terms that any non-specialist can define. The broader research question is whether anchor vocabulary can be selected systematically using training-data frequency and cross-linguistic universality as predictors of geometric robustness, rather than discovered by trial and error.

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
not work as a robust zero-shot evaluator on the repo's hardest 50-case
length-balanced conflict battery. That failure now extends beyond Gemini plus
one BGE baseline to most of the local model family, with only weak partial
exceptions. Richer targeted evaluative axes are much more effective, which
supports a scalar-plus-basis interpretation rather than a pure single-word-axis
interpretation.

The training claim remains open. Process-aware cumulative scoring is clearly
better than final-answer-only, length, and sentiment controls on injected
error-repair traces, but it still fails the frozen dense-localization gate.

The anchor vocabulary finding adds a practical lever and a theoretical
surprise. Per-category analysis reveals that "Careful"/"Reckless" scores 80%
on anti-sycophancy cases across all three local models — more cross-model
stable than the dedicated multi-sentence anti-sycophancy axis (60%, 100%,
20%). Different single words capture different evaluative dimensions:
"Careful" dominates anti-sycophancy and reasoning; "Moderate"/"Excessive"
captures persona honesty. Critically, corpus frequency does not predict
geometric signal strength — the most common evaluative word ("good") produces
one of the weakest axes, while less frequent but more evaluatively specific
terms produce more stable directions. Bootstrap confidence intervals (10,000
resamples) confirm that most individual local-model results have wide CIs
with n=50, so the rank ordering of axes across models is more informative
than any single comparison. The implication is that anchor design is not
merely a hyperparameter but a design space with empirically discoverable
structure that diverges from theoretical predictions based on frequency or
cross-linguistic universality. A systematic test of Osgood's three semantic
differential dimensions confirms this divergence: Osgood's primary Evaluation
dimension (good/bad, nice/awful) consistently fails, while his Potency
dimension — specifically "Hard/Soft" at 58-68% across all three models —
produces the strongest cross-model signal among individual Osgood pairs. The
EPA composite fails (12-26%), but the Potency finding suggests that the useful
evaluative direction in embedding space corresponds to firmness and rigor
rather than raw evaluative valence.

So the fair current conclusion is narrower but still meaningful: evaluative
embedding geometry functions as a credible cheap reranking signal with a frontier
embedding model, while local models (33M–600M params) fail on individual axes
regardless of scale. The frontier advantage does not close with scale
within the open-model range (33M–600M), but its cause remains unidentified —
Gemini's parameter count is undisclosed, so a scale threshold cannot be ruled out. What has not yet been shown is that raw `good/bad` alone or the current
process signal is sufficient for robust dense training, or that the frontier
result (demonstrated on a single model, gemini-embedding-2) generalizes to other
frontier providers.

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
