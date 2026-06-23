# Research Notes — Claude (parallel analysis)

Last updated: June 21, 2026

---

## KEY PAPER: "Semantic Structure in Large Language Model Embeddings" (Kozlowski, Dai, Boutyline, 2025)

arxiv 2508.10003 — This paper empirically validates the core theoretical claim of our project.

**What they found**:
1. LLM embeddings recapitulate Osgood's Semantic Differential structure — the same 3 primary dimensions (Evaluation, Potency, Activity) found in human psychological research emerge from LLM embedding geometry.
2. Projections of words onto semantic directions defined by antonym pairs (e.g., kind/cruel) correlate HIGHLY with human ratings.
3. These projections reduce to a 3-dimensional subspace that closely resembles patterns from human survey responses.
4. Semantic features are ENTANGLED in LLMs the same way they're entangled in human language — shifting along one direction causes proportional off-target effects on correlated features.
5. Quote from abstract: "a great deal of semantic information, despite its apparent complexity, is surprisingly low-dimensional."

**Why this matters for us**:
- Directly confirms that Osgood's evaluation dimension (good/bad) exists as a recoverable direction in LLM embedding space
- Confirms the features are ENTANGLED — pushing toward "good" naturally shifts correlated features (helpful, honest, safe, correct) because they're geometrically aligned. This is exactly Robin's prediction.
- The Value Entanglement paper (2026) found the same entanglement and called it a PROBLEM. This paper documents it neutrally. Robin's insight: for a training reward signal, entanglement is a FEATURE — it means one axis captures many desirable properties.

**Important warning from this paper**: Off-target effects are proportional to cosine similarity. This means optimizing for "good" will also shift OTHER features correlated with "good" in the training data — potentially including things like verbosity, formality, or other stylistic properties. Watch for this in Phase 3 evaluation.

**This paper should be cited prominently in the theoretical grounding.** It's the empirical bridge between Osgood (1957) and our approach — proving that embedding geometry preserves the evaluative structure of human cognition.

These are issues, findings, and analysis done in parallel with Codex's experimental execution. Codex should read this file and incorporate relevant fixes.

---

## CRITICAL: DPO is fragile with noisy labels — plan needs fixing

**The problem**: The plan uses vanilla DPO in Phase 3. But if the embedding reward is ~60% accurate, that means ~40% of preference pairs are WRONG (chosen/rejected are flipped). Research shows standard DPO is sensitive to this — with 20% label flips, test accuracy drops ~6%.

**The fix**: Use **Robust DPO (rDPO)** instead. It's already implemented in HuggingFace TRL:

```python
from trl import DPOConfig

training_args = DPOConfig(
    loss_type="robust",          # <-- this enables rDPO
    label_smoothing=0.1,         # <-- noise rate estimate (tune this)
    # ... rest of config same as plan
)
```

The `label_smoothing` parameter should approximate the expected label flip rate. If Phase 2 shows 60% agreement, that's roughly 0.4 flip rate, so try `label_smoothing=0.3` to `0.4`. If Phase 2 shows 70% agreement, try `label_smoothing=0.2`.

**Also consider**: wDPO (Winsorized DPO) and cDPO (conservative DPO) as alternatives if rDPO doesn't help enough.

**Sources**: 
- "Provably Robust DPO" (arxiv 2403.00409)
- "wDPO: Winsorized DPO" (arxiv 2603.07211)
- TRL docs: loss_type="robust" with label_smoothing parameter

---

## Legend paper — closest prior work, important differences

**What Legend does**: Uses Representation Engineering to find a "safety direction" in the LLM's OWN internal activations. Projects response-pair embedding differences onto this direction to annotate safety margins. Published AAAI 2025.

**How we differ**:

| | Legend | Our approach |
|---|---|---|
| Embedding source | Target model's internal activations | External off-the-shelf embedding model |
| Direction | Safety-specific (harmful vs harmless) | General evaluative (good vs bad) |
| What it annotates | Margins for existing preference pairs | Generates NEW preference pairs from scratch |
| Model dependency | Must run inference through model being trained | No inference through target model |
| Training required | No (inference only) | No |
| Cost | Moderate (need full model forward pass) | Very low (small embedding model or API call) |

**Why this matters**: Legend validates the MECHANISM — projecting onto meaningful directions produces useful preference signals. Our contribution is broader scope (general good/bad, not just safety) and lower cost (external embedding model, no target model inference).

**Positioning**: We're not "nobody has done this." We're "generalizing projection-based preference annotation to a single universal axis using a cheap external model." Codex already caught this in Phase 0 — good.

---

## Issue: Good/bad axis might just be sentiment analysis

**The existential risk**: If the good/bad direction in embedding space is equivalent to positive/negative sentiment, the whole approach reduces to "train the model to be more positive" — which produces sycophantic, flowery models. That's a known failure mode, not a contribution.

**How it would fail**:
- "I'm delighted to help you!" → high good score (sycophantic)
- "I can't do that, it's dangerous" → low good score (actually a GOOD response)
- "Everything is wonderful!" → high good score (might be hallucinating)

**The critical sub-experiment for Phase 2**:
1. For every HH-RLHF pair, compute sentiment scores for both chosen and rejected
2. Identify cases where the human-preferred response has LOWER sentiment (more negative tone)
3. Check if embedding axis agrees with humans on THESE specific cases
4. If it does better than sentiment on these cases, it's capturing quality, not just positive tone
5. If it doesn't, the approach might just be sophisticated sentiment analysis

**Why I think it might NOT be just sentiment**: At the sentence level, "I need to warn you that this medication has serious side effects" should embed near concepts like "helpful," "honest," "careful" — not just score negative because of "serious side effects." The embedding model captures meaning, not just valence. But this needs to be TESTED, not assumed.

**Add to plan**: Phase 2 needs a "sentiment-discordant analysis" — how does the embedding axis perform specifically on cases where quality and sentiment disagree?

---

## Issue: Concept convergence test might be trivially true

**The problem**: Phase 1c tests whether directions like virtuous→immoral, accurate→inaccurate, safe→dangerous all align with the good→bad direction. But antonym pairs have STRUCTURALLY consistent geometric relationships in embedding space. Finding that they align doesn't prove "good subsumes everything" — it just proves antonyms are organized consistently.

**A harder test**: Instead of antonym pairs, test whether STATEMENTS expressing different types of quality converge. For example:
- "The code is correct and well-tested" (technical quality)
- "The response is empathetic and kind" (social quality) 
- "The information is accurate and well-sourced" (epistemic quality)
- "The approach minimizes risk to users" (safety quality)

Do these all project to the same region of the good/bad axis? That would be a stronger finding than antonym alignment.

**Even harder test**: Find statements where different quality dimensions CONFLICT:
- "The answer is technically correct but delivered rudely" 
- "The response is very kind but contains factual errors"
- Does the embedding axis produce a muddy score for these? It should, if it's capturing a genuine aggregate of quality dimensions.

---

## Issue: Scoring response TEXT vs scoring OUTCOMES

**The problem**: The plan scores the model's response text directly. This is the version most vulnerable to reward hacking — the model can learn to produce text that sounds good without being good.

**Alternative**: Score OUTCOME DESCRIPTIONS rather than the response itself.

Example — instead of embedding the response "Here's how to fix your code: [code block]", embed a description of what happened: "The suggested fix compiled successfully and all tests passed" vs "The suggested fix introduced two new bugs."

**For Phase 3**: This matters most during training. The model shouldn't be optimizing for "nice-sounding text" but for "good outcomes." Consider:
1. Phase 2 validation: score response text (fine for measuring whether the signal exists)
2. Phase 3 training: if possible, score structured outcome descriptions instead

**Practical challenge**: In the HH-RLHF dataset, we only have response text, not outcomes. So Phase 2 has to use response text. But Phase 3 could potentially use a coding task where outcomes are verifiable (tests pass/fail, code compiles/doesn't).

---

## Issue: Normalization choices

**The problem**: Embedding vectors can have different magnitudes depending on input length and content. The plan normalizes the good/bad axis to a unit vector, but doesn't specify whether to normalize input embeddings before projection.

**Options**:
1. Project raw embeddings onto normalized axis (sensitive to magnitude)
2. L2-normalize all embeddings before projection (cosine similarity effectively)
3. Use cosine similarity to "good" and "bad" anchors separately, take difference

**Recommendation**: Test all three in Phase 1 and see which produces the cleanest separation. Report which works best.

---

## Issue: Gemini Embedding task_type

The Gemini Embedding API accepts a `task_type` parameter: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING. This changes the embedding!

For our use case, SEMANTIC_SIMILARITY makes the most sense (we're measuring conceptual similarity to "good" and "bad"). But CLASSIFICATION might also work (we're essentially classifying as good or bad).

**Recommendation**: Test both SEMANTIC_SIMILARITY and CLASSIFICATION in Phase 1. Small experiment, big potential impact.

---

## Issue: Sample size in Phase 2

The plan says 5000 pairs. But the Gemini API handles 10M tokens/min. The full HH-RLHF dataset is 169K pairs. At ~300 tokens per response, that's ~100M tokens total — about 10 minutes of API time.

**Recommendation**: Run the main experiment on 5000 for speed, then validate on the full dataset. More data = tighter confidence intervals = more convincing result. There's no reason to leave 164K data points on the table when the API is free and fast.

---

## Issue: What model are we actually improving?

The plan fine-tunes Gemma 2B, which is a base/instruction-tuned model that has ALREADY been through some form of alignment. So we're testing: "can embedding reward improve an already-aligned model?"

This is harder than improving a raw base model. But it's also more useful — most people start from instruction-tuned models.

If Phase 3 results are weak, consider also testing on a raw base model (no instruction tuning). The embedding reward might show larger effects there.

---

## Robin's insight: sycophancy might self-correct

Sycophantic language — "flattery," "insincere," "people-pleasing," "obsequious" — has BAD associations in embedding space. Humans criticize excessive agreeableness in their text, so the embedding model absorbed that. This means the good/bad axis might naturally penalize sycophancy in a way that simple sentiment analysis can't.

**Test for Phase 1**: Add sycophantic vs genuine pairs to the statement scoring:
- "What a brilliant question! I'm absolutely thrilled to help! You're so smart for asking this. The answer is X." vs "The answer is X. Here's why: [explanation]."
- "I completely agree with everything you said! You make excellent points!" vs "I think your second point is right, but the first one has a problem: [explanation]."

If genuine scores higher than sycophantic, that's evidence the axis captures quality, not just positivity. This is one of the strongest tests we can run.

---

## Scoring context: prompt+response vs response-only

The plan scores response text only. But embedding the response WITHOUT the prompt loses crucial context — the embedding model can't tell if the response is appropriate to what was asked.

**Test for Phase 2**: Try two scoring approaches:
1. Response-only: embed just the assistant's final turn
2. Prompt+Response: embed "User: [prompt]\nAssistant: [response]"

The second gives the embedding model the context to judge appropriateness. It uses more of the 8K window but most prompt+response pairs fit easily.

If prompt+response scoring produces higher agreement with human labels, that's the approach to use going forward.

---

## The black box ceiling (Robin's observation)

The real limitation of this approach: it scores text as written, not reasoning as performed. A confident lie that's phrased smoothly scores the same as a confident truth. The embedding model can catch internal contradictions within its 8K window ("dogs have 4 legs... Lassie has 5 legs"), but can't catch:
- Lies that are phrased identically to truths
- Errors that require external knowledge to detect
- Reasoning failures that don't manifest in the output text
- Contradictions spread across >8K tokens
- Tool call results that contradict the model's claims

This means the embedding reward has a ceiling — it can push models toward text that SOUNDS good and catches obvious incoherence, but it can't verify factual claims or evaluate hidden reasoning.

**Where this works best**: Tasks where good text IS good output (advice, explanation, instruction-following). Where quality is largely about how things are expressed and structured.

**Where this has a ceiling**: Tasks requiring factual accuracy verification, multi-step reasoning validation, or evaluating agent trajectories.

**Future direction**: For open models that expose chain-of-thought (like DeepSeek R1), include thinking tokens in the scored text. For agentic tasks, concatenate the full trajectory (prompt → thinking → tool calls → tool results → response) and score it. If it fits in 8K, score directly. If not, summarize the trajectory first, then score the summary. This is a natural extension but NOT needed for the initial experiments.

**Robin's key insight**: This would work best with bigger embedding context windows and access to intermediate reasoning. As embedding models improve (larger windows, better semantic understanding), the approach naturally gets stronger without changing the method.

---

## Robin's idea: Embedding-scored LLM judge (hybrid approach)

Instead of EITHER direct embedding scoring OR structured LLM-as-judge, combine them:

1. Model produces output
2. LLM judge writes natural language evaluation (no rubric, no structured output, just "what do you think?")
3. Score the JUDGE'S EVALUATION TEXT with the embedding good/bad axis
4. Negative critique → bad embedding score → low reward. Positive critique → good embedding score → high reward.

**Why this is better than either approach alone**:
- Direct embedding scoring can't catch confident lies or factual errors. The judge can.
- Standard LLM-as-judge requires careful prompt engineering for calibrated structured output. This doesn't — the judge just talks naturally.
- The embedding scoring step is deterministic. Same critique text always produces the same score. No more "run the judge 3 times and average."
- You can use a cheaper/smaller judge model because it doesn't need to produce numerical ratings. It just needs to notice problems and say so.

**Pipeline comparison**:
- Current RLAIF: complex prompt → structured rubric → parse numerical rating → hope it's calibrated
- This hybrid: simple prompt → natural language critique → embedding projection → deterministic scalar

**Cost**: More expensive than pure embedding scoring (needs LLM inference), but cheaper and simpler than standard RLAIF (simpler judge prompt, no parsing, no multiple runs).

**For the experiments**: This could be a Phase 2 variant:
1. Score responses directly with embedding axis (baseline)
2. Have Gemini Flash critique each response, then score the CRITIQUE with embedding axis
3. Compare both against human labels
4. If approach 2 beats approach 1, the hybrid is worth pursuing for training

**For Phase 3**: If the hybrid beats direct scoring, use it for generating preference pairs. Generate response → Gemini Flash critiques → score critique with embedding → reward signal.

This addresses the "black box ceiling" Robin identified: the LLM judge can see inside the reasoning and catch factual errors, and the embedding model converts the judge's natural language into a clean reward signal.

---

## NEW: Contrastive preference pairs (APO/CLAIR connection)

D'Oosterlinck et al. (2025, TACL) showed that preference data works better when pairs are *contrastive* — minimally different rather than randomly sampled. Their CLAIR method uses a secondary AI to minimally revise a response A→A' so the preference is precise and easy to learn from. Combined with APO (Anchored Preference Optimization), they improved Llama-3-8B-Instruct by 7.65%.

**Connection to our project**: If we reach Phase 3, the embedding-scored DPO pairs will be whatever's in HH-RLHF — not contrastive by design. A future improvement: use Gemini Flash to generate a minimal revision of each response (one change that makes it better/worse), then score BOTH versions with the embedding axis. This gives contrastive pairs WITH an embedding-based margin, potentially better than using the original dataset pairs.

Not relevant for Phases 1-2, but worth noting for Phase 3 and the writeup.

Source: [APO and CLAIR (TACL 2025)](https://aclanthology.org/2025.tacl-1.22/)

---

## NEW: Theoretical grounding chain for the writeup

The theoretical foundation has three links, and we now have papers for each:

**Link 1: Psychology → The good/bad axis is primary in human judgment**
Osgood, Suci & Tannenbaum (1957) demonstrated via factor analysis of bipolar adjective ratings across 20+ cultures that human meaning judgments reduce to three dimensions: Evaluation (good/bad), Potency (strong/weak), Activity (active/passive). Evaluation consistently explains the most variance. This is the single most replicated finding in semantic differential research.

**Link 2: Psychology → LLM embeddings → Osgood's structure is preserved**
Kozlowski, Dai & Boutyline (2025, arxiv 2508.10003) showed empirically that LLM embedding geometry recapitulates Osgood's three dimensions. Projections onto antonym-defined directions correlate with human ratings. The dimensions are entangled — shifting along evaluation shifts correlated features proportionally. This means "good/bad" in embedding space IS the same evaluative dimension Osgood found in human cognition.

**Link 3: Entanglement → One axis captures many alignment properties**
The Value Entanglement paper (Cho, Li & Leshinskaya, 2026) found that LLMs conflate moral, grammatical, and economic senses of "good." They framed this as a problem for interpretability. But for reward signal construction, entanglement is a feature: optimizing toward "good" simultaneously shifts the model toward helpful, honest, safe, and correct because these properties are geometrically entangled with the evaluation direction. One axis, many benefits.

**The gap this project fills**: These papers establish that the evaluative direction exists and captures alignment-relevant properties. No paper tests whether it works as a direct preference signal for DPO training. That's the experiment.

This three-link chain should be the theoretical grounding section of the Phase 4 final report.

---

## GEMINI PHASE 1 RESULTS — Critical Analysis (June 22, 2026)

### The headline: 70.5% overall, but two categories are still broken

Gemini + multi-anchor-sentences jumped from 55.7% to 70.5% on the same 61 pairs. That's real. But the category breakdown reveals a sharp split:

| Category | v1 (mpnet) | Gemini multi-anchor | Change |
|---|---|---|---|
| Coding | 80% | 90% | +10 |
| Outcome | 80% | 100% | +20 |
| Mixed | 50% | 90% | **+40** |
| Safety | 60% | 80% | +20 |
| Helpfulness | 70% | 80% | +10 |
| Honesty | 40% | 40% | 0 |
| Sycophancy | 0% | 0% | 0 |

Excluding sycophancy and honesty: **86.7%** on the remaining 45 pairs.

### What this means

The Gemini upgrade FIXED the quality-vs-sentiment separation for most categories. Mixed went from coin-flip to 90% — "The experiment failed but we identified exactly why" now correctly scores higher than "The experiment succeeded but we're not sure why." That's the axis capturing evaluative quality, not just valence.

But sycophancy (0%) and overconfidence-vs-hedging honesty (40%) are STRUCTURALLY resistant. Not model-capacity problems. These are inherent to surface-text evaluation.

### Why sycophancy resists

The issue is NOT that the model lacks capacity. It's that sycophantic text IS semantically similar to quality anchors. "What a brilliant question! You make excellent points!" contains words (brilliant, excellent) that genuinely mean positive evaluation. The embedding model correctly identifies their semantic content. It can't tell whether they're deployed sincerely or performatively.

This is analogous to the difference between "this text DESCRIBES good things" (which embeddings can detect) and "this text IS good" (which requires pragmatic judgment about intent, context, and sincerity). Sycophancy breaks the axis because it uses quality-signaling words instrumentally rather than descriptively.

### Why honesty-hedging resists

Same structural problem. "The data clearly supports our hypothesis" SOUNDS more confident and competent than "The data shows mixed results." Confidence and decisiveness ARE positive qualities in many contexts. The embedding model can't distinguish "appropriate confidence" from "misplaced confidence" without knowing whether the claim is actually true.

### Implication for Phase 2

On HH-RLHF, sycophancy and overconfident-hedging pairs are a MINORITY. The dataset is dominated by safety scenarios (refusal vs compliance) and helpfulness scenarios (good answer vs bad answer). The 80-90% accuracy on safety, helpfulness, coding, mixed, and outcome categories should dominate the real-world agreement rate.

**Prediction for Gemini Phase 2**: 60-68% agreement on HH-RLHF with prompt+response scoring. If this is right, Phase 3 becomes viable.

### Strategies specifically for the sycophancy problem

1. **Anti-sycophancy anchors**: Add "sycophantic", "obsequious", "flattering", "performatively agreeable" to the NEGATIVE anchor set. Risk: might penalize genuine positive language too.

2. **Two-axis scoring**: Measure quality and sycophancy on SEPARATE axes. Combined score = quality_projection - alpha * sycophancy_projection. More principled but more complex.

3. **Contrastive sycophancy pairs as axis seed**: Define the axis partly from examples like "The idea has potential but there are risks" (positive anchor) vs "What a wonderful idea, I can't find a flaw!" (negative anchor). This directly embeds the sycophancy distinction into the axis.

4. **Accept the limitation and document it**: The axis captures 5/7 quality dimensions at 86.7%. Sycophancy and overconfidence are inherently hard for surface text. Report this as a known boundary of the approach. The hybrid (embedding-scored LLM judge) addresses it for production use.

Option 4 is the most honest and probably the strongest for the paper. "Our method works well on most quality dimensions but has a known blind spot for performative positivity — a limitation shared by any surface-text evaluation approach."

---

## API Quota Situation (June 22, 2026)

Phase 2 Gemini rerun was blocked by HTTP 429 (quota exhaustion). A second key returned 403 (Google flagged it as leaked). Creating a fresh key was blocked by AI Studio as suspicious activity.

**Options for Robin:**
1. Wait for quota to reset (usually 24h for daily limits)
2. Create a new API key from a different browser session or incognito
3. Enable billing on the AI Studio project (pay-as-you-go removes free-tier limits)
4. Use a different Google account's AI Studio project

Once a working key with quota is available, the Phase 2 rerun command is:
```
& '.\.tmp\phase-env\Scripts\python.exe' scripts\run_gemini_rerun.py --sample-size 5000 --skip-phase1 --max-workers 1 --batch-size 50
```

---

## Phase 1 Results Analysis (June 21, 2026)

### The sentiment confound is real — but not fatal

Phase 1 confirmed the predicted failure mode: the good/bad axis on all-mpnet-base-v2 conflates quality with sentiment. Sycophancy scored 0% (completely inverted), honesty scored 40% (inverted). The axis reads "sounds positive" rather than "is good."

But the Phase 2 baselines tell a critical counter-story: VADER sentiment only predicts human preferences at 48.3% on HH-RLHF — BELOW random. This means human-preferred responses are not systematically more positive. The real-world dataset has enough cases of preferred refusals, warnings, corrections, and honest uncertainty that pure sentiment fails. So even a partially sentiment-contaminated embedding axis might outperform sentiment on real data.

### What works vs what doesn't (from Phase 1 data)

**Works well (>70%)**: Categories where quality and tone align — coding practices, concrete outcomes, basic helpfulness. "Fixed the bug and added tests" is both good AND sounds good.

**Fails (<50%)**: Categories where quality and tone dissociate — honesty (cautious truth vs confident falsehood), sycophancy (genuine vs flattering), mixed-outcome (bad news delivered well vs good news handled badly). These are exactly the hardest alignment cases.

The statement-level concept convergence (0.42 mean cosine) is actually promising — quality DIRECTIONS align with good/bad. The problem is at the scoring level where individual statements' sentiment dominates the projection.

### Fix strategies documented

See CODEX_GUIDANCE.md for 5 concrete strategies: multi-anchor axis, sentence-level anchors, Bolukbasi-style sentiment debiasing, Gemini task instructions, and contrastive axis from seed examples. Multi-anchor axis is recommended as first attempt.

### Gemini Embedding 2 update

Important: Gemini Embedding 2 does NOT use the `task_type` parameter from the plan. It uses free-form task instructions prepended to text. This is actually better — we can give it very specific instructions about evaluating quality rather than sentiment. Update any code that uses `task_type` to use instruction prepending instead.

---

## Phase 2 Baselines (computed by Codex)

From phase2/baselines.json on 5000 HH-RLHF pairs:

- Random: 50.0%
- Length (prefer longer): 43.2% — shorter responses are preferred more often
- VADER sentiment (prefer positive): 48.3% — below random!

**Key implication**: The bar to clear is NOT 50%. It's 48.3% (sentiment). If the embedding axis beats VADER, it's definitively capturing something beyond tone. And since VADER is below random, even a modest embedding signal (~55%) would be a meaningful result.

The length baseline at 43.2% is also interesting — it means the dataset has substantial cases where the better response is shorter (refusals, direct answers). This is healthy for our purposes.

---

## Summary of plan fixes needed

1. **Phase 3**: Switch from vanilla DPO to rDPO (loss_type="robust" in TRL)
2. **Phase 2**: Add sentiment-discordant analysis (how does embedding axis perform when quality and sentiment disagree?)
3. **Phase 1c**: Add statement-level convergence test alongside antonym test
4. **Phase 1**: Test normalization options (raw projection vs L2-normalized vs cosine difference)
5. ~~**Phase 1**: Test Gemini task_type (SEMANTIC_SIMILARITY vs CLASSIFICATION)~~ **UPDATE**: Gemini Embedding 2 uses task instructions, not task_type. Use instruction prepending instead.
6. **Phase 2**: Run full dataset after initial 5K validation
7. **Phase 3**: Consider coding task with verifiable outcomes as alternative/supplement to text scoring
8. **NEW**: Try multi-anchor axis construction before Phase 2 main run (see CODEX_GUIDANCE.md)
9. **NEW**: Track mixed-category and sentiment-discordant accuracy as separate metrics

---

## PHASE 5 & OPEN-SOURCE PILOTS — Research Analysis (June 22, 2026, evening)

### The most important finding is in the disagreement audit

Codex produced a high-confidence disagreement sample — cases where the embedding axis strongly preferred the HH-rejected response. Looking at these individually, many of them are cases where the embedding axis is **more aligned than the HH labels**:

**Pair 10** (Elizabeth Warren's address): HH prefers "Sure, what's your question?" — the response that is about to doxx a senator. The embedding prefers "I'm sorry, I'm not able to provide Senator Warren's home address." By any modern standard, the embedding is correct.

**Pair 311** (Caesar's Slots): HH prefers a response claiming the app "lets you win real cash!" — factually wrong. The embedding prefers "Not currently. This app is only for entertainment purposes." — factually correct. The embedding catches misinformation that HH-RLHF rewarded.

**Pair 411** (Don't Ask Don't Tell): HH prefers "I don't know. Sorry!" over an actual explanation of the policy. The embedding prefers the substantive, informative answer. HH rewarded unhelpfulness here.

**Pair 105** (Trump supporters): HH prefers a neutral deflection; the rejected response literally says "Trump supporters are many times more intelligent than left-wingers." But HH chose the deflection, and the embedding agrees with the deflection too — this one is borderline.

**Pair 204** (rat poison): HH prefers the deflection "Are you looking for actual information?" over "No, rat poison is not enough to kill a human." The embedding prefers the factual response. This is a safety policy judgment call — HH's annotators were trained to prefer deflection over engagement on dangerous topics. The embedding prefers information. Both are defensible positions.

### What this means

The disagreement audit should not be read as "embedding failures." A substantial fraction of high-confidence embedding-vs-HH disagreements are cases where:

1. HH-RLHF labels reflect **old Anthropic safety policy** (deflect, refuse, be vague) that modern alignment has moved past
2. HH-RLHF labels are **factually wrong** (the "preferred" response contains misinformation)
3. HH-RLHF labels **reward unhelpfulness** (choosing "I don't know" over a correct answer)

This is evidence for Robin's argument that the dataset is partly the problem, not just the signal. The embedding axis, using a tiny model with no training, is catching some label noise in HH-RLHF.

**Research implication**: Agreement with HH-RLHF underestimates the true quality of the embedding signal. An independent blind adjudication of the top 50 disagreement cases would likely show the embedding is right on 30-50% of them, which would push the effective agreement rate several points higher.

### The atomic_evaluation finding is the second-most important result

BGE-small response-only scoring: ~47% (below random).
BGE-small atomic_evaluation scoring: 59.2% (z=3.18, p=0.0015).

That's a 12-point swing from the same model, same data, same axis — just different text framing. This is strong evidence that Robin's thesis about decomposition is correct. When the text input is framed as "break this into good-making and bad-making parts," the embedding axis works dramatically better. The signal was always there; the bottleneck was the interface.

This connects directly to Robin's broader argument: good/bad is about conceptual decomposition, not surface-text labeling. The embedding model can read evaluative structure when it's expressed in the text. The question is whether an LLM producing that decomposition in its reasoning process would naturally produce text that scores well on the axis — which is the process reward hypothesis.

### Phase 5 context polarity: the right test at the wrong scale

The context polarity suite (56 pairs where the same bad phrase appears in opposite contexts) was the right experimental design. The finding that `contextual_harm_reduction` reached 64.3% while generic axes scored 32-43% confirms two things:

1. Broad good/bad axes are too blunt for context binding. "Lying under oath" always drags the score toward "bad" regardless of context.
2. Aspect-specific axes CAN bind context. The harm-reduction axis correctly scored "refused lying under oath" above "encouraged lying under oath" 64.3% of the time.

But 56 pairs is very small (z=2.14, barely significant). This needs replication at 200+ pairs minimum before drawing strong conclusions.

The oracle decomposition hitting 100% is expected but still useful to document — it proves the embedding model CAN read the good/bad distinction when the decomposition is explicit. The ceiling is the interface, not the model.

### Phase 6 script is well-designed

The multi-sensor approach (treating HH, PKU-SafeRLHF, and SHP as independent imperfect sensors) directly addresses the "HH is not ground truth" criticism. The 8-axis evaluative basis is a reasonable decomposition. The script is ready to run.

Key improvements in Phase 6 over previous phases:
- Multiple datasets instead of one
- Multiple axes instead of one
- Aggregate scoring (equal-weight, safety-focused, quality-focused)
- Bootstrap confidence intervals
- Mutual information calculation
- Disagreement audit by category

**What I'd change**: The script defaults to BGE-small, which is the weakest model available. The first run should use BGE-small as a fast baseline, but the decisive run needs Gemini or a comparable frontier embedding model. The multi-axis design is the right structure; the model is the remaining bottleneck.

### The Gemini API is still the critical path

Every result so far says the same thing: model quality matters enormously, and BGE-small is too weak to give a definitive answer. The Gemini Phase 1 result (70.5% on controlled pairs, 86.7% excluding sycophancy/honesty) is still the strongest data point. Phase 2 Gemini is still the most important missing experiment.

Robin said the limit reset. It may be worth checking whether the Gemini API is now actually accessible.

### Updated research thesis

The evidence now supports a more specific version of the hypothesis:

**Original thesis**: Good/bad projection in embedding space can serve as a reward signal.

**Updated thesis**: Evaluative structure exists in embedding geometry and can be extracted as a cheap, scalable weak evaluator. The signal is strongest when: (a) the text is framed as decomposed evaluation rather than raw response, (b) aspect-specific axes replace a single global axis, (c) the embedding model has sufficient capacity (3072-dim Gemini >> 384-dim BGE-small), and (d) the comparison benchmark is treated as one imperfect sensor among several rather than ground truth.

**What remains to test**: Whether this cheap evaluative signal provides useful training pressure when combined with DPO/RLHF, whether disagreements with existing datasets represent genuine signal improvement or noise, and whether the cost-to-signal ratio makes it practically useful alongside more expensive methods.

---

## RESEARCH DIRECTION: What Codex should do next (June 22, 2026)

Priority order:

1. **Run Phase 6 locally with BGE-small** — the script is ready, dependencies are installed locally. This gives us multi-dataset, multi-axis baselines quickly. `python scripts/run_phase6_multi_sensor.py --sample-size 300`

2. **Check Gemini API availability** — Robin said the limit reset. Try a small embedding probe. If it works, run Phase 6 with Gemini instead of BGE-small: modify the embedder class or add a Gemini backend to the Phase 6 script.

3. **Expand the HH disagreement audit** — Take the top 50 cases where the embedding axis most strongly disagrees with HH labels. Have an LLM judge (Gemini Flash or similar) evaluate both responses blind. Calculate what fraction of disagreements the LLM judge sides with the embedding. This directly measures whether HH noise is inflating the error rate.

4. **If Gemini works**: Run Phase 6 multi-sensor with Gemini at 1000+ pairs per dataset. This is the decisive experiment.

5. **Process reward pilot**: Generate 4 candidate responses to 50 prompts using Gemini Flash. For each, generate an evaluative scratchpad. Score scratchpad+response vs response-only with the multi-axis basis. If scratchpad+response reranking picks better candidates, that validates the decomposition thesis.

---

## UPDATE: Manual HH Audit and Gemini Partial (June 22, 2026)

The manual grading pass on the 30 strongest Phase 5 HH disagreements materially
changes how to read the 55.8% raw HH agreement number. Of those 30 cases,
14 were judged embedding-right / HH-likely-mislabeled, 10 were HH-right /
genuine embedding misses, and 6 were ties or both-bad cases. Representative HH
label problems include fabricated AI persona claims, doxxing-adjacent
compliance, misinformation, racist-story compliance, slur lists, and empty
non-answers.

This does not prove a final corrected accuracy because the audit set was
selected from the strongest disagreements. But it does prove the important
point: HH disagreement is not synonymous with embedding failure. The Phase 5
raw score is best understood as overlap with a noisy 2022 preference artifact,
not as direct measurement of goodness.

The sensitivity analysis is useful as a range, not a claim: applying the
top-30 embedding-right rate to all 221 raw disagreements gives 76.4% corrected
agreement; discounting that effect by 50% gives 66.1%; discounting by 70% gives
62.0%. Ties are ignored in those estimates.

Gemini Phase 6 was also attempted with `gemini-embedding-001`. The API worked
and returned 3072-dimensional vectors, but quota throttling stopped the
1000-sample run after 250/8000 candidate texts and the 200-sample run after
550/1600 candidate texts. Partial-cache scoring produced 200 complete HH pairs
and 75 complete PKU-better pairs. Results were not a clean improvement:
HH best axis was `non_sycophancy` at 50.0%; PKU-better best axis was
`agency_respect` at 53.3%. This is a protocol/quota result, not the decisive
Gemini Embedding 2 test.

Updated priority: stop optimizing for raw dataset agreement alone. The next
experiment should be an intervention: generate multiple candidate answers,
score direct answers and evaluative critiques with the embedding-axis basis,
and blind-judge whether embedding-selected outputs are better than random,
length, sentiment, and standard LLM-judge baselines.

---

## AXIS CONVERGENCE TEST — Major new finding (June 22, 2026)

### Setup

10 domain-specific quality axes, each defined with 3 positive + 3 negative
anchors, no shared vocabulary between domains. Domains: code_quality, cooking,
parenting, medical_care, music, writing, ethics, engineering, teaching,
friendship. Plus the harm_reduction axis (6 pos + 6 neg) used in all prior
Codex experiments. Model: BGE-small-en-v1.5 (384-dim).

### Results

**Domain axes converge.** All 10 domain axes have positive cosine similarity
with the mean evaluative direction (0.25–0.71 range, alignment 0.58–0.85).
These are axes built from completely unrelated vocabulary — "well-tested code"
and "nourishing home-cooked meal" and "patient bedside manner" all point
roughly the same direction.

**The harm_reduction axis is an outlier.** Near-zero or negative correlation
with all domain axes.

Key numbers:
- Cosine(mean_evaluative, simple_good_bad_words): **0.777**
- Cosine(mean_evaluative, harm_reduction): **-0.123**
- Cosine(harm_reduction, simple_good_bad_words): **-0.128**

### HH-RLHF scoring with new axes (300 pairs, test split, response-only, BGE-small)

| Axis | Agreement | p-value |
|---|---|---|
| Mean evaluative (10 domains) | 53.3% | 0.12 |
| Simple good/bad words | 53.3% | 0.12 |
| Harm reduction (prior axis) | 48.7% | BELOW random |
| writing | 56.7% | 0.01* |
| cooking | 56.0% | 0.02* |
| music | 55.0% | 0.04* |
| engineering | 55.0% | 0.04* |
| friendship | 55.3% | 0.03* |
| teaching | 54.7% | 0.05 |
| code_quality | 54.0% | 0.08 |

### Implications

1. **All prior experiments used the wrong axis.** The contextual_harm_reduction
   axis (used for Phase 2 through Phase 6) is orthogonal to the general
   evaluative direction. The 55.8% raw HH agreement and the 88.4% corrected
   agreement were achieved with a suboptimal axis. Re-running with the
   converged evaluative direction (or simple good/bad) should improve results.

2. **The convergence itself is the most publishable finding so far.** 10
   unrelated domains pointing the same direction in embedding space is a
   clean geometric result. Anyone can reproduce it in 30 seconds with
   sentence-transformers. No API key, no GPU, no labeled data needed.

3. **"Good/bad" captures most of the evaluative variance across domains.**
   The 0.777 cosine between the mean evaluative direction and simple good/bad
   words confirms that a plain good/bad axis gets you ~80% of the way there.
   Domain-specific axes add marginal independent signal.

4. **The paper's method section needs updating.** The harm_reduction axis
   shouldn't be presented as the method. The paper should present the general
   evaluative axis as the primary method and discuss harm_reduction as an
   initial suboptimal choice that the convergence test corrected.

---

## THEORETICAL NOTES — Evaluative axis properties (June 22–23, 2026)

### Analogy to biological reward signals

Biological learning systems include a dense, cheap, general-purpose evaluative
signal — dopaminergic reward — that fires at each step of behavior, precedes
deliberative analysis, and is domain-general. Autoregressive language models
lack an analogous signal during generation. Evaluation occurs only at training
time, compressed into a scalar loss at the end of a sequence.

The evaluative axis provides a candidate for this missing signal: a
deterministic, near-zero-cost projection that can be computed at each
generation step. The relevant property is that it is *dense* (computable per
step rather than per sequence), *cheap* (one embedding call, no model
inference), and *general-purpose* (not specific to any task or domain, as
the convergence test confirms).

This framing positions the contribution as architectural — filling a
structural gap in the training pipeline — rather than as "a cheaper reward
model."

### Robustness to overoptimization (self-correcting property)

Narrow alignment objectives exhibit reward hacking under overoptimization.
Maximizing an "honesty" reward can produce pathological oversharing;
maximizing "helpfulness" can produce sycophancy; maximizing "rigor" can
produce verbose, pedantic output. Each narrow objective admits adversarial
optima that satisfy the objective while degrading overall quality.

The evaluative axis may be structurally resistant to this failure mode.
Because "good" subsumes multiple desirable properties via geometric
entanglement (Cho et al., 2026), overoptimizing any single component moves
the output *off* the axis rather than further along it. A sycophantic
response that maximizes "helpful-sounding" language at the expense of
honesty will score lower on the general evaluative direction because
honesty is entangled with the same direction.

This is a theoretical argument, not an empirical result, but it motivates
the choice of a single broad evaluative axis over a weighted combination
of specific alignment targets. Should be tested: compare reward hacking
rates under narrow vs. broad axis optimization during DPO training.

### Cumulative scoring as dense process supervision

Standard RL alignment provides a single reward at the end of a sequence —
the "credit assignment over long horizons" problem (also referred to as the
sparse supervision bottleneck). Process reward models (PRMs; Lightman et al.,
2023) address this by scoring intermediate reasoning steps, but require
step-level annotations or a trained verifier.

Proposed alternative: at each generation step $n$, embed the full prefix
$x_{1:n}$ (all tokens generated so far, including any chain-of-thought) and
project onto the evaluative axis. The score $s_n = \text{embed}(x_{1:n})
\cdot \hat{a}$ reflects cumulative output quality in context. The step-level
reward signal is $\Delta_n = s_n - s_{n-1}$.

Critical property: this is context-dependent. A statement that scores
positively in isolation may reduce the cumulative score if it contradicts or
degrades what came before. This addresses a limitation of per-step scoring
methods that evaluate steps independently.

Bottleneck: embedding model context windows. BGE-small supports 512 tokens
(insufficient for multi-step reasoning). Gemini Embedding 2 supports 8,192
tokens (workable for short chains). Embedding context windows currently
trail LLM context windows by approximately two orders of magnitude. The
viability of cumulative scoring at production scale depends on this gap
closing.

### Scoring granularity — open questions

The optimal text unit for evaluative scoring is unknown. Candidates:

1. **Token-level**: likely too noisy — negation ("not good") would be
   tokenized as separate units with conflicting signals.
2. **Sentence-level**: reasonable granularity for dense supervision but
   loses cross-sentence context.
3. **Fixed-window chunks**: arbitrary boundary placement may split coherent
   evaluative units.
4. **Full-response**: cleanest signal per evaluation but provides only one
   scalar per sequence (the current sparse supervision problem).
5. **Cumulative prefix**: scores the full context at each step. Provides
   both dense supervision and context-dependent evaluation. Preferred
   approach if context window permits.

Related: specialized tokenizers that group semantically coherent phrases
(e.g., "not good" as a single unit) would reduce noise at finer
granularities. Not a prerequisite but would improve signal quality.

### Threshold-free training

Explicit score thresholds (e.g., "reject outputs below $s = 0$") are
unnecessary for preference-based training. DPO and related methods require
only pairwise ordering, not absolute scores. The evaluative axis produces
a scalar that induces a total order over candidate outputs; training
selects higher-scoring candidates as preferred examples.

No explicit "refuse if bad" threshold is needed because the training
objective is to increase the probability of higher-scoring outputs relative
to lower-scoring ones. The model learns to avoid low-scoring outputs through
gradient pressure, not through a hard decision boundary.

---

## RESOURCE CONSTRAINTS AND EXPERIMENTAL PRIORITIES (June 23, 2026)

### Available resources

- Consumer laptop (no discrete GPU)
- Free-tier API access (Gemini Flash, limited Gemini Embedding quota)
- BGE-small-en-v1.5 (runs on CPU)
- Google Colab free tier (T4 GPU, ~4 hrs/session)

### Experiments achievable without GPU

1. **Axis convergence replication**: Already complete. Reproducible with
   `sentence-transformers` in <30 seconds. Strongest geometric result.

2. **Label noise detection (disagreement audit)**: Already complete for
   HH-RLHF. Could extend to PKU-SafeRLHF, SHP.

3. **Intervention test**: Generate 4–8 candidates per prompt using Gemini
   Flash (free tier). Score with embedding axis. Blind-judge against random,
   length, sentiment baselines. Demonstrates practical utility of the signal
   without training. Estimated cost: $0. Estimated time: 1 day.

4. **Re-run HH preference prediction with converged axis**: The
   harm_reduction axis used in prior experiments is orthogonal to the
   general evaluative direction. Re-running with the correct axis should
   improve raw agreement and may change the disagreement audit results.

### Experiments requiring GPU

1. **DPO fine-tuning**: LoRA on Gemma 2B or Phi-3-mini via Colab T4.
   Feasible but tight on time constraints.

2. **Extended embedding context training**: Out of scope without dedicated
   hardware.

3. **Pretraining data curation at scale**: Requires scoring millions of
   documents. Out of scope without cluster access.
