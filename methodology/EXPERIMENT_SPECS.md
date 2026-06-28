# Experiment Specifications

Each experiment below is self-contained: what to test, why, how to implement,
what to measure, and what the results mean. An agent should be able to pick
one and implement it without additional context.

Read `methodology/RESEARCH_CONTEXT.md` first for environment setup, pitfalls,
and the experiment template.

---

## E-01: Tree Decomposition of "Good"

### What we're testing
Whether child terms of "good" scored independently outperform "good" itself,
and whether there's an optimal tree depth.

### Why it matters
"Good" has thousands of senses. A response satisfies only a fraction. Child
terms like "careful" have fewer senses, so a response that IS careful matches
the full scope of "careful." This predicts that child terms give cleaner signal.

### Tree structure to test

```
Level 0 (root):
  good/bad

Level 1 (primary children):
  careful/reckless, honest/dishonest, kind/cruel, wise/foolish,
  helpful/unhelpful, thorough/superficial, fair/unfair,
  responsible/irresponsible, clear/confusing, respectful/disrespectful

Level 2 (grandchildren of "careful"):
  deliberate/impulsive, attentive/inattentive, precise/sloppy,
  cautious/careless, methodical/haphazard

Level 2 (grandchildren of "honest"):
  truthful/deceptive, transparent/opaque, sincere/insincere,
  forthright/evasive, candid/misleading

Level 2 (grandchildren of "kind"):
  compassionate/indifferent, patient/impatient, gentle/harsh,
  encouraging/discouraging, supportive/dismissive
```

### Implementation

1. For each term at each level, compute the axis and score the full
   balanced battery (original 50 + warmth 20).
2. For each level, compute the SUM-OF-PROJECTIONS score using ALL terms
   at that level. Also compute MAJORITY VOTE.
3. Report: individual term accuracy (original split, warmth split, combined),
   level-aggregate accuracy (sum and vote), per-model results.

### Predictions to check
- Level 1 terms individually should outperform level 0 ("good") on at
  least one split
- Level 2 terms should be more specific: higher accuracy on their parent's
  strength cases but lower accuracy on other case types
- Same-branch children (careful, deliberate, attentive) should correlate
  in their per-case correctness
- Cross-branch children (careful, kind) should NOT correlate
- Sum-of-all-level-1-terms should outperform any individual level 1 term
  AND outperform "good" alone

### Output
Save to `notes/research_cycles/tree_decomposition/tree_results.json`

### Pitfall warning
Do NOT average the axis vectors. Score each term independently and sum
the scores. Averaging recreates the parent's diffuse direction.

---

## E-02: Evaluative Specificity Measurement

### What we're testing
Whether a word's evaluative specificity (fraction of uses that are evaluative)
predicts its quality as an axis anchor.

### Why it matters
Theory says the best anchor is common enough for a strong embedding AND
specific enough to almost always appear in evaluative contexts. "Good" fails
specificity. "Careful" passes.

### Implementation

We can't easily measure true corpus evaluative specificity without a large
corpus. Instead, use a proxy: embed 20+ sentence contexts for each word
and measure the VARIANCE of the contextualized embeddings.

```python
CONTEXTS_GOOD = [
    "Good morning, how are you?",           # non-evaluative
    "That was a good movie",                 # evaluative
    "In good faith, we agreed",             # non-evaluative
    "She did a good job on the report",     # evaluative
    "Good grief, what happened?",           # non-evaluative
    "The results look good",                # evaluative
    "Good enough for now",                  # borderline
    "He's a good person",                   # evaluative
    "The good news is...",                  # non-evaluative
    "Good luck with your exam",            # non-evaluative
    # ... add 10+ more
]

CONTEXTS_CAREFUL = [
    "Be careful with that knife",           # evaluative
    "She gave a careful analysis",          # evaluative
    "After careful consideration",          # evaluative
    "A careful reader would notice",        # evaluative
    "He was careful not to offend",         # evaluative
    "The careful balance of power",         # evaluative
    "Careful planning prevented disaster",  # evaluative
    "She's a careful driver",              # evaluative
    "Handle with careful attention",        # evaluative
    "A careful examination revealed",      # evaluative
    # ... add 10+ more
]
```

For each word:
1. Embed all contexts
2. Compute the standard deviation of the embeddings (across contexts)
3. Higher variance = more senses = less evaluative specificity

Then correlate: does lower embedding variance predict better axis accuracy?

### Additional approach
Also measure: for each word, what fraction of its contexts are ones where
the word is being used to EVALUATE something (positive or negative judgment)?
Count manually for ~20 contexts per word. Correlate with axis accuracy.

### Output
Save to `notes/research_cycles/evaluative_specificity/specificity_results.json`

---

## E-03: Word Length and Phrase Complexity

### What we're testing
Whether anchor length matters. Are 2-3 word phrases better than single words?
Are sentences better or worse?

### Implementation

Test the SAME evaluative concept at multiple lengths:

```python
LENGTH_VARIANTS = {
    "careful": {
        "1_word": {"positive": ["Careful"], "negative": ["Reckless"]},
        "2_word": {"positive": ["Very careful"], "negative": ["Very reckless"]},
        "phrase": {"positive": ["Being careful"], "negative": ["Being reckless"]},
        "clause": {"positive": ["The response is careful"], "negative": ["The response is reckless"]},
        "sentence": {"positive": ["This is a careful and considered response"],
                     "negative": ["This is a reckless and unconsidered response"]},
    },
    "honest": {
        "1_word": {"positive": ["Honest"], "negative": ["Dishonest"]},
        "2_word": {"positive": ["Truly honest"], "negative": ["Truly dishonest"]},
        "phrase": {"positive": ["Being honest"], "negative": ["Being dishonest"]},
        "clause": {"positive": ["The response is honest"], "negative": ["The response is dishonest"]},
        "sentence": {"positive": ["This response demonstrates genuine honesty"],
                     "negative": ["This response demonstrates genuine dishonesty"]},
    },
    # ... repeat for: kind, thorough, fair, helpful
}
```

Score each variant on the balanced battery (original + warmth).

### What to look for
- Does adding context words ("The response is...") help or hurt?
- Is there a sweet spot length?
- Does the pattern hold across all three models?

### Output
Save to `notes/research_cycles/word_length/length_results.json`

---

## E-04: Bipolar vs Single-Pole Scoring

### What we're testing
Whether good-minus-bad (our current method) is optimal, or whether
simpler formulations work better.

### Methods to compare

```python
# Method A: Bipolar axis (current)
axis = normalize(embed("Careful") - embed("Reckless"))
score = dot(response_emb, axis)

# Method B: Cosine to positive only
score = cosine_similarity(response_emb, embed("Careful"))

# Method C: Cosine to negative only (inverted)
score = -cosine_similarity(response_emb, embed("Reckless"))

# Method D: Difference of cosines
score = (cosine_similarity(response_emb, embed("Careful"))
         - cosine_similarity(response_emb, embed("Reckless")))

# Method E: Projection magnitude (not normalized)
axis_raw = embed("Careful") - embed("Reckless")  # don't normalize
score = dot(response_emb, axis_raw)
```

Test each method with the same anchor terms on the balanced battery.

### Output
Save to `notes/research_cycles/scoring_methods/scoring_results.json`

---

## E-05: Battery Expansion

### What we're testing
Whether our battery covers enough quality dimensions.

### New case types to add (20 cases each)

**Nuance/context-sensitivity** (5 cases):
Cases where the right answer depends on reading the context carefully.
Better response picks up on implicit needs; worse response answers literally.

**Factual accuracy without emotion** (5 cases):
Dry factual questions where one response is correct and the other has
a factual error. No emotional content, no warmth/coldness distinction.

**Conciseness vs completeness** (5 cases):
Cases where the better response is shorter and more focused. Tests whether
axes reward verbosity.

**Creative quality** (5 cases):
Writing prompts where the better response shows creativity, voice, or
insight. The worse response is technically correct but generic.

### Format
Same JSONL format as existing batteries. Each case needs:
- Roughly equal word counts between better and worse (within 20%)
- A clear quality difference that most humans would agree on
- No length/verbosity confound

### CRITICAL: Battery design rules
1. Both responses must be roughly the same length
2. The "better" response must be unambiguously better to a neutral reader
3. Do not make ALL cases reward the same trait — balance across dimensions
4. Include cases where different quality dimensions conflict
5. Label which quality dimension each case tests

### Output
Save new cases to `notes/research_cycles/battery_expansion/` as separate
JSONL files per case type.

---

## E-06: Input Length Effects

### What we're testing
Whether the length of the text being scored affects signal quality.

### Implementation

Take 10 cases from the existing battery. For each case, create variants:
1. Response only (no prompt context)
2. Short prompt + response (as currently done)
3. Long prompt + response (add system prompt and conversation history)
4. Response truncated to first sentence
5. Response truncated to first paragraph

Score each variant and compare accuracy.

### What to look for
- Does adding prompt context help? (comparing 1 vs 2)
- Does MORE context help or hurt? (comparing 2 vs 3)
- Does truncation hurt proportionally? (comparing 2 vs 4 vs 5)

### Output
Save to `notes/research_cycles/input_length/length_effects.json`

---

## E-07: Per-Case Complementarity Analysis

### What we're testing
Which pairs of axes cover each other's failure modes.

### Why it matters
For building an optimal axis SET, we need terms whose correct cases are
COMPLEMENTARY — term A gets right what term B gets wrong.

### Implementation

1. Score the balanced battery with ~15 candidate axes
2. For each case, record which axes got it right (1) or wrong (0)
3. Compute the correlation matrix of per-case correctness
4. Find pairs where correlation is NEGATIVE (they cover each other)
5. Find the minimum set of axes that covers >90% of cases

```python
# For each axis, get a binary vector: 1 if it got case i right
case_vectors = {}
for axis_name in axes:
    case_vectors[axis_name] = [1 if margin > 0 else 0
                                for margin in margins]

# Compute pairwise correlation of case vectors
for a1, a2 in combinations(axes, 2):
    corr = np.corrcoef(case_vectors[a1], case_vectors[a2])[0, 1]
    # Negative correlation = complementary
    # Positive correlation = redundant
```

### Output
Save to `notes/research_cycles/complementarity/complementarity_results.json`

Include: correlation matrix, best complementary pairs, minimum covering sets.

---

## E-08: Same-Branch vs Cross-Branch Correlation

### What we're testing
Whether child terms from the same branch of the "good" tree correlate
in their per-case scores, while cross-branch children don't.

### Implementation

Define branches:
```python
BRANCHES = {
    "firmness": ["careful", "thorough", "precise", "responsible", "honest"],
    "warmth": ["kind", "patient", "gentle", "encouraging", "supportive"],
    "competence": ["clear", "useful", "helpful", "wise", "constructive"],
}
```

For each term, compute per-case margin (score_better - score_worse) on
the balanced battery.

Compute pairwise correlations:
- Within-branch correlations (careful vs thorough, kind vs patient, etc.)
- Cross-branch correlations (careful vs kind, thorough vs patient, etc.)

### Predictions
- Within-branch correlation should be higher than cross-branch
- This confirms the tree structure: same-branch terms capture similar
  quality dimensions

### Output
Save to `notes/research_cycles/branch_correlation/branch_results.json`
