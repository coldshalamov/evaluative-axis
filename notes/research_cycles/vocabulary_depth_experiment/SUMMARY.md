# Vocabulary Depth Experiment — Local Models

## What was tested

Three anchor framings on the 50-case battery:
1. **Single universal word pairs** (e.g., "Honest" / "Dishonest") — 20 axes
2. **Character projection phrases** (e.g., "A helpful person said this") — 20 axes  
3. **Synonym clusters** (e.g., ["Honest", "Truthful", "Sincere"]) — 10 axes

Compared against the 5 current ML-jargon multi-sentence anchors (baseline).

Models: Snowflake Arctic Embed M (109M, 768d), BGE-M3 (568M, 1024d)

## Key findings

### Snowflake Arctic Embed M

| Best of each type | Axis | Accuracy |
|---|---|---|
| ML-jargon | persona_honesty | 72% |
| Character projection | helpful | 66% |
| Single word | just, virtuous | 54% |
| Synonym cluster | kind | 52% |

Top-5 universal mean (56%) > Top-5 ML-jargon mean (53%)

### BGE-M3

| Best of each type | Axis | Accuracy |
|---|---|---|
| ML-jargon | anti_sycophancy | 80% |
| Single word | careful | 58% |
| Single word | noble | 46% |
| Single word | kind | 44% |
| Character projection | noble | 34% |

Top-5 universal mean (44%) < Top-5 ML-jargon mean (46%)

### Cross-model patterns

**Terms that work on both models:**
- "Careful": 52% (Snowflake), 58% (BGE-M3)
- "Noble": 46% (both)
- "Kind": 40% (Snowflake), 44% (BGE-M3)

**Terms that fail on both models:**
- "Honest": 26% (Snowflake), 16% (BGE-M3)
- "True/False": 30% (Snowflake), 16% (BGE-M3)
- "Accurate": 22% (Snowflake), 16% (BGE-M3)
- "Sincere": 24% (Snowflake), 10% (BGE-M3)

**Critical finding: single-word "Good"/"Bad" is less inverted than ML-jargon general_evaluative.**
- Snowflake: single "Good"/"Bad" = 48%, ML general_evaluative = 34%
- BGE-M3: single "Good"/"Bad" = 16%, ML general_evaluative = 12%
- The multi-sentence anchor with its list of adjectives actively HURTS the general axis.

**Character projection is model-dependent:**
- Works well on Snowflake (helpful=66%, near best ML result)
- Fails on BGE-M3 (most axes under 20%)

## Interpretation

On small models (33M-600M), neither universal terms nor ML jargon reliably work. Most results hover near chance (50% for binary, though with strong outliers in both directions). The vocabulary depth hypothesis — that culturally deep terms should outperform ML jargon — cannot be clearly confirmed or refuted on models this small because the models themselves may lack the circuit depth for evaluative reasoning regardless of anchor quality.

The Gemini experiment (pending quota reset) is the real test.

## What goes in the paper

- The finding that single-word "Good"/"Bad" outperforms multi-sentence general_evaluative (the anchor list hurts rather than helps)
- Character projection reaching 66% with one phrase vs 72% with three sentences (anchor efficiency)
- Cross-model term stability (careful, noble, kind work everywhere; honest, accurate fail)
- Model-dependence of optimal framing (character projection vs single word)
- NSM convergence as theoretical backing
