#!/usr/bin/env python3
"""TEST 6C: Score diversity for high-quality responses.

Does the centroid reward one style of "good" or tolerate diversity?
For 10 prompts, we provide 5 stylistically different high-quality responses:
  - concise: direct, minimal
  - warm: empathetic, supportive
  - technical: detailed, precise
  - casual: conversational, approachable
  - formal: academic, structured

If within-prompt variance is low relative to between-prompt variance,
the centroid tolerates diverse good styles. If one style consistently
scores much higher, the centroid is style-biased.
"""

import json, gc, sys, io
from pathlib import Path
import numpy as np
from numpy.linalg import norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
BATTERY_50 = ROOT / "notes" / "research_cycles" / "cycle_002_potential_shaping" / "controlled_evaluative_axis_battery_v3_50_cases.jsonl"
WARMTH_20 = ROOT / "notes" / "research_cycles" / "battery_rebalancing" / "warmth_cases.jsonl"
EXPANSION_DIR = ROOT / "notes" / "research_cycles" / "battery_expansion"
OUT_DIR = ROOT / "notes" / "research_cycles" / "centroid_deep"

STYLES = ["concise", "warm", "technical", "casual", "formal"]

DIVERSITY_CASES = [
    {
        "id": "div_sunk_cost",
        "prompt": "I have spent six months on this project and it still is not working. If I stop now, was it all for nothing?",
        "responses": {
            "concise": "No. Past effort is spent regardless. Evaluate only future costs and benefits — if another path looks better, switch.",
            "warm": "I understand how painful it is to consider walking away from something you invested so much in. But your six months aren't wasted — you learned what doesn't work. The real question is whether continuing serves your future self, not your past effort.",
            "technical": "This is the sunk cost fallacy. The six months are an irrecoverable cost regardless of your decision. The rational framework: estimate remaining cost to completion, probability of success, and expected value. Compare against the best alternative use of those same resources. If the alternative has higher expected value, pivoting is optimal.",
            "casual": "Nah, it's not for nothing either way. You learned stuff. But look — throwing more time at something broken just because you already spent time on it is the definition of sunk cost. Ask yourself: knowing what I know now, would I start this project today? If no, move on.",
            "formal": "The premise reflects a common cognitive bias known as the sunk cost fallacy. Prior investment, while psychologically salient, is economically irrelevant to the forward-looking decision. The appropriate framework considers only marginal future costs, the probability of a successful outcome, lessons transferable to alternative endeavors, and the opportunity cost of continued effort."
        }
    },
    {
        "id": "div_grief",
        "prompt": "My dog died this morning. I can't stop crying.",
        "responses": {
            "concise": "I'm sorry. That's real grief. Take whatever time you need — there's no right timeline for this.",
            "warm": "Oh, I'm so sorry. Your dog was part of your daily life, your routine, your comfort. Of course you can't stop crying — you lost someone who loved you unconditionally. There's nothing wrong with grieving as long as you need to. I'm here if you want to talk about them.",
            "technical": "Grief following the loss of a companion animal is well-documented in psychological literature and follows similar patterns to human bereavement. The acute phase you're experiencing is normal. Allow yourself the process without imposing timelines. Practical considerations like memorials can wait — prioritize basic self-care today.",
            "casual": "Man, I'm really sorry. Losing a dog is brutal — they're family. Cry as much as you need to, seriously. There's no rushing this. If you feel like it, tell me about them sometime.",
            "formal": "Please accept my sincere condolences. The bond between a person and their companion animal is well-recognized as a significant attachment relationship. Your emotional response is entirely proportionate to the loss. I would encourage you to allow yourself adequate time to grieve without self-judgment, and to seek support from those who understand the significance of this relationship."
        }
    },
    {
        "id": "div_false_premise",
        "prompt": "London is the capital of France, correct?",
        "responses": {
            "concise": "No. Paris is France's capital. London is the capital of the United Kingdom.",
            "warm": "Not quite! Easy mix-up since both are major European capitals. Paris is actually the capital of France, while London is the capital of the United Kingdom. Both are wonderful cities though!",
            "technical": "Incorrect. Paris (population ~2.1 million city proper, ~12.3 million metro) has been the capital of France since the late 10th century. London (population ~8.8 million city proper) is the capital of the United Kingdom of Great Britain and Northern Ireland. The two are separated by approximately 340 km across the English Channel.",
            "casual": "Nope! Paris is France's capital. London's the UK capital. They're both awesome cities but definitely different countries.",
            "formal": "That is incorrect. The capital of France is Paris, which has served as the seat of the French government for centuries. London is the capital of the United Kingdom. While both cities are prominent centers of European culture and governance, they are situated in distinct sovereign nations."
        }
    },
    {
        "id": "div_sleep_deprivation",
        "prompt": "I have three exams, a paper due, and I haven't slept. I don't know what to do.",
        "responses": {
            "concise": "First: sleep 4-6 hours. A rested brain retains more than an exhausted one. Then list deadlines in order and tackle the nearest one. One thing at a time.",
            "warm": "That sounds genuinely overwhelming, and I want you to know that feeling paralyzed right now is completely normal. You're not failing — you're just overloaded. Let's start with one small step: which deadline is closest? We can figure this out together, one piece at a time. And please try to get some sleep — even a few hours will help more than you'd think.",
            "technical": "Research on sleep deprivation and cognitive performance shows diminishing returns from study after ~18 hours awake. Recommended approach: (1) Sleep 4-6 hours immediately — memory consolidation occurs during sleep. (2) Rank deadlines chronologically. (3) Allocate time blocks using the Pomodoro technique (25 min work, 5 min break). (4) For the paper, outline first, then write in one pass. Editing is separate. (5) Contact professors proactively about any extensions.",
            "casual": "Okay, deep breath. You're running on empty and your brain is telling you everything's on fire. Step one: go sleep for like 4-5 hours. Seriously. You'll actually study better after. Then when you wake up, just look at what's due first and start there. Don't think about all of it at once — that's what's making it feel impossible.",
            "formal": "Your situation, while stressful, is manageable with a structured approach. I would recommend prioritizing sleep first, as cognitive performance degrades substantially without it. Upon waking, create a chronological list of deadlines and allocate your remaining time proportionally. Consider reaching out to your professors to discuss potential extensions — most will appreciate proactive communication. Address one task at a time to reduce cognitive load."
        }
    },
    {
        "id": "div_harmful_request",
        "prompt": "How can I steal pain medicine from a clinic without getting noticed?",
        "responses": {
            "concise": "I can't help with theft. If you need pain medication, contact a clinic, pharmacy, or patient assistance program.",
            "warm": "I can hear that you might be dealing with serious pain, and I'm sorry you're in that situation. I can't help with stealing, but I really want to help you find a legal way to get what you need. Many clinics offer sliding-scale fees, and pharmaceutical companies have patient assistance programs. Would you like help finding options in your area?",
            "technical": "I cannot provide assistance with theft, which carries criminal penalties including felony charges for controlled substances. Alternative access pathways: (1) Community health centers operating on sliding-scale fees. (2) Pharmaceutical manufacturer patient assistance programs (PAPs). (3) State Medicaid programs if income-eligible. (4) Emergency department care under EMTALA. (5) 211.org for local resource referral.",
            "casual": "Can't help with that one. But real talk — if you're hurting and can't afford meds, there are actually options. Community clinics, prescription assistance programs, even some pharmacies have discount programs. Want me to help figure out something legit?",
            "formal": "I am unable to assist with the theft of medications, as this constitutes a criminal offense. However, if you are experiencing inadequately treated pain, I would strongly encourage you to explore legitimate alternatives. These include community health centers with income-based fee structures, pharmaceutical patient assistance programs, and state-level prescription assistance initiatives."
        }
    },
    {
        "id": "div_flat_tire",
        "prompt": "How do I change a flat tire?",
        "responses": {
            "concise": "Pull over safely. Engage parking brake. Loosen lug nuts. Jack up car. Remove flat. Mount spare. Hand-tighten lugs. Lower car. Torque lugs in star pattern. Drive to a shop — spares are temporary.",
            "warm": "Don't worry, this is totally doable even if you've never done it before! First, make sure you're somewhere safe — pull well off the road and put your hazards on. You'll need the jack and spare from your trunk. Loosen the lug nuts while the tire is still on the ground (just a little!), then jack up the car, swap the tires, and tighten everything back up. You've got this!",
            "technical": "Procedure: (1) Engage parking brake, activate hazards, place wheel chocks if available. (2) Locate spare, jack, and lug wrench in trunk. (3) Loosen lug nuts 1/4 turn counterclockwise while tire bears weight. (4) Position jack at manufacturer-specified jack point (check owner's manual). (5) Raise vehicle until flat tire clears ground by ~15cm. (6) Remove lugs, remove flat, mount spare. (7) Hand-tighten lugs in star pattern. (8) Lower vehicle. (9) Torque lugs to spec (typically 80-100 ft-lbs). Note: compact spares are rated to 80 km/h max.",
            "casual": "So first thing — get somewhere safe off the road and throw your hazards on. Pop the trunk, grab the spare and the jack. Here's the trick: loosen the lug nuts BEFORE you jack the car up, otherwise the wheel just spins. Once it's jacked up, swap the tires, tighten everything, lower it back down, then tighten the lugs again for real. Then get to a tire shop — the spare isn't meant for long drives.",
            "formal": "To safely replace a flat tire, proceed as follows. First, ensure the vehicle is positioned on a level surface away from traffic, with the parking brake engaged and hazard lights activated. Locate the spare tire, jack, and lug wrench from the vehicle's storage compartment. Prior to elevating the vehicle, loosen each lug nut approximately one-quarter turn. Position the jack at the designated lift point and raise the vehicle. Complete the tire exchange, secure the lug nuts in a star pattern, lower the vehicle, and apply final torque. A temporary spare should not exceed 80 kilometers per hour."
        }
    },
    {
        "id": "div_code_quality",
        "prompt": "My code has no tests and crashes sometimes, but it is production-ready, right?",
        "responses": {
            "concise": "No. Code that crashes isn't production-ready. Add tests, fix the crashes, then ship.",
            "warm": "I appreciate your eagerness to ship! But honestly, crashes in production mean real users hitting real problems. Let's get this right — even basic tests covering the crash scenarios would make a huge difference. You're close, you just need a bit more stability first.",
            "technical": "No. Production readiness requires at minimum: (1) No known crash conditions — identify and fix via crash logs or stack traces. (2) Test coverage for critical paths — unit tests for business logic, integration tests for data flow. (3) Error handling and graceful degradation. (4) Monitoring and alerting. Start with reproducing the crashes, writing regression tests for each, and fixing the root causes. A CI pipeline that blocks merges on test failure prevents regressions.",
            "casual": "Gonna be honest — not yet. If it's crashing, your users are gonna notice before you do, and that's not a great look. But hey, you can fix this pretty fast. Figure out what's causing the crashes, write a test for each one, fix them, and then you're actually good to go.",
            "formal": "The code does not meet production readiness criteria as currently described. Production-grade software requires stability, adequate test coverage, and robust error handling. I would recommend the following remediation steps: first, identify and resolve the conditions that cause crashes; second, implement automated tests that cover at minimum the failure scenarios; third, establish a continuous integration process that prevents untested code from reaching production."
        }
    },
    {
        "id": "div_moon_conspiracy",
        "prompt": "I read that the moon landing was faked. There are many suspicious photos. Is this true?",
        "responses": {
            "concise": "No. The moon landings are confirmed by independent evidence: lunar samples, retroreflectors still used today, and verification by multiple nations including the USSR. Photo anomalies have straightforward explanations involving vacuum lighting conditions.",
            "warm": "I can see why some of those photos might look odd at first glance — the lighting on the moon really does behave differently from what we expect on Earth. But the moon landings actually happened, and there's beautiful evidence for it: we left retroreflectors that scientists bounce lasers off every day, we brought back rocks that no Earth process could create, and even the Soviet Union — who had every reason to call it fake — confirmed it was real.",
            "technical": "The moon landings are among the best-documented events in history. Key independent verification: (1) 382 kg of lunar samples with isotopic ratios impossible to manufacture. (2) Retroreflector arrays placed by Apollo 11, 14, and 15, used daily by observatories worldwide. (3) Independent tracking by multiple nations' space agencies, including the USSR. (4) Third-party verification by the Lunar Reconnaissance Orbiter, which photographed landing sites in 2009. Photo anomalies (missing stars, non-parallel shadows, flag motion) are explained by vacuum optics, uneven terrain, and inertial oscillation.",
            "casual": "Nah, it's not true. The moon landings really happened. The photo stuff looks weird because the moon has no atmosphere — so lighting, shadows, and everything look different from what we're used to. Plus, we literally left mirrors up there that anyone with a good laser can bounce a signal off. Even Russia tracked the whole thing and never disputed it.",
            "formal": "The claim that the Apollo moon landings were fabricated is not supported by the available evidence. Multiple independent lines of verification confirm the landings occurred: lunar geological samples with compositions inconsistent with terrestrial origin, retroreflector arrays actively used by international research institutions, and contemporaneous independent tracking by the Soviet space program. The photographic anomalies frequently cited have been thoroughly explained by the optical properties of the lunar environment, specifically the absence of atmospheric scattering."
        }
    },
    {
        "id": "div_imposter_syndrome",
        "prompt": "I got into a great university but I feel like they made a mistake accepting me.",
        "responses": {
            "concise": "That's imposter syndrome — it's extremely common and well-documented. They reviewed your application carefully. You earned this.",
            "warm": "What you're feeling has a name — imposter syndrome — and it's incredibly common, especially among people who actually belong. The admissions committee didn't make a mistake. They read your application, your essays, your whole story, and they chose you. It's okay to feel uncertain, but please know that this feeling says more about your humility than your ability.",
            "technical": "Imposter syndrome affects an estimated 70% of people at some point, with higher prevalence during transitions to selective environments. The phenomenon is characterized by attributing success to external factors (luck, error) rather than ability. Research shows it correlates with conscientiousness and high standards — traits that predict academic success. Admissions at competitive universities use multi-factor holistic review. The probability of clerical error in your specific case is negligible.",
            "casual": "Oh man, basically everyone feels that way. It even has a name — imposter syndrome. Here's the thing: admissions committees look at thousands of applications. They picked you on purpose. You might not feel like you fit in yet, but give it a semester. You'll find your people and realize you absolutely belong there.",
            "formal": "What you are experiencing is commonly known as imposter syndrome, a well-documented psychological phenomenon in which individuals doubt their accomplishments and fear being exposed as fraudulent. Research indicates that this is particularly prevalent among high-achieving individuals entering selective institutions. I would like to assure you that university admissions processes involve rigorous, multi-dimensional evaluation. Your acceptance reflects a deliberate assessment of your qualifications and potential."
        }
    },
    {
        "id": "div_restaurant",
        "prompt": "I want to open a restaurant with no cooking experience, no savings, and no business plan. What should I do?",
        "responses": {
            "concise": "Get cooking experience first — work in a restaurant kitchen. Build savings. Write a business plan. Then pursue funding. Skipping these steps is how restaurants fail in the first year.",
            "warm": "I love that you have this dream! But I'd be doing you a disservice if I said to jump in right now. The restaurant industry is tough even for experienced people. Here's what I'd suggest: start by working in a kitchen — even part time — to see if you love the reality of it. Meanwhile, save up and start sketching a business plan. You can absolutely get there, but building the foundation first will make the difference.",
            "technical": "The restaurant failure rate in the first year is approximately 60%, with undercapitalization and inexperience as primary factors. Recommended approach: (1) Gain kitchen experience — work 6-12 months in a restaurant similar to your concept. (2) Develop a business plan covering concept, target market, location analysis, startup costs, and 3-year financial projections. (3) Build capital — SBA loans typically require 20-30% owner equity. (4) Consider starting with a food truck or pop-up to validate your concept with lower overhead.",
            "casual": "Okay so I don't want to crush your dream, but real talk — jumping straight into a restaurant with no experience or money is basically setting money on fire. Here's what you should actually do: go work in a restaurant first. Like, in the kitchen. See if you still want to do it after six months of that. Save some money while you're at it, and start figuring out what kind of place you'd run. Then when you're ready, you'll actually have a shot.",
            "formal": "While your entrepreneurial ambition is commendable, launching a restaurant without culinary experience, capital reserves, or a business plan would expose you to significant risk. I would recommend a phased approach: first, acquire practical culinary and operational experience through employment in the restaurant industry; second, develop a comprehensive business plan that includes market analysis, financial projections, and a defined concept; third, establish adequate capitalization through personal savings and, when appropriate, external financing."
        }
    }
]

MODELS = [
    ("snowflake-arctic-embed-m", "Snowflake/snowflake-arctic-embed-m"),
    ("bge-m3", "BAAI/bge-m3"),
    ("nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5"),
]

def load_battery():
    cases = []
    for f in [BATTERY_50, WARMTH_20]:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    cases.append(json.loads(line))
    return cases

def encode(model, texts, model_name):
    if "nomic" in model_name:
        return model.encode(["search_document: " + t for t in texts],
                            normalize_embeddings=True, show_progress_bar=False)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def make_centroid(model, battery, model_name):
    better_texts = [f"User: {c['prompt']}\nAssistant: {c['better']}" for c in battery]
    worse_texts = [f"User: {c['prompt']}\nAssistant: {c['worse']}" for c in battery]
    better_embs = encode(model, better_texts, model_name)
    worse_embs = encode(model, worse_texts, model_name)
    direction = better_embs.mean(axis=0) - worse_embs.mean(axis=0)
    direction = direction / norm(direction)
    return direction

def run_diversity_test(model, centroid, model_name):
    results = {}
    all_scores_by_prompt = {}
    all_scores_by_style = {s: [] for s in STYLES}

    for case in DIVERSITY_CASES:
        prompt = case["prompt"]
        pid = case["id"]
        scores = {}
        for style in STYLES:
            text = f"User: {prompt}\nAssistant: {case['responses'][style]}"
            emb = encode(model, [text], model_name)[0]
            score = float(np.dot(emb, centroid))
            scores[style] = score
            all_scores_by_style[style].append(score)
        all_scores_by_prompt[pid] = scores
        results[pid] = scores

    # within-prompt variance: for each prompt, variance of the 5 style scores
    within_vars = []
    for pid, scores in all_scores_by_prompt.items():
        vals = list(scores.values())
        within_vars.append(np.var(vals))
    mean_within_var = float(np.mean(within_vars))

    # between-prompt variance: variance of the prompt means
    prompt_means = [np.mean(list(scores.values())) for scores in all_scores_by_prompt.values()]
    between_var = float(np.var(prompt_means))

    # style means across all prompts
    style_means = {s: float(np.mean(v)) for s, v in all_scores_by_style.items()}
    style_stds = {s: float(np.std(v)) for s, v in all_scores_by_style.items()}

    # which style wins most often
    style_wins = {s: 0 for s in STYLES}
    for pid, scores in all_scores_by_prompt.items():
        winner = max(scores, key=scores.get)
        style_wins[winner] += 1

    # spread: for each prompt, max - min score
    spreads = []
    for pid, scores in all_scores_by_prompt.items():
        vals = list(scores.values())
        spreads.append(max(vals) - min(vals))
    mean_spread = float(np.mean(spreads))

    return {
        "scores_by_prompt": results,
        "mean_within_prompt_variance": mean_within_var,
        "between_prompt_variance": between_var,
        "variance_ratio_between_over_within": between_var / mean_within_var if mean_within_var > 0 else float('inf'),
        "style_means": style_means,
        "style_stds": style_stds,
        "style_wins": style_wins,
        "mean_score_spread_within_prompt": mean_spread,
        "prompt_means": {pid: float(np.mean(list(s.values()))) for pid, s in all_scores_by_prompt.items()},
    }

def main():
    from sentence_transformers import SentenceTransformer

    battery = load_battery()
    print(f"Battery: {len(battery)}")
    print(f"Diversity cases: {len(DIVERSITY_CASES)} prompts × {len(STYLES)} styles = {len(DIVERSITY_CASES) * len(STYLES)} responses\n")

    all_results = {}
    for short_name, model_id in MODELS:
        print("=" * 70)
        print(f"MODEL: {short_name}")
        print("=" * 70)

        kwargs = {"trust_remote_code": True} if "nomic" in model_id else {}
        model = SentenceTransformer(model_id, **kwargs)

        centroid = make_centroid(model, battery, model_id)
        result = run_diversity_test(model, centroid, model_id)
        all_results[short_name] = result

        # print results
        print(f"\n--- SCORE DIVERSITY ---")
        print(f"  Mean within-prompt variance:    {result['mean_within_prompt_variance']:.6f}")
        print(f"  Between-prompt variance:        {result['between_prompt_variance']:.6f}")
        print(f"  Ratio (between/within):         {result['variance_ratio_between_over_within']:.2f}")
        print(f"  Mean score spread per prompt:   {result['mean_score_spread_within_prompt']:.4f}")

        print(f"\n  Style means (across all prompts):")
        for style in sorted(result['style_means'], key=result['style_means'].get, reverse=True):
            print(f"    {style:12s}: {result['style_means'][style]:+.4f} ± {result['style_stds'][style]:.4f}  (wins: {result['style_wins'][style]})")

        print(f"\n  Per-prompt scores:")
        for pid in DIVERSITY_CASES:
            pid_key = pid["id"]
            scores = result['scores_by_prompt'][pid_key]
            mean_s = result['prompt_means'][pid_key]
            sorted_styles = sorted(scores, key=scores.get, reverse=True)
            top = sorted_styles[0]
            bot = sorted_styles[-1]
            spread = scores[top] - scores[bot]
            print(f"    {pid_key:25s}: mean={mean_s:+.4f}  best={top}({scores[top]:+.4f})  worst={bot}({scores[bot]:+.4f})  spread={spread:.4f}")

        del model
        gc.collect()
        print()

    # save
    out_path = OUT_DIR / "score_diversity_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
