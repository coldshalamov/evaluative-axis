# Cycle 006 Results

Date: June 27, 2026

## Output Artifacts

- `paper/draft.md`
- `paper/partner_packet_v1/brief.md`
- `paper/partner_packet_v1/packet_summary.json`
- `paper/partner_packet_v1/figures/figure_selection_lift.svg`
- `paper/partner_packet_v1/figures/figure_word_vs_targeted.svg`
- `paper/partner_packet_v1/figures/figure_process_signal.svg`

## Key Findings

### 1. The repo now has a single current-state packet

The new packet compresses the current serious evidence into one brief with
linked figures and explicit claim boundaries.

### 2. The packet keeps the strongest result first

The objective reranking evidence is now the lead practical claim rather than
older HH-centric correlation framing.

### 3. The packet preserves the honest negatives

It explicitly shows:

- raw `good/bad` failed on the 50-case conflict battery;
- cheap OSS embedders do not match Gemini on the frozen objective suites;
- the process-potential gate still fails.

### 4. The packet is reproducible

The figures and brief are generated from current saved JSON outputs rather than
hand-maintained numbers.

### 5. The paper front matter now matches the current evidence ladder

The abstract, contributions, conclusion, and experiment sections in
`paper/draft.md` now reflect:

- objective reranking as the strongest practical evidence;
- raw `good/bad` as a negative result in the current setup;
- process sensitivity as real but still below the frozen training-readiness
  gate.

## Interpretation

This does not add new empirical evidence. It makes the existing evidence much
more inspectable and much harder to misstate.

That matters because the project is now at the stage where presentation quality
directly affects whether external readers understand the claim boundaries.
