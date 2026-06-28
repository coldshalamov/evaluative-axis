# Cycle 009 Decision

Date: June 27, 2026

## Decision

Promote this cycle as the current model-landscape map for the clean direct-only
battery.

It strengthens the repo's capability-gap framing:

- not just BGE is weak
- there is a real local-model gradient
- but Gemini still stands well above the free/local pack on the direct-only
  battery

## What To Do Next

1. Use this sweep as the default answer when asked whether free Hugging Face
   models already match Gemini on the clean battery: they do not.
2. Keep the claim narrow. This cycle supports model-quality sensitivity, not a
   proof that parameter count alone is the cause.
3. If budget allows, run the raw `good/bad` vs proxy-word conflict protocol on
   one or two of the best local models from this sweep, especially
   `snowflake/snowflake-arctic-embed-m` or `jinaai/jina-embeddings-v2-small-en`.
4. Prefer future partner-facing claims to lean on:
   - objective reranking
   - targeted-axis controlled batteries
   - process-potential diagnostics
   rather than on raw broad-word axes alone.
