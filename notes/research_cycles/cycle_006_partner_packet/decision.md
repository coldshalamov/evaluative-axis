# Cycle 006 Decision

Date: June 27, 2026

## Decision

Use `paper/partner_packet_v1/brief.md` as the current external-facing summary
artifact for the repo.

The paper draft is now materially closer to the current evidence ladder, but it
still needs a full pass before it should replace the packet as the primary
external summary.

## What To Do Next

1. Update the full paper draft so it reflects:
   - the new packet structure throughout the whole manuscript, not just the
     front matter and new evidence sections;
   - objective reranking as the central practical evidence;
   - raw `good/bad` as an honest failure in the current setup;
   - process sensitivity as promising but below the frozen training gate.
2. Expand the packet with cost comparisons and confidence intervals once those
   are measured directly rather than estimated.
3. Keep the packet generator wired to saved outputs so future runs refresh the
   brief instead of creating stale prose.
