# Colab Testing Status

Date: June 22, 2026

## What Works

- Chrome browser control can claim and operate the Colab tab.
- The notebook runtime connects successfully to a Python 3 Google Compute backend.
- Short cells can be filled, run, and read back through the browser controller.
- A stdout smoke test completed: `print("colab-ok-2")` printed successfully.
- Dependency installation completed in Colab with `pip install -U sentence-transformers datasets vaderSentiment`.

## What Does Not Work Yet

- The formal `colab_mcp` bridge still does not unlock notebook-editing tools.
- `open_colab_browser_connection` timed out after the Colab runtime was connected.
- Source inspection shows `colab_mcp` starts an ephemeral localhost websocket and opens a Colab URL containing `mcpProxyToken` and `mcpProxyPort`; stale tabs and active MCP servers can point at different ports.
- Long code-cell pastes through browser control are unreliable because Colab's virtual editor can leave hidden cell fragments, causing syntax errors before code executes.

## Runtime Capability

- Runtime: Python 3 Google Compute backend
- GPU: unavailable
- `torch`: `2.11.0+cpu`
- `torch.cuda.is_available()`: `False`
- `nvidia-smi`: unavailable

## Decision

Use Colab browser control for short setup/probe cells only. For full experiment
runs, prefer the stable local script path or create/upload a real notebook file
instead of pasting long code into the scratchpad editor.
