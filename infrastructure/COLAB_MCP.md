# Colab MCP Workspace Setup

This workspace keeps Colab MCP opt-in. Nothing here adds a global Codex MCP
server, so other projects do not inherit Colab tool schemas.

## Start The Local Proxy

Run this in terminal 1:

```powershell
.\scripts\start-colab-mcp-http.ps1
```

This wraps the official stdio Colab MCP server with a localhost MCP proxy:

- Streamable HTTP: `http://127.0.0.1:8765/mcp`
- SSE: `http://127.0.0.1:8765/sse`
- Status: `http://127.0.0.1:8765/status`

Stop it with `Ctrl+C`.

## Launch Codex With Colab MCP

Run this in terminal 2:

```powershell
.\scripts\codex-with-colab-mcp.ps1
```

That launches Codex in this workspace with a temporary config override:

```powershell
-c 'mcp_servers.colab.url="http://127.0.0.1:8765/mcp"'
```

The override applies only to that Codex process. Normal Codex sessions in other
projects continue to use the usual global config and will not see the Colab MCP
tools.

## Why This Shape

The official `googlecolab/colab-mcp` server is a stdio MCP server. Codex can also
connect to Streamable HTTP MCP servers, so the local `mcp-proxy` bridge gives us
a clean on-demand boundary:

1. No permanent global MCP registration.
2. No always-running Colab server.
3. Tool schemas only appear in Codex sessions launched through this workspace
   script.
