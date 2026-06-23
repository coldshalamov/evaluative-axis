param(
    [int]$Port = 8765,
    [string]$HostName = "127.0.0.1",
    [switch]$DebugLogs
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command uvx -ErrorAction SilentlyContinue)) {
    throw "uvx was not found on PATH. Install uv first, then retry: pip install uv"
}

$proxyArgs = @(
    "mcp-proxy",
    "--host=$HostName",
    "--port=$Port",
    "--pass-environment"
)

if ($DebugLogs) {
    $proxyArgs += "--debug"
}

$proxyArgs += @(
    "--",
    "uvx",
    "git+https://github.com/googlecolab/colab-mcp"
)

Write-Host "Starting Colab MCP proxy on http://${HostName}:$Port"
Write-Host "Streamable HTTP endpoint: http://${HostName}:$Port/mcp"
Write-Host "SSE endpoint:             http://${HostName}:$Port/sse"
Write-Host "Stop it with Ctrl+C."
Write-Host ""

& uvx @proxyArgs
