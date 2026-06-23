param(
    [int]$Port = 8765,
    [string]$HostName = "127.0.0.1",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CodexArgs
)

$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$endpoint = "http://${HostName}:$Port/mcp"

try {
    Invoke-RestMethod -Uri "http://${HostName}:$Port/status" -TimeoutSec 2 | Out-Null
} catch {
    Write-Warning "Colab MCP proxy did not respond at http://${HostName}:$Port/status."
    Write-Warning "Start it in another terminal first: .\scripts\start-colab-mcp-http.ps1"
}

$urlConfig = "mcp_servers.colab.url=`"$endpoint`""

& codex `
    -C $workspaceRoot `
    -c $urlConfig `
    -c "mcp_servers.colab.enabled=true" `
    @CodexArgs
