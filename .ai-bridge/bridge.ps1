# VS Code AI Output Sync Bridge
# 把 VS Code 各 AI 扩展的输出自动同步到项目目录，供 OpenClaw Agent 读取

param(
    [string]$ProjectDir = "D:\project\DNN_Agent",
    [int]$IntervalSeconds = 5
)

$OutputDir = Join-Path $ProjectDir ".ai-bridge"
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    Write-Host "Created output dir: $OutputDir" -ForegroundColor Green
}

# Codex CLI 日志同步（已自动记录，只需创建软链接/快捷方式）
$CodexSessionsDir = Join-Path $env:USERPROFILE ".codex\sessions"
$CodexLink = Join-Path $OutputDir "codex-sessions"
if (-not (Test-Path $CodexLink)) {
    cmd /c mklink /J "$CodexLink" "$CodexSessionsDir" 2>$null
    if ($?) {
        Write-Host "Linked Codex sessions: $CodexLink -> $CodexSessionsDir" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "=== AI Output Bridge Running ===" -ForegroundColor Cyan
Write-Host "Project: $ProjectDir"
Write-Host "Output:  $OutputDir"
Write-Host "Interval: ${IntervalSeconds}s"
Write-Host ""
Write-Host "Now OpenClaw Agent can read:"
Write-Host "  - Codex CLI logs:    .ai-bridge/codex-sessions/"
Write-Host ""
Write-Host "For VS Code extensions (Copilot/ChatGPT/Claude), you need to:"
Write-Host "  1. Manually copy their output to .ai-bridge/copilot-chat.txt"
Write-Host "  2. Or use VS Code extension API (requires custom extension)"
Write-Host ""
Write-Host "Press Ctrl+C to stop."
Write-Host ""

while ($true) {
    # 更新一个心跳文件，让 Agent 知道桥接器在运行
    $heartbeat = @{
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss.fff")
        status = "running"
        project = $ProjectDir
    } | ConvertTo-Json -Depth 2
    
    $heartbeat | Set-Content -Path (Join-Path $OutputDir "heartbeat.json") -Encoding UTF8
    
    # 列出最新的 Codex 会话
    $latestCodex = Get-ChildItem -Path $CodexLink -Recurse -File -ErrorAction SilentlyContinue | 
        Sort-Object LastWriteTime -Descending | 
        Select-Object -First 1
    
    if ($latestCodex) {
        $meta = @{
            latestCodexSession = $latestCodex.FullName
            lastModified = $latestCodex.LastWriteTime.ToString("yyyy-MM-ddTHH:mm:ss")
            sizeMB = [math]::Round($latestCodex.Length / 1MB, 2)
        } | ConvertTo-Json -Depth 2
        $meta | Set-Content -Path (Join-Path $OutputDir "codex-meta.json") -Encoding UTF8
    }
    
    Start-Sleep -Seconds $IntervalSeconds
}
