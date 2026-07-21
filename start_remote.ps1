[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$Subdomain = "valuation-grid",
    [switch]$AllowEmptyPositions
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$ProjectRoot = $PSScriptRoot
$AppPath = Join-Path $ProjectRoot "app.py"
$PositionsPath = Join-Path $ProjectRoot "data\positions.json"
$LogsDir = Join-Path $ProjectRoot "logs"
$AppOutLog = Join-Path $LogsDir "app.stdout.log"
$AppErrLog = Join-Path $LogsDir "app.stderr.log"
$TunnelOutLog = Join-Path $LogsDir "localtunnel.stdout.log"
$TunnelErrLog = Join-Path $LogsDir "localtunnel.stderr.log"
$PublicUrl = "https://$Subdomain.loca.lt"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

function Get-ProcessSnapshot {
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
}

function Stop-ProcessSet([int[]]$ProcessIds) {
    foreach ($processId in ($ProcessIds | Sort-Object -Descending -Unique)) {
        if ($processId -eq $PID) {
            continue
        }
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
}

function Stop-ValuationGridProcesses {
    $all = Get-ProcessSnapshot
    $portOwnerIds = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
    $escapedAppPath = [regex]::Escape($AppPath)
    $fullAppPattern = '(?i)' + $escapedAppPath + '(?:[\s"]|$)'

    # Windows PowerShell 5.1 的 Get-Process 没有 CommandLine；必须使用 CIM。
    $appIds = @($all | Where-Object {
        $_.Name -match '^python(?:w)?\.exe$' -and
        $_.CommandLine -and
        ($_.CommandLine -match $fullAppPattern -or
         (($portOwnerIds -contains $_.ProcessId) -and
          $_.CommandLine -match '(?i)(^|[\s\"])app\.py([\s\"]|$)'))
    } | Select-Object -ExpandProperty ProcessId)

    $tunnelRoots = @($all | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match '(?i)localtunnel' -and
        $_.CommandLine -match "--port\s+$Port(?:\s|$)" -and
        $_.CommandLine -match "--subdomain\s+$([regex]::Escape($Subdomain))(?:\s|$)"
    } | Select-Object -ExpandProperty ProcessId)

    $tunnelIds = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($id in $tunnelRoots) {
        [void]$tunnelIds.Add([int]$id)
    }
    do {
        $added = $false
        foreach ($process in $all) {
            if ($tunnelIds.Contains([int]$process.ParentProcessId) -and
                -not $tunnelIds.Contains([int]$process.ProcessId)) {
                [void]$tunnelIds.Add([int]$process.ProcessId)
                $added = $true
            }
        }
    } while ($added)

    Stop-ProcessSet (@($tunnelIds))
    Stop-ProcessSet $appIds

    $deadline = (Get-Date).AddSeconds(10)
    do {
        $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if (-not $listener) {
            return
        }
        Start-Sleep -Milliseconds 200
    } while ((Get-Date) -lt $deadline)

    $owners = @($listener | Select-Object -ExpandProperty OwningProcess -Unique) -join ', '
    throw "端口 $Port 仍被进程 $owners 占用，已停止启动以避免误杀其他程序。"
}

function Assert-PositionsHealthy {
    if (-not (Test-Path -LiteralPath $PositionsPath)) {
        throw "缺少 $PositionsPath，拒绝启动。"
    }

    try {
        $raw = [IO.File]::ReadAllText($PositionsPath, [Text.UTF8Encoding]::new($false))
        $data = $raw | ConvertFrom-Json
    }
    catch {
        throw "positions.json 不是有效 UTF-8 JSON，拒绝启动：$($_.Exception.Message)"
    }

    if ($null -eq $data.funds) {
        throw "positions.json 缺少 funds，拒绝启动。"
    }
    $fundCount = @($data.funds.PSObject.Properties).Count
    if ($fundCount -eq 0 -and -not $AllowEmptyPositions) {
        throw "positions.json 当前为空。为防止误清空，默认拒绝启动；确认确实需要空仓时传入 -AllowEmptyPositions。"
    }
    return $fundCount
}

function Wait-LocalHealth($AppProcess) {
    $deadline = (Get-Date).AddSeconds(60)
    do {
        $AppProcess.Refresh()
        if ($AppProcess.HasExited) {
            $tail = if (Test-Path -LiteralPath $AppErrLog) {
                (Get-Content -Tail 30 -LiteralPath $AppErrLog) -join [Environment]::NewLine
            } else { "没有错误日志" }
            throw "app.py 启动后退出（exit=$($AppProcess.ExitCode)）。`n$tail"
        }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
            if ($health.status -eq "ok") {
                return
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    } while ((Get-Date) -lt $deadline)
    throw "本地服务 60 秒内未通过健康检查，请查看 $AppErrLog。"
}

function Stop-CurrentTunnel {
    $all = Get-ProcessSnapshot
    $ids = @($all | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match '(?i)localtunnel' -and
        $_.CommandLine -match "--port\s+$Port(?:\s|$)" -and
        $_.CommandLine -match "--subdomain\s+$([regex]::Escape($Subdomain))(?:\s|$)"
    } | Select-Object -ExpandProperty ProcessId)
    Stop-ProcessSet $ids
}

function Start-VerifiedTunnel($NpxPath) {
    for ($attempt = 1; $attempt -le 2; $attempt++) {
        Stop-CurrentTunnel
        Start-Sleep -Milliseconds 500

        $tunnel = Start-Process -FilePath $NpxPath `
            -ArgumentList @('--yes', 'localtunnel', '--port', "$Port", '--subdomain', $Subdomain) `
            -WorkingDirectory $ProjectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $TunnelOutLog `
            -RedirectStandardError $TunnelErrLog `
            -PassThru

        $deadline = (Get-Date).AddSeconds(60)
        do {
            try {
                $health = Invoke-RestMethod -Uri "$PublicUrl/health" `
                    -Headers @{'bypass-tunnel-reminder' = 'true'} `
                    -TimeoutSec 8
                if ($health.status -eq "ok") {
                    return $tunnel
                }
            }
            catch {
                Start-Sleep -Seconds 2
            }
        } while ((Get-Date) -lt $deadline)

        Stop-CurrentTunnel
        if ($attempt -lt 2) {
            Write-Host "公网隧道第 $attempt 次验证失败，正在重试……" -ForegroundColor Yellow
        }
    }

    $tail = if (Test-Path -LiteralPath $TunnelErrLog) {
        (Get-Content -Tail 30 -LiteralPath $TunnelErrLog) -join [Environment]::NewLine
    } else { "没有错误日志" }
    throw "localtunnel 两次启动均未通过公网健康检查。`n$tail"
}

$fundCount = Assert-PositionsHealthy
Stop-ValuationGridProcesses

$python = (Get-Command python.exe -ErrorAction Stop).Source
$npx = (Get-Command npx.cmd -ErrorAction Stop).Source
$env:PYTHONPATH = $ProjectRoot
$env:APP_PORT = $Port

$appProcess = Start-Process -FilePath $python `
    -ArgumentList @('-u', ('"{0}"' -f $AppPath)) `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $AppOutLog `
    -RedirectStandardError $AppErrLog `
    -PassThru

Wait-LocalHealth $appProcess
$tunnelProcess = Start-VerifiedTunnel $npx

Write-Host ""
Write-Host "valuation-grid 启动成功" -ForegroundColor Green
Write-Host "持仓校验: $fundCount 个" -ForegroundColor Green
Write-Host "本地: http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "远端: $PublicUrl" -ForegroundColor Green
Write-Host "日志: $LogsDir" -ForegroundColor DarkGray
Write-Host "app PID: $($appProcess.Id)  tunnel PID: $($tunnelProcess.Id)" -ForegroundColor DarkGray
