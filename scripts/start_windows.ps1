param([switch]$DownloadModel, [switch]$PrepareOnly, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
function Get-PythonCandidates {
    $localHint = Join-Path $PSScriptRoot 'python-local.txt'
    if (Test-Path -LiteralPath $localHint) { Get-Content -LiteralPath $localHint -Encoding UTF8 -First 1 }
    if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe' }
    if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX 'python.exe' }
    Get-Command python.exe -CommandType Application -All -ErrorAction SilentlyContinue | ForEach-Object Source
    foreach ($key in @('HKCU:\Software\Python\PythonCore\3.12\InstallPath','HKLM:\Software\Python\PythonCore\3.12\InstallPath')) {
        if (Test-Path $key) { (Get-Item $key).GetValue('ExecutablePath') }
    }
}
function Find-CompatiblePython($candidates, [switch]$ReadyOnly) {
    # 失效路径、Windows Store 占位程序、失效的 venv 都只影响自己的候选项。
    foreach ($raw in ($candidates | Select-Object -Unique)) {
        if (-not $raw) { continue }
        $candidate = $raw.Trim()
        if (-not [IO.Path]::IsPathRooted($candidate) -or $candidate -match '\\WindowsApps\\' -or
            -not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        if ($ReadyOnly -and $candidate.StartsWith($environmentRoot + '\',[StringComparison]::OrdinalIgnoreCase)) {
            $candidateEnvironment = Split-Path -Parent (Split-Path -Parent $candidate)
            if (-not (Test-Path -LiteralPath (Join-Path $candidateEnvironment 'stockpilot-ready.json') -PathType Leaf)) { continue }
        }
        try {
            if ($ReadyOnly) {
                & $candidate $bootstrapScript --probe 2>$null | Out-Null
            } else {
                & $candidate -I -c "import sys, struct, venv; sys.exit(not (sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8))" 2>$null | Out-Null
            }
            if ($LASTEXITCODE -eq 0) { return $candidate }
        } catch {
            continue
        }
    }
}
function Find-ReadyPython {
    $readyCandidates = @(Join-Path $projectRoot '.venv\Scripts\python.exe')
    if (Test-Path -LiteralPath $rememberedPython -PathType Leaf) {
        $readyCandidates += Get-Content -LiteralPath $rememberedPython -Encoding UTF8 -First 1 -ErrorAction SilentlyContinue
    }
    $readyCandidates += $managedPython
    $readyCandidates += @(Get-ChildItem -LiteralPath $environmentRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName 'Scripts\python.exe' })
    $readyCandidates += @(Get-PythonCandidates)
    Find-CompatiblePython $readyCandidates -ReadyOnly
}
function Get-DependencyKey {
    $specs = @(Get-ChildItem -LiteralPath $projectRoot -File -Filter 'requirements*.txt' | Sort-Object Name |
        ForEach-Object { $_.Name; [IO.File]::ReadAllText($_.FullName).Replace("`r`n", "`n") })
    $digest = [Security.Cryptography.SHA256]::Create()
    try {
        [BitConverter]::ToString($digest.ComputeHash([Text.Encoding]::UTF8.GetBytes(($specs -join "`n")))).Replace('-', '').ToLowerInvariant()
    } finally { $digest.Dispose() }
}
function Wait-EnvironmentLock {
    $lockPath = Join-Path $stateRoot 'prepare.lock'
    while ($true) {
        try { return [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') } catch [IO.IOException] {
            # 仅在另一进程持有锁时等待，其他磁盘错误直接报告。
            if (($_.Exception.GetBaseException().HResult -band 65535) -notin @(32, 33)) { throw }
            Write-Host '[1/4] Another StockPilot window is preparing the environment. Waiting to reuse it...'
            Start-Sleep -Seconds 5
        }
    }
}
function Install-StockPilotPython {
    New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
    $installer = Join-Path $runtimeRoot 'python-3.12.10-amd64.exe'
    Write-Host '[1/4] Downloading official Python 3.12.10 (Windows x64)...'
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $installer -UseBasicParsing -TimeoutSec 300 | Out-Null
    Write-Host '[1/4] Download complete. Checking publisher signature...'
    $signature = Get-AuthenticodeSignature -LiteralPath $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Python Software Foundation(?:,|$)') {
        throw 'Python installer signature verification failed. Installer was not executed.'
    }
    $installTarget = Join-Path $runtimeRoot 'python312'
    $installArgs = '/quiet InstallAllUsers=0 Include_pip=1 Include_test=0 Include_launcher=0 PrependPath=0 Shortcuts=0 TargetDir="' + $installTarget + '"'
    Write-Host '[1/4] Installing Python. Please keep this window open.'
    $process = Start-Process -FilePath $installer -ArgumentList $installArgs -WindowStyle Hidden -PassThru
    while (-not $process.WaitForExit(15000)) { Write-Host '[1/4] Python installer is still running...' }
    $process.Refresh()
    if ($process.ExitCode -notin @(0, 3010)) { throw "Python installation failed (exit $($process.ExitCode))." }
    $installedPython = Find-CompatiblePython (@((Join-Path $installTarget 'python.exe')) + @(Get-PythonCandidates))
    if (-not $installedPython) { throw 'Installer finished but a usable Python 3.12 x64 could not be located.' }
    return $installedPython
}
try {
    $projectRoot = Split-Path -Parent $PSScriptRoot
    Set-Location -LiteralPath $projectRoot
    $bootstrapScript = Join-Path $PSScriptRoot 'bootstrap.py'
    $stateRoot = Join-Path $env:LOCALAPPDATA 'StockPilot'
    $rememberedPython = Join-Path $stateRoot 'python-ready.txt'
    $runtimeRoot = Join-Path $stateRoot 'runtime'
    $environmentRoot = Join-Path $stateRoot 'environments'
    $managedEnvironment = Join-Path $environmentRoot (Get-DependencyKey)
    $managedPython = Join-Path $managedEnvironment 'Scripts\python.exe'
    $pythonExe = Find-ReadyPython
    if (-not $pythonExe) {
        if ($CheckOnly) { throw 'Python environment is missing. Run Start_StockPilot.bat to install.' }
        New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
        $setupLock = Wait-EnvironmentLock
        try {
            # 等待期间另一窗口可能已经完成安装。
            $pythonExe = Find-ReadyPython
            if (-not $pythonExe) {
                $managedMarker = Join-Path $managedEnvironment 'stockpilot-ready.json'
                if (Test-Path -LiteralPath $managedMarker -PathType Leaf) {
                    Remove-Item -LiteralPath $managedMarker
                }
                if (-not (Find-CompatiblePython @($managedPython))) {
                    $baseCandidates = @((Join-Path $runtimeRoot 'python312\python.exe'),
                        (Join-Path $projectRoot '.runtime\python312\python.exe')) + @(Get-PythonCandidates)
                    $basePython = Find-CompatiblePython $baseCandidates
                    if (-not $basePython) {
                        $basePython = Install-StockPilotPython
                    }
                    Write-Host "[1/4] Preparing shared StockPilot environment using $basePython"
                    & $basePython -m venv $managedEnvironment
                    if ($LASTEXITCODE -ne 0) { throw 'Could not create the shared Python environment.' }
                }
                & $managedPython $bootstrapScript --prepare-only --managed --remember
                if ($LASTEXITCODE -ne 0) { throw 'Could not prepare the shared StockPilot environment.' }
                $pythonExe = $managedPython
            }
        } finally { $setupLock.Dispose() }
    }
    Write-Host "[1/4] Using the existing StockPilot environment: $pythonExe"
    $bootstrapArgs = @($bootstrapScript, '--reuse')
    if ($DownloadModel) { $bootstrapArgs += '--download-model' }
    if ($PrepareOnly) { $bootstrapArgs += '--prepare-only' }
    if ($CheckOnly) { $bootstrapArgs += '--check' } else { $bootstrapArgs += '--remember' }
    & $pythonExe @bootstrapArgs
    if ($LASTEXITCODE -ne 0) { throw "StockPilot exited with code $LASTEXITCODE. Installation/network errors are shown above; retry after resolving them." }
} catch {
    Write-Host ("Startup failed at script line " + $_.InvocationInfo.ScriptLineNumber) -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
