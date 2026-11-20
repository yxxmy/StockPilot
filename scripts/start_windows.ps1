param([switch]$DownloadModel, [switch]$PrepareOnly, [switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
function Find-CompatiblePython {
    param([switch]$ReadyOnly)
    # 校验实际可执行的解释器，跳过 Windows Store 占位入口。
    $candidates = @(Get-Command python.exe -CommandType Application -All -ErrorAction SilentlyContinue | ForEach-Object Source)
    $localHint = Join-Path $PSScriptRoot 'python-local.txt'
    if (Test-Path -LiteralPath $localHint) { $candidates = @((Get-Content -LiteralPath $localHint -Encoding UTF8 -First 1)) + $candidates }
    if ($env:VIRTUAL_ENV) { $candidates += Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe' }
    if ($env:CONDA_PREFIX) { $candidates += Join-Path $env:CONDA_PREFIX 'python.exe' }
    foreach ($key in @('HKCU:\Software\Python\PythonCore\3.12\InstallPath','HKLM:\Software\Python\PythonCore\3.12\InstallPath')) {
        if (Test-Path $key) { $candidates += (Get-Item $key).GetValue('ExecutablePath') }
    }
    foreach ($candidate in $candidates) {
        if (-not $candidate -or $candidate -match '\\WindowsApps\\' -or -not (Test-Path -LiteralPath $candidate)) { continue }
        $valid = & $candidate -I -c "import sys, struct, venv; print(int(sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8))"
        if ($LASTEXITCODE -eq 0 -and $valid -eq '1') {
            if ($ReadyOnly) {
                & $candidate (Join-Path $PSScriptRoot 'bootstrap.py') --probe
                if ($LASTEXITCODE -ne 0) { continue }
            }
            return $candidate
        }
    }
}
try {
    $projectRoot = Split-Path -Parent $PSScriptRoot
    Set-Location -LiteralPath $projectRoot
    $pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
    $reuse = $false
    $localReady = $false
    if (Test-Path -LiteralPath $pythonExe) {
        & $pythonExe (Join-Path $PSScriptRoot 'bootstrap.py') --probe
        $localReady = $LASTEXITCODE -eq 0
    }
    if (-not $localReady) {
        $readyPython = Find-CompatiblePython -ReadyOnly
        if ($readyPython) {
            $pythonExe = $readyPython
            $reuse = $true
            Write-Host "[1/4] Using the existing StockPilot environment: $pythonExe"
        }
    }
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        if ($CheckOnly) { throw 'Python environment is missing. Run Start_StockPilot.bat to install.' }
        # 独立安装到项目目录，不更改系统PATH，也不需要管理员权限。
        $runtimeRoot = Join-Path $projectRoot '.runtime'
        $basePython = Join-Path $runtimeRoot 'python312\python.exe'
        if (-not (Test-Path -LiteralPath $basePython)) {
            $basePython = Find-CompatiblePython
        }
        if (-not $basePython) {
            New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
            $installer = Join-Path $runtimeRoot 'python-3.12.10-amd64.exe'
            Write-Host '[1/4] Downloading official Python 3.12.10 (Windows x64)...'
            [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
            Invoke-WebRequest 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe' -OutFile $installer -UseBasicParsing -TimeoutSec 300
            # 下载失败或签名不可信时停止，绝不执行未经验证的安装程序。
            Write-Host '[1/4] Download complete. Checking publisher signature...'
            $signature = Get-AuthenticodeSignature -LiteralPath $installer
            if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Python Software Foundation(?:,|$)') {
                throw 'Python installer signature verification failed. Installer was not executed.'
            }
            $installTarget = Join-Path $runtimeRoot 'python312'
            $installArgs = '/quiet InstallAllUsers=0 Include_pip=1 Include_test=0 Include_launcher=0 PrependPath=0 Shortcuts=0 TargetDir="' + $installTarget + '"'
            Write-Host '[1/4] Installing Python. Please keep this window open.'
            $process = Start-Process -FilePath $installer -ArgumentList $installArgs -WindowStyle Hidden -PassThru
            while (-not $process.WaitForExit(15000)) { Write-Host "[1/4] Python installer is still running..." }
            $process.Refresh()
            if ($process.ExitCode -notin @(0, 3010)) { throw "Python installation failed (exit $($process.ExitCode))." }
            $basePython = Join-Path $installTarget 'python.exe'
            if (-not (Test-Path -LiteralPath $basePython)) { $basePython = Find-CompatiblePython }
            if (-not $basePython) { throw 'Installer finished but a usable Python 3.12 x64 could not be located.' }
        }
        Write-Host "[1/4] Preparing the project environment using $basePython"
        if (Test-Path -LiteralPath (Join-Path $projectRoot '.venv')) {
            throw 'An incomplete .venv already exists. Rename it manually before retrying; no files were deleted.'
        }
        & $basePython -m venv (Join-Path $projectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    }
    $bootstrapArgs = @((Join-Path $PSScriptRoot 'bootstrap.py'))
    if ($reuse) { $bootstrapArgs += '--reuse' }
    if ($DownloadModel) { $bootstrapArgs += '--download-model' }
    if ($PrepareOnly) { $bootstrapArgs += '--prepare-only' }
    if ($CheckOnly) { $bootstrapArgs += '--check' }
    & $pythonExe @bootstrapArgs
    if ($LASTEXITCODE -ne 0) { throw "StockPilot exited with code $LASTEXITCODE. Installation/network errors are shown above; retry after resolving them." }
} catch {
    Write-Host ("Startup failed at script line " + $_.InvocationInfo.ScriptLineNumber) -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
