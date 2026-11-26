"""启动器离线测试：外部安装用明确替身，不安装或请求正式接口。"""
from types import SimpleNamespace
import os
from unittest.mock import Mock
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import bootstrap, launch_local


@pytest.mark.skipif(shutil.which('powershell.exe') is None, reason='Windows PowerShell required')
def test_windows_missing_environment_branch_preserves_chinese_comments(tmp_path):
    # 真实执行Windows PowerShell，不下载、不安装；在网络边界主动中止。
    source = Path(bootstrap.__file__).with_name('start_windows.ps1')
    assert source.read_bytes().startswith(b'\xef\xbb\xbf')
    script_dir = tmp_path / 'share with spaces' / 'scripts'
    script_dir.mkdir(parents=True)
    target = script_dir / source.name
    target.write_bytes(source.read_bytes())
    escaped = str(target).replace("'", "''")
    command = "function Get-Command { }; function Test-Path { if ($args[-1] -like 'HK*') { return $false }; Microsoft.PowerShell.Management\\Test-Path @args }; function Invoke-WebRequest { throw 'TEST_ONLY_NETWORK_STOP' }; & '" + escaped + "'"
    local_app_data = tmp_path / 'local app data'
    env = {k:v for k,v in os.environ.items() if k.upper() not in ('VIRTUAL_ENV', 'CONDA_PREFIX', 'LOCALAPPDATA')}
    env['LOCALAPPDATA'] = str(local_app_data)
    result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', command],
                            capture_output=True, timeout=30, env=env)
    assert result.returncode == 1
    assert b'TEST_ONLY_NETWORK_STOP' in result.stdout
    assert b'Downloading official Python' in result.stdout
    assert (local_app_data / 'StockPilot/runtime').is_dir()
    assert not (script_dir.parent / '.venv').exists()


def test_prepare_reuses_verified_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, 'ROOT', tmp_path)
    (tmp_path / '.venv').mkdir()
    calls = []
    monkeypatch.setattr(bootstrap, 'run_checked', calls.append)
    monkeypatch.setattr(bootstrap, 'installed_requirements_match', lambda: False)
    bootstrap.prepare()
    assert 'install' in calls[0]
    assert (tmp_path / '.venv/stockpilot-ready.json').exists()
    calls.clear()
    monkeypatch.setattr(bootstrap, 'installed_requirements_match', lambda: True)
    bootstrap.prepare()
    assert len(calls) == 2
    assert not any('install' in call for call in calls)


def test_install_failure_does_not_mark_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, 'ROOT', tmp_path)
    (tmp_path / '.venv').mkdir()
    def fail(arguments):
        raise bootstrap.subprocess.CalledProcessError(1, 'TEST ONLY installer')
    monkeypatch.setattr(bootstrap, 'run_checked', fail)
    with pytest.raises(bootstrap.subprocess.CalledProcessError):
        bootstrap.prepare()
    assert not (tmp_path / '.venv/stockpilot-ready.json').exists()


def test_check_only_never_installs_or_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, 'ROOT', tmp_path)
    calls = []
    monkeypatch.setattr(bootstrap, 'run_checked', calls.append)
    bootstrap.prepare(check_only=True)
    assert len(calls) == 2
    assert not list(tmp_path.iterdir())


def test_requirements_change_invalidates_fingerprint(tmp_path, monkeypatch):
    requirement = tmp_path / 'requirements.txt'
    monkeypatch.setattr(bootstrap, 'REQUIREMENTS', (requirement,))
    requirement.write_text('TEST ONLY version 1')
    original = bootstrap.dependency_fingerprint()
    requirement.write_text('TEST ONLY version 2')
    assert bootstrap.dependency_fingerprint() != original


def test_dead_server_does_not_open_browser(monkeypatch):
    process = Mock()
    process.poll.return_value = 1
    browser = Mock()
    monkeypatch.setattr(launch_local.subprocess, 'Popen', Mock(return_value=process))
    monkeypatch.setattr(launch_local.webbrowser, 'open', browser)
    assert launch_local.launch() == 1
    browser.assert_not_called()
    process.terminate.assert_not_called()


def test_healthy_server_opens_own_url_and_waits(monkeypatch):
    process = Mock()
    process.poll.return_value = 0
    process.wait.return_value = 0
    monkeypatch.setattr(launch_local, 'choose_port', lambda: 18555)
    popen = Mock(return_value=process)
    monkeypatch.setattr(launch_local.subprocess, 'Popen', popen)
    ready = Mock()
    browser = Mock(return_value=True)
    monkeypatch.setattr(launch_local, 'wait_ready', ready)
    monkeypatch.setattr(launch_local.webbrowser, 'open', browser)
    assert launch_local.launch() == 0
    ready.assert_called_once_with(process, 'http://127.0.0.1:18555')
    browser.assert_called_once_with('http://127.0.0.1:18555')
    command = popen.call_args.args[0]
    assert command[command.index('--server.address') + 1] == '127.0.0.1'
    assert command[command.index('--server.headless') + 1] == 'true'


def test_startup_timeout_stops_only_owned_process(monkeypatch):
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(launch_local.subprocess, 'Popen', Mock(return_value=process))
    monkeypatch.setattr(launch_local, 'wait_ready', Mock(side_effect=RuntimeError('TEST ONLY timeout')))
    browser = Mock()
    monkeypatch.setattr(launch_local.webbrowser, 'open', browser)
    assert launch_local.launch() == 1
    process.terminate.assert_called_once()
    browser.assert_not_called()


def test_health_checks_ignore_system_proxy(monkeypatch):
    response = Mock()
    response.status = 200
    response.read.return_value = b'ok'
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    opener = Mock()
    opener.open.return_value = response
    build = Mock(return_value=opener)
    monkeypatch.setattr(launch_local, 'build_opener', build)
    launch_local.wait_ready(SimpleNamespace(poll=lambda: None), 'http://127.0.0.1:18555')
    assert build.call_args.args[0].proxies == {}
    opener.open.assert_called_once_with('http://127.0.0.1:18555/_stcore/health', timeout=2)

def test_ready_environment_without_marker_skips_pip_install(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, 'ROOT', tmp_path)
    (tmp_path / '.venv').mkdir()
    monkeypatch.setattr(bootstrap, 'installed_requirements_match', lambda: True)
    calls = []
    monkeypatch.setattr(bootstrap, 'run_checked', calls.append)
    bootstrap.prepare()
    assert calls and all('install' not in call for call in calls)


def test_probe_rejects_failed_import_even_when_versions_match(monkeypatch):
    monkeypatch.setattr(bootstrap, 'installed_requirements_match', lambda: True)
    runner = Mock(side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)])
    monkeypatch.setattr(bootstrap.subprocess, 'run', runner)
    assert not bootstrap.probe()
    assert runner.call_count == 2


def test_ready_file_remembers_the_actual_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.setattr(bootstrap.sys, 'executable', str(tmp_path / 'another folder/Scripts/python.exe'))
    bootstrap.remember_environment()
    marker = tmp_path / 'StockPilot/python-ready.txt'
    assert marker.read_text(encoding='utf-8').strip() == bootstrap.sys.executable
    assert list(marker.parent.iterdir()) == [marker]


@pytest.mark.skipif(os.name != 'nt' or shutil.which('powershell.exe') is None, reason='Windows PowerShell required')
def test_new_extraction_reuses_environment_from_previous_folder(tmp_path):
    source = Path(bootstrap.__file__).with_name('start_windows.ps1')
    previous = tmp_path / 'first download'
    current = tmp_path / 'second download'
    ready_python = previous / '.venv/Scripts/python.exe'
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(previous / '.venv')],
                   capture_output=True, check=True, timeout=30)
    state = tmp_path / 'local app data'
    env = {**os.environ, 'LOCALAPPDATA': str(state), 'STOCKPILOT_TEST_READY_PYTHON': str(ready_python)}
    for project in (previous, current):
        scripts = project / 'scripts'
        scripts.mkdir(parents=True, exist_ok=True)
        target = scripts / source.name
        target.write_bytes(source.read_bytes())
        (scripts / 'bootstrap.py').write_text("""import os
import sys
from pathlib import Path
if Path(sys.executable) != Path(os.environ['STOCKPILOT_TEST_READY_PYTHON']):
    sys.exit(1)
if '--probe' not in sys.argv:
    assert '--reuse' in sys.argv and '--prepare-only' in sys.argv
    ready_file = Path(os.environ['LOCALAPPDATA']) / 'StockPilot/python-ready.txt'
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.write_text(sys.executable, encoding='utf-8')
""", encoding='utf-8')
        result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                 '-File', str(target), '-PrepareOnly'],
                                capture_output=True, env=env, timeout=30)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert b'Using the existing StockPilot environment' in result.stdout
        assert b'Preparing shared StockPilot environment' not in result.stdout
    assert not (current / '.venv').exists()
    assert not (state / 'StockPilot/environments').exists()


@pytest.mark.skipif(os.name != 'nt' or shutil.which('powershell.exe') is None, reason='Windows PowerShell required')
def test_simultaneous_extractions_prepare_one_shared_environment(tmp_path):
    source = Path(bootstrap.__file__).with_name('start_windows.ps1')
    state = tmp_path / 'local app data'
    env = {**os.environ, 'LOCALAPPDATA': str(state), 'STOCKPILOT_TEST_PYTHON': sys._base_executable}
    command = ("function Get-Command { [pscustomobject]@{Source=$env:STOCKPILOT_TEST_PYTHON} }; "
               "function Test-Path { if ($args[-1] -like 'HK*') { return $false }; "
               "Microsoft.PowerShell.Management\\Test-Path @args }; "
               "& $env:STOCKPILOT_TEST_LAUNCHER -PrepareOnly")
    scripts = []
    for name in ('first download', 'second download'):
        folder = tmp_path / name / 'scripts'
        folder.mkdir(parents=True)
        target = folder / source.name
        target.write_bytes(source.read_bytes())
        (folder / 'bootstrap.py').write_text("""import sys
import time
from pathlib import Path
ready = Path(sys.prefix) / 'test-ready.txt'
if '--probe' in sys.argv:
    sys.exit(not ready.exists())
if '--managed' in sys.argv:
    assert '--prepare-only' in sys.argv
    time.sleep(1)
    with ready.open('a', encoding='utf-8') as log:
        log.write('prepared\\n')
    (Path(sys.prefix) / 'stockpilot-ready.json').touch()
else:
    assert '--reuse' in sys.argv and ready.exists()
""", encoding='utf-8')
        scripts.append(target)

    processes = [
        subprocess.Popen(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', command],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env={**env, 'STOCKPILOT_TEST_LAUNCHER': str(target)})
        for target in scripts
    ]
    try:
        for process in processes:
            output = process.communicate(timeout=45)
            assert process.returncode == 0, output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    environments = list((state / 'StockPilot/environments').iterdir())
    assert len(environments) == 1
    assert (environments[0] / 'test-ready.txt').read_text(encoding='utf-8') == 'prepared\n'
    assert all(not (target.parent.parent / '.venv').exists() for target in scripts)
