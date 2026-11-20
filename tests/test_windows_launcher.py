"""启动器离线测试：外部安装用明确替身，不安装或请求正式接口。"""
from types import SimpleNamespace
import os
from unittest.mock import Mock
from pathlib import Path
import shutil
import subprocess

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
    result = subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', command],
                            capture_output=True, timeout=30, env={k:v for k,v in os.environ.items() if k.upper() not in ('VIRTUAL_ENV','CONDA_PREFIX')})
    assert result.returncode == 1
    assert b'TEST_ONLY_NETWORK_STOP' in result.stdout
    assert b'Downloading official Python' in result.stdout
    assert (script_dir.parent / '.runtime').is_dir()
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
