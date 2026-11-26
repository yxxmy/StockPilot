"""Windows启动器的环境准备；不读取密钥，不触发行情或模型分析。"""
import argparse
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = tuple(sorted(ROOT.glob('requirements*.txt')))
INSTALL = ['-m', 'pip', 'install', '-r', 'requirements-ui.txt', '-r', 'requirements-forecast.txt']
CHECKS = (
    ['-m', 'pip', 'check'],
    # PDF 模型在用户查询正文时加载，避免可选的 PyTorch DLL 阻断行情首页。
    ['-c', 'import arch, streamlit, plotly, akshare, pandas, langgraph, sqlite3'],
)


def dependency_fingerprint():
    digest = hashlib.sha256()
    digest.update(f'{sys.version_info[:3]}:{sys.executable}'.encode())
    for path in REQUIREMENTS:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_checked(arguments):
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def installed_requirements_match():
    # 先在本机读取包版本，判断完整环境，无需连接软件源。
    import struct
    from importlib.metadata import PackageNotFoundError, version
    if sys.version_info[:2] != (3, 12) or struct.calcsize('P') != 8:
        return False
    try:
        from packaging.requirements import Requirement
        for path in REQUIREMENTS:
            for line in path.read_text(encoding='utf-8').splitlines():
                if not line.strip() or line.startswith(('#', '-r ')):
                    continue
                req = Requirement(line.strip())
                if not req.specifier.contains(version(req.name)):
                    return False
    except (ImportError, PackageNotFoundError):
        return False
    return True


def probe():
    if not installed_requirements_match():
        return False
    return all(subprocess.run([sys.executable, *arguments], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
               for arguments in CHECKS)


def prepare(*, check_only=False, managed=False):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('This launcher requires Python 3.12. Existing environment was not replaced.')
    needs_install = not check_only and not installed_requirements_match()
    if needs_install:
        print('[2/4] Installing dependencies; first installation may take several minutes.', flush=True)
        run_checked(INSTALL)
    print('[3/4] Checking dependencies...', flush=True)
    try:
        for arguments in CHECKS:
            run_checked(arguments)
    except subprocess.CalledProcessError:
        if check_only or not managed or needs_install:
            raise
        print('[2/4] Repairing the shared environment after a failed dependency check.', flush=True)
        run_checked([*INSTALL, '--force-reinstall'])
        for arguments in CHECKS:
            run_checked(arguments)
    if not check_only:
        environment = Path(sys.prefix) if managed else ROOT / '.venv'
        marker = environment / 'stockpilot-ready.json'
        marker.write_text(json.dumps({'fingerprint': dependency_fingerprint()}), encoding='utf-8')


def remember_environment():
    directory = Path(os.environ['LOCALAPPDATA']) / 'StockPilot'
    pending = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory, delete=False) as stream:
            pending = Path(stream.name)
            stream.write(sys.executable + '\n')
        pending.replace(directory / 'python-ready.txt')
    except OSError:
        print('The environment is usable, but its path could not be saved for reuse.', flush=True)
    finally:
        if pending is not None:
            with suppress(OSError):
                pending.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--download-model', action='store_true')
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--managed', action='store_true')
    parser.add_argument('--remember', action='store_true')
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args(argv)
    if args.probe:
        return 0 if probe() else 1
    os.chdir(ROOT)
    try:
        prepare(check_only=args.check or args.reuse, managed=args.managed)
        if args.check:
            return 0
        if args.remember:
            remember_environment()
        if args.download_model:
            run_checked(['-m', 'rag', 'download-model'])
            return 0
        if not args.prepare_only:
            from launch_local import launch
            return launch()
        return 0
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, subprocess.CalledProcessError):
        print('Environment/startup failed. See the error above. No automatic analysis was submitted.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
