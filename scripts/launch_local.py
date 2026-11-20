"""启动自己的本地服务，健康检查成功后才打开网页。"""
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def choose_port():
    # 每次选空闲端口，不连接其他项目或TEST ONLY验收服务。
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def wait_ready(process, url, *, timeout=180):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('The local server exited before it became ready.')
        try:
            with opener.open(url + '/_stcore/health', timeout=2) as response:
                if response.status == 200 and response.read(32).strip() == b'ok' and process.poll() is None:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError('Local server startup timed out. See terminal output.')


def launch(*, open_browser=True):
    port = choose_port()
    url = f'http://127.0.0.1:{port}'
    print('[4/4] Starting StockPilot. Keep this window open; Ctrl+C stops the service.', flush=True)
    process = subprocess.Popen([
        sys.executable, str(ROOT / 'web_app.py'), '--server.address', '127.0.0.1',
        '--server.port', str(port), '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
    ], cwd=ROOT)
    try:
        wait_ready(process, url)
        print(f'StockPilot is ready: {url}', flush=True)
        if open_browser and not webbrowser.open(url):
            print('Open the address above in your browser.', flush=True)
        return process.wait()
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        # 仅停止本次启动的子进程，不杀死其他Python/Streamlit进程。
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == '__main__':
    raise SystemExit(launch())
