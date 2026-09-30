"""Tiện ích chạy tiến trình nền (API, Streamlit) và Cloudflare Tunnel.

Dùng được cả trên Colab/Linux lẫn Windows.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
PROCS: dict[str, subprocess.Popen] = {}
IS_WINDOWS = platform.system() == "Windows"

CLOUDFLARED_ASSETS = {
    "Linux": "cloudflared-linux-amd64",
    "Windows": "cloudflared-windows-amd64.exe",
}


def use_utf8_console() -> None:
    """Console Windows mặc định cp1252 — in tiếng Việt sẽ lỗi/hỏng chữ nếu không đổi sang UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)


def download(url: str, dest: Path) -> Path:
    dest = Path(dest)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print("↓", url, flush=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def kill_port(port: int) -> None:
    """Giải phóng cổng đang bị chiếm. Bỏ qua nếu chưa cài psutil."""
    try:
        import psutil
    except ImportError:
        return
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN and conn.pid:
            try:
                psutil.Process(conn.pid).kill()
            except psutil.Error:
                pass


def start(name: str, cmd: list[str], port: int | None = None, env: dict | None = None) -> subprocess.Popen:
    if name in PROCS and PROCS[name].poll() is None:
        PROCS[name].terminate()
        PROCS[name].wait(10)
    if port is not None:
        kill_port(port)
    LOGS.mkdir(exist_ok=True)
    log = open(LOGS / f"{name}.log", "w", encoding="utf-8")
    PROCS[name] = subprocess.Popen(
        cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}
    )
    return PROCS[name]


def log_tail(name: str, n: int = 3000) -> str:
    path = LOGS / f"{name}.log"
    return path.read_text(encoding="utf-8", errors="replace")[-n:] if path.exists() else "(chưa có log)"


def wait_http(url: str, name: str, timeout: int = 900, ready=lambda r: r.ok):
    import requests

    t0 = time.time()
    while time.time() - t0 < timeout:
        if PROCS[name].poll() is not None:
            raise RuntimeError(f"{name} đã dừng:\n{log_tail(name)}")
        try:
            r = requests.get(url, timeout=5)
            if ready(r):
                return r
        except requests.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"{name} chưa sẵn sàng sau {timeout}s — xem logs/{name}.log")


def start_api(port: int = 8000, timeout: int = 900, env: dict | None = None) -> dict:
    """Khởi động FastAPI rồi chờ tới khi mô hình nạp xong."""
    start("api", [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", str(port)], port, env)
    t0 = time.time()
    body = wait_http(f"http://localhost:{port}/api/health", "api", timeout).json()
    print(f"API sẵn sàng sau {time.time() - t0:.0f}s:", body, flush=True)
    failed = [m for m, ok in body["models"].items() if not ok]
    if failed:
        print("⚠️ Mô hình nạp lỗi:", failed, "— xem logs/api.log", flush=True)
    return body


def tunnel(name: str, port: int, timeout: int = 60) -> str:
    """Mở Cloudflare Quick Tunnel tới một cổng local, trả về URL công khai."""
    system = platform.system()
    if system not in CLOUDFLARED_ASSETS:
        raise RuntimeError(f"Tunnel chưa hỗ trợ trên {system} — hãy chạy trên Colab/Linux")
    asset = CLOUDFLARED_ASSETS[system]
    exe = download(f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}", ROOT / asset)
    if not IS_WINDOWS:
        exe.chmod(0o755)

    key = f"tunnel_{name}"
    if key in PROCS and PROCS[key].poll() is None:
        PROCS[key].terminate()
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / f"{key}.log"
    PROCS[key] = subprocess.Popen(
        [str(exe), "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
        stdout=open(log_path, "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(timeout):
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log_path.read_text(encoding="utf-8", errors="replace"))
        if m:
            return m.group(0)
        time.sleep(1)
    raise TimeoutError("Không tạo được tunnel — xem " + str(log_path))


GIST_API = "https://api.github.com/gists"
GIST_FILE = "api.json"


def secret(name: str) -> str | None:
    """Đọc từ biến môi trường, hoặc từ Colab Secrets khi chạy trên Colab.

    Giá trị được cắt khoảng trắng/xuống dòng vì dán vào Colab Secrets rất dễ dính kèm.
    """
    if os.environ.get(name):
        return os.environ[name].strip()
    try:
        from google.colab import userdata  # chỉ có trên Colab

        value = userdata.get(name)
        return value.strip() if value else None
    except Exception:
        return None


def publish_api_url(url: str, gist_id: str | None = None) -> bool:
    """Công bố địa chỉ backend hiện tại vào gist để giao diện ở origin khác tự nối.

    Giao diện đọc gist này qua `VITE_API_DISCOVERY` (xem README mục 6.2). Cần GH_TOKEN (scope `gist`)
    và GIST_ID, đặt qua biến môi trường hoặc Colab Secrets. Thiếu thì chỉ in nhắc, không làm gì.

    Lưu ý trên Colab: `!python scripts/serve.py` là tiến trình con, không đọc được `userdata` của kernel,
    nên phải đặt biến môi trường trong kernel trước khi chạy (tiến trình con kế thừa).
    """
    import requests

    gist_id = gist_id or secret("GIST_ID")
    token = secret("GH_TOKEN")
    if not gist_id or not token:
        print(
            "ℹ️ Chưa công bố địa chỉ backend: thiếu GH_TOKEN hoặc GIST_ID.\n"
            "   Trên Colab, `!python …` là tiến trình con nên phải đặt trước trong kernel:\n"
            "     from google.colab import userdata\n"
            "     os.environ['GH_TOKEN'] = userdata.get('GH_TOKEN')\n"
            "     os.environ['GIST_ID'] = userdata.get('GIST_ID')\n"
            "   (cũng cần bật Notebook access cho hai secret — xem README mục 6.2)",
            flush=True,
        )
        return False

    content = {"api": url, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    r = requests.patch(
        f"{GIST_API}/{gist_id}",
        json={"files": {GIST_FILE: {"content": json.dumps(content, indent=2)}}},
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        timeout=20,
    )
    if r.status_code >= 300:
        print(f"⚠️ Không cập nhật được gist công bố ({r.status_code}): {r.text[:200]}", flush=True)
        return False
    print(f"📣 Đã công bố địa chỉ backend: {url}", flush=True)
    return True
