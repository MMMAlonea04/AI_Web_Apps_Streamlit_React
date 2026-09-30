"""Chạy backend, hai giao diện và tunnel công khai.

    python scripts/serve.py all            # build React → API → Streamlit → tunnel
    python scripts/serve.py api            # chỉ FastAPI
    python scripts/serve.py streamlit
    python scripts/serve.py web            # npm install + npm run build
    python scripts/serve.py tunnel         # Cloudflare Tunnel cho 8000 và 8501 (Colab/Linux)

Trên Windows chỉ cần `api` + `streamlit` + `web`; tunnel nên chạy trên Colab.

Link tunnel đổi mỗi phiên, nên sau khi mở tunnel script tự công bố địa chỉ backend vào gist (cần
GH_TOKEN + GIST_ID) để giao diện React ở origin khác — ví dụ Netlify — tự nối. Xem README mục 6.2.
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.colab_utils import (  # noqa: E402
    log_tail,
    publish_api_url,
    start,
    start_api,
    tunnel,
    use_utf8_console,
    wait_http,
)

use_utf8_console()


def node_version() -> tuple[int, int] | None:
    node = shutil.which("node")
    if not node:
        return None
    raw = subprocess.run([node, "-v"], capture_output=True, text=True).stdout.strip().lstrip("v")
    try:
        major, minor = raw.split(".")[:2]
        return int(major), int(minor)
    except ValueError:
        return None


def ensure_node() -> None:
    """Vite 8 cần Node ^20.19 hoặc >= 22.12 — Colab mặc định cài Node cũ hơn."""
    ver = node_version()
    if ver and ((ver[0] == 20 and ver[1] >= 19) or ver >= (22, 12)):
        print("node:", ".".join(map(str, ver)), flush=True)
        return

    have = ".".join(map(str, ver)) if ver else "chưa cài"
    print(f"⚠️ Node {have} — Vite 8 cần ≥ 20.19 hoặc ≥ 22.12", flush=True)
    if platform.system() == "Linux" and hasattr(os, "geteuid") and os.geteuid() == 0:
        print("→ Cài Node 22 từ nodesource…", flush=True)
        for cmd in ("curl -fsSL https://deb.nodesource.com/setup_22.x | bash -", "apt-get install -y nodejs"):
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("node:", ".".join(map(str, node_version() or ())), flush=True)
        return

    raise SystemExit(
        "Hãy cài Node ≥ 22.12 (https://nodejs.org/en/download) rồi chạy lại,\n"
        "hoặc bỏ qua bước build React: python scripts/serve.py api\n"
        "(API vẫn phục vụ web/dist nếu bản build đã có sẵn)."
    )


def build_web() -> None:
    web = ROOT / "web"
    if not (web / "package.json").exists():
        raise SystemExit("Không thấy web/package.json")
    ensure_node()
    npm = shutil.which("npm") or "npm"
    for cmd in ([npm, "install", "--no-audit", "--no-fund", "--loglevel=error"], [npm, "run", "build"]):
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=web, check=True)
    dist = web / "dist"
    print("React build:", dist if dist.exists() else "⚠️ không thấy web/dist")


def run_streamlit(port: int, api_port: int) -> None:
    start("streamlit", [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",  # cần tắt khi chạy sau proxy/tunnel
    ], port, env={"API_URL": os.environ.get("API_URL", f"http://localhost:{api_port}")})
    wait_http(f"http://localhost:{port}/_stcore/health", "streamlit", timeout=120)
    print(f"Streamlit: http://localhost:{port}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all", choices=["all", "api", "streamlit", "web", "tunnel"])
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--streamlit-port", type=int, default=8501)
    parser.add_argument("--no-tunnel", action="store_true", help="bỏ qua bước mở tunnel")
    parser.add_argument("--enabled-models", default=None, help="ghi đè ENABLED_MODELS cho API")
    parser.add_argument("--api-timeout", type=int, default=900)
    parser.add_argument("--publish", default=None, help="địa chỉ backend công bố vào gist (mặc định: link tunnel của API)")
    args = parser.parse_args()

    env = {"ENABLED_MODELS": args.enabled_models} if args.enabled_models else None
    urls: dict[str, str] = {}

    if args.command in ("all", "web"):
        build_web()
        if args.command == "web":
            return 0

    if args.command in ("all", "api", "tunnel"):
        start_api(args.api_port, timeout=args.api_timeout, env=env)
        urls["API"] = f"http://localhost:{args.api_port}"
        urls["Swagger"] = f"http://localhost:{args.api_port}/docs"

    if args.command in ("all", "streamlit"):
        run_streamlit(args.streamlit_port, args.api_port)
        urls["Streamlit"] = f"http://localhost:{args.streamlit_port}"

    if (args.command in ("all", "tunnel")) and not args.no_tunnel:
        for name, port in (("react", args.api_port), ("streamlit", args.streamlit_port)):
            try:
                urls[name.capitalize()] = tunnel(name, port)
            except Exception as exc:
                print(f"⚠️ Tunnel {name}: {exc}", flush=True)

    published = args.publish or urls.get("React")
    if published:
        publish_api_url(published)

    print("\n--- Địa chỉ ---")
    for name, url in urls.items():
        print(f"{name:>9}: {url}")
    print("\nLog trong:", (ROOT / 'logs').relative_to(ROOT))
    if args.command == "all" and not urls.get("React"):
        print("\nGợi ý: nếu API không lên, xem\n" + log_tail("api", 2000))
    return 0


if __name__ == "__main__":
    sys.exit(main())
