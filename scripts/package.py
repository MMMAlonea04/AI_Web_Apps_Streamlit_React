"""Đóng gói dự án thành zip để nộp hoặc đẩy lên nơi khác.

    python scripts/package.py            # -> dist/ai_web_apps.zip
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.colab_utils import use_utf8_console  # noqa: E402

use_utf8_console()

SKIP = (
    "web/node_modules",
    "web/dist",
    "logs",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "cloudflared",
    "runs",
    "data/flowers",
    "data/coco128",
    "data/coco128.zip",
    "data/flower_photos.tgz",
    "dist",
)


def main() -> None:
    out_dir = ROOT / "dist"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "ai_web_apps.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if not p.is_file() or any(s in rel for s in SKIP):
                continue
            z.write(p, f"ai_web_apps/{rel}")
    print(f"{out} · {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
