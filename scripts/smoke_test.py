"""Kiểm thử khói toàn bộ API. Cần artifacts đã sinh và API đang chạy.

    python scripts/build_artifacts.py all
    python scripts/serve.py api          # ở terminal khác
    python scripts/smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("APP_ROOT", str(ROOT))

import requests  # noqa: E402

from config import DATA_DIR  # noqa: E402
from scripts.colab_utils import use_utf8_console  # noqa: E402

use_utf8_console()

API = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
FLOWERS_DIR = DATA_DIR / "flowers" / "flower_photos"


def main() -> int:
    from ultralytics.utils import ASSETS

    health = requests.get(f"{API}/api/health", timeout=10).json()
    print("health:", health)
    enabled = [m for m, ok in health["models"].items() if ok]

    flower_img = next((FLOWERS_DIR / "sunflowers").glob("*.jpg")).read_bytes()
    bus_img = Path(ASSETS / "bus.jpg").read_bytes()

    if "classifier" in enabled:
        r = requests.post(f"{API}/api/classify", files={"file": flower_img}, data={"top_k": 3}, timeout=120)
        r.raise_for_status()
        print("classify:", r.json()["predictions"][0], "·", r.headers["X-Process-Time-ms"], "ms")

    if "detector" in enabled:
        r = requests.post(f"{API}/api/detect", files={"file": bus_img}, data={"conf": 0.3}, timeout=120)
        r.raise_for_status()
        print("detect:", r.json()["summary"])

    if "retrieval" in enabled:
        r = requests.post(f"{API}/api/search/text", json={"query": "a red flower", "k": 3}, timeout=120)
        r.raise_for_status()
        print("search/text:", [(x["label"], x["score"]) for x in r.json()["results"]])
        assert requests.get(API + r.json()["results"][0]["url"], timeout=30).headers["content-type"].startswith("image/")

        r = requests.post(f"{API}/api/search/image", files={"file": flower_img}, data={"k": 3}, timeout=120)
        r.raise_for_status()
        print("search/image:", [(x["label"], x["score"]) for x in r.json()["results"]])

        assert requests.get(f"{API}/api/gallery/999999", timeout=60).status_code == 404

    if "llm" in enabled:
        r = requests.post(f"{API}/api/chat/sync",
                          json={"message": "Phí giao hàng cho đơn 200.000đ là bao nhiêu?"}, timeout=300)
        r.raise_for_status()
        print("chat:", r.json()["answer"][:200])

        with requests.post(f"{API}/api/chat", json={"message": "Bảo hành đồ gia dụng bao lâu?"},
                           stream=True, timeout=300) as s:
            s.encoding = "utf-8"
            events = [json.loads(line[6:]) for line in s.iter_lines(decode_unicode=True) if line.startswith("data: ")]
        print("chat SSE:", [e["type"] for e in events][:5], "…",
              "".join(e.get("text", "") for e in events)[:120])

    # Các ca lỗi phải trả mã HTTP đúng (400/413/404 khi mô hình đã nạp, 503 khi chưa)
    status = requests.post(f"{API}/api/classify", files={"file": b"not an image"}, timeout=60).status_code
    assert status == (400 if "classifier" in enabled else 503), status
    assert requests.post(f"{API}/api/search/text", json={"query": ""}, timeout=60).status_code == 422
    assert requests.get(f"{API}/api/health", timeout=30).json()["status"] == "ok"
    print("✅ Tất cả smoke test đạt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
