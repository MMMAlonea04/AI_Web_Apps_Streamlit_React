"""Đóng gói repo thành nội dung Space rồi đẩy lên Hugging Face Spaces (Docker SDK).

    hf auth login                                   # token có quyền write, hoặc đặt HF_TOKEN
    python scripts/deploy_space.py --repo <user>/ai-web-apps --dry-run
    python scripts/deploy_space.py --repo <user>/ai-web-apps

Chỉ file Space cần được đưa lên `dist/space/`, nên README dự án không bị thay bằng thẻ Space
(`deploy/hf-space/README.md` được copy thành `README.md` của Space). Trọng số CLIP, MiniLM và
Qwen do container tải lúc khởi động, không nằm trong image.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.colab_utils import use_utf8_console  # noqa: E402

use_utf8_console()

STAGE = ROOT / "dist" / "space"
SPACE_README = ROOT / "deploy" / "hf-space" / "README.md"
FILES = ("Dockerfile", "config.py", "requirements.txt")
DIRS = ("api", "core", "web", "data/kb", "data/gallery", "artifacts")
SKIP = ("node_modules", "dist", "__pycache__", ".pytest_cache")


def build_stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for rel in FILES:
        dest = STAGE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    for rel in DIRS:
        src = ROOT / rel
        if not src.exists():
            raise SystemExit(f"❌ Thiếu {rel}/ — Space sẽ lỗi lúc chạy")
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            sub = path.relative_to(src).as_posix()
            if any(part in SKIP for part in sub.split("/")):
                continue
            dest = STAGE / rel / sub
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    shutil.copy2(SPACE_README, STAGE / "README.md")


def copy_sources() -> list[str]:
    """Nguồn của mọi lệnh COPY trong Dockerfile (bỏ qua COPY --from=<stage khác>)."""
    sources = []
    for raw in (STAGE / "Dockerfile").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.upper().startswith("COPY "):
            continue
        args = [tok for tok in line.split()[1:] if not tok.startswith("--")]
        if len(args) >= 2 and not args[0].startswith("/"):
            sources.append(args[0].rstrip("/"))
    return sources


def report() -> list[tuple[str, int, int]]:
    """(thư mục gốc, số file, số byte) để in ra trước khi đẩy."""
    rows = []
    for rel in DIRS:
        files = [p for p in (STAGE / rel).rglob("*") if p.is_file()]
        rows.append((rel, len(files), sum(p.stat().st_size for p in files)))
    return rows


def space_subdomain(repo_id: str) -> str:
    return repo_id.replace("/", "-").replace("_", "-").replace(".", "-").lower()


def upload(repo_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("❌ Thiếu huggingface_hub — chạy: pip install -r requirements-dev.txt")

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as exc:
        raise SystemExit(f"❌ Chưa đăng nhập Hugging Face (hf auth login, hoặc đặt HF_TOKEN): {exc}")
    print("Đăng nhập:", user, flush=True)

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=STAGE,
        commit_message="Deploy backend AI Web Apps từ GitHub MMMAlonea04/AI_Web_Apps_Streamlit_React",
    )
    print("✅ Đã đẩy lên", f"https://huggingface.co/spaces/{repo_id}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=os.environ.get("HF_SPACE_ID"), help="<user>/<tên-space>; hoặc đặt HF_SPACE_ID")
    parser.add_argument("--dry-run", action="store_true", help="chỉ dựng staging và in nội dung, không đẩy")
    parser.add_argument("--private", action="store_true", help="tạo Space ở chế độ private")
    args = parser.parse_args()

    build_stage()
    missing = [s for s in copy_sources() if not any(STAGE.glob(s))]
    if missing:
        print("❌ Dockerfile COPY những thứ không có trong staging:", ", ".join(missing), flush=True)
        return 1

    rows = report()
    total_files = sum(n for _, n, _ in rows) + len(FILES) + 1
    total_bytes = sum(sz for _, _, sz in rows) + sum((STAGE / rel).stat().st_size for rel in FILES)
    print(f"Staging: dist/space · {total_files} file · {total_bytes / 1e6:.1f} MB", flush=True)
    for rel, n, size in rows:
        print(f"  {rel + '/':<16} {n:>4} file · {size / 1e6:>6.1f} MB", flush=True)

    if not args.repo:
        print("\n⚠️ Chưa có --repo (hoặc HF_SPACE_ID) — dừng ở bước dựng staging.", flush=True)
        return 0

    _, health = space_urls(args.repo)
    if args.dry_run:
        print("\n(dry-run) Sẽ tạo/cập nhật Space rồi upload thư mục trên.", flush=True)
    else:
        upload(args.repo, args.private)

    print(f"""
Bước tiếp theo:
  1. Theo dõi build ở tab Logs: https://huggingface.co/spaces/{args.repo} (~5–10 phút)
  2. Settings → Variables and secrets, thêm
       CORS_ORIGINS=https://<tên-site>.netlify.app,http://localhost:5173
  3. Chờ Space Running rồi kiểm: curl {health}/api/health
     (lần đầu mất 1–3 phút vì tải trọng số CLIP, MiniLM, Qwen)
  4. Netlify: đặt biến môi trường VITE_API_URL={health} rồi deploy lại giao diện""", flush=True)
    return 0


def space_urls(repo_id: str) -> tuple[str, str]:
    return f"https://huggingface.co/spaces/{repo_id}", f"https://{space_subdomain(repo_id)}.hf.space"


if __name__ == "__main__":
    sys.exit(main())
