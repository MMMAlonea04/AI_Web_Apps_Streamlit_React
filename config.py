"""Cấu hình tập trung. Mọi giá trị đều ghi đè được bằng biến môi trường."""
import os
from pathlib import Path

try:
    import torch
except ImportError:  # CI/test API chỉ cần fastapi, không cần cài torch
    torch = None

ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parent))
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

# Mô hình (đổi tên model = đổi biến môi trường, không sửa code)
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", str(ART_DIR / "detector" / "yolo11n.pt"))
CLIP_MODEL = os.environ.get("CLIP_MODEL", "openai/clip-vit-base-patch32")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
LLM_MODEL = os.environ.get(
    "LLM_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct" if DEVICE == "cuda" else "Qwen/Qwen2.5-0.5B-Instruct",
)

# Bật/tắt từng mô hình để tiết kiệm bộ nhớ, ví dụ ENABLED_MODELS="classifier,detector"
ENABLED_MODELS = {
    m.strip() for m in os.environ.get("ENABLED_MODELS", "classifier,detector,retrieval,llm").split(",") if m.strip()
}

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "8"))
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:8501").split(",")

# Ngưỡng điểm cosine tối thiểu để coi là "tài liệu có liên quan" (chatbot RAG).
# Số đo thực tế: câu hỏi trong phạm vi đạt 0.55–0.67, câu ngoài phạm vi chỉ 0.10–0.12.
# Dưới ngưỡng này thì không gọi LLM — tránh để model trả lời bằng kiến thức ngoài tài liệu.
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.30"))


def resolve_path(path: str) -> Path:
    """Dữ liệu lưu đường dẫn tương đối so với ROOT để mang sang máy khác (Docker, HF Spaces)."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
