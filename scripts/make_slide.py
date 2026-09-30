"""Sinh slide thuyết trình từ số liệu thật trong artifacts/.

    python scripts/make_slide.py

Tạo 2 file trong docs/:
  slide-cach-lam.pptx          — 1 trang tóm tắt (đúng yêu cầu "1 slide")
  slide-cach-lam-5-trang.pptx  — bản 5 trang để trình bày

Số trên slide đọc trực tiếp từ artifacts/*/metrics.json nên không bị lệch khỏi kết quả thật.
"""
from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
GITHUB = "github.com/MMMAlonea04/AI_Web_Apps_Streamlit_React"

INK = RGBColor(0x1B, 0x24, 0x33)
ACC = RGBColor(0x0B, 0x6E, 0x99)
MUT = RGBColor(0x5E, 0x6A, 0x7A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xEE, 0xF3, 0xF6)


def load(path: str) -> dict:
    f = ROOT / path
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


CLF, DET = load("artifacts/classifier/metrics.json"), load("artifacts/detector/metrics.json")
RET, RAG = load("artifacts/retrieval/metrics.json"), load("artifacts/rag_metrics.json")
P10 = RET.get("text_to_image_precision@10") or {}


def f4(d: dict, key: str, fallback: str = "n/a") -> str:
    return f"{d[key]:.4f}" if key in d else fallback


# ------------------------------------------------------------------ tiện ích
def new_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def box(s, left, top, width, height, lines, size=11, bold=False, color=INK,
        align=PP_ALIGN.LEFT, space=5, anchor=None):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    if anchor:
        tf.vertical_anchor = anchor
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_after = Pt(space)
        for r in p.runs:
            r.font.size, r.font.bold, r.font.color.rgb = Pt(size), bold, color
    return tb


def rect(s, left, top, width, height, fill=LIGHT):
    sh = s.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def table(s, left, top, rows, widths, size=9.5):
    shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top),
                               Inches(sum(widths)), Inches(0.34 * len(rows)))
    tbl = shape.table
    for col, w in zip(tbl.columns, widths):
        col.width = Inches(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = val
            cell.margin_left = cell.margin_right = Pt(5)
            cell.margin_top = cell.margin_bottom = Pt(2)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.bold = i == 0 or (j == 0 and i > 0)
                    r.font.color.rgb = WHITE if i == 0 else INK
    return tbl


def head(s, title: str, sub: str | None = None):
    rect(s, 0, 0, 13.333, 0.06, ACC)
    box(s, 0.5, 0.28, 12.4, 0.5, [title], 22, True, ACC)
    if sub:
        box(s, 0.5, 0.83, 12.4, 0.3, [sub], 11, False, MUT)


def band(s, text: str, size=9):
    rect(s, 0.5, 6.95, 12.35, 0.42, LIGHT)
    box(s, 0.62, 7.01, 12.1, 0.32, [text], size, False, MUT)


def pic(s, name: str, left, top, width=None, height=None):
    s.shapes.add_picture(str(SHOTS / name), Inches(left), Inches(top),
                         Inches(width) if width else None, Inches(height) if height else None)


def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    return prs


FEATURES = [
    ("1. Phân loại ảnh", "ResNet-18 fine-tune · 5 loài hoa", "daisy 91.2% trong 11.6 ms"),
    ("2. Phát hiện đối tượng", "YOLO11n · 80 lớp COCO", "cake 93.0% · fork 96.1% · 108 ms"),
    ("3. Tìm kiếm ảnh", "CLIP ViT-B/32 + FAISS · kho 628 ảnh", "5/5 kết quả đúng loài · 20 ms"),
    ("4. Chatbot RAG", "Qwen2.5 + MiniLM + FAISS · 6 tài liệu", "trả lời đúng kèm Nguồn (3) · 3,1 s"),
]

AI_DECL = [
    "Hỗ trợ làm bài:",
    "   Kimi Code CLI (Moonshot AI) — mô hình deepseek-flash",
    "",
    "Mô hình trong sản phẩm:",
    "   ResNet-18 · YOLO11n 8.4.165 (AGPL-3.0)",
    "   CLIP ViT-B/32 · MiniLM-L12-v2 · Qwen2.5-1.5B",
    "",
    "Thư viện: torch 2.14.0 · ultralytics 8.4.165 ·",
    "   transformers 5.17.0 · faiss-cpu 1.15.1 ·",
    "   fastapi 0.141.1 · streamlit 1.64.0 · vite 8.3.1",
]

LIMITS = [
    "•  Gõ tiếng Việt KHÔNG DẤU làm truy xuất sai tài liệu",
    "    (0.674 → 0.359), chatbot trả lời sai theo.",
    "•  mAP đo trên COCO128 chính là dữ liệu mô hình đã học.",
    "•  Hit@1 chỉ trên 10 câu — mẫu nhỏ.",
    "•  API chưa có xác thực và giới hạn tần suất.",
]


def page_features(prs, title, sub, with_images: bool, fields):
    """Bảng 4 chức năng; dùng chung cho bản 1 trang và bản 5 trang."""
    s = new_slide(prs)
    head(s, title, sub)
    y = 1.35
    for name, model, res in fields:
        rect(s, 0.5, y, 12.35, 0.95)
        box(s, 0.65, y + 0.06, 3.4, 0.3, [name], 12, True, ACC)
        box(s, 4.1, y + 0.06, 4.2, 0.3, [model], 10.5, False, INK)
        box(s, 8.4, y + 0.06, 4.3, 0.3, [res], 10, False, MUT)
        y += 1.05
    if with_images:
        for i, (name, cap) in enumerate([("01-streamlit-tong-quan.jpg", "1. Phân loại — Streamlit"),
                                         ("05-react-phat-hien.jpg", "2. Phát hiện — React"),
                                         ("06-react-tim-anh.jpg", "3. Tìm kiếm ảnh — React")]):
            pic(s, name, 0.5 + i * 3.2, y + 0.15, width=3.0)
            box(s, 0.5 + i * 3.2, y + 1.8, 3.0, 0.3, [cap], 9.5, True, MUT)
    return s


# ========================================================== BẢN 1 TRANG
def build_one_page() -> Path:
    """Đúng 1 trang: yêu cầu của thầy là '1 slide ngắn gọn về cách làm'."""
    prs = new_prs()
    s = new_slide(prs)
    head(s, "Cách làm — 3 tầng, 5 bước, 4 chức năng chạy thật",
         "Vườn Hoa AI · Lập trình Web nâng cao · nộp 01/10/2026 · " + GITHUB)
    box(s, 0.5, 1.3, 6.2, 0.3, ["CÁCH LÀM"], 13, True, ACC)
    box(s, 0.5, 1.68, 6.2, 3.5, [
        "1.  Tách 3 tầng: core/ chỉ suy luận → api/ bọc HTTP → giao diện chỉ gọi API.",
        "2.  Một nơi giữ mô hình: FastAPI nạp 4 mô hình một lần; Streamlit và React đều là client mỏng.",
        "3.  Cấu hình bằng biến môi trường (config.py): đổi mô hình, bật/tắt mô hình, đổi ngưỡng — không sửa code.",
        "4.  Sinh trọng số và chỉ mục trên Colab T4 (build_artifacts.py, 5 stage cố định) rồi mang về repo.",
        "5.  Đo trước khi nộp: mỗi mô hình lưu metrics.json; độ trễ lấy từ header X-Process-Time-ms.",
        "6.  Kiểm thử: 7 test pytest chạy 0,4 giây không cần GPU.",
    ], 10.5, False, INK, space=7)
    box(s, 0.5, 5.15, 6.2, 0.3, ["KIẾN TRÚC"], 13, True, ACC)
    box(s, 0.5, 5.5, 6.2, 1.3, [
        "Trình duyệt → Streamlit (8501) ─┐",
        "                                ├→ FastAPI (8000) → core/ (4 mô hình)",
        "Trình duyệt → React (web/dist) ─┘",
        "Sau khi build React: một cổng, một link, không cần CORS.",
    ], 9.5, False, MUT, space=2)

    box(s, 6.9, 1.3, 5.95, 0.3, ["4 CHỨC NĂNG + SỐ ĐO"], 13, True, ACC)
    rows = [
        ("Chức năng", "Số đo thực tế"),
        ("1. Kính lúp hoa", f"accuracy {f4(CLF,'test_accuracy')} · 39 ms"),
        ("2. Mắt thần vườn", f"mAP50 {f4(DET,'mAP50')} · 65 ms"),
        ("3. Album hoa", f"Precision@5 {f4(RET,'image_to_image_precision@5')} · 20 ms"),
        ("4. Cô làm vườn (RAG)", f"Hit@1 {RAG.get('hit@1',0):.2f} · 3.105 ms"),
    ]
    table(s, 6.9, 1.68, rows, [3.0, 2.95], 9.5)
    box(s, 6.9, 3.65, 5.95, 2.5, [
        "KHAI BÁO AI ĐÃ DÙNG",
        "   Kimi Code CLI (Moonshot AI) — deepseek-flash",
        "   Mô hình trong sản phẩm: ResNet-18 · YOLO11n 8.4.165",
        "   (AGPL-3.0) · CLIP ViT-B/32 · MiniLM · Qwen2.5-1.5B",
        "",
        "HẠN CHẾ ĐÃ BIẾT",
        "   •  Gõ tiếng Việt không dấu làm truy xuất sai tài liệu.",
        "   •  mAP đo trên COCO128 — chính dữ liệu mô hình đã học.",
        "   •  API chưa có xác thực và giới hạn tần suất.",
    ], 9, False, INK, space=2)
    band(s, f"Clone repo là chạy được ngay: đã kèm artifacts/ và data/gallery/  ·  {GITHUB}")
    out = ROOT / "docs" / "slide-cach-lam.pptx"
    prs.save(out)
    return out


# ========================================================== BẢN 5 TRANG
def build_five_pages() -> Path:
    prs = new_prs()

    # --- 1
    s = new_slide(prs)
    head(s, "Vườn Hoa AI — 4 chức năng AI sau một backend FastAPI",
         "Lập trình Web nâng cao · nộp 01/10/2026 · " + GITHUB)
    box(s, 0.5, 1.35, 5.6, 0.3, ["THẦY GIAO GÌ"], 13, True, ACC)
    box(s, 0.5, 1.72, 5.6, 2.2, [
        "•  Một trang web chứa đủ 4 chức năng AI trong notebook của thầy",
        "•  Mỗi chức năng phải chạy mượt trên web",
        "•  Kèm link GitHub, README có ảnh giao diện, 1 slide về cách làm",
        "•  Được dùng AI nhưng phải ghi rõ công cụ và phiên bản",
    ], 11, False, INK, space=7)
    box(s, 0.5, 4.15, 5.6, 0.3, ["LÀM TRÊN CLOUD, KHÔNG CẦN GPU Ở NHÀ"], 13, True, ACC)
    box(s, 0.5, 4.52, 5.6, 2.2, [
        "Sinh trọng số và chỉ mục trên Google Colab T4 (16 GB VRAM) rồi",
        "mang kết quả về repo. Máy nhóm chỉ có GPU AMD + MX150 2 GB —",
        "không đủ để huấn luyện, nhưng đủ để chạy lại và kiểm thử.",
        "",
        "Nhờ vậy mọi bước đều tái lập được bằng script, không phụ thuộc",
        "vào một máy cụ thể nào.",
    ], 11, False, INK, space=5)
    box(s, 6.5, 1.35, 6.4, 0.3, ["4 CHỨC NĂNG ĐÃ LÀM"], 13, True, ACC)
    y = 1.75
    for name, model, res in FEATURES:
        rect(s, 6.5, y, 6.4, 1.05)
        box(s, 6.65, y + 0.08, 6.1, 0.3, [name], 12, True, ACC)
        box(s, 6.65, y + 0.38, 6.1, 0.28, [model], 10, False, INK)
        box(s, 6.65, y + 0.66, 6.1, 0.28, [res], 9.5, False, MUT)
        y += 1.18
    band(s, f"Kết quả: 1 backend FastAPI (8 endpoint) + 2 giao diện web dùng chung  ·  {GITHUB}")

    # --- 2
    s = new_slide(prs)
    head(s, "Cách làm (1/3) — Kiến trúc: tách 3 tầng, một nơi giữ mô hình",
         "Đây là nguyên tắc xuyên suốt, quyết định mọi lựa chọn phía sau")
    box(s, 0.9, 1.5, 11.5, 1.2, [
        "   Trình duyệt  ──►  Streamlit (8501)            ┐",
        "                                                  ├──►  FastAPI (8000)  ──►  core/   (4 mô hình, nạp 1 lần)",
        "   Trình duyệt  ──►  React  (web/dist)     ┘",
    ], 12, False, INK, space=6)
    box(s, 0.9, 2.5, 11.5, 0.3,
        ["Sau khi npm run build, FastAPI phục vụ luôn web/dist  →  một cổng, một link, không cần CORS."], 10, False, MUT)
    cols = [
        ("core/", ["Chỉ chứa suy luận.", "Không biết gì về web.", "4 lớp: ImageClassifier,", "ObjectDetector, ImageSearch,", "RAGChatbot"]),
        ("api/", ["Bọc core/ thành HTTP.", "8 endpoint, tài liệu tự sinh", "tại /docs.", "Nạp 4 mô hình MỘT lần lúc", "khởi động (lifespan)."]),
        ("Giao diện", ["streamlit_app.py và web/", "chỉ gọi API — không giữ", "mô hình nào.", "Hai giao diện dùng chung", "một backend."]),
    ]
    for i, (name, lines) in enumerate(cols):
        x = 0.9 + i * 3.85
        rect(s, x, 3.0, 3.6, 2.5)
        box(s, x + 0.15, 3.1, 3.3, 0.3, [name], 13, True, ACC)
        box(s, x + 0.15, 3.5, 3.3, 1.9, lines, 10, False, INK, space=3)
    band(s, "Đổi mô hình KHÔNG phải sửa giao diện  ·  cấu hình bằng biến môi trường trong config.py (ENABLED_MODELS, LLM_MODEL, RAG_MIN_SCORE…)")

    # --- 3
    s = new_slide(prs)
    head(s, "Cách làm (2/3) — Quy trình 5 bước và 3 vấn đề đã xử lý")
    box(s, 0.5, 1.3, 7.6, 0.3, ["QUY TRÌNH"], 13, True, ACC)
    steps = [
        ("1", "Trích code notebook thành repo", "core/ · api/ · web/ · streamlit_app.py · tests/ — mỗi thứ một chỗ"),
        ("2", "Sinh artifacts trên Colab T4", "build_artifacts.py gom 20 cell rời thành 5 stage cố định: data → classifier → detector → retrieval → rag"),
        ("3", "Backend FastAPI", "8 endpoint; chatbot trả lời bằng SSE để chữ chạy ra dần"),
        ("4", "Hai giao diện dùng chung API", "Streamlit cho demo nhanh, React cho sản phẩm; React build xong thì FastAPI phục vụ luôn"),
        ("5", "Kiểm thử", "7 test pytest chạy trong 0,4 giây không cần GPU + smoke test 8 endpoint qua HTTP"),
    ]
    y = 1.68
    for n, t, d in steps:
        rect(s, 0.5, y, 0.5, 0.5, ACC)
        box(s, 0.5, y + 0.06, 0.5, 0.4, [n], 15, True, WHITE, PP_ALIGN.CENTER)
        box(s, 1.15, y + 0.02, 6.95, 0.3, [t], 11.5, True, INK)
        box(s, 1.15, y + 0.3, 6.95, 0.5, [d], 9.5, False, MUT)
        y += 0.98
    box(s, 8.4, 1.3, 4.5, 0.3, ["3 VẤN ĐỀ ĐÃ XỬ LÝ"], 13, True, ACC)
    probs = [
        ("Mô hình bị nạp trùng ở cả hai giao diện",
         "Tốn gấp đôi RAM/VRAM và lệch kết quả. → Gom về một backend duy nhất, hai giao diện chỉ gọi HTTP."),
        ("Chỉ mục ảnh ghi đường dẫn tuyệt đối",
         "Mang sang máy khác là hỏng. → Lưu đường dẫn tương đối theo APP_ROOT, dùng .as_posix() để không lệ thuộc Windows/Linux."),
        ("Chatbot trả lời bằng kiến thức ngoài tài liệu",
         "Hỏi “Thủ đô của Pháp” thì đáp “Paris”. → Thêm ngưỡng RAG_MIN_SCORE: điểm truy xuất quá thấp thì từ chối thay vì gọi LLM."),
    ]
    y = 1.68
    for t, d in probs:
        rect(s, 8.4, y, 4.45, 1.55)
        box(s, 8.55, y + 0.08, 4.15, 0.45, [t], 10.5, True, ACC)
        box(s, 8.55, y + 0.5, 4.15, 1.0, [d], 9, False, INK, space=3)
        y += 1.7
    band(s, "Cả 3 vấn đề đều phát hiện bằng cách chạy thật và đo, không phải bằng đọc code.")

    # --- 4
    s = new_slide(prs)
    head(s, "Cách làm (3/3) — Kết quả: 4 chức năng đều chạy thật",
         "Số lấy từ artifacts/*/metrics.json sinh trên Colab T4; độ trễ là thời gian xử lý phía server (p50)")
    rows = [
        ("Chức năng", "Mô hình", "Kết quả chạy thật", "Độ trễ p50"),
        ("1. Phân loại ảnh", "ResNet-18 fine-tune, 5 loài", f"daisy 91.2% · accuracy {f4(CLF,'test_accuracy')}", "39 ms"),
        ("2. Phát hiện đối tượng", "YOLO11n, 80 lớp COCO", f"{{bus:1, person:4}} · mAP50 {f4(DET,'mAP50')}", "65 ms"),
        ("3. Tìm kiếm ảnh", "CLIP ViT-B/32 + FAISS, 628 ảnh", f"5/5 đúng loài · Precision@5 {f4(RET,'image_to_image_precision@5')}", "20 ms"),
        ("4. Chatbot RAG", "Qwen2.5 + MiniLM + FAISS", f"đúng “7 ngày” + Nguồn (3) · Hit@1 {RAG.get('hit@1',0):.2f}", "3.105 ms"),
    ]
    table(s, 0.5, 1.35, rows, [2.5, 3.0, 5.2, 1.65])
    for i, (name, cap) in enumerate([("01-streamlit-tong-quan.jpg", "1. Phân loại — Streamlit"),
                                     ("05-react-phat-hien.jpg", "2. Phát hiện — React"),
                                     ("06-react-tim-anh.jpg", "3. Tìm kiếm ảnh — React")]):
        pic(s, name, 0.5 + i * 3.2, 3.3, width=3.0)
        box(s, 0.5 + i * 3.2, 4.95, 3.0, 0.3, [cap], 9.5, True, MUT)
    chat = Image.open(SHOTS / "07-react-chatbot.jpg")
    chat = chat.crop((0, int(chat.height * 0.40), chat.width, int(chat.height * 0.80)))
    buf = BytesIO()
    chat.save(buf, "JPEG", quality=88)
    buf.seek(0)
    s.shapes.add_picture(buf, Inches(10.15), Inches(3.3), width=Inches(2.7))
    box(s, 10.15, 4.95, 2.7, 0.6, ["4. Chatbot RAG — trên điện thoại: trả lời kèm Nguồn (3)"], 9.5, True, MUT)
    band(s, "Ảnh gốc độ phân giải đầy đủ ở docs/screenshots/ · README nhúng cả 8 ảnh")

    # --- 5
    s = new_slide(prs)
    head(s, "Số đo, kiểm thử và khai báo sử dụng AI")
    box(s, 0.5, 1.3, 7.6, 0.3, ["SỐ ĐO THỰC TẾ (artifacts/*/metrics.json)"], 13, True, ACC)
    rows = [
        ("Mô hình", "Chỉ số", "Số đo", "Trên bao nhiêu mẫu"),
        ("ResNet-18", "test accuracy / macro-F1", f"{f4(CLF,'test_accuracy')} / {f4(CLF,'test_macro_f1')}", "367 ảnh test"),
        ("YOLO11n", "mAP50 / mAP50-95", f"{f4(DET,'mAP50')} / {f4(DET,'mAP50_95')}", "128 ảnh COCO128"),
        ("CLIP + FAISS", "Precision@5 / @10",
         f"{f4(RET,'image_to_image_precision@5')} / {(sum(P10.values())/len(P10) if P10 else 0):.4f}", "50 + 5 truy vấn"),
        ("Qwen + MiniLM", "Hit@1 / Hit@3", f"{RAG.get('hit@1',0):.2f} / {RAG.get('hit@3',0):.2f}", "10 câu hỏi"),
    ]
    table(s, 0.5, 1.68, rows, [1.5, 2.3, 1.9, 1.9])
    box(s, 0.5, 4.05, 7.6, 1.5, [
        "•  Độ trễ server (p50 / p95): phân loại 39/71 ms · phát hiện 65/88 ms ·",
        "    tìm ảnh 20/37 ms · chatbot 3.105/4.728 ms",
        "•  Nút cổ chai là Cloudflare Tunnel chứ không phải mô hình: vòng-trip 559 ms",
        "    trong khi server chỉ xử lý 39 ms",
        "•  Kiểm thử: 7 test pytest (0,4 s, không cần GPU) + smoke test 8 endpoint",
    ], 10, False, INK, space=4)
    box(s, 8.4, 1.3, 4.45, 0.3, ["KHAI BÁO AI ĐÃ DÙNG"], 13, True, ACC)
    rect(s, 8.4, 1.68, 4.45, 2.1)
    box(s, 8.55, 1.78, 4.15, 1.9, AI_DECL, 9, False, INK, space=2)
    box(s, 8.4, 3.95, 4.45, 0.3, ["HẠN CHẾ ĐÃ BIẾT"], 13, True, ACC)
    box(s, 8.55, 4.3, 4.3, 2.5, LIMITS, 9.5, False, INK, space=4)
    band(s, f"Repo: {GITHUB}  ·  clone về chạy được ngay (đã kèm artifacts/ và data/gallery/, 99 MB)")

    out = ROOT / "docs" / "slide-cach-lam-5-trang.pptx"
    prs.save(out)
    return out


def report(path: Path) -> None:
    prs = Presentation(path)
    print(f"  {path.relative_to(ROOT).as_posix()}: {len(prs.slides)} trang, "
          f"{path.stat().st_size / 1024:.0f} KB")
    for i, s in enumerate(prs.slides, 1):
        pics = sum(1 for sh in s.shapes if sh.shape_type == 13)
        tbls = sum(1 for sh in s.shapes if sh.has_table)
        print(f"     trang {i}: {len(s.shapes):2} shape · {tbls} bảng · {pics} ảnh")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    print("Sinh slide từ số liệu trong artifacts/ …")
    for path in (build_one_page(), build_five_pages()):
        report(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
