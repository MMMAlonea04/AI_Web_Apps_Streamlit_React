# AI Web Apps — Streamlit & React

Bốn ứng dụng AI sau **một** backend FastAPI, với hai giao diện web (Streamlit và React).

| # | Ứng dụng | Mô hình | Dữ liệu (tự tải) | Chỉ số |
|---|---|---|---|---|
| 1 | Nhận diện loài hoa | ResNet-18 fine-tune | TF Flowers — 3.670 ảnh, 5 lớp | Accuracy, macro-F1, ma trận nhầm lẫn |
| 2 | Phát hiện đối tượng | YOLO11n (COCO, 80 lớp) | COCO128 — 128 ảnh có nhãn | mAP50, mAP50-95 |
| 3 | Tìm kiếm ảnh | CLIP ViT-B/32 + FAISS | COCO128 + 100 ảnh/loài hoa | Precision@5, Precision@10 |
| 4 | Trợ lý khách hàng | Qwen2.5-Instruct + MiniLM + FAISS (RAG) | 6 tài liệu chính sách ShopLite | Hit@1, Hit@3 |

```
Trình duyệt ──► Streamlit (8501) ─┐
                                  ├──► FastAPI (8000) ──► core/ (4 mô hình, nạp 1 lần)
Trình duyệt ──► React (web/dist) ─┘
```

`core/` chỉ chứa suy luận · `api/` bọc thành HTTP · `streamlit_app.py` và `web/` chỉ là giao diện.
Đổi mô hình không phải sửa giao diện.

## Ảnh giao diện

Chụp từ phiên chạy thật trên **Colab T4 (Tesla T4, 16 GB VRAM)**, truy cập qua Cloudflare Tunnel.
Danh sách đầy đủ và ý nghĩa từng ảnh: [docs/screenshots/](docs/screenshots/).

<table>
<tr>
<td width="50%"><a href="docs/screenshots/01-streamlit-tong-quan.jpg"><img src="docs/screenshots/01-streamlit-tong-quan.jpg" alt="Tổng quan Streamlit"></a><br>
<sub><b>Streamlit — khởi động</b>: sidebar báo <code>Backend: 🟢 cuda</code>, cả 4 mô hình đã nạp ✅, và <b>chức năng 1 (phân loại ảnh)</b> trả về <code>daisy 91.2%</code> trong 11.6 ms</sub></td>
<td width="50%"><a href="docs/screenshots/02-streamlit-phat-hien.jpg"><img src="docs/screenshots/02-streamlit-phat-hien.jpg" alt="Streamlit - phát hiện đối tượng"></a><br>
<sub><b>Chức năng 2 — phát hiện đối tượng</b> (Streamlit): YOLO11n vẽ hộp lên ảnh, trả về <code>{person: 1, potted plant: 1}</code> kèm bảng toạ độ</sub></td>
</tr>
<tr>
<td width="50%"><a href="docs/screenshots/03-streamlit-tim-anh.jpg"><img src="docs/screenshots/03-streamlit-tim-anh.jpg" alt="Streamlit - tìm kiếm ảnh"></a><br>
<sub><b>Chức năng 3 — tìm kiếm ảnh</b> (Streamlit): gõ câu mô tả <i>“yellow sunflowers in a field”</i>, trả về ảnh xếp theo điểm tương đồng</sub></td>
<td width="50%"><a href="docs/screenshots/04-react-phan-loai.jpg"><img src="docs/screenshots/04-react-phan-loai.jpg" alt="React - phân loại ảnh"></a><br>
<sub><b>Giao diện thứ hai (React)</b> — chức năng 1: <code>roses 75.0% · tulips 19.6% · sunflowers 2.1%</code>, 8 ms</sub></td>
</tr>
<tr>
<td width="50%"><a href="docs/screenshots/05-react-phat-hien.jpg"><img src="docs/screenshots/05-react-phat-hien.jpg" alt="React - phát hiện đối tượng"></a><br>
<sub><b>React</b> — chức năng 2: <code>fork 96.1% · cake 93.0% · dining table 31.4%</code>, bảng 3 cột Lớp / Độ tin cậy / Hộp</sub></td>
<td width="50%"><a href="docs/screenshots/06-react-tim-anh.jpg"><img src="docs/screenshots/06-react-tim-anh.jpg" alt="React - tìm kiếm ảnh"></a><br>
<sub><b>React</b> — chức năng 3: lưới kết quả kèm nhãn và điểm, tìm được cả bằng câu mô tả lẫn ảnh mẫu</sub></td>
</tr>
<tr>
<td width="50%"><a href="docs/screenshots/07-react-chatbot.jpg"><img src="docs/screenshots/07-react-chatbot.jpg" alt="React - chatbot RAG"></a><br>
<sub><b>Chức năng 4 — chatbot RAG</b>: trả lời đúng <i>“7 ngày”</i> theo tài liệu và có <code>Nguồn (3)</code> để người đọc tự kiểm</sub></td>
<td width="50%"><a href="docs/screenshots/08-tren-dien-thoai.jpg"><img src="docs/screenshots/08-tren-dien-thoai.jpg" alt="Trên điện thoại"></a><br>
<sub><b>Chạy trên điện thoại</b> — cùng một link công khai, không cần cài gì</sub></td>
</tr>
</table>

## 1. Yêu cầu môi trường

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| Python | **3.11 – 3.14** | Đã thử trên Windows/Python 3.14 với torch 2.14 CPU + faiss-cpu 1.15.1 |
| Node | **≥ 22.12** | Vite 8 yêu cầu Node ≥ 20.19 hoặc ≥ 22.12 |
| GPU | không bắt buộc | CPU chạy được cả 4 mô hình; tự giảm còn 1 epoch / 800 ảnh train và LLM 0.5B |

## 2. Chạy trên máy (Windows / Linux / macOS)

```bash
py -3 -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python scripts/build_artifacts.py all  # sinh artifacts (~15 phút trên GPU, 1–2 giờ trên CPU)
python scripts/serve.py api
```

Trên máy **không có GPU NVIDIA**, `pip install torch` từ PyPI đã là bản CPU nên không cần làm gì thêm.
Chạy trên CPU vẫn ra đủ 4 bộ artifacts, nhưng classifier chỉ học 1 epoch trên 800 ảnh
(test accuracy thực đo ~0.68, so với ~0.9x khi chạy đủ 5 epoch trên GPU) — **dùng CPU để thử luồng,
không dùng để lấy số nộp bài.**

Mở `http://localhost:8000/docs` để thử API, hoặc chạy giao diện:

```bash
python scripts/serve.py streamlit                      # http://localhost:8501
cd web && npm install && npm run build                 # React build → API phục vụ tại :8000
python scripts/serve.py web                            # hoặc để script tự chạy 2 lệnh trên
```

`python scripts/serve.py all` chạy một lượt: build React → API → Streamlit → tunnel.

## 3. Chạy trên Google Colab (khuyến nghị — T4, 16 GB VRAM)

**Bật GPU trước:** Runtime → Change runtime type → **T4 GPU**. `config.py` tính `DEVICE` lúc import,
nên đổi runtime sau khi đã chạy thì phải Restart.

Mỗi ô dán riêng một cell. Dùng `requirements-colab.txt` (không cài lại torch vì Colab đã có bản khớp CUDA):

```python
!git clone https://github.com/MMMAlonea04/AI_Web_Apps_Streamlit_React.git /content/ai_web_apps
```

```python
%cd /content/ai_web_apps          # %cd giữ thư mục cho các cell sau; !cd thì không
```

```python
!pip install -q -r requirements-colab.txt
```

```python
!python scripts/build_artifacts.py all          # ~15–25 phút trên T4
```

Sinh xong artifacts thì **sao lưu ngay** — `/content` mất khi hết phiên. Phải mount Drive trước:

```python
from google.colab import drive; drive.mount('/content/drive')
```

```python
!mkdir -p /content/drive/MyDrive/ai_web_apps && cp -r artifacts data/gallery /content/drive/MyDrive/ai_web_apps/
```

Rồi chạy demo và lấy link công khai:

```python
!python scripts/serve.py all
```

`serve.py` tự cài Node 22 nếu bản Node của Colab quá cũ cho Vite 8 (chỉ chạy được khi có quyền root,
tức là trên Colab/Colab-like). Nếu chỉ muốn API, dùng `python scripts/serve.py api`.

> Link `*.trycloudflare.com` chỉ sống khi phiên Colab còn chạy. Triển khai lâu dài xem mục 6.

## 4. Từng bước sinh artifacts

`scripts/build_artifacts.py` gom các cell rời của notebook thành 5 stage chạy theo thứ tự cố định:

| Stage | Việc | Kết quả |
|---|---|---|
| `data` | Tải TF Flowers, COCO128 + biểu đồ EDA | `data/flowers/`, `data/coco128/`, `artifacts/figures/01_du_lieu.png` |
| `classifier` | Fine-tune ResNet-18, chọn checkpoint theo val, đánh giá 1 lần trên test | `artifacts/classifier/{model.pt,classes.json,split.json,metrics.json}` |
| `detector` | Tải YOLO11n, chạy ảnh mẫu, eval trên COCO128 | `artifacts/detector/{yolo11n.pt,metrics.json}` |
| `retrieval` | Gắn nhãn COCO bằng YOLO, dựng gallery, encode CLIP, build FAISS, đo Precision | `data/gallery/`, `artifacts/retrieval/{index.faiss,meta.json,metrics.json}` |
| `rag` | Chunk tài liệu, embedding MiniLM, eval Hit@k, nạp LLM hỏi thử | `artifacts/{rag_metrics.json,rag_probe.json}` |

```bash
python scripts/build_artifacts.py data                 # chỉ tải dữ liệu
python scripts/build_artifacts.py classifier --epochs 5 --batch-size 64
```

Batch size tự giảm còn 16 nếu VRAM < 6 GB (MX150 2 GB), nên không bị `CUDA out of memory`.

## 5. Kiểm thử

```bash
python -m pytest                      # test API bằng mô hình giả, ~2 giây, không cần GPU
```

Test trong `tests/test_api.py` thay mô hình thật bằng bản giả qua `MODELS` của `api.main`, nên không cần
torch hay artifacts. Cùng lệnh này chạy trong GitHub Actions (`.github/workflows/ci.yml`).

Kiểm thử thật (cần artifacts): `python scripts/smoke_test.py` khi API đang chạy.

## 6. Triển khai

| Phương án | Phù hợp | Tóm tắt | Chi phí |
|---|---|---|---|
| **Hugging Face Spaces (Docker)** | API + React, một link | Tạo Space loại *Docker*, đẩy repo (đã có `Dockerfile`, cổng 7860) | Miễn phí, CPU 2 vCPU/16 GB |
| **Streamlit Community Cloud** | Giao diện Streamlit | Đẩy repo có `streamlit_app.py` + `requirements-streamlit.txt`, đặt secret `API_URL` trỏ tới API | Miễn phí |
| **Render / Railway / Fly.io** | API container | Kết nối repo, dùng `Dockerfile` | Có gói miễn phí |
| **VPS có GPU** | Dự án thật | `docker run --gpus all`, đặt sau Nginx + HTTPS | Theo máy |
| **Vercel / Netlify** | Chỉ React | Build `web/`, đặt `VITE_API_URL` = địa chỉ API, bật CORS ở API | Miễn phí |

`Dockerfile` **cần `artifacts/` và `data/gallery/` có sẵn** trước khi build — chạy `build_artifacts.py` trước,
hoặc để Space tải từ Hugging Face Hub lúc khởi động.

Lưu ý khi lên production: gói miễn phí không có GPU nên đặt `ENABLED_MODELS` gọn và
`LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct`; đặt `CORS_ORIGINS` đúng tên miền giao diện; không commit API key.

## 7. Biến môi trường

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `APP_ROOT` | thư mục chứa `config.py` | Gốc dự án, mọi đường dẫn tính từ đây |
| `ENABLED_MODELS` | `classifier,detector,retrieval,llm` | Mô hình được nạp, ví dụ `classifier,detector` |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` (GPU) / `-0.5B-` (CPU) | Mô hình sinh |
| `EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding cho RAG |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | Tìm kiếm ảnh |
| `YOLO_WEIGHTS` | `artifacts/detector/yolo11n.pt` | Trọng số phát hiện đối tượng |
| `MAX_UPLOAD_MB` | `8` | Giới hạn ảnh tải lên |
| `RAG_MIN_SCORE` | `0.30` | Điểm cosine tối thiểu để coi là tài liệu liên quan; dưới ngưỡng thì chatbot từ chối thay vì gọi LLM |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:8501` | Origin được gọi API |
| `API_URL` | `http://localhost:8000` | (Streamlit) địa chỉ backend |

## 8. API

| Method | Endpoint | Đầu vào | Đầu ra |
|---|---|---|---|
| GET | `/api/health` | — | trạng thái, thiết bị, mô hình đã nạp |
| POST | `/api/classify` | `file`, `top_k` | `predictions`, `confident`, `latency_ms` |
| POST | `/api/detect` | `file`, `conf` | `detections`, `summary`, `image` (base64 đã vẽ hộp) |
| POST | `/api/search/text` | `{query, k}` | `results` [{id, label, score, url}] |
| POST | `/api/search/image` | `file`, `k` | như trên |
| GET | `/api/gallery/{id}` | — | file ảnh trong kho |
| POST | `/api/chat` | `{message, history}` | SSE: `sources` → `token`… → `done` |
| POST | `/api/chat/sync` | như trên | `{answer, sources}` |

Tài liệu tương tác: `/docs`.

## 9. Docker

```bash
python scripts/build_artifacts.py all
docker build -t ai-web-apps .
docker run -p 7860:7860 ai-web-apps      # mở http://localhost:7860
```

## 10. Số đo

Đo ngày 2026-09-29 trên **Colab T4 (Tesla T4, 16 GB VRAM)**, gọi qua Cloudflare Tunnel.

| Endpoint | n | Server p50 | Server p95 | Vòng-trip p50 | Vòng-trip p95 |
|---|---|---|---|---|---|
| `/api/classify` | 20 | 39 ms | 71 ms | 559 ms | 850 ms |
| `/api/detect` | 12 | 65 ms | 88 ms | 3802 ms | 3887 ms |
| `/api/search/text` | 20 | 20 ms | 37 ms | 585 ms | 680 ms |
| `/api/search/image` | 20 | 44 ms | 66 ms | 605 ms | 697 ms |
| `/api/chat/sync` | 5 | 3105 ms | 4728 ms | 3527 ms | 5161 ms |

- **Server** = header `X-Process-Time-ms`, chỉ thời gian xử lý bên trong FastAPI.
- **Vòng-trip** = thời gian client thấy, gồm cả Cloudflare Tunnel (miễn phí, không SLA).
- `/api/detect` trả ảnh base64 ~368 KB nên vòng-trip chậm hơn server ~60 lần — **nút cổ chai là truyền ảnh, không phải mô hình.**
- Chất lượng chatbot: truy xuất đúng tài liệu 10/10, trả lời đúng ~6–7/10 trên 10 câu hỏi chuẩn — chi tiết ở [MODEL_CARD.md](MODEL_CARD.md).

Chỉ số mô hình (accuracy, F1, mAP, Precision@k, Hit@k) và cách đo: [MODEL_CARD.md](MODEL_CARD.md),
số thô ở [docs/measurements/](docs/measurements/).

## 11. Sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `git status` hiện hàng nghìn file lạ | Bạn chạy notebook gốc **trong thư mục repo** — nó tạo dự án riêng ở `ai_web_apps/`. Xoá thư mục đó, và chạy notebook trên Colab hoặc dùng `scripts/build_artifacts.py` |
| `CUDA out of memory` khi chạy API | Đặt `ENABLED_MODELS` ít hơn, hoặc thêm `--batch-size 16` khi train |
| `/api/health` báo mô hình `false` | Thiếu file trong `artifacts/` — xem `logs/api.log`, chạy lại stage tương ứng |
| Tải model Hugging Face chậm / lỗi 429 | `huggingface_hub.login()` bằng token miễn phí |
| Link tunnel không mở được | Tunnel hết hạn khi phiên Colab ngắt — chạy lại `serve.py tunnel` |
| Streamlit upload báo 403 sau proxy | Giữ `--server.enableXsrfProtection false` (đã có trong `serve.py`) |
| `npm run build` lỗi | Node phải ≥ 22.12 |
| Chatbot trả lời bịa | Kiểm `artifacts/rag_metrics.json` (Hit@3), tăng `k`, dùng `LLM_MODEL` lớn hơn |

Chỉ số đã đo: `artifacts/classifier/metrics.json`, `artifacts/detector/metrics.json`,
`artifacts/retrieval/metrics.json`, `artifacts/rag_metrics.json`.

Giới hạn, rủi ro và **vấn đề giấy phép** (YOLO11n là AGPL-3.0): xem [MODEL_CARD.md](MODEL_CARD.md).

## Khai báo sử dụng AI

Bài tập cho phép dùng AI nhưng yêu cầu ghi rõ công cụ và phiên bản, nên khai báo tách thành hai tầng.

### Tầng 1 — AI hỗ trợ trong quá trình làm bài

| Công cụ | Phiên bản / mô hình | Dùng vào việc gì |
|---|---|---|
| **Kimi Code CLI** (Moonshot AI) | mô hình `deepseek-flash` | Trích code từ notebook của thầy thành repo chạy được; viết `scripts/build_artifacts.py`, `scripts/serve.py`, `scripts/smoke_test.py`, `tests/`, `.gitignore`, `.gitattributes`, `Dockerfile`; tìm và sửa lỗi; chạy kiểm thử API và đo độ trễ; viết README và `MODEL_CARD.md` |
| `<công cụ khác nếu có>` | `<version>` | `<dùng làm gì>` |

Phần nào của nhóm, phần nào của AI: kiến trúc, mô hình và code gốc lấy từ notebook của thầy.
Nhóm chạy, kiểm chứng trên Colab T4 và trên link công khai, chụp ảnh minh chứng, quyết định nội dung báo cáo.
AI hỗ trợ chuyển notebook thành repo, sửa lỗi phát hiện khi chạy, và viết tài liệu.

### Tầng 2 — Các mô hình AI nằm trong sản phẩm

| Mô hình | Phần mềm / phiên bản | Giấy phép | Vai trò |
|---|---|---|---|
| ResNet-18 | torchvision 0.29.0, trọng số `IMAGENET1K_V1` rồi fine-tune | BSD-3-Clause | Phân loại 5 loài hoa |
| YOLO11n | ultralytics 8.4.165 | **AGPL-3.0** | Phát hiện 80 lớp COCO |
| CLIP ViT-B/32 | transformers 5.17.0, `openai/clip-vit-base-patch32` | MIT | Mã hoá ảnh và câu chữ để tìm kiếm |
| paraphrase-multilingual-MiniLM-L12-v2 | sentence-transformers 6.1.0 | Apache-2.0 | Embedding tài liệu cho RAG |
| Qwen2.5-1.5B-Instruct | transformers 5.17.0 (GPU) · `-0.5B-Instruct` khi chạy CPU | Apache-2.0 | Sinh câu trả lời của chatbot |

Thư viện chính: `torch 2.14.0`, `torchvision 0.29.0`, `faiss-cpu 1.15.1`, `fastapi 0.141.1`, `uvicorn`,
`pydantic`, `streamlit 1.64.0`, `vite 8.3.1`, `React 19.2.0`, `Node 22`.

> **YOLO11n là AGPL-3.0.** Nếu đưa sản phẩm này lên mạng cho người khác dùng thì AGPL buộc phải mở
> mã nguồn toàn bộ. Dùng cho mục đích học tập thì không vấn đề, nhưng phải nêu rõ. Chi tiết ở
> [MODEL_CARD.md](MODEL_CARD.md) mục 5.
