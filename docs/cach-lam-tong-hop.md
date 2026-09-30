# Tổng hợp cách làm — Vườn Hoa AI (Streamlit & React)

Tài liệu này gom **toàn bộ đường đi** của dự án: dữ liệu lấy ở đâu, 4 ứng dụng dùng mô hình gì và đo
được bao nhiêu, cách sinh artifacts, cách kiểm thử, cách triển khai công khai, và những lỗi đã gặp cùng
cách sửa. Số liệu đều là **đo thật** trong `artifacts/`; chi tiết giới hạn và rủi ro ở
[MODEL_CARD.md](../MODEL_CARD.md), số thô ở [measurements/](measurements/).

**Sáu câu tóm tắt:** một backend FastAPI giữ 4 mô hình, hai giao diện (Streamlit và React) chỉ gọi HTTP.
Dữ liệu tự tải bằng script (TF Flowers 3.670 ảnh, COCO128, 8 tài liệu chăm sóc hoa tiếng Việt). Bốn ứng dụng:
phân loại hoa (accuracy **0.9510**), phát hiện đối tượng (mAP50 **0.6696**), tìm kiếm ảnh (Precision@5
**0.8760**), chatbot RAG (Hit@1 **1.00** trên 10 câu). Triển khai: backend chạy trên **Colab T4 +
Cloudflare Tunnel** (miễn phí, có GPU), giao diện trên **Netlify**, hai bên tự nối với nhau qua một
**gist công bố** để không phải sửa gì khi link tunnel đổi mỗi phiên.

## 1. Kiến trúc

```
Trình duyệt ──► Streamlit (8501) ─┐
                                  ├──► FastAPI (8000/7860) ──► core/ (4 mô hình, nạp 1 lần)
Trình duyệt ──► React (web/dist) ─┘
```

| Tầng | Thư mục | Trách nhiệm |
|---|---|---|
| Suy luận | `core/` | Chỉ 4 lớp mô hình: `ImageClassifier`, `ObjectDetector`, `ImageSearch`, `RAGChatbot` |
| HTTP | `api/` | Bọc thành endpoint, kiểm ảnh tải lên, đo `X-Process-Time-ms`, CORS |
| Giao diện | `streamlit_app.py`, `web/` | Chỉ gọi API, không chứa mô hình |
| Cấu hình | `config.py` | Mọi giá trị đổi được bằng biến môi trường (đổi model không phải sửa code) |

Điểm cốt lõi: **đổi mô hình = đổi biến môi trường**, giao diện không phải sửa. Cả 4 mô hình nạp một lần
lúc khởi động; một mô hình nạp lỗi chỉ khiến `/api/health` báo `false` chứ không làm sập server.

## 2. Dữ liệu dùng ở đâu

| Bộ dữ liệu | Dùng cho | Quy mô | Cách có được | Ghi chú |
|---|---|---|---|---|
| **TF Flowers** | Ứng dụng 1 (huấn luyện) | 3.670 ảnh / 5 lớp (daisy, dandelion, roses, sunflowers, tulips) | `python scripts/build_artifacts.py data` (tự tải) | Chia phân tầng 80/10/10 = 2.936 / 367 / 367, cố định `SEED=42`, lưu ở `artifacts/classifier/split.json`. Tăng cường chỉ áp cho tập train |
| **COCO128** | Ứng dụng 2 (đánh giá) + kho ảnh của ứng dụng 3 | 128 ảnh, 929 đối tượng | như trên | Ảnh này nằm trong tập COCO mà YOLO đã học → mAP đo được **lạc quan hơn thực tế** |
| **Kho ảnh gallery** | Ứng dụng 3 (truy vấn) | **628 ảnh** = 128 COCO128 + 100 × 5 loài hoa | Sinh ở stage `retrieval` | Nhãn COCO do chính YOLO gán → 95 nhãn khác nhau. Ảnh nằm trong `data/gallery/` (đã commit) |
| **8 tài liệu chăm sóc hoa** | Ứng dụng 4 (RAG) | 8 file Markdown | Có sẵn ở `data/kb/` | Tự biên soạn cho chủ đề hoa: chăm sóc cơ bản, 5 loài hoa, sâu bệnh, hoa theo mùa |

Dữ liệu thô (`data/flowers/`, `data/coco128/`) **không commit** — bị chặn trong `.gitignore` vì tải lại
được bằng script; chỉ `data/kb/` và `data/gallery/` (39 MB) nằm trong repo để bản Docker chạy được ngay.

## 3. Bốn ứng dụng và số đo thật

| # | Ứng dụng | Mô hình | Endpoint | Chỉ số đo thật |
|---|---|---|---|---|
| 1 | Nhận diện loài hoa | ResNet-18 fine-tune từ ImageNet | `POST /api/classify` | test accuracy **0.9510**, macro-F1 **0.9509** (367 ảnh test) |
| 2 | Phát hiện đối tượng | YOLO11n (COCO, 80 lớp, không huấn luyện lại) | `POST /api/detect` | mAP50 **0.6696**, mAP50-95 **0.5024** (128 ảnh COCO128) |
| 3 | Tìm kiếm ảnh | CLIP ViT-B/32 + FAISS `IndexFlatIP` | `POST /api/search/text`, `/api/search/image`, `GET /api/gallery/{id}` | ảnh→ảnh Precision@5 **0.8760** (50 truy vấn); chữ→ảnh Precision@10 **1.00** (5 truy vấn, bài đo dễ) |
| 4 | Cô làm vườn — chatbot chăm sóc hoa | Qwen2.5-Instruct + MiniLM + FAISS top-3 | `POST /api/chat` (SSE), `POST /api/chat/sync` | Hit@1 **1.00**, Hit@3 **1.00** (10 câu, đo trên kho tài liệu cũ — cần đo lại); chất lượng trả lời thực tế **~6–7/10** |

Cách huấn luyện ứng dụng 1: transfer learning, thay lớp `fc` thành 5 đầu ra, AdamW + OneCycleLR, label
smoothing 0.1, AMP fp16, **chọn checkpoint theo tập validation** và chỉ báo cáo **một lần** trên tập test
(lịch sử 5 epoch ghi trong `artifacts/classifier/metrics.json`).

Hai con số "đẹp" cần đọc cho đúng: Precision@10 = 1.00 là bài đo dễ (câu `"a photo of daisy"` gần như
trùng cách gán nhãn kho ảnh), và Hit@1 = 1.00 chỉ trên 10 câu — mẫu quá nhỏ. Con số đáng tin nhất là
**accuracy 0.9510 trên 367 ảnh test tách hẳn khỏi train**.

## 4. Sinh artifacts (5 stage)

`scripts/build_artifacts.py` gom các cell rời của notebook thành 5 stage chạy theo thứ tự cố định:

| Stage | Việc | Kết quả |
|---|---|---|
| `data` | Tải TF Flowers, COCO128 + biểu đồ EDA | `data/flowers/`, `data/coco128/`, `artifacts/figures/01_du_lieu.png` |
| `classifier` | Fine-tune ResNet-18, chọn checkpoint theo val, đánh giá 1 lần trên test | `artifacts/classifier/{model.pt,classes.json,split.json,metrics.json}` |
| `detector` | Tải YOLO11n, chạy ảnh mẫu, eval trên COCO128 | `artifacts/detector/{yolo11n.pt,metrics.json}` |
| `retrieval` | Gắn nhãn COCO bằng YOLO, dựng gallery, encode CLIP, build FAISS, đo Precision | `data/gallery/`, `artifacts/retrieval/{index.faiss,meta.json,metrics.json}` |
| `rag` | Chunk tài liệu, embedding MiniLM, eval Hit@k, nạp LLM hỏi thử | `artifacts/{rag_metrics.json,rag_probe.json}` |

```bash
python scripts/build_artifacts.py all                       # ~15–25 phút trên Colab T4, 1–2 giờ trên CPU
python scripts/build_artifacts.py classifier --epochs 5 --batch-size 64
```

Chạy CPU vẫn ra đủ 4 bộ artifacts nhưng **số không dùng để nộp**: classifier tự giảm còn 1 epoch / 800 ảnh
(train ~0.68 thay vì ~0.95) và LLM chuyển sang `Qwen2.5-0.5B-Instruct`, bản 0.5B kém hơn hẳn về an toàn
(đã đo: cùng câu injection xin OTP, 1.5B từ chối còn 0.5B trả lời đưa mã). Batch size tự giảm còn 16 nếu
VRAM < 6 GB nên không bị `CUDA out of memory`.

## 5. Chạy trên máy và trên Colab

```bash
# trên máy
py -3 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python scripts/serve.py api          # http://localhost:8000/docs
python scripts/serve.py streamlit    # http://localhost:8501
cd web && npm install && npm run build && python scripts/serve.py web
```

```python
# trên Colab (Runtime → T4 GPU trước)
!git clone -b develop https://github.com/MMMAlonea04/AI_Web_Apps_Streamlit_React.git /content/ai_web_apps
%cd /content/ai_web_apps
!pip install -q -r requirements-colab.txt     # không cài lại torch vì Colab đã có bản khớp CUDA
!python scripts/serve.py all                  # build React → API → Streamlit → tunnel
```

`serve.py all` tự cài Node 22 nếu Node của Colab quá cũ cho Vite 8 (cần ≥ 20.19 / ≥ 22.12). Biến môi
trường quan trọng: `ENABLED_MODELS`, `LLM_MODEL`, `CORS_ORIGINS`, `RAG_MIN_SCORE`, `MAX_UPLOAD_MB` — bảng
đầy đủ ở [README mục 7](../README.md).

## 6. Kiểm thử và đo

```bash
python -m pytest                     # 7 test API bằng mô hình giả, ~2 giây, không cần GPU/artifacts
python scripts/smoke_test.py         # kiểm thật (cần artifacts + API đang chạy)
API_URL=https://<link-tunnel> python scripts/smoke_test.py    # kiểm luôn bản đã triển khai
```

`smoke_test.py` đi hết 8 ca: health, classify, detect, search bằng chữ, search bằng ảnh, ảnh gallery trả
`image/*`, chat thường + chat SSE, và các ca lỗi phải đúng mã 400 / 413 / 404 / 422. Script lấy ảnh mẫu từ
`data/gallery` khi chưa tải bộ Flowers, nên chạy được cả với backend ở xa.

Độ trễ đo trên Colab T4 (gọi qua Cloudflare Tunnel, header `X-Process-Time-ms`):

| Endpoint | Server p50 | Server p95 | Vòng-trip p50 | Vòng-trip p95 |
|---|---|---|---|---|
| `/api/classify` | 39 ms | 71 ms | 559 ms | 850 ms |
| `/api/detect` | 65 ms | 88 ms | 3802 ms | 3887 ms |
| `/api/search/text` | 20 ms | 37 ms | 585 ms | 680 ms |
| `/api/search/image` | 44 ms | 66 ms | 605 ms | 697 ms |
| `/api/chat/sync` | 3105 ms | 4728 ms | 3527 ms | 5161 ms |

Bài học từ số đo: `/api/detect` trả ảnh base64 ~368 KB nên vòng-trip chậm hơn server **~60 lần** — nút cổ
chai là **truyền ảnh**, không phải mô hình. Và lần gọi đầu tiên của mỗi mô hình chậm hơn hẳn lần sau
(classify 1.060 ms ở lần đầu so với 34 ms khi đã nóng) — khi báo cáo độ trễ nên ghi rõ "đã warm-up".

## 7. Triển khai công khai

### 7.1 Hai bản chạy công khai

| Thành phần | Nơi chạy | Địa chỉ | Sống khi nào |
|---|---|---|---|
| Backend FastAPI + 4 mô hình | Colab T4 + Cloudflare Tunnel | `https://<tên-ngẫu-nhiên>.trycloudflare.com` | chỉ khi phiên Colab còn chạy |
| Giao diện React | Netlify | `https://ai-web-aapp.netlify.app` | luôn (file tĩnh) |

Vì link tunnel đổi mỗi phiên, chính link tunnel cũng phục vụ luôn bản React đã build → cách gọn nhất để
có *một* link là mở thẳng link tunnel (cùng origin, không cần CORS).

### 7.2 Vấn đề trung tâm: frontend ở origin cố định, backend đổi địa chỉ

`VITE_API_URL` là biến **build-time**, nên nếu ghim cứng thì mỗi phiên phải build lại giao diện. Cách giải
quyết đã chọn — **đọc địa chỉ API lúc chạy**, theo thứ tự ưu tiên (`web/src/api.js`):

```
?api=<địa chỉ>  →  window.API_URL  →  gist công bố  →  localStorage  →  VITE_API_URL (lúc build)  →  cùng origin
```

Trong đó "gist công bố" là một **gist công khai** chứa `{"api": "<link tunnel hiện tại>"}`, và `serve.py`
tự cập nhật gist đó sau mỗi lần mở tunnel (`scripts/colab_utils.py::publish_api_url`). Kết quả: mỗi phiên
chỉ cần chạy `serve.py all`, mở Netlify là vào thẳng, không phải dán gì.

Muốn tự dựng lại cơ chế này:

1. Tạo **public gist** với file `api.json` nội dung `{"api": ""}` → lấy ID ở cuối URL.
2. Tạo GitHub token **có quyền ghi gist** (classic: scope `gist`; fine-grained: *User permissions →
   Gists → Read and write*). **Token phải thuộc đúng tài khoản sở hữu gist** — PAT chỉ sửa được gist của chính nó.
3. Đặt vào Colab Secrets: `GH_TOKEN`, `GIST_ID` (bật *Notebook access* cho cả hai).
4. Frontend trỏ vào gist: `VITE_API_DISCOVERY=https://api.github.com/gists/<id>` lúc build, **hoặc** mở
   một lần `https://<tên-site>.netlify.app/?discovery=https://api.github.com/gists/<id>` (được ghi nhớ).

Cell Colab chuẩn cho mỗi phiên — chú ý phải đẩy secret ra biến môi trường **trước** khi chạy:

```python
import os
from google.colab import userdata
os.environ["GH_TOKEN"] = userdata.get("GH_TOKEN")     # !python là tiến trình con, không đọc được userdata
os.environ["GIST_ID"] = userdata.get("GIST_ID")
os.environ["CORS_ORIGINS"] = "https://ai-web-aapp.netlify.app"
!python scripts/serve.py all                          # log sẽ in: 📣 Đã công bố địa chỉ backend: …
```

### 7.3 CORS

`config.py` mặc định chỉ cho `localhost`. Khi deploy phải đặt `CORS_ORIGINS` **trước khi** khởi động API
(API đọc biến này lúc chạy), đúng origin và **không có dấu `/` ở cuối** — trình duyệt gửi Origin không có
dấu `/`, thêm vào là không bao giờ khớp và preflight trả 400.

### 7.4 Giao diện React lên Netlify

`netlify.toml` khai base `web`, `npm run build`, publish `dist`, `NODE_VERSION=22`. Hai cách đưa lên:

- **Nối repo GitHub** (khuyến nghị): Netlify tự build mỗi lần push, biến môi trường đặt một lần trong UI.
- **Kéo-thả** `web/dist`: Netlify không có chỗ sửa file, nên biến build-time phải có **lúc build tại máy**
  (`web/.env` theo mẫu `web/.env.example`, hoặc `VITE_API_DISCOVERY=… npm run build`). Đừng dùng trang
  `app.netlify.com/drop` cho site đã có vì nó tạo **site mới → tên miền mới → CORS lệch**; hãy vào site cũ
  → tab **Deploys** → kéo thả.

### 7.5 Hugging Face Spaces và các phương án khác

Từ 2026, Hugging Face **chỉ cho tài khoản trả phí tạo Space chạy compute** (Gradio/Docker) — *"require a
paid plan to create: PRO for personal accounts"*; hardware `CPU Basic` (2 vCPU/16 GB) vẫn miễn phí nhưng
phải có PRO mới tạo được Space Docker. **Static Spaces thì miễn phí cho mọi người**, kể cả có bước build
(`sdk: static` + `app_build_command: npm run build` + `app_file: dist/index.html`).

| Phương án | Phù hợp | Ghi chú |
|---|---|---|
| **Colab + Cloudflare Tunnel** (đang dùng) | demo, bảo vệ | 0đ, có GPU T4 nên chatbot dùng Qwen 1.5B; link chết khi hết phiên |
| **Hugging Face Spaces PRO** | API + React, một link 24/7 | $9/tháng, huỷ được; `Dockerfile` + `scripts/deploy_space.py` đã viết sẵn cho đường này |
| **HF Static Space** | chỉ React | 0đ, cùng tài khoản HF |
| **Google Cloud Run** | API container 24/7 | free tier 180.000 vCPU-giây + 360.000 GiB-giây/tháng ≈ 25 giờ xử lý với 2 vCPU/4 GiB; cần thẻ |
| **Render / Koyeb / Fly.io** | API container | RAM free thường 512 MB–1 GB → phải `ENABLED_MODELS=classifier,detector` (bỏ LLM) |
| **Streamlit Community Cloud** | giao diện Streamlit | 0đ, đặt secret `API_URL` trỏ về backend |

## 8. Lỗi đã gặp và cách sửa

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| Docker image phình thêm ~100 MB | `chown -R` sau `COPY` nhân bản layer, mà HF Spaces chạy container bằng UID 1000 | Tạo user `-u 1000` **trước** mọi `COPY` rồi dùng `COPY --chown` |
| Image local kéo theo hàng trăm MB dữ liệu thô | `COPY data/ data/` | Copy tường minh `data/kb` và `data/gallery` |
| Browser báo lỗi CORS dù đã set `CORS_ORIGINS` | Giá trị có dấu `/` ở cuối (hoặc có dấu cách sau dấu phẩy — `config.py` cắt chuỗi theo `,` mà không trim) | Nhập đúng origin, không dấu `/`, không khoảng trắng |
| Trang tĩnh gọi `/api/...` ra 404 | Thiếu địa chỉ backend lúc build | Đặt `VITE_API_URL`/`VITE_API_DISCOVERY`, hoặc dùng `?api=`/`?discovery=` lúc chạy |
| Colab: log in `ℹ️ Chưa công bố địa chỉ backend…` dù Secrets đã đúng | `!python …` là **tiến trình con**, không đọc được `userdata` của kernel | Export `os.environ[...] = userdata.get(...)` trong kernel trước khi chạy (tiến trình con kế thừa) |
| `git push` 403 dù đang là chủ repo | Token thiếu **Contents: write** (fine-grained) hoặc máy đang cache credential của tài khoản khác | Cấp đúng quyền; push bằng `-c credential.helper=` để không dùng credential cũ |
| `PATCH /gists/{id}` trả 403 | PAT chỉ sửa được gist **của chính nó** (gist thuộc tài khoản khác) hoặc thiếu quyền `gists=write` | Dùng token của đúng chủ gist, cấp quyền ghi gist |
| Muốn tạo Docker Space nhưng bị chặn | Chính sách 2026: Space chạy compute cần PRO | Dùng Colab + tunnel (miễn phí), hoặc HF Static Space cho frontend |
| Một mô hình nạp lỗi làm sao biết? | — | `api/main.py` bọc `try/except` từng mô hình; `/api/health` liệt kê `models` true/false, lỗi ghi vào `logs/api.log` |
| Chatbot trả lời ngoài tài liệu | Truy xuất lấy đoạn không liên quan nhưng điểm vẫn trên ngưỡng | `RAG_MIN_SCORE` (0.30) chặn nhóm điểm rất thấp; nhưng hai nhóm **chồng lấn** (trong phạm vi thấp nhất 0.556 vs ngoài phạm vi 0.561) nên không nâng ngưỡng thêm được — cần reranker |

## 9. Hạn chế đã biết và hướng phát triển

- **Chất lượng ứng dụng 4 là điểm yếu nhất**: trả lời đúng ~6–7/10, có ca từ chối sai, có ca bịa; **gõ
  không dấu làm truy xuất sụp** (`0.674` → `0.359`) và trả lời sai — lỗi nghiêm trọng nhất đã biết. Hướng
  sửa: thêm **reranker cross-encoder**, hoặc thay embedding/LLM tốt hơn (chỉ đổi biến môi trường).
- **Không có xác thực, không có rate limit** trên API; ảnh tải lên chỉ kiểm kích thước và định dạng.
- **Chưa có test tự động chống prompt injection** (mới có 4 câu thăm dò ghi ở `artifacts/rag_probe.json`).
- **Mẫu đo nhỏ**: Hit@k trên 10 câu, Precision@10 trên 5 truy vấn; detector eval trên chính dữ liệu COCO
  mà mô hình đã học.
- **Link demo phụ thuộc phiên Colab** — muốn 24/7 phải chuyển sang host trả phí (mục 7.5).
- **Giấy phép**: YOLO11n là **AGPL-3.0** → đưa lên mạng cho người khác dùng là buộc mở mã nguồn; notebook
  gốc còn **xoá `LICENSE.txt`** của TF Flowers. Chi tiết ở [MODEL_CARD.md](../MODEL_CARD.md) mục 5.

## 10. Bản đồ file

| Đường dẫn | Nội dung |
|---|---|
| `core/` | `classifier.py`, `detector.py`, `retrieval.py`, `llm.py` — chỉ suy luận |
| `api/main.py` | 8 endpoint, CORS, đo thời gian, phục vụ luôn `web/dist` |
| `config.py` | Cấu hình tập trung, đọc từ biến môi trường |
| `streamlit_app.py` · `web/` | Hai giao diện (Streamlit và React) |
| `scripts/build_artifacts.py` | 5 stage sinh artifacts |
| `scripts/serve.py` | Chạy API/giao diện/tunnel, tự công bố link vào gist |
| `scripts/deploy_space.py` | Đóng gói + đẩy lên Hugging Face Spaces (Docker) |
| `scripts/smoke_test.py` | Kiểm thật 8 ca, chạy được với backend ở xa |
| `tests/test_api.py` | 7 test bằng mô hình giả, chạy trong CI |
| `artifacts/` · `data/kb/` · `data/gallery/` | Trọng số, chỉ số đo, tài liệu và kho ảnh |
| `deploy/hf-space/README.md` · `netlify.toml` | Cấu hình triển khai |

## 11. Khai báo sử dụng AI và giấy phép

- AI hỗ trợ trong quá trình làm bài: **Kimi Code CLI** (mô hình `deepseek-flash`) — chuyển notebook của
  thầy thành repo chạy được, viết `build_artifacts.py`, `serve.py`, `smoke_test.py`, `tests/`, `Dockerfile`,
  `netlify.toml`, `deploy_space.py`; tìm và sửa lỗi; chạy kiểm thử và đo độ trễ; viết README/`MODEL_CARD.md`;
  cá nhân hoá thương hiệu Vườn Hoa AI, chatbot Cô làm vườn và kho tài liệu chăm sóc hoa.
  Kiến trúc, mô hình và code gốc lấy từ notebook của thầy; nhóm chạy, kiểm chứng và chịu trách nhiệm nội dung.
- Mô hình trong sản phẩm: ResNet-18 (BSD-3-Clause), **YOLO11n (AGPL-3.0 — cần chú ý)**, CLIP (MIT),
  MiniLM (Apache-2.0), Qwen2.5 (Apache-2.0).
