---
title: AI Web Apps API
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: FastAPI backend cho 4 ứng dụng AI
---

# AI Web Apps — backend

FastAPI giữ cả 4 mô hình (ResNet-18 phân loại hoa, YOLO11n phát hiện đối tượng, CLIP + FAISS tìm
kiếm ảnh, Qwen2.5 + MiniLM chatbot RAG), chạy trên CPU. Bản React tĩnh cũng được phục vụ tại `/`.

- `GET /api/health` — trạng thái, thiết bị, mô hình đã nạp
- `POST /api/classify`, `/api/detect`, `/api/search/text`, `/api/search/image`, `/api/chat`, `/api/chat/sync`
- `/docs` — tài liệu tương tác

Container khởi động rồi tải trọng số CLIP, MiniLM và Qwen từ Hugging Face Hub (1–3 phút cho lần
đầu); giao diện gọi `/api/health` lại mỗi 3 giây nên vẫn vào được trong lúc chờ.

Mã nguồn, model card, số đo và cách chạy trên máy:
[MMMAlonea04/AI_Web_Apps_Streamlit_React](https://github.com/MMMAlonea04/AI_Web_Apps_Streamlit_React)
