# Ảnh giao diện

Ảnh chụp từ phiên chạy thật trên **Colab T4 (Tesla T4, 16 GB VRAM)**, truy cập qua Cloudflare Tunnel.
README ở thư mục gốc nhúng các ảnh này ở mục “Ảnh giao diện”.

| # | File | Giao diện | Chứng minh |
|---|---|---|---|
| 1 | `01-streamlit-tong-quan.jpg` | Streamlit | Sidebar `Backend: 🟢 cuda` + cả 4 mô hình ✅, chức năng 1 trả `daisy 91.2%` trong 11.6 ms |
| 2 | `02-streamlit-phat-hien.jpg` | Streamlit | Chức năng 2 — ảnh đã vẽ hộp + bảng toạ độ `{person: 1, potted plant: 1}` |
| 3 | `03-streamlit-tim-anh.jpg` | Streamlit | Chức năng 3 — tìm bằng câu mô tả, lưới ảnh kèm điểm |
| 4 | `04-react-phan-loai.jpg` | React | Chức năng 1 — `roses 75.0% · tulips 19.6% · sunflowers 2.1%`, 8 ms |
| 5 | `05-react-phat-hien.jpg` | React | Chức năng 2 — `fork 96.1% · cake 93.0% · dining table 31.4%` + bảng 3 cột |
| 6 | `06-react-tim-anh.jpg` | React | Chức năng 3 — lưới kết quả kèm nhãn và điểm |
| 7 | `07-react-chatbot.jpg` | React (điện thoại) | Chức năng 4 — trả lời “7 ngày” theo tài liệu + `Nguồn (3)` |
| 8 | `08-tren-dien-thoai.jpg` | React (điện thoại) | Chạy trên điện thoại qua cùng một link công khai |

Cả 4 chức năng đều có ảnh minh chứng; chức năng 1–3 có ở **cả hai giao diện**.

Ảnh được chuẩn hoá còn **JPEG, chiều rộng tối đa 1400 px, quality 88** để repo nhẹ (tổng ~1,4 MB).

## Ảnh gốc

Chưa chuẩn hoá nằm ở `E:\ai-web-apps\` trên máy (14 file, tên như `streamlit_phanloai.png`,
`react_phathiendoituong_mobile_1.jpg`…). Cần chụp lại thì chạy `notebooks/colab_runbook.ipynb`
trên Colab T4 rồi bấm theo cột “Làm gì” trong README.

Hai ảnh không dùng: `streamlit_chatbot_mobile.jpg` và `react_chatbot.png` chụp lúc chatbot còn trả lời
sai câu ngoài phạm vi (đã sửa bằng `RAG_MIN_SCORE`, xem `MODEL_CARD.md` mục 4).
