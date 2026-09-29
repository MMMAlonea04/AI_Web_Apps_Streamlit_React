# Ảnh giao diện

Ảnh chụp từ phiên chạy thật trên **Colab T4**, truy cập qua Cloudflare Tunnel.
README ở thư mục gốc nhúng các ảnh này.

| File | Chứng minh |
|---|---|
| `01-tong-quan-streamlit.png` | Sidebar: `Backend: 🟢 cuda`, cả 4 mô hình đã nạp ✅ |
| `02-phan-loai.png` | Chức năng 1 — phân loại ảnh hoa, kết quả `sunflowers 88.6%` |
| `03-phat-hien.png` | Chức năng 2 — phát hiện đối tượng, ảnh đã vẽ hộp + bảng toạ độ |
| `04-tim-anh.png` | Chức năng 3 — tìm ảnh bằng câu mô tả, lưới kết quả kèm điểm |
| `05-chatbot-nguon.png` | Chức năng 4 — chatbot RAG trả lời **và mở phần "Nguồn đã dùng"** |
| `06-ca-mo-hinh-sai.png` | Ca mô hình sai: `bus.jpg` bị đoán là `tulips` và hệ thống **cảnh báo "không chắc chắn"** thay vì khẳng định sai |
| `07-react-phat-hien.png` | Giao diện thứ hai (React) — cùng backend, bố cục khác |
| `08-react-chatbot.png` | React: chatbot streaming + phần `Nguồn (3)` mở ra |
| `09-swagger-docs.png` | 8 endpoint của backend FastAPI tại `/docs` |
| `10-dien-thoai.png` | Chạy trên điện thoại (tuỳ chọn) |

Nếu chụp lại: chạy `notebooks/colab_runbook.ipynb` trên Colab T4, rồi làm theo cột "Làm gì" trong README.
