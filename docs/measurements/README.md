# Số đo thực tế

Số thô cho bảng tổng hợp trong [MODEL_CARD.md](../../MODEL_CARD.md). Đo ngày **2026-09-29**.

Ba chỉ số đầu đo **qua API công khai** của phiên chạy trên **Colab T4 (Tesla T4, 16 GB VRAM)** —
tức là đo đúng cái mà người dùng thật sẽ gọi, không phải đo mô hình trong notebook.
Riêng detector đo ở **local CPU** (AMD Ryzen 5 5600, torch 2.14.0+cpu) với cùng trọng số YOLO11n chính thức.

## 1. `classifier.json` — ResNet-18, 367 ảnh test

```
test_accuracy = 0.9510   (349/367 đúng)
test_macro_f1 = 0.9509
```

Cách đo: tái tạo đúng cách chia trong `scripts/build_artifacts.py` (SEED=42, phân tầng 80/10/10)
trên cùng bộ TF Flowers, rồi gửi **toàn bộ 367 ảnh test** qua `POST /api/classify` (4 luồng, 54 giây,
0 lỗi) và tính accuracy / macro-F1 bằng scikit-learn.

F1 từng lớp: `daisy 0.984 · dandelion 0.967 · sunflowers 0.964 · roses 0.915 · tulips 0.925`.
Lớp yếu nhất là **roses** (0.915) — hay bị nhầm với tulips.

> Tập test được **tái tạo tại local** chứ không lấy `artifacts/classifier/split.json` từ Colab.
> Cách chia là hàm thuần của (SEED, danh sách tệp) nên trùng nhau nếu bộ ảnh giống nhau
> (đã kiểm: 5 lớp đúng số lượng 633/898/641/699/799 = 3.670 ảnh như notebook in ra).

## 2. `detector.json` — YOLO11n, COCO128

```
APP_ROOT=<thư mục dự án> python scripts/build_artifacts.py detector
```
```
mAP50 = 0.6696   mAP50-95 = 0.5024
```
128 ảnh, 929 đối tượng, 10 giây trên CPU. Tổng thể: precision 0.663 · recall 0.589.

> **COCO128 chính là dữ liệu mô hình đã học** khi huấn luyện trên COCO, nên con số này
> lạc quan hơn so với ảnh mới. Đây là điểm yếu của phép đo, không phải điểm mạnh của mô hình.

## 3. `retrieval.json` — CLIP ViT-B/32 + FAISS, kho 628 ảnh

```
Precision@5  (ảnh → ảnh) = 0.9440    # 50 truy vấn, 10 ảnh mỗi loài
Precision@10 (chữ → ảnh) = 1.0000    # 5 truy vấn "a photo of <loài>"
gallery_size             = 628
```

`gallery_size = 628` xác nhận bằng tìm kiếm nhị phân trên `/api/gallery/{id}`: id 627 trả 200, id 628 trả 404.

> **Precision@10 = 1.00 là bài đo dễ**, đừng đọc thành "tìm ảnh hoàn hảo": câu truy vấn
> `"a photo of daisy"` gần như trùng với cách gán nhãn của kho ảnh, nên gần như chắc chắn đúng.
> Con số 0.9440 của Precision@5 có ý nghĩa hơn.

## 4. `rag.json` — chất lượng truy xuất của chatbot

```
Hit@1 = 1.00    Hit@3 = 1.00    (10 câu hỏi chuẩn)
```

> Chỉ 10 câu — mẫu quá nhỏ để coi là kết luận. Và **Hit@k chỉ đo khâu truy xuất, không đo câu trả lời**:
> chất lượng câu trả lời là ~6–7/10, xem [MODEL_CARD.md](../../MODEL_CARD.md) mục 4.

## Đo lại

```bash
# 1) Sinh artifacts trên Colab T4 (hoặc local CPU, chậm hơn nhiều)
python scripts/build_artifacts.py all

# 2) Detector: lấy mAP
python scripts/build_artifacts.py detector

# 3) Ba chỉ số còn lại: chạy API rồi đo qua HTTP
python scripts/serve.py api
```
