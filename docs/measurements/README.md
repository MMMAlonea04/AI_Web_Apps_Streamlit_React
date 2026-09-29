# Số đo thực tế

Số thô cho bảng tổng hợp trong [MODEL_CARD.md](../../MODEL_CARD.md).
Sinh ngày **2026-09-29** trên **Colab T4 (Tesla T4, 16 GB VRAM)** bằng `python scripts/build_artifacts.py all`.

Bốn file `.json` trong thư mục này là **bản sao nguyên văn** của
`artifacts/{classifier,detector,retrieval}/metrics.json` và `artifacts/rag_metrics.json`.
Trọng số, chỉ mục và ảnh kho đi kèm nằm ở `artifacts/` và `data/gallery/` — tức là clone repo về
chạy được ngay, không phải huấn luyện lại.

## Kiểm chứng độc lập

Sau khi Colab sinh số, nhóm **đo lại từ bên ngoài** qua API công khai (gọi HTTP như người dùng thật)
để xem số có trung thực không:

| Chỉ số | Trong `artifacts/` | Đo lại độc lập | |
| --- | --- | --- | --- |
| Classifier test accuracy | 0.9509536784741145 | **0.9509536784741145** | trùng khít |
| Classifier macro-F1 | 0.9508784565931061 | **0.9508784565931061** | trùng khít |
| Detector mAP50 | 0.669649061510062 | **0.669649061510062** | trùng khít |
| Detector mAP50-95 | 0.5024436941561841 | **0.5024436941561841** | trùng khít |
| Retrieval Precision@5 | 0.8760 | 0.9440 | **lệch** — xem giải thích bên dưới |
| Retrieval Precision@10 | 1.0000 | 1.0000 | bằng nhau |
| RAG Hit@1 / Hit@3 | 1.00 / 1.00 | 1.00 / 1.00 | bằng nhau |
| `gallery_size` | 628 | 628 | bằng nhau |

**Vì sao Precision@5 lệch:** khác **tập truy vấn**, không khác mô hình. Script lấy 50 truy vấn
từ chính kho ảnh (mỗi ảnh truy vấn đều nằm trong chỉ mục, bỏ kết quả trùng chính nó) → 0.8760.
Lần đo lại lấy 50 ảnh ngẫu nhiên từ toàn bộ 3.670 ảnh của bộ dữ liệu, phần lớn **không** nằm trong
kho, nên câu trả lời dễ hơn → 0.9440. Cùng một mô hình, cùng một chỉ mục 628 vector.

> Bài học ghi vào báo cáo: Precision@k **phải nói rõ tập truy vấn**, nếu không thì con số vô nghĩa.

## 1. `classifier.json` — ResNet-18, phân loại 5 loài hoa

```
test_accuracy = 0.9510    (349/367 đúng)
test_macro_f1 = 0.9509
epochs = 5 · batch_size = 64 · device = cuda
```

Chia phân tầng 80/10/10, `SEED=42`: train 2.936 · val 367 · test 367 ảnh.
F1 từng lớp: `daisy 0.984 · dandelion 0.967 · sunflowers 0.964 · roses 0.915 · tulips 0.925`.
Lớp yếu nhất là **roses** (0.915) — hay bị nhầm với tulips.

Lần đo lại đã **tái tạo đúng cách chia tại local** rồi gửi cả 367 ảnh test qua `POST /api/classify`
(4 luồng, 54 giây, 0 lỗi) và ra **đúng từng chữ số**. Hai đường đi khác nhau ra cùng kết quả —
tập test cố định đúng như thiết kế.

## 2. `detector.json` — YOLO11n, COCO128

```
mAP50 = 0.6696    mAP50-95 = 0.5024
```
128 ảnh, 929 đối tượng. Trên CPU local chạy hết 10 giây.

> **COCO128 chính là dữ liệu mô hình đã học** khi huấn luyện trên COCO, nên con số này
> lạc quan hơn so với ảnh mới. Đây là điểm yếu của phép đo, không phải điểm mạnh của mô hình.

## 3. `retrieval.json` — CLIP ViT-B/32 + FAISS, kho 628 ảnh

```
image_to_image_precision@5  = 0.8760
text_to_image_precision@10  = 1.0000   (mỗi loài đều 1.00)
gallery_size                = 628      (128 ảnh COCO + 100 ảnh × 5 loài hoa)
model                       = openai/clip-vit-base-patch32
```

`gallery_size = 628` cũng xác nhận được từ bên ngoài bằng tìm kiếm nhị phân trên `/api/gallery/{id}`:
id 627 trả 200, id 628 trả 404.

> **Precision@10 = 1.00 là bài đo dễ**, đừng đọc thành "tìm ảnh hoàn hảo": câu truy vấn
> `"a photo of daisy"` gần như trùng với cách gán nhãn của kho ảnh nên gần như chắc chắn đúng.
> Con số 0.8760 của Precision@5 có ý nghĩa hơn.

## 4. `rag.json` — chất lượng truy xuất của chatbot

```
hit@1 = 1.00 · hit@3 = 1.00 · n_questions = 10
embed_model = sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
llm         = Qwen/Qwen2.5-1.5B-Instruct
```

> Chỉ 10 câu — mẫu quá nhỏ để coi là kết luận. Và **Hit@k chỉ đo khâu truy xuất, không đo câu trả lời**:
> chất lượng câu trả lời là ~6–7/10, xem [MODEL_CARD.md](../../MODEL_CARD.md) mục 4.
> `artifacts/rag_probe.json` lưu 4 câu thăm dò gồm cả câu ngoài phạm vi và câu prompt injection.

## Đo lại

```bash
python scripts/build_artifacts.py all      # sinh lại toàn bộ artifacts/ (Colab T4: 15–25 phút)
python scripts/build_artifacts.py detector # chỉ lấy lại mAP (nhanh, chạy được trên CPU)
python scripts/serve.py api                # đo lại qua HTTP như lần kiểm chứng độc lập
```
