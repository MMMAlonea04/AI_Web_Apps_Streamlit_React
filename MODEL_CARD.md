# Model Card — AI Web Apps

Phiên bản `v1.0` · Cập nhật 2026-09-29 · Nhóm thực hiện: `<tên các thành viên>`

Bốn mô hình sau **một** backend FastAPI. Tài liệu này nêu dữ liệu, chỉ số, giới hạn và cách dùng đúng/sai.

> ⚠️ Các ô `<...>` là số **phải điền sau khi chạy** `python scripts/build_artifacts.py all` trên Colab.
> Nguồn số: `artifacts/classifier/metrics.json`, `artifacts/detector/metrics.json`,
> `artifacts/retrieval/metrics.json`, `artifacts/rag_metrics.json`.
> **Không điền số ước lượng — số bịa trong model card là lỗi nghiêm trọng.**

## Bảng tổng hợp

| # | Mô hình | Nguồn | Giấy phép | Chỉ số chính | Số đo thực tế |
| --- | --- | --- | --- | --- | --- |
| 1 | ResNet-18 fine-tune | torchvision `IMAGENET1K_V1` | BSD-3-Clause | test accuracy, macro-F1 | `<...>` / `<...>` |
| 2 | YOLO11n | Ultralytics (COCO) | **AGPL-3.0** (xem mục 5) | mAP50, mAP50-95 | `<...>` / `<...>` |
| 3 | CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | MIT | Precision@5 (ảnh→ảnh), Precision@10 (chữ→ảnh) | `<...>` / `<...>` |
| 4 | Qwen2.5-Instruct + MiniLM | `Qwen/Qwen2.5-1.5B-Instruct`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | Hit@1, Hit@3 | `<...>` / `<...>` |

Phần cứng đo: `<GPU T4 16 GB / CPU ...>`. Thời gian suy luận p50/p95 ghi ở README (nhiệm vụ C5).

## 1. ResNet-18 — Phân loại loài hoa

- **Mục đích:** ảnh một bông hoa → tên loài + độ tin cậy, trong 5 lớp cố định.
- **Dữ liệu:** TF Flowers, 3.670 ảnh / 5 lớp (daisy, dandelion, roses, sunflowers, tulips). Chia phân tầng 80/10/10, cố định `SEED=42`, danh sách chỉ số lưu ở `artifacts/classifier/split.json`. Tăng cường dữ liệu **chỉ** áp cho tập train.
- **Cách huấn luyện:** transfer learning từ ImageNet, thay lớp `fc` thành 5 đầu ra; AdamW + OneCycleLR, label smoothing 0.1, AMP fp16, chọn checkpoint theo tập **validation**, báo cáo **một lần** trên tập test.
- **Giới hạn:** chỉ biết 5 loài. Ảnh ngoài 5 lớp **vẫn bị gán nhãn** — hệ thống chỉ hạ cờ `confident=false` khi điểm cao nhất < 0.5, chứ không từ chối. Ảnh chụp cận cảnh nhiều bông, ảnh vẽ, ảnh đen trắng đều dễ sai.
- **Rủi ro:** đặt tên loài sai có thể gây hại nếu dùng cho mục đích nhận diện thực vật ăn được/độc.
- **Dùng đúng:** gợi ý tên loài cho mục đích học tập, gắn nhãn sơ bộ có người kiểm lại.
- **Dùng sai:** kết luận thay chuyên gia, dùng để quyết định an toàn (cây độc, thuốc).

## 2. YOLO11n — Phát hiện đối tượng

- **Mục đích:** khoanh vùng mọi đối tượng trong ảnh, trả nhãn + độ tin cậy + toạ độ hộp.
- **Dữ liệu:** **không huấn luyện**. Dùng trọng số có sẵn của COCO (80 lớp). Đánh giá trên COCO128 (128 ảnh có nhãn, chính là ảnh mà mô hình đã thấy khi huấn luyện COCO) — **đây là điểm yếu của phép đo: mAP trên COCO128 lạc quan hơn so với dữ liệu mới.**
- **Giới hạn:** 80 lớp COCO thiên về cảnh phương Tây; không có lớp đặc thù Việt Nam (biển số, đặc sản, cây trồng...). Vật nhỏ và vật bị che khuất bị bỏ sót nhiều. Kết quả phụ thuộc mạnh vào ngưỡng `conf` (mặc định 0.25).
- **Rủi ro:** bỏ sót người trong ảnh an ninh; nhận nhầm nhãn "person" có thể gây hậu quả khi dùng tự động.
- **Dùng đúng:** hỗ trợ tìm kiếm/đếm đối tượng, gợi ý cho người gán nhãn.
- **Dùng sai:** ra quyết định tự động không có người xác nhận (an ninh, giao thông).

## 3. CLIP ViT-B/32 + FAISS — Tìm kiếm ảnh

- **Mục đích:** gõ mô tả hoặc tải ảnh mẫu → trả các ảnh gần nhất trong kho.
- **Dữ liệu:** kho ảnh dựng sẵn gồm COCO128 (nhãn = đối tượng YOLO phát hiện được) + 100 ảnh mỗi loài hoa, tổng `<gallery_size>`. Vector đã chuẩn hoá, chỉ mục `IndexFlatIP` = cosine.
- **Giới hạn:** **truy vấn văn bản chỉ hiệu quả bằng tiếng Anh** — CLIP gốc không hỗ trợ tiếng Việt. Kho ảnh nhỏ nên con số Precision đo được **không suy ra được** cho kho 10.000 ảnh. CLIP thiên lệch về văn hoá/đối tượng phương Tây (thiên lệch dữ liệu huấn luyện LAION/OpenAI). Kết quả nhạy với cách diễn đạt truy vấn.
- **Rủi ro:** thiên lệch trong xếp hạng ảnh (giới tính, màu da, quốc tịch) nếu dùng để lọc/tuyển chọn con người.
- **Dùng đúng:** tìm ảnh trong kho nội bộ, khám phá nội dung.
- **Dùng sai:** dùng điểm tương đồng để phân loại hoặc đánh giá con người; dùng cho kho ảnh ngoài phân phối dữ liệu đã đo.

## 4. Qwen2.5-Instruct + MiniLM + FAISS — Chatbot RAG

- **Mục đích:** trả lời câu hỏi về chính sách cửa hàng, **chỉ dựa trên tài liệu** và ghi nguồn.
- **Dữ liệu:** 6 tài liệu Markdown **giả lập** của "ShopLite" trong `data/kb/`. Chunk theo tiêu đề `##`, tối đa 600 ký tự. Embedding đa ngữ MiniLM + FAISS top-3.
- **Chỉ số:** Hit@1, Hit@3 trên 10 câu hỏi kiểm thử (`scripts/build_artifacts.py::EVAL_QA`) — mẫu quá nhỏ để coi là kết luận chắc chắn.
- **Giới hạn:** tài liệu không phải chính sách thật của doanh nghiệp nào. Bản `Qwen2.5-0.5B-Instruct` trên CPU yếu rõ rệt so với 1.5B trên GPU — cùng một câu hỏi có thể ra câu trả lời khác nhau. Truy xuất sai đoạn thì câu trả lời sai dù LLM không bịa.
- **Hành vi đo được** (API công khai, Colab T4, Qwen2.5-1.5B-Instruct, 10 câu hỏi chuẩn):
  - **Truy xuất đúng tài liệu 10/10.** Khâu truy xuất không phải điểm yếu.
  - **Trả lời đúng ~6–7/10.** Ba dạng lỗi đã ghi nhận:
    - *Từ chối sai (false refusal):* "Đơn 250.000đ ở Đà Nẵng phí ship bao nhiêu?" — tài liệu `giao_hang.md` có đáp án (35.000đ ngoại thành) và đã được lấy đúng ở top-1 (0.557), mô hình vẫn trả lời "Tài liệu không cung cấp thông tin".
    - *Bịa:* "Quên mật khẩu thì làm sao?" — trả lời về quét mã QR code, không có trong tài liệu nào.
    - *Trả lời lệch mục:* "Một điểm thưởng quy đổi được bao nhiêu tiền?" — trích quy tắc **tích** điểm thay vì quy tắc **dùng** điểm.
  - Nguyên nhân nằm ở **khâu sinh**, không phải khâu truy xuất. Thử bằng `LLM_MODEL=Qwen/Qwen2.5-3B-Instruct` (đổi biến môi trường, không sửa code) là cách kiểm chứng rẻ nhất.
- **Nhạy cảm với dấu tiếng Việt:** cùng một câu hỏi gõ **không dấu** làm điểm tương đồng sụp và truy xuất lấy sai tài liệu — "Bảo hành đồ gia dụng bao lâu?" cho `bao_hanh.md` @0.674 và trả lời đúng "24 tháng"; "Bao hanh do gia dung bao lau?" chỉ còn 0.359, lấy `thanh_toan.md` và mô hình trả lời "3 tháng". Người dùng gõ không dấu (rất phổ biến) sẽ nhận câu trả lời sai. Đây là **lỗi nghiêm trọng nhất đã biết** của ứng dụng 4.
- **Rủi ro:**
  - **Bịa (hallucination):** system prompt yêu cầu chỉ trả lời theo TÀI LIỆU và từ chối khi thiếu thông tin, nhưng không đảm bảo tuyệt đối.
  - **Prompt injection:** đã có 4 câu kiểm thử (`artifacts/rag_probe.json`), gồm câu yêu cầu bỏ qua hướng dẫn và câu hỏi ngoài phạm vi. System prompt có ghi rõ nội dung TÀI LIỆU là dữ liệu tham khảo, không phải mệnh lệnh. **Không có bộ test tự động chống injection** — đây là việc còn thiếu.
  - **Rò rỉ thông tin:** mô hình không bao giờ được trả mật khẩu/OTP; nếu tài liệu đưa vào `data/kb/` có dữ liệu cá nhân thì mô hình có thể đọc ra.
  - **Lịch sử hội thoại** được gửi lên API ở mỗi lượt (giữ 3 lượt gần nhất) — không có xác thực người dùng.
- **Dùng đúng:** trả lời câu hỏi tra cứu, luôn hiển thị nguồn để người đọc tự kiểm.
- **Dùng sai:** dùng làm tư vấn pháp lý/y tế; tin câu trả lời khi không mở phần "Nguồn".

## 5. Vấn đề giấy phép — đọc trước khi dùng thương mại

- **Ultralytics YOLO11n là AGPL-3.0.** Nếu bạn đưa sản phẩm này lên mạng cho người khác dùng, AGPL buộc phải mở mã nguồn toàn bộ. Muốn dùng thương mại kín thì phải mua giấy phép Ultralytics Enterprise, hoặc thay bằng mô hình giấy phép thoáng (ví dụ DETR/RT-DETR của HF).
- Bộ ảnh **TF Flowers**: notebook gốc **xoá `LICENSE.txt`** của bộ dữ liệu (cell "1) TF Flowers"). Đây là vấn đề tuân thủ — cần tra lại giấy phép gốc trước khi công bố sản phẩm.
- **COCO128**: ảnh lấy từ COCO, giấy phép ảnh không đồng nhất (phần lớn từ Flickr). Ảnh trong `data/gallery/` thừa hưởng giấy phép gốc, không phải giấy phép của repo này.
- CLIP (MIT), Qwen2.5 (Apache-2.0), MiniLM (Apache-2.0), torchvision (BSD-3-Clause): dùng được cho cả mục đích thương mại.

## 6. Giới hạn chung của cả hệ thống

1. **Không có xác thực và không có giới hạn tần suất** trên API — ai biết địa chỉ cũng gọi được. Phải thêm rate limit trước khi mở cho người dùng thật.
2. **Ảnh tải lên chỉ được kiểm kích thước (8 MB) và định dạng**, không kiểm nội dung.
3. **Không ghi log câu hỏi/ảnh của người dùng** — tốt cho quyền riêng tư, nhưng không có dữ liệu để giám sát chất lượng.
4. Mọi kết quả đều là **gợi ý của mô hình**, không phải kết luận. Giao diện có câu cảnh báo ở tab Chatbot; các tab khác thì chưa.

## 7. Cách kiểm chứng lại

```bash
python scripts/build_artifacts.py all      # sinh lại số đo, ghi vào artifacts/
python -m pytest                           # 7 test API, không cần GPU
python scripts/smoke_test.py               # cần API đang chạy + artifacts
```
