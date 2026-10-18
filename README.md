# GestureLab — Nhận dạng cử chỉ bằng TF-C

**GestureLab** là ứng dụng web minh họa bài toán **nhận dạng cử chỉ từ chuỗi tín hiệu gia tốc ba trục (X, Y, Z)**. Ứng dụng sử dụng mô hình **Time-Frequency Contrastive Learning (TF-C)** với hai nhánh trích xuất đặc trưng miền thời gian và miền tần số, kết hợp bộ phân loại được fine-tune trên dữ liệu **SGR**.

> **Phạm vi:** Bản demo này sử dụng **Transformer Encoder của TF-C gốc**, **không sử dụng DINOv3**. Phần thay backbone DINOv3 là một nhiệm vụ nghiên cứu riêng.

**Trải nghiệm trực tuyến:** [gesturelab-tfc-demo.streamlit.app](https://gesturelab-tfc-demo.streamlit.app/)  
**Dữ liệu tham khảo và dữ liệu tự thu thập:** [Dataset TF-C trên Kaggle](https://www.kaggle.com/datasets/nguynngctin1881/dataset-tf-c)

## 1. Tổng quan và bài toán

GestureLab tiếp nhận chuỗi gia tốc ba trục và dự đoán **một trong sáu cử chỉ**:

| Nhãn | Ý nghĩa |
|---|---|
| `typing` | Gõ phím |
| `teeth` | Đánh răng |
| `drinking` | Uống nước |
| `writing` | Viết chữ |
| `clapping` | Vỗ tay |
| `folding` | Gấp đồ |

Quy trình xử lý:

```text
Mẫu SGR / CSV / NPZ / Sensor Logger ZIP
                    ↓
       Kiểm tra và xử lý tín hiệu
                    ↓
          Tensor [N, 3, 206]
                    ↓
       TF-C Time + Frequency Encoders
                    ↓
         SGR Classifier (6 lớp)
                    ↓
 Nhãn dự đoán + điểm Softmax + biểu đồ
```

Mô hình được **khởi tạo từ checkpoint encoder đã pre-train trên HAR** (trong quá trình thử nghiệm từng đặt tên MAR, là HAR được đổi tên) rồi **fine-tune trên SGR**. Đầu ra của ứng dụng là **sáu cử chỉ SGR**, không phải bốn nhãn hoạt động HAR.

## 2. Chức năng chính

- **Thử nhanh mẫu SGR:** chọn loại cử chỉ và mẫu thử có sẵn để dự đoán mà không cần tải file.
- **Tải dữ liệu:** hỗ trợ `.csv`, `.npz` và `.zip` xuất từ Sensor Logger có `Accelerometer.csv`.
- **Trực quan hóa:** hiển thị tín hiệu X/Y/Z theo từng cửa sổ, kèm độ lệch chuẩn (Std), biên độ đỉnh (Peak), RMS và nhận xét thống kê.
- **Dự đoán:** hiển thị nhãn, điểm Softmax của lớp được chọn và biểu đồ điểm dự đoán cho cả sáu lớp.
- **Đối chiếu:** báo trạng thái khớp/chưa khớp khi đầu vào có nhãn tham chiếu; không tự suy đoán nhãn thật cho file không có nhãn.
- **Xuất dữ liệu:** tải kết quả suy luận theo cửa sổ dưới dạng CSV.

Các chỉ số Std, Peak và RMS là **thống kê tín hiệu**, không phải một mô hình AI khác hoặc phép xác định cử chỉ độc lập. Điểm Softmax cũng không được hiểu là mức độ chính xác đã được bảo đảm trong mọi điều kiện.

## 3. Kiến trúc mô hình và kết quả thử nghiệm

Ứng dụng sử dụng TF-C với **hai Transformer Encoder**, mỗi nhánh gồm **hai Transformer layers**. Đặc trưng của hai nhánh được đưa qua projection heads, ghép lại và truyền vào classifier sáu lớp. Mô hình nhận tensor đầu vào `[N, 3, 206]`.

- **Nguồn pre-training:** HAR (dữ liệu gia tốc điện thoại, bốn nhãn hoạt động).
- **Fine-tuning / downstream:** SGR (dữ liệu gia tốc đồng hồ thông minh, sáu nhãn cử chỉ).
- **Số epoch fine-tuning trong lần xuất mô hình:** 15.
- **Checkpoint triển khai:** `models/tfc_har_to_sgr_safe.pt` (gồm encoder và classifier; chạy suy luận bằng CPU).

Kết quả do script huấn luyện/đánh giá của dự án ghi nhận:

| Chỉ số | Giá trị |
|---|---:|
| Test Accuracy | **80,67%** |
| Test Macro-F1 | **80,84%** |
| Test Loss | **0,8236** |

**Lưu ý khi diễn giải:** Trong lần chạy này, validation Macro-F1 tốt nhất khoảng **40,10%**, thấp hơn đáng kể so với kết quả test. Cần kiểm tra thêm cách chia dữ liệu và quy trình đánh giá trước khi dùng các chỉ số trên để kết luận khả năng tổng quát hóa. Đây là **demo học thuật**, không phải hệ thống đã được kiểm định trong thực tế.

## 4. Cấu trúc mã nguồn

```text
.
├── app.py                          # Giao diện và điều khiển quy trình Streamlit
├── data_io.py                      # Đọc, kiểm tra và chuẩn bị CSV / NPZ / ZIP
├── model_runtime.py                # Định nghĩa TF-C gốc và suy luận sáu lớp
├── prepare_checkpoint.py           # Chuyển checkpoint tự huấn luyện sang bản deploy
├── requirements.txt                # Các thư viện cần thiết
├── .streamlit/
│   └── config.toml                 # Giao diện Streamlit
├── models/
│   └── tfc_har_to_sgr_safe.pt       # Encoder + classifier đã xuất
└── samples/
    ├── sgr_examples.npz            # Mẫu SGR để dùng thử
    └── sgr_normalized_csv_example.csv
```

## 5. Cách chạy trên máy cá nhân

**Yêu cầu:** Python 3.11 và Conda (khuyến nghị). Không cần GPU và không cần cài DINOv3.

```bash
# Tạo môi trường riêng (chỉ cần chạy một lần)
conda create -n tfc-web python=3.11 -y
conda activate tfc-web

# Di chuyển tới thư mục chứa app.py
cd <THU_MUC_DU_AN>

# Cài thư viện
python -m pip install -r requirements.txt

# Chạy website
python -m streamlit run app.py
```

Mở trình duyệt tại **http://localhost:8501**.

Tệp `models/tfc_har_to_sgr_safe.pt` phải tồn tại để bật nút dự đoán. Nếu bạn có checkpoint tự huấn luyện nhưng chưa chuyển đổi, sử dụng **`prepare_checkpoint.py`** theo hướng dẫn trong mã nguồn và **chỉ nạp checkpoint từ nguồn tin cậy**. Ứng dụng không nhận tệp `.pt` do khách truy cập tải lên.

## 6. Định dạng dữ liệu đầu vào

| Loại | Yêu cầu |
|---|---|
| **CSV** | Có cột `x`, `y`, `z`; hỗ trợ thêm `time` hoặc `seconds_elapsed`. Có thể tải CSV từ Sensor Logger. |
| **NPZ** | Chứa mảng `samples` dạng `[N, 3, 206]` hoặc `[N, 206, 3]`; có thể có `labels`. |
| **ZIP** | File xuất Sensor Logger chứa `Accelerometer.csv`. |

Với chuỗi liên tục, ứng dụng tạo các **cửa sổ 206 điểm, bước trượt 100 điểm**. Người dùng cần chọn đúng chế độ **dữ liệu thô** hoặc **đã chuẩn hóa theo SGR**. Ứng dụng không tự coi tín hiệu có tần số lấy mẫu khác nhau là tương đương về mặt thực nghiệm.

**Giới hạn dữ liệu:** Model được huấn luyện cho **cử chỉ từ cảm biến đồng hồ thông minh**. Dữ liệu gia tốc từ điện thoại có thể được đọc và đưa qua pipeline về mặt kỹ thuật, nhưng các dự đoán trên nguồn dữ liệu khác SGR **chưa được xác thực độ chính xác**. Dữ liệu HAR tự thu thập (Walking, Stairs, Sitting, Standing) **không** đồng nghĩa với sáu nhãn của classifier SGR.

## 7. Triển khai trực tuyến

Ứng dụng được triển khai bằng **Streamlit Community Cloud** từ repository GitHub này, với:

- **Nhánh triển khai:** `main`
- **Entry point:** `app.py`
- **Phụ thuộc:** `requirements.txt`
- **Mô hình:** `models/tfc_har_to_sgr_safe.pt`

**Link demo:** https://gesturelab-tfc-demo.streamlit.app/

Ứng dụng có thể cần thời gian khởi động lại sau khi ở trạng thái ngủ. Khi thử nghiệm, nên chọn mẫu SGR trước để kiểm tra mô hình, sau đó mới tải CSV/NPZ/ZIP của riêng bạn.

## 8. Nguồn tham khảo và ghi nhận

- **TF-C — mã nguồn nghiên cứu gốc:** [mims-harvard/TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining).
- **Dữ liệu HAR/SGR và dữ liệu tự thu thập phục vụ dự án:** [Dataset TF-C trên Kaggle](https://www.kaggle.com/datasets/nguynngctin1881/dataset-tf-c). Dữ liệu HAR/SGR có nguồn gốc bên ngoài; cần ghi nhận nguồn và tuân thủ giấy phép phân phối tương ứng.
- **Streamlit:** [streamlit.io](https://streamlit.io/).

GestureLab là **ứng dụng minh họa được phát triển dựa trên TF-C**, không phải giao diện chính thức do nhóm tác giả TF-C công bố. Các kết quả thực nghiệm được nêu theo log của phiên bản dự án này, không đại diện cho toàn bộ kết quả trong bài báo gốc.
