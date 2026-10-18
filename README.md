# GestureLab — TF-C Gesture Recognition (HAR → SGR)

Ứng dụng **Streamlit** nhận dạng **6 cử chỉ SGR** từ tín hiệu gia tốc ba trục bằng **TF-C Transformer gốc** được pre-train trên HAR (MAR) và fine-tune trên SGR. Không dùng DINOv3 ở phần deploy này.

Ứng dụng có giao diện tiếng Việt: **thử mẫu SGR có sẵn**, **tải CSV, NPZ hoặc Sensor Logger ZIP**, **xem X/Y/Z**, **xem điểm dự đoán từng lớp**, và **tải kết quả CSV**. Tất cả dự đoán đều đến từ checkpoint được huấn luyện; không có chế độ dự đoán mô phỏng.

## Giao diện v2 — Đã cải tiến (không thay đổi model)

Bản v2 chỉ thay đổi giao diện `app.py`, theme `.streamlit/config.toml` và cố định thêm thư viện vẽ biểu đồ trong `requirements.txt`. Các file `model_runtime.py`, `data_io.py`, dữ liệu mẫu và logic suy luận **giữ nguyên**. Không cần huấn luyện lại checkpoint.

Cải tiến chính:
- Chọn nhanh theo **6 loại cử chỉ** và lựa chọn mẫu tham chiếu SGR.
- Thẻ **kết quả nhanh** xuất hiện ngay dưới nút phân loại, không cần cuộn tới cuối trang.
- Có đối chiếu **nhãn dự đoán / nhãn thực tế / trạng thái khớp hoặc chưa khớp** khi dữ liệu có nhãn. Với file CSV không có nhãn, giao diện không tự suy đoán đúng/sai.
- Biểu đồ điểm Softmax **thanh ngang** đọc được tên các cử chỉ tiếng Việt.
- Biểu đồ gia tốc, dữ liệu CSV/NPZ/ZIP, bảng và nút tải kết quả vẫn hoạt động theo luồng cũ.
- Ghim `numpy==1.26.4`, `pyarrow==17.0.0` để tránh lỗi xung đột PyArrow khi vẽ biểu đồ.

**Cập nhật dự án đang chạy:** Sao lưu `app.py`, thay bằng bản v2, cập nhật `.streamlit/config.toml` và `requirements.txt`. Giữ nguyên thư mục `models/` đang có chứa checkpoint thật. Kích hoạt môi trường `tfc-web`, chạy `python -m pip install -r requirements.txt`, sau đó `python -m streamlit run app.py`. Nếu chỉ muốn dùng giao diện mới ở môi trường đã cài đầy đủ, bạn có thể thay riêng `app.py`.

**Lưu ý kiểm thử:** Cú pháp Python của các file đã được kiểm tra. Ứng dụng cần được chạy và kiểm tra với checkpoint `models/tfc_har_to_sgr_safe.pt` trên máy của bạn để xác nhận toàn bộ dự đoán, vì checkpoint đã huấn luyện không nằm trong bản ZIP này.

## 1. Cấu trúc dự án

```text
TFC_SGR_Streamlit/
├── app.py                         # Giao diện Streamlit
├── model_runtime.py               # TF-C Encoder + Classifier đúng bản đã huấn luyện
├── data_io.py                     # Đọc CSV / NPZ / ZIP, tạo cửa sổ 206 điểm
├── prepare_checkpoint.py          # Chuyển checkpoint tin cậy thành dạng an toàn
├── requirements.txt
├── .streamlit/config.toml
├── samples/
│   ├── sgr_examples.npz           # 12 mẫu SGR test thật (2 mẫu/lớp)
│   └── sgr_normalized_csv_example.csv
└── models/
    ├── tfc_har_to_sgr_safe.pt      # BẠN cần tạo từ checkpoint huấn luyện local
    └── training_results.json       # Tùy chọn: sao chép báo cáo kết quả local
```

**Lưu ý:** Không kèm checkpoint đã huấn luyện trong ZIP này vì checkpoint đó đang nằm trên máy Windows của bạn. Không tạo hay thay bằng trọng số giả để trình diễn.

## 2. Chuẩn bị mô hình bạn đã huấn luyện

Bạn đã có checkpoint:

```text
C:\AI_projects\TFC_HAR_SGR_Train\exported_model\tfc_har_to_sgr_best.pt
```

File gốc lưu một số metadata dưới dạng NumPy scalar, dẫn tới lỗi khi nạp với `weights_only=True`. Dùng công cụ đi kèm để chuyển nó thành checkpoint chỉ gồm **tensor và kiểu dữ liệu Python an toàn**, rồi nạp bằng `weights_only=True` trên Cloud.

Trên CMD Windows, sau khi giải nén dự án:

```bat
conda activate C:\Users\admin\miniconda3\envs\tfc
cd /d C:\AI_projects\TFC_SGR_Streamlit
python prepare_checkpoint.py --source "C:\AI_projects\TFC_HAR_SGR_Train\exported_model\tfc_har_to_sgr_best.pt"
copy "C:\AI_projects\TFC_HAR_SGR_Train\exported_model\training_results.json" "models\training_results.json"
```

Lệnh đầu tạo:

```text
models/tfc_har_to_sgr_safe.pt
```

Công cụ chỉ chạy **trên máy của bạn** và đọc checkpoint bạn tự huấn luyện với `weights_only=False` một lần. **Không** chạy nó đối với checkpoint từ người lạ. Ứng dụng Streamlit chỉ tải model đã chuyển đổi bằng `weights_only=True` và **không** chấp nhận người dùng upload `.pt` hoặc pickle.

## 3. Chạy giao diện trên Windows

Để không ảnh hưởng môi trường `tfc` Python 3.8 dùng cho train, khuyên tạo môi trường riêng cho giao diện **không có DINOv3**:

```bat
conda create -n tfc-web python=3.11 -y
conda activate tfc-web
cd /d C:\AI_projects\TFC_SGR_Streamlit
python -c "import sys; print(sys.executable)"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Ứng dụng mở ở `http://localhost:8501`.

Nếu thiếu model, giao diện vẫn có thể hiển thị dữ liệu mẫu và đồ thị nhưng nút dự đoán sẽ bị khóa. Bạn phải hoàn thành mục 2 để chạy inference thật.

## 4. Hỗ trợ CSV/NPZ

- **CSV:** `x,y,z` với ít nhất 206 dòng; hỗ trợ thêm `seconds_elapsed` hoặc `time`, kể cả thứ tự cột `time,seconds_elapsed,z,y,x` do Sensor Logger xuất. Cắt cửa sổ **206 điểm**, bước trượt **100 điểm**, tối đa 100 cửa sổ mỗi lượt. **Không tự nội suy** theo tần số lấy mẫu vì tần số nguồn SGR không được nêu rõ trong gói dữ liệu đã dùng train.
- **NPZ:** khóa `samples`, shape `[N,3,206]` hoặc `[N,206,3]`, có thể có `labels`. Chỉ nạp khi `allow_pickle=False`.
- **Sensor Logger ZIP:** chứa một `Accelerometer.csv` duy nhất.
- **Mẫu thử:** 12 mẫu **đã chuẩn hóa** được rút trực tiếp từ SGR test, không phải dữ liệu giả. `samples/sgr_normalized_csv_example.csv` là một mẫu CSV đã chuẩn hóa; khi tải mẫu này, chọn **Đã chuẩn hóa theo SGR**.

Nếu tải **gia tốc thô**, ứng dụng dùng mean/std từ file `data/SGR/README.md` của gói dữ liệu huấn luyện:

| Trục | Mean (train SGR) | Std (train SGR) |
|---|---:|---:|
| X | -2.00637245 | 5.04262161 |
| Y | -4.74350595 | 4.82831144 |
| Z | 3.29379106 | 4.80061388 |

**Quan trọng về tính khoa học:** Model được fine-tune cho **cử chỉ trên đồng hồ thông minh** gồm `typing, teeth, drinking, writing, clapping, folding`. CSV/ZIP từ cảm biến **điện thoại** vẫn đọc được về kỹ thuật nhưng khác nguồn thiết bị, hướng đặt và phân bố. Không thể dùng model SGR này để nhận dạng chính xác bốn nhãn HAR `Walking/Stairs/Sitting/Standing` mà không huấn luyện lại cho HAR.

## 5. Deploy lên Streamlit Community Cloud

1. Push các file dự án lên **GitHub repository của bạn**, bao gồm `models/tfc_har_to_sgr_safe.pt` đã kiểm tra và hai file demo trong `samples/`. Kiểm tra quyền công khai checkpoint/dataset trước khi push.
2. Vào [Streamlit Community Cloud](https://share.streamlit.io/), đăng nhập bằng GitHub, chọn **Create app**.
3. Chọn repository của bạn, branch chính, và đặt **Main file path = `app.py`**.
4. Deploy rồi chờ môi trường cài `requirements.txt`. Ứng dụng chỉ chạy CPU (không yêu cầu GPU).
5. Mở link công khai, chọn mẫu **SGR** → **Phân loại bằng TF-C**, thử cả file CSV mẫu. Kiểm tra tab **Về mô hình** xem báo cáo kết quả (nếu đã sao chép JSON).

Tránh push token Hugging Face, khóa API, dữ liệu cá nhân hoặc bất kỳ file nhạy cảm nào. App có thể ngủ khi không có truy cập; người chấm có thể đánh thức bằng link.

## 6. Tham khảo và ghi nhận

- TF-C pretraining: https://github.com/mims-harvard/TFC-pretraining
- Dữ liệu đã đăng Kaggle: https://www.kaggle.com/datasets/nguynngctin1881/dataset-tf-c
- Đây là ứng dụng nghiên cứu được phát triển thêm, **không phải giao diện chính thức của nhóm tác giả TF-C**.
