# Nghiên cứu kỹ thuật phân đoạn ảnh dựa trên Deep Learning

## Mô tả dự án

Đây là một dự án nghiên cứu và ứng dụng các kỹ thuật phân đoạn ảnh dựa trên Deep Learning để phân tích và xử lý ảnh y tế, đặc biệt là ảnh X-quang phổi. Mục tiêu chính của dự án là phát triển các mô hình học sâu có khả năng phân đoạn các vùng quan trọng trên ảnh phổi, từ đó hỗ trợ quá trình chẩn đoán và điều trị các bệnh lý như viêm phổi hoặc ung thư phổi.

### **Tác giả**
- Nguyễn Gia Khang
- Dương Thị Diệu Ái

**Giảng viên hướng dẫn:** Thạc sĩ Nguyễn Tấn Phú

## Mục tiêu nghiên cứu
1. **Tìm hiểu các mô hình phân đoạn ảnh y tế**:
   - Fully Convolutional Networks (FCN)
   - SegNet
   - U-Net
   - U-Net++
2. **Xây dựng và huấn luyện mô hình**:
   - Tiền xử lý dữ liệu: chuẩn hóa, tăng cường dữ liệu, giảm kích thước ảnh.
   - Huấn luyện và tối ưu hóa các mô hình học sâu.
3. **Đánh giá hiệu suất mô hình**:
   - Sử dụng các chỉ số như Dice Coefficient, IoU, Precision, Recall.
4. **Đề xuất cải tiến**:
   - Tối ưu hóa mô hình phù hợp với các đặc điểm riêng của dữ liệu ảnh X-quang phổi.

## Cấu trúc dự án

```
project-root/
|
├── .venv/                  # Môi trường ảo (được bỏ qua trong .gitignore)
├── .vscode/                # Cấu hình của VSCode (được bỏ qua trong .gitignore)
├── data/                   # Thư mục dữ liệu (được bỏ qua trong .gitignore)
│   ├── raw/                # Dữ liệu gốc
│   ├── processed/          # Dữ liệu đã qua xử lý
├── experiments/            # Kết quả thử nghiệm mô hình (được bỏ qua trong .gitignore)
├── logs/                   # Log huấn luyện (được bỏ qua trong .gitignore)
├── notebooks/              # Notebook Jupyter minh họa và thử nghiệm
├── src/                    # Mã nguồn chính
│   ├── models/             # Định nghĩa các kiến trúc mạng (U-Net, U-Net++, ...)
│   ├── utils/              # Các tiện ích xử lý dữ liệu và tính toán
│   ├── train.py            # Script huấn luyện mô hình
│   ├── evaluate.py         # Script đánh giá mô hình
├── README.md               # Tài liệu dự án (file này)
├── requirements.txt        # Các thư viện cần thiết
└── .gitignore              # File cấu hình Git để loại trừ các file không cần thiết
```

## Hướng dẫn sử dụng

### **1. Cài đặt môi trường**

1. Clone repo về máy:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Tạo môi trường ảo và cài đặt thư viện:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Với Linux/Mac
   .\.venv\Scripts\activate  # Với Windows
   pip install -r requirements.txt
   ```

3. Đảm bảo rằng bạn đã cài đặt TensorFlow GPU nếu có GPU hỗ trợ:
 ```bash
 pip install tensorflow-gpu
 ```


### **2. Chuẩn bị dữ liệu**
- Tải tập dữ liệu X-quang phổi từ nguồn Kaggle (Pneumothorax Chest X-ray Images and Masks).
- Đặt dữ liệu vào thư mục `data/raw/` và chạy các script trong `src/utils/` để tiền xử lý dữ liệu.

### **3. Huấn luyện và đeánh giá mô hình**

Để huấn luyện và đánh giá mô hình phân đoạn ảnh phổi, bạn có thể thực hiện trực tiếp trong Jupyter Notebooks.

## Kết quả thực nghiệm

Các mô hình được thử nghiệm bao gồm:
- **U-Net++:** Hiệu năng tốt nhất với Dice Coefficient đạt 0.7985.
- **U-Net:** Hiệu năng ổn định với Dice Coefficient đạt 0.7421 (khi tối ưu tham số).
- **SegNet:** Phù hợp với bài toán đơn giản, hiệu năng trung bình.
- **FCN:** Hiệu năng thấp nhất, không phù hợp với bài toán phức tạp.

### **Bảng kết quả tiêu biểu:**
| Mô hình  | Dice Coefficient | IoU   | Precision | Recall |
|----------|------------------|-------|-----------|--------|
| U-Net++  | 0.7985           | 0.6647| 0.8621    | 0.7437 |
| U-Net    | 0.7421           | 0.5899| 0.8697    | 0.6471 |
| SegNet   | 0.6549           | 0.4869| 0.8351    | 0.5387 |
| FCN      | 0.2954           | 0.1733| 0.2881    | 0.3031 |

## Định hướng phát triển
- Mở rộng nghiên cứu sang các loại dữ liệu y tế khác như CT, MRI.
- Thử nghiệm các kiến trúc tiên tiến hơn như Attention U-Net, DeepLabV3+.
- Tích hợp các mô hình vào hệ thống hỗ trợ chẩn đoán y tế thực tế.

## Liên hệ
- Nguyễn Gia Khang: [khangng162@gmail.com](mailto:khangng162@gmail.com)
- Dương Thị Diệu Ái: [dieuai256@gmail.com](mailto:dieuai256@gmail.com)

---
*Đồ án được thực hiện tại Trường Đại học Kỹ thuật - Công nghệ Cần Thơ, 2024.*
