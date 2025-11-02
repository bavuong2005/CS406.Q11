<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>CS406.Q11 - XỬ LÝ ẢNH VÀ ỨNG DỤNG</b></h1>
<h1 align="center"><b>IMAGE PROCESSING AND APPLICATIONS</b></h1>
<h2 align="center"><b>LAB 04</b></h2>

# Introduction
* **Full name:** Mai Lê Bá Vương
* **Student ID:** 23521821
* **Lecturer:** Cáp Phạm Đình Thăng - *thangcpd@uit.edu.vn*

# Demo Phân loại Cảnh vật (VGG, ResNet, ViT)

Đây là một ứng dụng web xây dựng bằng Streamlit cho phép người dùng upload ảnh cảnh vật tự nhiên và nhận dự đoán từ ba mô hình Deep Learning khác nhau: **VGG16**, **ResNet50**, và **ViT-B16**.

## Demo

<img src="https://raw.githubusercontent.com/bavuong2005/CS406.Q11/refs/heads/main/23521821_Lab_4/demo.gif" alt="Demo"></img>

## Cài đặt và Chạy

### 1. Yêu cầu
* Python 3.8+
* Các file trọng số (`.h5`) đã huấn luyện của 3 mô hình.
* **Lưu ý:** Do file trọng số rất nặng, chúng **không** được lưu trên Git. Vui lòng xem hướng dẫn tải ở **Bước 4** dưới đây.

### 2. Hướng dẫn

1.  **Clone repository này:**

    ```bash
    git clone https://github.com/bavuong2005/CS406.Q11.git
    ```

2.  **Di chuyển vào thư mục Lab 4:**
    (Các bước tiếp theo phải được thực hiện từ bên trong thư mục của Lab 4)

    ```bash
    cd CS406.Q11/23521821_Lab_4
    ```

3.  **Tạo và kích hoạt môi trường ảo:**

    ```bash
    python -m venv myenv
    
    # Trên Windows
    myenv\Scripts\activate
    
    # Trên macOS/Linux
    source myenv/bin/activate
    ```

4.  **Cài đặt các thư viện cần thiết:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Tải và thêm file trọng số (BẮT BUỘC):**

    Vui lòng tải 3 file trọng số (`.weights.h5`) từ output của Kaggle Notebook tại link sau:

    **Link tải trọng số: [Kaggle Output - Model Weights](https://www.kaggle.com/code/vuong2901/cs406-lab4/output)**

    Sau khi tải về:

    a. Tạo một thư mục tên là `models` (nếu chưa có).
    
    b. Sao chép 3 file trọng số vào thư mục `models`.

6.  **Chạy ứng dụng Streamlit:**

    ```bash
    streamlit run app.py
    ```

    Trình duyệt sẽ tự động mở lên địa chỉ `http://localhost:8501`.

## 📂 Cấu trúc Thư mục
    23521821_Lab_4/
    │
    ├── app.py             # File Streamlit chính để chạy ứng dụng
    │
    ├── models/             # Thư mục chứa các file trọng số đã huấn luyện
    │   ├── vgg16.weights.h5
    │   ├── resnet50.weights.h5
    │   └── vit_b16.weights.h5
    │
    ├── utils.py           # File chứa các hàm hỗ trợ (tiền xử lý, load mô hình)
    │
    └── requirements.txt   # File chứa các thư viện Python cần thiết

## Các mô hình được sử dụng
Dự án này so sánh hiệu quả của 3 kiến trúc mô hình phổ biến:
* **VGG16:** Một mô hình CNN truyền thống với các tầng tích chập sâu.
* **ResNet50:** Một mô hình CNN sử dụng các kết nối tắt (skip connections) để giải quyết vấn đề vanishing gradient.
* **ViT-B16 (Vision Transformer):** Một kiến trúc hiện đại dựa trên cơ chế "attention" của Transformer, vốn ban đầu được thiết kế cho NLP.