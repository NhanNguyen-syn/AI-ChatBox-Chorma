# Chroma AI Chat - Dalat Hasfarm

Một ứng dụng AI Chat nội bộ được xây dựng cho Dalat Hasfarm, cho phép tương tác mạnh mẽ với các mô hình ngôn ngữ lớn (LLM), tích hợp tìm kiếm web, và cung cấp một giao diện quản trị toàn diện.

---

## 🚀 Công nghệ sử dụng

- **Backend:** FastAPI (Python)
- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Cơ sở dữ liệu Vector:** ChromaDB
- **Cơ sở dữ liệu Quan hệ:** SQLite
- **LLM:** Hỗ trợ OpenAI (GPT-4, GPT-3.5) và các mô hình local qua Ollama.
- **Deployment:** Docker (tùy chọn)

---

## 🌟 Tính năng nổi bật

### Dành cho Người dùng

- **Giao diện Chat hiện đại:** Tương tác tự nhiên với AI.
- **Tương tác với tài liệu:** Tải lên và "trò chuyện" với file PDF, DOCX, TXT. Hệ thống tự động bóc tách và hiểu nội dung.
- **Tích hợp tìm kiếm Web:** AI có khả năng tìm kiếm thông tin mới nhất trên Internet để trả lời câu hỏi.
- **Lịch sử Chat:** Lưu trữ và xem lại các cuộc trò chuyện trước đây.
- **Giao diện Sáng/Tối:** Chuyển đổi giao diện linh hoạt để bảo vệ mắt.

### Dành cho Quản trị viên

- **Dashboard quản trị:** Theo dõi thống kê hệ thống, số lượng người dùng, tài liệu, và các chỉ số quan trọng khác.
- **Quản lý Người dùng:** Mời, kích hoạt/vô hiệu hóa, và phân quyền (admin/user) cho các tài khoản.
- **Quản lý Tài liệu:** Giám sát các tài liệu đã được tải lên và trạng thái xử lý của chúng.
- **Quản lý FAQs:** Xây dựng và quản lý kho câu hỏi thường gặp để AI trả lời nhanh và chính xác.
- **Tùy chỉnh Thương hiệu:** Thay đổi logo, tên ứng dụng và màu sắc chủ đạo để phù hợp với nhận diện thương hiệu của công ty.

---

## 🛠️ Hướng dẫn Cài đặt & Khởi chạy

### Yêu cầu

- **Python 3.10+**
- **Node.js 18+** và **npm**
- **(Tùy chọn) Tesseract OCR:** Cần thiết cho việc đọc file PDF dạng ảnh. Tải về [tại đây](https://github.com/tesseract-ocr/tesseract).

### 1. Cài đặt Backend

```bash
# Di chuyển vào thư mục backend
cd chroma_project_demo/backend

# Cài đặt các thư viện Python cần thiết
pip install -r requirements.txt
```

### 2. Cấu hình Backend

Tạo một file tên là `.env` trong thư mục `backend` và sao chép nội dung từ file `.env.example` (nếu có) hoặc sử dụng mẫu dưới đây.

**Nội dung file `.env` mẫu:**

```env
# CHỌN NỀN TẢNG LLM
# Đặt USE_OPENAI=1 để dùng OpenAI, hoặc 0 để dùng mô hình local (Ollama)
USE_OPENAI=1

# CẤU HÌNH OPENAI (nếu USE_OPENAI=1)
OPENAI_API_KEY="sk-YOUR_API_KEY_HERE"
OPENAI_CHAT_MODEL="gpt-4o-mini"
OPENAI_EMBED_MODEL="text-embedding-3-large"

# CẤU HÌNH OCR (để đọc file ảnh/PDF scan)
ENABLE_OCR="1"
OCR_LANG="vie+eng" # Ngôn ngữ nhận dạng: vie (Việt), eng (Anh)
# Đường dẫn tới Tesseract trên Windows (chỉnh lại nếu cần)
TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"

# CẤU HÌNH TÌM KIẾM WEB
ALLOW_WEB_SEARCH="1"

# CẤU HÌNH BOOTSTRAP ADMIN
# Đặt BOOTSTRAP_ADMIN=1 để tự động tạo tài khoản admin khi khởi động
BOOTSTRAP_ADMIN="1"
ADMIN_USERNAME="admin"
ADMIN_EMAIL="admin@example.com"
ADMIN_PASSWORD="admin123"
```

### 3. Cài đặt Frontend

```bash
# Di chuyển từ backend ra frontend
cd ../frontend

# Cài đặt các gói Node.js
npm install
```

### 4. Chạy ứng dụng

**a. Chạy Backend Server:**

Mở terminal tại thư mục `chroma_project_demo/backend` và chạy:

```bash
uvicorn main:app --reload
```

> Server sẽ khởi động tại `http://localhost:8000`. Lần chạy đầu tiên, các bảng trong cơ sở dữ liệu và tài khoản admin (nếu được bật) sẽ được tự động tạo.

**b. Chạy Frontend App:**

Mở một terminal khác tại thư mục `chroma_project_demo/frontend` và chạy:

```bash
npm run dev
```

> Ứng dụng sẽ mở trong trình duyệt tại `http://localhost:3002` (hoặc một port khác nếu 3002 đã bận).

---

## 🔑 Tài khoản Demo

Bạn có thể sử dụng các tài khoản sau để trải nghiệm ứng dụng:

- **Tài khoản Admin:**

  - **Username:** `SG0510`
  - **Password:** `admin123`
- **Tài khoản User:**

  - **Username:** `SG1297`
  - **Password:** `123123`
