# Chroma AI Chat - DalatHasfarm

## Giới thiệu

Đây là một ứng dụng AI Chat nội bộ được xây dựng cho DalatHasfarm, sử dụng ChromaDB làm vector store và cho phép tương tác với các mô hình ngôn ngữ lớn (LLM) như OpenAI GPT hoặc các mô hình local qua Ollama. Giao diện quản trị cho phép cấu hình hệ thống, quản lý người dùng và tài liệu.

## Yêu cầu hệ thống

Trước khi bắt đầu, hãy đảm bảo bạn đã cài đặt các phần mềm sau:

* **Python 3.10+**
* **Node.js 18+** và **npm**
* **(Tùy chọn) Ollama:** Nếu bạn muốn chạy các mô hình LLM local.
* **(Tùy chọn) Docker:** Để chạy ChromaDB một cách dễ dàng.

## Hướng dẫn cài đặt

### 1. Cài đặt Backend

Di chuyển vào thư mục backend và cài đặt các dependencies:

```bash
cd chroma_project_demo/backend
pip install -r requirements.txt
```

### 2. Cấu hình Backend

Backend sử dụng các biến môi trường để cấu hình. Tạo một file tên là `.env` trong thư mục `backend` với nội dung sau:

**File `.env` mẫu:**

```env
# --- Cấu hình mô hình AI ---
# Chọn 1 để sử dụng OpenAI, 0 để dùng các mô hình local (ví dụ: Ollama)
USE_OPENAI=1

# Nếu USE_OPENAI=1, hãy cung cấp API key của bạn
OPENAI_API_KEY="sk-YOUR_API_KEY_HERE"

# Tên các mô hình OpenAI (có thể thay đổi nếu muốn)
OPENAI_CHAT_MODEL="gpt-4o-mini"
OPENAI_EMBED_MODEL="text-embedding-3-large"

# --- Cấu hình OCR (để đọc file PDF/ảnh) ---
# Bật/tắt tính năng OCR (1 là bật, 0 là tắt)
ENABLE_OCR="1"
# Ngôn ngữ cho OCR, ví dụ: "vie+eng" cho tiếng Việt và tiếng Anh
OCR_LANG="vie+eng"
# Đường dẫn đến file thực thi Tesseract OCR (chỉ cần trên Windows)
TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Cấu hình Chat ---
# Số lượt chat gần nhất để đưa vào context
HISTORY_TURNS="5"
# Bật/tắt tính năng tóm tắt cuộc trò chuyện
ROLLING_SUMMARY_ENABLED="1"
ROLLING_SUMMARY_MAX_MESSAGES="30"

# --- Cấu hình tìm kiếm Web ---
# Bật/tắt tính năng tìm kiếm web (1 là bật, 0 là tắt)
ALLOW_WEB_SEARCH="1"
# Chế độ bổ sung thông tin từ web (1 là tự động, 0 là thủ công)
AUGMENT_MODE="1"
# Số lượng kết quả tìm kiếm web
WEB_SEARCH_RESULTS="3"
```

### 3. Khởi tạo cơ sở dữ liệu và Admin

Chạy các lệnh sau từ thư mục `backend` để tạo database và tài khoản admin đầu tiên:

```bash
# Tạo các bảng trong database
python -c "from database import Base, engine; Base.metadata.create_all(bind=engine)"

# Tạo tài khoản admin
python init_admin.py
```

Bạn sẽ được yêu cầu nhập thông tin cho tài khoản admin.

### 4. Cài đặt Frontend

Di chuyển vào thư mục frontend và cài đặt các dependencies:

```bash
cd ../frontend
npm install
```

## Chạy ứng dụng

### 1. Chạy Backend

Mở một terminal, di chuyển đến `chroma_project_demo/backend` và chạy lệnh:

```bash
env:USE_OPENAI="1"
$env:OPENAI_API_KEY="your-api-key" 
$env:OPENAI_EMBED_MODEL="text-embedding-3-large" 
$env:OPENAI_CHAT_MODEL="gpt-4o-mini" 
$env:ENABLE_OCR="1" 
$env:OCR_LANG="vie+eng" 
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:HISTORY_TURNS="5"
$env:ROLLING_SUMMARY_ENABLED="1"
$env:ROLLING_SUMMARY_MAX_MESSAGES=”30”
$env:ALLOW_WEB_SEARCH=”1”
$env:AUGMENT_MODE=”1”
$env:WEB_SEARCH_RESULTS=”3”
uvicorn main:app --reload 
```

Backend API sẽ chạy tại `http://localhost:8000`.

### 2. Chạy Frontend

Mở một terminal khác, di chuyển đến `chroma_project_demo/frontend` và chạy lệnh:

```bash
npm run dev
```

Ứng dụng sẽ tự động mở trong trình duyệt tại `http://localhost:3002`.

## Các tính năng chính

* **Chatbot AI:** Giao diện chat để tương tác với LLM.
* **Admin Dashboard:**
  * **Tổng quan:** Xem thống kê hệ thống.
  * **Quản lý User:** Kích hoạt/vô hiệu hóa, cấp/hủy quyền admin.
  * **Thương hiệu:** Tùy chỉnh logo, tên và màu sắc chủ đạo của ứng dụng.

## Tài khoản admin và user đã tạo để test

## admin

```
Tài khoản : admin
Mật khẩu : admin123
```

## user

```
Tài khoản : DevNhan
Mật khẩu : admin123
```
