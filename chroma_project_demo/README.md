# Chroma AI Chat - DalatHasfarm

## Giới thiệu

Đây là một ứng dụng AI Chat nội bộ được xây dựng cho DalatHasfarm, sử dụng ChromaDB làm vector store và cho phép tương tác với các mô hình ngôn ngữ lớn (LLM) như OpenAI GPT hoặc các mô hình local qua Ollama. Giao diện quản trị cho phép cấu hình hệ thống, quản lý người dùng và tài liệu.

## Yêu cầu hệ thống

Trước khi bắt đầu, hãy đảm bảo bạn đã cài đặt các phần mềm sau:

*   **Python 3.10+**
*   **Node.js 18+** và **npm**
*   **(Tùy chọn) Ollama:** Nếu bạn muốn chạy các mô hình LLM local.
*   **(Tùy chọn) Docker:** Để chạy ChromaDB một cách dễ dàng.

## Hướng dẫn cài đặt

### 1. Cài đặt Backend

Di chuyển vào thư mục backend và cài đặt các dependencies:

```bash
cd chroma_project_demo/backend
pip install -r requirements.txt
```

### 2. Cấu hình Backend

Backend sử dụng các biến môi trường để cấu hình. Bạn có thể tạo một file `.env` trong thư mục `backend` hoặc thiết lập chúng trực tiếp trong terminal.

**File `.env` mẫu:**
```env
# Chọn 1 để sử dụng OpenAI, 0 để dùng Ollama/local models
USE_OPENAI=1

# Nếu USE_OPENAI=1, hãy cung cấp API key của bạn
OPENAI_API_KEY="sk-...

# (Tùy chọn) Tên các mô hình OpenAI
OPENAI_CHAT_MODEL="gpt-4o-mini"
OPENAI_EMBED_MODEL="text-embedding-3-large"

# (Tùy chọn) Tên các mô hình Ollama nếu không dùng OpenAI
# OLLAMA_CHAT_MODEL="llama2"
# OLLAMA_EMBED_MODEL="llama2"
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

Ứng dụng sẽ tự động mở trong trình duyệt tại `http://localhost:3000`.

## Các tính năng chính

*   **Chatbot AI:** Giao diện chat để tương tác với LLM.
*   **Admin Dashboard:**
    *   **Tổng quan:** Xem thống kê hệ thống.
    *   **Quản lý User:** Kích hoạt/vô hiệu hóa, cấp/hủy quyền admin.
    *   **Cấu hình AI:** Thay đổi mô hình AI, embedding model, ngưỡng similarity và max tokens.
    *   **Thương hiệu:** Tùy chỉnh logo, tên và màu sắc chủ đạo của ứng dụng.

