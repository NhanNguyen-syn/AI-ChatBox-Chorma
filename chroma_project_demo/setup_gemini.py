#!/usr/bin/env python3
"""
Setup script để cài đặt và test Google Gemini cho Chroma Project
"""

import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """Chạy command và hiển thị kết quả"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} thành công!")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} thất bại!")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} lỗi: {e}")
        return False

def check_python_version():
    """Kiểm tra Python version"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Cần Python 3.8 trở lên!")
        return False
    return True

def install_dependencies():
    """Cài đặt dependencies"""
    print("\n📦 Cài đặt dependencies...")
    
    # Cài đặt từ requirements.txt
    if not run_command("pip install -r backend/requirements.txt", "Cài đặt requirements.txt"):
        return False
    
    # Cài đặt thêm packages cần thiết
    packages = [
        "google-generativeai==0.3.2",
        "sentence-transformers==2.2.2",
        "pytesseract==0.3.10",
        "Pillow==10.0.1"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Cài đặt {package}"):
            print(f"⚠️  Không thể cài {package}, tiếp tục...")
    
    return True

def test_gemini_connection():
    """Test kết nối Gemini"""
    print("\n🧪 Test kết nối Google Gemini...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Chưa có GEMINI_API_KEY!")
        print("   Hướng dẫn:")
        print("   1. Truy cập: https://makersuite.google.com/app/apikey")
        print("   2. Tạo API key")
        print("   3. Thêm vào file .env: GEMINI_API_KEY=your-key-here")
        return False
    
    try:
        # Test import
        import google.generativeai as genai
        print("✅ Import google-generativeai thành công")
        
        # Test API connection
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        response = model.generate_content("Xin chào, bạn có thể trả lời bằng tiếng Việt không?")
        if response and response.text:
            print("✅ Kết nối Gemini thành công!")
            print(f"   Response: {response.text[:100]}...")
            return True
        else:
            print("❌ Gemini không trả về response")
            return False
            
    except Exception as e:
        print(f"❌ Test Gemini thất bại: {e}")
        return False

def test_sentence_transformers():
    """Test sentence transformers"""
    print("\n🧪 Test sentence-transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(["Hello world", "Xin chào"])
        print(f"✅ Sentence-transformers hoạt động! Embedding shape: {embeddings.shape}")
        return True
    except Exception as e:
        print(f"❌ Sentence-transformers lỗi: {e}")
        return False

def check_tesseract():
    """Kiểm tra Tesseract OCR"""
    print("\n🔍 Kiểm tra Tesseract OCR...")
    
    tesseract_paths = [
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        "/usr/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
        "tesseract"
    ]
    
    for path in tesseract_paths:
        if run_command(f'"{path}" --version', f"Test Tesseract tại {path}"):
            print(f"✅ Tesseract tìm thấy tại: {path}")
            return True
    
    print("❌ Không tìm thấy Tesseract!")
    print("   Hướng dẫn cài đặt:")
    print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   macOS: brew install tesseract tesseract-lang")
    print("   Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-vie")
    return False

def main():
    """Main setup function"""
    print("🚀 SETUP GOOGLE GEMINI CHO CHROMA PROJECT")
    print("=" * 50)
    
    # Kiểm tra Python version
    if not check_python_version():
        sys.exit(1)
    
    # Cài đặt dependencies
    if not install_dependencies():
        print("❌ Cài đặt dependencies thất bại!")
        sys.exit(1)
    
    # Load .env file nếu có
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env file")
    except:
        print("⚠️  Không tìm thấy .env file hoặc python-dotenv")
    
    # Test các components
    gemini_ok = test_gemini_connection()
    st_ok = test_sentence_transformers()
    tesseract_ok = check_tesseract()
    
    # Tổng kết
    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ SETUP:")
    print(f"   Google Gemini: {'✅' if gemini_ok else '❌'}")
    print(f"   Sentence Transformers: {'✅' if st_ok else '❌'}")
    print(f"   Tesseract OCR: {'✅' if tesseract_ok else '❌'}")
    
    if gemini_ok and st_ok:
        print("\n🎉 SETUP THÀNH CÔNG!")
        print("   Bạn có thể khởi động server và test chat với tài liệu!")
        print("   Chạy: cd backend && python -m uvicorn main:app --reload")
    else:
        print("\n⚠️  SETUP CHƯA HOÀN CHỈNH")
        print("   Vui lòng xem lại các lỗi ở trên và khắc phục")

if __name__ == "__main__":
    main()
