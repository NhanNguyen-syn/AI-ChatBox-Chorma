#!/usr/bin/env python3
"""
Setup script để test OpenAI cho Chroma Project
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
        "openai>=1.10.0",
        "pytesseract==0.3.10",
        "Pillow==10.0.1"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Cài đặt {package}"):
            print(f"⚠️  Không thể cài {package}, tiếp tục...")
    
    return True

def test_openai_connection():
    """Test kết nối OpenAI"""
    print("\n🧪 Test kết nối OpenAI...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Chưa có OPENAI_API_KEY!")
        print("   Hướng dẫn:")
        print("   1. Truy cập: https://platform.openai.com/api-keys")
        print("   2. Tạo API key")
        print("   3. Thêm vào file .env: OPENAI_API_KEY=sk-your-key-here")
        return False
    
    if not api_key.startswith('sk-'):
        print("❌ OPENAI_API_KEY không đúng định dạng (phải bắt đầu bằng 'sk-')")
        return False
    
    try:
        # Test import
        import openai
        print("✅ Import openai thành công")
        
        # Test API connection
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Xin chào, bạn có thể trả lời bằng tiếng Việt không?"}],
            max_tokens=100
        )
        
        if response and response.choices:
            print("✅ Kết nối OpenAI thành công!")
            print(f"   Response: {response.choices[0].message.content[:100]}...")
            
            # Test embeddings
            embed_response = client.embeddings.create(
                model="text-embedding-3-small",
                input="Test embedding"
            )
            if embed_response and embed_response.data:
                print("✅ OpenAI Embeddings hoạt động!")
                print(f"   Embedding dimension: {len(embed_response.data[0].embedding)}")
            
            return True
        else:
            print("❌ OpenAI không trả về response")
            return False
            
    except Exception as e:
        print(f"❌ Test OpenAI thất bại: {e}")
        if "insufficient_quota" in str(e):
            print("   💳 Lỗi: Tài khoản OpenAI hết quota hoặc chưa có billing")
            print("   Vui lòng kiểm tra: https://platform.openai.com/usage")
        elif "invalid_api_key" in str(e):
            print("   🔑 Lỗi: API key không hợp lệ")
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

def check_env_file():
    """Kiểm tra file .env"""
    print("\n📄 Kiểm tra file .env...")
    
    if not os.path.exists('.env'):
        print("❌ Không tìm thấy file .env!")
        print("   Tạo file .env từ template:")
        print("   cp .env.example .env")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
    
    if 'USE_OPENAI=1' in content:
        print("✅ USE_OPENAI=1 đã được cấu hình")
    else:
        print("⚠️  Chưa cấu hình USE_OPENAI=1")
    
    if 'OPENAI_API_KEY=' in content and 'sk-' in content:
        print("✅ OPENAI_API_KEY đã được cấu hình")
    else:
        print("⚠️  Chưa cấu hình OPENAI_API_KEY")
    
    return True

def main():
    """Main setup function"""
    print("🚀 SETUP OPENAI CHO CHROMA PROJECT")
    print("=" * 50)
    
    # Kiểm tra Python version
    if not check_python_version():
        sys.exit(1)
    
    # Kiểm tra .env file
    check_env_file()
    
    # Load .env file nếu có
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Loaded .env file")
    except:
        print("⚠️  Không tìm thấy .env file hoặc python-dotenv")
    
    # Cài đặt dependencies
    if not install_dependencies():
        print("❌ Cài đặt dependencies thất bại!")
        sys.exit(1)
    
    # Test các components
    openai_ok = test_openai_connection()
    tesseract_ok = check_tesseract()
    
    # Tổng kết
    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ SETUP:")
    print(f"   OpenAI API: {'✅' if openai_ok else '❌'}")
    print(f"   Tesseract OCR: {'✅' if tesseract_ok else '❌'}")
    
    if openai_ok:
        print("\n🎉 SETUP THÀNH CÔNG!")
        print("   Bạn có thể khởi động server và test chat với tài liệu!")
        print("   Chạy: cd backend && python -m uvicorn main:app --reload")
        print("\n💡 Tips:")
        print("   - Sử dụng gpt-4o-mini để tiết kiệm chi phí")
        print("   - Monitor usage tại: https://platform.openai.com/usage")
    else:
        print("\n⚠️  SETUP CHƯA HOÀN CHỈNH")
        print("   Vui lòng xem lại các lỗi ở trên và khắc phục")
        print("   Đặc biệt kiểm tra OPENAI_API_KEY và billing account")

if __name__ == "__main__":
    main()
