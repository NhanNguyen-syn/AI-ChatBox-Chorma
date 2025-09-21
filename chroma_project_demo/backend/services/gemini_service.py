"""
Google Gemini AI Service - Thay thế cho OpenAI/Ollama
Sử dụng Gemini Pro miễn phí với khả năng đọc hiểu tài liệu tốt
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
# Avoid importing sentence_transformers at module import time to prevent HF hub issues
import numpy as np

class GeminiService:
    def __init__(self):
        # Khởi tạo Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Khởi tạo embedding model local (miễn phí) - lazy import to avoid HF hub issues
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[Gemini] Initialized with local embeddings")
        except Exception as e:
            print(f"[Gemini] Warning: Could not load embedding model: {e}")
            self.embedding_model = None
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Tạo embeddings cho danh sách text"""
        if not self.embedding_model:
            # Fallback: tạo embeddings đơn giản dựa trên hash
            return [[hash(text) % 1000 / 1000.0] * 384 for text in texts]
        
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"[Gemini] Embedding error: {e}")
            return [[0.0] * 384 for _ in texts]
    
    def chat_with_context(self, message: str, context: str = "") -> Tuple[str, int, int]:
        """
        Chat với Gemini sử dụng context từ tài liệu
        Returns: (response, input_tokens, output_tokens)
        """
        try:
            # Tạo prompt với context
            if context.strip():
                prompt = f"""Bạn là trợ lý AI thông minh của công ty Dalat Hasfarm. 
Hãy trả lời câu hỏi dựa trên thông tin trong tài liệu được cung cấp.

THÔNG TIN TỪ TÀI LIỆU:
{context}

CÂU HỎI: {message}

HƯỚNG DẪN TRẢ LỜI:
1. Trả lời bằng tiếng Việt
2. Dựa vào thông tin trong tài liệu để trả lời chính xác
3. Nếu có số liệu, bảng biểu thì trình bày rõ ràng
4. Nếu không tìm thấy thông tin, hãy nói rõ "Không tìm thấy thông tin này trong tài liệu"
5. Trả lời chi tiết và hữu ích

TRẢ LỜI:"""
            else:
                prompt = f"""Bạn là trợ lý AI của công ty Dalat Hasfarm. 
Câu hỏi: {message}

Hãy trả lời bằng tiếng Việt một cách hữu ích. Nếu cần thông tin từ tài liệu cụ thể, hãy yêu cầu người dùng upload tài liệu liên quan."""

            # Gọi Gemini API
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                # Ước tính token count (Gemini không trả về chính xác)
                input_tokens = len(prompt.split()) * 1.3  # Ước tính
                output_tokens = len(response.text.split()) * 1.3
                
                return response.text.strip(), int(input_tokens), int(output_tokens)
            else:
                return "Xin lỗi, tôi không thể tạo phản hồi lúc này.", 0, 0
                
        except Exception as e:
            print(f"[Gemini] Chat error: {e}")
            return f"Lỗi khi xử lý: {str(e)}", 0, 0
    
    def extract_keywords(self, text: str) -> List[str]:
        """Trích xuất từ khóa quan trọng từ text"""
        try:
            prompt = f"""Trích xuất 5-10 từ khóa quan trọng nhất từ văn bản sau (chỉ trả về danh sách từ khóa, mỗi từ một dòng):

{text[:1000]}"""
            
            response = self.model.generate_content(prompt)
            if response and response.text:
                keywords = [kw.strip() for kw in response.text.strip().split('\n') if kw.strip()]
                return keywords[:10]  # Giới hạn 10 từ khóa
            return []
        except Exception as e:
            print(f"[Gemini] Keyword extraction error: {e}")
            # Fallback: tách từ đơn giản
            words = text.lower().split()
            return [w for w in words if len(w) > 3][:10]

# Singleton instance
_gemini_service = None

def get_gemini_service() -> GeminiService:
    """Lấy instance của GeminiService"""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service

def test_gemini_connection():
    """Test kết nối Gemini"""
    try:
        service = get_gemini_service()
        response, _, _ = service.chat_with_context("Xin chào, bạn có thể giúp tôi không?")
        print(f"[Gemini] Test successful: {response[:100]}...")
        return True
    except Exception as e:
        print(f"[Gemini] Test failed: {e}")
        return False

if __name__ == "__main__":
    # Test script
    test_gemini_connection()
