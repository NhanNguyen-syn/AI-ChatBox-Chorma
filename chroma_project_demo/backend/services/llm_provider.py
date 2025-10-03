import os
from typing import Any, Dict, List, Optional, Tuple

# Import provider clients
from openai import OpenAI

# --- Provider Configuration ---
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"

# Lazy init OpenAI to avoid raising during import if key missing
try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if USE_OPENAI else None
except Exception as _e:
    print(f"[LLM Provider] OpenAI init failed: {_e}")
    openai_client = None

_gemini_service = None

def _get_gemini():
    global _gemini_service
    if _gemini_service is None:
        try:
            # Import lazily to avoid import-time errors if deps are missing
            from services.gemini_service import get_gemini_service  # type: ignore
            _gemini_service = get_gemini_service()
        except Exception as e:
            print(f"[LLM Provider] Gemini unavailable: {e}")
            _gemini_service = None
    return _gemini_service

# Define provider priority
PRIMARY_PROVIDER = os.getenv("PRIMARY_LLM_PROVIDER", "openai").lower()
SECONDARY_PROVIDER = os.getenv("SECONDARY_LLM_PROVIDER", "gemini").lower()

# --- Unified Interface ---

def get_chat_response(
    message: str,
    context: str = "",
    history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, int, str]:
    """Get a chat response from the best available provider, with failover.

    Returns:
        Tuple[str, int, str]: (response_text, tokens_used, provider_name)
    """
    providers = [PRIMARY_PROVIDER, SECONDARY_PROVIDER]

    # Normalize history to expected schema
    safe_history: List[Dict[str, str]] = []
    if history:
        for m in history:
            try:
                role = str(m.get("role", "user"))
                content = str(m.get("content", ""))
                if content:
                    safe_history.append({"role": role, "content": content})
            except Exception:
                continue

    system_msg: Optional[Dict[str, str]] = None
    system_prompt = (
        "Bạn là một trợ lý AI nội bộ của Dalat Hasfarm, chỉ được phép sử dụng thông tin do công ty cung cấp. "
        '**QUY TẮC BẮT BUỘC:**\n'
        '1.  **PHẠM VI NỘI BỘ:** Chỉ được phép trả lời câu hỏi của người dùng dựa **DUY NHẤT** vào thông tin trong mục \'Ngữ cảnh\' dưới đây. Tuyệt đối không sử dụng kiến thức bên ngoài hoặc thông tin bạn tự biết.\n'
        '2.  **TRÍCH DẪN CHÍNH XÁC:** Khi có số liệu, bảng biểu, hoặc dữ liệu cụ thể trong ngữ cảnh, hãy **trích dẫn trực tiếp và chính xác** các thông tin đó.\n'
        '3.  **TRỰC TIẾP, NGẮN GỌN:** Luôn đi thẳng vào câu trả lời, không giới thiệu chung chung, không thêm các câu từ không cần thiết.\n'
        "4.  **KHÔNG CÓ DỮ LIỆU:** Nếu 'Ngữ cảnh' không chứa thông tin để trả lời câu hỏi, bạn chỉ được phép nói một câu duy nhất: 'Tôi không tìm thấy thông tin này trong tài liệu nội bộ của công ty.'\n\n"
        f"Ngữ cảnh:\n---\n{context}\n---"
    )
    system_msg = {"role": "system", "content": system_prompt}

    for provider in providers:
        try:
            if provider == "openai" and openai_client:
                msgs: List[Dict[str, str]] = []
                if system_msg:
                    msgs.append(system_msg)
                msgs.extend(safe_history)
                msgs.append({"role": "user", "content": message})

                response = openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
                    messages=msgs,
                    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
                )
                text = (response.choices[0].message.content or "").strip()
                tokens = getattr(response, "usage", None).total_tokens if getattr(response, "usage", None) else 0
                return text, tokens, "openai"

            elif provider == "gemini":
                gs = _get_gemini()
                if gs:
                    res_text, in_toks, out_toks = gs.chat_with_context(message=message, context=context)
                    return res_text, int(in_toks or 0) + int(out_toks or 0), "gemini"

        except Exception as e:
            print(f"[LLM Provider] {provider.upper()} failed: {e}. Trying next provider.")
            continue

    # If all providers fail
    return "Xin lỗi, hiện tại tôi không thể trả lời câu hỏi của bạn. Vui lòng thử lại sau.", 0, "none"

