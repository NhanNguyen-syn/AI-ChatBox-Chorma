import uuid
import os
import uuid
import base64
import json
from datetime import datetime, timezone
from typing import List


from typing import List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from typing import Optional, List
import json

import pandas as pd
from io import StringIO

from PIL import Image
from services.ocr_utils import ocr_with_confidence, preprocess_image

from typing import TypedDict, Literal, Any as _Any

class DecomposedQuery(TypedDict):
    intent: Literal["COMPARISON", "SYNTHESIS", "CAUSALITY", "SIMPLE_QA"]
    retrieval_strategy: Literal["parallel", "sequential"]
    sub_questions: list[str]

DEFAULT_SIMPLE_QUERY: DecomposedQuery = {
    "intent": "SIMPLE_QA",
    "retrieval_strategy": "parallel",
    "sub_questions": []
}

# New: Intent classification tailored for Excel/CRM queries
class NLQIntentResult(TypedDict):
    # Vietnamese categories as requested
    intent: Literal["LOOKUP", "FILTER_LIST", "COMPARISON", "AGGREGATION", "DESCRIPTIVE"]
    # Key-value pairs for later SQL filters (e.g., {"sku": "F01012PA"})
    entities: dict[str, _Any]

# Optional local embeddings + flashrank reranker
try:
    from sentence_transformers import SentenceTransformer
    _st_model: Optional[SentenceTransformer] = None
except Exception:
    SentenceTransformer = None  # type: ignore
    _st_model = None  # type: ignore

# Optional CrossEncoder reranker
rerank_model = None
try:
    from sentence_transformers import CrossEncoder
    try:
        # This model is lightweight and fast, suitable for quick reranking.
        rerank_model = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2', max_length=512)
        print("[Chat] CrossEncoder reranker initialized successfully.")
    except Exception as e:
        print(f"[Chat] Could not initialize CrossEncoder for reranking: {e}")
        rerank_model = None
except Exception as e:
    print(f"[Chat] sentence-transformers not available for reranking: {e}")
    rerank_model = None
import os

# Provider selection flags
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Defaults; will be overridden by DB configs at runtime where applicable
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
openai_client = None
if USE_OPENAI:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("[Chat] Using OpenAI for chat and embeddings")
    except Exception as e:
        print(f"Warning: Could not init OpenAI client: {e}")
        USE_OPENAI = False
# Toggle debug logs (set DEBUG_LOGS=1 in .env to enable)
DEBUG_LOGS = os.getenv("DEBUG_LOGS", "0") == "1"
# Strict RAG mode: if enabled, refuse to answer when no relevant context is found
# Default OFF to ensure the assistant still answers even when retrieval misses context.
STRICT_CONTEXT_ONLY = os.getenv("STRICT_CONTEXT_ONLY", "0") == "1"

# ---- Runtime System Config loader (from DB) ----
from database import SessionLocal, SystemConfig  # type: ignore

_config_cache = {"ts": 0.0, "map": {}}

def _get_system_config_map(force: bool = False) -> dict:
    """Load key/value configs from SystemConfig, cached for 30s."""
    import time
    global _config_cache
    if not force and (time.time() - _config_cache.get("ts", 0)) < 30:
        return _config_cache.get("map", {})
    try:
        db = SessionLocal()
        rows = db.query(SystemConfig).all()
        m = {r.key: r.value for r in rows}
        _config_cache = {"ts": time.time(), "map": m}
        db.close()
        return m
    except Exception:
        return _config_cache.get("map", {})

def _cfg_chat_model(default: str | None = None) -> str:
    m = _get_system_config_map()
    return m.get("chat_model") or default or OPENAI_CHAT_MODEL

def _cfg_embed_model(default: str | None = None) -> str:
    m = _get_system_config_map()
    return m.get("embed_model") or default or OPENAI_EMBED_MODEL

def _cfg_max_tokens(default: int = 800) -> int:
    m = _get_system_config_map()
    try:
        return int(m.get("max_tokens", default))
    except Exception:
        return default

def _cfg_similarity_threshold(default: float = 0.45) -> float:
    m = _get_system_config_map()
    try:
        return float(m.get("similarity_threshold", default))
    except Exception:
        return default

# Number of recent user turns to augment retrieval with (for query expansion)
try:
    HISTORY_USER_TURNS = max(1, int(os.getenv("HISTORY_USER_TURNS", "3")))
except Exception:
    HISTORY_USER_TURNS = 3

# Number of chat history pairs (user+assistant) to include in model prompts
# Example: HISTORY_TURNS=5 -> include up to 10 recent messages (5 pairs)
try:
    HISTORY_TURNS = max(1, int(os.getenv("HISTORY_TURNS", "5")))
except Exception:
    HISTORY_TURNS = 5
HISTORY_MAX_HISTORY_MESSAGES = HISTORY_TURNS * 2
# Temporary context TTL (minutes) for session-scoped uploads
try:
    TEMP_CONTEXT_TTL_MINUTES = max(5, int(os.getenv("TEMP_CONTEXT_TTL_MINUTES", "240")))
except Exception:
    TEMP_CONTEXT_TTL_MINUTES = 240

# Max upload size for temporary context files (MB)
try:
    MAX_CONTEXT_FILE_MB = max(1, int(os.getenv("MAX_CONTEXT_FILE_MB", "15")))
except Exception:
    MAX_CONTEXT_FILE_MB = 15


def dlog(msg: str):
    if DEBUG_LOGS:
        print(msg)

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from fastapi import UploadFile, File, Form
# Rolling summary configuration
ROLLING_SUMMARY_ENABLED = os.getenv("ROLLING_SUMMARY_ENABLED", "1") == "1"
try:
    ROLLING_SUMMARY_MAX_MESSAGES = max(8, int(os.getenv("ROLLING_SUMMARY_MAX_MESSAGES", "30")))
except Exception:
    ROLLING_SUMMARY_MAX_MESSAGES = 30

from typing import List

from database import get_db, ChatSession, ChatMessage, User, get_chroma_collection, get_all_chroma_collections, CrmProduct, OcrText, AllowanceTable, ExcelRow, engine, TemporaryContext
from auth.jwt_handler import verify_token
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Import Gemini service
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from services.gemini_service import get_gemini_service
    GEMINI_AVAILABLE = True
    print("[Chat] Gemini service imported successfully")
except Exception as e:
    print(f"[Chat] Gemini service not available: {e}")
    GEMINI_AVAILABLE = False

router = APIRouter()
security = HTTPBearer()

# Verify user using HTTP Bearer like users router

def verify_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    payload = verify_token(credentials.credentials)
    user = db.query(User).filter(User.username == payload["sub"]).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid user")
    return user

# Pydantic models
class ChatMessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatMessageResponse(BaseModel):
    id: str
    message: str
    response: str
    timestamp: datetime
    session_id: str
    sources: Optional[List[dict]] = None
    suggestions: Optional[List[str]] = None

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

# Initialize models
try:
    if USE_OPENAI and openai_client:
        embeddings = None  # we'll call OpenAI embedding endpoint directly
        llm = None         # we'll call OpenAI chat endpoint directly
    else:
        # Try Ollama first
        try:
            embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "llama2"))
            llm = Ollama(model=os.getenv("OLLAMA_CHAT_MODEL", "llama2"))
            print("[Chat] Using Ollama for chat+embeddings")
        except Exception as oll_err:
            print(f"[Chat] Ollama unavailable: {oll_err}")
            embeddings = None
            llm = None
            # Fallback to local sentence-transformers if available (embeddings only)
            if SentenceTransformer is not None:
                try:
                    _st_model = SentenceTransformer(os.getenv("ST_EMBED_MODEL", "all-MiniLM-L6-v2"))
                    print("[Chat] Using sentence-transformers for embeddings")
                except Exception as st_err:
                    print(f"[Chat] sentence-transformers init failed: {st_err}")

except Exception as e:
    print(f"Warning: Could not initialize models: {e}")
    embeddings = None
    llm = None

import unicodedata

def _normalize_vi(s: str) -> str:
    try:
        return unicodedata.normalize('NFD', (s or '')).encode('ascii', 'ignore').decode('ascii').lower().strip()
    except Exception:
        return (s or '').lower().strip()

def _is_greeting(msg: str) -> bool:
    s = (msg or "").strip()
    if not s:
        return False
    # Emoji/siêu ngắn coi như chào
    if any(ch in s for ch in ["👋","🤝","🙏","🙂","😊"]) and len(s) <= 8:
        return True
    n = _normalize_vi(s).strip("!.?,;: ")
    pats = [
        'chao', 'xin chao', 'chao ban', 'chao bot', 'alo', 'alo ban',
        'hi', 'hi there', 'hello', 'hello there', 'hey', 'yo', 'sup',
        'chao buoi sang', 'chao buoi chieu', 'chao buoi toi', 'chao moi nguoi',
        'e chao', 'em chao', 'chao ad', 'xin chao ad', 'chao shop',
        'good morning', 'good afternoon', 'good evening', 'good day'
    ]
    if any(n == p or n.startswith(p) for p in pats):
        return True
    if n in {"hi","hello","hey","yo","alo","chao"}:
        return True
    return False

def _requires_internal_docs(msg: str) -> bool:
    """Kiểm tra câu hỏi có cần tài liệu nội bộ không"""
    n = _normalize_vi(msg)
    # Các từ khóa yêu cầu tài liệu nội bộ
    internal_keywords = [
        'san pham', 'gia ca', 'gia ban', 'khuyen mai', 'giam gia',
        'dich vu', 'cong ty', 'shop', 'cua hang', 'chi nhanh',
        'lien he', 'dia chi', 'so dien thoai', 'email',
        'chinh sach', 'bao hanh', 'doi tra', 'van chuyen',
        'thanh toan', 'the tin dung', 'chuyen khoan',
        'don hang', 'ma don hang', 'tracking', 'giao hang',
        'catalog', 'danh muc', 'menu', 'bang gia'
    ]
    return any(kw in n for kw in internal_keywords)


# --- Text-to-SQL Agent Components ---
from sqlalchemy import MetaData

def get_db_schema_for_llm(engine) -> str:
    """Inspects the database and returns a simplified schema for the LLM prompt."""
    try:

        meta = MetaData()
        meta.reflect(bind=engine)
        schema_lines = []

        # Whitelist tables that are useful for Text-to-SQL
        allowed_tables = ["crm_products", "excel_rows", "allowance_tables"]

        for table_name in allowed_tables:
            if table_name in meta.tables:
                table = meta.tables[table_name]
                schema_lines.append(f"Table `{table.name}` has columns:")
                for column in table.columns:
                    col_type = str(column.type)
                    # Add hints for important columns
                    comment = ""
                    if column.name in ["sku", "name", "normalized_text", "sheet_name"]:
                        comment = " (good for filtering)"
                    if column.name == 'data_json':
                        comment = " (contains full row data as JSON)"

                    schema_lines.append(f"  - `{column.name}` (type: {col_type}){comment}")
                schema_lines.append("") # Add a blank line for readability

        return "\n".join(schema_lines)
    except Exception as e:
        print(f"[DB Schema] Failed to generate schema: {e}")
        return "Error: Could not retrieve database schema."


def _execute_sql_and_format_results(sql_query: str, engine) -> str:
    """Executes a read-only SQL query and formats the results as a Markdown table."""
    if not sql_query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."
    try:
        with engine.connect() as connection:
            result = connection.execute(sql_text(sql_query))
            rows = result.mappings().all()
            if not rows:
                return "No results found."

            headers = list(rows[0].keys())
            md_lines = [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |"
            ]
            for row in rows:
                md_lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")

            return "\n".join(md_lines)
    except Exception as e:
        return f"Error executing SQL query: {e}"


def _generate_sql_query_with_llm(user_query: str, db_schema: str) -> str:
    """Uses an LLM to generate a safe SQL query from a natural language question."""
    if not (USE_OPENAI and openai_client):
        return ""

    prompt = f"""
You are an expert Text-to-SQL assistant. Your task is to convert a user's question into a single, valid SQLite SELECT statement.

**Rules:**
1.  **ONLY `SELECT`:** You must only generate `SELECT` statements. Do not generate `INSERT`, `UPDATE`, `DELETE`, `DROP`, or any other type of statement.
2.  **Use the Schema:** Base your query on the provided database schema. Do not invent table or column names.
3.  **Be Specific:** Use `WHERE` clauses to filter data according to the user's question. Use `LIKE` for partial text matches.
4.  **Handle Ambiguity:** If a column name is ambiguous (e.g., 'name'), prefer the one from the most relevant table (e.g., `crm_products.name`).
5.  **Return Only SQL:** Your entire output should be just the raw SQL statement. Do not include explanations, backticks, or any other text.

**Database Schema:**
```
{db_schema}
```

**User Question:** "{user_query}"

**Generated SQL Statement:**
    """
    try:
        res = openai_client.chat.completions.create(
            model=_cfg_chat_model(OPENAI_CHAT_MODEL),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=min(300, _cfg_max_tokens(300)),
        )
        sql_query = (res.choices[0].message.content or "").strip()
        # Final safety check
        if sql_query.strip().upper().startswith("SELECT"):
            return sql_query
        return "" # Return empty if it's not a SELECT query
    except Exception as e:
        print(f"[Text2SQL] LLM query generation failed: {e}")
        return ""

def run_text_to_sql_agent(user_query: str, engine) -> Optional[str]:
    """Orchestrates the Text-to-SQL process."""
    print("[Agent] Attempting to run Text-to-SQL agent...")
    db_schema = get_db_schema_for_llm(engine)
    if "Error:" in db_schema:
        return None

    sql_query = _generate_sql_query_with_llm(user_query, db_schema)
    if not sql_query:
        print("[Agent] Text-to-SQL LLM failed to generate a valid SELECT query.")
        return None

    print(f"[Agent] Generated SQL: {sql_query}")
    result = _execute_sql_and_format_results(sql_query, engine)
    print(f"[Agent] SQL execution result length: {len(result)}")

    if "Error:" in result or "No results found." in result:
        return None

    return result

# ------- CRM retrieval helpers -------

# --- Web Search Agent Components ---
# Optional dependency: duckduckgo_search
try:
    from duckduckgo_search import DDGS  # type: ignore
    _DDG_AVAILABLE = True
except Exception as e:
    print(f"[WebSearch] duckduckgo_search not available: {e}")
    _DDG_AVAILABLE = False
import asyncio


async def _get_search_results(query: str, num_results: int = 3) -> str:
    """Performs a web search and returns a concatenated string of snippets."""
    if not _DDG_AVAILABLE:
        return ""
    snippets = []
    try:
        async with DDGS() as ddgs:  # type: ignore
            results = await asyncio.gather(*[ddgs.text(query, region='wt-wt', safesearch='off', timelimit='y', max_results=num_results)])
            for r in results[0]:
                snippet = r.get('body', '')
                if snippet:
                    snippets.append(f"Source: {r.get('href', '')}\nSnippet: {snippet}")
    except Exception as e:
        print(f"[WebSearch] DuckDuckGo search failed: {e}")
        return ""
    return "\n\n---\n\n".join(snippets)

async def run_web_search_agent(user_query: str) -> Optional[str]:
    """Orchestrates the web search process and synthesizes an answer."""
    print("[Agent] Attempting to run Web Search agent...")
    if not (USE_OPENAI and openai_client):
        return None

    search_context = await _get_search_results(user_query)
    if not search_context:
        print("[Agent] Web search returned no context.")
        return None

    prompt = f"""
You are a research assistant. Your task is to answer the user's question based *only* on the provided web search results.

**Rules:**
1.  **Synthesize, Don't Copy:** Combine information from the snippets to form a coherent answer. Do not just copy-paste the snippets.
2.  **Cite Sources:** At the end of your answer, list the URLs of the sources you used.
3.  **Be Concise:** Keep the answer focused on the user's question.
4.  **If Unsure, State It:** If the search results do not contain enough information to answer the question, say "Dựa trên kết quả tìm kiếm, tôi không thể tìm thấy câu trả lời chính xác cho câu hỏi này."

**Web Search Results:**
```
{search_context}
```

**User Question:** "{user_query}"

**Synthesized Answer:**
"""
    try:
        res = openai_client.chat.completions.create(
            model=_cfg_chat_model(OPENAI_CHAT_MODEL),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=min(500, _cfg_max_tokens(500)),
        )
        answer = (res.choices[0].message.content or "").strip()
        print(f"[Agent] Web Search agent synthesized answer of length: {len(answer)}")
        return answer
    except Exception as e:
        print(f"[WebSearch] LLM synthesis failed: {e}")
        return None


# --- Tool Router & Orchestrator ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI as LangchainChatOpenAI # Alias to avoid conflict with our own client
from typing import Literal

class ToolChoice(BaseModel):
    """A choice of which tool to use to answer the user's question."""
    tool_name: Literal["vector_search", "text_to_sql", "web_search", "general_conversation"] = Field(
        ..., description="The name of the tool to use."
    )

async def run_tool_router(user_query: str) -> str:
    """Uses an LLM to decide which tool is best suited for the user's query."""
    print(f"[Router] Routing query: '{user_query}'")
    if not (USE_OPENAI and openai_client):
        # Default to vector search if the router isn't available
        return "vector_search"

    tool_descriptions = """
    - `vector_search`: Use for questions about company policies, product information from documents, internal regulations, and general knowledge contained within the uploaded files.
    - `text_to_sql`: Use for questions that require querying a database with structured data, such as calculating totals, averages, listing items with specific criteria, or analyzing data from tables like crm_products, allowances, or excel_rows.
    - `web_search`: Use for questions about recent events, news, general knowledge not found in internal documents, or information about external entities (e.g., 'what is the capital of France?').
    - `general_conversation`: Use for greetings, chit-chat, or when no specific tool is needed to answer the question.
    """

    prompt_template = f"""
You are an expert AI router. Your task is to determine the best tool to use to answer the user's question based on the tool descriptions below. You must choose one and only one tool.

**Tools:**
{tool_descriptions}

**User Question:**
"{{user_query}}"

Based on the user's question, which tool should be used?
"""

    try:
        # Use a separate LangChain client for this structured output task
        lc_chat = LangchainChatOpenAI(model=_cfg_chat_model(OPENAI_CHAT_MODEL), temperature=0, openai_api_key=OPENAI_API_KEY)
        structured_llm = lc_chat.with_structured_output(ToolChoice)

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | structured_llm

        result = await chain.ainvoke({{"user_query": user_query}})
        print(f"[Router] Decision: {result.tool_name}")
        return result.tool_name
    except Exception as e:
        print(f"[Router] Failed to route query: {e}. Defaulting to vector_search.")
        return "vector_search"

from sqlalchemy import or_

def _extract_keywords(msg: str) -> list[str]:
    s = (_normalize_vi(msg) or "").replace("\n", " ")
    toks = [t.strip() for t in s.split() if len(t.strip()) >= 3]
    # pick up token after patterns like 'ma', 'ma sp', 'sku', 'code'
    keys = []
    for i, t in enumerate(toks):
        if t in {"ma", "sku", "code"} and i+1 < len(toks):
            keys.append(toks[i+1])
    # also include all tokens of length>=3 (dedup)
    for t in toks:
        if t not in keys:
            keys.append(t)
    return keys[:6]


def find_crm_products_context(db: Session, user_query: str, limit: int = 10) -> tuple[str, list[dict]]:
    """Search CrmProduct using FTS5 and format results as a Markdown table."""
    try:
        keys = _extract_keywords(user_query)
        if not keys:
            return "", []

        # Use FTS5 for searching
        match_query = " OR ".join([f'"{k}"' for k in keys if k])
        sql = sql_text("""
            SELECT c.* FROM crm_products c
            JOIN crm_products_fts f ON f.id = c.id
            WHERE f.crm_products_fts MATCH :match
            ORDER BY rank
            LIMIT :limit
        """)
        with engine.connect() as conn:
            res = conn.execute(sql, {"match": match_query, "limit": limit})
            rows = res.mappings().all()

        if not rows:
            return "", []

        # Format as Markdown table
        headers = ["SKU", "Tên sản phẩm", "Giá", "Danh mục", "Mô tả"]
        md_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        sources = []
        for r in rows:
            price = str(r.get('price') or "")
            if price and r.get('currency'):
                price += f" {r.get('currency')}"

            desc = (r.get('description') or "")
            desc_short = (desc[:80] + '...') if len(desc) > 80 else desc
            desc_short = desc_short.replace("\n", " ")

            md_lines.append("| " + " | ".join([
                str(r.get('sku') or ""),
                str(r.get('name') or ""),
                price,
                str(r.get('category') or ""),
                desc_short
            ]) + " |")
            if r.get('source_filename'):
                cite = {"title": f"CRM: {r.get('source_filename')}", "row": r.get('row_index')}
                if cite not in sources:
                    sources.append(cite)

        ctx = "**Bảng dữ liệu sản phẩm (CRM) phù hợp:**\n" + "\n".join(md_lines)
        return ctx, sources
    except Exception as e:
        print(f"[Chat] CRM FTS search failed: {e}")
        return "", []

# ------- OCR SQL retrieval helpers -------

from sqlalchemy import text as sql_text
from database import engine

_VN_SYNONYMS = {
    "phu cap": ["phụ cấp", "tro cap", "phuc loi", "allowance"],
    "an trua": ["ăn trưa", "bua trua", "com trua"],
    "muc": ["mức", "gia", "so tien", "bao nhieu"],
    "tang": ["tăng", "dieu chinh", "nang", "tang them"],
    "tp.hcm": ["hcm", "tp hcm", "tp.hcm", "ho chi minh", "hồ chí minh"],
    "ha noi": ["hà nội", "ha noi"],
    "can tho": ["cần thơ", "can tho"],
}

def _expand_keywords(keys: list[str]) -> list[str]:
    expanded = set()
    for k in keys:
        nk = _normalize_vi(k)
        if nk:
            expanded.add(nk)
        for root, syns in _VN_SYNONYMS.items():
            if root in nk:
                for s in syns:
                    expanded.add(_normalize_vi(s))
    return list(expanded)


def find_ocr_sql_context(db: Session, user_query: str, limit: int = 5) -> tuple[str, list[dict]]:
    try:
        qnorm = _normalize_vi(user_query)
        base_keys = _extract_keywords(user_query)
        keys = _expand_keywords(base_keys)
        if not keys:
            return "", []
        rows: list[OcrText] = []
        # 1) FTS5 search (best-effort)
        try:
            match = " OR ".join([f'"{k}"' for k in keys if k])
            sql = sql_text("""
                SELECT o.id FROM ocr_texts o
                JOIN ocr_texts_fts f ON f.rowid = o.rowid
                WHERE f.normalized_content MATCH :match
                LIMIT :lim
            """)
            with engine.connect() as conn:
                res = conn.execute(sql, {"match": match, "lim": limit * 20})
                ids = [r[0] for r in res.fetchall()]
            if ids:
                rows = db.query(OcrText).filter(OcrText.id.in_(ids)).all()
        except Exception as _fts_err:
            dlog(f"[Chat] FTS search skipped: {_fts_err}")
        # 2) LIKE fallback
        if not rows:
            filters = [OcrText.normalized_content.ilike(f"%{_normalize_vi(k)}%") for k in keys]
            q = db.query(OcrText)
            if filters:
                q = q.filter(or_(*filters))
            rows = q.limit(limit * 20).all()
        if not rows:
            return "", []
        # Rank and build context with citations
        wants_number = any(w in qnorm for w in ["bao nhieu", "muc", "gia", "so tien", "tang"]) or any(ch.isdigit() for ch in user_query)
        q_tokens = [t for t in qnorm.split() if len(t) >= 3]
        def rank_score(row: OcrText) -> float:
            content_norm = (row.normalized_content or _normalize_vi(row.content or ""))
            toks = [t for t in content_norm.split() if len(t) >= 3]
            if not toks:
                return 0.0
            hits = sum(1 for t in q_tokens if t in content_norm)
            num_bonus = 0.2 if (wants_number and any(ch.isdigit() for ch in (row.content or ""))) else 0.0
            table_bonus = 0.15 if (row.block_type == 'table') else 0.0
            return hits / max(1, len(set(toks))) + num_bonus + table_bonus
        rows.sort(key=rank_score, reverse=True)
        top_rows = rows[:limit]
        snippets: list[str] = []
        sources: list[dict] = []
        for r in top_rows:
            content = (r.content or "").strip()
            if not content:
                continue
            cite = f"Nguồn: {r.source_filename or '(unknown)'}"
            if r.page:
                cite += f", trang {r.page}"
            if r.section:
                cite += f", mục {r.section}"
            if r.block_type:
                cite += f" ({r.block_type})"
            snippets.append(f"- {content[:800]}\n  [{cite}]")
            sources.append({"title": r.source_filename or "(unknown)", "page": r.page, "section": r.section, "block_type": r.block_type})
        if not snippets:
            return "", []
        ctx = "Trích từ tài liệu:\n" + "\n".join(snippets)
        return ctx, sources
    except Exception as e:
        print(f"[Chat] OCR SQL search failed: {e}")
        return "", []

# ------- Allowance Table SQL retrieval helpers -------

def find_allowance_sql_context(db: Session, user_query: str, limit: int = 10) -> tuple[str, list[dict]]:
    """Search structured AllowanceTable by keywords and format concise context."""
    q_norm = _normalize_vi(user_query)
    keywords = ["phu cap", "tro cap", "muc an", "tien an", "allowance"]
    if not any(kw in q_norm for kw in keywords):
        return "", []

    try:
        # Extract region keywords from the query
        region_filters = []
        for region_kw in _all_region_keywords:
            if _normalize_vi(region_kw) in q_norm:
                region_filters.append(AllowanceTable.khu_vuc.ilike(f"%{region_kw}%"))

        q = db.query(AllowanceTable)
        if region_filters:
            q = q.filter(or_(*region_filters))

        rows = q.order_by(AllowanceTable.created_at.desc()).limit(limit).all()
        if not rows:
            return "", []

        lines = []
        sources = []
        for r in rows:
            line = f"- Khu vực: {r.khu_vuc} | Phụ cấp mới: {_format_vnd(r.phu_cap_moi)}đ (tăng {_format_vnd(r.muc_tang)}đ so với mức cũ {_format_vnd(r.phu_cap_cu)}đ)"
            lines.append(line)
            if r.source_filename:
                cite = {"title": f"SQL: {r.source_filename}", "page": r.page}
                if cite not in sources:
                    sources.append(cite)

        ctx = "Thông tin Phụ cấp từ SQL:\n" + "\n".join(lines)
        return ctx, sources
    except Exception as e:
        print(f"[Chat] Allowance SQL search failed: {e}")
        return "", []


# ------- Raw Excel SQL retrieval helpers -------

def find_excel_sql_context(db: Session, user_query: str, limit: int = 15) -> tuple[str, list[dict]]:
    """Search ExcelRow using FTS5 and format results as a dynamic Markdown table."""
    try:
        keys = _extract_keywords(user_query)
        if not keys:
            return "", []

        match_query = " OR ".join([f'"{k}"' for k in keys if k])
        sql = sql_text("""
            SELECT e.* FROM excel_rows e
            JOIN excel_rows_fts f ON f.id = e.id
            WHERE f.normalized_text MATCH :match
            ORDER BY rank
            LIMIT :limit
        """)
        with engine.connect() as conn:
            res = conn.execute(sql, {"match": match_query, "limit": limit})
            rows = res.mappings().all()

        if not rows:
            return "", []

        # Dynamically build a Markdown table from the JSON data
        all_headers = set(['Dòng', 'Sheet'])
        parsed_rows = []
        for r in rows:
            row_data = json.loads(r.get('data_json') or '{}')
            row_data['Dòng'] = r.get('row_index')
            row_data['Sheet'] = r.get('sheet_name')
            all_headers.update(row_data.keys())
            parsed_rows.append(row_data)

        # Select a reasonable number of columns to display
        display_headers = sorted(list(all_headers), key=lambda h: (h not in ['Dòng', 'Sheet'], h))[:8]

        md_lines = ["| " + " | ".join(map(str, display_headers)) + " |", "| " + " | ".join(["---"] * len(display_headers)) + " |"]
        sources = []
        for row_data in parsed_rows:
            line_values = [str(row_data.get(h) or "").replace("\n", " ") for h in display_headers]
            md_lines.append("| " + " | ".join(line_values) + " |")
            cite = {"title": f"Excel: {rows[0].get('source_filename')}", "sheet": row_data.get('Sheet'), "row": row_data.get('Dòng')}
            if cite not in sources:
                sources.append(cite)

        source_file = rows[0].get('source_filename') if rows else ''
        ctx = f"**Bảng dữ liệu từ file Excel ({source_file}):**\n" + "\n".join(md_lines)
        return ctx, sources
    except Exception as e:
        print(f"[Chat] Excel FTS search failed: {e}")
        return "", []


def _simple_title_heuristic(user_text: str, ai_text: str = "") -> str:
    base = (user_text or ai_text or "").strip()
    if not base:
        return "Đoạn Chat"
    # Lấy câu đầu
    for sep in ["\n", ".", "?", "!", ";"]:
        if sep in base:
            base = base.split(sep)[0]
            break
    n = _normalize_vi(base)
    # Bỏ các cụm dư thừa thường gặp
    drops = [
        "toi muon ", "minh muon ", "cho minh hoi ", "cho toi hoi ",
        "toi can ", "minh can ", "muon tim hieu ", "tim hieu ",
        "xin chao ", "chao ", "vui long ", "hoi ve ", "ve ",
    ]
    for d in drops:
        if n.startswith(d):
            base = base[len(d):]
            break
    # Cắt gọn tối đa 8-10 từ
    words = base.strip().split()
    short = " ".join(words[:10])
    # Viết hoa chữ cái đầu
    if short:
        short = short[0].upper() + short[1:]
    return short or "Đoạn Chat"


def _suggest_chat_title(user_text: str, ai_text: str = "") -> str:
    # Ưu tiên heuristic – nhanh, không phụ thuộc model
    title = _simple_title_heuristic(user_text, ai_text)
    try:
        if USE_OPENAI and openai_client:
            prompt = (
                "Hãy đặt một tiêu đề tiếng Việt hay, ngắn gọn (tối đa 8 từ) cho đoạn chat dưới đây. "
                "Không dùng dấu chấm câu, không mở đầu bằng 'Hỏi về'. Chỉ trả về tiêu đề.\n\n"
                f"User: {user_text}\nAssistant: {ai_text[:200]}"
            )
            res = openai_client.chat.completions.create(
                model=_cfg_chat_model(OPENAI_CHAT_MODEL),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=min(24, _cfg_max_tokens(24)),
            )
            cand = (res.choices[0].message.content or "").strip()
            if 0 < len(cand) <= 60:
                title = cand
    except Exception:
        pass
    # Cuối cùng cắt về 60 ký tự để hiển thị gọn
    return (title or "Đoạn Chat").strip()[:60]

# --- Rolling Summary & Follow-up Suggestions ---

def _build_transcript(messages: list[dict], limit: int = 50) -> str:
    lines: list[str] = []
    for m in messages[-limit:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "User" if role == "user" else ("Assistant" if role == "assistant" else "System")
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _update_session_summary(db: Session, session: ChatSession, history: list[dict]):
    try:
        if not ROLLING_SUMMARY_ENABLED:
            return
        # Only attempt when OpenAI is available
        if not (USE_OPENAI and openai_client):
            return
        transcript = _build_transcript(history, limit=ROLLING_SUMMARY_MAX_MESSAGES)
        if not transcript:
            return
        prompt = (
            "Bạn là trợ lý tóm tắt hội thoại. Hãy tóm tắt ngắn gọn bằng tiếng Việt, 3-5 gạch đầu dòng, "
            "nêu rõ thông tin định lượng (mức tiền, ngày, địa điểm) nếu có, để dùng làm ngữ cảnh cho các lượt hỏi tiếp theo.\n\n"
            f"Hội thoại:\n{transcript}\n\nTóm tắt:"
        )
        res = openai_client.chat.completions.create(
            model=_cfg_chat_model(OPENAI_CHAT_MODEL),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=min(220, _cfg_max_tokens(220)),
        )
        summary = (res.choices[0].message.content or "").strip()
        if summary:
            # Cap to ~1k chars to avoid bloat
            session.summary = summary[:1000]
            session.updated_at = datetime.now(timezone.utc)
            db.add(session)
    except Exception as e:
        print(f"[Chat] Rolling summary failed: {e}")


def _suggest_followups(user_msg: str, ai_text: str) -> list[str]:
    try:
        if not (USE_OPENAI and openai_client):
            return []
        prompt = (
            "Dựa trên câu hỏi và câu trả lời sau, hãy đề xuất 2-3 câu hỏi tiếp theo hữu ích. "
            "Mỗi câu tối đa 12 từ, tiếng Việt, không đánh số. Trả về JSON dạng {\"suggestions\": [\"...\", \"...\"]}.\n\n"
            f"Câu hỏi: {user_msg}\nCâu trả lời: {ai_text}\nKết quả:"
        )
        res = openai_client.chat.completions.create(
            model=_cfg_chat_model(OPENAI_CHAT_MODEL),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=min(120, _cfg_max_tokens(120)),
            response_format={"type": "json_object"},
        )
        import json as _json
        data = _json.loads(res.choices[0].message.content or "{}")
        suggestions = data.get("suggestions") or []
        # Keep 3 concise items
        return [s.strip() for s in suggestions if isinstance(s, str) and s.strip()][:3]
    except Exception as e:
        print(f"[Chat] Suggest follow-ups failed: {e}")
        return []



import time

def _rerank_and_prune_context(query: str, context: str, sources: list[dict]) -> tuple[str, list[dict]]:
    """Uses a CrossEncoder reranker to prune and re-order the context and sources based on relevance."""
    if not context or not sources or not rerank_model:
        print("[Chat] Reranker not available or context empty, skipping.")
        return context, sources

    try:
        # Rerank the sources based on the query
        sentence_pairs = [(query, s.get("content", "")) for s in sources]
        scores = rerank_model.predict(sentence_pairs)

        # Combine sources with their scores and sort
        scored_sources = sorted(zip(scores, sources), key=lambda x: x[0], reverse=True)

        # Filter and select the top N sources (e.g., top 5 with score > 0.05)
        # This threshold helps remove documents that are clearly irrelevant.
        top_sources = [source for score, source in scored_sources if score > 0.05][:5]

        if not top_sources:
            return "", []

        # Reconstruct the context from the top sources
        pruned_context = "\n\n".join([s.get("content", "") for s in top_sources])
        print(f"[Chat] Reranked and pruned context. Kept {len(top_sources)}/{len(sources)} sources.")
        return pruned_context, top_sources
    except Exception as e:
        print(f"[Chat] Reranking failed: {e}")
        return context, sources

def _execute_table_calculation(table_md: str, operation: str, column: str) -> str:
    """
    Parses a Markdown table and performs a simple calculation on a specified column.
    Returns the result as a formatted string.
    """
    try:
        # Convert Markdown table to a DataFrame
        lines = [line.strip() for line in table_md.strip().split('\n')]
        if len(lines) < 2: return "Lỗi: Bảng không hợp lệ."

        # Clean lines by removing leading/trailing pipes
        cleaned_lines = [re.sub(r'^\s*\|\s*|\s*\|\s*$', '', line) for line in lines]
        header_line = cleaned_lines[0]
        separator_line = cleaned_lines[1]
        data_lines = cleaned_lines[2:]

        # Check for separator line validity
        if not re.match(r'^[\|\s:-]+$', separator_line):
            return "Lỗi: Dòng phân tách của bảng Markdown không hợp lệ."

        # Use StringIO to simulate reading a CSV, splitting by pipe
        csv_data = "\n".join([header_line] + data_lines)
        df = pd.read_csv(StringIO(csv_data), sep='|', skipinitialspace=True)

        # Clean up column names and data
        df.columns = [col.strip() for col in df.columns]
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        # Find the target column, allowing for partial matches
        target_col = None
        for col_name in df.columns:
            if column.strip().lower() in col_name.lower():
                target_col = col_name
                break
        if not target_col:
            return f"Lỗi: Không tìm thấy cột '{column}'. Các cột có sẵn: {', '.join(df.columns)}"

        # Convert column to numeric, coercing errors
        numeric_series = pd.to_numeric(df[target_col], errors='coerce').dropna()
        if numeric_series.empty:
            return f"Lỗi: Cột '{target_col}' không chứa dữ liệu số hợp lệ."

        # Perform operation
        result = None
        op = operation.upper()
        if op == "SUM":
            result = numeric_series.sum()
        elif op == "AVG":
            result = numeric_series.mean()
        elif op == "MAX":
            result = numeric_series.max()
        elif op == "MIN":
            result = numeric_series.min()
        else:
            return f"Lỗi: Phép tính '{operation}' không được hỗ trợ."

        return f"{result:,.2f}" # Format with commas and 2 decimal places

    except Exception as e:
        print(f"[TableCalc] Error executing calculation: {e}")
        return f"Lỗi khi xử lý bảng: {e}"

def get_ai_response(
    message: str,
    context: str = "",
    history: list[dict] | None = None,
    intent: str = "SIMPLE_QA"
) -> tuple[str, int, int]:
    """Return (response_text, tokens_used, response_time_ms)."""
    start = time.time()

    def _looks_like_ack(msg: str) -> bool:
        m = (msg or "").strip().lower()
        return m in {"ok", "oke", "okay", "dạ", "vâng", "uhm", "ừ", "ờ", "được", "okie", "okiee", "yes"}

    def _looks_like_too_generic(msg: str) -> bool:
        m = (msg or "").strip().lower()
        if len(m) <= 2: return True
        generic_patterns = ["đây là gì", "cái này là gì", "là gì", "what is this", "this is?"]
        return any(p in m for p in generic_patterns)

    def _wants_brief(msg: str) -> bool:
        m = (msg or "").strip().lower()
        brief_phrases = ["nói ngắn gọn", "ngắn gọn", "tóm gọn", "tóm tắt", "summary", "brief"]
        return any(p in m for p in brief_phrases)

    if (not context or not context.strip()) and (_looks_like_ack(message) or _looks_like_too_generic(message)):
        return ("Xin lỗi, tôi không chắc về thông tin này", 0, int((time.time() - start) * 1000))

    try:
        parsed = parse_allowance_table(context)
        exact = try_answer_allowance_question(message, parsed)
        if exact:
            return exact, 0, int((time.time() - start) * 1000)
    except Exception as _pe:
        dlog(f"[Chat] allowance parser skipped: {_pe}")

    try:
        if USE_OPENAI and openai_client:
            brief = _wants_brief(message)
            # Dynamic System Prompt selection logic here...
            # Chính sách chung của DalatHasfarm (base policy)
            base_policy = (
                "Bạn là trợ lý AI nội bộ của DalatHasfarm, được xây dựng để hỗ trợ nhân viên và quản lý tra cứu dữ liệu, truy cập thông tin nội bộ, và giải đáp chính xác các vấn đề nghiệp vụ, nhằm nâng cao hiệu quả công việc và tối ưu hóa quy trình vận hành.\n\n"
                "**QUY TẮC BẮT BUỘC:**\n\n"
                "1. **NGUỒN DỮ LIỆU DUY NHẤT:** Chỉ sử dụng thông tin từ cơ sở dữ liệu nội bộ DalatHasfarm (bao gồm: tài liệu PDF, file Excel, báo cáo, chính sách, quy định, hướng dẫn quy trình đã được tải lên hệ thống).\n\n"
                "2. **XÁC THỰC TRƯỚC KHI TRẢ LỜI:**\n"
                "   - Nếu tìm thấy thông tin: Mở đầu bằng MỘT trong các mẫu sau (chọn 1, phù hợp ngữ cảnh):\n"
                "     • \"Dựa trên tài liệu nội bộ của DalatHasfarm, ...\"\n"
                "     • \"Theo tài liệu nội bộ DalatHasfarm, ...\"\n"
                "     • \"Căn cứ vào tài liệu nội bộ đã cung cấp, ...\"\n"
                "     • \"Từ các tài liệu nội bộ hiện có, ...\"\n"
                "     • \"Dựa trên dữ liệu nội bộ (Excel/PDF) của DalatHasfarm, ...\"\n"
                "     • \"Theo dữ liệu trong file PDF/Excel của DalatHasfarm, ...\"\n"
                "     • \"Tôi muốn kiểm tra thông tin, ...\"\n"
                "     • \"Kết quả trả lời dựa trên thông tin nội bộ đã tải lên, ...\"\n"
                "     • \"Căn cứ vào dữ liệu quản trị nội bộ của DalatHasfarm, ...\"\n"
                "     • \"Theo hồ sơ nội bộ DalatHasfarm, ...\"\n"
                "     • \"Căn cứ trên dữ liệu quản lý nội bộ, ...\"\n"
                "     • \"Cho tôi thông tin, ...\"\n"
                "   - Nếu là câu hỏi tiếp theo (follow‑up) trong cùng chủ đề và cùng nguồn ngữ cảnh (ví dụ: người dùng dùng \"tiếp theo\", \"vậy còn\", \"còn ở\", \"thế còn\"; hoặc tham chiếu kết quả trước):\n"
                "     • Có thể lược bỏ câu mở đầu dài, trả lời ngắn gọn trực tiếp nhưng VẪN trích dẫn nguồn ở cuối.\n"
                "   - Nếu không tìm thấy: Trả lời chính xác \"Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu nội bộ của DalatHasfarm. Vui lòng liên hệ bộ phận liên quan để được hỗ trợ trực tiếp.\"\n\n"
                "3. **CẤM TUYỆT ĐỐI:**\n"
                "   - Không bịa đặt, suy đoán, hoặc tạo ra thông tin không có trong dữ liệu\n"
                "   - Không sử dụng kiến thức bên ngoài về DalatHasfarm từ internet.\n"
                "   - Không đưa ra khuyến nghị cá nhân vượt quá phạm vi tài liệu.\n"
                "   - Không tự động tổng quát hóa thông tin khi dữ liệu chỉ áp dụng cho một trường hợp cụ thể.\n"
                "   - Không đưa ra khuyến nghị cá nhân vượt quá phạm vi tài liệu.\n\n"
                "4. **ĐỊNH DẠNG TRẢ LỜI:**\n"
                "   - Trả lời ngắn gọn, súc tích, đúng trọng tâm\n"
                "   - Sử dụng ngôn ngữ chuyên nghiệp nhưng thân thiện\n"
                "   - Nếu dữ liệu là danh sách, dùng bullet point để trình bày rõ ràng"
                "   - Khi trích xuất từ bảng dữ liệu, giữ nguyên định dạng bảng Markdown\n\n"
                "5. **XỬ LÝ CÂU HỎI PHỨC TẠP:**\n"
                "   - Nếu câu hỏi yêu cầu thông tin từ nhiều tài liệu, tổng hợp một cách logic và rõ ràng\n"
                "   - Nếu có mâu thuẫn giữa các tài liệu, nêu rõ và ưu tiên tài liệu mới nhất\n"
                "   - Đối với số liệu, luôn trích xuất chính xác từ nguồn gốc\n\n"
                "6. **PHẠM VI HỖ TRỢ:** Chỉ trả lời các câu hỏi liên quan đến hoạt động nội bộ DalatHasfarm như: chính sách nhân sự, quy định công ty, quy trình làm việc, thông tin sản phẩm, báo cáo tài chính, phúc lợi nhân viên, cơ cấu tổ chức."
            )

            if intent == "COMPARISON":
                sys_prompt = base_policy + (
                    "\n\nHƯỚNG DẪN RIÊNG (SO SÁNH/ĐỐI CHIẾU):\n"
                    "- Trình bày có cấu trúc, làm rõ điểm giống/khác; ưu tiên bảng Markdown khi phù hợp.\n"
                    "- Chỉ dùng thông tin trong tài liệu nội bộ; nếu có mâu thuẫn, nêu rõ và ưu tiên tài liệu mới nhất.\n"
                )
            elif intent == "SYNTHESIS":
                sys_prompt = base_policy + (
                    "\n\nHƯỚNG DẪN RIÊNG (TỔNG HỢP):\n"
                    "- Kết nối thông tin từ nhiều tài liệu một cách logic, rõ ràng, mạch lạc.\n"
                    "- Ưu tiên dữ liệu bảng; nếu có mâu thuẫn, nêu rõ và ưu tiên tài liệu mới nhất.\n"
                )
            else:  # Default to SIMPLE_QA
                sys_prompt = base_policy + (
                    "\n\nLƯU Ý THỰC THI: Khi người dùng yêu cầu tính toán trên bảng (tổng, trung bình, lớn nhất, nhỏ nhất), hãy tính toán chính xác dựa trên dữ liệu bảng.\n"
                    "Nếu cần gọi công cụ tính toán bảng, trả về một JSON object duy nhất theo định dạng sau (không trả lời trực tiếp):\n"
                    '```json\n{"tool_name": "table_calculator", "table": "<toàn bộ bảng dạng Markdown>", "operation": "<SUM|AVG|MAX|MIN>", "column": "<tên cột>"}\n```'
                )

            user_prompt = f"Thông tin từ tài liệu:\n{context}\n\nCâu hỏi: {message}" if context.strip() else message
            msgs = [{"role": "system", "content": sys_prompt}]
            if history: msgs.extend(history[-HISTORY_MAX_HISTORY_MESSAGES:])
            msgs.append({"role": "user", "content": user_prompt})

            resp = openai_client.chat.completions.create(
                model=_cfg_chat_model(OPENAI_CHAT_MODEL), messages=msgs, temperature=0.2 if brief else 0.4, max_tokens=_cfg_max_tokens(800))
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, 'usage', None)
            tokens = usage.total_tokens if usage and hasattr(usage, 'total_tokens') else 0

            tool_call_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if tool_call_match:
                try:
                    tool_data = json.loads(tool_call_match.group(1))
                    if tool_data.get("tool_name") == "table_calculator":
                        print(f"[Chat] Received tool call: {tool_data}")
                        calc_result = _execute_table_calculation(
                            tool_data.get("table", ""), tool_data.get("operation", ""), tool_data.get("column", ""))
                        print(f"[Chat] Tool execution result: {calc_result}")

                        synthesis_prompt = f"Dựa trên câu hỏi gốc và kết quả tính toán, hãy đưa ra câu trả lời tự nhiên.\nCâu hỏi: '{message}'\nKết quả: '{calc_result}'\nCâu trả lời:"

                        final_resp = openai_client.chat.completions.create(
                            model=OPENAI_CHAT_MODEL, messages=[{"role": "user", "content": synthesis_prompt}], temperature=0.1, max_tokens=200)
                        final_text = (final_resp.choices[0].message.content or "").strip()

                        final_usage = getattr(final_resp, 'usage', None)
                        final_tokens = final_usage.total_tokens if final_usage and hasattr(final_usage, 'total_tokens') else 0
                        total_tokens = tokens + final_tokens

                        return final_text, total_tokens, int((time.time() - start) * 1000)
                except Exception as e:
                    print(f"[Chat] Tool call processing failed: {e}")
                    return f"Lỗi khi xử lý yêu cầu công cụ: {e}", tokens, int((time.time() - start) * 1000)

            return text, tokens, int((time.time() - start) * 1000)

        # Fallback for other models...
        if 'llm' in globals() and llm:
            prompt = f"Context: {context}\n\nQuestion: {message}\n\nAnswer:" if context else f"Question: {message}\n\nAnswer:"
            text = (llm.invoke(prompt) or "").strip()
            return text, 0, int((time.time() - start) * 1000)

        return ("Xin lỗi, mình chưa có đủ ngữ cảnh đáng tin cậy để trả lời câu hỏi này.", 0, int((time.time() - start) * 1000))
    except Exception as e:
        return (f"Error generating response: {str(e)}", 0, int((time.time() - start) * 1000))

# ---- Lightweight structured extractors for admin tables (e.g., phụ cấp) ----
import re

def _to_int_vn(num_str: str) -> int | None:
    try:
        digits = re.sub(r"[^0-9]", "", num_str or "")
        return int(digits) if digits else None
    except Exception:
        return None



_all_region_keywords = ["hà nội", "hcm", "tp.hcm", "tp hcm", "hồ chí minh", "cần thơ", "đà nẵng", "nha trang", "ban mê thuột", "ban me thuot", "qui nhơn", "qui nhon", "đà lạt", "da lat"]


def _format_vnd(n: int | None) -> str:
    if n is None:
        return ""
    s = f"{n:,}".replace(",", ".")
    return s


def parse_allowance_table(text: str) -> dict:
    """Parse admin notice like Dalat Hasfarm allowance table.
    Returns {'groups': [{'regions':[...], 'current':int, 'new':int, 'delta':int}], 'effective_date': 'dd/mm/yyyy'} or {}.
    """
    if not text:
        return {}
    low = text.lower()
    # Effective date
    eff = None
    m = re.search(r"hiệu lực\s*từ\s*ngày\s*(\d{1,2}/\d{1,2}/\d{4})", low)
    if m:
        eff = m.group(1)
    # Find main row: Hà Nội, HCM, Cần Thơ ... numbers
    groups = []
    # Capture a line containing 2-3 cities then 3 numbers
    m1 = re.search(r"(hà nội[^\n]*hcm[^\n]*cần thơ[^\n]*)(\n|\r| )+([0-9\.\s,]+)(\n|\r| )+([0-9\.\s,]+)(\n|\r| )+([0-9\.\s,]+)", low)
    if m1:
        cur = _to_int_vn(m1.group(3))
        new = _to_int_vn(m1.group(5))
        delta = _to_int_vn(m1.group(7))
        groups.append({
            "regions": ["Hà Nội", "HCM", "Cần Thơ"],
            "current": cur,
            "new": new,
            "delta": delta,
        })
    # Capture central-highland block numbers
    m2 = re.search(r"(miền\s*trung[^\n]*cao\s*nguyên[^\n]*)(?:.|\n|\r)+?([0-9\.\s,]+)\s+([0-9\.\s,]+)\s+([0-9\.\s,]+)", low)
    if m2:
        cur = _to_int_vn(m2.group(2))
        new = _to_int_vn(m2.group(3))
        delta = _to_int_vn(m2.group(4))
        # Extract bullet list regions in this block
        regions = []
        for line in low.splitlines():
            if "➤" in line or "-" in line:
                for kw in ["đà nẵng", "nha trang", "ban mê thuột", "ban me thuot", "qui nhơn", "qui nhon", "đà lạt", "da lat"]:
                    if kw in line and kw.title() not in regions:
                        regions.append(kw.title())
        if not regions:
            regions = ["Đà Nẵng", "Nha Trang", "Ban Mê Thuột", "Qui Nhơn", "Đà Lạt", "…"]
        groups.append({
            "regions": regions,
            "current": cur,
            "new": new,
            "delta": delta,
        })
    return {"groups": groups, "effective_date": eff} if groups else {}


def try_answer_allowance_question(question: str, parsed: dict) -> str | None:
    q = (question or "").lower()
    if not parsed or not parsed.get("groups"):
        return None
    if "phụ cấp" not in q and "phu cap" not in q:
        return None
    eff = parsed.get("effective_date")
    # Flatten mapping region -> group
    region_map: dict[str, dict] = {}
    for g in parsed["groups"]:
        for r in g.get("regions", []):
            region_map[(r or "").lower()] = g
        # explicit aliases: only map if this group represents HCM/Hồ Chí Minh
        if any(("hcm" in (r or "").lower()) or ("hồ chí minh" in (r or "").lower()) for r in g.get("regions", [])):
            for k in ["hcm", "tp.hcm", "tp hcm", "hồ chí minh", "ho chi minh"]:
                region_map[k] = g
    # Find target region in question
    target = None
    for key in region_map.keys():
        if key and key in q:
            target = key
            break
    if target:
        g = region_map[target]
        cur, new, delta = g.get("current"), g.get("new"), g.get("delta")
        region_name = target.title().replace("Tp.Hcm", "TP.HCM")
        core = f"{_format_vnd(new)} đồng/tháng (tăng {_format_vnd(delta)} so với {_format_vnd(cur)})."
        return (f"Mức phụ cấp ăn trưa tại {region_name} sau điều chỉnh là {core}" +
                (f" Hiệu lực từ {eff}." if eff else ""))
    # If ask to list or generic
    if any(k in q for k in ["chi tiết", "kể chi tiết", "các khu vực", "từng khu vực", "bao nhiêu", "mức"]):
        lines = []
        for g in parsed["groups"]:
            area = ", ".join(g.get("regions", [])[:6])
            lines.append(f"- {area}: {_format_vnd(g.get('current'))} → {_format_vnd(g.get('new'))} (tăng {_format_vnd(g.get('delta'))})")
        suffix = f"Hiệu lực từ {eff}." if eff else ""
        return ("Mức phụ cấp ăn trưa được điều chỉnh như sau:\n" + "\n".join(lines) + ("\n" + suffix if suffix else ""))
    return None


def _encode_query(text: str):
    # Ưu tiên sử dụng OpenAI embeddings (tốt nhất cho tiếng Việt)
    if USE_OPENAI and openai_client:
        try:
            resp = openai_client.embeddings.create(model=_cfg_embed_model(OPENAI_EMBED_MODEL), input=text)
            return resp.data[0].embedding
        except Exception as e:
            print(f"[Chat] OpenAI embedding failed: {e}")

    # Fallback to Gemini embeddings
    if GEMINI_AVAILABLE:
        try:
            gemini_service = get_gemini_service()
            embeddings_list = gemini_service.generate_embeddings([text])
            if embeddings_list:
                return embeddings_list[0]
        except Exception as e:
            print(f"[Chat] Gemini embedding failed: {e}")

    # Fallback to Ollama
    if embeddings:
        return embeddings.embed_query(text)

    # Fallback to local sentence-transformers
    if '_st_model' in globals() and _st_model is not None:
        return _st_model.encode(text).tolist()

    raise RuntimeError("No embeddings backend available")


def get_relevant_context(message: str, collection) -> str:
    """Hybrid search (dense + keyword) with robust fallbacks and a relevance gate."""
    try:
        print(f"[Chat] Searching for: '{message}'")
        candidates: list[str] = []

        # 1) Dense vector search
        try:
            query_vec = _encode_query(message)
            print(f"[Chat] Query embedding generated: {len(query_vec)} dimensions")
            dense_results = collection.query(query_embeddings=[query_vec], n_results=8)
            dense_docs = dense_results.get('documents', [[]])[0] if dense_results else []
            candidates.extend([d for d in dense_docs if d])
            dlog(f"[Chat] Dense search found {len(dense_docs)} documents")
        except Exception as e1:
            dlog(f"[Chat] Dense search skipped: {e1}")

        # 2) Sparse keyword via query_texts (requires embedding function on collection)
        keywords = [w for w in message.split() if len(w) > 2][:5]
        try:
            for kw in keywords:
                r = collection.query(query_texts=[kw], n_results=3)
                if r.get('documents') and r['documents'][0]:
                    for d in r['documents'][0]:
                        if d and d not in candidates:
                            candidates.append(d)
        except Exception as e2:
            print(f"[Chat] query_texts keyword search unavailable: {e2}")

        # 3) Pure keyword fallback using where_document contains (no embeddings)
        if not candidates:
            try:
                toks = [w for w in message.replace('\n', ' ').split() if len(w) > 2][:5]
                for tk in toks:
                    try:
                        res = collection.get(where_document={"$contains": tk}, limit=5)
                        docs = res.get('documents') or []
                        if docs and isinstance(docs[0], list):
                            docs = docs[0]
                        for d in docs:
                            if d and d not in candidates:
                                candidates.append(d)
                    except Exception:
                        break
            except Exception as e3:
                print(f"[Chat] where_document fallback failed: {e3}")

        # 4) Last resort: sample & keyword-overlap ranking
        if not candidates:
            try:
                res = collection.get(limit=50)
                docs = res.get('documents') or []
                if docs and isinstance(docs[0], list):
                    docs = docs[0]
                kws = [w.lower() for w in message.split() if len(w) > 2]
                def score(doc: str) -> int:
                    low = (doc or '').lower()
                    return sum(1 for k in kws if k in low)
                ranked = sorted([d for d in docs if d], key=score, reverse=True)
                candidates.extend(ranked[:5])
            except Exception as e4:
                print(f"[Chat] global sample fallback failed: {e4}")

        # Trim and optionally rerank + relevance gate
        candidates = candidates[:10]
        best_score = None
        if candidates and '_ranker' in globals() and _ranker is not None and RerankRequest is not None:
            try:
                passages_to_rerank = [{"id": i, "text": c} for i, c in enumerate(candidates)]
                rerank_request = RerankRequest(query=message, passages=passages_to_rerank)
                reranked_results = _ranker.rerank(rerank_request)

                scores = [r['score'] for r in reranked_results]
                best_score = float(max(scores)) if scores else None

                topk = [r['text'] for r in reranked_results[:5]]

            except Exception as e4:
                print(f"[Chat] Reranker failed: {e4}")
                topk = candidates[:5]
        else:
            topk = candidates[:5]

        # Relevance threshold (env/db overrideable)
        try:
            thresh = float(_cfg_similarity_threshold(0.45))
        except Exception:
            thresh = 0.45
        if best_score is not None and best_score < thresh:
            print(f"[Chat] Context rejected by relevance gate (score={best_score:.3f} < {thresh})")
            return ""

        final_context = "\n".join(topk)
        print(f"[Chat] Final context length: {len(final_context)} chars")
        if final_context:
            preview = (final_context[:200] + "...") if len(final_context) > 200 else final_context
            print(f"[Chat] Context preview: {preview}")
        return final_context
    except Exception as e:
        print(f"Error querying ChromaDB (hybrid): {e}")
        return ""

# Like get_relevant_context but supports querying across multiple collections.
# Return: only the context text (concatenated top docs).
def _extract_entities_for_filtering(query: str) -> Optional[dict]:
    """Uses an LLM to extract key-value pairs from a query for metadata filtering."""
    if not (USE_OPENAI and openai_client):
        return None
    try:
        prompt = f'''
From the user's question, extract key entities that can be used for filtering data.
Return a JSON object where keys are potential column names (in snake_case) and values are the extracted values.
Only extract specific, concrete values (like names, codes, numbers). Ignore vague requests.

Example 1:
Question: "thông tin sản phẩm Rocca painted Mono có mã F01016PA"
Result: {{"ten_san_pham": "Rocca painted Mono", "ma_san_pham": "F01016PA"}}

Example 2:
Question: "giá của Laptop Pro là bao nhiêu"
Result: {{"ten_san_pham": "Laptop Pro"}}

Example 3:
Question: "sản phẩm nào có giá 7600"
Result: {{"gia": 7600}}

Example 4:
Question: "liệt kê các máy tính"
Result: {{}}

Example 5:
Question: "thông tin về sản phẩm Tulip Lima"
Result: {{"ten_san_pham": "Tulip Lima"}}

User Question: "{query}"
Result:
'''
        res = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL, # A fast model is fine
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        json_str = (res.choices[0].message.content or "{}").strip()
        entities = json.loads(json_str)
        # Basic validation: ensure values are not empty strings or None
        validated_entities = {k: v for k, v in entities.items() if v is not None and str(v).strip() != ''}
        if validated_entities:
            print(f"[Chat] Extracted entities for filtering: {validated_entities}")
            return validated_entities
        return None
    except Exception as e:
        print(f"[Chat] Entity extraction for filtering failed: {e}")
        return None


def _validate_context_relevance(query: str, context: str) -> bool:
    """Uses an LLM to validate if the context is truly relevant to the user's query."""
    if not (USE_OPENAI and openai_client) or not context.strip():
        return True # Default to true if validation is not possible
    try:
        prompt = f'''
You are a strict data validation assistant. Your task is to determine if the provided CONTEXT document is *directly and specifically* about the main subject of the USER QUESTION.

The CONTEXT must contain substantial details about the subject. A passing mention is not enough.

For example, if the question is about "Product X", the context must be the specifications, description, or data sheet for "Product X". A document that only mentions "Product X" in a list of company assets is NOT a valid context.

Does the provided CONTEXT contain a direct and specific answer to the USER QUESTION?
Respond with only "YES" or "NO".

USER QUESTION: "{query}"

CONTEXT:
---
{context[:2000]}
---

ANSWER (YES or NO):
'''
        res = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL, # A fast model is fine
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        decision = (res.choices[0].message.content or "").strip().upper()
        print(f'[Chat] Context validation for "{query[:30]}..." returned: {decision}')
        return "YES" in decision
    except Exception as e:
        print(f"[Chat] Context validation failed: {e}")
        return True # Fail open to avoid accidentally blocking good results

def get_context_with_sources(queries: list[str], collections: list[Any]) -> tuple[str, list[dict]]:
    """Query ChromaDB with a two-stage (filtered -> broad) search, then rerank."""
    if not queries or not collections:
        return "", []

    original_query = queries[0]
    all_results: list[tuple[float, str, dict]] = []

    try:
        # 1. Generate embeddings for the queries
        if USE_OPENAI and openai_client:
            res = openai_client.embeddings.create(model=_cfg_embed_model(OPENAI_EMBED_MODEL), input=queries)
            query_embeddings = [r.embedding for r in res.data]
        elif _st_model:
            query_embeddings = [e.tolist() for e in _st_model.encode(queries)]
        else:
            return "", []

        # 2. Stage 1: Filtered Search
        where_filter = _extract_entities_for_filtering(original_query)

        # --- Start: Source Prioritization Logic ---
        search_collections = collections
        # Expand product-related keys to handle various Excel headers
        PRODUCT_METADATA_KEYS = {
            "ten_san_pham", "ma_san_pham", "gia", "san_pham", "ten", "sku", "ma", "ma_hang", "ma_hang_hoa",
            "product_name", "product", "item_code"
        }
        is_product_query = bool(where_filter) and any(k in where_filter for k in PRODUCT_METADATA_KEYS)

        if is_product_query:
            excel_collections = [c for c in collections if getattr(c, 'name', '').startswith('excel_data_')]
            if excel_collections:
                print(f"[Chat] Product query detected. Prioritizing {len(excel_collections)} Excel collections.")
                search_collections = excel_collections
            else:
                print("[Chat] Product query detected, but no Excel collections found. Searching all collections.")
        # --- End: Source Prioritization Logic ---

        def _normalize_key(s: str) -> str:
            import re, unicodedata
            try:
                s2 = unicodedata.normalize('NFKD', s or '')
                s2 = ''.join([c for c in s2 if not unicodedata.combining(c)])
            except Exception:
                s2 = s or ''
            s2 = s2.lower()
            s2 = re.sub(r"[^a-z0-9_]+", "_", s2)
            return s2.strip("_")

        def _map_filter_keys_to_collection(where_dict: dict, collection) -> dict:
            """Map filter keys to the closest metadata keys present in this collection."""
            try:
                sample = collection.get(limit=1)
                metas = (sample or {}).get('metadatas', []) or []
                sample_keys = set()
                if metas:
                    for k in metas[0].keys():
                        sample_keys.add(k)
                if not sample_keys:
                    return where_dict
                norm_map = { _normalize_key(k): k for k in sample_keys }
                mapped = {}
                for k, v in (where_dict or {}).items():
                    nk = _normalize_key(k)
                    target = norm_map.get(nk)
                    if not target:
                        # fallback: try any sample key that contains tokens of nk
                        for sk_norm, orig in norm_map.items():
                            if nk in sk_norm or sk_norm in nk:
                                target = orig; break
                    mapped[target or k] = v
                if mapped != where_dict:
                    print(f"[Chat] Mapped filter keys for collection '{getattr(collection,'name','?')}': {where_dict} -> {mapped}")
                return mapped
            except Exception as _e:
                print(f"[Chat] Could not map filter keys for collection {getattr(collection,'name','?')}: {_e}")
                return where_dict

        if where_filter:
            print(f"[Chat] Stage 1: Performing filtered search with (raw): {where_filter}")
            for collection in search_collections:
                try:
                    mapped_where = _map_filter_keys_to_collection(where_filter, collection)
                    results = collection.query(
                        query_embeddings=query_embeddings,
                        n_results=10,
                        where=mapped_where,
                        include=["metadatas", "documents", "distances"]
                    )
                    print(f"[Chat] Stage1 results from '{getattr(collection,'name','?')}': {len(results.get('ids', [[]])[0])} hits")
                    for i in range(len(results.get("ids", []))):
                        for j in range(len(results["ids"][i])):
                            dist = results["distances"][i][j]
                            doc = results["documents"][i][j]
                            meta = results["metadatas"][i][j]
                            _prev = (doc or "")[:160]
                            try:
                                _prev = _prev.replace("\n", " ")
                            except Exception:
                                pass
                            print(f"    • score={1.0 - dist:.3f} doc[:160]=" + _prev + f" | meta={meta}")
                            all_results.append((1.0 - dist, doc, meta))
                except Exception as e:
                    print(f"Error in filtered search on {getattr(collection, 'name', '?')}: {e}")

        # 3. Stage 2: Broad Search (if filtered search yields no results)
        if not all_results:
            if where_filter:
                print("[Chat] Stage 1 yielded no results. Proceeding to Stage 2: Broad Search.")
            else:
                print("[Chat] No entities found for filtering. Proceeding directly to Stage 2: Broad Search.")

            for collection in collections:
                try:
                    results = collection.query(
                        query_embeddings=query_embeddings,
                        n_results=10,
                        include=["metadatas", "documents", "distances"]
                    )
                    print(f"[Chat] Stage2 results from '{getattr(collection,'name','?')}': {len(results.get('ids', [[]])[0])} hits")
                    for i in range(len(results.get("ids", []))):
                        for j in range(len(results["ids"][i])):
                            dist = results["distances"][i][j]
                            doc = results["documents"][i][j]
                            meta = results["metadatas"][i][j]
                            _prev2 = (doc or "")[:160]
                            try:
                                _prev2 = _prev2.replace("\n", " ")
                            except Exception:
                                pass
                            print(f"    • score={1.0 - dist:.3f} doc[:160]=" + _prev2 + f" | meta={meta}")
                            all_results.append((1.0 - dist, doc, meta))
                except Exception as e:
                    print(f"Error in broad search on {getattr(collection, 'name', '?')}: {e}")

        if not all_results:
            # Extra fallback: keyword 'contains' search for product-like queries on Excel collections
            try:
                if is_product_query and where_filter:
                    for val in [str(v) for v in where_filter.values() if v]:
                        for collection in search_collections:
                            try:
                                kw = str(val).strip()
                                if not kw: continue
                                res = collection.get(where_document={"$contains": kw}, limit=7)
                                docs = res.get("documents", []) or []
                                metas = res.get("metadatas", []) or []
                                ids = res.get("ids", []) or []
                                print(f"[Chat] Fallback where_document contains '{kw}' -> {len(docs)} hits in {getattr(collection,'name','?')}")
                                for i in range(min(len(docs), 7)):
                                    doc = docs[i]; meta = metas[i] if i < len(metas) else {}
                                    all_results.append((0.51, doc, meta))  # neutral score
                            except Exception as _wd:
                                print(f"[Chat] where_document fallback failed on {getattr(collection,'name','?')}: {_wd}")
            except Exception as _fb:
                print(f"[Chat] Fallback keyword search error: {_fb}")

        if not all_results:
            return "", []

        # 4. Consolidate and Rank Results
        all_results.sort(key=lambda x: x[0], reverse=True)
        unique_docs = {doc: meta for _, doc, meta in all_results} # Deduplicate
        candidates = list(unique_docs.items())[:15]

        # 5. Rerank and Prune
        source_candidates = [{
            "content": doc,
            **(meta if meta else {})
        } for doc, meta in candidates]

        pruned_context, final_sources = _rerank_and_prune_context(original_query, "", source_candidates)

        # 6. Final validation
        if not _validate_context_relevance(original_query, pruned_context):
            print(f"[Chat] Context rejected by final validation for query: '{original_query}'")
            return "", []

        # Log the final context before returning
        try:
            print(f"[Chat] Final context preview (first 500 chars): {(pruned_context[:500] if pruned_context else '')}")
            print(f"[Chat] Final sources sample: {final_sources[:2]}")
        except Exception as e_log:
            print(f"[Chat] Final logging failed: {e_log}")

        return pruned_context, final_sources
    except Exception as e:
        print(f"Error querying ChromaDB (with sources): {e}")
        import traceback
        traceback.print_exc()
        return "", []

async def _sequential_retrieval(original_message: str, sub_questions: list[str], collections) -> tuple[str, list[dict]]:
    """
    Performs sequential retrieval where the context from one step informs the next.
    """
    print(f"[Chat] Performing sequential retrieval for: {sub_questions}")
    final_context_docs: dict[str, dict] = {} # Use dict to auto-handle duplicates {doc: meta}
    previous_step_summary = ""

    for i, question in enumerate(sub_questions):
        # Augment the sub-question with the summary of the previous step's findings
        if previous_step_summary:
            current_query = f"{question} (liên quan đến: {previous_step_summary})"
        else:
            current_query = question

        print(f"[Chat] Sequential step {i+1}: Searching for '{current_query}'")
        context, sources = get_context_with_sources([current_query], collections)

        if context:
            # Update the combined context
            context_docs = context.split('\n')
            for doc_idx, doc in enumerate(context_docs):
                if doc and doc not in final_context_docs:
                    meta = sources[doc_idx] if doc_idx < len(sources) else {}
                    final_context_docs[doc] = meta

            # Create a summary of the newly found context to inform the next step
            if USE_OPENAI and openai_client:
                try:
                    summary_prompt = f"Dựa vào thông tin sau, hãy tóm tắt ý chính trong một câu ngắn gọn để làm ngữ cảnh cho bước tìm kiếm tiếp theo. Ngữ cảnh: '{context[:1000]}'"
                    # This is a synchronous call, so no 'await' is needed.
                    res = openai_client.chat.completions.create(
                        model=OPENAI_CHAT_MODEL,
                        messages=[{"role": "user", "content": summary_prompt}],
                        temperature=0.0,
                        max_tokens=100,
                    )
                    previous_step_summary = (res.choices[0].message.content or "").strip()
                    print(f"[Chat] Sequential step {i+1} summary: {previous_step_summary}")
                except Exception as e:
                    print(f"[Chat] Failed to summarize context for sequential retrieval: {e}")
                    previous_step_summary = context[:150] # Fallback to simple truncation
            else:
                # Fallback if OpenAI is not available
                previous_step_summary = context[:150]

    # Final reranking of all collected documents against the original message
    all_candidates = list(final_context_docs.items())
    source_candidates = [{
        "content": doc,
        **(meta if meta else {})
    } for doc, meta in all_candidates]

    # Rerank and prune the combined results from all sequential steps
    pruned_context, final_sources = _rerank_and_prune_context(original_message, "", source_candidates)

    return pruned_context, final_sources

# Build an augmented query using recent user history.
# We will use this ONLY as a fallback when current-question retrieval is weak.
def build_augmented_query(current_message: str, history: list[dict], min_user_msgs: int = 3) -> str:
    def _is_generic(msg: str) -> bool:
        m = (msg or "").strip().lower()
        if len(m) <= 6:
            return True
        generic_phrases = [
            "nói ngắn gọn", "ngắn gọn", "tóm gọn", "tóm tắt", "nói tiếp", "tiếp theo", "tiếp tục",
            "ok", "oke", "vậy thôi", "đúng rồi", "chuẩn", "tiếp đi", "nữa"
        ]
        return any(p in m for p in generic_phrases)

    # Collect last user turns (keep order)
    recent_user_msgs: list[str] = []
    for h in history or []:
        if h.get("role") == "user":
            c = (h.get("content") or "").strip()
            if c:
                recent_user_msgs.append(c)
    recent_user_msgs = recent_user_msgs[-min_user_msgs:]

    if _is_generic(current_message):
        base = " ".join(recent_user_msgs)
    else:
        base = (current_message or "").strip()
        if recent_user_msgs:
            base = base + " \n" + " \n".join(recent_user_msgs)
    return base.strip() or (current_message or "")

# Return (primary_query, secondary_query). Primary = current message; secondary = augmented or None if generic current.
def build_dual_queries(current_message: str, history: list[dict], min_user_msgs: int = 3) -> tuple[str, str | None]:
    augmented = build_augmented_query(current_message, history, min_user_msgs)
    # If the current message is generic (e.g., "giá cụ thể?"), use the augmented as primary.
    cm = (current_message or "").strip()
    if len(cm) <= 2:
        return augmented, None
    low = cm.lower()
    generic_markers = ["nói ngắn gọn", "ngắn gọn", "tóm", "tiếp", "nữa", "cụ thể", "giá cụ thể", "sao", "bao nhiêu?"]
    if any(m in low for m in generic_markers):
        return augmented, None
    # Otherwise, prioritize the exact current question and keep augmented as fallback
    return current_message, (augmented if augmented != current_message else None)




def _expand_query_with_llm(message: str) -> list[str]:
    """Sử dụng LLM để sinh các truy vấn thay thế/liên quan."""
    # Chỉ chạy khi có OpenAI và câu hỏi đủ dài
    if not (USE_OPENAI and openai_client and len(message) > 10):
        return [message]
    try:
        prompt = (
            "Bạn là một trợ lý tìm kiếm hữu ích, chuyên mở rộng truy vấn của người dùng. "
            "Dựa trên câu hỏi của người dùng, hãy tạo ra 2 câu hỏi thay thế hoặc từ khóa tìm kiếm liên quan có khả năng tìm thấy thông tin phù hợp. "
            "Giữ các truy vấn ngắn gọn và bằng tiếng Việt. Chỉ trả về một danh sách JSON chứa các chuỗi. "
            "Ví dụ: Người dùng hỏi 'chính sách nghỉ phép'. Bạn trả về: "
            '["quy định về ngày nghỉ phép năm", "thủ tục xin nghỉ phép"]\n\n'
            f'Câu hỏi của người dùng: "{message}"\n'
            "Danh sách JSON các truy vấn thay thế:"
        )
        res = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
            response_format={"type": "json_object"},  # Yêu cầu output dạng JSON
        )
        content = (res.choices[0].message.content or "").strip()
        import json
        data = json.loads(content)
        # Xử lý cả hai trường hợp: list hoặc object chứa list
        if isinstance(data, list):
            expanded = [str(q) for q in data if isinstance(q, str)]
        elif isinstance(data, dict):
            # Find the first key in the dict that holds a list
            key = next((k for k, v in data.items() if isinstance(v, list)), None)
            expanded = [str(q) for q in data.get(key or '', []) if isinstance(q, str)]
        else:
            expanded = []

        # Combine original query with expanded ones (max 3 total)
        all_queries = [message] + expanded[:2]
        print(f"[Chat] Expanded query: {message} -> {all_queries}")
        return all_queries
    except Exception as e:
        print(f"[Chat] Query expansion failed: {e}")
        return [message] # Always return original query on error

def _analyze_and_decompose_query(message: str) -> DecomposedQuery:
    """
    Analyzes the user's query to determine complexity, intent, and sub-questions for multi-step retrieval.
    """
    # Only run for reasonably long queries where multi-step is plausible
    if not (USE_OPENAI and openai_client and len(message) > 20):
        return {**DEFAULT_SIMPLE_QUERY}

    try:
        prompt = f'''
You are an expert query analyzer. Your task is to decompose a complex user question into simpler sub-questions for a Retrieval-Augmented Generation (RAG) system. You must also determine the user's primary intent and suggest a retrieval strategy.

Analyze the user's question: "{message}"

Follow these steps:
1.  **Determine Intent**: Classify the user's primary goal. Choose one from:
    *   `COMPARISON`: The user wants to compare/contrast two or more items (e.g., "compare policy A and B").
    *   `SYNTHESIS`: The user wants a summary or explanation that requires combining information from multiple topics (e.g., "what is the relationship between vacation policy and project deadlines?").
    *   `CAUSALITY`: The user is asking for a cause-and-effect relationship (e.g., "what was the reason for the Q3 budget cut and what was its impact?").
    *   `SIMPLE_QA`: The question is likely answerable from a single piece of context.

2.  **Suggest Retrieval Strategy**:
    *   `parallel`: Use for `COMPARISON` or `SYNTHESIS` where sub-questions are independent and can be searched at the same time.
    *   `sequential`: Use for `CAUSALITY` or questions with dependencies, where the answer to one sub-question is needed for the next.

3.  **Decompose into Sub-Questions**: If the intent is NOT `SIMPLE_QA`, break the main question down into 2-4 simple, self-contained questions. If the intent is `SIMPLE_QA`, return an empty list.

Respond ONLY with a valid JSON object in the following format. Do not add any explanations.
{{
  "intent": "...",
  "retrieval_strategy": "...",
  "sub_questions": ["...", "..."]
}}
'''
        res = openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        content = (res.choices[0].message.content or "").strip()

        data = json.loads(content)

        # Validate the structure and types
        validated_data: DecomposedQuery = {
            "intent": data.get("intent", "SIMPLE_QA"),
            "retrieval_strategy": data.get("retrieval_strategy", "parallel"),
            "sub_questions": [str(q) for q in data.get("sub_questions", []) if isinstance(q, str)]
        }

        # If it's simple, sub_questions should be empty.
        if validated_data["intent"] == "SIMPLE_QA":
            validated_data["sub_questions"] = []


        # Return the validated result
        return validated_data

    except Exception as e:
        print(f"[Chat] _analyze_and_decompose_query failed: {e}")
        return {**DEFAULT_SIMPLE_QUERY}


# ---------- Intent & Entity Extraction tailored for Excel/CRM ----------

def _map_nlq_to_generation_intent(nlq_intent: str) -> str:
    """Map Vietnamese NLQ intent to generation prompt variants."""
    m = (nlq_intent or "").upper()
    if m == "COMPARISON":
        return "COMPARISON"
    if m == "AGGREGATION":
        return "AGGREGATION"
    # Lookup / Filter_List / Descriptive -> simple QA style
    return "SIMPLE_QA"


def analyze_nlq_intent_and_entities(query: str) -> NLQIntentResult:
    """Classify the user query into the 5 Vietnamese categories and extract entities.
    Uses OpenAI when available; otherwise falls back to heuristics.
    """
    q = (query or "").strip()
    # LLM path
    if USE_OPENAI and openai_client:
        try:
            prompt = (
                "Bạn là tác nhân phân tích câu hỏi tự nhiên cho hệ thống hỏi đáp dữ liệu Excel/CRM. "
                "Hãy phân loại câu hỏi vào đúng một trong các nhóm: \n"
                "- LOOKUP (Tra cứu: tìm 1 đối tượng cụ thể, ví dụ tìm theo mã/sku, hỏi giá của 1 sản phẩm)\n"
                "- FILTER_LIST (Lọc & Liệt kê: trả về danh sách nhiều đối tượng theo điều kiện)\n"
                "- COMPARISON (So sánh: so sánh 2 hay nhiều đối tượng)\n"
                "- AGGREGATION (Tính toán & Tổng hợp: tổng/đếm/trung bình/min/max)\n"
                "- DESCRIPTIVE (Mô tả: câu hỏi dạng văn bản tự do, chính sách, mô tả)\n\n"
                "Đồng thời trích xuất thực thể làm bộ lọc SQL ở dạng JSON key-value (ví dụ: sku, ma, ma_san_pham, ten_san_pham, danh_muc, sheet, cot, gia_tri, so_luong,...).\n"
                "Chỉ trả về JSON, đúng schema: {\"intent\": \"...\", \"entities\": {...}}.\n\n"
                f"Câu hỏi: \"{q}\"\nKết quả:" )
            res = openai_client.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=220,
                response_format={"type": "json_object"},
            )
            raw = (res.choices[0].message.content or "{}").strip()
            data = json.loads(raw)
            intent = str(data.get("intent", "DESCRIPTIVE")).upper()
            entities = data.get("entities") or {}
            if not isinstance(entities, dict):
                entities = {}
            # Normalize keys to snake_case-like
            def _norm_key(s: str) -> str:
                import re, unicodedata
                try:
                    s2 = unicodedata.normalize('NFKD', s or '')
                    s2 = ''.join([c for c in s2 if not unicodedata.combining(c)])
                except Exception:
                    s2 = s or ''
                s2 = s2.lower()
                s2 = re.sub(r"[^a-z0-9_]+", "_", s2)
                return s2.strip("_")
            ents_norm = { _norm_key(str(k)): v for k, v in entities.items() if v is not None and str(v).strip() != '' }
            final_intent = intent if intent in {"LOOKUP","FILTER_LIST","COMPARISON","AGGREGATION","DESCRIPTIVE"} else "DESCRIPTIVE"
            # Cast to the specific Literal type to satisfy the type checker
            validated_intent: Literal["LOOKUP", "FILTER_LIST", "COMPARISON", "AGGREGATION", "DESCRIPTIVE"] = final_intent # type: ignore
            return {"intent": validated_intent, "entities": ents_norm}
        except Exception as e:
            print(f"[Chat] NLQ intent analysis (LLM) failed: {e}")
    # Heuristic fallback
    n = _normalize_vi(q)
    def has_any(words: list[str]) -> bool:
        return any(w in n for w in words)
    if has_any(["so sanh", "doi chieu", "compare"]):
        intent = "COMPARISON"
    elif has_any(["tong", "tong cong", "trung binh", "trung binh cong", "dem", "so luong", "sum", "avg", "min", "max"]):
        intent = "AGGREGATION"
    elif has_any(["liet ke", "danh sach", "thuoc danh muc", "loc "]):
        intent = "FILTER_LIST"
    elif has_any(["ma ", "sku", "code", "gia cua", "thong tin chi tiet", "chi tiet cua"]):
        intent = "LOOKUP"
    else:
        intent = "DESCRIPTIVE"
    return {"intent": intent, "entities": {}}


def _table_like_confident(ctx: str) -> bool:
    if not ctx:
        return False
    lines = [l for l in ctx.splitlines() if l.strip()]
    tbl_rows = sum(1 for l in lines if l.strip().startswith("| "))
    return tbl_rows >= 3

async def _extract_text_from_upload_file(file: UploadFile) -> tuple[str, dict]:
    """Extract text from a single uploaded file (PDF/Excel/Image/CSV)
    Returns (content_text, summary_dict)
    """
    name = file.filename or "uploaded_file"
    ext = name.split(".")[-1].lower()
    raw = await file.read()
    summary = {"filename": name, "type": ext, "size": len(raw)}
    try:
        if ext == "pdf":
            try:
                from langchain_community.document_loaders import PyPDFLoader  # type: ignore
                import os, uuid as _uuid
                tmp = f"/tmp/{_uuid.uuid4()}_{name}"
                with open(tmp, "wb") as fh:
                    fh.write(raw)
                loader = PyPDFLoader(tmp)
                docs = loader.load()
                os.remove(tmp)
                pages = [d.page_content or "" for d in docs]
                summary["pages"] = len(pages)
                return ("\n\n".join(pages).strip(), summary)
            except Exception:
                try:
                    from PyPDF2 import PdfReader  # type: ignore
                    import io as _io
                    reader = PdfReader(_io.BytesIO(raw))
                    pages = [p.extract_text() or "" for p in reader.pages]
                    summary["pages"] = len(pages)
                    return ("\n\n".join(pages).strip(), summary)
                except Exception as e:
                    return (f"", {**summary, "error": f"PDF parse failed: {e}"})
        elif ext in ("xlsx", "xls"):
            import pandas as pd
            import io as _io
            try:
                excel_engine = "openpyxl" if ext == "xlsx" else "xlrd"
                xls = pd.read_excel(_io.BytesIO(raw), sheet_name=None, engine=excel_engine, dtype=str)
            except Exception:
                xls = pd.read_excel(_io.BytesIO(raw), sheet_name=None, dtype=str)
            lines: list[str] = []
            sheet_cnt = 0
            row_cnt = 0
            for sh, df in (xls or {}).items():
                sheet_cnt += 1
                if df is None or df.empty:
                    continue
                df = df.dropna(how='all').dropna(axis=1, how='all')
                df.columns = [str(c).strip() for c in df.columns]
                for idx, row in df.iterrows():
                    row_cnt += 1
                    parts = []
                    for col, val in row.items():
                        if val is not None and str(val).strip() != "":
                            parts.append(f"{col} là '{val}'")
                    if parts:
                        lines.append(f"Dòng {idx + 2}: " + ", ".join(parts) + ".")
            txt = ("\n".join(lines)).strip()
            return txt, {**summary, "sheets": sheet_cnt, "rows": row_cnt}
        elif ext in ("csv",):
            import pandas as pd
            import io as _io
            try:
                df = pd.read_csv(_io.BytesIO(raw), dtype=str)
            except Exception:
                df = pd.read_csv(_io.BytesIO(raw), dtype=str, encoding="cp1258", sep=None, engine="python")
            lines: list[str] = []
            df = df.dropna(how='all').dropna(axis=1, how='all')
            df.columns = [str(c).strip() for c in df.columns]
            for idx, row in df.iterrows():
                parts = []
                for col, val in row.items():
                    if val is not None and str(val).strip() != "":
                        parts.append(f"{col} là '{val}'")
                if parts:
                    lines.append(f"Dòng {idx + 2}: " + ", ".join(parts) + ".")
            txt = ("\n".join(lines)).strip()
            return txt, {**summary, "rows": len(df)}
        elif ext in ("png", "jpg", "jpeg", "gif", "webp"):
            import io as _io
            pil = Image.open(_io.BytesIO(raw))
            pil = preprocess_image(pil)
            text, conf = ocr_with_confidence(pil)
            return text or "", {**summary, "ocr_confidence": conf}
        else:
            return "", {**summary, "error": "Unsupported file type"}
    except Exception as e:
        return "", {**summary, "error": str(e)}


def _get_session_temp_context(db: Session, session_id: str, user_id: str) -> str:
    """Fetch valid temporary context for session (and purge expired)."""
    try:
        now = datetime.utcnow()
        # Purge global expired to keep table small
        db.query(TemporaryContext).filter(TemporaryContext.expires_at != None, TemporaryContext.expires_at < now).delete()
        db.commit()
    except Exception:
        db.rollback()
    row = db.query(TemporaryContext).filter(TemporaryContext.session_id == session_id, TemporaryContext.user_id == user_id).order_by(TemporaryContext.created_at.desc()).first()
    if not row:
        return ""
    if row.expires_at and row.expires_at < datetime.utcnow():
        try:
            db.delete(row)
            db.commit()
        except Exception:
            db.rollback()
        return ""
    return row.content or ""


@router.post("/upload-context-file")
async def upload_context_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    """Upload a file to be used as TEMPORARY context for this chat session.
    It will not be indexed into global vector DB. Replaces previous temp context for the session.
    """
    # Validate/ensure session belongs to user
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    content, meta = await _extract_text_from_upload_file(file)
    if not content.strip():
        raise HTTPException(status_code=400, detail=f"Không trích xuất được nội dung từ tệp: {meta.get('error','no content')}")
    # Replace existing temp context for this session/user
    try:
        db.query(TemporaryContext).filter(TemporaryContext.session_id == session_id, TemporaryContext.user_id == user.id).delete()
        from datetime import timedelta as _td
        expires_at = datetime.utcnow() + _td(minutes=TEMP_CONTEXT_TTL_MINUTES)
        row = TemporaryContext(session_id=session_id, user_id=user.id, filename=meta.get('filename'), file_type=meta.get('type'), file_size=meta.get('size'), summary=json.dumps({k: v for k, v in meta.items() if k not in {'error'}}, ensure_ascii=False), content=content[:200000], expires_at=expires_at)
        db.add(row)
        db.commit(); db.refresh(row)
    except Exception as e:
        db.rollback()

@router.delete("/upload-context-file")
async def clear_context_file(
    session_id: str = Form(...),
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    try:
        db.query(TemporaryContext).filter(TemporaryContext.session_id == session_id, TemporaryContext.user_id == user.id).delete()
        db.commit()
        return {"message": "Đã xóa ngữ cảnh tạm thời cho phiên này."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Không thể xóa ngữ cảnh tạm thời: {e}")









@router.post("/send", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    """Send a message and get AI response (now with basic conversation history)."""
    print(f"[Chat] Received message from user {user.username}: {request.message}")

    try:
        # user is already provided by verify_user

        # Get or create chat session
        if request.session_id:
            session = db.query(ChatSession).filter(
                ChatSession.id == request.session_id,
                ChatSession.user_id == user.id
            ).first()
            if not session:
                raise HTTPException(status_code=404, detail="Chat session not found")
        else:
            session = ChatSession(
                id=str(uuid.uuid4()),
                user_id=user.id,



                title=request.message[:50] + "..." if len(request.message) > 50 else request.message
            )
            db.add(session)
            db.commit()
            db.refresh(session)

        # Build recent history based on configured HISTORY_TURNS (N pairs => 2N messages)
        recent_msgs = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.timestamp.desc()).limit(HISTORY_MAX_HISTORY_MESSAGES).all()
        history = []
        for m in reversed(recent_msgs):
            if m.is_user:
                if m.message:
                    history.append({"role": "user", "content": m.message})
            else:
                if m.response:
                    history.append({"role": "assistant", "content": m.response})
        # Prepend rolling summary (if any) as a system message to give background context
        if getattr(session, "summary", None):
            history = [{"role": "system", "content": f"Tóm tắt hội thoại trước:\n{session.summary}"}] + history

        # Get ChromaDB collections (query across all backends for fresh data)


        # Load temporary per-session context if user uploaded a file
        temp_ctx = _get_session_temp_context(db, session.id, user.id)

        # --- Agentic Layer: Decide which tool to use ---
        context = ""


        # (Fallback Agent) If no internal context is found, try a web search as a last resort
        if not final_context.strip() and _should_try_web_search(request.message):
            web_context = await run_web_search_agent(request.message)
            if web_context:
                print("[Agent] Web Search agent succeeded. Using its result as context.")
                final_context = web_context
                all_sources = [{"title": "Web Search", "content": web_context}] # Overwrite sources






        sources = []






        final_context_docs = {}




        # 1. (Tool) Attempt to use Text-to-SQL for structured data queries
        if _should_try_text_to_sql(request.message):
            sql_result = run_text_to_sql_agent(request.message, engine)



            if sql_result:
                print("[Agent] Text-to-SQL agent succeeded. Using its result as context.")
                context = sql_result
                sources.append({"title": "Database Query", "content": sql_result})

        # 2. (Tool) If Text-to-SQL is not used or fails, proceed with standard RAG from VectorDB
        if not context:


            collections = get_all_chroma_collections() or []
            if not collections:
                c = get_chroma_collection()
                if c:
                    collections = [c]

            # Debug: Check collection status
            try:
                total = 0
                for c in collections[:3]:
                    cnt = c.count()
                    total += cnt
                    dlog(f"[Chat] Collection '{getattr(c, 'name', 'unknown')}' has {cnt} docs")
                dlog(f"[Chat] Total docs across {len(collections)} collections: {total}")
            except Exception as e:
                dlog(f"[Chat] Could not check collections: {e}")

            # Step 1: Classify NLQ intent/entities (SQL-first policy controller)
            nlq = analyze_nlq_intent_and_entities(request.message)

            # Step 2: Retrieve from structured SQL sources first
            try:
                allowance_ctx, allowance_sources = find_allowance_sql_context(db, request.message)
            except Exception:
                allowance_ctx, allowance_sources = "", []
            try:
                crm_ctx, crm_sources = find_crm_products_context(db, request.message)
            except Exception:
                crm_ctx, crm_sources = "", []
            try:
                excel_ctx, excel_sources = find_excel_sql_context(db, request.message)
            except Exception:
                excel_ctx, excel_sources = "", []
            try:
                ocr_ctx, ocr_sources = find_ocr_sql_context(db, request.message)
            except Exception:
                ocr_ctx, ocr_sources = "", []

            structured_parts = [p for p in [allowance_ctx, crm_ctx, excel_ctx, ocr_ctx] if p]
            structured_sources = (allowance_sources or []) + (crm_sources or []) + (excel_sources or []) + (ocr_sources or [])

            # Decide if we need vector search as fallback
            prefer_structured = nlq["intent"] in {"LOOKUP", "FILTER_LIST", "AGGREGATION"}
            has_confident_table = _table_like_confident(crm_ctx) or _table_like_confident(excel_ctx)
            use_vector = not structured_parts or (not prefer_structured and not has_confident_table)

            vector_context, vector_sources = "", []
            if use_vector:
                # Analyze query complexity and decomposed retrieval only if needed for vector
                query_analysis = _analyze_and_decompose_query(request.message)
                if query_analysis["intent"] == "SIMPLE_QA" or not query_analysis["sub_questions"]:
                    retrieval_queries = _expand_query_with_llm(request.message)
                    vector_context, vector_sources = get_context_with_sources(retrieval_queries, collections)
                else:
                    if query_analysis["retrieval_strategy"] == "sequential":
                        vector_context, vector_sources = await _sequential_retrieval(
                            request.message, query_analysis["sub_questions"], collections
                        )
                    else:
                        retrieval_queries = [request.message] + query_analysis["sub_questions"]
                        vector_context, vector_sources = get_context_with_sources(retrieval_queries, collections)
                # Prune vector context for focus
                vector_context, vector_sources = _prune_context_with_llm(request.message, vector_context, vector_sources)

            # Combine contexts: structured first, vector as supplemental
            merged_contexts = []
            merged_contexts.extend([m.strip() for m in structured_parts if m and m.strip()])
            if vector_context:
                merged_contexts.append(vector_context)
            final_context = "\n\n".join(merged_contexts)

            all_sources = structured_sources + (vector_sources or [])


            # (Fallback Agent) If no internal context is found, try a web search as a last resort
            if not final_context.strip() and _should_try_web_search(request.message):
                web_context = await run_web_search_agent(request.message)
                if web_context:
                    print("[Agent] Web Search agent succeeded. Using its result as context.")
                    final_context = web_context
                    all_sources = [{"title": "Web Search", "content": web_context}] # Overwrite sources

            print(f"[Chat] Final context length: {len(final_context)} chars for query: '{request.message}'")
            if final_context:
                preview = (final_context[:250] + "...") if len(final_context) > 250 else final_context
                print(f"[Chat] Context preview: {preview}")

            # Get AI response with history
            if _is_greeting(request.message):
                ai_text = "Xin chào! Mình là trợ lý AI nội bộ của DalatHasfarm. Mình có thể hỗ trợ bạn tra cứu thông tin hoặc giải đáp công việc gì hôm nay?"
                tokens_used, response_time_ms = 0, 0
            elif STRICT_CONTEXT_ONLY and not (final_context and final_context.strip()) and _requires_internal_docs(request.message):
                ai_text = (
                    "Xin lỗi, mình chưa tìm thấy thông tin liên quan trong tài liệu nội bộ để trả lời câu hỏi này. "
                    "Bạn có thể mô tả cụ thể hơn, hoặc tải/thêm tài liệu liên quan. Mình sẽ không đoán để tránh trả lời sai."
                )
                tokens_used, response_time_ms = 0, 0
            else:
                ai_text, tokens_used, response_time_ms = get_ai_response(
                    request.message,
                    final_context,
                    history=history,
                    intent=_map_nlq_to_generation_intent(nlq.get("intent", "DESCRIPTIVE"))
                )

            # Build follow-up suggestions (UI can surface these as clickable chips)
            suggestions = _suggest_followups(request.message, ai_text)

        # Save user message
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session.id,
            message=request.message,
            response="",  # do not duplicate AI reply on user row
            is_user=True,
            timestamp=datetime.now(timezone.utc),
            tokens_used=tokens_used,
            response_time=response_time_ms
        )
        db.add(user_message)

        # Save AI response
        ai_message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session.id,
            message="",
            response=ai_text,
            is_user=False,
            timestamp=datetime.now(timezone.utc),
            tokens_used=0,
            response_time=0,
            sources=json.dumps(all_sources, ensure_ascii=False) if all_sources else None
        )
        db.add(ai_message)

        # Update session
        session.updated_at = datetime.now(timezone.utc)

        db.commit()
        # Auto-title the session (first turn)
        try:
            default_title = request.message[:50] + ("..." if len(request.message) > 50 else "")
            if (not session.title) or session.title in ("Đoạn Chat", default_title) or session.title.startswith((request.message or "")[:20]):
                new_title = _suggest_chat_title(request.message, ai_text)
                if new_title and new_title != session.title:
                    session.title = new_title
                    session.updated_at = datetime.now(timezone.utc)
                    db.commit()
        except Exception as _e:
            print(f"[Chat] Auto-title failed: {_e}")


        return ChatMessageResponse(
            id=ai_message.id,
            message=request.message,
            response=ai_text,
            timestamp=ai_message.timestamp,
            session_id=session.id,
            sources=all_sources
        )




    except Exception as e:
        print(f"[Chat] Error processing message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.post("/send-with-files", response_model=ChatMessageResponse)
async def send_message_with_files(
    message: str = Form(""),
    session_id: str | None = Form(None),
    files: List[UploadFile] = File(None),
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    """Send a message with attached files (pdf/txt/doc/docx/images). Extract text, enrich context, and answer.
    - Images are OCR'd via OpenAI Vision when USE_OPENAI=1; otherwise ignored.
    - Documents are chunked and embedded into the context but not persisted to DB/Chroma.
    """
    print(f"[Chat] Received message+files from {user.username}: {message} | {len(files)} files")

    # Prepare or get chat session
    if session_id:
        session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
    else:
        session = ChatSession(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title=(message or (files[0].filename if files else ""))[:50]
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    # Build recent history based on configured HISTORY_TURNS (N pairs => 2N messages)
    recent_msgs = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.timestamp.desc()).limit(HISTORY_MAX_HISTORY_MESSAGES).all()
    history = []
    for m in reversed(recent_msgs):
        if m.is_user:
            if m.message:
                history.append({"role": "user", "content": m.message})
        else:
            if m.response:
                history.append({"role": "assistant", "content": m.response})

    # Aggregate context from files
    extracted_texts: list[str] = []
    saved_image_urls: list[str] = []  # URLs under /static for History rendering
    vision_captions: list[str] = []
    vision_tags: list[str] = []
    vision_entities: list[str] = []

    async def read_upload(uf: UploadFile) -> bytes:
        return await uf.read()

    import io, zipfile
    for f in files or []:
        name = f.filename or ""
        ext = name.split(".")[-1].lower()
        try:
            raw = await read_upload(f)
            # PDF/TXT
            if ext == "pdf":
                # Prefer LangChain PyPDFLoader; fallback to PyPDF2 when unavailable
                try:
                    from langchain_community.document_loaders import PyPDFLoader  # type: ignore
                    tmp = f"/tmp/{uuid.uuid4()}_{name}"
                    with open(tmp, "wb") as fh:
                        fh.write(raw)
                    loader = PyPDFLoader(tmp)
                    docs = loader.load()
                    os.remove(tmp)
                    extracted_texts.extend([d.page_content for d in docs])
                except Exception:
                    try:
                        from PyPDF2 import PdfReader  # type: ignore
                        import io as _io
                        reader = PdfReader(_io.BytesIO(raw))
                        pages = [p.extract_text() or "" for p in reader.pages]
                        extracted_texts.append("\n".join(pages))
                    except Exception as _pe:
                        print(f"[Chat] PDF parse fallback failed: {_pe}")
            elif ext in ("txt",):
                extracted_texts.append(raw.decode(errors="ignore"))
            elif ext in ("docx",):
                # Minimal .docx text extraction via zipfile
                try:
                    with zipfile.ZipFile(io.BytesIO(raw)) as z:
                        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
                    # Strip XML tags (naive)
                    text = xml.replace("<w:p>", "\n").replace("</w:p>", "\n").replace("<w:t>", "").replace("</w:t>", "")
                    # Remove residual tags
                    import re as _re
                    text = _re.sub(r"<[^>]+>", "", text)
                    extracted_texts.append(text)
                except Exception as e:
                    print(f"[Chat] DOCX parse error: {e}")
            elif ext in ("jpg", "jpeg", "png", "gif", "webp"):
                # Persist image to static so History can render it
                try:
                    os.makedirs(os.path.join("static", "chat_uploads"), exist_ok=True)
                    safe_name = f"{uuid.uuid4()}_{os.path.basename(name)}"
                    disk_path = os.path.join("static", "chat_uploads", safe_name)
                    with open(disk_path, "wb") as imgf:
                        imgf.write(raw)
                    public_url = f"/static/chat_uploads/{safe_name}"
                    saved_image_urls.append(public_url)
                except Exception as e:
                    print(f"[Chat] Could not save image for history: {e}")
                # OCR & caption via OpenAI Vision if enabled
                if USE_OPENAI and openai_client:
                    try:
                        b64 = base64.b64encode(raw).decode("utf-8")
                        img_url = f"data:image/{ext};base64,{b64}"
                        # 1) OCR-like text extraction
                        prompt_text = ("Hãy trích xuất mọi chữ/tables chính từ ảnh dưới dạng văn bản tiếng Việt,"
                                       " không cần mô tả thừa.")
                        vis_text = openai_client.chat.completions.create(
                            model=OPENAI_CHAT_MODEL,


                            messages=[
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt_text},
                                    {"type": "image_url", "image_url": {"url": img_url}}
                                ]}
                            ],
                            temperature=0.0,
                        )
                        extracted_texts.append(vis_text.choices[0].message.content or "")
                        # 2) Caption + tags/entities for recognition
                        prompt_cap = os.getenv(
                            "VISION_CAPTION_PROMPT",
                            "Mô tả ngắn gọn nội dung ảnh bằng tiếng Việt. Sau đó trả về JSON đúng định dạng: {\"caption\": string, \"tags\": string[], \"entities\": string[]}"
                        )
                        vis_cap = openai_client.chat.completions.create(
                            model=OPENAI_CHAT_MODEL,
                            messages=[
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt_cap},
                                    {"type": "image_url", "image_url": {"url": img_url}}
                                ]}
                            ],
                            temperature=0.2,
                        )
                        cap_raw = (vis_cap.choices[0].message.content or "").strip()
                        try:
                            data = json.loads(cap_raw)
                            cap = str(data.get("caption", "")).strip()
                            tags = [str(t).strip() for t in (data.get("tags") or [])][:20]
                            ents = [str(t).strip() for t in (data.get("entities") or [])][:20]
                        except Exception:
                            cap = cap_raw
                            tags, ents = [], []
                        if cap:
                            vision_captions.append(cap)
                        if tags:
                            vision_tags.extend(tags)
                        if ents:
                            vision_entities.extend(ents)
                    except Exception as e:
                        print(f"[Chat] Vision analyze failed: {e}")
            else:
                print(f"[Chat] Unsupported file type for inline context: {ext}")
        except Exception as e:
            print(f"[Chat] Error reading file {name}: {e}")

    # Build message content including image markdown so History can render
    msg_with_images = message
    if saved_image_urls:
        md_imgs = "\n".join([f"![{os.path.basename(u)}]({u})" for u in saved_image_urls])
        msg_with_images = (message + "\n\n" + md_imgs) if message else md_imgs

    # Combine with Chroma context + search by image OCR/Vision (augment with last N user turns)
    collections = get_all_chroma_collections() or []
    aug_query2 = build_augmented_query(message or "", history, min_user_msgs=HISTORY_USER_TURNS)
    chroma_ctx, chroma_sources = get_context_with_sources([aug_query2], collections) if aug_query2 else ("", [])

    files_ctx = "\n\n".join(extracted_texts[:10])

    related_ctx_parts: list[str] = []
    if embeddings and extracted_texts and collections:
        try:
            for t in extracted_texts[:5]:
                q = t[:1000]
                if not q.strip(): continue
                q_emb = embeddings.embed_query(q)
                for coll in collections:
                    res = coll.query(query_embeddings=[q_emb], n_results=2)
                    if res and res.get('documents'):
                        docs_list = res['documents']
                        if docs_list and docs_list[0]:
                            related_ctx_parts.extend(docs_list[0])
        except Exception as e:
            print(f"[Chat] related search by OCR failed: {e}")
    related_ctx = "\n\n".join(related_ctx_parts)

    vision_summary_parts: list[str] = []
    if vision_captions: vision_summary_parts.append("; ".join(vision_captions))
    if vision_tags: vision_summary_parts.append("Tags: " + ", ".join(sorted({t for t in vision_tags if t})[:20]))
    if extracted_texts: vision_summary_parts.append(("OCR: " + " ".join(extracted_texts))[:1000])
    vision_summary = " ".join(vision_summary_parts).strip()
    vision_ctx, vision_sources = get_context_with_sources([vision_summary], collections) if vision_summary else ("", [])

    ctx_parts: list[str] = []
    all_sources: list[dict] = []
    if vision_ctx: ctx_parts.append(vision_ctx); all_sources.extend(vision_sources)
    if chroma_ctx: ctx_parts.append(chroma_ctx); all_sources.extend(chroma_sources)
    if files_ctx: ctx_parts.append(files_ctx)
    if related_ctx: ctx_parts.append(related_ctx)
    full_ctx = "\n\n".join(ctx_parts).strip()

    if not (vision_ctx or chroma_ctx or related_ctx):
        fallback_desc = "; ".join(vision_captions[:2])
        tag_line = ", ".join(sorted({t for t in vision_tags})[:10])
        if fallback_desc or tag_line or files_ctx:
            full_ctx = f"Mô tả ảnh (Vision): {fallback_desc}\nTags: {tag_line}\nOCR: {files_ctx[:500]}".strip()

    if _is_greeting(message):
        ai_text, tokens_used, response_time_ms = "Xin chào! Mình là trợ lý AI nội bộ của DalatHasfarm. Mình có thể hỗ trợ bạn tra cứu thông tin hoặc giải đáp công việc gì hôm nay?", 0, 0
    elif STRICT_CONTEXT_ONLY and not full_ctx.strip() and _requires_internal_docs(message):
        ai_text, tokens_used, response_time_ms = ("Xin lỗi, mình chưa tìm thấy thông tin...", 0, 0)
    else:
        ai_text, tokens_used, response_time_ms = get_ai_response(message or "(Người dùng gửi tệp)", full_ctx, history=history)

    user_message = ChatMessage(id=str(uuid.uuid4()), session_id=session.id, message=msg_with_images, response="", is_user=True, timestamp=datetime.now(timezone.utc), tokens_used=tokens_used, response_time=response_time_ms)
    db.add(user_message)
    ai_message = ChatMessage(id=str(uuid.uuid4()), session_id=session.id, message="", response=ai_text, is_user=False, timestamp=datetime.now(timezone.utc), tokens_used=0, response_time=0, sources=json.dumps(all_sources, ensure_ascii=False) if all_sources else None)
    db.add(ai_message)
    session.updated_at = datetime.now(timezone.utc)

    try:
        base_msg = (message or (files[0].filename if files else ""))
        default_title = (base_msg or "")[:50]
        if (not session.title) or session.title in ("Đoạn Chat", default_title) or session.title.startswith((base_msg or "")[:20]):
            new_title = _suggest_chat_title(base_msg, ai_text)
            if new_title and new_title != session.title:
                session.title = new_title
                session.updated_at = datetime.now(timezone.utc)
                db.commit()
    except Exception as _e:
        print(f"[Chat] Auto-title (files) failed: {_e}")

    db.commit()

    return ChatMessageResponse(id=ai_message.id, message=msg_with_images, response=ai_text, timestamp=ai_message.timestamp, session_id=session.id, sources=all_sources)


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_chat_sessions(
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    """Get all chat sessions for the current user"""

    # user provided by verify_user

    sessions = db.query(ChatSession).filter(
        ChatSession.user_id == user.id
    ).order_by(ChatSession.updated_at.desc()).all()

    return [
        ChatSessionResponse(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at
        )
        for session in sessions
    ]

@router.patch("/sessions/{session_id}", response_model=ChatSessionResponse)
async def rename_chat_session(
    session_id: str,
    new_title: str,
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    """Rename a chat session"""
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == user.id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    title = (new_title or "").strip()[:60] or "Đoạn Chat"
    session.title = title
    session.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(session)
    return ChatSessionResponse(id=session.id, title=session.title, created_at=session.created_at, updated_at=session.updated_at)


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_chat_messages(
    session_id: str,
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    """Get all messages in a chat session"""

    # user provided by verify_user

    # Verify session belongs to user
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.timestamp).all()

    return [
        ChatMessageResponse(
            id=message.id,
            message=message.message,
            response=message.response,
            timestamp=message.timestamp,
            session_id=session_id
        )
        for message in messages
    ]

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    """Delete a chat session and all its messages"""

    # user is provided by verify_user
    # Verify session belongs to user
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == user.id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Delete all messages in the session
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()

    # Delete the session
    db.delete(session)
    db.commit()

    return {"message": "Chat session deleted successfully"}

@router.post("/stream")
async def stream_message(
    request: ChatMessageRequest,
    user: User = Depends(verify_user),
    db: Session = Depends(get_db)
):
    """Stream assistant response progressively for better UX (text-only; no files)."""
    print(f"[Chat] (stream) Received message from user {user.username}: {request.message}")

    # Prepare or create session
    session = None
    if request.session_id:
        session = db.query(ChatSession).filter(ChatSession.id == request.session_id, ChatSession.user_id == user.id).first()
    if not session:
        session = ChatSession(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title=request.message[:50] + "..." if len(request.message) > 50 else request.message
        )
        db.add(session)
        db.commit()
        db.refresh(session)

    # Build history
    history: list[dict] = []
    msgs = db.query(ChatMessage).filter(ChatMessage.session_id == session.id).order_by(ChatMessage.timestamp).all()
    for m in msgs[-HISTORY_MAX_HISTORY_MESSAGES:]:
        if m.is_user:
            history.append({"role": "user", "content": m.message or ""})
        else:
            history.append({"role": "assistant", "content": m.response or ""})

    # Retrieval with augmented query
    collections = get_all_chroma_collections() or []
    context, sources = get_context_with_sources([request.message], collections)

    try:
        crm_ctx, crm_sources = find_crm_products_context(db, request.message)
    except Exception:
        crm_ctx, crm_sources = "", []
    try:
        ocr_ctx, ocr_sources = find_ocr_sql_context(db, request.message)
    except Exception:
        ocr_ctx, ocr_sources = "", []

    merged_contexts = []
    if crm_ctx: merged_contexts.append(crm_ctx)
    if ocr_ctx: merged_contexts.append(ocr_ctx)
    if context: merged_contexts.append(context)
    final_context = "\n\n".join([m.strip() for m in merged_contexts if m and m.strip()])
    all_sources = (sources or []) + (crm_sources or []) + (ocr_sources or [])


    async def _gen():
        full_text = ""
        started = time.time()
        try:
            sid_line = f"<<SID:{session.id}>>"
            yield sid_line.encode("utf-8")

            if _is_greeting(request.message):
                full_text = "Xin chào! Mình là trợ lý AI nội bộ của DalatHasfarm. Mình có thể hỗ trợ bạn tra cứu thông tin hoặc giải đáp công việc gì hôm nay?"
                yield full_text.encode("utf-8")
            elif STRICT_CONTEXT_ONLY and not (final_context and final_context.strip()) and _requires_internal_docs(request.message):
                full_text = "Xin lỗi, mình chưa tìm thấy thông tin liên quan trong tài liệu nội bộ để trả lời câu hỏi này."
                yield full_text.encode("utf-8")
            else:
                buffer: list[str] = []
                if USE_OPENAI and openai_client:
                    try:
                        user_prompt = f"Thông tin từ tài liệu:\n{final_context}\n\nCâu hỏi: {request.message}" if final_context.strip() else request.message
                        messages = [{"role": "system", "content": "Bạn là một trợ lý AI chuyên nghiệp..."}]
                        if history: messages.extend(history[-HISTORY_MAX_HISTORY_MESSAGES:])
                        messages.append({"role": "user", "content": user_prompt})
                        stream = openai_client.chat.completions.create(model=OPENAI_CHAT_MODEL, messages=messages, stream=True)
                        for ev in stream:
                            delta = getattr(getattr(ev.choices[0], 'delta', None), 'content', None)
                            if delta:
                                buffer.append(delta)
                                yield delta.encode("utf-8")
                        full_text = "".join(buffer).strip()
                    except Exception as oe:
                        print(f"[Chat] OpenAI stream failed: {oe}")

                if not full_text:
                    full_text, _, _ = get_ai_response(request.message, final_context, history=history)
                    yield full_text.encode("utf-8")
        finally:
            elapsed = int((time.time() - started) * 1000)
            user_msg = ChatMessage(id=str(uuid.uuid4()), session_id=session.id, message=request.message, response="", is_user=True, timestamp=datetime.now(timezone.utc), response_time=elapsed)
            db.add(user_msg)
            ai_msg = ChatMessage(id=str(uuid.uuid4()), session_id=session.id, message="", response=full_text, is_user=False, timestamp=datetime.now(timezone.utc), sources=json.dumps(all_sources, ensure_ascii=False) if all_sources else None)
            db.add(ai_msg)

            try:
                if (not session.title) or session.title == "Đoạn Chat" or session.title.startswith(request.message[:20]):
                    new_title = _suggest_chat_title(request.message, full_text)
                    if new_title and new_title != session.title:
                        session.title = new_title
            except Exception as _e:
                print(f"[Chat] Auto-title(stream) failed: {_e}")

            session.updated_at = datetime.now(timezone.utc)
            db.commit()

    return StreamingResponse(_gen(), media_type="text/plain; charset=utf-8")
