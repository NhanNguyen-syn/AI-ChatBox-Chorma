import uuid
import os
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session, joinedload
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LCDocument
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from services.security import validate_meta, antivirus_scan_bytes
from services.storage import save_bytes, S3_ENABLED
from services.background_jobs import enqueue_file_processing
# Optional: docx support via docx2txt loader
try:
    from langchain_community.document_loaders import Docx2txtLoader
    HAS_DOCX = True
except Exception:
    Docx2txtLoader = None  # type: ignore
    HAS_DOCX = False
from langchain_community.embeddings import OllamaEmbeddings
from typing import Optional

# Optional sentence-transformers fallback
try:
    from sentence_transformers import SentenceTransformer
    _st_model: Optional[SentenceTransformer] = None
except Exception:
    SentenceTransformer = None  # type: ignore
    _st_model = None  # type: ignore
import chromadb

# Provider selection: OpenAI or Ollama (default)
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"
openai_client = None
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

if USE_OPENAI:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embeddings = None
        print("[Embeddings] Using OpenAI embeddings")
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI client: {e}")
        print("Falling back to Ollama embeddings")
        USE_OPENAI = False

# Initialize Ollama embeddings (fallback)


from database import (
    get_db,
    Document,
    DocumentChunk,
    User,
    get_chroma_collection,
    get_chroma_collection_for_backend,
    OcrText,
    AllowanceTable,
)
from auth.jwt_handler import verify_token
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

router = APIRouter()
security = HTTPBearer()

# Auth dependency: extract user from Bearer token and ensure admin
from fastapi import status

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    return verify_token(credentials.credentials)

def require_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    payload = verify_token(credentials.credentials)
    user = db.query(User).filter(User.username == payload.get("sub")).first()
    if not user or not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user

class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    uploaded_by: str
    uploaded_at: str
    file_size: int
    status: str
    # Extra indexing info for observability
    backend: Optional[str] = None
    collection: Optional[str] = None
    chunks_indexed: Optional[int] = None

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int

# Initialize embeddings backend
try:
    embeddings = None
    if USE_OPENAI:
        embeddings = None  # handled via OpenAI client later
    else:
        # First try Ollama if available
        try:
            embeddings = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "llama2"))
            print("[Embeddings] Using OllamaEmbeddings")
        except Exception as ollama_err:
            print(f"[Embeddings] Ollama not available: {ollama_err}")
            # Fallback to sentence-transformers (local, no server) if installed
            if SentenceTransformer is not None:
                try:
                    _st_model = SentenceTransformer(os.getenv("ST_EMBED_MODEL", "all-MiniLM-L6-v2"))
                    print("[Embeddings] Using sentence-transformers fallback")
                except Exception as st_err:
                    print(f"[Embeddings] sentence-transformers init failed: {st_err}")
except Exception as e:
    print(f"[Embeddings] Initialization error: {e}")
# ---------------- PDF extraction and layout-aware chunking helpers ----------------
import re
from typing import Tuple

# Heuristic: keep table blocks intact; keep header with data rows when we must split
# Optimal chunking based on user request (aiming for 500-1000 tokens)
# ~4 chars/token -> 3000 chars is ~750 tokens
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "4000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "500"))


def _table_to_tsv(table: list[list[str | None]]) -> str:
    rows: list[str] = []
    for r in table or []:
        cells = [(c if c is not None else "").strip() for c in r]
        rows.append("\t".join(cells))
    return "\n".join(rows).strip()


def _split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    # Normalize line breaks; keep paragraph boundaries
    parts = re.split(r"\n{2,}", text.strip())
    # Merge tiny lines with next paragraph to avoid fragmentation
    merged: list[str] = []
    buf = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 80 and buf:
            buf.append(p)
        else:
            if buf:
                merged.append("\n\n".join(buf))
            buf = [p]
    if buf:
        merged.append("\n\n".join(buf))
    return merged


# ---------------- Semantic chunking helpers ----------------
from typing import Iterable
import math


def _split_sentences(text: str) -> list[str]:
    # Lightweight sentence splitter for vi/en
    if not text:
        return []
    # Normalize new lines to help boundary detection
    t = re.sub(r"[\r]+", "\n", text)
    # Protect table-like lines to not split within
    lines = [ln for ln in t.split("\n")]
    sents: list[str] = []
    buf = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            if buf:
                sents.append(" ".join(buf).strip())
                buf = []
            continue
        # If looks like a bullet item, flush previous
        if re.match(r"^[-*•▪►➤]\s+", ln):
            if buf:
                sents.append(" ".join(buf).strip())
                buf = []
            sents.append(ln)
            continue
        # Split by sentence enders but keep short fragments together
        parts = re.split(r"(?<=[\.!?…])\s+", ln)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) < 30:
                buf.append(p)
            else:
                if buf:
                    p = " ".join(buf + [p])
                    buf = []
                sents.append(p)
    if buf:
        sents.append(" ".join(buf).strip())
    return [s for s in sents if s]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Return embeddings for texts using OpenAI if available, else sentence-transformers if present.
    Fallback to simple hashing vector to avoid failure in dev environments.
    """
    if not texts:
        return []
    try:
        # Reuse OpenAI path from this module if available
        if USE_OPENAI and openai_client:
            # Batch in chunks to respect token limits
            embs: list[list[float]] = []
            MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
            for i in range(0, len(texts), 50):
                batch = texts[i:i+50]
                resp = openai_client.embeddings.create(model=MODEL, input=batch)
                embs.extend([d.embedding for d in resp.data])
            return embs
    except Exception as _oe:
        print(f"[Semantic] OpenAI embed failed, fallback: {_oe}")
    try:
        if _st_model is not None:
            return [list(map(float, v)) for v in _st_model.encode(texts, convert_to_numpy=True)]
    except Exception as _se:
        print(f"[Semantic] ST embed failed, fallback: {_se}")
    # Fallback: 64-dim hashing vector
    dim = 64
    vecs: list[list[float]] = []
    for s in texts:
        v = [0.0]*dim
        for i, ch in enumerate(s.encode("utf-8")):
            v[i % dim] += (ch % 23) / 10.0
        vecs.append(v)
    return vecs


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _semantic_chunk_text(text: str, target_chars: int = DEFAULT_CHUNK_SIZE, min_sim: float = 0.55) -> list[str]:
    sents = _split_sentences(text)
    if not sents:
        return [text.strip()] if text.strip() else []
    embs = _embed_texts(sents)
    chunks: list[str] = []
    buf: list[str] = []
    cur_len = 0
    for i, sent in enumerate(sents):
        e_prev = embs[i-1] if i > 0 else None
        e_cur = embs[i]
        sim_ok = True
        if e_prev is not None:
            sim_ok = (_cosine(e_prev, e_cur) >= min_sim)
        # Start new chunk if size exceeded or semantic drop
        if buf and (cur_len + len(sent) > target_chars or not sim_ok):
            chunks.append(" ".join(buf).strip())
            buf = [sent]
            cur_len = len(sent)
        else:
            buf.append(sent)
            cur_len += len(sent)
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]


def _semantic_chunk_documents(documents: list[LCDocument]) -> list[LCDocument]:
    out: list[LCDocument] = []
    for d in documents:
        txt = getattr(d, 'page_content', '') or ''
        md = getattr(d, 'metadata', {}) or {}
        if not txt.strip():
            continue
        parts = _semantic_chunk_text(txt)
        for p in parts:
            out.append(LCDocument(page_content=p, metadata=dict(md)))
    # Respect overlap by lightly merging very small trailing chunks
    return out


def _is_probable_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    # All caps or ends with ':' or typical keywords
    if s.endswith(":"):
        return True
    up = re.sub(r"[^A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỵ ]", "", s)
    if up and up == s.upper() and len(s) <= 80:
        return True
    keywords = ["khu vực", "phụ cấp", "bảng", "table", "mức", "đơn giá", "giá", "phụ lục"]
    return any(k in s.lower() for k in keywords)


def _layout_aware_chunk_blocks(blocks: list[LCDocument]) -> list[LCDocument]:
    chunks: list[LCDocument] = []
    # Track latest heading/section to attach as metadata
    current_section = None
    buf_text = ""
    buf_meta: dict | None = None
    def flush_text():
        nonlocal buf_text, buf_meta
        if buf_text.strip():
            text = buf_text
            while len(text) > DEFAULT_CHUNK_SIZE:
                cut = DEFAULT_CHUNK_SIZE
                m = list(re.finditer(r"\n\n", text[:DEFAULT_CHUNK_SIZE]))
                if m:
                    cut = m[-1].start()
                part = text[:cut].strip()
                if part:
                    meta_out = dict(buf_meta or {})
                    if current_section and not meta_out.get('section'):
                        meta_out['section'] = current_section
                    chunks.append(LCDocument(page_content=part, metadata=meta_out))
                text = text[cut - min(DEFAULT_CHUNK_OVERLAP, 80):]
            if text.strip():
                meta_out = dict(buf_meta or {})
                if current_section and not meta_out.get('section'):
                    meta_out['section'] = current_section
                chunks.append(LCDocument(page_content=text.strip(), metadata=meta_out))
        buf_text = ""
        buf_meta = None

    for b in blocks:
        md = getattr(b, 'metadata', {}) or {}
        btype = md.get('block_type')
        content = getattr(b, 'page_content', '') or ''
        if btype == 'table':
            # flush current text, then keep table as a single chunk
            flush_text()
            if current_section and not md.get('section'):
                md = {**md, 'section': current_section}
            chunks.append(LCDocument(page_content=content, metadata=md))
            continue
        # text block: split into paragraphs; keep heading lines with following paragraph
        paras = _split_paragraphs(content)
        for p in paras:
            first_line = p.strip().splitlines()[0] if p.strip() else ""
            if _is_probable_heading(first_line):
                # finalize previous section and update current_section
                flush_text()
                current_section = first_line.strip(':').strip()
                buf_meta = {**md, 'section': current_section}
                buf_text = p.strip() + "\n\n"
            else:
                if not buf_meta:
                    buf_meta = dict(md)
                    if current_section:
                        buf_meta['section'] = current_section
                if len(buf_text) + len(p) > DEFAULT_CHUNK_SIZE * 1.2:
                    flush_text()
                    buf_meta = dict(md)
                    if current_section:
                        buf_meta['section'] = current_section
                buf_text += p.strip() + "\n\n"
    flush_text()
    return chunks


def extract_pdf_blocks_with_tables(file_path: str, base_metadata: Optional[dict] = None) -> list[LCDocument]:
    """Extract PDF into blocks preserving tables when possible using pdfplumber.
    Returns a list of LangChain Documents with metadata including page, block_index, block_type.
    """
    documents: list[LCDocument] = []
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(file_path) as pdf:
            for pidx, page in enumerate(pdf.pages):
                md_base = {"source": os.path.basename(file_path), "type": "pdf", "page": pidx + 1}
                if base_metadata:
                    md_base.update(base_metadata)
                # 1) tables first (so we can remove them from text if needed)
                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                # 2) text
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                # Emit text block
                if text.strip():
                    documents.append(LCDocument(page_content=text.strip(), metadata={**md_base, "block_type": "text", "block_index": len(documents)}))
                # Emit tables as TSV blocks
                for tidx, t in enumerate(tables):
                    tsv = _table_to_tsv(t)
                    if tsv.strip():
                        md = {**md_base, "block_type": "table", "table_index": tidx, "block_index": len(documents)}
                        documents.append(LCDocument(page_content=tsv, metadata=md))
    except Exception as e:
        print(f"[PDF] pdfplumber extraction failed: {e}")
        # Fallbacks are handled later in process_file_content
    return documents


def validate_numeric_tables(docs: list[LCDocument]) -> list[LCDocument]:
    """Run simple validations for common admin tables (e.g., allowance delta = new - current)."""
    for d in docs:
        md = getattr(d, 'metadata', {}) or {}
        if md.get('block_type') != 'table':
            continue
        text = (getattr(d, 'page_content', '') or '').lower()
        # detect allowance table by header keywords present in first line
        first_line = text.splitlines()[0] if text else ''
        if all(k in first_line for k in ["khu vực", "phụ cấp", "mới"]):
            # try to find numbers on subsequent lines; if delta present, check consistency
            import math
            num = lambda s: int(re.sub(r"[^0-9]", "", s)) if re.search(r"\d", s) else None
            for line in text.splitlines()[1:]:
                cols = [c.strip() for c in line.split("\t")]
                if len(cols) < 4:
                    continue
                cur, new, delta = num(cols[-3]), num(cols[-2]), num(cols[-1])
                if cur is not None and new is not None and delta is not None:
                    if new - cur != delta:
                        md.setdefault('validation_warnings', []).append({
                            'type': 'delta_mismatch', 'line': line, 'expected': new - cur, 'found': delta,
                        })
                    d.metadata = md
    return docs

    embeddings = None

def is_scanned_pdf(file_path: str) -> bool:
    """Check if a PDF is likely scanned by checking for text on the first page."""
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            if not pdf.pages:
                return False  # Empty PDF is not considered scanned
            first_page = pdf.pages[0]
            text = first_page.extract_text() or ""
            # Heuristic: if very few characters are found, it's probably an image/scan.
            if len(text.strip()) < 50:
                print(f"[PDF Classify] PDF '{os.path.basename(file_path)}' is likely SCANNED.")
                return True
            print(f"[PDF Classify] PDF '{os.path.basename(file_path)}' is TEXT-BASED.")
            return False
    except Exception as e:
        print(f"[PDF Classify] Error checking PDF type, assuming text-based: {e}")
        return False

def extract_pdf_tables_with_camelot(file_path: str, base_metadata: Optional[dict] = None) -> list[LCDocument]:
    """Extract tables from a PDF using Camelot for higher accuracy."""
    documents: list[LCDocument] = []
    try:
        import camelot
        # Use lattice mode for tables with clear grid lines, which is common in business docs.
        tables = camelot.read_pdf(file_path, pages='all', flavor='lattice', suppress_stdout=True)
        if not tables:
            print("[Camelot] No tables found with lattice, trying stream mode.")
            tables = camelot.read_pdf(file_path, pages='all', flavor='stream', suppress_stdout=True)

        for t in tables:
            tsv = t.df.to_csv(sep='\t', index=False, header=True)
            if tsv.strip():
                md_base = {"source": os.path.basename(file_path), "type": "pdf", "page": t.page}
                if base_metadata:
                    md_base.update(base_metadata)
                md = {**md_base, "block_type": "table", "extraction_method": "camelot"}
                documents.append(LCDocument(page_content=tsv, metadata=md))
        print(f"[Camelot] Extracted {len(documents)} tables from '{os.path.basename(file_path)}'.")
    except Exception as e:
        print(f"[Camelot] Table extraction failed: {e}")
    return documents

def process_file_content(db: Session, file_path: str, file_type: str, base_metadata: Optional[dict] = None) -> List[LCDocument]:
    """Process file content and split into chunks"""
    try:
        documents: List[object] = []
        if file_type == "pdf":
            # 1. Classify PDF as text-based or scanned
            scanned = is_scanned_pdf(file_path)

            # 2. Process based on type
            if not scanned:
                # --- TEXT-BASED PDF PIPELINE (CAMELOT + PDFPLUMBER) ---
                # 1. Prioritize table extraction with Camelot
                camelot_tables = extract_pdf_tables_with_camelot(file_path, base_metadata)
                documents.extend(camelot_tables)

                # 2. Get text blocks from pdfplumber, excluding its tables to avoid duplication
                plumber_blocks = extract_pdf_blocks_with_tables(file_path, base_metadata)
                text_blocks = [b for b in plumber_blocks if getattr(b, 'metadata', {}).get('block_type') != 'table']
                documents.extend(text_blocks)

                # 3. If we have any content, chunk and return it
                if documents:
                    validated_docs = validate_numeric_tables(documents)
                    ingest_allowance_tables_to_sql(db, validated_docs)
                    chunks = _layout_aware_chunk_blocks(validated_docs)
                    for i, ch in enumerate(chunks):
                        md = getattr(ch, 'metadata', {}) or {}
                        md.update({'chunk_index': i})
                        ch.metadata = md
                    if chunks:
                        return chunks  # Return early with high-quality extracted content

            # --- SCANNED PDF / FALLBACK OCR PIPELINE ---
            if os.getenv("ENABLE_OCR", "1") == "1":
                try:
                    import pypdfium2 as pdfium  # type: ignore
                    pieces: List[str] = []
                    pages_md: List[dict] = []
                    doc = pdfium.PdfDocument(file_path)
                    for idx, page in enumerate(doc):
                        bmp = page.render(scale=2.0)
                        pil_img = bmp.to_pil()
                        proc = preprocess_image(pil_img)
                        ocr_text, conf = ocr_with_confidence(proc)
                        # Try detect simple tables and keep as extra block if found
                        tsv = detect_tsv_from_ocr_lines(ocr_text)
                        if tsv:
                            md_table = {"source": os.path.basename(file_path), "type": "pdf", "page": idx + 1, "block_type": "table", "ocr_confidence": conf}
                            if base_metadata:
                                md_table.update(base_metadata)
                            documents.append(LCDocument(page_content=tsv, metadata=md_table))
                        if ocr_text and ocr_text.strip():
                            pieces.append(ocr_text.strip())
                            pages_md.append({"page": idx + 1, "ocr_confidence": conf})

                    if pieces:
                        # Build per-page text documents from OCR results
                        if not documents: # Only add text blocks if no tables were found for same page
                            if pages_md and len(pages_md) == len(pieces):
                                for i, t in enumerate(pieces):
                                    md = {"source": os.path.basename(file_path), "type": "pdf", "page": pages_md[i]["page" ], "block_type": "text", "ocr_confidence": pages_md[i]["ocr_confidence"]}
                                    if base_metadata:
                                        md.update(base_metadata)
                                    documents.append(LCDocument(page_content=t, metadata=md))
                            else:
                                text = "\n".join(pieces).strip()
                                md = {"source": os.path.basename(file_path), "type": "pdf", "block_type": "text"}
                                if base_metadata:
                                    md.update(base_metadata)
                                documents.append(LCDocument(page_content=text, metadata=md))
                except Exception as ocr_err:
                    print(f"[OCR] PDF OCR failed: {ocr_err}")

            # OpenAI Vision fallback for PDFs (when OCR/text is empty)
            if (not documents) and os.getenv("ENABLE_VISION_OCR_FALLBACK", "0") == "1" and USE_OPENAI and openai_client:
                try:
                    import base64, io
                    import pypdfium2 as pdfium  # type: ignore
                    pages_text: list[tuple[int, str]] = []
                    doc = pdfium.PdfDocument(file_path)
                    max_pages = min(len(doc), int(os.getenv("VISION_OCR_MAX_PAGES", "3")))
                    model = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
                    for idx in range(max_pages):
                        page = doc[idx]
                        bmp = page.render(scale=2.0)
                        pil_img = bmp.to_pil()
                        buf = io.BytesIO(); pil_img.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        img_url = f"data:image/png;base64,{b64}"
                        prompt_text = ("Hãy trích xuất toàn bộ chữ/bảng từ trang PDF dưới dạng văn bản tiếng Việt. "
                                       "Giữ nguyên số liệu và cấu trúc bảng đơn giản nếu có.")
                        vis = openai_client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": img_url}}
                            ]}],
                            temperature=0.0,
                        )
                        content = (vis.choices[0].message.content or "").strip()
                        pages_text.append((idx + 1, content))
                    # Build documents
                    texts = [t for _, t in pages_text if t]
                    if texts:
                        documents = []
                        for pg, t in pages_text:
                            if not t:
                                continue
                            md = {"source": os.path.basename(file_path), "type": "pdf", "page": pg}
                            if base_metadata:
                                md.update(base_metadata)
                            documents.append(LCDocument(page_content=t, metadata=md))
                        if not documents:
                            combined = "\n".join(texts).strip()
                            if combined:
                                md = {"source": os.path.basename(file_path), "type": "pdf"}
                                if base_metadata:
                                    md.update(base_metadata)
                                documents = [LCDocument(page_content=combined, metadata=md)]
                        print("[OCR] Used OpenAI Vision fallback for PDF")
                except Exception as _ve:
                    print(f"[OCR] Vision fallback failed for PDF: {_ve}")

            if not documents:
                # Fallback: placeholder document if everything failed
                text = text or f"Tài liệu PDF: {os.path.basename(file_path)}\nKhông thể trích xuất nội dung text tự động. Có thể là file scan hoặc PDF được bảo vệ."
                print(f"[WARNING] Could not extract text from PDF: {file_path}")
                md = {"source": os.path.basename(file_path), "type": "pdf"}
                if base_metadata:
                    md.update(base_metadata)
                documents = [LCDocument(page_content=text, metadata=md)]
        elif file_type in ("txt", "docx", "doc"):
            content = ""
            if file_type == "txt":
                try:
                    import chardet
                    with open(file_path, "rb") as f:
                        result = chardet.detect(f.read())
                        encoding = result['encoding'] or 'utf-8'
                    with open(file_path, "r", encoding=encoding, errors='ignore') as f:
                        content = f.read()
                except Exception:
                    raise HTTPException(status_code=400, detail="Could not read TXT file. Ensure it's a valid text file.")
            elif HAS_DOCX and Docx2txtLoader is not None:
                loader = Docx2txtLoader(file_path)
                content = "\n\n".join([doc.page_content for doc in loader.load()])

            if not content.strip():
                documents = []
            else:
                # Use the same layout-aware logic as PDFs for high-quality text chunking
                paragraphs = _split_paragraphs(content)
                doc_blocks = []
                md = {"source": os.path.basename(file_path), "type": file_type, "block_type": "text"}
                if base_metadata:
                    md.update(base_metadata)
                for p in paragraphs:
                    doc_blocks.append(LCDocument(page_content=p, metadata=md))

                # Now, apply the intelligent chunking
                chunks = _layout_aware_chunk_blocks(doc_blocks)
                for i, ch in enumerate(chunks):
                    md_chunk = getattr(ch, 'metadata', {}) or {}
                    md_chunk.update({'chunk_index': i})
                    ch.metadata = md_chunk
                return chunks # Return early, skipping the generic splitter

        elif file_type in ("png", "jpg", "jpeg"):
            try:
                from PIL import Image
                pil_img = Image.open(file_path)
                # Preprocess image to improve OCR quality
                processed_img = preprocess_image(pil_img)
                # OCR with confidence scoring
                text, confidence = ocr_with_confidence(processed_img)

                if (not text or not text.strip()) and os.getenv("ENABLE_VISION_OCR_FALLBACK", "0") == "1" and USE_OPENAI and openai_client:
                    # Vision fallback for images
                    import base64
                    with open(file_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    img_url = f"data:image/*;base64,{b64}"
                    prompt_text = ("Hãy trích xuất mọi chữ/bảng chính từ ảnh dưới dạng văn bản tiếng Việt.")
                    model = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
                    vis = openai_client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": img_url}}
                        ]}],
                        temperature=0.0,
                    )
                    text = clean_ocr_text((vis.choices[0].message.content or "").strip())
                    confidence = 99  # Assume high confidence for Vision API

                # Detect tables in OCR output
                documents = []
                tsv_table = detect_tsv_from_ocr_lines(text)
                md = {"source": os.path.basename(file_path), "type": "image", "ocr_confidence": confidence}
                if base_metadata:
                    md.update(base_metadata)

                if tsv_table:
                    md_tbl = {**md, "block_type": "table"}
                    documents.append(LCDocument(page_content=tsv_table, metadata=md_tbl))
                if text:
                    md_txt = {**md, "block_type": "text"}
                    documents.append(LCDocument(page_content=text, metadata=md_txt))

                if not documents:
                     documents = [LCDocument(page_content="", metadata=md)] # Ensure at least one doc

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Image processing error: {e}. Hint: Install Tesseract/OpenCV.")
        elif file_type in ("xlsx", "xls"):
            # Excel reader (all sheets) -> text
            try:
                import pandas as pd  # type: ignore
                engine = "openpyxl" if file_type == "xlsx" else "xlrd"
                try:
                    xls = pd.read_excel(file_path, sheet_name=None, engine=engine, dtype=str)
                except Exception:
                    # Fallback: let pandas auto-detect engine
                    xls = pd.read_excel(file_path, sheet_name=None, dtype=str)
                pieces: list[str] = []
                for sheet, df in (xls or {}).items():
                    if df is None:
                        continue
                    # Fill NaN with empty and build TSV-like text
                    df = df.fillna("")
                    header = "\t".join([str(c) for c in df.columns.tolist()])
                    rows = ["\t".join([str(v) for v in row]) for row in df.values.tolist()]
                    sheet_text = f"[Sheet: {sheet}]\n" + header + "\n" + "\n".join(rows)
                    pieces.append(sheet_text)
                text = ("\n\n".join(pieces)).strip()
                if not text:
                    text = f"Bảng tính rỗng: {os.path.basename(file_path)}"
                md = {"source": os.path.basename(file_path), "type": "excel"}
                if base_metadata:
                    md.update(base_metadata)
                documents = [LCDocument(page_content=text, metadata=md)]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Excel parse error: {e}. Hãy cài pandas + openpyxl (và xlrd cho .xls)")
        elif file_type == "csv":
            try:
                import pandas as pd  # type: ignore
                # Try utf-8 first then cp1258
                try:
                    df = pd.read_csv(file_path, dtype=str)
                except Exception:
                    df = pd.read_csv(file_path, dtype=str, encoding="cp1258", sep=None, engine="python")
                df = df.fillna("")
                header = "\t".join([str(c) for c in df.columns.tolist()])
                rows = ["\t".join([str(v) for v in row]) for row in df.values.tolist()]
                text = header + "\n" + "\n".join(rows)
                md = {"source": os.path.basename(file_path), "type": "csv"}
                if base_metadata:
                    md.update(base_metadata)
                documents = [LCDocument(page_content=text, metadata=md)]
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Split documents into chunks using semantic chunking first
        chunks = _semantic_chunk_documents(documents)
        if not chunks:
            # Fallback to character-based splitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)
        # Gắn metadata chi tiết theo chunk
        for i, ch in enumerate(chunks):
            md = getattr(ch, 'metadata', {}) or {}
            md.update({'chunk_index': i})
            ch.metadata = md
        if not chunks:
            raise HTTPException(status_code=400, detail="Không trích xuất được nội dung (kết quả rỗng).")
        return chunks

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

# ---------------- CRM structured ingestion helpers ----------------
import json
import unicodedata
from sqlalchemy.orm import Session
from database import CrmProduct

def _normalize_vi(s: str) -> str:
    try:
        s = unicodedata.normalize('NFKD', s or '')
        s = ''.join([c for c in s if not unicodedata.combining(c)])
        return ''.join(ch for ch in s.lower() if ch.isalnum() or ch in [' ', '_', '-', '.', ':']).strip()
    except Exception:
        return (s or '').lower().strip()


# Import robust OCR utilities (Vietnamese-aware)
from services.ocr_utils import clean_ocr_text, preprocess_image, ocr_with_confidence, detect_tsv_from_ocr_lines  # type: ignore

# Map possible header names to canonical fields (normalized to ASCII by _normalize_vi)
_KW_SKU = {
    "ma", "ma sp", "ma san pham", "sku", "product code", "code",
    "ma hang", "ma hang hoa", "item code"
}
_KW_NAME = {
    "ten", "ten sp", "ten san pham", "name", "product name", "title",
    "ten hang", "ten hang hoa", "dien giai", "mo ta"
}
_KW_PRICE = {
    "gia", "gia ban", "price", "unit price", "don gia", "gia vnd"
}
_KW_CURRENCY = {"tien te", "currency", "don vi tien", "vnd", "vnd."}
_KW_CATEGORY = {"danh muc", "category", "loai", "nhom"}
_KW_DESC = {"mo ta", "mo ta san pham", "description", "dien giai", "ghi chu", "note"}


def _map_columns(df_columns: list[str]) -> dict:
    mapped = {"sku": None, "name": None, "price": None, "currency": None, "category": None, "description": None}
    norm_cols = {_normalize_vi(c): c for c in df_columns}
    for keyset, field in [(_KW_SKU, "sku"), (_KW_NAME, "name"), (_KW_PRICE, "price"), (_KW_CURRENCY, "currency"), (_KW_CATEGORY, "category"), (_KW_DESC, "description")]:
        for k in keyset:
            if k in norm_cols:
                mapped[field] = norm_cols[k]
                break
    return mapped



def _auto_header_dataframe(df):
    """Try to detect a proper header row when the first row is not header.
    Heuristic: within first 10 rows, find a row that contains at least two header-like tokens
    matching any of our known Vietnamese/English synonyms.
    Returns a new DataFrame with that row as header; falls back to original df.
    """
    try:
        import pandas as pd
        if df is None or df.empty:
            return df
        # If columns already look good (not many 'Unnamed:'), keep
        colnames = [str(c or "").strip().lower() for c in df.columns.tolist()]
        unnamed = sum(1 for c in colnames if c.startswith("unnamed") or c == "")
        if unnamed <= max(1, len(colnames)//3):
            return df
        # Build set of known header keywords
        known = set()
        for s in (_KW_SKU | _KW_NAME | _KW_PRICE | _KW_CURRENCY | _KW_CATEGORY | _KW_DESC):
            known.add(s)
        # Scan first up to 10 rows
        head = df.head(10)
        for i, row in enumerate(head.values.tolist()):
            toks = []
            for cell in row:
                try:
                    toks.append(_normalize_vi(str(cell)))
                except Exception:
                    continue
            hits = sum(1 for t in toks if t in known)
            if hits >= 2:
                # use this row as header
                new_cols = [str(c) for c in df.iloc[i].tolist()]
                rest = df.iloc[i+1:].copy()
                rest.columns = new_cols
                # strip whitespace column names
                rest.columns = [c.strip() for c in rest.columns]
                return rest
        return df
    except Exception as _ah_err:
        print(f"[Excel] auto header detection failed: {_ah_err}")
        return df


def _heuristic_map_columns(df) -> dict:
    """Fallback mapping when header-based mapping fails.
    - Picks a likely 'name' column: longest average text, low numeric ratio
    - Picks a likely 'price' column: highest numeric-like ratio
    Returns a partial colmap like {"name": colname, "price": colname} using original column names.
    """
    try:
        import pandas as pd
        cand = {"name": None, "price": None}
        if df is None or df.empty:
            return cand
        cols = list(df.columns)
        def _num_like(v: str) -> bool:
            s = str(v or "").strip()
            return any(ch.isdigit() for ch in s)
        best_name, best_len = None, -1.0
        best_price, best_ratio = None, -1.0
        sample = df.head(200)
        for c in cols:
            vals = sample[c].astype(str).tolist()
            num_ratio = sum(1 for v in vals if _num_like(v)) / max(1, len(vals))
            avg_len = sum(len(str(v)) for v in vals) / max(1, len(vals))
            # name: long text, not dominated by numbers
            if avg_len > best_len and num_ratio < 0.6:
                best_len, best_name = avg_len, c
            # price: dominated by numbers
            if num_ratio > best_ratio and num_ratio > 0.4:
                best_ratio, best_price = num_ratio, c
        if best_name:
            cand["name"] = best_name
        if best_price and best_price != best_name:
            cand["price"] = best_price
        return cand
    except Exception as _he:
        print(f"[CRM] Heuristic column mapping failed: {_he}")
        return {"name": None, "price": None}


def ingest_dataframe_to_crm(db: Session, df, source_filename: str) -> int:
    inserted = 0
    try:
        if df is None or df.empty:
            print("[CRM] Empty DataFrame – nothing to ingest")
            return 0
        # Try to detect header if columns look wrong
        df = _auto_header_dataframe(df)
        df = df.fillna("")
        print(f"[CRM] Incoming columns: {list(df.columns)}")
        colmap = _map_columns(df.columns.tolist())
        print(f"[CRM] Initial column mapping: {colmap}")
        # If we still can't find sku/name columns, try using the first row as header once
        if not (colmap.get("sku") or colmap.get("name")) and len(df) > 1:
            try:
                new_cols = [str(c) for c in df.iloc[0].tolist()]
                df2 = df.iloc[1:].copy()
                df2.columns = [c.strip() for c in new_cols]
                df2 = df2.fillna("")
                colmap2 = _map_columns(df2.columns.tolist())
                print(f"[CRM] After first-row header attempt, mapping: {colmap2}")
                if colmap2.get("sku") or colmap2.get("name"):
                    print(f"[CRM] Header fixed by first-row detection: {colmap2}")
                    df, colmap = df2, colmap2
                else:
                    # As a last resort, use heuristic mapping to at least get name/price
                    hmap = _heuristic_map_columns(df2)
                    print(f"[CRM] Heuristic mapping: {hmap}")
                    if hmap.get("name"):
                        # Merge heuristic fields into colmap using existing column names
                        for k in ["name", "price"]:
                            if hmap.get(k):
                                colmap[k] = hmap[k]
                        df = df2
            except Exception as _hdr_err:
                print(f"[CRM] First-row header attempt failed: {_hdr_err}")
        # If still no sku and no name, try heuristic on current df
        if not (colmap.get("sku") or colmap.get("name")):
            hmap = _heuristic_map_columns(df)
            print(f"[CRM] Heuristic on original df: {hmap}")
            if hmap.get("name"):
                for k in ["name", "price"]:
                    if hmap.get(k):
                        colmap[k] = hmap[k]
        print(f"[CRM] Final column mapping used: {colmap}")
        for idx, row in enumerate(df.to_dict(orient="records")):
            data = {k: (row.get(v) if v else "") for k, v in colmap.items()}
            # Only save rows that have at least sku or name
            if not (str(data.get("sku") or "").strip() or str(data.get("name") or "").strip()):
                continue
            prod = CrmProduct(
                source_filename=source_filename,
                row_index=idx,
                sku=str(data.get("sku") or "").strip(),
                name=str(data.get("name") or "").strip(),
                price=str(data.get("price") or "").strip(),
                currency=str(data.get("currency") or "").strip(),
                category=str(data.get("category") or "").strip(),
                description=str(data.get("description") or "").strip(),
                attributes=json.dumps(row, ensure_ascii=False)
            )
            db.add(prod)
            inserted += 1
        if inserted:
            db.commit()
    except Exception as e:
        print(f"[Files] CRM ingest failed: {e}")
        db.rollback()
        inserted = 0
    return inserted

def _normalize_allowance_value(s: str) -> Optional[int]:
    """Convert currency string like '650.000d' or '1,200,000' to integer."""
    if not s or not isinstance(s, str):
        return None
    # Remove currency symbols, dots, and whitespace, then convert to int
    cleaned = re.sub(r"[.,đd\s]", "", s.lower())
    if cleaned.isdigit():
        return int(cleaned)
    return None

def ingest_allowance_tables_to_sql(db: Session, docs: list[LCDocument]):
    """Parse allowance table documents and save structured data to SQL."""
    inserted_count = 0
    for doc in docs:
        md = getattr(doc, 'metadata', {})
        if md.get('block_type') != 'table':
            continue

        content = getattr(doc, 'page_content', '')
        lines = content.strip().split('\n')
        if len(lines) < 2:
            continue

        header = [h.lower().strip() for h in lines[0].split('\t')]
        # Heuristic to find the columns we need
        try:
            khu_vuc_idx = header.index("khu vực")
            cu_idx = header.index("phụ cấp cũ")
            moi_idx = header.index("phụ cấp mới")
            tang_idx = header.index("mức tăng")
        except ValueError:
            # If header doesn't match exactly, skip this table for structured ingestion
            continue

        for row_str in lines[1:]:
            cols = row_str.split('\t')
            if len(cols) <= max(khu_vuc_idx, cu_idx, moi_idx, tang_idx):
                continue

            khu_vuc = cols[khu_vuc_idx].strip()
            phu_cap_cu = _normalize_allowance_value(cols[cu_idx])
            phu_cap_moi = _normalize_allowance_value(cols[moi_idx])
            muc_tang = _normalize_allowance_value(cols[tang_idx])

            if khu_vuc and phu_cap_moi is not None:
                entry = AllowanceTable(
                    source_filename=md.get('filename', 'unknown'),
                    page=md.get('page'),
                    khu_vuc=khu_vuc,
                    phu_cap_cu=phu_cap_cu,
                    phu_cap_moi=phu_cap_moi,
                    muc_tang=muc_tang,
                    raw_row_text='\t'.join(cols)
                )
                db.add(entry)
                inserted_count += 1

    if inserted_count > 0:
        try:
            db.commit()
            print(f"[SQL Ingest] Inserted {inserted_count} rows into allowance_tables.")
        except Exception as e:
            print(f"[SQL Ingest] Error committing allowance data: {e}")
            db.rollback()
    return inserted_count


@router.post("/upload", response_model=DocumentResponse)
async def upload_file(
    file: UploadFile = File(...),
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Upload a file and process it for ChromaDB (immediate indexing)."""

    print(f"[Upload] Start upload: filename='{file.filename}', by={admin_user.username}")

    # Validate file type (by extension + mime)
    allowed_types = ["pdf", "txt", "doc", "docx", "png", "jpg", "jpeg", "xlsx", "xls", "csv"]
    excel_mimes = {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/octet-stream",  # some browsers send this for .xlsx
    }
    file_extension = (file.filename or "").split(".")[-1].lower()
    if file_extension not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed types: {', '.join(allowed_types)}"
        )
    # If it's an Excel ext but mime looks odd, just warn; do not block
    try:
        ct = getattr(file, "content_type", None)
        if file_extension in ("xlsx", "xls") and ct and ct not in excel_mimes:
            print(f"[Upload][Warn] Unexpected Excel MIME: {ct} for {file.filename}")
    except Exception:
        pass
    print(f"[Upload] Detected type={file_extension}")

    # Validate metadata early
    ok, reason = validate_meta(file.filename or "", getattr(file, "content_type", None), 0)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    # Create uploads directory if it doesn't exist
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save file (stream to disk for large files)
    file_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{file.filename}")
    file_bytes = b""
    try:
        total_written = 0
        CHUNK = 1024 * 1024  # 1MB
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                buffer.write(chunk)
                total_written += len(chunk)
                # Keep small files in memory for AV/S3; cap at 30MB to avoid memory blowup
                if len(file_bytes) < 30 * 1024 * 1024:
                    file_bytes += chunk
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    # Re-validate with actual size and antivirus
    ok, reason = validate_meta(file.filename or "", getattr(file, "content_type", None), total_written)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)
    clean, note = antivirus_scan_bytes(file_bytes if file_bytes else open(file_path, 'rb').read())
    if not clean:
        # Delete saved file if unsafe
        try:
            os.remove(file_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"Upload rejected by antivirus: {note}")



    # === Background Data Pipeline ===
    try:
        # Save document record with pending status
        document = Document(
            id=str(uuid.uuid4()),
            filename=file.filename,
            file_path=file_path,
            file_type=file_extension,
            uploaded_by=admin_user.id,
            file_size=total_written,
            status="pending"
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        # Enqueue the processing job
        enqueue_file_processing(file_path, document.id)

        return DocumentResponse(
            id=document.id,
            filename=document.filename,
            file_type=document.file_type,
            uploaded_by=admin_user.username,
            uploaded_at=document.uploaded_at.isoformat(),
            file_size=document.file_size,
            status=document.status
        )
    except Exception as e:
        # Clean up file if DB operation fails
        try:
            os.remove(file_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Error enqueuing file for processing: {e}")

@router.post("/upload-chunk")
async def upload_chunk(
    upload_id: str = Form(...),
    index: int = Form(...),
    total: int = Form(...),
    filename: str = Form(...),
    file: UploadFile = File(...),
    admin_user: User = Depends(require_admin),
):
    """Receive a chunk and store on disk. Call /files/upload-chunk/finish to assemble & index.
    This avoids loading large files in memory and allows parallel chunk uploads from the client.
    """
    # Basic validation
    if not upload_id or total <= 0 or index < 0 or index >= total:
        raise HTTPException(status_code=400, detail="Invalid chunk parameters")

    chunk_dir = os.path.join("uploads", ".chunks", upload_id)
    os.makedirs(chunk_dir, exist_ok=True)
    # Persist chunk
    chunk_path = os.path.join(chunk_dir, f"{index:06d}.part")
    try:
        CHUNK = 1024 * 1024
        with open(chunk_path, "wb") as out:
            while True:
                data = await file.read(CHUNK)
                if not data:
                    break
                out.write(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving chunk: {e}")

    return {"ok": True, "upload_id": upload_id, "index": index, "total": total}


@router.post("/upload-chunk/finish", response_model=DocumentResponse)
async def finish_upload(
    upload_id: str = Form(...),
    filename: str = Form(...),
    total: int = Form(...),
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Assemble uploaded chunks and index the file like /files/upload."""
    chunk_dir = os.path.join("uploads", ".chunks", upload_id)
    if not os.path.isdir(chunk_dir):
        raise HTTPException(status_code=400, detail="Upload session not found")

    # Assemble
    final_path = os.path.join("uploads", f"{uuid.uuid4()}_{filename}")
    try:
        with open(final_path, "wb") as dest:
            for i in range(int(total)):
                part_path = os.path.join(chunk_dir, f"{i:06d}.part")
                if not os.path.exists(part_path):
                    raise HTTPException(status_code=400, detail=f"Missing chunk {i}")
                with open(part_path, "rb") as src:
                    dest.write(src.read())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error assembling chunks: {e}")
    finally:
        # Cleanup chunk files
        try:
            for name in os.listdir(chunk_dir):
                try:
                    os.remove(os.path.join(chunk_dir, name))
                except Exception:
                    pass
            os.rmdir(chunk_dir)
        except Exception:
            pass

    # Index assembled file (reuse same pipeline)
    file_extension = filename.split(".")[-1].lower()
    try:
        base_md = {
            "filename": filename,
            "uploader": admin_user.username,
            "uploaded_at": datetime.utcnow().isoformat(),
            "file_type": file_extension,
        }
        chunks = process_file_content(db, final_path, file_extension, base_metadata=base_md)
    except Exception as e:
        # Clean up file if processing fails
        try:
            os.remove(final_path)
        except Exception:
            pass
        raise e

    # Add to ChromaDB (same as /upload)
    try:
        if not chunks:
            raise HTTPException(status_code=400, detail="Không có nội dung nào để lưu (chunks trống)")
        texts = [getattr(ch, 'page_content', '') for ch in chunks]
        metadatas = [getattr(ch, 'metadata', {}) for ch in chunks]
        chunk_embeddings: list[list[float]] = []
        backend = "openai" if (USE_OPENAI and openai_client) else "ollama"
        if USE_OPENAI and openai_client:
            print(f"[Index] Using OpenAI embeddings model={OPENAI_EMBED_MODEL}")
            for text in texts:
                resp = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
                vec = resp.data[0].embedding
                chunk_embeddings.append(vec)
        else:
            if embeddings:
                print("[Index] Using OllamaEmbeddings backend")
                for text in texts:
                    vec = embeddings.embed_query(text)
                    chunk_embeddings.append(vec)
            elif _st_model is not None:
                print("[Index] Using sentence-transformers fallback for embeddings")
                chunk_embeddings = [_st_model.encode(t).tolist() for t in texts]
                backend = "st"
            else:
                raise Exception("Embeddings backend is not available: cần Ollama/OpenAI hoặc cài sentence-transformers")
        emb_dim = len(chunk_embeddings[0]) if chunk_embeddings else None
        collection = get_chroma_collection_for_backend(backend, emb_dim) or get_chroma_collection()
        coll_name = getattr(collection, 'name', 'unknown') if collection else 'unknown'
        print(f"[Index] Adding {len(texts)} chunks to collection='{coll_name}' backend={backend} dim={emb_dim}")
        ids = [str(uuid.uuid4()) for _ in chunks]
        if not ids:
            raise HTTPException(status_code=400, detail="Không tạo được ID cho tài liệu")
        for md in metadatas:
            md["backend"] = backend
            md["collection"] = coll_name
        if not collection:
            raise HTTPException(status_code=500, detail="ChromaDB collection is not available")
        collection.add(
            embeddings=chunk_embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"[Index] Added {len(ids)} chunks to '{coll_name}' successfully")
    except Exception as e:
        try:
            os.remove(final_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Error adding to vector database: {str(e)}")

    # Save document record
    # Save to storage for serving (S3/local) and get URL
    try:
        with open(final_path, 'rb') as fh:
            data = fh.read()
        public_url, _ = save_bytes('docs', filename, data)
        file_url = public_url
    except Exception as _se:
        print(f"[UploadChunk] Storage save failed: {_se}")
        file_url = None

    document = Document(
        id=str(uuid.uuid4()),
        filename=filename,
        file_path=final_path,
        file_type=file_extension,
        uploaded_by=admin_user.id,
        file_size=os.path.getsize(final_path),
        file_url=file_url
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Persist OCR/parsed text into SQL for AI to read directly
    try:
        saved = 0
        for ch in chunks:
            content = getattr(ch, 'page_content', '')
            if not content:
                continue
            norm = _normalize_vi(clean_ocr_text(content))
            md = getattr(ch, 'metadata', {}) or {}
            page = md.get('page') if isinstance(md.get('page'), int) else None
            db.add(OcrText(document_id=document.id,
                           source_filename=document.filename,
                           page=page,
                           chunk_index=md.get('chunk_index'),
                           content=content,
                           normalized_content=norm))
            saved += 1
        if saved:
            db.commit()
            print(f"[Files] Saved {saved} text chunks to SQL (ocr_texts)")
    except Exception as e:
        print(f"[Files] Warning: could not persist OCR text: {e}")
        db.rollback()

    # Structured CRM ingestion (Excel/CSV)
    try:
        ext = file_extension
        inserted_rows = 0
        if ext in ("xlsx", "xls"):
            import pandas as pd  # type: ignore
            try:
                xls = pd.read_excel(final_path, sheet_name=None, dtype=str)
            except Exception:
                xls = pd.read_excel(final_path, sheet_name=None, dtype=str, engine=None)
            for _sheet, _df in (xls or {}).items():
                if _df is not None:
                    inserted_rows += ingest_dataframe_to_crm(db, _df, filename)
        elif ext == "csv":
            import pandas as pd  # type: ignore
            try:
                df = pd.read_csv(final_path, dtype=str)
            except Exception:
                df = pd.read_csv(final_path, dtype=str, encoding="cp1258", sep=None, engine="python")
            inserted_rows += ingest_dataframe_to_crm(db, df, filename)
        if inserted_rows:
            print(f"[Files] CRM ingested {inserted_rows} rows from {filename}")
    except Exception as e:
        print(f"[Files] CRM ingest warning: {e}")

    # Track chunk IDs
    try:
        for cid in ids:
            db.add(DocumentChunk(document_id=document.id, chunk_id=cid, collection=getattr(collection, 'name', 'unknown')))
        db.commit()
    except Exception as track_err:
        print(f"[Files] Warning: could not track chunk IDs: {track_err}")
        db.rollback()

    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        file_type=document.file_type,
        uploaded_by=admin_user.username,
        uploaded_at=document.uploaded_at.isoformat(),
        file_size=document.file_size,
        backend=backend,
        collection=coll_name,
        chunks_indexed=len(ids)
    )


@router.get("/documents", response_model=DocumentListResponse)
async def get_documents(
    page: int = 1,
    limit: int = 10,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Get paginated uploaded documents"""
    try:
        page = max(1, int(page))
        limit = max(1, min(100, int(limit)))

        base_q = db.query(Document)
        total = base_q.count()
        documents = (
            base_q.options(joinedload(Document.uploader))
            .order_by(Document.uploaded_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
            .all()
        )

        items: list[DocumentResponse] = []
        for doc in documents:
            uploader_name = doc.uploader.username if doc.uploader else "Unknown"
            items.append(DocumentResponse(
                id=doc.id,
                filename=doc.filename or "",
                file_type=doc.file_type or "",
                uploaded_by=uploader_name,
                uploaded_at=(doc.uploaded_at.isoformat() if doc.uploaded_at else datetime.utcnow().isoformat()),
                file_size=int(doc.file_size or 0),
                status=(doc.status or "pending")
            ))

        return DocumentListResponse(documents=items, total=int(total or 0))
    except Exception as e:
        import traceback
        print("[Files] get_documents error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class BatchDeleteRequest(BaseModel):
    ids: List[str]

@router.post("/documents/batch-delete")
async def batch_delete_documents(
    request: BatchDeleteRequest,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Batch delete documents by a list of IDs."""
    if not request.ids:
        raise HTTPException(status_code=400, detail="No document IDs provided")

    docs_to_delete = db.query(Document).filter(Document.id.in_(request.ids)).all()
    if not docs_to_delete:
        return {"message": "No matching documents found to delete."}

    collection_map = {}
    paths_to_remove = []
    doc_ids_to_delete = [doc.id for doc in docs_to_delete]

    for doc in docs_to_delete:
        if doc.file_path and os.path.exists(doc.file_path):
            paths_to_remove.append(doc.file_path)
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).all()
        for chunk in chunks:
            if chunk.collection_name not in collection_map:
                collection_map[chunk.collection_name] = []
            collection_map[chunk.collection_name].append(chunk.id)

    for coll_name, chunk_ids in collection_map.items():
        try:
            collection = get_chroma_collection(coll_name)
            if chunk_ids:
                collection.delete(ids=chunk_ids)
                print(f"[Delete] Deleted {len(chunk_ids)} chunks from Chroma collection '{coll_name}'.")
        except Exception as e:
            print(f"[Delete] Error deleting from Chroma collection '{coll_name}': {e}")

    db.query(DocumentChunk).filter(DocumentChunk.document_id.in_(doc_ids_to_delete)).delete(synchronize_session=False)
    db.query(Document).filter(Document.id.in_(doc_ids_to_delete)).delete(synchronize_session=False)

    for path in paths_to_remove:
        try:
            os.remove(path)
            print(f"[Delete] Removed file from disk: {path}")
        except OSError as e:
            print(f"[Delete] Error removing file {path}: {e}")

    db.commit()
    return {"message": f"Successfully deleted {len(doc_ids_to_delete)} documents."}


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete a document and remove from ChromaDB"""

    # Get document
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from ChromaDB precisely using stored chunk IDs
    try:
        # Fetch tracked chunks for this document
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).all()
        if not chunks:
            print("[Files] No chunk mapping found for document; skipping vector delete.")
        else:
            # Group chunk IDs by collection name
            by_collection: dict[str, list[str]] = {}
            for ch in chunks:
                by_collection.setdefault(ch.collection or "", []).append(ch.chunk_id)
            # Connect to Chroma and delete per collection
            chroma_client = chromadb.PersistentClient(path="./chroma_db")
            for cname, ids_to_delete in by_collection.items():
                try:
                    coll = chroma_client.get_collection(name=cname)
                    coll.delete(ids=ids_to_delete)
                    print(f"[Files] Deleted {len(ids_to_delete)} chunks from collection '{cname}'")
                except Exception as ce:
                    print(f"[Files] Could not delete chunks from '{cname}': {ce}")
            # Remove chunk rows from DB
            db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()
            db.commit()
    except Exception as e:
        print(f"Warning: Could not remove from ChromaDB: {e}")
        db.rollback()



class DocumentBulkDeleteRequest(BaseModel):
    ids: List[str]


@router.delete("/documents")
async def bulk_delete_documents(
    payload: DocumentBulkDeleteRequest,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Bulk delete documents by IDs.
    This matches the frontend's DELETE /files/documents with a JSON body {"ids": [...]}
    and avoids Method Not Allowed by explicitly supporting DELETE on /documents.
    """
    deleted: list[str] = []
    failed: dict[str, str] = {}

    for doc_id in (payload.ids or []):
        try:
            document = db.query(Document).filter(Document.id == doc_id).first()
            if not document:
                failed[doc_id] = "not_found"
                continue
            # Remove from ChromaDB
            try:
                chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).all()
                if chunks:
                    by_collection: dict[str, list[str]] = {}
                    for ch in chunks:
                        by_collection.setdefault(ch.collection or "", []).append(ch.chunk_id)
                    chroma_client = chromadb.PersistentClient(path="./chroma_db")
                    for cname, ids_to_delete in by_collection.items():
                        try:
                            coll = chroma_client.get_collection(name=cname)
                            coll.delete(ids=ids_to_delete)
                        except Exception as ce:
                            print(f"[Files] Bulk delete: could not delete chunks from '{cname}': {ce}")
                    # Remove chunk rows
                    db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()
                    db.commit()
            except Exception as e:
                print(f"[Files] Bulk delete chroma warning for {doc_id}: {e}")
                db.rollback()
            # Delete file on disk
            try:
                if document.file_path and os.path.exists(document.file_path):
                    os.remove(document.file_path)
            except Exception as e:
                print(f"[Files] Bulk delete file warning for {doc_id}: {e}")
            # Delete DB record
            db.delete(document)
            db.commit()
            deleted.append(doc_id)
        except Exception as e:
            db.rollback()
            failed[doc_id] = str(e)

    return {"deleted": deleted, "failed": failed, "total_deleted": len(deleted)}

    # Delete file
    try:
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
    except Exception as e:
        print(f"Warning: Could not delete file: {e}")


# Compatibility endpoint for frontend bulk delete
@router.post("/documents/delete")
async def compat_bulk_delete_documents(
    payload: DocumentBulkDeleteRequest,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Frontend sends POST to /documents/delete for bulk actions. Route to the new DELETE handler."""
    return await bulk_delete_documents(payload, admin_user, db)

