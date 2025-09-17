"""
Ingestion pipeline for Excel, PDF, and Images with light structure tagging.
- Excel: read with pandas, clean, and return row-wise JSON records per sheet
- PDF: extract text with PyMuPDF; if scanned, OCR pages via Tesseract
- Image: OCR via Tesseract with preprocessing
- Structure tagging: detect headings, lists, and tables; emit tagged segments

This module is self-contained and can be reused by API endpoints or batch jobs.
"""
from __future__ import annotations
import io
import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

from PIL import Image

from .ocr_utils import (
    preprocess_image,
    ocr_with_confidence,
    detect_tsv_from_ocr_lines,
    clean_ocr_text,
)


# ---------------------- Data Structures ----------------------
@dataclass
class Segment:
    type: str  # 'heading' | 'paragraph' | 'list' | 'table'
    content: str
    level: Optional[int] = None  # for headings only
    page: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class IngestResult:
    text: str  # full plain text
    segments: List[Segment]
    summary: Dict[str, Any]  # metadata such as page_count, avg_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "segments": [asdict(s) for s in self.segments],
            "summary": self.summary,
        }


# ---------------------- Helpers ----------------------
_heading_num_re = re.compile(r"^(\d+(?:[\.)])(?:\d+[\.)])?\s+|^[IVXLC]+\.|^Mục\s+\d+", re.IGNORECASE)
_heading_caps_re = re.compile(r"^[A-ZÀ-Ỹ0-9][A-ZÀ-Ỹ0-9\s\-_/]{5,}$")
_bullet_re = re.compile(r"^\s*(?:[-*•▪►➤]|\d+\)|\d+\.|[a-z]\))\s+")
_table_bar_re = re.compile(r"\|.*\|")


def _is_heading(line: str) -> Tuple[bool, Optional[int]]:
    s = (line or "").strip()
    if not s or len(s) < 3:
        return False, None
    if _heading_num_re.search(s):
        # crude level inference based on number of dots
        lvl = s.split(".")
        level = min(3, max(1, len([x for x in lvl if x.strip().rstrip(")").rstrip('.')])) )
        return True, level
    if _heading_caps_re.match(s) and len(s.split()) <= 12:
        return True, 1
    return False, None


def _is_bullet(line: str) -> bool:
    return bool(_bullet_re.match((line or "").strip()))


def _detect_table_block(lines: List[str], start_idx: int) -> Tuple[Optional[str], int]:
    """Try to detect a table starting at start_idx. Returns (markdown_table, consumed_lines).
    Heuristics:
    - markdown pipes present
    - OCR TSV from spaced alignment
    - CSV-ish comma separated with >= 3 columns (best-effort)
    """
    block = []
    i = start_idx
    # collect up to blank line
    while i < len(lines) and lines[i].strip():
        block.append(lines[i])
        i += 1
    raw = "\n".join(block)

    # 1) Markdown pipe table
    if sum(1 for ln in block if _table_bar_re.search(ln)) >= 2:
        # ensure header separator exists
        hdr, *rest = [ln.strip() for ln in block]
        if not re.search(r"\|\s*-+\s*\|", "\n".join(rest)):
            cols = [c.strip() for c in hdr.strip("|").split("|")]
            sep = "| " + " | ".join(["---"] * len(cols)) + " |"
            raw = "| " + " | ".join(cols) + " |\n" + sep + "\n" + "\n".join(
                ["| " + " | ".join([c.strip() for c in ln.strip("|").split("|")]) + " |" for ln in rest]
            )
        return raw, i - start_idx

    # 2) OCR spaced TSV detection
    tsv = detect_tsv_from_ocr_lines("\n".join(block))
    if tsv:
        rows = [r.split("\t") for r in tsv.splitlines()]
        if rows:
            md = "| " + " | ".join(rows[0]) + " |\n" + "| " + " | ".join(["---"] * len(rows[0])) + " |\n"
            for r in rows[1:]:
                md += "| " + " | ".join(r) + " |\n"
            return md.strip(), i - start_idx

    # 3) CSV-ish heuristic (>=3 columns with commas)
    if sum(1 for ln in block if ln.count(",") >= 2) >= 2:
        def _csv_to_md(csv_text: str) -> str:
            df = pd.read_csv(io.StringIO(csv_text))
            headers = list(df.columns)
            lines = ["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
            for _, row in df.iterrows():
                vals = [str(row[h]) for h in headers]
                lines.append("| " + " | ".join(vals) + " |")
            return "\n".join(lines)
        return _csv_to_md("\n".join(block)), i - start_idx

    return None, 0


def analyze_structure(text: str, page: Optional[int] = None) -> List[Segment]:
    """Split text into tagged segments. Lightweight, heuristic-based.
    - Detect headings (numbered and CAPS)
    - Detect bullet/numbered lists
    - Detect tables (markdown/TSV/CSV-ish)
    - Remaining content becomes paragraphs
    """
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    segs: List[Segment] = []

    i = 0
    buf: List[str] = []

    def flush_para():
        nonlocal buf
        block = "\n".join([b for b in buf if b.strip()]).strip()
        if block:
            segs.append(Segment(type="paragraph", content=block, page=page))
        buf = []

    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            flush_para()
            i += 1
            continue

        # Heading detection
        is_h, lvl = _is_heading(ln)
        if is_h:
            flush_para()
            segs.append(Segment(type="heading", content=ln.strip(), level=lvl or 1, page=page))
            i += 1
            continue

        # List block
        if _is_bullet(ln):
            flush_para()
            items = []
            while i < len(lines) and _is_bullet(lines[i]):
                items.append(re.sub(_bullet_re, "", lines[i]).strip())
                i += 1
            segs.append(Segment(type="list", content="\n".join(["- " + it for it in items]), page=page))
            continue

        # Table block
        table_md, consumed = _detect_table_block(lines, i)
        if consumed >= 2 and table_md:
            flush_para()
            segs.append(Segment(type="table", content=table_md, page=page))
            i += consumed
            continue

        # Default accumulate paragraph
        buf.append(ln)
        i += 1

    flush_para()
    return segs


# ---------------------- Excel Processing ----------------------

def _normalize_header(h: Any) -> str:
    s = str(h or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip("()[]{}:;|#")
    return s


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        # Try numeric
        try:
            df[c] = pd.to_numeric(df[c], errors="ignore")
        except Exception:
            pass
        # Strip strings
        try:
            df[c] = df[c].apply(lambda x: str(x).strip() if isinstance(x, str) else x)
        except Exception:
            pass
    return df


def process_excel(file_bytes: bytes, max_rows_per_sheet: int = 5000) -> Dict[str, Any]:
    """Read an Excel workbook and return structured records by sheet.
    Returns dict: {
        "sheets": [
            {"name": str, "rows": [ {"row_index": int, "data": {...}, "normalized_text": str}, ... ]}
        ],
        "summary": {"sheet_count": int, "row_count": int}
    }
    """
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets_out: List[Dict[str, Any]] = []
    total_rows = 0

    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet, header=0)
        except Exception:
            # Fallback: no header
            raw = pd.read_excel(xls, sheet_name=sheet, header=None)
            # infer header row: the first row with >= 2 non-null cells
            header_row = 0
            for r in range(min(10, len(raw))):
                if raw.iloc[r].notna().sum() >= 2:
                    header_row = r
                    break
            df = pd.read_excel(xls, sheet_name=sheet, header=header_row)

        # Drop fully empty rows/cols
        df = df.dropna(how="all")
        df = df.loc[:, df.notna().sum() > 0]
        df.columns = [_normalize_header(c) or f"Col_{i+1}" for i, c in enumerate(df.columns)]
        df = _coerce_types(df)

        rows_out: List[Dict[str, Any]] = []
        for idx, row in df.iterrows():
            if len(rows_out) >= max_rows_per_sheet:
                break
            data = {str(col): (None if pd.isna(val) else val) for col, val in row.items()}
            # Build normalized text for search (space-separated values)
            flat_values = " ".join([str(v) for v in data.values() if v is not None])
            norm_text = clean_ocr_text(flat_values)
            rows_out.append({
                "row_index": int(idx) + 1,
                "data": data,
                "normalized_text": norm_text,
            })
        total_rows += len(rows_out)
        sheets_out.append({"name": sheet, "rows": rows_out})

    return {
        "sheets": sheets_out,
        "summary": {"sheet_count": len(sheets_out), "row_count": total_rows},
    }


# ---------------------- PDF & Image Processing ----------------------

def _ocr_pil_image(pil: Image.Image) -> Tuple[str, Optional[int]]:
    try:
        pre = preprocess_image(pil)
        text, conf = ocr_with_confidence(pre)
        return text, conf
    except Exception:
        return "", None


def process_pdf(file_bytes: bytes, ocr_on_scanned: bool = True) -> IngestResult:
    """Extract text from a PDF using PyMuPDF; OCR when needed.
    - For pages with little or no text and ocr_on_scanned=True, render to image and OCR.
    - Returns IngestResult with per-page structure tagging.
    """
    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) is not installed")

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    all_text_parts: List[str] = []
    segments: List[Segment] = []
    ocr_pages = 0
    page_confs: List[int] = []

    for i in range(len(doc)):
        page = doc[i]
        txt = page.get_text("text") or ""
        txt = clean_ocr_text(txt)
        used_ocr = False
        conf_val: Optional[int] = None

        if ocr_on_scanned and len(txt.strip()) < 20:
            # Consider scanned: render and OCR at 2x zoom
            try:
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                txt, conf_val = _ocr_pil_image(img)
                used_ocr = True
                ocr_pages += 1
            except Exception:
                pass

        if not txt:
            continue

        all_text_parts.append(txt)
        page_segs = analyze_structure(txt, page=i + 1)
        segments.extend(page_segs)
        if conf_val is not None:
            page_confs.append(conf_val)

    full_text = "\n\n".join(all_text_parts).strip()
    avg_conf = int(sum(page_confs) / len(page_confs)) if page_confs else None
    return IngestResult(
        text=full_text,
        segments=segments,
        summary={
            "pages": len(doc),
            "ocr_pages": ocr_pages,
            "avg_confidence": avg_conf,
        },
    )


def process_image(file_bytes: bytes) -> IngestResult:
    pil = Image.open(io.BytesIO(file_bytes))
    txt, conf = _ocr_pil_image(pil)
    segs = analyze_structure(txt)
    return IngestResult(
        text=txt,
        segments=segs,
        summary={"pages": 1, "ocr_pages": 1, "avg_confidence": conf},
    )


# ---------------------- Orchestrator ----------------------

SUPPORTED_TYPES = {"pdf", "png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff", "xlsx", "xls"}


def ingest_any(filename: str, file_bytes: bytes) -> Dict[str, Any]:
    """Convenience entrypoint that routes by extension and returns a JSON-serializable result.
    For Excel, returns a dict from process_excel(). For PDFs/Images, returns IngestResult.to_dict().
    """
    ext = (filename or "").split(".")[-1].lower()
    if ext in {"xlsx", "xls"}:
        return process_excel(file_bytes)
    if ext == "pdf":
        return process_pdf(file_bytes).to_dict()
    if ext in {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"}:
        return process_image(file_bytes).to_dict()
    raise ValueError(f"Unsupported file type: {ext}")

