"""
OCR utilities focused on Vietnamese business documents.
- Optional OpenCV-based preprocessing (grayscale, threshold, denoise, deskew)
- Post-OCR cleanup
- OCR with confidence via pytesseract.image_to_data
- Simple table detection from monospaced OCR lines
"""
from __future__ import annotations
import os
from typing import Tuple, List, Optional

try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

from PIL import Image, ImageFilter, ImageOps  # type: ignore
import pytesseract  # type: ignore
import re
import unicodedata


import numpy as np



def normalize_vi(s: str) -> str:
    try:
        s = unicodedata.normalize('NFKD', s or '')
        s = ''.join([c for c in s if not unicodedata.combining(c)])
        s = s.lower()
        s = re.sub(r"\s+", " ", s)
        return s.strip()
    except Exception:
        return (s or '').lower().strip()


# Dictionary of common Vietnamese OCR errors and their corrections
VI_CORRECTIONS = {
    # Common errors with diacritics (decomposed form -> composed form)
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
    # Common word misspellings
    "chưong": "chương",
    "giao duc": "giáo dục",
    "qui định": "quy định",
    "qui chế": "quy chế",
    "kí": "ký",
}
# Create a regex pattern for efficient replacement
_vi_correction_re = re.compile(r'\b(' + '|'.join(re.escape(key) for key in VI_CORRECTIONS.keys()) + r')\b', re.IGNORECASE)

def clean_ocr_text(text: str) -> str:
    t = (text or "")
    if not t:
        return t

    # 1. Unicode Normalization to NFC for consistent representation
    t = unicodedata.normalize('NFC', t)

    # 2. Apply Vietnamese-specific corrections
    t = _vi_correction_re.sub(lambda m: VI_CORRECTIONS[m.group(1).lower()], t)

    # 3. Basic whitespace and line break cleaning
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Join hyphenated line breaks: "pho-\n bien" -> "phobien" (handle with care)
    t = re.sub(r"(\w)-[\n\s]+(\w)", r"\1\2", t)
    # Merge line breaks splitting words/sentences
    t = re.sub(r"([0-9A-Za-zÀ-ỹ,])\n([0-9a-zà-ỹ])", r"\1 \2", t)
    # Normalize bullets
    t = t.replace("•", "-").replace("▪", "-").replace("►", "-").replace("➤", "-")
    # Collapse whitespace
    t = re.sub(r"[\t\x0b\x0c]+", " ", t)
    t = re.sub(r"[ ]{2,}", " ", t)
    # Trim trailing spaces on lines
    t = "\n".join([ln.strip() for ln in t.splitlines()])
    return t.strip()


def _deskew_image(img: 'np.ndarray') -> 'np.ndarray':
    """Deskew an image using OpenCV."""
    # Threshold and find contours
    thresh = cv2.bitwise_not(img)
    coords = np.column_stack(np.where(thresh > 0))
    # Get minimal bounding box and its angle
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # Rotate the image to correct for the skew
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(pil_img: Image.Image) -> Image.Image:
    """Enhance image for OCR (deskew, grayscale, denoise, threshold). Uses OpenCV if available."""
    try:
        if HAS_CV2:
            import numpy as np  # type: ignore
            img = np.array(pil_img.convert("RGB"))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # 1. Deskew
            deskewed = _deskew_image(gray)
            # 2. Adaptive threshold for uneven lighting
            thr = cv2.adaptiveThreshold(deskewed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 35, 11)
            # 3. Morphological opening to remove dots/noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
            # 4. Optional slight dilation to connect broken glyphs
            final = cv2.dilate(opened, kernel, iterations=1)
            return Image.fromarray(final)
        else:
            # PIL fallback (no deskew)
            img = pil_img.convert("L")  # grayscale
            img = ImageOps.autocontrast(img)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            return img.point(lambda x: 0 if x < 160 else 255, mode='1').convert('L')
    except Exception as e:
        print(f"[OCR Preprocess] Failed: {e}")
        return pil_img


def ocr_with_confidence(pil_img: Image.Image, lang: str = None) -> Tuple[str, Optional[int]]:
    """Run Tesseract and return (clean_text, avg_confidence0..100).
    Uses pytesseract.image_to_data to compute average confidence among valid tokens.
    """
    try:
        if lang is None:
            lang = os.getenv("OCR_LANG", "vie+eng")
        # Respect TESSERACT_CMD if provided
        tcmd = os.getenv("TESSERACT_CMD") or r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        try:
            if os.path.exists(tcmd):
                from pytesseract import pytesseract as _pt  # type: ignore
                _pt.tesseract_cmd = tcmd
        except Exception:
            pass
        data = pytesseract.image_to_data(pil_img, lang=lang, output_type=pytesseract.Output.DICT)
        # Build text and compute confidence
        words = data.get("text", []) or []
        confs = data.get("conf", []) or []
        lines: List[str] = []
        current_line = 0
        for txt, ln in zip(words, data.get("line_num", [0]*len(words))):
            if ln != current_line:
                if current_line != 0:
                    lines.append(" ".join(buf))
                buf = []
                current_line = ln
            else:
                buf = locals().get('buf', [])
            if txt and txt.strip():
                buf.append(txt.strip())
        if locals().get('buf'):
            lines.append(" ".join(locals()['buf']))
        raw_text = "\n".join([l.strip() for l in lines if l.strip()])
        # Average confidence ignoring -1
        valid = [int(c) for c in confs if str(c).isdigit() and int(c) >= 0]
        avg_conf = int(sum(valid)/len(valid)) if valid else None
        text = clean_ocr_text(raw_text)
        return text, avg_conf
    except pytesseract.TesseractNotFoundError as e:
        print(f"[OCR] FATAL: Tesseract is not installed or configured correctly. {e}")
        # Re-raise this critical error so the caller MUST handle it.
        raise e
    except Exception as e:
        print(f"[OCR] Tesseract failed with a general error: {e}")
        return "", None


def detect_tsv_from_ocr_lines(text: str) -> Optional[str]:
    """Detect simple aligned tables from OCR text by splitting on 2+ spaces.
    Returns TSV string if a table-like structure is detected; otherwise None.
    """
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    # Identify rows that have at least 3 columns split by >=2 spaces
    rows = []
    for ln in lines:
        parts = re.split(r"\s{2,}", ln.strip())
        if len(parts) >= 3 and sum(1 for p in parts if p.strip()) >= 3:
            rows.append([p.strip() for p in parts])
    if len(rows) >= 2:
        # Normalize column count using the mode
        from collections import Counter
        cnt = Counter(len(r) for r in rows)
        col = cnt.most_common(1)[0][0]
        rows = [r[:col] + [""]*(col-len(r)) if len(r) < col else r[:col] for r in rows]
        return "\n".join(["\t".join(r) for r in rows])
    return None

