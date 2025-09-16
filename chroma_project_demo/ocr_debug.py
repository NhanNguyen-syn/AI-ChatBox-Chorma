#!/usr/bin/env python3
"""
Quick OCR debug for scanned PDFs on Windows/macOS/Linux.
- Verifies Tesseract installation and language packs
- Tries pypdfium2 -> PIL -> pytesseract to extract text
- Prints short previews so you can confirm OCR is working

Usage:
  python ocr_debug.py --pdf "path/to/file.pdf" [--lang vie+eng] [--pages 1-3]

Notes:
- Make sure to `pip install pypdfium2 pytesseract pillow`
- On Windows, install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
  and set TESSERACT_CMD in .env or environment if not on PATH
"""

import argparse
import os
import sys
from typing import List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def check_tesseract() -> Tuple[bool, str]:
    import subprocess
    cand = [
        os.getenv("TESSERACT_CMD") or "",
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        "/usr/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
        "tesseract",
    ]
    for path in cand:
        if not path:
            continue
        try:
            out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                return True, path
        except Exception:
            continue
    # Last attempt: plain 'tesseract'
    try:
        out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return True, "tesseract"
    except Exception:
        pass
    return False, ""


def ocr_pdf(pdf_path: str, lang: str, page_start: int, page_end: int) -> List[str]:
    import pypdfium2 as pdfium
    import pytesseract

    # Respect TESSERACT_CMD if provided
    tcmd = os.getenv("TESSERACT_CMD")
    if tcmd and os.path.exists(tcmd):
        try:
            from pytesseract import pytesseract as _pt  # type: ignore
            _pt.tesseract_cmd = tcmd
        except Exception:
            pass

    doc = pdfium.PdfDocument(pdf_path)
    n_pages = len(doc)
    page_start = max(1, page_start)
    page_end = min(n_pages, page_end if page_end > 0 else n_pages)
    texts: List[str] = []
    for i in range(page_start - 1, page_end):
        page = doc[i]
        bmp = page.render(scale=2.0)
        pil = bmp.to_pil()
        txt = pytesseract.image_to_string(pil, lang=lang)
        texts.append((txt or "").strip())
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to scanned PDF")
    ap.add_argument("--lang", default=os.getenv("OCR_LANG", "vie+eng"), help="Tesseract languages, e.g. vie+eng")
    ap.add_argument("--pages", default="1-2", help="Pages range, e.g. 1-2, 2-5")
    args = ap.parse_args()

    pdf = args.pdf
    if not os.path.exists(pdf):
        print(f"❌ PDF not found: {pdf}")
        sys.exit(1)

    # Check Tesseract
    ok, path = check_tesseract()
    if ok:
        print(f"✅ Tesseract found at: {path}")
        # List langs if possible
        try:
            import subprocess
            out = subprocess.run([path, "--list-langs"], capture_output=True, text=True)
            langs = [l.strip() for l in out.stdout.splitlines() if l.strip() and not l.startswith("List of")]
            print(f"🗣  Installed languages: {', '.join(langs[:20])}{' ...' if len(langs) > 20 else ''}")
            if 'vie' not in ''.join(langs):
                print("⚠️  'vie' not found. Please install Vietnamese language pack.")
        except Exception:
            pass
    else:
        print("❌ Tesseract not found. Set TESSERACT_CMD or install Tesseract.")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)

    # Parse pages
    try:
        if "-" in args.pages:
            s, e = args.pages.split("-", 1)
            p1, p2 = int(s), int(e)
        else:
            p1 = int(args.pages)
            p2 = p1
    except Exception:
        p1, p2 = 1, 2

    # OCR
    try:
        texts = ocr_pdf(pdf, args.lang, p1, p2)
        total_chars = sum(len(t) for t in texts)
        print(f"✅ OCR done. Pages {p1}-{p2}. Total chars: {total_chars}")
        for i, t in enumerate(texts, start=p1):
            snippet = (t[:300] + "...") if len(t) > 300 else t
            print("\n— Page", i, "—")
            print(snippet if snippet else "(empty)")
        if total_chars == 0:
            print("⚠️  OCR returned empty text. Check scan quality and language setting (e.g., --lang vie+eng).")
    except Exception as e:
        print(f"❌ OCR error: {e}")
        print("• Ensure you installed: pypdfium2, pytesseract, pillow")
        print("• Verify TESSERACT_CMD points to the correct binary")
        sys.exit(1)


if __name__ == "__main__":
    main()

