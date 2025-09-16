#!/usr/bin/env python3
"""
Quick verification tool for Excel -> ChromaDB ingestion.

What it does:
- Connects to the local SQLite DB to find the most recently uploaded Excel file (table: excel_files)
- Opens its dedicated ChromaDB collection
- Prints a handful of (document, metadata) pairs exactly as stored
- Loads the original Excel file via pandas and prints a few raw rows for side-by-side comparison

Usage:
  python chroma_project_demo/verify_excel_ingestion.py

Note:
- This script is read-only and safe to run. It assumes the repo layout:
    chroma_project_demo/
      backend/
        chroma_chat.db
        chroma_db/
- If paths differ, adjust BASE_DIR below.
"""
import os
import sys
import json
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
DB_PATH = os.path.join(BACKEND_DIR, "chroma_chat.db")
CHROMA_PATH = os.path.join(BACKEND_DIR, "chroma_db")


def _fmt_dt(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        return str(ts)
    except Exception:
        return str(ts)


def get_latest_excel_file_row():
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"SQLite DB not found at {DB_PATH}")
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT id, filename, file_path, collection_name, uploaded_at
            FROM excel_files
            ORDER BY uploaded_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("No rows found in excel_files. Upload an Excel file first in Admin UI.")
        return dict(row)
    finally:
        con.close()


def print_chroma_samples(collection_name: str, limit: int = 8):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    try:
        coll = client.get_collection(collection_name)
    except Exception as e:
        # Some chroma versions require explicit keyword
        coll = client.get_collection(name=collection_name)

    cnt = coll.count()
    print(f"\n[Chroma] Collection '{collection_name}' has {cnt} vectors")
    sample = coll.get(limit=limit)
    docs = sample.get("documents", []) or []
    metas = sample.get("metadatas", []) or []
    ids = sample.get("ids", []) or []

    for i in range(min(limit, len(ids))):
        doc = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        print("\n--- Chroma Record", i + 1, "---")
        print("ID:", ids[i])
        print("Metadata keys:", list((meta or {}).keys()))
        print("Metadata sample:", json.dumps(meta, ensure_ascii=False)[:400])
        print("Document preview:", (doc or "")[:350].replace("\n", " "))


def print_excel_rows(file_path: str, limit: int = 6):
    import pandas as pd
    if not os.path.exists(file_path):
        print(f"[WARN] Excel file no longer exists on disk: {file_path}")
        return
    ext = (os.path.splitext(file_path)[1] or "").lower()
    try:
        if ext in (".xlsx", ".xls"):
            try:
                engine = "openpyxl" if ext == ".xlsx" else "xlrd"
                sheets = pd.read_excel(file_path, sheet_name=None, engine=engine, dtype=str)
            except Exception:
                sheets = pd.read_excel(file_path, sheet_name=None, dtype=str)
        elif ext == ".csv":
            sheets = {"Sheet1": pd.read_csv(file_path, dtype=str)}
        else:
            print(f"[WARN] Unsupported Excel extension: {ext}")
            return
    except Exception as e:
        print(f"[ERROR] Cannot read Excel file: {e}")
        return

    print("\n[Excel] Raw rows (first few per sheet):")
    for sh, df in (sheets or {}).items():
        if df is None or df.empty:
            continue
        print(f"\n-- Sheet: {sh} --")
        print("Columns:", list(df.columns))
        print(df.head(limit).fillna("").to_string(index=False))


if __name__ == "__main__":
    print("=== Verify Excel Ingestion ===")
    print("Base:", BASE_DIR)
    try:
        row = get_latest_excel_file_row()
        print("Latest ExcelFile row:")
        print(json.dumps({k: (row.get(k)) for k in ["id", "filename", "collection_name", "uploaded_at"]}, ensure_ascii=False, indent=2))
        collection_name = row.get("collection_name")
        file_path = row.get("file_path")

        if collection_name:
            print_chroma_samples(collection_name)
        else:
            print("[ERROR] No collection_name stored for this Excel file.")

        if file_path:
            print_excel_rows(file_path)
        else:
            print("[ERROR] No file_path stored.")

        print("\nDone.")
    except Exception as e:
        print("[FATAL]", e)
        import traceback
        traceback.print_exc()

