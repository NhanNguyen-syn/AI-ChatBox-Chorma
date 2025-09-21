import os
import uuid
from sqlalchemy.orm import Session
from database import Document, DocumentChunk, OcrText, get_db
from routers.files import process_file_content, _normalize_vi, clean_ocr_text
from langchain_community.embeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
import chromadb

# This function will be executed by the RQ worker.
def process_file_and_embed(file_path: str, document_id: str):
    db: Session = next(get_db())
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            print(f"[Worker] Document with id {document_id} not found.")
            return

        # Reuse the existing processing logic
        chunks = process_file_content(db, file_path, document.file_type, base_metadata={})

        if not chunks:
            document.status = "failed"
            db.commit()
            return

        texts = [getattr(ch, 'page_content', '') for ch in chunks]
        metadatas = [getattr(ch, 'metadata', {}) for ch in chunks]
        
        # Embedding logic (simplified for worker context)
        # In a real app, this would be more robust, handling different backends.
        embeddings_model = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "llama2"))
        chunk_embeddings = embeddings_model.embed_documents(texts)

        # Add to ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(name=os.getenv("CHROMA_COLLECTION", "documents"))
        ids = [str(uuid.uuid4()) for _ in texts]
        collection.add(embeddings=chunk_embeddings, documents=texts, metadatas=metadatas, ids=ids)

        # Update document status and save chunks
        document.status = "completed"
        for cid in ids:
            db.add(DocumentChunk(document_id=document.id, chunk_id=cid, collection=collection.name))
        
        # Save OCR text if any
        for chunk in chunks:
            md = getattr(chunk, 'metadata', {})
            content = getattr(chunk, 'page_content', '')
            if md.get('block_type') in ['text', 'table'] and content:
                db.add(OcrText(
                    document_id=document.id, 
                    source_filename=document.filename, 
                    page=md.get('page'), 
                    content=content, 
                    normalized_content=_normalize_vi(clean_ocr_text(content))
                ))

        db.commit()
        print(f"[Worker] Successfully processed and embedded file: {document.filename}")

    except Exception as e:
        print(f"[Worker] Error processing file {document_id}: {e}")
        if 'document' in locals() and document:
            document.status = "failed"
            db.commit()
    finally:
        db.close()

