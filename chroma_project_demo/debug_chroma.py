#!/usr/bin/env python3
"""
Debug script để kiểm tra ChromaDB và tìm lỗi tại sao không truy xuất được dữ liệu
"""

import os
import sys
sys.path.append('backend')

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings

def debug_chromadb():
    """Debug ChromaDB collection"""
    print("🔍 DEBUG CHROMADB")
    print("=" * 50)
    
    try:
        # Kết nối ChromaDB
        chroma_client = chromadb.PersistentClient(
            path="backend/chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        print("✅ Kết nối ChromaDB thành công")
        
        # Lấy collection theo backend
        use_openai = os.getenv("USE_OPENAI", "0") == "1"
        if use_openai:
            emb = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").replace(":","_").replace("-","_")
            collection_name = f"documents_openai_{emb}"
        else:
            emb = os.getenv("OLLAMA_EMBED_MODEL", "llama2").replace(":","_").replace("-","_")
            collection_name = f"documents_ollama_{emb}"
        try:
            collection = chroma_client.get_collection(name=collection_name)
            print(f"✅ Tìm thấy collection: {collection_name}")
        except Exception as e:
            print(f"❌ Không tìm thấy collection: {e}")
            
            # Liệt kê tất cả collections
            collections = chroma_client.list_collections()
            print(f"📋 Các collections có sẵn: {[c.name for c in collections]}")
            
            if not collections:
                print("❌ Không có collection nào! Cần upload tài liệu trước.")
                return
            
            # Sử dụng collection đầu tiên
            collection = collections[0]
            print(f"🔄 Sử dụng collection: {collection.name}")
        
        # Kiểm tra số lượng documents
        count = collection.count()
        print(f"📊 Số lượng documents: {count}")
        
        if count == 0:
            print("❌ Collection rỗng! Cần upload tài liệu.")
            return
        
        # Lấy sample documents
        print("\n📄 SAMPLE DOCUMENTS:")
        sample = collection.get(limit=5)
        documents = sample.get('documents', [])
        metadatas = sample.get('metadatas', [])
        ids = sample.get('ids', [])
        
        for i, (doc_id, doc, metadata) in enumerate(zip(ids, documents, metadatas)):
            print(f"\nDoc {i+1}:")
            print(f"  ID: {doc_id}")
            print(f"  Metadata: {metadata}")
            print(f"  Content: {doc[:200]}...")
        
        # Test search
        print("\n🔍 TEST SEARCH:")
        test_queries = [
            "phụ cấp ăn trưa",
            "lương",
            "Dalat Hasfarm",
            "tăng",
            "khu vực"
        ]
        
        for query in test_queries:
            print(f"\nTìm kiếm: '{query}'")
            try:
                # Test với query_texts (keyword search)
                results = collection.query(
                    query_texts=[query],
                    n_results=3,
                    include=["documents", "metadatas", "distances"]
                )
                
                docs = results.get('documents', [[]])[0]
                distances = results.get('distances', [[]])[0]
                
                print(f"  Tìm thấy: {len(docs)} documents")
                for j, (doc, dist) in enumerate(zip(docs, distances)):
                    print(f"    Doc {j+1}: distance={dist:.3f}, content={doc[:100]}...")
                    
            except Exception as e:
                print(f"  ❌ Lỗi query_texts: {e}")
                
                # Fallback: where_document contains
                try:
                    results = collection.get(
                        where_document={"$contains": query},
                        limit=3
                    )
                    docs = results.get('documents', [])
                    print(f"  Fallback tìm thấy: {len(docs)} documents")
                    for j, doc in enumerate(docs[:2]):
                        print(f"    Doc {j+1}: {doc[:100]}...")
                except Exception as e2:
                    print(f"  ❌ Lỗi where_document: {e2}")
        
        # Test embeddings
        print("\n🧮 TEST EMBEDDINGS:")
        try:
            # Test OpenAI embeddings
            if os.getenv("USE_OPENAI") == "1" and os.getenv("OPENAI_API_KEY"):
                import openai
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input="test embedding"
                )
                embedding = response.data[0].embedding
                print(f"✅ OpenAI embeddings: {len(embedding)} dimensions")
                
                # Test vector search
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=3,
                    include=["documents", "distances"]
                )
                docs = results.get('documents', [[]])[0]
                distances = results.get('distances', [[]])[0]
                print(f"  Vector search tìm thấy: {len(docs)} documents")
                for j, (doc, dist) in enumerate(zip(docs, distances)):
                    print(f"    Doc {j+1}: distance={dist:.3f}")
                    
            else:
                print("⚠️  OpenAI embeddings không được cấu hình")
                
        except Exception as e:
            print(f"❌ Lỗi test embeddings: {e}")
        
        print("\n" + "=" * 50)
        print("✅ DEBUG HOÀN THÀNH")
        
    except Exception as e:
        print(f"❌ Lỗi debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chromadb()
