import os
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from dotenv import load_dotenv

# ----------------------------------------
# Load .env
# ----------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

client = OpenAI(api_key=OPENAI_KEY)

# ----------------------------------------
# Chroma client
# ----------------------------------------
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# --------------------------
# DEBUG PRINTS
# --------------------------
print("=== DEBUG CHROMA ===")
print("CHROMA_DIR:", CHROMA_DIR)
print("Collections at startup:", chroma_client.list_collections())
print("=====================")


# ----------------------------------------
# Get or create collection
# ----------------------------------------
def get_collection(name: str):
    try:
        collection = chroma_client.get_collection(name)
        print(f"[DEBUG] Found existing collection: {name}")
        return collection
    except Exception as e:
        print(f"[DEBUG] Creating new collection: {name}")
        # Δημιουργία collection με σωστό μοντέλο και με ΑΠΑΓΟΡΕΥΣΗ auto-embeddings
        return chroma_client.create_collection(
            name=name,
            metadata={"embedding_model": EMBED_MODEL},
            embedding_function=None  # δεν επιτρέπει στην Chroma να κάνει δικά της embeddings
        )


# ----------------------------------------
# Create embeddings
# ----------------------------------------
def embed(text: str):
    """Generate an OpenAI embedding vector."""
    resp = client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return resp.data[0].embedding


# ----------------------------------------
# Normalize file names
# ----------------------------------------
def safe_filename(name: str) -> str:
    """Convert filename into safe unique key."""
    stem = uuid.uuid4().hex
    return stem


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    length = len(text)
    start = 0

    while start < length:
        end = min(start + chunk_size, length)
        segment = text[start:end]
        chunks.append(segment)
        start += chunk_size - overlap

    return chunks


# ----------------------------------------
# Add document into RAG DB
# ----------------------------------------
def rag_add_document(text: str, metadata: dict, collection: str):
    try:
        col = get_collection(collection)
        
        # Κόψε το κείμενο αν είναι πολύ μεγάλο
        MAX_TEXT_LENGTH = 1000000  # 1 εκατομμύριο χαρακτήρες
        if len(text) > MAX_TEXT_LENGTH:
            print(f"⚠️ Text too long ({len(text)}), truncating to {MAX_TEXT_LENGTH}")
            text = text[:MAX_TEXT_LENGTH]
        
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        
        ids = []
        docs = []
        metas = []
        embeds = []
        
        # Προσθήκη chunks ένα-ένα για να αποφύγουμε memory issues
        for idx, chunk in enumerate(chunks[:50]):  # Μέγιστο 50 chunks
            cid = safe_filename(metadata.get("filename", "doc")) + f"_{idx}"
            ids.append(cid)
            docs.append(chunk)
            metas.append(metadata)
            
            try:
                embeds.append(embed(chunk))
            except Exception as e:
                print(f"⚠️ Embedding error for chunk {idx}: {e}")
                continue
        
        if ids:  # Μόνο αν έχουμε embeddings
            col.add(
                ids=ids,
                documents=docs,
                metadatas=metas,
                embeddings=embeds
            )
            print(f"✅ Added {len(ids)} chunks to {collection}")
            return {"status": "added", "chunks": len(ids)}
        else:
            print(f"❌ No chunks added to {collection}")
            return {"status": "error", "message": "No embeddings generated"}
            
    except Exception as e:
        print(f"❌ RAG add error: {e}")
        return {"status": "error", "message": str(e)}

# ----------------------------------------
# RAG SEARCH
# ----------------------------------------
def rag_search(query: str, collection: str, top_k: int = 3):
    col = get_collection(collection)

    # Δημιουργούμε εμείς το query embedding
    q_embed = embed(query)

    res = col.query(
        query_embeddings=[q_embed],
        n_results=top_k
    )

    return {
        "ids": res.get("ids", [[]])[0],
        "documents": res.get("documents", [[]])[0],
        "metadatas": res.get("metadatas", [[]])[0],
    }