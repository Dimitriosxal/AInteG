import os
import hashlib
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# --------------------------------------
# Local Embedding Model (Unicode Safe)
# --------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

class LocalEmbeddingFunction:
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return model.encode(input).tolist()

emb_func = LocalEmbeddingFunction()

# --------------------------------------
# ChromaDB Init
# --------------------------------------
DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
client = PersistentClient(path=DB_DIR)

def get_collection(name: str):
    try:
        return client.get_collection(name)
    except:
        return client.create_collection(
            name=name,
            embedding_function=emb_func
        )

# --------------------------------------
# ADD document
# --------------------------------------
def rag_add_document(text: str, metadata: dict, collection: str):
    col = get_collection(collection)

    original_name = metadata.get("filename", "")
    safe_id = hashlib.md5(original_name.encode("utf-8")).hexdigest()

    clean_metadata = {
        "safe_name": safe_id,
        "type": metadata.get("type", "general"),
        "original_filename": original_name
    }

    col.upsert(
        ids=[safe_id],
        documents=[text],
        metadatas=[clean_metadata]
    )

    return safe_id

# --------------------------------------
# QUERY
# --------------------------------------
def rag_search(query: str, collection: str, top_k: int = 3):
    col = get_collection(collection)
    return col.query(query_texts=[query], n_results=top_k)
