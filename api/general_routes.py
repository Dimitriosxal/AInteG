from fastapi import APIRouter, UploadFile, File
from pathlib import Path
from pydantic import BaseModel

from core.integrations.rag_adapter import rag_add_document, rag_search

# Router με prefix και tag
router = APIRouter(prefix="/general", tags=["general"])

# -----------------------------
# MODELS
# -----------------------------
class QueryModel(BaseModel):
    query: str
    top_k: int = 3

# -----------------------------
# UPLOAD DIRECTORY
# -----------------------------
UPLOAD_DIR = Path("uploads/general")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# UPLOAD ROUTE
# -----------------------------
@router.post("/upload")
async def upload_general(file: UploadFile = File(...)):
    path = UPLOAD_DIR / file.filename

    # Save the uploaded file
    content = await file.read()
    path.write_bytes(content)

    # Convert to text (UTF-8 safe)
    try:
        text = content.decode("utf-8")
    except:
        text = content.decode("latin-1", errors="ignore")

    # Store into ChromaDB
    rag_add_document(
        text=text,
        metadata={"filename": file.filename, "type": "general"},
        collection="general"
    )

    return {"status": "ok", "filename": file.filename}

# -----------------------------
# SEARCH ROUTE
# -----------------------------
@router.post("/search")
async def search_general(query: QueryModel):
    results = rag_search(
        query=query.query,
        collection="general",
        top_k=query.top_k
    )
    return results
