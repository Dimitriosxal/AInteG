from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from core.integrations.rag_adapter import rag_add_document, rag_search
import asyncio
import pdfplumber
router = APIRouter(prefix="/general", tags=["general"])

UPLOAD_DIR = Path("uploads/general")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_general(file: UploadFile = File(...)):
    try:
        content = await file.read()
        path = UPLOAD_DIR / file.filename
        path.write_bytes(content)
        
        # Για μεγάλα PDFs, απλά αποθήκευση χωρίς processing
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > 5:
            return {
                "status": "warning",
                "filename": file.filename,
                "message": "File saved but RAG processing skipped (too large)"
            }
        
        text = ""  # <-- ΠΡΟΣΘΗΚΗ ΑΥΤΗΣ ΤΗΣ ΓΡΑΜΜΗΣ
        
        # Αν είναι PDF, χρησιμοποίησε pdfplumber
        if file.filename.lower().endswith('.pdf'):
            try:
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                print(f"📄 PDF extracted {len(text)} characters")
            except Exception as e:
                print(f"PDF extraction error: {e}")
                text = "PDF extraction failed"
        else:
            # Για txt αρχεία
            try:
                text = content.decode("utf-8", errors="ignore")
            except Exception:
                text = ""

        # Αν το κείμενο είναι πολύ μικρό, προειδοποίηση
        if len(text.strip()) < 10:
            return {
                "status": "warning",
                "message": f"Little or no text extracted from file",
                "filename": file.filename,
                "text_preview": text[:200] if text else ""
            }

        # Προσθήκη στο RAG
        rag_result = rag_add_document(
            text=text,
            metadata={"filename": file.filename, "type": "general"},
            collection="general"
        )

        return {
            "status": "ok",
            "filename": file.filename,
            "text_length": len(text),
            "rag_result": rag_result
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Upload failed: {str(e)[:100]}"
        }

@router.post("/search")
async def search_general(query: dict):
    q = query.get("query")
    top_k = query.get("top_k", 3)

    results = rag_search(q, collection="general", top_k=top_k)
    return results


@router.get("/debug")
async def debug_rag():
    """Debug endpoint to test RAG"""
    from core.integrations.rag_adapter import rag_search
    from core.integrations.rag_adapter import get_collection
    
    # Test 1: Basic search
    test_result = rag_search("test", "general", 3)
    
    # Test 2: Check collection
    col = get_collection("general")
    count = col.count()
    
    # Check what's in the collection
    sample = col.peek() if hasattr(col, 'peek') else {}
    
    return {
        "rag_search_result": test_result,
        "collection_count": count,
        "collection_sample": sample,
        "test_query": "test"
    }