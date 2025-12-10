from fastapi import APIRouter, UploadFile, File
from pathlib import Path

from core.ocr.invoice_ocr import ocr_to_text
from core.integrations.rag_adapter import rag_add_document, rag_search
from core.invoice.parser import parse_invoice_text
from models.rag_models import QueryRequest


router = APIRouter(prefix="/invoices", tags=["invoices"])

UPLOAD_DIR = Path("uploads/invoices")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# UPLOAD + OCR + PARSE
# -------------------------
@router.post("/upload")
async def upload_invoice(file: UploadFile = File(...)):

    path = UPLOAD_DIR / file.filename
    content = await file.read()
    path.write_bytes(content)

    text = ocr_to_text(str(path), file.filename)

    if not text or len(text.strip()) < 20:
        return {
            "status": "error",
            "message": "OCR failed or returned empty text.",
            "ocr_preview": text,
        }

    # Store in vector DB
    rag_add_document(
        text=text,
        metadata={"filename": file.filename, "type": "invoice"},
        collection="invoices"
    )

    parsed = parse_invoice_text(text)

    return {
        "status": "ok",
        "filename": file.filename,
        "ocr_preview": text[:2000],
        "parsed_invoice": parsed
    }


# -------------------------
# SEARCH (RAG for invoices)
# -------------------------
@router.post("/search")
async def search_invoice_collection(query: QueryRequest):
    """
    RAG search specifically for invoices.
    Called from Streamlit chat mode.
    """
    return rag_search(
        collection="invoices",
        query=query.query,
        top_k=query.top_k
    )
