from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import asyncio

from core.ocr.invoice_ocr import ocr_to_text
from core.integrations.rag_adapter import rag_add_document
from core.invoice.parser import parse_invoice_text

router = APIRouter(prefix="/invoices", tags=["invoices"])

UPLOAD_DIR = Path("uploads/invoices")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_invoice(file: UploadFile = File(...)):
    try:
        # Timeout protection for reading the file
        try:
            content = await asyncio.wait_for(file.read(), timeout=30.0)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Upload timeout")

        # Save file
        path = UPLOAD_DIR / file.filename
        path.write_bytes(content)

        # OCR
        text = ocr_to_text(str(path), file.filename)

        if len(text.strip()) < 20:
            return {
                "status": "error",
                "message": "OCR failed: too little text",
                "ocr_preview": text
            }

        # Store in RAG
        rag_add_document(
            text=text,
            metadata={"filename": file.filename, "type": "invoice"},
            collection="invoices"
        )

        # Parse
        parsed = parse_invoice_text(text)

        return {
            "status": "ok",
            "filename": file.filename,
            "ocr_preview": text[:2000],
            "parsed_invoice": parsed
        }

    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "error",
            "message": f"Internal backend error: {e}"
        }



@router.post("/search")
async def search_invoice(query: dict):
    from core.integrations.rag_adapter import rag_search

    q = query.get("query")
    top_k = query.get("top_k", 3)

    return rag_search(q, collection="invoices", top_k=top_k)