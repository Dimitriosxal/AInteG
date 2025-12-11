# api/chunked_routes.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import hashlib
import time

router = APIRouter(prefix="/chunked", tags=["chunked_upload"])

UPLOAD_TEMP_DIR = Path("uploads/temp")
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 5 * 1024 * 1024  # 5MB chunks

@router.post("/start")
async def start_upload(filename: str, total_size: int):
    """Start a chunked upload session"""
    file_id = hashlib.md5(f"{filename}_{time.time()}".encode()).hexdigest()
    temp_dir = UPLOAD_TEMP_DIR / file_id
    temp_dir.mkdir(exist_ok=True)
    
    return {
        "file_id": file_id,
        "chunk_size": CHUNK_SIZE,
        "total_chunks": (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE
    }

@router.post("/chunk/{file_id}/{chunk_index}")
async def upload_chunk(file_id: str, chunk_index: int, chunk: UploadFile = File(...)):
    """Upload a single chunk"""
    temp_dir = UPLOAD_TEMP_DIR / file_id
    if not temp_dir.exists():
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    chunk_path = temp_dir / f"chunk_{chunk_index}"
    content = await chunk.read()
    chunk_path.write_bytes(content)
    
    return {"status": "ok", "chunk": chunk_index}

@router.post("/complete/{file_id}")
async def complete_upload(file_id: str, filename: str):
    """Complete upload by assembling chunks"""
    temp_dir = UPLOAD_TEMP_DIR / file_id
    if not temp_dir.exists():
        raise HTTPException(status_code=404, detail="Upload session not found")
    
    # Βρες όλα τα chunks
    chunks = sorted(temp_dir.glob("chunk_*"), key=lambda x: int(x.name.split("_")[1]))
    
    # Συνένωσε τα chunks
    final_path = Path("uploads/general") / filename
    with open(final_path, "wb") as final_file:
        for chunk_path in chunks:
            with open(chunk_path, "rb") as chunk_file:
                final_file.write(chunk_file.read())
    
    # Καθάρισε τα temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    return {"status": "ok", "filename": filename}