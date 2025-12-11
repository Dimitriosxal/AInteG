# api/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import asyncio

# Import routers
from api.general_routes import router as general_router
from api.invoice_routes import router as invoice_router

app = FastAPI(
    title="AInteG Backend API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    max_upload_size=50 * 1024 * 1024,  # 50MB limit
)

# ================ TIMEOUT MIDDLEWARE ================
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Αύξηση timeout για upload requests
    if request.method == "POST" and ("/upload" in request.url.path):
        timeout = 300.0  # 5 λεπτά για uploads
    else:
        timeout = 30.0   # 30 δευτερόλεπτα για άλλα requests
    
    try:
        response = await asyncio.wait_for(call_next(request), timeout=timeout)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content={
                "status": "error",
                "message": f"Request timeout after {timeout} seconds"
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }
        )
# ====================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (αυτά είναι τα πραγματικά endpoints)
app.include_router(general_router)
app.include_router(invoice_router)

# Health endpoints
@app.get("/")
async def root():
    return {
        "message": "AInteG Backend API", 
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/general", "/invoices", "/docs", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AInteG Backend on http://127.0.0.1:8001")
    print("📚 API Documentation: http://127.0.0.1:8001/docs")
    print("⚙️  Upload timeout: 5 minutes")
    print("📏 Max file size: 50MB")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")