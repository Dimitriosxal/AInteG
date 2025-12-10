from fastapi import FastAPI
from api.general_routes import router as general_router
from api.invoice_routes import router as invoice_router

app = FastAPI(title="AInteG Backend")

app.include_router(general_router)
app.include_router(invoice_router)

@app.get("/health")
def health():
    return {"status": "ok"}
