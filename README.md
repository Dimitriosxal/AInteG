📘 AInteG – AI Document Automation System

AInteG is an intelligent, modular system designed to automate document processing for small businesses.
It combines FastAPI, RAG (ChromaDB), Hybrid OCR, LLM invoice parsing, and a Streamlit console for a seamless end-to-end workflow.

🚀 Features
🔎 RAG Document Search

Stores documents in ChromaDB

Generates embeddings via OpenAI

Allows question-answering over documents

Chunking + metadata preserved

🧾 Invoice Automation

Hybrid OCR pipeline:
✔ Tesseract (local)
✔ OpenAI Vision (cloud)

AI invoice parser (structured JSON output)

Regex fallback when LLM fails

Supports PDF & images (jpg/png)

⚡ FastAPI Backend

Endpoints for document upload

Invoice OCR & parsing

RAG search

Clean, modular architecture

🖥 Streamlit Management Console

File upload UI

RAG chat over documents & invoices

Invoice previews

OCR preview

Backend health monitoring

🏗 Project Structure
AInteG/
│── api/                # FastAPI routes (general, invoices)
│── core/               # OCR, RAG adapter, parser
│── models/             # Pydantic models
│── console/            # Streamlit app (UI)
│── requirements.txt    # Backend dependencies
│── .env.template       # Example env file (safe)
│── .gitignore
│── README.md


❗ Important runtime folders like uploads/ and chroma_db/
are NOT included (auto-created at runtime).

⚙️ Installation
1) Clone the repo
git clone https://github.com/Dimitriosxal/AInteG.git
cd AInteG/backend

2) Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

3) Install backend dependencies
pip install -r requirements.txt

4) Create .env file

Use .env.template as the base.

OPENAI_API_KEY=your_key_here
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_DB_DIR=./chroma_db

5) Run the backend
uvicorn api.main:app --reload --port 8001


Backend docs:
👉 http://127.0.0.1:8001/docs

🖥 Running the Streamlit Console
cd ../console
pip install -r requirements.txt
streamlit run streamlit_app.py

🧱 Tech Stack

Python 3.10+

FastAPI

Streamlit

ChromaDB

Tesseract OCR

OpenAI GPT-4o Vision & embeddings

PyMuPDF

pdfplumber

🔐 Security Notes

.env is ignored

No API keys included

Upload folders removed from repo

ChromaDB local vector data removed
