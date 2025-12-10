import os
import re
import json
from pathlib import Path

import requests
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# =====================================================
# LOAD .env (bulletproof)
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

API_URL = "http://127.0.0.1:8001"

# =====================================================
# Streamlit basic config
# =====================================================
st.set_page_config(page_title="AInteG Console", layout="centered")
st.title("AInteG Management Console")
st.markdown("### Διάλεξε κατηγορία (General ή Invoice)")

# Φάκελοι uploads (για file manager)
GENERAL_UPLOAD_DIR = Path("uploads/general")
INVOICE_UPLOAD_DIR = Path("uploads/invoices")
GENERAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INVOICE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# OpenAI client helper
# =====================================================
def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Δεν ρίχνουμε exception — επιστρέφουμε None και δίνουμε μήνυμα στο UI
        return None
    return OpenAI(api_key=api_key)


# =====================================================
# Helpers
# =====================================================
def highlight_snippet(text: str, query: str, max_len: int = 400) -> str:
    """Μικρό snippet γύρω από το query, με bold στο query."""
    if not text:
        return ""

    idx = text.lower().find(query.lower())
    if idx != -1:
        start = max(0, idx - 120)
        end = min(len(text), idx + 120)
        snippet = text[start:end]
    else:
        snippet = text[:max_len]

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    snippet = pattern.sub(lambda m: f"**{m.group(0)}**", snippet)

    return snippet


def list_files(dir_path: Path):
    files = []
    if dir_path.exists():
        for p in sorted(dir_path.iterdir(), key=lambda x: x.name):
            if p.is_file():
                files.append(p)
    return files


# =====================================================
# RAG CHAT CORE (safe & limited)
# =====================================================
def rag_chat(scope: str, query: str, top_k: int = 3, debug: bool = False):
    """
    scope: 'general' ή 'invoices'
    - Καλεί το αντίστοιχο /search endpoint.
    - Παίρνει ΜΕΧΡΙ 3 chunks.
    - Περιορίζει το συνολικό context σε ~4000 χαρακτήρες.
    - Κάνει κλήση στο OpenAI και επιστρέφει απάντηση + λίγα metadata.
    """
    endpoint = f"{API_URL}/{scope}/search"

    try:
        resp = requests.post(endpoint, json={"query": query, "top_k": top_k})
    except Exception as e:
        return {
            "answer": f"Σφάλμα HTTP προς backend: {e}",
            "contexts": [],
            "metadatas": [],
            "raw": None,
        }

    try:
        raw = resp.json()
    except Exception:
        return {
            "answer": f"Το backend δεν επέστρεψε έγκυρο JSON. HTTP {resp.status_code}",
            "contexts": [],
            "metadatas": [],
            "raw": None,
        }

    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    # Chroma: documents = [[chunk1, chunk2, ...]]
    if docs and isinstance(docs[0], list):
        contexts = docs[0]
    else:
        contexts = docs

    if metas and isinstance(metas[0], list):
        metadatas = metas[0]
    else:
        metadatas = metas

    # --- LIMIT chunks to avoid huge prompts ---
    MAX_CHUNKS = 3
    contexts = contexts[:MAX_CHUNKS]
    metadatas = metadatas[:MAX_CHUNKS]

    if not contexts:
        return {
            "answer": "Δεν βρήκα σχετικές πληροφορίες στο RAG. Ανέβασε πρώτα κάποια αρχεία.",
            "contexts": [],
            "metadatas": [],
            "raw": raw,
        }

    # --- Build context block with char limit ---
    CONTEXT_CHAR_LIMIT = 4000
    context_block = ""
    for i, chunk in enumerate(contexts):
        meta = metadatas[i] if i < len(metadatas) else {}
        source_info = f"(source: {meta.get('filename', 'unknown')}, chunk: {meta.get('chunk', '-')})"
        piece = f"---\n{source_info}\n{chunk}\n\n"

        if len(context_block) + len(piece) > CONTEXT_CHAR_LIMIT:
            context_block += "\n...[TRUNCATED]...\n"
            break

        context_block += piece

    client = get_client()
    if client is None:
        return {
            "answer": "Δεν βρέθηκε OPENAI_API_KEY στο .env. Συμπλήρωσέ το και ξανατρέξε την εφαρμογή.",
            "contexts": contexts,
            "metadatas": metadatas,
            "raw": raw,
        }

    system_msg = (
        "Είσαι βοηθός RAG του AInteG. "
        "Απαντάς ΣΥΝΤΟΜΑ, στα ελληνικά, χρησιμοποιώντας ΜΟΝΟ τις πληροφορίες που σου δίνονται. "
        "Αν δεν βρίσκεις απάντηση στο context, πες ξεκάθαρα ότι δεν υπάρχει αρκετή πληροφορία."
    )

    user_msg = (
        "Χρησιμοποίησε τις παρακάτω πληροφορίες (context) για να απαντήσεις στην ερώτηση.\n\n"
        f"{context_block}\n\n"
        f"Ερώτηση: {query}"
    )

    try:
        chat_resp = client.chat.completions.create(
            model=os.getenv("MODEL_CHAT", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        answer = chat_resp.choices[0].message.content
    except Exception as e:
        answer = f"Σφάλμα LLM: {e}"

    return {
        "answer": answer,
        "contexts": contexts,
        "metadatas": metadatas,
        "raw": raw if debug else None,
    }


# =====================================================
# MODE SELECTION
# =====================================================
mode = st.radio(
    "Κατηγορία:",
    ["General", "Invoice"],
    horizontal=True,
)

st.divider()

# ===================================================================
# ========================  GENERAL MODE  ============================
# ===================================================================
if mode == "General":
    st.subheader("📄 General Chat + Upload + Files")
    st.write("Ανέβασε αρχεία (PDF/TXT) και κάνε ερωτήσεις πάνω στο περιεχόμενό τους.")

    # -------------------------
    # Chat-RAG (πάνω)
    # -------------------------
    st.markdown("### 💬 Chat-RAG (General)")

    if "general_chat" not in st.session_state:
        st.session_state.general_chat = []

    debug_general = st.checkbox("Debug mode (δείξε RAG αποτελέσματα)", key="general_debug")

    # ιστορικό
    for msg in st.session_state.general_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ρώτα κάτι πάνω στα general έγγραφά σου:")
    if user_input:
        st.session_state.general_chat.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Ψάχνω στο RAG..."):
                result = rag_chat("general", user_input, top_k=3, debug=debug_general)
                answer = result["answer"]
                st.markdown(answer)

                contexts = result.get("contexts") or []
                metadatas = result.get("metadatas") or []

                if contexts:
                    st.markdown("### 📚 Σχετικό απόσπασμα:")
                    for i, ctx in enumerate(contexts):
                        meta = metadatas[i] if i < len(metadatas) else {}
                        snippet = highlight_snippet(ctx, user_input)
                        st.markdown(
                            f"*{meta.get('filename', 'unknown')}* (chunk {meta.get('chunk', '-')})"
                        )
                        st.markdown(f"> {snippet}")
                        break  # δείχνουμε μόνο το πρώτο snippet για καθαρό UI

                if debug_general and result.get("raw") is not None:
                    st.markdown("### 🔧 RAW RAG RESPONSE")
                    st.json(result["raw"])

        st.session_state.general_chat.append({"role": "assistant", "content": answer})

    st.divider()

    # -------------------------
    # Upload (κάτω)
    # -------------------------
    st.markdown("### 🔼 Upload (General)")
    gen_file = st.file_uploader("Επίλεξε PDF ή TXT:", type=["pdf", "txt"], key="gen_upload")

    if gen_file is not None:
        if st.button("📤 Upload (General)"):
            files = {"file": (gen_file.name, gen_file.getvalue())}
            r = requests.post(f"{API_URL}/general/upload", files=files)

            st.write("Status:", r.status_code)
            try:
                st.json(r.json())
            except Exception:
                st.error("Το backend δεν επέστρεψε έγκυρο JSON.")
                st.code(r.text)

    st.divider()

    # -------------------------
    # File Manager (General)
    # -------------------------
    st.markdown("### 📁 Αρχεία (uploads/general)")
    gen_files = list_files(GENERAL_UPLOAD_DIR)
    if not gen_files:
        st.info("Δεν υπάρχουν αρχεία στον φάκελο uploads/general.")
    else:
        for p in gen_files:
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.write(p.name)
            with col2:
                size_kb = p.stat().st_size / 1024
                st.write(f"{size_kb:.1f} KB")
            with col3:
                if st.button("🗑️", key=f"del_gen_{p.name}"):
                    try:
                        p.unlink()
                        st.success(f"Διαγράφηκε: {p.name}")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Σφάλμα διαγραφής: {e}")

# ===================================================================
# ========================  INVOICE MODE  ============================
# ===================================================================
else:
    st.subheader("🧾 Invoices Chat + Upload (OCR) + Files")
    st.write("Ανέβασε τιμολόγια (PDF/JPG/PNG) για OCR, parsing και RAG αναζήτηση.")

    # -------------------------
    # Chat-RAG (Invoices) ΠΑΝΩ
    # -------------------------
    st.markdown("### 💬 Chat-RAG (Invoices)")

    if "invoice_chat" not in st.session_state:
        st.session_state.invoice_chat = []

    debug_invoices = st.checkbox("Debug mode (δείξε RAG αποτελέσματα)", key="invoice_debug")

    for msg in st.session_state.invoice_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    inv_query = st.chat_input("Ρώτα κάτι πάνω στα invoices σου:")
    if inv_query:
        st.session_state.invoice_chat.append({"role": "user", "content": inv_query})

        with st.chat_message("assistant"):
            with st.spinner("Ψάχνω στα invoices..."):
                result = rag_chat("invoices", inv_query, top_k=3, debug=debug_invoices)
                answer = result["answer"]
                st.markdown(answer)

                contexts = result.get("contexts") or []
                metadatas = result.get("metadatas") or []

                if contexts:
                    st.markdown("### 📚 Σχετικό απόσπασμα από invoice:")
                    for i, ctx in enumerate(contexts):
                        meta = metadatas[i] if i < len(metadatas) else {}
                        snippet = highlight_snippet(ctx, inv_query)
                        st.markdown(
                            f"*{meta.get('filename', 'unknown')}* (chunk {meta.get('chunk', '-')})"
                        )
                        st.markdown(f"> {snippet}")
                        break

                if debug_invoices and result.get("raw") is not None:
                    st.markdown("### 🔧 RAW RAG RESPONSE (Invoices)")
                    st.json(result["raw"])

        st.session_state.invoice_chat.append({"role": "assistant", "content": answer})

    st.divider()

    # -------------------------
    # Upload + OCR (κάτω)
    # -------------------------
    st.markdown("### 🔼 Upload Invoice (OCR)")
    inv_file = st.file_uploader(
        "Επίλεξε invoice (PDF/JPG/PNG):",
        type=["pdf", "jpg", "jpeg", "png"],
        key="inv_upload"
    )

    if inv_file is not None:
        if st.button("📤 Upload & OCR"):
            files = {"file": (inv_file.name, inv_file.getvalue())}
            r = requests.post(f"{API_URL}/invoices/upload", files=files)

            st.write("Status:", r.status_code)
            try:
                data = r.json()
            except Exception:
                st.error("Το backend δεν επέστρεψε έγκυρο JSON.")
                st.code(r.text)
            else:
                st.success("OCR & parsing ολοκληρώθηκαν!")

                st.markdown("### 📝 OCR Preview (ως text)")
                st.code(data.get("ocr_preview", ""))

                st.markdown("### 📦 Parsed Invoice JSON")
                st.json(data.get("parsed_invoice", {}))

                if "price_changes" in data:
                    st.markdown("### 🔍 Πιθανές αλλαγές τιμών")
                    pcs = data.get("price_changes", [])
                    if pcs:
                        for pc in pcs:
                            st.warning(
                                f"**{pc['product']}**\n\n"
                                f"Old price: {pc['old_price']}\n"
                                f"New price: {pc['new_price']}"
                            )
                    else:
                        st.info("Δεν εντοπίστηκαν αλλαγές τιμών.")

                if "mlg_candidates" in data:
                    st.markdown("### 🏷️ MLG Candidates")
                    mlg = data.get("mlg_candidates", [])
                    if mlg:
                        st.json(mlg)
                    else:
                        st.info("Δεν υπάρχουν MLG items.")

    st.divider()

    # -------------------------
    # File Manager (Invoices)
    # -------------------------
    st.markdown("### 📁 Αρχεία (uploads/invoices)")
    inv_files = list_files(INVOICE_UPLOAD_DIR)
    if not inv_files:
        st.info("Δεν υπάρχουν αρχεία στον φάκελο uploads/invoices.")
    else:
        for p in inv_files:
            col1, col2, col3 = st.columns([4, 2, 1])
            with col1:
                st.write(p.name)
            with col2:
                size_kb = p.stat().st_size / 1024
                st.write(f"{size_kb:.1f} KB")
            with col3:
                if st.button("🗑️", key=f"del_inv_{p.name}"):
                    try:
                        p.unlink()
                        st.success(f"Διαγράφηκε: {p.name}")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Σφάλμα διαγραφής: {e}")
