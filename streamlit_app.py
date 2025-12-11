import os
import json
from pathlib import Path
import requests
import streamlit as st
from openai import OpenAI
import re
from dotenv import load_dotenv
import time
import sys
from PIL import Image
import io

# =====================================
# LOAD ENVIRONMENT VARIABLES
# =====================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Αρχική διεύθυνση API
API_URL = "http://127.0.0.1:8001"

# Ορισμός ρύθμισης σελίδας
st.set_page_config(
    page_title="AInteG Management Console",
    layout="centered",
    page_icon="🤖"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 10px 0;
    }
    
    /* Βελτιωμένα tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AInteG Management Console")
st.caption("RAG-powered document management system")

# =====================================
# BACKEND CONNECTION TEST
# =====================================
def test_backend_connection():
    """Test if backend is reachable"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to backend"
    except Exception as e:
        return False, str(e)

# =====================================
# DISPLAY BACKEND STATUS
# =====================================
with st.sidebar:
    st.header("🔧 Σύνδεση Backend")
    
    if st.button("🔄 Έλεγχος Σύνδεσης"):
        is_connected, message = test_backend_connection()
        if is_connected:
            st.success("✅ Backend είναι online!")
            if isinstance(message, dict):
                st.json(message)
        else:
            st.error(f"❌ Backend offline: {message}")
    
    st.divider()
    
    # Manual API URL input
    st.subheader("⚙️ Ρυθμίσεις API")
    api_url_input = st.text_input("API URL", value=API_URL, key="api_url_sidebar")
    
    if st.button("Εφαρμογή νέας URL"):
        st.session_state.api_url = api_url_input
        st.rerun()
    
    # Use session state for API URL
    if "api_url" in st.session_state:
        API_URL = st.session_state.api_url
    else:
        st.session_state.api_url = API_URL

# =====================================
# SIMPLE UPLOAD FUNCTION
# =====================================
def simple_upload(file, endpoint):
    """Simple upload function without complex progress bars"""
    if file is None:
        return {"error": True, "message": "Δεν επιλέχθηκε αρχείο"}
    
    try:
        # Show upload status
        status = st.empty()
        status.info(f"📤 Αποστολή {file.name}...")
        
        # Show file size
        file_size_mb = file.size / (1024 * 1024)
        if file_size_mb > 10:
            status.info(f"📤 Αποστολή {file.name} ({file_size_mb:.1f}MB - μπορεί να πάρει λίγο χρόνο)...")
        
        # Simple upload with increased timeout for large files
        files = {"file": (file.name, file.getvalue())}
        
        # Adjust timeout based on file size
        if file_size_mb > 20:
            timeout = 180  # 3 λεπτά για πολύ μεγάλα αρχεία
        elif file_size_mb > 5:
            timeout = 120  # 2 λεπτά για μεσαία αρχεία
        else:
            timeout = 60   # 1 λεπτό για μικρά αρχεία
        
        response = requests.post(
            f"{API_URL}/{endpoint}",
            files=files,
            timeout=timeout
        )
        
        status.empty()
        
        if response.status_code == 200:
            result = response.json()
            status_value = result.get("status")
            
            # ΑΥΤΗ είναι η σωστή έλεγξη:
            if status_value == "ok":
                return {"success": True, "data": result}
            elif status_value == "warning":
                return {"success": True, "data": result, "warning": True}  # Επίσης success αλλά με warning
            else:
                return {"error": True, "message": result.get("message", "Άγνωστο σφάλμα")}
        else:
            return {"error": True, "message": f"HTTP {response.status_code}: {response.text[:200]}"}
            
    except requests.exceptions.Timeout:
        return {"error": True, "message": f"⏰ Timeout ({timeout}s) - Το αρχείο είναι πολύ μεγάλο ή αργή σύνδεση"}
    except requests.exceptions.ConnectionError:
        return {"error": True, "message": "🔌 Δεν μπορώ να συνδεθώ με τον server"}
    except Exception as e:
        return {"error": True, "message": f"⚠️ Σφάλμα: {str(e)}"}

# =====================================
# ENHANCED RAG CHAT FUNCTION
# =====================================
def enhanced_rag_chat(scope: str, query: str, top_k: int = 3, chat_history=None):
    """Enhanced RAG chat with better context handling"""
    try:
        # Search for documents
        endpoint = f"{API_URL}/{scope}/search"
        resp = requests.post(
            endpoint, 
            json={"query": query, "top_k": top_k}, 
            timeout=30
        )
        
        if resp.status_code != 200:
            return {
                "answer": "⚠️ Σφάλμα στην αναζήτηση εγγράφων",
                "contexts": [],
                "metadatas": [],
                "error": True
            }
        
        data = resp.json()
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        
        if not docs:
            return {
                "answer": "Δεν βρέθηκαν σχετικά έγγραφα στη βάση δεδομένων.",
                "contexts": [],
                "metadatas": []
            }
        
        # Build enhanced context with conversation history
        system_prompt = "Απάντησε με βάση τα έγγραφα. Αν δεν υπάρχει πληροφορία στα έγγραφα, πες 'Δεν βρέθηκε πληροφορία στα έγγραφα'."
        
        if chat_history and len(chat_history) > 0:
            history_text = "Προηγούμενη συζήτηση:\n"
            for msg in chat_history[-4:]:  # Τα τελευταία 4 μηνύματα
                role = "Χρήστης" if msg.get("role") == "user" else "Βοηθός"
                history_text += f"{role}: {msg.get('content', '')}\n"
            system_prompt = history_text + "\n" + system_prompt
        
        # Build document context
        context_parts = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            source_info = f"[Πηγή {i+1}]"
            if meta and 'filename' in meta:
                source_info += f" από {meta['filename']}"
            if meta and 'page' in meta:
                source_info += f" (σελίδα {meta['page']})"
            
            context_parts.append(f"{source_info}:\n{doc[:400]}{'...' if len(doc) > 400 else ''}")
        
        context = "\n\n".join(context_parts)
        
        # OpenAI call
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Έγγραφα:\n{context}\n\nΕρώτηση: {query}\n\nΑπάντησε με βάση ΜΟΝΟ τα παραπάνω έγγραφα:"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "contexts": docs,
            "metadatas": metas,
            "sources_count": len(docs)
        }
        
    except Exception as e:
        return {
            "answer": f"⚠️ Σφάλμα κατά την επεξεργασία: {str(e)}",
            "contexts": [],
            "metadatas": [],
            "error": True
        }

# =====================================
# SHOW FILE PREVIEW FUNCTION
# =====================================
def show_file_preview(file):
    """Show small preview of uploaded file"""
    if hasattr(file, 'type') and file.type and file.type.startswith('image/'):
        try:
            img = Image.open(io.BytesIO(file.getvalue()))
            img.thumbnail((150, 150))
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(img, width=100)
            with col2:
                st.write(f"**{file.name}**")
                st.caption(f"Μέγεθος: {file.size/1024:.1f} KB")
        except Exception as e:
            st.write(f"📄 {file.name}")
    elif hasattr(file, 'type') and file.type and 'pdf' in file.type.lower():
        st.write(f"📄 {file.name} (PDF)")
    else:
        st.write(f"📄 {file.name}")

# =====================================
# FILE MANAGER
# =====================================
def list_files(path: Path):
    try:
        return sorted([p for p in path.iterdir() if p.is_file()], key=lambda x: x.name)
    except Exception:
        return []

# =====================================
# CREATE UPLOAD DIRECTORIES
# =====================================
GENERAL_UPLOAD_DIR = Path("uploads/general")
INVOICE_UPLOAD_DIR = Path("uploads/invoices")
GENERAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INVOICE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# INITIALIZE SESSION STATE
# =====================================
if "general_chat" not in st.session_state:
    st.session_state.general_chat = []
if "invoice_chat" not in st.session_state:
    st.session_state.invoice_chat = []
if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0
if "current_chat_input" not in st.session_state:
    st.session_state.current_chat_input = ""

# =====================================
# MAIN APP - CHECK BACKEND CONNECTION
# =====================================

st.markdown("### 🔍 Έλεγχος σύνδεσης με το backend...")
is_connected, message = test_backend_connection()

if not is_connected:
    st.markdown(f'<div class="error-box">❌ Δεν μπορώ να συνδεθώ στο backend: {message}</div>', unsafe_allow_html=True)
    
    st.markdown("### 🔧 Προσπάθησε τα εξής:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Επανεκκίνηση Backend"):
            st.info("Εκτέλεσε στον terminal: python main.py")
    
    with col2:
        if st.button("🔗 Άλλαξε API URL"):
            new_url = st.text_input("Νέα API URL:", value="http://localhost:8001", key="new_url")
            if new_url:
                st.session_state.api_url = new_url
                st.rerun()
    
    with col3:
        if st.button("📱 Test Connection"):
            st.code("""
            # Στον terminal:
            curl http://localhost:8001/health
            """)
    
    st.stop()  # Stop execution if backend is not connected

# Backend is connected - show success
st.markdown('<div class="success-box">✅ Συνδέθηκε με το backend!</div>', unsafe_allow_html=True)

# Mode selector
mode = st.radio(
    "Επιλογή Κατηγορίας:",
    ["Γενικά Έγγραφα", "Τιμολόγια"],
    horizontal=True,
    key="mode_selector"
)

st.divider()

# ======================================================================
# ===========================   GENERAL MODE   ==========================
# ======================================================================
if mode == "Γενικά Έγγραφα":
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📤 Upload", "📁 Αρχεία"])
    
    with tab1:
        st.header("💬 Chat για Γενικά Έγγραφα")
        
        # Εμφάνιση ιστορικού chat
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.general_chat:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
                        
                        # Προσθήκη πηγών αν υπάρχουν
                        if message.get("contexts") and len(message["contexts"]) > 0:
                            with st.expander(f"🔍 Πηγές ({len(message['contexts'])})"):
                                for i, (doc, meta) in enumerate(zip(message.get("contexts", []), 
                                                                  message.get("metadatas", []))):
                                    st.markdown(f"**Πηγή {i+1}**")
                                    if meta and 'filename' in meta:
                                        st.caption(f"Αρχείο: {meta['filename']}")
                                    st.text(doc[:400] + "..." if len(doc) > 400 else doc)
                                    st.divider()
        
        # Εισαγωγή χρήστη
        user_input = st.chat_input("Ρωτήστε κάτι για τα έγγραφά σας...")
        
        if user_input:
            # Προσθήκη ερώτησης χρήστη στο ιστορικό
            st.session_state.general_chat.append({"role": "user", "content": user_input})
            
            # Εμφάνιση ερώτησης
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Απάντηση του assistant
            with st.chat_message("assistant"):
                with st.spinner("🔍 Αναζήτηση εγγράφων και δημιουργία απάντησης..."):
                    try:
                        # Χρήση enhanced RAG chat
                        result = enhanced_rag_chat(
                            "general", 
                            user_input, 
                            top_k=3,
                            chat_history=st.session_state.general_chat
                        )
                        
                        # Εμφάνιση απάντησης
                        st.markdown(result["answer"])
                        
                        # Προσθήκη πηγών
                        if result.get("contexts") and len(result["contexts"]) > 0:
                            with st.expander(f"🔍 Πηγές ({len(result['contexts'])})"):
                                for i, (doc, meta) in enumerate(zip(result["contexts"], result["metadatas"])):
                                    st.markdown(f"**Πηγή {i+1}**")
                                    if meta and 'filename' in meta:
                                        st.caption(f"Αρχείο: {meta['filename']}")
                                    st.text(doc[:400] + "..." if len(doc) > 400 else doc)
                                    st.divider()
                        
                        # Προσθήκη απάντησης στο ιστορικό
                        st.session_state.general_chat.append({
                            "role": "assistant", 
                            "content": result["answer"],
                            "contexts": result.get("contexts", []),
                            "metadatas": result.get("metadatas", [])
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Σφάλμα: {str(e)}"
                        st.error(error_msg)
                        st.session_state.general_chat.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
        
        # Κουμπί καθαρισμού chat
        if st.button("🧹 Καθαρισμός Chat", use_container_width=True):
            st.session_state.general_chat = []
            st.rerun()
    
    with tab2:
        st.header("📤 Upload Γενικών Εγγράφων")
        
        file = st.file_uploader(
            "Επιλέξτε αρχείο (PDF, TXT)",
            type=["pdf", "txt"],
            key=f"gen_upload_{st.session_state.upload_counter}"
        )
        
        if file:
            st.info(f"📄 Επιλέχθηκε: {file.name}")
            
            if st.button("Ανέβασμα", type="primary", use_container_width=True):
                with st.spinner("📤 Αποστολή αρχείου..."):
                    result = simple_upload(file, "general/upload")
                    
                    if result.get("success"):
                        st.success(f"✅ Το {file.name} ανέβηκε επιτυχώς!")
                        
                        if result.get("warning"):
                            st.info(f"⚠️ {result['data'].get('message', 'Προειδοποίηση')}")
                        
                        # Εμφάνιση λεπτομερειών
                        with st.expander("📊 Λεπτομέρειες"):
                            st.json(result["data"])
                        
                        st.session_state.upload_counter += 1
                        time.sleep(1)
                        st.rerun()
                    elif result.get("error"):
                        st.error(f"❌ {result['message']}")
    
    with tab3:
        st.header("📁 Γενικά Αρχεία")
        
        files = list_files(GENERAL_UPLOAD_DIR)
        
        if not files:
            st.info("Δεν υπάρχουν ανεβασμένα αρχεία.")
        else:
            st.write(f"**Σύνολο αρχείων:** {len(files)}")
            
            for idx, file_path in enumerate(files):
                col1, col2, col3 = st.columns([6, 2, 1])
                
                with col1:
                    file_size = file_path.stat().st_size / 1024  # KB
                    file_date = time.ctime(file_path.stat().st_mtime)
                    st.write(f"📄 **{file_path.name}**")
                    st.caption(f"Μέγεθος: {file_size:.1f} KB | Τροποποιήθηκε: {file_date}")
                
                with col2:
                    if st.button("📄 Προεπισκόπηση", key=f"preview_{idx}"):
                        try:
                            with open(file_path, 'rb') as f:
                                if file_path.suffix.lower() == '.txt':
                                    content = f.read().decode('utf-8', errors='ignore')
                                    with st.expander(f"Προεπισκόπηση: {file_path.name}"):
                                        st.text_area("Περιεχόμενο", content[:2000], height=300)
                                elif file_path.suffix.lower() == '.pdf':
                                    st.info("PDF προεπισκόπηση (απαιτείται ειδική βιβλιοθήκη)")
                        except Exception as e:
                            st.error(f"Σφάλμα: {e}")
                
                with col3:
                    if st.button("🗑️", key=f"del_{idx}"):
                        try:
                            file_path.unlink()
                            st.success(f"Το αρχείο {file_path.name} διαγράφηκε!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Σφάλμα: {e}")

# ======================================================================
# ===========================   INVOICE MODE   ==========================
# ======================================================================
else:  # Τιμολόγια
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📤 Upload", "📁 Αρχεία"])
    
    with tab1:
        st.header("💬 Chat για Τιμολόγια")
        
        # Εμφάνιση ιστορικού chat
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.invoice_chat:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
                        
                        # Προσθήκη πηγών αν υπάρχουν
                        if message.get("contexts") and len(message["contexts"]) > 0:
                            with st.expander(f"🔍 Πηγές ({len(message['contexts'])})"):
                                for i, (doc, meta) in enumerate(zip(message.get("contexts", []), 
                                                                  message.get("metadatas", []))):
                                    st.markdown(f"**Πηγή {i+1}**")
                                    if meta and 'filename' in meta:
                                        st.caption(f"Αρχείο: {meta['filename']}")
                                    st.text(doc[:400] + "..." if len(doc) > 400 else doc)
                                    st.divider()
        
        # Εισαγωγή χρήστη
        user_input = st.chat_input("Ρωτήστε κάτι για τα τιμολόγιά σας...")
        
        if user_input:
            # Προσθήκη ερώτησης χρήστη στο ιστορικό
            st.session_state.invoice_chat.append({"role": "user", "content": user_input})
            
            # Εμφάνιση ερώτησης
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Απάντηση του assistant
            with st.chat_message("assistant"):
                with st.spinner("🔍 Αναζήτηση τιμολογίων και δημιουργία απάντησης..."):
                    try:
                        # Χρήση enhanced RAG chat
                        result = enhanced_rag_chat(
                            "invoices", 
                            user_input, 
                            top_k=3,
                            chat_history=st.session_state.invoice_chat
                        )
                        
                        # Εμφάνιση απάντησης
                        st.markdown(result["answer"])
                        
                        # Προσθήκη πηγών
                        if result.get("contexts") and len(result["contexts"]) > 0:
                            with st.expander(f"🔍 Πηγές ({len(result['contexts'])})"):
                                for i, (doc, meta) in enumerate(zip(result["contexts"], result["metadatas"])):
                                    st.markdown(f"**Πηγή {i+1}**")
                                    if meta and 'filename' in meta:
                                        st.caption(f"Αρχείο: {meta['filename']}")
                                    st.text(doc[:400] + "..." if len(doc) > 400 else doc)
                                    st.divider()
                        
                        # Προσθήκη απάντησης στο ιστορικό
                        st.session_state.invoice_chat.append({
                            "role": "assistant", 
                            "content": result["answer"],
                            "contexts": result.get("contexts", []),
                            "metadatas": result.get("metadatas", [])
                        })
                        
                    except Exception as e:
                        error_msg = f"❌ Σφάλμα: {str(e)}"
                        st.error(error_msg)
                        st.session_state.invoice_chat.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
        
        # Κουμπί καθαρισμού chat
        if st.button("🧹 Καθαρισμός Chat", use_container_width=True, key="clear_inv_chat"):
            st.session_state.invoice_chat = []
            st.rerun()
    
    with tab2:
        st.header("📤 Upload Τιμολογίου")
        
        # Single file upload
        st.subheader("📄 Απλό Ανέβασμα")
        file = st.file_uploader(
            "Επιλέξτε τιμολόγιο (PDF, JPG, PNG)",
            type=["pdf", "jpg", "jpeg", "png"],
            key=f"inv_upload_{st.session_state.upload_counter}"
        )
        
        if file:
            show_file_preview(file)
            
            if st.button("Ανέβασμα & OCR", type="primary", use_container_width=True):
                with st.spinner("📤 Ανέβασμα και επεξεργασία..."):
                    result = simple_upload(file, "invoices/upload")
                    
                    if result.get("success"):
                        st.success(f"✅ Το {file.name} επεξεργάστηκε επιτυχώς!")
                        st.balloons()
                        
                        data = result.get("data", {})
                        
                        # Εμφάνιση αποτελεσμάτων
                        with st.expander("📊 Σύνοψη"):
                            st.json(data)
                        
                        if data.get("ocr_preview"):
                            with st.expander("🔤 OCR Προεπισκόπηση"):
                                st.text(data["ocr_preview"][:1500])
                        
                        if data.get("parsed_invoice"):
                            with st.expander("📋 Δομημένα Δεδομένα"):
                                st.json(data["parsed_invoice"])
                        
                        st.session_state.upload_counter += 1
                        time.sleep(2)
                        st.rerun()
                    
                    elif result.get("error"):
                        st.error(f"❌ {result['message']}")
        
        st.divider()
        
        # Batch upload
        st.subheader("📦 Πολλαπλά Αρχεία")
        uploaded_files = st.file_uploader(
            "Επιλέξτε πολλά αρχεία",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"batch_upload_{st.session_state.upload_counter}"
        )
        
        if uploaded_files:
            st.info(f"📁 Επιλέχθηκαν {len(uploaded_files)} αρχεία")
            
            if st.button("Ανέβασμα Όλων", type="secondary", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.info(f"📤 Ανέβασμα {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                    
                    result = simple_upload(uploaded_file, "invoices/upload")
                    results.append({
                        "filename": uploaded_file.name,
                        "result": result
                    })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.empty()
                
                # Display summary
                success_count = sum(1 for r in results if r["result"].get("success"))
                st.success(f"✅ {success_count}/{len(results)} αρχεία ανέβηκαν επιτυχώς!")
                
                with st.expander("📋 Λεπτομερή Αποτελέσματα"):
                    for res in results:
                        filename = res["filename"]
                        result = res["result"]
                        
                        if result.get("success"):
                            st.success(f"✅ {filename}")
                        else:
                            st.error(f"❌ {filename}: {result.get('message', 'Σφάλμα')}")
                
                if st.button("🔄 Καθαρισμός Λίστας", use_container_width=True):
                    st.session_state.upload_counter += 1
                    st.rerun()
    
    with tab3:
        st.header("📁 Αρχεία Τιμολογίων")
        
        files = list_files(INVOICE_UPLOAD_DIR)
        
        if not files:
            st.info("Δεν υπάρχουν ανεβασμένα τιμολόγια.")
        else:
            st.write(f"**Σύνολο τιμολογίων:** {len(files)}")
            
            for idx, file_path in enumerate(files):
                col1, col2, col3 = st.columns([6, 2, 1])
                
                with col1:
                    file_size = file_path.stat().st_size / 1024  # KB
                    file_date = time.ctime(file_path.stat().st_mtime)
                    
                    # Προσδιορισμός τύπου αρχείου
                    if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        file_icon = "🖼️"
                    elif file_path.suffix.lower() == '.pdf':
                        file_icon = "📄"
                    else:
                        file_icon = "📎"
                    
                    st.write(f"{file_icon} **{file_path.name}**")
                    st.caption(f"Μέγεθος: {file_size:.1f} KB | Τροποποιήθηκε: {file_date}")
                
                with col2:
                    if st.button("👁️ Προεπισκόπηση", key=f"inv_preview_{idx}"):
                        try:
                            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                img = Image.open(file_path)
                                img.thumbnail((300, 300))
                                st.image(img, caption=file_path.name)
                            elif file_path.suffix.lower() == '.pdf':
                                st.info("PDF προεπισκόπηση (απαιτείται ειδική βιβλιοθήκη)")
                        except Exception as e:
                            st.error(f"Σφάλμα: {e}")
                
                with col3:
                    if st.button("🗑️", key=f"inv_del_{idx}"):
                        try:
                            file_path.unlink()
                            st.success(f"Το αρχείο {file_path.name} διαγράφηκε!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Σφάλμα: {e}")

# =====================================
# FOOTER
# =====================================
st.divider()
st.caption(f"AInteG Management Console | Backend: {API_URL} | © 2024")