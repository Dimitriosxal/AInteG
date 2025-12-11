import io
import os
import fitz
import pytesseract
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import base64
load_dotenv()

# -----------------------------------------
# Setup
# -----------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------------------
# Utility: OCR quality scoring
# -----------------------------------------
def score_text(text: str) -> float:
    """Returns score 0–1 based on readability."""
    if not text:
        return 0.0

    alpha = sum(c.isalpha() for c in text)
    ratio = alpha / max(1, len(text))

    if len(text) < 30:
        ratio *= 0.3

    return ratio


# -----------------------------------------
# TESSERACT OCR
# -----------------------------------------
def ocr_image_tesseract(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img, lang="ell+eng")
    except Exception:
        return ""


def ocr_pdf_tesseract(path: str) -> str:
    text = ""
    try:
        pdf = fitz.open(path)
        for page in pdf:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_bytes))
            page_text = pytesseract.image_to_string(img, lang="ell+eng")
            text += "\n" + page_text
        return text
    except Exception:
        return ""


# -----------------------------------------
# OPENAI OCR (Vision)
# -----------------------------------------
def openai_ocr_image(image_bytes: bytes) -> str:
    """OCR για εικόνα με OpenAI Vision."""
    try:
        # Encode to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # ή "gpt-4o" για καλύτερο OCR
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[ERROR] OpenAI OCR failed: {e}")
        return ""


def openai_ocr_pdf(path: str) -> str:
    """OCR για PDF - βελτιωμένη έκδοση."""
    text = ""
    try:
        pdf = fitz.open(path)
        
        # Περίγραψε το max pages για να μην κολλάει
        max_pages = min(20, len(pdf))  # Μέχρι 20 σελίδες
        
        for page_num in range(max_pages):
            page = pdf[page_num]
            
            # Μείωσε DPI για ταχύτερη επεξεργασία
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            
            page_text = openai_ocr_image(img_bytes)
            if page_text:
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            # Προαιρετικό: Προσθήκη delay για rate limits
            import time
            if page_num % 5 == 0:
                time.sleep(0.5)
        
        return text
        
    except Exception as e:
        print(f"[ERROR] PDF OCR failed: {e}")
        return ""


# -----------------------------------------
# MAIN HYBRID OCR FUNCTION
# -----------------------------------------
def ocr_to_text(path: str, filename: str) -> str:
    print("[DEBUG] ocr_to_text START")

    filename = filename.lower()

    # Load bytes
    try:
        with open(path, "rb") as f:
            file_bytes = f.read()
        print("[DEBUG] File loaded")
    except Exception as e:
        print("[DEBUG] FAILED to load file:", e)
        return ""

    is_image = filename.endswith((".jpg", ".jpeg", ".png"))
    is_pdf = filename.endswith(".pdf")

    # -----------------------
    # 1) Tesseract OCR
    # -----------------------
    print("[DEBUG] Running Tesseract...")
    if is_image:
        t_text = ocr_image_tesseract(file_bytes)
    else:
        t_text = ocr_pdf_tesseract(path)
    print("[DEBUG] Tesseract DONE (length:", len(t_text), ")")

    # -----------------------
    # 2) OpenAI OCR
    # -----------------------
    print("[DEBUG] Running OpenAI OCR...")
    if is_image:
        o_text = openai_ocr_image(file_bytes)
    else:
        o_text = openai_ocr_pdf(path)
    print("[DEBUG] OpenAI OCR DONE (length:", len(o_text), ")")

    # -----------------------
    # 3) SCORE
    # -----------------------
    print("[DEBUG] Scoring...")
    score_t = score_text(t_text)
    score_o = score_text(o_text)
    print("[DEBUG] Scoring DONE")
    print(f"[DEBUG] Scores → Tesseract: {score_t:.3f}, OpenAI: {score_o:.3f}")

    # -----------------------
    # 4) PICK BEST
    # -----------------------
    if score_o > score_t:
        print("[DEBUG] SELECTED: OpenAI OCR")
        return o_text

    print("[DEBUG] SELECTED: Tesseract OCR")
    return t_text
