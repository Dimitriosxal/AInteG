import io
import os
import fitz
import pytesseract
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------------------
# Utility: quality score for OCR text
# -----------------------------------------
def score_text(text: str) -> float:
    if not text:
        return 0

    # % of alphabetic characters
    alpha = sum(c.isalpha() for c in text)
    ratio = alpha / max(1, len(text))

    # penalize extremely short or gibberish blocks
    if len(text) < 30:
        ratio *= 0.3

    return ratio


# -----------------------------------------
# Tesseract OCR for images
# -----------------------------------------
def ocr_image_tesseract(image_bytes: bytes) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang="ell+eng")
        return text
    except Exception:
        return ""


# -----------------------------------------
# Tesseract OCR for PDFs
# -----------------------------------------
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
# OpenAI Vision OCR (image bytes)
# -----------------------------------------
def ocr_image_openai(image_bytes: bytes) -> str:
    try:
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Extract all text from this image:"},
                        {"type": "input_image", "image": image_bytes}
                    ]
                }
            ]
        )
        return result.choices[0].message.content or ""
    except Exception:
        return ""


# -----------------------------------------
# OpenAI Vision OCR for PDFs
# -----------------------------------------
def ocr_pdf_openai(path: str) -> str:
    text = ""
    try:
        pdf = fitz.open(path)
        for page in pdf:
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Extract all text from this image:"},
                            {"type": "input_image", "image": img_bytes}
                        ]
                    }
                ]
            )
            page_text = resp.choices[0].message.content or ""
            text += "\n" + page_text

        return text
    except Exception:
        return ""


# -----------------------------------------
# MAIN HYBRID OCR ENTRY
# -----------------------------------------
def ocr_to_text(path: str, filename: str):
    filename = filename.lower()

    # Load file bytes once
    content = open(path, "rb").read()

    # Decide if image or pdf
    is_image = filename.endswith((".jpg", ".jpeg", ".png"))
    is_pdf = filename.endswith(".pdf")

    # 1) Run Tesseract
    if is_image:
        t_text = ocr_image_tesseract(content)
    else:
        t_text = ocr_pdf_tesseract(path)

    # 2) Run OpenAI Vision
    if is_image:
        o_text = ocr_image_openai(content)
    else:
        o_text = ocr_pdf_openai(path)

    # 3) Score both
    score_t = score_text(t_text)
    score_o = score_text(o_text)

    print(f"[OCR] Tesseract Score: {score_t:.3f}, OpenAI Score: {score_o:.3f}")

    # 4) Pick best
    if score_o >= score_t:
        return o_text
    else:
        return t_text
