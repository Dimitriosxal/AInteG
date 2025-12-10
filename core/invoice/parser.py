import os
import json
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------------
# LOAD .env από το ROOT του project
# -------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]   # AInteG/
load_dotenv(BASE_DIR / ".env")

def get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------------------
# SYSTEM PROMPT για LLM parsing
# -------------------------------------
INVOICE_SYSTEM_PROMPT = """
You are an expert system for extracting structured data from OCR invoice text.
Return ONLY valid JSON. No explanations.

Extract:
- customer name (string)
- vat number if found (string or null)
- invoice number (string)
- invoice date (string or null)
- series (string or null)

- product lines:
    - description
    - quantity (number)
    - unit_price (number)
    - line_total (number)

- totals:
    - subtotal
    - vat_amount
    - grand_total

If a field is missing set it to null.
"""


# ----------------------------------------------------------
# REGEX FALLBACK — used if GPT fails
# ----------------------------------------------------------
def regex_fallback(text: str):
    def clean(t):
        return re.sub(r"[ \t]+", " ", t)

    t = clean(text)

    # Supplier
    supplier = None
    for line in t.split("\n")[:10]:
        if line.strip().isupper() and len(line.strip()) > 3:
            supplier = line.strip()
            break

    # Invoice number
    invoice_num = None
    patterns = [
        r"ΤΙΜ(?:\.)?\s*№?\s*(\d+)",
        r"ΤΙΜΟΛΟΓΙΟ\s*№?\s*(\d+)",
        r"INV\s*(\d+)",
        r"Invoice\s*(\d+)",
    ]
    for p in patterns:
        m = re.search(p, t, re.IGNORECASE)
        if m:
            invoice_num = m.group(1)
            break

    # Total amount (last price)
    prices = re.findall(r"(\d{1,3}(?:\.\d{3})*,\d{2})", t)
    total = prices[-1] if prices else None

    # Product lines
    products = []
    pattern = r"(.*?)[ ]+(\d+)[ ]+(\d+(?:,\d{2})?)[ ]+(\d+(?:,\d{2})?)"

    for line in t.split("\n"):
        m = re.search(pattern, line)
        if m:
            products.append({
                "description": m.group(1).strip(),
                "quantity": m.group(2),
                "unit_price": m.group(3),
                "line_total": m.group(4),
            })

    return {
        "supplier": supplier,
        "invoice_number": invoice_num,
        "total_amount": total,
        "products": products
    }


# ----------------------------------------------------------
# MAIN PARSER — LLM → fallback_regex
# ----------------------------------------------------------
def parse_invoice_text(text: str) -> dict:

    client = get_client()
    user_prompt = f"Extract the invoice data from the following OCR text:\n\n{text}"

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": INVOICE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        return {
            "source": "fallback_regex",
            "error": str(e),
            "data": regex_fallback(text)
        }
