import sys, os
import pdfplumber

def extract_text(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""     # extract page text; handle None for image-only pages
            text_parts.append(t)
    return "\n".join(text_parts).strip()    # join pages; trim leading/trailing whitespace

if __name__ == "__main__":
    path = sys.argv[1]                      # allow CLI usage: python extract_text.py file.pdf
    print(extract_text(path))
