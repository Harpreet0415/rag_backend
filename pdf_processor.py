import PyPDF2
import re
import os
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> Dict:
    """Extract text from a PDF file along with page metadata."""
    pages_data = []
    full_text = ""
    
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = clean_text(text)
            if text.strip():
                pages_data.append({
                    "page_num": page_num + 1,
                    "text": text
                })
                full_text += f"\n\n[Page {page_num + 1}]\n{text}"
    
    return {
        "full_text": full_text.strip(),
        "pages": pages_data,
        "num_pages": num_pages
    }


def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Fix hyphenated line breaks
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    return text.strip()


def chunk_text(pages_data: List[Dict], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict]:
    """Split page texts into overlapping chunks with page metadata."""
    chunks = []
    
    for page_info in pages_data:
        page_num = page_info["page_num"]
        text = page_info["text"]
        
        words = text.split()
        if not words:
            continue
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text_str = " ".join(chunk_words)
            
            if len(chunk_text_str.strip()) > 50:  # skip very short chunks
                chunks.append({
                    "text": chunk_text_str,
                    "page_num": page_num,
                    "chunk_id": len(chunks),
                    "start_word": start,
                    "end_word": min(end, len(words))
                })
            
            if end >= len(words):
                break
            start += chunk_size - chunk_overlap
    
    return chunks
