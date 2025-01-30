import fitz  # type: ignore # PyMuPDF
from transformers import AutoTokenizer, AutoModel
import torch

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

pdf_path = r"C:\Users\rajes\OneDrive\Documents\GitHub\ARP\2023-annual-report.pdf"
pdf_text = extract_text_from_pdf(pdf_path)


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap  # Overlap for context
    return chunks

chunks = chunk_text(pdf_text)

print(chunks[0])

# Load a pre-trained model
model_name = "sentence-transformers/all-MiniLM-L6-v2"       
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

vectorized_chunks = [embed_text(chunk) for chunk in chunks]
