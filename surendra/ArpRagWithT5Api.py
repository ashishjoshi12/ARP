from flask import Flask, request, jsonify
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pdfplumber
import os

# Initialize Flask app
app = Flask(__name__)

# Load necessary models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
t5_model = T5ForConditionalGeneration.from_pretrained('T5-large')
t5_tokenizer = T5Tokenizer.from_pretrained('T5-large')

# Global variables
text_data = ""  # You would load this from your PDF
text_chunks = []
faiss_index = None

# Step 1: Load PDF and Preprocess Data
def extract_text_tables(pdf_path):
    print(f"Extracting text and tables from {pdf_path}...")
    with pdfplumber.open(pdf_path) as pdf:
        all_text = []
        all_tables = []
        for page in pdf.pages:
            page_text = page.extract_text()
            all_text.append(page_text)
            tables = page.extract_tables()
            all_tables.extend(tables)
        print(f"Extracted {len(all_text)} pages of text.")
        print(f"Extracted {len(all_tables)} tables.")
        return "\n".join(all_text), all_tables

# Preprocess and split the extracted text (only runs when the server starts)
def preprocess_pdf(pdf_path):
    global text_data, text_chunks, faiss_index

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    print(f"Preprocessing PDF: {pdf_path}")
    
    text_data, table_data = extract_text_tables(pdf_path)
    text_chunks = text_data.split('\n')
    print(f"Text split into {len(text_chunks)} chunks.")

    embeddings = embedding_model.encode(text_chunks, convert_to_numpy=True)
    print(f"Generated embeddings for {len(text_chunks)} text chunks.")

    # Setup FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    print("FAISS index created and embeddings added.")

# Path to your PDF (replace with actual path)
pdf_path = "D:/surendra/annualReview/gxocompany.pdf"  
preprocess_pdf(pdf_path)

# Function to retrieve relevant chunks
def retrieve(query, k=5):
    print(f"Retrieving top {k} relevant chunks for query: {query}")
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, k)
    print(f"Found {len(distances[0])} matches.")
    return [text_chunks[i] for i in indices[0]]

# Function to generate commentary based on context and query
def generate_commentary(context, query):
    print(f"Generating commentary for query: {query}")
    input_text = f"question: {query} context: {context}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)
    response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated response: {response[:100]}...")  # Print the first 100 characters for brevity
    return response

# Flask route to handle the query
@app.route("/ask/", methods=["POST"])
def ask_question():
    try:
        req_data = request.get_json()
        query = req_data['question']
        print(f"Received query: {query}")
        retrieved_chunks = retrieve(query)
        context = " ".join(retrieved_chunks)
        response = generate_commentary(context, query)
        print(f"Sending response: {response[:100]}...")  # Print the first 100 characters for brevity
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

# Optional Health Check route
@app.route("/health", methods=["GET"])
def health_check():
    print("Health check received.")
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    app.run(debug=True)


#curl command to call api curl -X POST http://localhost:5000/ask/ -H "Content-Type: application/json" -d "{\"question\": \"how much is the annual revenue growth for 2023\"}"