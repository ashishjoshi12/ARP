#A library for creating web applications with simple Python scripts.
import streamlit as st  

#A class for reading PDF files
from PyPDF2 import PdfReader 

#A text splitter for breaking down large texts into smaller chunks.
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os  #Provides a way to interact with the operating system.

#Embeddings model from LangChain for Google Generative AI.
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#Google Generative AI library
import google.generativeai as genai

#A library for similarity search and clustering of dense vectors.
from langchain.vectorstores import FAISS

#A class for interacting with the Google Generative AI model for chat-based conversations.
from langchain_google_genai import ChatGoogleGenerativeAI

#A function to load a question-answering chain from LangChain.
from langchain.chains.question_answering import load_qa_chain

#A template for creating prompts for the conversational model
from langchain.prompts import PromptTemplate

#A function to load environment variables from a file (.env).
from dotenv import load_dotenv


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



#Function to extract text from PDF files:
#Takes a list of PDF files as input.
#Iterates through each PDF file, reads its pages, and extracts text.
#Concatenates the text from all pages and returns it.
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


#Function to split text into chunks:
#Takes a text as input.
#Uses a text splitter to break the text into chunks of specified size and overlap.
#Returns the list of text chunks.
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#Function to create a vector store from text chunks:
#Takes a list of text chunks as input.
#Uses Google Generative AI embeddings to generate embeddings for each chunk.
#Creates a FAISS vector store from the embeddings and saves it locally.
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


#Function to get a conversational chain:
#Defines a template for prompts.
#Configures the Google Generative AI model for chat-based conversations.
#Creates a conversational chain using the loaded model and prompt template.
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


#Function to handle user input and generate responses:
#Uses Google Generative AI embeddings to load the FAISS vector store.
#Performs similarity search to find relevant documents for the user's question.
#Calls the conversational chain to generate a response based on the user's question and context.
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])



#Main function for the Streamlit web application:
#Configures the Streamlit page.
#Provides a simple web interface for users to input questions and upload PDF files.
#Processes PDF files, extracts text, creates vector stores, and handles user interactions
def main():
    st.set_page_config("Chat PDF")
    st.header("PDF Summarizing")

    user_question = st.text_input("Ask a Question")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit ", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Have patience..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Start Asking your questions")

if __name__ == "__main__":
    main()