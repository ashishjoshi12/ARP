# langgraph_orchestration.py

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import requests
from typing import Dict, Any, List
import json
import os

# --- Custom Company Model Wrapper ---
class CompanyLLM:
    def __init__(self, ingest_url: str, query_url: str):
        self.ingest_url = ingest_url
        self.query_url = query_url

    def ask(self, documents: str, question: str) -> str:
        ingest_resp = requests.post(self.ingest_url, json={"docs": documents})
        token = ingest_resp.json()["token"]
        query_resp = requests.post(self.query_url, json={"question": question, "token": token})
        return query_resp.json()["answer"]

# --- Load API Specs from File ---
def load_api_specs(file_path: str) -> List[Document]:
    with open(file_path, 'r') as f:
        specs = json.load(f)
    return [Document(page_content=spec["content"], metadata={"name": spec["name"]}) for spec in specs]

# --- Embedding & Vector Store Setup ---
def get_vectorstore_from_specs(specs: List[Document], embedding) -> FAISS:
    if not os.path.exists("api_spec_index"):
        vectorstore = FAISS.from_documents(specs, embedding)
        vectorstore.save_local("api_spec_index")
    else:
        vectorstore = FAISS.load_local("api_spec_index", embedding)
    return vectorstore

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
specs = load_api_specs("api_specs.json")
vectorstore = get_vectorstore_from_specs(specs, embedding)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- State Definition ---
class GraphState(Dict[str, Any]):
    question: str
    docs: List[Document]
    selected_api: str
    api_response: Any
    final_answer: str

# --- Helper Functions ---
def join_docs(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])

# --- Nodes ---
def retrieve_api_docs(state: GraphState) -> GraphState:
    docs = retriever.invoke(state["question"])
    return {**state, "docs": docs}

def select_api_node(llm: CompanyLLM):
    def inner(state: GraphState) -> GraphState:
        spec_text = join_docs(state["docs"])
        api_name = llm.ask(spec_text, state["question"])
        return {**state, "selected_api": api_name}
    return inner

def call_selected_api(state: GraphState) -> GraphState:
    api_name = state["selected_api"]
    # Dummy implementation — customize this with actual logic
    if api_name == "getUserTransactions":
        resp = requests.get("https://internal.api/users/transactions?userId=123")
        return {**state, "api_response": resp.json()}
    return {**state, "api_response": {"error": "Unknown API"}}

def generate_final_answer(llm: CompanyLLM):
    def inner(state: GraphState) -> GraphState:
        context = f"""
        User Question: {state['question']}

        API Specs:
        {join_docs(state['docs'])}

        API Response:
        {state['api_response']}
        """
        final = llm.ask(context, "Provide a final answer based on this.")
        return {**state, "final_answer": final}
    return inner

# --- Graph Definition ---
def create_langgraph_flow(llm: CompanyLLM):
    graph = StateGraph(GraphState)

    graph.add_node("retrieve_docs", retrieve_api_docs)
    graph.add_node("select_api", select_api_node(llm))
    graph.add_node("call_api", call_selected_api)
    graph.add_node("final_answer", generate_final_answer(llm))

    graph.set_entry_point("retrieve_docs")
    graph.add_edge("retrieve_docs", "select_api")
    graph.add_edge("select_api", "call_api")
    graph.add_edge("call_api", "final_answer")
    graph.add_edge("final_answer", END)

    return graph.compile()

# --- Usage ---
if __name__ == "__main__":
    llm = CompanyLLM(
        ingest_url="https://your.company.api/ingest",
        query_url="https://your.company.api/query"
    )

    workflow = create_langgraph_flow(llm)

    question = "How do I get user transactions for user ID 123?"
    result = workflow.invoke({"question": question})
    print("Final Answer:\n", result["final_answer"])
