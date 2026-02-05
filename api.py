from fastapi import FastAPI
from rag import get_qa_chain

app = FastAPI()

qa = get_qa_chain()


@app.get("/")
def home():
    return {"message": "Pharma RAG API running"}


@app.post("/ask")
def ask(question: str):
    result = qa(question)
    return {"answer": result["result"]}
