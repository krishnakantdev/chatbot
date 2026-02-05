from dotenv import load_dotenv
load_dotenv()

# Modern imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA


# -----------------------------
# Load documents
# -----------------------------
def load_documents():
    loader = PyPDFLoader("data/drug_manual.pdf")
    return loader.load()


# -----------------------------
# Split into chunks
# -----------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


# -----------------------------
# Create vector DB (FREE embeddings)
# -----------------------------
def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


# -----------------------------
# Build QA chain (Groq LLM)
# -----------------------------
def build_qa_chain(db):

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa


# -----------------------------
# Main entry
# -----------------------------
def get_qa_chain():
    docs = load_documents()
    chunks = split_documents(docs)
    db = create_vector_store(chunks)
    return build_qa_chain(db)
