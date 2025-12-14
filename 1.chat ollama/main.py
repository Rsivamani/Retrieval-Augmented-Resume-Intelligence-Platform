import streamlit as st
import tempfile
import os
import hashlib
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------
st.set_page_config(page_title="Resume Intelligence System", layout="centered")
st.title("📄 Resume Intelligence System")

# --------------------------------------------------
# Vector DB (Persistent)
# --------------------------------------------------
@st.cache_resource
def load_vector_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory="./resume_chroma",
        embedding_function=embeddings
    )

db = load_vector_db()
st.caption(f"📦 Indexed chunks: {db._collection.count()}")

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = Ollama(model="llama3", temperature=0)

# --------------------------------------------------
# Prompt (NO LIES, NO OVERCLAIMS)
# --------------------------------------------------
PROMPT = PromptTemplate.from_template("""
You are a resume question-answering system.

Rules:
- Use ONLY the provided context.
- If the context is insufficient, respond with:
  "INSUFFICIENT CONTEXT IN RESUME"
- Do NOT infer, assume, or fabricate information.

Context:
{context}

Question:
{question}

Answer:
""")

# --------------------------------------------------
# Resume-Specific Utilities
# --------------------------------------------------
RESUME_SECTIONS = [
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "summary"
]

def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def detect_section(text: str) -> str:
    text_lower = text.lower()
    for section in RESUME_SECTIONS:
        if section in text_lower:
            return section
    return "general"

# --------------------------------------------------
# Resume Ingestion (FAANG-LEVEL)
# --------------------------------------------------
def ingest_resume(uploaded_file):
    file_bytes = uploaded_file.read()
    file_hash = compute_file_hash(file_bytes)

    # Deduplication check
    existing = db.similarity_search(
        "resume",
        filter={"file_hash": file_hash},
        k=1
    )
    if existing:
        st.warning("⚠️ This resume is already indexed.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyMuPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    documents: List[Document] = []

    for page in pages:
        section = detect_section(page.page_content)
        chunks = splitter.split_text(page.page_content)

        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "file_hash": file_hash,
                        "section": section,
                        "source": uploaded_file.name
                    }
                )
            )

    db.add_documents(documents)
    db.persist()
    os.remove(tmp_path)

    st.success("✅ Resume indexed successfully")

# --------------------------------------------------
# Upload UI
# --------------------------------------------------
st.subheader("Upload Resume PDF")
uploaded_file = st.file_uploader("Upload resume", type=["pdf"])

if uploaded_file:
    with st.spinner("Indexing resume..."):
        ingest_resume(uploaded_file)

# --------------------------------------------------
# Question Answering
# --------------------------------------------------
st.subheader("Ask Resume Questions")
question = st.text_input("Enter your question")

SIMILARITY_THRESHOLD = 0.35
TOP_K = 4

if question:
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": TOP_K,
            "score_threshold": SIMILARITY_THRESHOLD
        }
    )

    docs = retriever.invoke(question)

    if not docs:
        st.warning("INSUFFICIENT CONTEXT IN RESUME")
    else:
        context = "\n\n".join(
            f"[{doc.metadata.get('section','general')}] {doc.page_content}"
            for doc in docs
        )

        final_prompt = PROMPT.format(
            context=context,
            question=question
        )

        with st.spinner("Generating answer..."):
            answer = llm.invoke(final_prompt)

        st.subheader("🧠 Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Evidence"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Chunk {i} | Section: {doc.metadata['section']}**")
                st.write(doc.page_content)