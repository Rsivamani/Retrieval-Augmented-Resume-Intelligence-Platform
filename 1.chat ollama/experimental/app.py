import streamlit as st
import tempfile
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Resume RAG", layout="centered")
st.title("Resume Pdf Web Browswer")

# ------------------ Load Vector DB (Cached) ------------------
@st.cache_resource
def load_vector_db():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory="./resume_chroma",
        embedding_function=embeddings
    )

db = load_vector_db()
st.caption(f"📦 Documents in DB: {db._collection.count()}")

# ------------------ LLM + Prompt ------------------
llm = Ollama(model="llama3", temperature=0)

prompt = PromptTemplate.from_template("""
You are a resume assistant.
Answer ONLY using the context below.
If the answer is not present, say "Not found in resume".

Context:
{context}

Question:
{question}
""")

# ------------------ PDF Upload ------------------
st.subheader("Upload Resume PDF")

uploaded_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

def ingest_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    db.add_documents(chunks)
    db.persist()

    os.remove(tmp_path)

if uploaded_file:
    with st.spinner("Processing and saving resume..."):
        ingest_pdf(uploaded_file)
    st.success("✅ Resume added succesfully")

# ------------------ Question Input ------------------
st.subheader("💬 Ask Questions")

question = st.text_input(
    "Ask a question about the uploaded resume",
    key="resume_question"
)

# ------------------ RAG Pipeline ------------------
if question:
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(question)

    if not docs:
        st.warning("Not found in resume.")
    else:
        context = "\n".join(doc.page_content for doc in docs)

        final_prompt = prompt.format(
            context=context,
            question=question
        )

        with st.spinner("Thinking..."):
            answer = llm.invoke(final_prompt)

        st.subheader("🧠 Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Context"):
            for i, doc in enumerate(docs, 1):
                st.write(f"**Chunk {i}:**")
                st.write(doc.page_content)
